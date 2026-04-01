#include "inference_xnor.h"
#include "matmul.h"
#include "weights_xnor.h"
#include "act_scales_xnor.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// -----------------------------------------------------------------------
// Unpack binary weights (flash) -> int8 {-1, +1}
// -----------------------------------------------------------------------

static int8_t *unpack_weights(const uint8_t *packed, int rows, int cols,
                                int cols_padded, int use_psram)
{
    int n = rows * cols_padded;
    int8_t *buf;
    if (use_psram) {
        buf = (int8_t *)heap_caps_aligned_alloc(16, n, MALLOC_CAP_SPIRAM);
    } else {
        buf = (int8_t *)heap_caps_aligned_alloc(16, n, MALLOC_CAP_INTERNAL);
    }
    if (!buf) return NULL;
    memset(buf, 0, n);

    int bytes_per_row = (cols + 7) / 8;
    for (int r = 0; r < rows; r++) {
        const uint8_t *src_row = packed + r * bytes_per_row;
        int8_t *dst_row = buf + r * cols_padded;
        for (int c = 0; c < cols; c++) {
            dst_row[c] = ((src_row[c >> 3] >> (c & 7)) & 1) ? 1 : -1;
        }
    }
    return buf;
}

// -----------------------------------------------------------------------
// Copy packed binary weights to aligned SRAM
// -----------------------------------------------------------------------

static uint8_t *copy_packed_to_sram(const uint8_t *src, int rows, int cols)
{
    int bytes = (rows * cols + 7) / 8;
    int bytes_padded = (bytes + 15) & ~15;
    uint8_t *buf = (uint8_t *)heap_caps_aligned_alloc(16, bytes_padded, MALLOC_CAP_INTERNAL);
    if (buf) {
        memset(buf, 0, bytes_padded);
        memcpy(buf, src, bytes);
    }
    return buf;
}

// -----------------------------------------------------------------------
// BN threshold folding
// -----------------------------------------------------------------------

static void compute_bn_thresholds_fixed(
    const float *bn_mean, const float *bn_var,
    const float *bn_gamma, const float *bn_beta, float bn_eps,
    const float *w_scale, float input_scale,
    int n, int32_t *thr_out, int32_t *flip_out)
{
    for (int i = 0; i < n; i++) {
        float std = sqrtf(bn_var[i] + bn_eps);
        float bn_thr = bn_mean[i] - bn_beta[i] * std / bn_gamma[i];
        float acc_thr = bn_thr / (w_scale[i] * input_scale);
        thr_out[i] = (int32_t)roundf(acc_thr);
        flip_out[i] = (bn_gamma[i] < 0.0f) ? -1 : 0;
    }
}

static void compute_bn_thresholds_partial(
    const float *bn_mean, const float *bn_var,
    const float *bn_gamma, const float *bn_beta, float bn_eps,
    const float *w_scale,
    int n, float *partial_out, int32_t *flip_out)
{
    for (int i = 0; i < n; i++) {
        float std = sqrtf(bn_var[i] + bn_eps);
        float bn_thr = bn_mean[i] - bn_beta[i] * std / bn_gamma[i];
        partial_out[i] = bn_thr / w_scale[i];
        flip_out[i] = (bn_gamma[i] < 0.0f) ? -1 : 0;
    }
}

// -----------------------------------------------------------------------
// Binarize from int32 acc using precomputed thresholds
// -----------------------------------------------------------------------

void binarize_from_acc(const int32_t *acc, const int32_t *thr,
                        const int32_t *flip, uint8_t *bits, int n)
{
    int bytes = (n + 7) / 8;
    memset(bits, 0, bytes);
    for (int i = 0; i < n; i++) {
        int32_t diff = acc[i] - thr[i];
        int32_t val = diff ^ flip[i];
        if (val >= 0) {
            bits[i >> 3] |= (1 << (i & 7));
        }
    }
}

// -----------------------------------------------------------------------
// Init
//
// enc_fc1:  SRAM int8 unpacked (256KB in 384KB block, single PIE call)
// pred_fc1: SRAM int8 unpacked, split across two memory regions
// enc_fc2:  SRAM packed binary (XNOR + popcount)
// pred_fc2: SRAM packed binary (XNOR + popcount)
// -----------------------------------------------------------------------

void model_xnor_init(ModelXnor *m)
{
    m->name = "xnor";

    int obs_padded = (OBS_DIM + 15) & ~15;
    int pred_input_dim = PREDICTOR_HISTORY * LATENT_DIM;
    int pred_padded = (pred_input_dim + 15) & ~15;

    m->obs_padded = obs_padded;
    m->pred_input_padded = pred_padded;

    // enc_fc1: SRAM (256KB in largest block, single PIE call)
    m->enc_fc1_w = unpack_weights(enc_fc1_weight, ENCODER_HIDDEN,
                                   OBS_DIM, obs_padded, 0);

    // fc2: copy packed to SRAM (eliminates flash cache misses)
    m->enc_fc2_w  = copy_packed_to_sram(enc_fc2_weight, LATENT_DIM, ENCODER_HIDDEN);
    m->pred_fc2_w = copy_packed_to_sram(pred_fc2_weight, LATENT_DIM, PREDICTOR_HIDDEN);

    // pred_fc1: split across two SRAM regions
    {
        int bytes_per_row = pred_padded;
        int total_rows = PREDICTOR_HIDDEN;
        int bytes_per_row_packed = (pred_input_dim + 7) / 8;

        // Try largest chunk that fits
        int rows_a = total_rows;
        int8_t *buf_a = NULL;
        while (rows_a > 0) {
            int sz = rows_a * bytes_per_row;
            buf_a = (int8_t *)heap_caps_aligned_alloc(16, sz, MALLOC_CAP_INTERNAL);
            if (buf_a) break;
            rows_a -= 16;
        }

        int rows_b = total_rows - rows_a;
        int8_t *buf_b = NULL;
        if (rows_b > 0) {
            int sz = rows_b * bytes_per_row;
            buf_b = (int8_t *)heap_caps_aligned_alloc(16, sz, MALLOC_CAP_INTERNAL);
        }

        // Unpack into chunk A
        for (int r = 0; r < rows_a; r++) {
            const uint8_t *src = pred_fc1_weight + r * bytes_per_row_packed;
            int8_t *dst = buf_a + r * bytes_per_row;
            memset(dst, 0, bytes_per_row);
            for (int c = 0; c < pred_input_dim; c++) {
                dst[c] = ((src[c >> 3] >> (c & 7)) & 1) ? 1 : -1;
            }
        }

        // Unpack into chunk B
        for (int r = 0; r < rows_b; r++) {
            const uint8_t *src = pred_fc1_weight + (rows_a + r) * bytes_per_row_packed;
            int8_t *dst = buf_b + r * bytes_per_row;
            memset(dst, 0, bytes_per_row);
            for (int c = 0; c < pred_input_dim; c++) {
                dst[c] = ((src[c >> 3] >> (c & 7)) & 1) ? 1 : -1;
            }
        }

        m->pred_fc1_w_a = buf_a;
        m->pred_fc1_w_b = buf_b;
        m->pred_fc1_rows_a = rows_a;
        m->pred_fc1_rows_b = rows_b;

        ESP_LOGI("xnor", "pred_fc1 split: %d + %d rows (%dKB + %dKB SRAM)",
                 rows_a, rows_b,
                 (rows_a * bytes_per_row) / 1024,
                 (rows_b * bytes_per_row) / 1024);
    }

    // fc2 scales
    m->enc_fc2_scale  = enc_fc2_scale;
    m->pred_fc2_scale = pred_fc2_scale;

    // Activation scales
    float enc_input_scale = ACT_SCALE_OBS / 127.0f;
    m->obs_scale    = ACT_SCALE_OBS;
    m->latent_scale = ACT_SCALE_ENC_FC2_OUT;

    // Encoder BN: fixed int32 thresholds
    m->enc_bn_thr  = (int32_t *)malloc(ENCODER_HIDDEN * sizeof(int32_t));
    m->enc_bn_flip = (int32_t *)malloc(ENCODER_HIDDEN * sizeof(int32_t));
    compute_bn_thresholds_fixed(
        enc_fc1_bn_mean, enc_fc1_bn_var,
        enc_fc1_bn_gamma, enc_fc1_bn_beta, ENC_FC1_BN_EPS,
        enc_fc1_scale, enc_input_scale,
        ENCODER_HIDDEN, m->enc_bn_thr, m->enc_bn_flip);

    // Predictor BN: partial thresholds
    m->pred_bn_thr_partial = (float *)malloc(PREDICTOR_HIDDEN * sizeof(float));
    m->pred_bn_flip = (int32_t *)malloc(PREDICTOR_HIDDEN * sizeof(int32_t));
    compute_bn_thresholds_partial(
        pred_fc1_bn_mean, pred_fc1_bn_var,
        pred_fc1_bn_gamma, pred_fc1_bn_beta, PRED_FC1_BN_EPS,
        pred_fc1_scale,
        PREDICTOR_HIDDEN, m->pred_bn_thr_partial, m->pred_bn_flip);

    // Inference buffers
    int max_hidden = ENCODER_HIDDEN;
    if (PREDICTOR_HIDDEN > max_hidden) max_hidden = PREDICTOR_HIDDEN;
    int max_dim = max_hidden;
    if (LATENT_DIM > max_dim) max_dim = LATENT_DIM;

    m->buf_obs_q       = (int8_t *)heap_caps_aligned_alloc(16, obs_padded, MALLOC_CAP_INTERNAL);
    m->buf_pred_q      = (int8_t *)heap_caps_aligned_alloc(16, pred_padded, MALLOC_CAP_INTERNAL);
    m->buf_acc          = (int32_t *)heap_caps_aligned_alloc(16, max_dim * sizeof(int32_t), MALLOC_CAP_INTERNAL);
    m->buf_hidden_bits  = (uint8_t *)heap_caps_aligned_alloc(16, (max_hidden + 7) / 8, MALLOC_CAP_INTERNAL);
    m->buf_latent       = (float *)malloc(LATENT_DIM * sizeof(float));
    m->buf_pred_input   = (float *)malloc(pred_input_dim * sizeof(float));
    m->buf_pred_latent  = (float *)malloc(LATENT_DIM * sizeof(float));

    memset(m->buf_obs_q, 0, obs_padded);
    memset(m->buf_pred_q, 0, pred_padded);
}

// -----------------------------------------------------------------------
// Encode
// -----------------------------------------------------------------------

void xnor_encode(ModelXnor *m, const float *obs)
{
    // Quantise observation to int8
    float inv_scale = 127.0f / m->obs_scale;
    for (int j = 0; j < OBS_DIM; j++) {
        int v = (int)(obs[j] * inv_scale + 0.5f);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        m->buf_obs_q[j] = (int8_t)v;
    }

    // fc1: single PIE call from SRAM
    pie_matmul_s8_xacc(m->enc_fc1_w, m->buf_obs_q, m->buf_acc,
                        ENCODER_HIDDEN, m->obs_padded);

    // BN folded: int32 threshold -> packed bits
    binarize_from_acc(m->buf_acc, m->enc_bn_thr, m->enc_bn_flip,
                       m->buf_hidden_bits, ENCODER_HIDDEN);

    // fc2: XNOR + popcount from SRAM
    matmul_xnor(m->enc_fc2_w, m->buf_hidden_bits, m->buf_acc,
                 LATENT_DIM, ENCODER_HIDDEN);

    // Scale to float latent
    for (int i = 0; i < LATENT_DIM; i++) {
        m->buf_latent[i] = m->enc_fc2_scale[i] * (float)m->buf_acc[i];
    }
}

// -----------------------------------------------------------------------
// Predict
// -----------------------------------------------------------------------

void xnor_predict(ModelXnor *m)
{
    int pred_input_dim = PREDICTOR_HISTORY * LATENT_DIM;

    // Quantise predictor input to int8
    float absmax = 0.0f;
    for (int j = 0; j < pred_input_dim; j++) {
        float a = fabsf(m->buf_pred_input[j]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-10f) absmax = 1e-10f;

    float input_scale = absmax / 127.0f;
    float inv_input_scale = 1.0f / input_scale;

    memset(m->buf_pred_q, 0, m->pred_input_padded);
    for (int j = 0; j < pred_input_dim; j++) {
        float v = m->buf_pred_input[j] * inv_input_scale;
        int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        m->buf_pred_q[j] = (int8_t)q;
    }

    // fc1: two PIE calls from SRAM (split across memory regions)
    pie_matmul_s8_xacc(m->pred_fc1_w_a, m->buf_pred_q, m->buf_acc,
                        m->pred_fc1_rows_a, m->pred_input_padded);
    if (m->pred_fc1_rows_b > 0) {
        pie_matmul_s8_xacc(m->pred_fc1_w_b, m->buf_pred_q,
                            m->buf_acc + m->pred_fc1_rows_a,
                            m->pred_fc1_rows_b, m->pred_input_padded);
    }

    // BN folded: compute int32 thresholds from partial + runtime input_scale
    float inv_is = 1.0f / input_scale;
    int32_t pred_thr[PREDICTOR_HIDDEN];
    for (int i = 0; i < PREDICTOR_HIDDEN; i++) {
        pred_thr[i] = (int32_t)roundf(m->pred_bn_thr_partial[i] * inv_is);
    }

    binarize_from_acc(m->buf_acc, pred_thr, m->pred_bn_flip,
                       m->buf_hidden_bits, PREDICTOR_HIDDEN);

    // fc2: XNOR + popcount from SRAM
    matmul_xnor(m->pred_fc2_w, m->buf_hidden_bits, m->buf_acc,
                 LATENT_DIM, PREDICTOR_HIDDEN);

    // Scale to float
    for (int i = 0; i < LATENT_DIM; i++) {
        m->buf_pred_latent[i] = m->pred_fc2_scale[i] * (float)m->buf_acc[i];
    }
}

// -----------------------------------------------------------------------
// MSE
// -----------------------------------------------------------------------

float xnor_mse(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum / (float)n;
}