#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "model_config.h"
#include "physics.h"
#include "inference_xnor.h"
#include "matmul.h"

static const char *TAG = "xnor";

#define TRAJ_LEN         80
#define ANOMALY_STEP     40
#define N_TIMING_ITERS   1000
#define N_VOE_SEEDS      10

// -----------------------------------------------------------------------
// PRNG
// -----------------------------------------------------------------------

static uint32_t rng_state = 42;

static uint32_t xorshift32(void)
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static float rand_uniform(float lo, float hi)
{
    float t = (float)(xorshift32() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    return lo + t * (hi - lo);
}

// -----------------------------------------------------------------------
// Trajectory generation
// -----------------------------------------------------------------------

static void generate_trajectory(float *obs_out, int traj_len, int anomaly_type,
                                 int anomaly_step, uint32_t seed)
{
    rng_state = seed;
    BallState s;
    s.x  = rand_uniform(2.0f, BOX_SIZE - 2.0f);
    s.y  = rand_uniform(2.0f, BOX_SIZE - 2.0f);
    s.vx = rand_uniform(-4.0f, 4.0f);
    s.vy = rand_uniform(-4.0f, 4.0f);

    float gravity_current = GRAVITY;

    for (int t = 0; t < traj_len; t++) {
        if (t == anomaly_step && anomaly_type != 0) {
            if (anomaly_type == 1) {
                s.x = rand_uniform(1.0f, BOX_SIZE - 1.0f);
                s.y = rand_uniform(1.0f, BOX_SIZE - 1.0f);
            } else if (anomaly_type == 2) {
                s.vx = -s.vx;
                s.vy = -s.vy;
            } else if (anomaly_type == 3) {
                gravity_current = -gravity_current;
            }
        }
        rasterise(s, obs_out + t * OBS_DIM);
        s = physics_step(s, gravity_current, DT, RESTITUTION);
    }
}

// -----------------------------------------------------------------------
// Surprise computation
// -----------------------------------------------------------------------

static int compute_surprise(ModelXnor *m, const float *obs,
                             int traj_len, float *surprise_out)
{
    int H     = PREDICTOR_HISTORY;
    int n_out = traj_len - H - JEPA_STEPS + 1;

    float (*z_all)[LATENT_DIM] = (float (*)[LATENT_DIM])malloc(traj_len * LATENT_DIM * sizeof(float));
    if (!z_all) return 0;

    for (int t = 0; t < traj_len; t++) {
        xnor_encode(m, obs + t * OBS_DIM);
        memcpy(z_all[t], m->buf_latent, LATENT_DIM * sizeof(float));
    }

    for (int i = 0; i < n_out; i++) {
        int t_start = i + H - 1;
        float hist[PREDICTOR_HISTORY][LATENT_DIM];
        for (int h = 0; h < H; h++)
            memcpy(hist[h], z_all[t_start - H + 1 + h], LATENT_DIM * sizeof(float));

        float step_err = 0.0f;
        for (int k = 1; k <= JEPA_STEPS; k++) {
            for (int h = 0; h < H; h++)
                memcpy(m->buf_pred_input + h * LATENT_DIM, hist[h], LATENT_DIM * sizeof(float));

            xnor_predict(m);
            step_err += xnor_mse(m->buf_pred_latent, z_all[t_start + k], LATENT_DIM);

            for (int h = 0; h < H - 1; h++)
                memcpy(hist[h], hist[h + 1], LATENT_DIM * sizeof(float));
            memcpy(hist[H - 1], m->buf_pred_latent, LATENT_DIM * sizeof(float));
        }
        surprise_out[i] = step_err / (float)JEPA_STEPS;
    }

    free(z_all);
    return n_out;
}

// -----------------------------------------------------------------------
// Peak surprise near anomaly
// -----------------------------------------------------------------------

static float peak_surprise(const float *surprise, int n_out, int anomaly_step)
{
    int t_off = PREDICTOR_HISTORY - 1;
    int anom_idx = anomaly_step - t_off;
    int lo = anom_idx - JEPA_STEPS;
    int hi = anom_idx + JEPA_STEPS + 1;
    if (lo < 0) lo = 0;
    if (hi > n_out) hi = n_out;

    float peak = 0.0f;
    for (int i = lo; i < hi; i++)
        if (surprise[i] > peak) peak = surprise[i];
    return peak;
}

// -----------------------------------------------------------------------
// Timing
// -----------------------------------------------------------------------

static float bench_model(ModelXnor *m, const float *obs)
{
    for (int i = 0; i < 10; i++) {
        xnor_encode(m, obs);
        memset(m->buf_pred_input, 0, PREDICTOR_HISTORY * LATENT_DIM * sizeof(float));
        xnor_predict(m);
    }

    int64_t t0 = esp_timer_get_time();
    for (int iter = 0; iter < N_TIMING_ITERS; iter++) {
        xnor_encode(m, obs);
        memcpy(m->buf_pred_input + (PREDICTOR_HISTORY - 1) * LATENT_DIM,
               m->buf_latent, LATENT_DIM * sizeof(float));
        xnor_predict(m);
    }
    int64_t elapsed = esp_timer_get_time() - t0;
    return (float)elapsed / (float)N_TIMING_ITERS;
}

// -----------------------------------------------------------------------
// app_main
// -----------------------------------------------------------------------

void app_main(void)
{
    ESP_LOGI(TAG, "XNOR-Net LeWorldModel -- ESP32-P4 inference");
    ESP_LOGI(TAG, "CPU freq: %d MHz", CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    ESP_LOGI(TAG, "Grid: %dx%d  Latent: %d  Hidden: %d/%d  History: %d  Steps: %d",
             GRID_SIZE, GRID_SIZE, LATENT_DIM, ENCODER_HIDDEN, PREDICTOR_HIDDEN,
             PREDICTOR_HISTORY, JEPA_STEPS);

    vTaskDelay(pdMS_TO_TICKS(500));

    // ---------------------------------------------------------------
    // Init model
    // ---------------------------------------------------------------

    ModelXnor model;
    model_xnor_init(&model);

    if (!model.enc_fc1_w || !model.pred_fc1_w_a ||
        !model.enc_fc2_w || !model.pred_fc2_w) {
        ESP_LOGE(TAG, "Weight allocation failed!");
        ESP_LOGE(TAG, "enc_fc1=%p pred_fc1_a=%p pred_fc1_b=%p enc_fc2=%p pred_fc2=%p",
                 model.enc_fc1_w, model.pred_fc1_w_a, model.pred_fc1_w_b,
                 model.enc_fc2_w, model.pred_fc2_w);
        return;
    }

    // ---------------------------------------------------------------
    // Memory usage
    // ---------------------------------------------------------------

    printf("\n--- Memory Usage ---\n");

    size_t total_sram = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    size_t free_sram  = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t free_psram  = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    printf("%-14s  %10s  %10s  %10s\n", "Region", "Total", "Free", "Used");
    printf("------------------------------------------------------\n");
    printf("%-14s  %8zuKB  %8zuKB  %8zuKB\n", "SRAM",
           total_sram / 1024, free_sram / 1024, (total_sram - free_sram) / 1024);
    printf("%-14s  %8zuKB  %8zuKB  %8zuKB\n", "PSRAM",
           total_psram / 1024, free_psram / 1024, (total_psram - free_psram) / 1024);

    int total_params = ENCODER_HIDDEN * OBS_DIM
                     + LATENT_DIM * ENCODER_HIDDEN
                     + PREDICTOR_HIDDEN * (PREDICTOR_HISTORY * LATENT_DIM)
                     + LATENT_DIM * PREDICTOR_HIDDEN;

    int obs_padded = model.obs_padded;
    int pred_padded = model.pred_input_padded;
    int pred_input_dim = PREDICTOR_HISTORY * LATENT_DIM;

    int enc_fc1_ram  = ENCODER_HIDDEN * obs_padded;
    int pred_fc1_ram = PREDICTOR_HIDDEN * pred_padded;
    int enc_fc2_ram  = (((LATENT_DIM * ENCODER_HIDDEN + 7) / 8) + 15) & ~15;
    int pred_fc2_ram = (((LATENT_DIM * PREDICTOR_HIDDEN + 7) / 8) + 15) & ~15;

    printf("\n%-14s  %10s  %10s\n", "Weights", "Bytes", "Location / Method");
    printf("--------------------------------------------------------------\n");
    printf("%-14s  %10d  SRAM, single PIE XACC call\n", "enc_fc1", enc_fc1_ram);
    printf("%-14s  %10d  SRAM split %d+%d rows, 2 PIE calls\n", "pred_fc1",
           pred_fc1_ram, model.pred_fc1_rows_a, model.pred_fc1_rows_b);
    printf("%-14s  %10d  SRAM, XNOR + popcount\n", "enc_fc2", enc_fc2_ram);
    printf("%-14s  %10d  SRAM, XNOR + popcount\n", "pred_fc2", pred_fc2_ram);
    printf("%-14s  %10d  params total\n", "Model", total_params);
    printf("\nSRAM used:      %zuKB\n", (total_sram - free_sram) / 1024);

    size_t min_free = heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL);
    printf("Min SRAM free:  %zuKB\n", min_free / 1024);

    // ---------------------------------------------------------------
    // VoE evaluation
    // ---------------------------------------------------------------

    float *obs      = (float *)malloc(TRAJ_LEN * OBS_DIM * sizeof(float));
    float *surprise = (float *)malloc(TRAJ_LEN * sizeof(float));
    if (!obs || !surprise) {
        ESP_LOGE(TAG, "Failed to allocate trajectory buffers");
        return;
    }

    const char *anomaly_names[] = { "normal", "teleport", "vel_flip", "gravity" };
    int n_anomalies = 4;

    printf("\n--- VoE: XNOR-Net ---\n");
    printf("%-14s", "Model");
    for (int a = 0; a < n_anomalies; a++)
        printf("  %12s", anomaly_names[a]);
    printf("\n");
    for (int i = 0; i < 14 + n_anomalies * 14; i++) putchar('-');
    printf("\n");

    printf("%-14s", model.name);
    for (int a = 0; a < n_anomalies; a++) {
        float peak_sum = 0.0f;
        for (int s = 0; s < N_VOE_SEEDS; s++) {
            generate_trajectory(obs, TRAJ_LEN, a, ANOMALY_STEP, 1000 + s);
            int n_out = compute_surprise(&model, obs, TRAJ_LEN, surprise);
            peak_sum += peak_surprise(surprise, n_out, ANOMALY_STEP);
        }
        printf("  %12.6f", peak_sum / N_VOE_SEEDS);
    }
    printf("\n");

    // ---------------------------------------------------------------
    // Timing
    // ---------------------------------------------------------------

    generate_trajectory(obs, 1, 0, 999, 77);

    printf("\n--- Inference Timing (%d iterations) ---\n", N_TIMING_ITERS);
    float us = bench_model(&model, obs);
    printf("%-14s  %10.1f us/frame\n", model.name, us);

    // ---------------------------------------------------------------
    // Per-layer timing breakdown
    // ---------------------------------------------------------------

    printf("\n--- Per-Layer Timing (%d iterations) ---\n", N_TIMING_ITERS);

    int64_t t0, t_enc_quant = 0, t_enc_fc1 = 0, t_enc_bn = 0;
    int64_t t_enc_fc2 = 0, t_enc_scale = 0;
    int64_t t_pred_quant = 0, t_pred_fc1 = 0, t_pred_bn = 0;
    int64_t t_pred_fc2 = 0, t_pred_scale = 0;

    for (int iter = 0; iter < N_TIMING_ITERS; iter++) {
        // --- Encode ---

        t0 = esp_timer_get_time();
        {
            float inv_s = 127.0f / model.obs_scale;
            for (int j = 0; j < OBS_DIM; j++) {
                int v = (int)(obs[j] * inv_s + 0.5f);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                model.buf_obs_q[j] = (int8_t)v;
            }
        }
        t_enc_quant += esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        pie_matmul_s8_xacc(model.enc_fc1_w, model.buf_obs_q, model.buf_acc,
                            ENCODER_HIDDEN, obs_padded);
        t_enc_fc1 += esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        binarize_from_acc(model.buf_acc, model.enc_bn_thr, model.enc_bn_flip,
                           model.buf_hidden_bits, ENCODER_HIDDEN);
        t_enc_bn += esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        matmul_xnor(model.enc_fc2_w, model.buf_hidden_bits, model.buf_acc,
                      LATENT_DIM, ENCODER_HIDDEN);
        t_enc_fc2 += esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int i = 0; i < LATENT_DIM; i++)
            model.buf_latent[i] = model.enc_fc2_scale[i] * (float)model.buf_acc[i];
        t_enc_scale += esp_timer_get_time() - t0;

        // --- Predict ---

        memcpy(model.buf_pred_input + (PREDICTOR_HISTORY - 1) * LATENT_DIM,
               model.buf_latent, LATENT_DIM * sizeof(float));

        t0 = esp_timer_get_time();
        {
            float absmax = 0.0f;
            for (int j = 0; j < pred_input_dim; j++) {
                float a = fabsf(model.buf_pred_input[j]);
                if (a > absmax) absmax = a;
            }
            if (absmax < 1e-10f) absmax = 1e-10f;
            float is = absmax / 127.0f;
            float inv_is = 1.0f / is;
            memset(model.buf_pred_q, 0, pred_padded);
            for (int j = 0; j < pred_input_dim; j++) {
                float v = model.buf_pred_input[j] * inv_is;
                int q = (int)(v + (v >= 0 ? 0.5f : -0.5f));
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                model.buf_pred_q[j] = (int8_t)q;
            }
        }
        t_pred_quant += esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        pie_matmul_s8_xacc(model.pred_fc1_w_a, model.buf_pred_q,
                            model.buf_acc, model.pred_fc1_rows_a, pred_padded);
        if (model.pred_fc1_rows_b > 0) {
            pie_matmul_s8_xacc(model.pred_fc1_w_b, model.buf_pred_q,
                                model.buf_acc + model.pred_fc1_rows_a,
                                model.pred_fc1_rows_b, pred_padded);
        }
        t_pred_fc1 += esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        {
            float absmax = 0.0f;
            for (int j = 0; j < pred_input_dim; j++) {
                float a = fabsf(model.buf_pred_input[j]);
                if (a > absmax) absmax = a;
            }
            if (absmax < 1e-10f) absmax = 1e-10f;
            float is = absmax / 127.0f;
            float inv_is = 1.0f / is;
            int32_t pred_thr[PREDICTOR_HIDDEN];
            for (int i = 0; i < PREDICTOR_HIDDEN; i++)
                pred_thr[i] = (int32_t)roundf(model.pred_bn_thr_partial[i] * inv_is);
            binarize_from_acc(model.buf_acc, pred_thr, model.pred_bn_flip,
                               model.buf_hidden_bits, PREDICTOR_HIDDEN);
        }
        t_pred_bn += esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        matmul_xnor(model.pred_fc2_w, model.buf_hidden_bits, model.buf_acc,
                      LATENT_DIM, PREDICTOR_HIDDEN);
        t_pred_fc2 += esp_timer_get_time() - t0;

        t0 = esp_timer_get_time();
        for (int i = 0; i < LATENT_DIM; i++)
            model.buf_pred_latent[i] = model.pred_fc2_scale[i] * (float)model.buf_acc[i];
        t_pred_scale += esp_timer_get_time() - t0;
    }

    float n = (float)N_TIMING_ITERS;
    printf("%-20s  %10s\n", "Operation", "us/frame");
    printf("--------------------------------------\n");
    printf("%-20s  %10.1f\n", "enc_quant",       t_enc_quant / n);
    printf("%-20s  %10.1f\n", "enc_fc1 (PIE)",    t_enc_fc1 / n);
    printf("%-20s  %10.1f\n", "enc_bn_thr",      t_enc_bn / n);
    printf("%-20s  %10.1f\n", "enc_fc2 (XNOR)",   t_enc_fc2 / n);
    printf("%-20s  %10.1f\n", "enc_scale",       t_enc_scale / n);
    printf("%-20s  %10.1f\n", "pred_quant",      t_pred_quant / n);
    printf("%-20s  %10.1f\n", "pred_fc1 (PIE)",   t_pred_fc1 / n);
    printf("%-20s  %10.1f\n", "pred_bn_thr",     t_pred_bn / n);
    printf("%-20s  %10.1f\n", "pred_fc2 (XNOR)",  t_pred_fc2 / n);
    printf("%-20s  %10.1f\n", "pred_scale",      t_pred_scale / n);

    float total_enc = (t_enc_quant + t_enc_fc1 + t_enc_bn + t_enc_fc2 + t_enc_scale) / n;
    float total_pred = (t_pred_quant + t_pred_fc1 + t_pred_bn + t_pred_fc2 + t_pred_scale) / n;
    printf("--------------------------------------\n");
    printf("%-20s  %10.1f\n", "Encoder total", total_enc);
    printf("%-20s  %10.1f\n", "Predictor total", total_pred);
    printf("%-20s  %10.1f\n", "Grand total", total_enc + total_pred);

    // ---------------------------------------------------------------
    // Surprise trace
    // ---------------------------------------------------------------

    printf("\n--- Surprise Trace (teleport at step %d) ---\n", ANOMALY_STEP);
    generate_trajectory(obs, TRAJ_LEN, 1, ANOMALY_STEP, 42);
    int n_out = compute_surprise(&model, obs, TRAJ_LEN, surprise);
    int t_off = PREDICTOR_HISTORY - 1;
    for (int i = 0; i < n_out; i++) {
        printf("  step %3d: %.6f", i + t_off, surprise[i]);
        if (i + t_off == ANOMALY_STEP) printf("  <-- anomaly");
        printf("\n");
    }

    free(obs);
    free(surprise);

    ESP_LOGI(TAG, "All benchmarks complete.");
}