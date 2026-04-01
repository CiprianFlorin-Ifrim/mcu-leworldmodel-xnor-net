#pragma once

#include "model_config.h"
#include <stdint.h>

typedef struct {
    const char *name;

    // fc1 weights
    const int8_t *enc_fc1_w;      // SRAM, int8 unpacked (direct PIE)
    const int8_t *pred_fc1_w_a;   // SRAM chunk A, int8 unpacked
    const int8_t *pred_fc1_w_b;   // SRAM chunk B, int8 unpacked
    int pred_fc1_rows_a;
    int pred_fc1_rows_b;

    // fc2 weights
    const uint8_t *enc_fc2_w;     // SRAM, packed binary (XNOR + popcount)
    const uint8_t *pred_fc2_w;    // SRAM, packed binary (XNOR + popcount)

    // fc2 per-row weight scales
    const float *enc_fc2_scale;
    const float *pred_fc2_scale;

    // Encoder BN folded into int32 thresholds
    int32_t *enc_bn_thr;
    int32_t *enc_bn_flip;

    // Predictor BN partial thresholds
    float   *pred_bn_thr_partial;
    int32_t *pred_bn_flip;

    // Activation scales
    float obs_scale;
    float latent_scale;

    // Padded column sizes
    int obs_padded;
    int pred_input_padded;

    // Inference buffers
    int8_t  *buf_obs_q;
    int8_t  *buf_pred_q;
    int32_t *buf_acc;
    uint8_t *buf_hidden_bits;
    float   *buf_latent;
    float   *buf_pred_input;
    float   *buf_pred_latent;
} ModelXnor;

void model_xnor_init(ModelXnor *m);
void xnor_encode(ModelXnor *m, const float *obs);
void xnor_predict(ModelXnor *m);
float xnor_mse(const float *a, const float *b, int n);
void binarize_from_acc(const int32_t *acc, const int32_t *thr,
                        const int32_t *flip, uint8_t *bits, int n);