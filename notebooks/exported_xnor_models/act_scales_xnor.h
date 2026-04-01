#pragma once

// Per-layer activation scales for XNOR-Net full integer inference.
// Calibrated from 1000 trajectories x 100 steps with 10% headroom.

#define ACT_SCALE_OBS 1.09999987f
#define ACT_SCALE_ENC_FC1_OUT 1.65806071f
#define ACT_SCALE_ENC_FC2_OUT 3.90328345f
#define ACT_SCALE_PRED_FC1_OUT 4.30894527f
#define ACT_SCALE_PRED_FC2_OUT 3.99759729f
