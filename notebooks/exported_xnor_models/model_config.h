#pragma once

// Model architecture constants (XNOR-Net)
#define OBS_DIM           256
#define LATENT_DIM        128
#define ENCODER_HIDDEN    1024
#define PREDICTOR_HIDDEN  512
#define PREDICTOR_HISTORY 3
#define JEPA_STEPS        3
#define GRID_SIZE         16

// Physics sim constants
#define BOX_SIZE          10.0f
#define GRAVITY           -2.0f
#define DT                0.05f
#define RESTITUTION       0.95f
#define BALL_RADIUS       1.5f
