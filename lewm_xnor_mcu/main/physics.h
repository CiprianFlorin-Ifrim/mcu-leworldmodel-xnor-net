#pragma once

#include "model_config.h"

typedef struct {
    float x, y, vx, vy;
} BallState;

// Advance ball by one Euler step with elastic wall bounces.
BallState physics_step(BallState s, float gravity, float dt, float restitution);

// Render ball as Gaussian blob on GRID_SIZE x GRID_SIZE grid.
// Output: obs[OBS_DIM] in [0, 1] range.
void rasterise(BallState s, float *obs);
