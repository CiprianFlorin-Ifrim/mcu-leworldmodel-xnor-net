#include "physics.h"
#include <math.h>

BallState physics_step(BallState s, float gravity, float dt, float restitution)
{
    float vy_new = s.vy + gravity * dt;
    float x_new  = s.x  + s.vx * dt;
    float y_new  = s.y  + vy_new * dt;

    float vx = s.vx;

    if (x_new < 0.0f) {
        x_new = -x_new;
        vx    = -vx * restitution;
    } else if (x_new > BOX_SIZE) {
        x_new = 2.0f * BOX_SIZE - x_new;
        vx    = -vx * restitution;
    }

    if (y_new < 0.0f) {
        y_new  = -y_new;
        vy_new = -vy_new * restitution;
    } else if (y_new > BOX_SIZE) {
        y_new  = 2.0f * BOX_SIZE - y_new;
        vy_new = -vy_new * restitution;
    }

    return (BallState){ x_new, y_new, vx, vy_new };
}


void rasterise(BallState s, float *obs)
{
    float cell_size = BOX_SIZE / GRID_SIZE;
    float sigma2    = 2.0f * BALL_RADIUS * BALL_RADIUS;

    for (int row = 0; row < GRID_SIZE; row++) {
        float cy = row * cell_size + cell_size * 0.5f;
        float dy = cy - s.y;
        for (int col = 0; col < GRID_SIZE; col++) {
            float cx = col * cell_size + cell_size * 0.5f;
            float dx = cx - s.x;
            obs[row * GRID_SIZE + col] = expf(-(dx * dx + dy * dy) / sigma2);
        }
    }
}
