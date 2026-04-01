#pragma once

#include <stdint.h>

extern void pie_matmul_s8_xacc(const int8_t *W, const int8_t *x, int32_t *y,
                                int rows, int cols);

void matmul_xnor(const uint8_t *W_packed, const uint8_t *x_packed,
                  int32_t *y, int rows, int cols);