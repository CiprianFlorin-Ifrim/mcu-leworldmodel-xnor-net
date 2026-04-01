#include "matmul.h"
#include <string.h>

static inline int fast_popcount(uint32_t x)
{
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}

void matmul_xnor(const uint8_t *W_packed, const uint8_t *x_packed,
                  int32_t *y, int rows, int cols)
{
    int bytes_per_row = (cols + 7) / 8;
    int words_per_row = (cols + 31) / 32;
    int pad_bits = words_per_row * 32 - cols;

    for (int r = 0; r < rows; r++) {
        const uint32_t *w_row = (const uint32_t *)(W_packed + r * bytes_per_row);
        const uint32_t *x_words = (const uint32_t *)x_packed;
        int32_t agree = 0;

        for (int w = 0; w < words_per_row; w++) {
            uint32_t xnor_val = ~(w_row[w] ^ x_words[w]);
            agree += fast_popcount(xnor_val);
        }

        agree -= pad_bits;
        y[r] = 2 * agree - cols;
    }
}