.section .text
    .align  4

    .global pie_matmul_s8_xacc
    .type   pie_matmul_s8_xacc, @function

// pie_matmul_s8_xacc(const int8_t *W, const int8_t *x, int32_t *y,
//                    int rows, int cols)
//
// Native int8 x int8 matrix-vector multiply using PIE XACC.
// XACC is a 40-bit scalar accumulator that sums all 16 products
// from esp.vmulas.s8.xacc into a single value. No overflow for
// vectors up to ~4M elements.
//
// Based on esp-dl dl_esp32p4_dotprod_i8k8o16 pattern.
//
// a0=W, a1=x, a2=y, a3=rows, a4=cols

pie_matmul_s8_xacc:
    addi    sp, sp, -28
    sw      s0,  0(sp)
    sw      s1,  4(sp)
    sw      s2,  8(sp)
    sw      s3, 12(sp)
    sw      s4, 16(sp)
    sw      s5, 20(sp)
    sw      s6, 24(sp)

    mv      s0, a0              // W base
    mv      s1, a1              // x base (constant)
    mv      s2, a2              // y output pointer
    mv      s3, a3              // rows remaining
    mv      s4, a4              // cols

    beqz    s3, .Ldone

    // n_bulk = cols / 32 (2 vmulas per bulk iteration = 32 elements)
    srli    s5, s4, 5
    // cols_bulk = n_bulk * 32
    slli    t0, s5, 5
    // has_tail = (cols != cols_bulk), i.e. remainder >= 16
    sub     s6, s4, t0
    snez    s6, s6              // s6 = 1 if tail, 0 if not

.Lrow_loop:
    esp.zero.xacc

    mv      a1, s0              // W row pointer
    mv      a2, s1              // x pointer (reset each row)

    // Pre-load first pair of vectors
    esp.vld.128.ip  q0, a1, 16
    esp.vld.128.ip  q1, a2, 16

    mv      t0, s5              // bulk counter
    beqz    t0, .Ltail_check

.Lbulk_loop:
    // Fused: load q2 from W, then MAC q0*q1 into XACC while loading q3 from x
    esp.vld.128.ip  q2, a1, 16
    esp.vmulas.s8.xacc.ld.ip  q3, a2, 16, q0, q1

    // Fused: load q0 from W, then MAC q2*q3 into XACC while loading q1 from x
    esp.vld.128.ip  q0, a1, 16
    esp.vmulas.s8.xacc.ld.ip  q1, a2, 16, q2, q3

    addi    t0, t0, -1
    bnez    t0, .Lbulk_loop

.Ltail_check:
    beqz    s6, .Lextract

    // Process remaining 16 elements (already in q0, q1 from pre-load or last bulk iter)
    esp.vmulas.s8.xacc  q0, q1

.Lextract:
    // Extract XACC to t3 with shift=0 (full 32-bit precision)
    li      a3, 0
    esp.srs.s.xacc  t3, a3

    // Store int32 result for this row
    sw      t3, 0(s2)
    addi    s2, s2, 4

    // Advance W to next row
    add     s0, s0, s4
    addi    s3, s3, -1
    bnez    s3, .Lrow_loop

.Ldone:
    lw      s0,  0(sp)
    lw      s1,  4(sp)
    lw      s2,  8(sp)
    lw      s3, 12(sp)
    lw      s4, 16(sp)
    lw      s5, 20(sp)
    lw      s6, 24(sp)
    addi    sp, sp, 28
    ret

    .size pie_matmul_s8_xacc, . - pie_matmul_s8_xacc