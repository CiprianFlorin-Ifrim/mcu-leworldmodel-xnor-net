# XNOR-Net LeWorldModel -- Baremetal MCU Inference on ESP32-P4

Binary-weight neural network for physics anomaly detection, deployed on a $3.50 microcontroller with custom RISC-V SIMD assembly. 655K parameters, 1,129 microseconds per frame, 10x anomaly detection ratio.

## Overview

This project implements a JEPA (Joint Embedding Predictive Architecture) world model trained with XNOR-Net binary activations, deployed as a baremetal C inference engine on the ESP32-P4 RISC-V MCU. The model detects physics violations (teleportation, velocity reversal, gravity inversion) in a 2D bouncing ball simulation by computing "surprise" -- the prediction error between expected and observed latent states.

The key contributions are:

- XNOR-Net training from scratch with hard sign activations, clipped STE, and L1 mean weight scaling
- Full-integer BatchNorm folding into int32 thresholds (zero float ops in the encoder fc1 path)
- ESP32-P4 PIE (Processor Instruction Extension) SIMD assembly for int8 dot products at 16 MACs/cycle
- Software XNOR + popcount for binary-binary layers (no hardware Zbb/cpop on ESP32-P4)
- Split SRAM allocation across fragmented memory regions for all-SRAM deployment
- Per-layer timing instrumentation and systematic optimization (11.5x speedup over scalar baseline)

## Architecture

```
Observation (16x16 float pixels)
  |
  v
Encoder fc1: int8_input x binary_weights --> PIE XACC (256 cols, 1024 rows)
  |
  v
BatchNorm --> sign() : folded into int32 threshold compare (NO FLOAT)
  |
  v
Encoder fc2: binary_input x binary_weights --> XNOR + popcount (1024 cols, 128 rows)
  |
  v
Scale to float latent (128-dim)
  |
  v
Predictor fc1: int8_input x binary_weights --> PIE XACC (384 cols, 512 rows)
  |
  v
BatchNorm --> sign() : partial threshold (1 float mul/element)
  |
  v
Predictor fc2: binary_input x binary_weights --> XNOR + popcount (512 cols, 128 rows)
  |
  v
Scale to float predicted latent (128-dim)
```

fc1 layers use PIE XACC because the input is continuous (int8), not binary. Only fc2 layers are true XNOR. This matches the XNOR-Net paper (Rastegari et al., 2016): "avoid binarization at the first and last layer."

## Model

| Parameter | Value |
|-----------|-------|
| Architecture | JEPA encoder-predictor |
| Latent dim | 128 |
| Encoder hidden | 1024 |
| Predictor hidden | 512 |
| Predictor history | 3 frames |
| JEPA steps | 3 |
| Total parameters | 655,360 |
| Weight precision | Binary (+1/-1) |
| Activation precision | Binary (after BN + sign) |
| Weight storage | Packed bits (82 KB) |

### Training Recipe

The XNOR-Net training required specific techniques to converge. Multiple failed approaches are documented below.

Failed approaches (45K param model):
- Hard sign from scratch: diverged immediately
- Clipped STE: diverged slower but still unstable
- Progressive tanh annealing: worked during ramp, catastrophic jump at temperature 50-54 (gradient of tanh(T*x) is pathological at high T: T*sech^2(T*x) = 50 at x=0, 0 at x=0.05)

Working recipe (655K param model, 8x wider):
- Hard sign() from epoch 1 with clipped STE (gradient=1 if |w|<=1)
- L1 mean weight scale (paper equation 6, proven optimal)
- Adam optimizer (paper recommends for XNOR, not SGD)
- LR=1e-3, step decay 0.1x every 100 epochs
- Batch size 8192 (critical: smooths STE gradient noise)
- SIGReg lambda=2.0
- 400 epochs

Result: val_pred=0.070, teleport ratio=14x (working anomaly detector).

## ESP32-P4 Deployment

### Hardware

- Chip: ESP32-P4, dual-core RISC-V at 360 MHz
- ISA: RV32IMAFCZc + Xhwlp + Xai (PIE)
- SRAM: 768 KB (fragmented: 384 KB + 155 KB + 18 KB + smaller blocks)
- PSRAM: 32 MB (200 MHz x16, ~400 MB/s)
- PIE: 128-bit vector registers, 16 int8 MACs/cycle via XACC accumulator

### Memory Layout

All weights in SRAM (no PSRAM streaming needed):

| Weight | Size | Location | Method |
|--------|------|----------|--------|
| enc_fc1 | 256 KB | SRAM (384 KB block) | Single PIE XACC call |
| pred_fc1 | 192 KB | SRAM (split 320+192 rows across 2 blocks) | 2 PIE XACC calls |
| enc_fc2 | 16 KB | SRAM (packed binary) | XNOR + popcount |
| pred_fc2 | 8 KB | SRAM (packed binary) | XNOR + popcount |
| Total SRAM used | 545 KB | | |
| SRAM free | 84 KB | | |

The pred_fc1 split allocation was necessary because no single SRAM region had 192 KB free after enc_fc1 consumed the largest block. The allocator tries the largest contiguous chunk first, then allocates the remainder in the next available region.

### BatchNorm Folding

The chain: accumulator --> dequantize --> BatchNorm --> sign()

```
sign(gamma * (w_scale * in_scale * acc - mean) / sqrt(var + eps) + beta)
```

Precomputed threshold on the raw int32 accumulator:

```
threshold = (mean - beta * sqrt(var + eps) / gamma) / (w_scale * in_scale)
```

For the encoder: in_scale = obs_scale/127 is constant (observation range is fixed), so the threshold is a compile-time int32 constant. Zero float operations in the entire fc1 path.

For the predictor: in_scale varies per call (dynamic absmax quantization of latent inputs), so we store partial thresholds (threshold / w_scale) and multiply by 1/in_scale at runtime. One float multiply and one round per element -- cheaper than full BN.

The sign flip for negative gamma is handled by XOR: `bit = ((acc - threshold) ^ flip) >= 0` where flip is 0 or 0xFFFFFFFF.

### PIE Assembly (matmul_pie.S)

The PIE XACC kernel loads 16 int8 elements at a time into vector registers and accumulates the dot product in the 40-bit scalar accumulator (XACC). Key findings:

- XACC (40-bit scalar accumulator) is correct for dot products, NOT QACC (256-bit vector)
- Register t3 (x28) is reserved as the PIE address register -- do not use as a general-purpose temp
- PIE requires data in internal SRAM, not flash or PSRAM
- The kernel processes one row per call, caller iterates over rows

### Popcount

ESP32-P4 does NOT have the RISC-V Zbb extension (no hardware cpop instruction). The march flag confirms: `rv32imafc_zicsr_zifencei_xesppie` -- no `_zbb`.

Three approaches were tried:
1. `__builtin_popcount()` -- compiler generates software loop (~107 cycles per 32 bits)
2. Inline asm `"cpop %0, %1"` -- assembler error, Zbb not recognized
3. Raw encoding `.insn i 0x13, 1, %0, %1, 0x602` -- assembled but crashed at runtime (Illegal instruction, MCAUSE=2)

Final solution: the classic bit-parallel Hamming weight algorithm (~12 cycles per 32 bits):

```c
static inline int fast_popcount(uint32_t x)
{
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    return (x * 0x01010101u) >> 24;
}
```

### VoE (Violation of Expectation) Results

Averaged over 10 seeds, 80-frame trajectories, anomaly at step 40:

| Model | Normal | Teleport | Vel Flip | Gravity |
|-------|--------|----------|----------|---------|
| XNOR (MCU) | 0.584 | 6.038 | 0.749 | 0.608 |

Teleport detection ratio: 6.038 / 0.584 = 10.3x. The model reliably detects teleportation anomalies. Velocity flip and gravity inversion produce weaker but measurable signals.

These numbers are bit-exact across all optimization stages -- the BN folding and SRAM split do not change the mathematical result.

## Optimization Journey

### XNOR Model (655K params)

| Optimization | us/frame | Speedup | Key change |
|-------------|----------|---------|------------|
| Scalar C (no PIE) | 12,962 | 1.0x | Baseline |
| Per-row unpack + PIE | 7,295 | 1.8x | PIE for fc1, scalar fc2 |
| + fast_popcount | 6,560 | 2.0x | Replace __builtin_popcount |
| + SRAM fc2 | 2,826 | 4.6x | Eliminate flash cache misses on fc2 |
| + All SRAM, split pred_fc1 | 1,129 | 11.5x | No PSRAM, 2 PIE calls for pred_fc1 |

### Per-Layer Breakdown (Final)

| Operation | us/frame | % of total |
|-----------|----------|------------|
| enc_fc1 (PIE, SRAM) | 280 | 25% |
| enc_fc2 (XNOR, SRAM) | 287 | 25% |
| enc_bn_thr (int32) | 59 | 5% |
| enc_scale (float) | 5 | <1% |
| pred_fc1 (PIE, SRAM split) | 211 | 19% |
| pred_fc2 (XNOR, SRAM) | 148 | 13% |
| pred_bn_thr (partial float) | 97 | 9% |
| pred_scale (float) | 5 | <1% |
| Quantize (enc + pred) | 37 | 3% |
| **Total** | **1,129** | **100%** |

fc1 (PIE) and fc2 (XNOR) are roughly balanced. BN thresholds and quantization are 17% combined. The lack of hardware popcount costs ~435 us (38%) -- with Zbb cpop this would drop to ~50 us, giving ~740 us total.

### Comparison with INT8 Model (45K params, original LeWorldModel)

| Model | Params | us/frame | VoE teleport ratio |
|-------|--------|----------|---------------------|
| INT8 QAT (PIE XACC) | 45K | 78 | ~25x |
| XNOR-Net (PIE + popcount) | 655K | 1,129 | 10.3x |

The INT8 model is faster and has better anomaly detection. XNOR requires 14x more parameters for lower quality. This confirms the theoretical analysis: for models that fit in SRAM, INT8 + PIE is superior.

## Key Findings

1. **No MCU has hardware popcount.** ESP32-P4 (no Zbb), Cortex-M7 (no VCNT), Cortex-M55/M85 Helium (no VCNT in MVE). VCNT exists only in ARM Neon (Cortex-A, not MCU class). Software popcount at ~12 cycles/32 bits is universal.

2. **XNOR wins only when memory-bound.** For models exceeding SRAM, streaming INT8 weights from PSRAM at 400 MB/s makes PIE idle 93% of the time. XNOR weights are 8x smaller, reducing bandwidth demand. For SRAM-resident models, INT8 PIE at 16 MACs/cycle is 6x faster than software XNOR.

3. **INT4 is the optimal precision for large MCU models.** 2x less bandwidth than INT8, negligible quality loss, unpack cost hidden behind external memory latency with double-buffered DMA.

4. **ESP32-P4 PIE has 2-4x more INT8 throughput than any ARM MCU** (16 MACs/cycle vs 4 for Cortex-M7, 8 for Helium). But when memory-bound from external RAM, bandwidth is the bottleneck and compute advantage is wasted.

5. **SRAM fragmentation is a real engineering problem.** The ESP32-P4's 768 KB is split across multiple non-contiguous regions. Fitting a 192 KB weight matrix required a custom split allocator with two PIE calls.

## File Structure

```
main/
  main.c              -- Trajectory generation, VoE evaluation, timing
  inference_xnor.h    -- Model struct, function declarations
  inference_xnor.c    -- Init, encode, predict, BN folding, SRAM split
  matmul.h             -- PIE and XNOR matmul declarations
  matmul.c             -- XNOR + fast_popcount implementation
  matmul_pie.S         -- PIE XACC assembly kernel
  weights_xnor.h       -- Packed binary weights, BN params, scales
  act_scales_xnor.h    -- Calibrated activation scales
  model_config.h       -- Architecture constants
  physics.h/c          -- Bouncing ball simulation
```

## Building

Requires ESP-IDF v5.5.3 with ESP32-P4 support and PSRAM enabled.

```bash
idf.py set-target esp32p4
idf.py build
idf.py flash monitor
```

## References

- Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary Neural Networks", ECCV 2016
- LeCun et al., "LeWorldModel: Learning Predictive World Models", 2025
- ESP32-P4 Technical Reference Manual, Espressif Systems
- ESP-DL PIE assembly examples (esp-dl repository)

## License

Research use only. Model weights and inference code are provided as-is for educational purposes.
