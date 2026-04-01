# mcu-leworldmodel — ESP32-P4

JEPA LeWorldModel inference on the ESP32-P4 using PIE SIMD.

Four quantisation levels (FP32, INT8, ternary, binary) running the same
bouncing ball world model. FP32 uses scalar float matmul. INT8, ternary,
and binary all use the PIE `esp.vmulas.s8.qacc` instruction (16 INT8 MACs
per cycle) with on-the-fly activation quantisation.

## Setup

### 1. Export weights from Python

Run the export cell in `jepa_quantization.ipynb`. This produces:
- `weights_fp32.h` — float arrays
- `weights_int8.h` — int8 arrays + per-row scales

For the PIE path, ternary and binary weights must be exported as **unpacked
int8 arrays** (not bit-packed). Add this cell to the notebook and run it:

```python
# Export ternary and binary as unpacked int8 for PIE SIMD
for qname in ['ternary', 'binary']:
    info = BEST_MODELS[qname]
    enc, pred = info['encoder'], info['predictor']
    modules = {'encoder': enc, 'predictor': pred}

    lines = f'#pragma once\n\n#include <stdint.h>\n#include "model_config.h"\n\n'

    for mod_name, layer_name, c_prefix in LAYER_NAMES:
        layer = getattr(modules[mod_name], layer_name)
        w = layer.weight.detach().cpu().float()

        # Quantise to int8 values
        scale = (w.pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt()
        if qname == 'ternary':
            threshold = 0.5 * scale
            q = torch.sign(w) * (w.abs() > threshold).float()
        else:
            q = torch.sign(w).float()

        q_int8 = q.to(torch.int8)
        flat_q = q_int8.reshape(-1).numpy()
        flat_s = scale.squeeze(-1).numpy()

        rows, cols = w.shape
        w_vals = ', '.join(str(int(v)) for v in flat_q)
        s_vals = ', '.join(f'{v:.8f}f' for v in flat_s)

        lines += f'static const int8_t {c_prefix}_weight[{len(flat_q)}] = {{\n    {w_vals}\n}}; // ({rows}, {cols})\n'
        lines += f'static const float {c_prefix}_scale[{len(flat_s)}] = {{\n    {s_vals}\n}};\n\n'

    path = os.path.join(EXPORT_DIR, f'weights_{qname}_pie.h')
    with open(path, 'w') as f:
        f.write(lines)
    print(f'Saved: {path}  ({os.path.getsize(path):,} bytes)')
```

### 2. Copy headers

```bash
cp exported_models/model_config.h        main/
cp exported_models/weights_fp32.h        main/
cp exported_models/weights_int8.h        main/
cp exported_models/weights_ternary_pie.h main/
cp exported_models/weights_binary_pie.h  main/
```

### 3. Build and flash

```bash
source ~/.espressif/v5.5.3/esp-idf/export.sh

# First time only
idf.py set-target esp32p4

# Build
idf.py build

# Flash (adjust port for your board)
cd build && python -m esptool --chip esp32p4 --port /dev/cu.usbmodem1101 \
    --baud 921600 write_flash @flash_args
```

### 4. Monitor

```bash
idf.py -p /dev/cu.usbmodem1101 monitor
```

## How It Works

### Inference Pipeline

```
observation (float32, 256 values)
  → quantise activation to int8 (find absmax, scale, round, clamp)
  → PIE int8 × int8 MAC (esp.vmulas.s8.qacc, 16 elements/cycle)
  → dequantise int32 → float32 (weight_scale * act_scale * accumulator)
  → ReLU
  → repeat for next layer
```

FP32 uses scalar float matmul (no PIE). The three quantised variants
(INT8, ternary, binary) all use the same PIE assembly — the SIMD
processes all elements regardless of sparsity. Ternary zeros cost nothing
extra because `esp.vmulas.s8.qacc` multiplies all 16 lanes every cycle.

### PIE Assembly

Adapted from `mcu-ternary-matmul` Experiment 3 optimised assembly:
- **Cached activation path** (cols == 64): load x once into q4-q7
- **Bulk + tail path** (cols > 16): 4-pair unrolled (64 elements/iter)
  + single-pair tail

### Model Architecture

```
Encoder:    256  → Linear → 128  → ReLU → Linear → 32
Predictor:  96   → Linear → 64   → ReLU → Linear → 32
                    (3 × 32 concatenated history)
```

45K parameters. At INT8: 44 KB. Fits in <6% of ESP32-P4 SRAM.

## Files

```
main/
  model_config.h         Architecture constants (from Python)
  weights_*.h            Weight arrays (from Python)
  physics.h/c            Ball simulation and rasterisation
  inference.h/c          Encode, predict, MSE
  matmul.h/c             FP32 scalar + PIE int8 wrapper
  matmul_pie.S           PIE assembly (esp.vmulas.s8.qacc)
  model_fp32.c           FP32 model init
  model_int8.c           INT8 model init (PIE)
  model_ternary.c        Ternary model init (PIE, int8 storage)
  model_binary.c         Binary model init (PIE, int8 storage)
  main.c                 app_main, VoE evaluation, timing
  CMakeLists.txt
CMakeLists.txt           Root ESP-IDF project file
```

## Hardware

- ESP32-P4, RISC-V HP at 360 MHz (v1.x silicon)
- 768 KB SRAM
- PIE: 8 × 128-bit Q registers, esp.vmulas.s8.qacc (16 INT8 MACs/cycle)
