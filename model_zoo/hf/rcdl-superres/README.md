---
license: bsd-3-clause
library_name: rcdl
pipeline_tag: image-to-image
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- super-resolution
- real-esrgan
---

# Real-ESRGAN Compact ×4 for RK3588: super-resolution

Compiled RKNN models (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no ONNX export, no calibration, no rknn-toolkit2. Built with
rknn-toolkit2 2.3.2, decoded and measured with
[**RCDL**](https://github.com/ruisv/rcdl). RCDL is a C++17 / Python library that
drives the NPU, RGA and VPU on these boards as one zero-copy pipeline.

Upstream: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
`realesr-general-x4v3`, the compact general-purpose ×4 model.

> [!TIP]
> **Redistributable, including commercially.** Real-ESRGAN is BSD-3-Clause, and
> these weights were released by its copyright holder. See [Licence](#licence).

## Files

| file | what it is | size |
|---|---|---|
| `realesr_general_x4v3_128_fp16_rk3588.rknn` | 128×128 tile → 512×512, fp16 (the default) | 3.5 MB |
| `realesr_general_x4v3_128_i8_rk3588.rknn` | the same, int8: about 1.6× faster and visibly noisier | 1.9 MB |

Each file upscales one fixed 128×128 RGB tile. RCDL's `SuperResolver` cuts a
frame of any size into overlapping tiles, runs one inference per tile, and
cross-fades the results back together. Both builds take 0..255: bytes for int8,
float32 still in 0..255 for fp16. The `/255` is inside the model.

## Measured on RK3588S

| build | agreement with the float ONNX | edge energy | one 128×128 tile |
|---|---|---|---|
| **fp16** (default) | **63.2 dB** | 55.3 (as the float model) | 70–82 ms |
| int8 | 31.5 dB | 66.4, past the original's 62.3 | 40–53 ms |

librknnrt 2.3.2, NPU driver 0.9.8, one NPU core. The cost is linear in tiles:
480×270 → 1920×1080 is 15 tiles and 1237 ms in fp16, 796 ms in int8.

**Judge this family by sharpness, not PSNR.** It is trained perceptually: it
invents plausible texture rather than the blur that minimises squared error.
Over six ×4 downscale-and-upscale pairs it scores *below* bicubic on PSNR while
being much sharper:

| | PSNR vs the original | mean gradient magnitude |
|---|---|---|
| bicubic ×4 | 28.35 dB | 40.5 |
| **the model** | **24.55 dB** | **55.3** |
| the original | — | 62.3 |

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
curl -LO https://huggingface.co/ruisv/rcdl-superres/resolve/main/realesr_general_x4v3_128_fp16_rk3588.rknn
```

```python
import cv2, rcdl

sr = rcdl.Engine("realesr_general_x4v3_128_fp16_rk3588.rknn").upscaler(overlap=16)
big = rcdl.upscale(sr, cv2.imread("small.jpg"))     # BGR uint8 in, BGR uint8 out
print(sr.scale, sr.tile, sr.last_tile_count)        # 4, 128, tiles the last call ran
```

## What to know before deploying

**fp16 is the default because int8's extra edge energy is not detail.** The int8
build's edges go past the original image's. That is quantization noise landing
on edges. Take int8 only when 1.6× matters more than that noise.

**Feed the fp16 build 0..255, not 0..1.** The model divides by 255 itself, and a
caller that normalises first gets a dark, wrong image. `SuperResolver` writes
whatever type the Engine reports. The trap only bites callers who drive the
Engine by hand.

## Licence

BSD-3-Clause, on the Real-ESRGAN code and on these weights, which its copyright
holder released.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and these compiled files* only.

How each build was made, and the input contract for every model, are
documented in RCDL's
[`docs/MODELS.md`](https://github.com/ruisv/rcdl/blob/main/docs/MODELS.md). The
licence status of every model RCDL has verified is in
[`model_zoo/`](https://github.com/ruisv/rcdl/tree/main/model_zoo).
