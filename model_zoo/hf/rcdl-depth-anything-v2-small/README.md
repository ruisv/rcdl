---
license: apache-2.0
library_name: rcdl
base_model:
- depth-anything/Depth-Anything-V2-Small
pipeline_tag: depth-estimation
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- depth-estimation
- monocular-depth
---

# Depth-Anything-V2-Small for RK3588: monocular relative depth

Compiled RKNN models (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no ONNX export, no calibration, no rknn-toolkit2. int8 builds from
rknn-toolkit2 2.3.2, decoded and measured with
[**RCDL**](https://github.com/ruisv/rcdl). RCDL is a C++17 / Python library that
drives the NPU, RGA and VPU on these boards as one zero-copy pipeline.

Upstream: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
**Small** (ViT-S encoder, DPT head).

> [!TIP]
> **Redistributable, including commercially.** Depth-Anything-V2-Small is
> Apache-2.0 on the code and on the weights. The Base and larger checkpoints are
> CC-BY-NC-4.0 and are **not** here. See [Licence](#licence).

## Files

| file | input | size |
|---|---|---|
| `depth_anything_v2_vits_308_rk3588.rknn` | 308×308 uint8 RGB (the default) | 28.3 MB |
| `depth_anything_v2_vits_rk3588.rknn` | 518×518 uint8 RGB, the network's native size | 31.9 MB |

The output is `[1, H, W]` **relative inverse depth**: disparity, where a larger
value is nearer. It is not distance, and there is no metric build of this
model. ImageNet mean/std are folded into the model, so the NPU takes raw bytes.

## Measured on RK3588S

Against the untouched fp32 ONNX, run on the identical letterboxed input:

| build | frame → depth map | NPU / CPU | cosine vs fp32 | Spearman | normalised MAE |
|---|---|---|---|---|---|
| 518×518 | 963 ms | 681 / 252 ms | 0.9839 | 0.9717 | 0.080 |
| **308×308** | **330 ms** | **142 / 108 ms** | **0.9894** | **0.9951** | **0.059** |

librknnrt 2.3.2, NPU driver 0.9.8, one NPU core.

**The smaller build wins on both speed and accuracy.** Speed: ViT attention is
quadratic in token count, and 308 has 22² patches against 518's 37². Accuracy is
the surprise: fewer, larger patches quantize better here. The CPU share is the
DPT head's `align_corners` upsamples, which run outside the NPU.

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
curl -LO https://huggingface.co/ruisv/rcdl-depth-anything-v2-small/resolve/main/depth_anything_v2_vits_308_rk3588.rknn
```

```python
import cv2, numpy as np, rcdl

dep = rcdl.Engine("depth_anything_v2_vits_308_rk3588.rknn").depth_estimator()
m = rcdl.estimate_depth(dep, cv2.imread("room.jpg"))    # mapped back to source pixels
disparity = np.asarray(m.data)                           # m.width x m.height
cv2.imwrite("depth.png", np.asarray(rcdl.depth_colorize(m)))
```

## What to know before deploying

**int8 keeps the order of depths, not their values.** Read the map as "which
pixel is nearer", and normalise per frame for display, which
`depth_estimator()` does by default (`normalize`). `depth_estimator(inverse=True)`
turns disparity back into a relative depth. Leave it off for visualisation,
because disparity is what the network was trained to be smooth in.

## Licence

Apache-2.0, on the Depth-Anything-V2 code and on the **Small** weights. The
Base and Large checkpoints are released under CC-BY-NC-4.0. A build made from
them would not be redistributable for commercial use, which is why only Small
is published.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and these compiled files* only.

How each build was made, and the input contract for every model, are
documented in RCDL's
[`docs/MODELS.md`](https://github.com/ruisv/rcdl/blob/main/docs/MODELS.md). The
licence status of every model RCDL has verified is in
[`model_zoo/`](https://github.com/ruisv/rcdl/tree/main/model_zoo).
