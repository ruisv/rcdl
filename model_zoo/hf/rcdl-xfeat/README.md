---
license: apache-2.0
library_name: rcdl
pipeline_tag: image-feature-extraction
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- xfeat
- local-features
- feature-matching
---

# XFeat for RK3588: sparse keypoints, descriptors and matching

A compiled RKNN model (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no ONNX export, no calibration, no rknn-toolkit2. An int8 build from
rknn-toolkit2 2.3.2, decoded and measured with
[**RCDL**](https://github.com/ruisv/rcdl). RCDL is a C++17 / Python library that
drives the NPU, RGA and VPU on these boards as one zero-copy pipeline.

Upstream: [XFeat](https://github.com/verlab/accelerated_features) (CVPR 2024),
accelerated features for lightweight image matching.

> [!TIP]
> **Redistributable, including commercially.** XFeat is Apache-2.0, and its
> weights are committed in the same Apache-2.0 repository. See
> [Licence](#licence).

## Files

| file | what it is | size |
|---|---|---|
| `xfeat_640x480_i8_rk3588.rknn` | 640×480 grey input → dense descriptor, keypoint and reliability maps | 1.3 MB |

The three outputs are `feats [1,64,60,80]`, `keypoints [1,65,60,80]` and
`reliability [1,1,60,80]`, int8, NCHW. RCDL's decoder turns them into up to 4096
keypoints with 64-d descriptors. Positions are in the original frame's pixels,
and descriptor rows are L2-normalised.

## Measured on RK3588S

Scored against a warp that is known exactly: a photograph rotated 12°, scaled
0.85 and shifted, so every keypoint has one correct answer.

| build | keypoints | matches | within 3 px | median error |
|---|---|---|---|---|
| ONNX float (CPU) | 4096 + 4096 | 2192 | 77.4% | 1.35 px |
| RKNN int8, toolkit simulator | 4096 + 4096 | 2022 | 77.5% | 1.34 px |
| **this file, on the NPU** | 4096 + 4096 | 2033 | **76.6%** | **1.36 px** |

Descriptors reach cosine 0.9978 (minimum 0.9910) against the float model's, on
the 3348 keypoints both builds detected at the same pixel. As a control, two
*different* scenes give 69 mutual matches, where the warped pair gives 2033.

| stage | time |
|---|---|
| extract: preprocessing, NPU and decode | 56–85 ms, of which 35 ms is the NPU |
| match 4096 × 4096 | 213–243 ms on 8 threads (824 ms on one) |

librknnrt 2.3.2, NPU driver 0.9.8, one NPU core.

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
curl -LO https://huggingface.co/ruisv/rcdl-xfeat/resolve/main/xfeat_640x480_i8_rk3588.rknn
```

```python
import cv2, rcdl

e  = rcdl.Engine("xfeat_640x480_i8_rk3588.rknn", float_inputs=[0])
ex = e.feature_extractor()                         # config=rcdl.XfeatConfig()

fa = rcdl.extract_features(ex, cv2.imread("a.jpg"))
fb = rcdl.extract_features(ex, cv2.imread("b.jpg"))
pairs, cosines = rcdl.match_features(fa, fb)       # (M,2) indices, (M,) cosines
print(len(fa), len(fb), "features,", len(pairs), "mutual matches")
# fa.xy[pairs[:, 0]] and fb.xy[pairs[:, 1]] go straight into cv2.findHomography
```

## What to know before deploying

**`float_inputs=[0]` is not optional.** The graph starts after XFeat's
InstanceNorm, so its input is a normalised map, not image bytes. RCDL computes
that map on the CPU and writes it as float. `FeatureExtractor` refuses an
Engine opened without the flag rather than feeding it bytes.

**Matching costs more than inference.** It is `O(|a|·|b|·64)`, about 10⁹
multiply-adds at 4096 keypoints a side. `XfeatConfig.top_k` decides whether a
pair costs milliseconds or a quarter of a second. Fewer points are not free:
`top_k = 1024` cut matching to 72 ms, but known-warp agreement fell to 62.7%.
The highest-scoring points are also the most repetitive ones.

**int8 is measured equivalent to float here.** That is a finding for this
model, not a rule. Several other networks in RCDL's registry ship fp16 because
int8 broke them.

## Licence

Apache-2.0, on the XFeat code and on the weights committed in its repository.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and this compiled file* only.

How the build was made, and the input contract for every model, are documented
in RCDL's
[`docs/MODELS.md`](https://github.com/ruisv/rcdl/blob/main/docs/MODELS.md). The
licence status of every model RCDL has verified is in
[`model_zoo/`](https://github.com/ruisv/rcdl/tree/main/model_zoo).
