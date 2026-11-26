---
license: apache-2.0
library_name: rcdl
base_model:
- google/siglip-base-patch16-224
pipeline_tag: zero-shot-image-classification
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- siglip
- image-embedding
---

# SigLIP base/16 for RK3588: image embeddings and zero-shot classification

A compiled RKNN model (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no ONNX export, no rknn-toolkit2. It is the **image** tower of SigLIP,
fp16, built with rknn-toolkit2 2.3.2. It comes with a text-embedding table for
the 80 COCO labels, so zero-shot classification works out of the box. Decoded
and measured with [**RCDL**](https://github.com/ruisv/rcdl). RCDL is a C++17 /
Python library that drives the NPU, RGA and VPU on these boards as one
zero-copy pipeline.

Upstream: [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224).

> [!TIP]
> **Redistributable, including commercially.** Google releases SigLIP under
> Apache-2.0, and the text table is that same model's output. See
> [Licence](#licence).

## Files

| file | what it is | size |
|---|---|---|
| `siglip_b16_224_fp16_rk3588.rknn` | image encoder: 224×224 RGB → 768-d unit vector | 182.9 MB |
| `siglip_b16_224_text_coco80.npy` | text embeddings of the 80 COCO labels, `[80, 768]` | 240 KB |
| `siglip_b16_224_labels.txt` | the 80 labels, one per line, in the table's row order | 1 KB |

Only the image tower runs on the board. Once a label set is chosen, the text side
is a fixed table: it was computed on the conversion host with SigLIP's own prompt
form, `a photo of a {label}.`, which scores measurably better than a bare noun.
Zero-shot classification is then one dot product per label.

## Measured on RK3588S

| frame | top three labels (cosine) |
|---|---|
| `bus.jpg` | **bus +0.092**, car +0.040, train +0.031 |
| `bird.jpg` | **bird +0.080**, kite +0.016, elephant +0.010 |
| `space_shuttle_224.jpg` | **airplane +0.055**, boat +0.034, car +0.021 |

One frame to one embedding takes 96 ms (librknnrt 2.3.2, NPU driver 0.9.8, one
NPU core). The raw cosines are small for this model family. What matters is
the margin over the runner-up.

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
hf download ruisv/rcdl-siglip-b16 --local-dir siglip    # or curl each file from resolve/main/
```

```python
import cv2, numpy as np, rcdl

emb = rcdl.Engine("siglip/siglip_b16_224_fp16_rk3588.rknn").embedder()
table = np.load("siglip/siglip_b16_224_text_coco80.npy")                  # [80, 768]
labels = [l.strip() for l in open("siglip/siglip_b16_224_labels.txt") if l.strip()]

v = np.asarray(rcdl.embed(emb, cv2.imread("bus.jpg")), np.float32).reshape(-1)
v /= np.linalg.norm(v)
sims = table @ v
for i in np.argsort(-sims)[:3]:
    print(labels[i], f"{sims[i]:+.3f}")
```

`rcdl.embed(emb, img, box=...)` embeds a crop instead of the whole frame.

## What to know before deploying

**This is the fp16 build because int8 destroys it.** The int8 build compiles,
runs and returns well-formed unit vectors. Every label then lands within noise
of every other: `bus.jpg` comes back as `skateboard +0.005`, with `suitcase` and
`handbag` a thousandth behind. Nothing errors, and the embedding space has
collapsed. This is how a ViT typically behaves under post-training quantization.

**Another vocabulary needs no reconversion.** Swap the table, not the model:
run the text tower of the same checkpoint on any machine over your labels, with
the same prompt form, and save the embeddings in your label file's row order.

## Licence

Apache-2.0, as Google releases SigLIP. The text table and labels are derived
from that same model and carry the same terms.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and these compiled files* only.

How the build was made, and the input contract for every model, are documented
in RCDL's
[`docs/MODELS.md`](https://github.com/ruisv/rcdl/blob/main/docs/MODELS.md). The
licence status of every model RCDL has verified is in
[`model_zoo/`](https://github.com/ruisv/rcdl/tree/main/model_zoo).
