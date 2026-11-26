---
license: apache-2.0
library_name: rcdl
base_model:
- PaddlePaddle/PP-OCRv4_mobile_det
- PaddlePaddle/PP-OCRv4_mobile_rec
- PaddlePaddle/PP-OCRv5_mobile_det
- PaddlePaddle/PP-OCRv5_server_rec
- PaddlePaddle/PP-OCRv6_medium_det
- PaddlePaddle/PP-OCRv6_medium_rec
pipeline_tag: image-to-text
tags:
- rknn
- rockchip
- rk3588
- npu
- edge
- ocr
- text-detection
- text-recognition
---

# PP-OCR v4 · v5 · v6 for RK3588: text detection, direction and recognition

Compiled RKNN models (`.rknn`) for the **Rockchip RK3588 / RK3588S** NPU, ready
to load: no Paddle export, no calibration, no rknn-toolkit2. Built with
rknn-toolkit2 2.3.2, decoded and measured with
[**RCDL**](https://github.com/ruisv/rcdl). RCDL is a C++17 / Python library that
drives the NPU, RGA and VPU on these boards as one zero-copy pipeline.

Upstream: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): PP-OCRv4
mobile, PP-OCRv5 (mobile detection, server recognition) and PP-OCRv6 medium.

> [!TIP]
> **Redistributable, including commercially.** PaddleOCR is Apache-2.0 on the
> code and on every one of these models' cards. See [Licence](#licence).

## Files

| file | what it is | size |
|---|---|---|
| `ppocrv6_medium_det_rk3588.rknn` | text detection (DBNet), 480×480, int8 | 17.0 MB |
| `ppocrv6_medium_rec_logits_rk3588.rknn` | text recognition, 48×320, fp16, emits logits | 38.3 MB |
| `ppocrv5_det_rk3588.rknn` | text detection, v5 mobile, 480×480, int8 | 2.5 MB |
| `ppocrv5_server_rec_logits_rk3588.rknn` | text recognition, v5 server, 48×320, fp16, emits logits | 42.2 MB |
| `ppocrv4_det_rk3588.rknn` | text detection, v4 mobile, int8 | 2.5 MB |
| `ppocrv4_rec_rk3588.rknn` | text recognition, v4 mobile (CRNN) | 6.7 MB |
| `ppocr_cls_rk3588.rknn` | text-line direction, 0° or 180°, 192×48 | 0.8 MB |
| `ppocr_keys_v6_18710.txt` | dictionary for the v6 recogniser | 73 KB |
| `ppocr_keys_v5_18385.txt` | dictionary for the v5 recogniser | 72 KB |
| `ppocr_keys_v1_6625.txt` | dictionary for the v4 recogniser | 26 KB |

The three generations are complete stacks: a detector, then one recognition per
detected line. The direction classifier is shared and goes between the two.

## Measured on RK3588S

| | v4 | v5 | v6 |
|---|---|---|---|
| detection, one frame (NPU) | 42 ms | as v4 | 62 ms |
| recognition, one line | 36 ms | 88 ms | 40 ms |
| mean line confidence on the sample page | 0.661 | 0.928 | 0.930 |

librknnrt 2.3.2, NPU driver 0.9.8, one NPU core. All three detectors find the
same regions on the sample page, and the v5 detector measures the same as v4.
The difference is in the recognisers: v5 and v6 agree on the one character v4
reads differently, and v6 is the most confident and close to v4's speed.

## Use it

On the board (linux-aarch64):

```bash
conda create -n rcdl -c https://mirrors.ruis.ai/conda -c conda-forge rcdl
conda activate rcdl
hf download ruisv/rcdl-ppocr --local-dir ppocr    # or curl each file from resolve/main/
```

```python
import cv2, numpy as np, rcdl

img = cv2.imread("page.jpg")

cfg = rcdl.OcrDetConfig()
cfg.bin_thresh, cfg.box_thresh, cfg.unclip_ratio = 0.2, 0.45, 1.4   # PP-OCRv6's own DB constants
det = rcdl.Engine("ppocr/ppocrv6_medium_det_rk3588.rknn").text_detector(config=cfg)

rcfg = rcdl.OcrRecConfig()
rcfg.apply_softmax = True        # the softmax is deliberately not in this graph
rec = rcdl.Engine("ppocr/ppocrv6_medium_rec_logits_rk3588.rknn").text_recognizer(
    "ppocr/ppocr_keys_v6_18710.txt", config=rcfg,
    input_scale=1.0, input_shift=0.0, model_input="rgb888")

for box in rcdl.detect_text(det, img, "bgr888"):
    pts = np.asarray(box.pts, np.float32).reshape(4, 2)
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[3] - pts[2])))
    h = int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])))
    if w < 4 or h < 4 or h > w:   # a vertical line needs rotating upright first
        continue
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w, h))
    line = rcdl.recognize_text(rec, crop, "bgr888")
    print(f"{line.score:.2f}", line.text)
```

When lines may be upside down, run the direction classifier on each crop before
recognising it:

```python
angle = rcdl.Engine("ppocr/ppocr_cls_rk3588.rknn").text_angle_classifier()
r = angle.process(np.ascontiguousarray(crop), crop.shape[1], crop.shape[0], "bgr888")
if r.flip180:
    crop = cv2.rotate(crop, cv2.ROTATE_180)
```

## What to know before deploying

**The v5 and v6 recognisers emit logits, on purpose.** The 18k-way softmax is
the one operation that does not survive this runtime's fp16 path. In the graph,
its output sums to anywhere between 0.64 and 1.00 per step and has no peak.
CTC needs only the argmax, so the softmax moved to the CPU. Decode with
`apply_softmax=True`, and feed raw 0..255 (`input_scale=1, input_shift=0`).

**The dictionary must match the recogniser.** v6 has 18710 classes, v5 18385 and
v4 6625. A recogniser paired with the wrong file decodes to plausible-looking
wrong characters rather than failing. The right file for each one ships here.

**Use v6's own detection thresholds.** PP-OCRv6 publishes its own DB constants
(`bin 0.2, box 0.45, unclip 1.4`). The library defaults are the earlier
generations' constants.

**The direction classifier is not optional on rotated input.** A CTC recogniser
fed an upside-down line does not fail; it returns confident nonsense. The
classifier takes BGR at 192×48, with the softmax in the graph.

## Licence

Apache-2.0, on the PaddleOCR code and on every model card these builds come
from. The dictionaries are PaddleOCR's own files.

**RCDL itself is Apache-2.0 and is unrelated to these terms.** It is a
general-purpose runtime that loads any `.rknn`. The licence above applies to
*these weights and these compiled files* only.

How each build was made, and the input contract for every model, are
documented in RCDL's
[`docs/MODELS.md`](https://github.com/ruisv/rcdl/blob/main/docs/MODELS.md). The
licence status of every model RCDL has verified is in
[`model_zoo/`](https://github.com/ruisv/rcdl/tree/main/model_zoo).
