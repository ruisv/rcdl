# Models

RCDL consumes finished `.rknn` files. Conversion from ONNX (rknn-toolkit2, PTQ,
`accuracy_analysis`, hybrid quantization) runs on an **x86 host** and is owned by
a separate model-zoo project — the toolkit is x86-only, and the board carries
only the runtime.

`scripts/fetch_models.sh` stages the registry below into `models/` (gitignored).
Paths come from `scripts/local.env`; anything unset is reported as MISSING rather
than silently skipped.

```bash
scripts/fetch_models.sh --list     # show the registry
scripts/fetch_models.sh            # stage everything it can find
```

## What the decoders need to be told

Two model properties are **not** discoverable from the `.rknn` file and must be
recorded per model. Getting either wrong degrades accuracy silently rather than
failing:

| Property | Where it goes | How to tell |
|---|---|---|
| **Input channel order** (`rgb` vs `bgr`) | `PipelineConfig::model_input` | A property of the export (`--input-order`). The rknn_model_zoo YOLO exports are **RGB**; an OpenCV-native build is usually BGR. Wrong order costs a few points of confidence and shifts marginal detections. |
| **Whether the activation is in the graph** | `apply_sigmoid` / `apply_softmax` | Run the model once and look at the raw head output: values in [0,1] that behave like probabilities mean the export folded the sigmoid in. `looksLikeProbabilities()` in `tasks/classification.h` does this check. |

Everything else — grid sizes, class count, DFL `reg_max`, strides, NCHW vs NHWC,
quantization scale and zero point — is read **from the model** at construction by
`resolveYoloHead()` and its siblings, which throw with the full output signature
printed when a model does not match the head they were asked for. A mismatched
model fails loudly instead of decoding garbage.

## Registry

Measured on RK3588 (librknnrt 2.3.2, driver 0.9.8), converted with
rknn-toolkit2 2.3.2 from the airockchip model zoo.

| Model | Task | Input | Order | Head layout |
|---|---|---|---|---|
| `yolov8n_rk3588.rknn` | detection | 640×640 u8 | rgb | 9 outputs: per scale `box[1,64,H,W]`, `cls[1,80,H,W]`, `sum[1,1,H,W]`, NCHW. **Sigmoid in-graph.** |
| `yolo11n_rk3588.rknn` | detection | 640×640 u8 | rgb | same as yolov8n |
| `yolov8n-seg_rk3588.rknn` | instance seg | 640×640 u8 | rgb | 13 outputs: per scale `box(64)`, `cls(80)`, `sum(1)`, `mc(32)` + `proto[1,32,160,160]`, NCHW |
| `yolov8n-pose_rk3588.rknn` | pose | 640×640 u8 | rgb | 4 outputs: per scale **fused** `[1,65,H,W]` = 64 DFL + 1 class, plus `keypoints[1,17,3,8400]` f16. **Sigmoid applied on the CPU.** |
| `yolov8n-obb_rk3588.rknn` | oriented boxes | 640×640 u8 | rgb | 4 outputs: per scale **fused** `[1,79,H,W]` = 64 DFL + 15 DOTA classes, plus `angle[1,1,8400]` (sigmoid in-graph; decode as `(a − 0.25)·π`). **Class sigmoid on the CPU.** |
| `ppocrv4_det_rk3588.rknn` | text detection | DBNet probability map | — | thresholded map → contours → unclipped quads |
| `ppocrv4_rec_rk3588.rknn` | text recognition | CRNN | — | CTC greedy decode against `data/ppocr_keys_*.txt` |
| `ppocr_cls_rk3588.rknn` | text-line direction | 192×48 u8 | **bgr** | `[1,2]` f16, **softmax in-graph**: argmax is the label (0 upright / 1 rotated 180°), its value the score |
| `ppocrv5_det_rk3588.rknn` | text detection | 480×480 u8 | rgb | as the v4 detector, and measurably the same on the sample page — see below |
| `ppseg_rk3588.rknn` | semantic seg | 512×512 u8 | rgb | `[1,19,512,512]` NCHW int8 **logits** (PP-LiteSeg, Cityscapes 19 classes) — argmax on the CPU |
| `retinaface_rk3588.rknn` | face + 5 landmarks | 320×320 u8 | **bgr** | anchor-based SSD head: `boxes[1,4200,4]`, `scores[1,4200,2]` (**softmax in-graph**), `landmarks[1,4200,10]`. 4200 priors = `(40²+20²+10²)×2` from `steps {8,16,32}`, `min_sizes {{16,32},{64,128},{256,512}}`, variances `[0.1, 0.2]` |
| `resnet18_rk3588.rknn` | classification | 224×224 u8 | rgb | `[1,1000]`, softmax on the CPU |
| `depth_anything_v2_vits_308_rk3588.rknn` | monocular depth | 308×308 u8 | rgb | `[1,308,308]` int8 — **relative inverse depth (disparity, big = near)**, no activation. ImageNet mean/std baked into the `.rknn` preprocessing, so the NPU takes raw u8 RGB |
| `depth_anything_v2_vits_rk3588.rknn` | monocular depth | 518×518 u8 | rgb | as above at the network's native resolution; slower **and** less accurate — see below |
| `osnet_x0_25_msmt17_rk3588.rknn` | appearance embedding (person ReID) | 128×256 u8 | rgb | `[1,512]` int8, L2-normalised on read-out. Crops are **squashed, not letterboxed** — see below |

**PP-OCRv5: the detector ships, the recogniser cannot.** Both were converted
from the PaddleOCR 3.x mobile exports with the same recipes used for v4.

*Detection* is a straight non-regression: 480×480 int8, and on
`data/images/ocr.jpg` it finds the same 16 regions as the v4 build, which yield
the same 15 strings at the same mean confidence. No improvement is demonstrable
on one page, so that is all this claims; the parity is pinned by a test so a
future divergence surfaces.

*Recognition* is the interesting failure, and it is a hardware one:

| build | result on the 16 sample lines |
|---|---|
| Paddle fp32 (the original) | 16/16 exact, scores 0.95–0.999 |
| RKNN float16, **toolkit simulator** | matches the original, mean winning probability 0.9997 |
| RKNN float16, **on the NPU** | **1/16 exact**, mean 0.28, characters dropped mid-line |
| RKNN int8 | 0/16, mean 0.21 |
| RKNN int16 | `rknn_run` fails to submit at the first Conv on this driver |

The graph converts correctly — the simulator proves that — and the board still
gets it wrong: fed the identical input tensor, the NPU returns a flattened
softmax (peak 0.47 where the simulator has 1.00) and the argmax at the first
character collapses to the CTC blank. An 18385-class head in float16 is past
what this runtime keeps accurate, and neither quantization escape is open. So
**PP-OCRv4 remains the deployed recogniser** and the v5 build is not in the
registry. This is the sharper form of a lesson already in this file: a model
that compiles, and even one that a simulator reproduces exactly, is not
therefore a model that works.

**The direction classifier is fed differently from everything else, and it
matters.** PP-OCR fits a line crop to the model's HEIGHT, caps the width at the
input's, anchors it top-left and pads the remainder — `ocrLineFitWidth()`. Feed
it the centred letterbox the rest of the library uses and, measured on the 16
lines of `data/images/ocr.jpg`, it drops from **16/16 orientations right (mean
confidence 0.98) to 9/16 upright and 11/16 rotated (0.78)**. Nothing errors; the
model is simply looking at a thin band of text between two thick bars, which is
not what it was trained on.

The model is `ch_ppocr_mobile_v2.0_cls`, exported to ONNX with paddle2onnx and
quantized int8 against 96 line crops cut out of the sample page by RCDL's own
PP-OCRv4 detector — both orientations, plus a brightness and a blur variant of
each. Validated against the **Paddle fp32 original on the same crops**: on a
held-out split (calibrate on 10 of the 16 lines, measure on the other 6) the
int8 build agrees with Paddle on **100% of labels with a probability MAE of
0.0000 (max 0.0003)**; over all 96 crops, 100% of labels and MAE 0.003. Both
agree with the intended orientation on every line except the 17×73 vertical
strip, which is a column of stacked glyphs rather than a text line and which
both score ~0.6 either way — below the 0.9 flip gate, which is the point of
having one.

**Why the head is in the pipeline at all**: a CTC recogniser fed an upside-down
line does not fail. On this page the rotated lines come back as the empty string
or a single stray bracket at score ≤ 0.6 — the content is gone, with nothing in
the result to say so. With the classifier in front, all of them read exactly as
they do upright.

**RetinaFace wants BGR.** The vendor's Python demo converts BGR→RGB before
inference, but the converted `.rknn` does not behave that way: fed RGB it finds
only one of the two faces in `zidane.jpg` (the second peaks at 0.129, below any
usable threshold); fed BGR it finds both at 0.995 and 0.949. It does not error
and the landmarks do not scatter — it just silently loses most of the recall.
This is exactly the class of thing the "input channel order" column exists for.

**OSNet crops are squashed, never letterboxed.** A ReID tower is trained on
detection crops resized straight to 128×256, aspect ratio thrown away. Feeding
it an aspect-preserving letterbox instead puts grey bars in an input that has
never seen any, and the failure is quiet: the vectors still come out unit-length
and still look like embeddings, they are just less separable. `ImageEmbedder`
squashes for this reason and returns no `LetterboxInfo` — an embedding has no
coordinates, so there is no geometry to invert.

Measured on the four people in `bus.jpg`, int8, 13.5 ms per crop (9.9 ms of it
NPU):

| Pair | Cosine |
|---|---|
| Different people | 0.38 – 0.47 |
| Same person, crop 4% tighter | 0.960 |
| Same person, from a 2× upscaled frame | 0.993 |

Against the fp32 ONNX on the same crops, the int8 vectors land at cosine 0.9922
(min 0.9874) and preserve the pairwise-similarity structure (Pearson 0.9928,
largest single similarity shifted by 0.027). As the header in `tasks/embedding.h`
warns, an int8 embedding is coarser than the float model's, so **any similarity
threshold must be measured with the same quantized model** rather than taken
from a paper.

**Depth-Anything-V2-Small emits disparity, not depth.** The head is relative
*inverse* depth: large values are near, the scale is arbitrary, and the only
sensible presentation is the per-frame min-max normalise `DepthConfig::normalize`
does by default. `DepthConfig::inverse` turns it back into a depth if you want
one; leave it off for visualisation, because disparity is what the network was
trained to be smooth in. There is no metric build of this model — do not read
its output as distance.

**The 308×308 build beats the native 518×518 one on both axes.** This is
counter-intuitive enough to be worth the measurement. Against the untouched fp32
ONNX run on the *identical* letterboxed input:

| Build | Frame (end to end) | NPU / CPU split | cosine vs fp32 | Spearman | normalised MAE |
|---|---|---|---|---|---|
| 518×518 | 963 ms | 681 / 252 ms | 0.9839 | 0.9717 | 0.080 |
| **308×308** | **330 ms** | **142 / 108 ms** | **0.9894** | **0.9951** | **0.059** |

The speed comes from ViT attention, which is quadratic in token count
(37² → 22² patches). The *accuracy* is the surprise: fewer, larger patches
quantize better here, so the smaller build is the one the tests and examples
use. The 518 build stays in the registry for anyone who needs the native
resolution.

Two further measured facts about this model:

* **A quarter of the frame is CPU, and it is the DPT head's upsamples.** Five
  `Resize` layers run on the CPU because they are `mode=linear` with
  `coordinate_transformation_mode=align_corners`, which RKNPU2 has no NPU
  implementation for. Rewriting them to `half_pixel` in the ONNX does move four
  of the five onto the NPU (330 → 290 ms), but it costs real accuracy —
  cosine 0.9894 → 0.9742, Spearman 0.9951 → 0.9648 — so **it is not what ships**.
  The fifth resize is a 1.75× scale that is structural to the DPT head at any
  input size, and no NPU path exists for it.
* **int8 compresses the top of the range.** On `bus.jpg` the fp32 reference
  spans [0.06, 8.51] where the NPU gives [0.00, 6.68]. Rank order is what
  survives quantization (Spearman 0.995), absolute values are not — another
  reason the decoder normalises per frame.

**The letterbox padding gets a depth too.** The model sees a square canvas, so a
non-square frame arrives padded, and the network happily predicts depth for the
grey bars. `DepthEstimator::postprocess(lb)` projects only the real image region
back onto the frame, which is why the normalised map does not reach 1.0 when the
brightest disparity landed in the padding.

**Note the pose and OBB exports differ structurally from detection**: they
concatenate the box and class channels into one tensor per scale, and they do
*not* fold the class sigmoid into the graph. That is not a quirk of one export —
it is what the vendor's own reference implementation reads — so the decoders
support both the fused and the separate-branch shapes and choose from the
model's output signature.

## Measured performance

RK3588S, single-frame latency unless stated:

| Workload | Result |
|---|---|
| ResNet-18 int8, one NPU core | 4.0 ms |
| ResNet-18 int8, three pinned contexts | 707 fps aggregate |
| Detection pipeline, ResNet-18 canvas, synchronous | 99 fps |
| Same, `AsyncDetectionPipeline` with 3 workers | 481 fps (4.86×) |
| 1080p H.264 decode | 324 fps |
| 4K H.265 decode | 244 fps |
| 1080p H.264 encode from decoded dma-bufs | 197 fps |
| decode → encode → decode luma PSNR | 47.2 dB |
| JPEG decode (810×1080) | 4.1 ms; matches libjpeg to 0.02 mean luma |
| YOLOv8n 640, full VPU → RGA → NPU → overlay → VPU per frame | 26.9 ms (37.2 fps); post-processing is 4.1 ms of that, down from 29.9 ms before the score-sum pre-filter |
| PP-LiteSeg 2048×1024 → 512×512 → full-res label map | 77 ms (dominated by the CPU upscale of the label map, not the NPU) |
| Depth-Anything-V2-Small 308×308, frame → depth in source pixels | 330 ms (142 ms NPU, 108 ms CPU — the CPU part is the DPT head's `align_corners` upsamples, see above) |
| Same at the native 518×518 | 963 ms |
| OSNet x0.25 ReID, detection box → 512-d vector | 13.5 ms per crop (9.9 ms NPU) |

## Adding a model

1. Convert it on the x86 host and copy the `.rknn` to wherever
   `RCDL_CONVERT_MODELS` points.
2. Add a `name|convert|path` line to `REGISTRY` in `scripts/fetch_models.sh`.
3. Add a row to the table above, **including the input order and where the
   activation lives** — that is the information the file itself does not carry.
4. Add a board test that skips cleanly when the model is absent
   (`tests/board_models.py::require_model`), so the suite stays green on a
   machine that has only some of the registry staged.

## A note on class-name tables

`segmentation.h` ships **VOC-21** names and palette, but `ppseg_rk3588.rknn` is a
**Cityscapes-19** model — the label indices are correct either way (argmax is
argmax), but the names and colours will not match that model. Pass your own
table rather than trusting `vocClassName()` for a model that is not VOC-trained.
The same caution applies to `cocoClassName()` for any detector not trained on
COCO, and to the DOTA-15 names for OBB.
