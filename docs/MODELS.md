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
| `yolo26n_rk3588.rknn` | detection | 640×640 u8 | rgb | 9 outputs, same shape as yolov8n except the box branch is **4 channels, not 64: this head has no DFL**. Exported from the NMS-free head's one2one branch |
| `yolo26n-seg_rk3588.rknn` | instance seg | 640×640 u8 | rgb | 13 outputs, the v8-seg layout with 4-channel box branches. Prototypes are built from **all three scales**, not P3 alone |
| `yolo26n-pose_rk3588.rknn` | pose | 640×640 u8 | rgb | 9 outputs: per scale box(4), cls(1, sigmoid on the CPU), keypoints(51) raw. **Decode with `kpt_decode="cell_relative_whole"`** — see below |
| `yolo26n-obb_rk3588.rknn` | oriented boxes | 640×640 u8 | rgb | 9 outputs: per scale box(4), cls(15), angle(1). **The angle is in RADIANS** — decode with `angle_bias=0`, `angle_scale=1` |
| `yolo26n-cls_rk3588.rknn` | classification | 224×224 u8 | rgb | `[1,1000]` with the **softmax IN the graph** — pass `apply_softmax=False` |
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
| `xfeat_640x480_i8_rk3588.rknn` | sparse local features | 640×480 **f32** | — (grey) | 3 outputs: `feats[1,64,60,80]`, `keypoints[1,65,60,80]`, `reliability[1,1,60,80]`, all int8 NCHW. **The input is a normalised map, not image bytes** — see below |
| `realesr_general_x4v3_128_fp16_rk3588.rknn` | ×4 super-resolution | 128×128 **f32** | rgb | `[1,3,512,512]` f16 NCHW in [0,1]. Input is float32 **still in 0..255** — the ÷255 is in the model |
| `realesr_general_x4v3_128_i8_rk3588.rknn` | ×4 super-resolution | 128×128 u8 | rgb | as above, int8. ~1.6× faster, 31 dB from the float model — see below |
| `neuflow_v2_512x384_fp16_rk3588.rknn` | dense optical flow | 512×384 ×2 **f32** | **bgr** | `[1,2,384,512]` f16 NCHW, pixels (+u right, +v down). Inputs are 0..255 BGR floats. **Needs a custom operator** — see below |
| `edge_sam_3x_encoder_fp16_rk3588.rknn` | promptable seg — encoder | 1024×1024 **f32** | rgb | `[1,256,64,64]` f16 embedding. Input is float32 in 0..255; SAM's mean/std are folded into the `.rknn` |
| `edge_sam_3x_decoder_fp16_rk3588.rknn` | promptable seg — decoder | embedding + 2 prompt points | — | `scores[1,4]`, `masks[1,4,256,256]` f16 logits. The embedding input is **NHWC** where the encoder's output is NCHW |
| `rtmw_s_133_256x192_fp16_rk3588.rknn` | whole-body pose (133 kpts) | 192×256 **f32** | rgb | `simcc_x[1,133,384]` + `simcc_y[1,133,512]` f16. Top-down: one person per inference. Input is float32 in 0..255 |
| `yoloe_11s_coco80_rk3588.rknn` | open-vocab detection | 640×640 u8 | rgb | 9 outputs, the yolov8n layout with `cls[1,80,H,W]`. The class axis is **80 words chosen at conversion time**; pair it with `yoloe_11s_coco80_rk3588.labels.txt` |
| `yoloe_11s_streetwear_rk3588.rknn` | open-vocab detection | 640×640 u8 | rgb | same head with `cls[1,6,H,W]` — six prompts COCO has no class for. Pair with its own `.labels.txt` |
| `yoloe_11s_coco80_seg_rk3588.rknn` | open-vocab instance seg | 640×640 u8 | rgb | the same checkpoint with the mask branch: 13 outputs, the yolov8n-seg layout. `InstanceSegmenter` reads it unchanged |
| `yolo26n_sem_640_i8_rk3588.rknn` | semantic seg | 640×640 u8 | rgb | `[1,19,80,80]` int8 **logits** (Cityscapes 19, the same taxonomy as PP-LiteSeg) — argmax on the CPU, `segToSource` upscales |
| `yolo26n_sem_640_fp16_rk3588.rknn` | semantic seg | 640×640 **f32** | rgb | the same, float. Reproduces its ONNX to 99.6% of pixels, but its float32 input is off the zero-copy path — see below |
| `yolop_cut_640_i8_rk3588.rknn` | panoptic driving | 640×640 u8 | rgb | 5 outputs: three **anchor-based** raw heads `[1,18,H,W]` (3 priors × (5 + 1 class), NCHW, **unactivated**) + `drivable[1,2,640,640]` + `lane[1,2,640,640]` (sigmoid in-graph). ImageNet mean/std baked in |
| `arcface_r50_112_fp16_rk3588.rknn` | face recognition (identity) | 112×112 **f32** | rgb | `[1,512]` f16 embedding, L2-normalised on read-out. Input is float32 **still in 0..255** — the (x−127.5)/127.5 is in the model. The crop is a **five-point warp**, not a box — see below |
| `yolop_cut_640_fp16_rk3588.rknn` | panoptic driving | 640×640 **f32** | rgb | the same 5 outputs, float. Keeps the lane mask the int8 build blurs — but its float32 input cannot be letterboxed straight into the NPU tensor, so it is off the zero-copy path. See below |

**YOLO26 needs no decoder changes, and that is the whole point of resolving the
head from the model.** Its box branch carries 4 channels where YOLOv8/YOLO11
carry 64, because the DFL is gone; `resolveYoloHead()` reads that off the
signature and reports `plain LTRB` instead of `DFL reg_max=16`. Measured on
`bus.jpg`, all three generations return the same five objects:

| model | detections | infer | postproc |
|---|---|---|---|
| YOLOv8n | 1 bus + 4 people, top score 0.887 | 26 ms | 13 ms |
| YOLO11n | 1 bus + 4 people, top score 0.948 | 33–37 ms | 13–18 ms |
| YOLO26n | 1 bus + 4 people, top score 0.924 | 36–38 ms | **6–8 ms** |

(Per frame over 30 frames after warm-up, two runs; postprocessing is CPU work
and shares the board, hence the ranges.)

Postprocessing roughly halves because there is no DFL to reduce — 8400 cells no
longer need a 4×16 softmax-and-sum each. Inference does **not** improve: YOLO11n
and YOLO26n both run slower than YOLOv8n on this NPU despite fewer FLOPs, which
is worth knowing before treating a newer generation as an upgrade.

The rest of the family follows, with one decoder change and two more export
traps of the same kind:

* **Instance segmentation** needed nothing new. Its prototype branch is built
  from all three feature scales rather than P3 alone, which only matters to the
  export patch. On `bus.jpg` v8n-seg finds the bus and all four people including
  the one cropped by the left edge (0.318); 26n-seg finds the bus and the three
  unambiguous people at 0.84–0.91 and leaves the edge case below threshold.
* **Pose** needed a new keypoint decode, and it is the sharpest trap here.
  YOLOv8/YOLO11 predict the joint offset in HALF cells — `(2*raw + grid) *
  stride` — and **YOLO26 dropped the doubling**: `(raw + grid + 0.5) * stride`.
  Nothing in the tensor says which. Decoded with the older formula, the joints
  land at roughly twice their distance from the cell centre: measured on the two
  clear people in `bus.jpg`, 16 of 16 confident joints sit inside their own
  person's box with `kpt_decode="cell_relative_whole"` and only **5 of 16** with
  `cell_relative` — a skeleton that still draws and is still wrong.
* **Oriented boxes** changed convention: v8 maps its angle through
  `(sigmoid(v) - 0.25) * pi` inside the graph, YOLO26 regresses **radians** and
  applies nothing, so `ObbConfig` gained `angle_scale` (pi for the older
  fraction-of-a-half-turn form, 1 for radians). Decoded with the wrong one the
  boxes still land on the objects, just rotated wrongly — and because rotated
  NMS merges by IoU, even the object COUNT changes. Checked against the
  framework's own angles for `obb.jpg`: all four aircraft agree within 0.12 rad
  with the right convention (comparing modulo pi/2, since RCDL regularises every
  box to w >= h and the reference does not), and the largest is 0.35 rad out
  with v8's.
* **Classification** puts its softmax in the graph, where ResNet-18 emits
  logits. Softmax it a second time and the ARGMAX survives — top-3 remains space
  shuttle, submarine, catamaran — while the score collapses from 0.939 to 0.003
  over 1000 classes. Nothing errors, and a test that only checked the predicted
  class would pass.
* Pose also splits its keypoint branch in two: `cv4` is a feature TRUNK, and the
  joints come from `cv4_kpts` on top of it (`cv4_sigma` is a training-only
  uncertainty). Exporting `cv4` yields its 85 channels, which look exactly like
  17 joints × 5 and decode into a plausible skeleton of nonsense.

Two things about the detection export, both of which produce a plausible-looking
model when got wrong:

* An NMS-free head (`end2end = True`) carries **two** sets of branches — the
  one2many `cv2`/`cv3` used in training and the `one2one_*` pair the model
  actually infers with. Exporting the former gives boxes from a head the
  deployed model does not use. RCDL's build takes the one2one branch and still
  runs NMS, which is a no-op on a duplicate-free head and costs nothing.
* The first attempt here was converted with an **ultralytics too old to know
  YOLO26** (8.3.167). It loaded the checkpoint without complaint into a
  v8-shaped graph, and both PyTorch and the resulting `.rknn` confidently
  reported a *refrigerator* on `bus.jpg`. The conversion chain was correct
  throughout; the source model was not. Check the framework can run the weights
  before converting them.

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

**XFeat's input is not an image, and the Engine has to be told.** The reference
graph starts with a channel mean and an InstanceNorm — a per-image, data-dependent
statistic, and exactly the thing int8 quantizes badly — so the exported graph
begins one step later and takes the normalised map, roughly ±3, as float. A
quantized RKNN input is normally presented to the runtime as u8 image bytes, and
that path has **no negative range at all**: half of this map would clip to the
zero point, and the head would still return keypoints that look like keypoints.
So the Engine is opened with `float_inputs = {0}` (Python: `float_inputs=[0]`),
and `FeatureExtractor` refuses to construct on an Engine that was not:

```python
engine = rcdl.Engine("models/xfeat_640x480_i8_rk3588.rknn", float_inputs=[0])
ex = engine.feature_extractor()
```

**int8 is measured equivalent to float here**, which is a per-model finding and
not a rule (see the PP-OCRv5 recogniser above for the opposite). Optical flow is
the standard cautionary tale for feature-space models in int8; this one is fine.
Scoring uses ground truth that can be *manufactured*: rotate a photograph 12°,
scale it 0.85 and shift it, and the correct correspondence for every keypoint is
where the warp puts it.

| Build | keypoints | matches | within 3 px | median error |
|---|---|---|---|---|
| ONNX float (CPU) | 4096 + 4096 | 2192 | 77.4% | 1.35 px |
| RKNN int8, toolkit simulator | 4096 + 4096 | 2022 | 77.5% | 1.34 px |
| **RKNN int8, on the NPU** | 4096 + 4096 | 2033 | **76.6%** | **1.36 px** |

Descriptors from the int8 build reach cosine 0.9978 (min 0.9910) against the
float model's, on the 3348 of 4096 keypoints both builds detected at the same
pixel. The control that makes the match count mean something: the same extractor
on two *different* scenes finds 69 mutual matches where the warped pair finds
2033.

Cost, on `bus.jpg` at 640×480:

| Stage | Time |
|---|---|
| extract (CPU preproc + NPU + CPU decode) | 56–85 ms, of which 35 ms is `rknn_run` |
| match, 4096 × 4096 × 64-d | 213–243 ms with 8 threads (824 ms single-threaded) |

**Matching, not inference, is the budget.** It is `O(|a|·|b|·64)` — about 10⁹
multiply-adds at the reference `top_k` of 4096 — so `XfeatConfig::top_k` is the
knob that decides whether a pair costs milliseconds or a quarter second. It is
not free accuracy-wise: dropping to `top_k = 1024` cut matching to 72 ms but the
known-warp agreement fell to 62.7%, because the highest-scoring points are also
the most repetitive ones.

**Dense optical flow is not in the registry, and the reason is the runtime.**
NeuFlow v2 converts cleanly (the toolkit's simulator reproduces the float ONNX to
0.03 px on known shifts), but its correlation lookups are `GridSample`, which
**librknnrt 2.3.2 on this board does not implement at all** — not on the NPU and
not on its CPU fallback list. The toolkit lowers the op as a generic CPU node
without complaint; the runtime then reports `Unsupport CPU op: GridSample` and
**`rknn_init` segfaults**, so this is not something a caller can catch. The
supported route is a custom operator (`rknn_register_custom_ops`, declared at
conversion time), which is a piece of work RCDL has not done yet.

**Super-resolution: the input is 0..255 in both builds, and one of them is
float.** The `÷255` is folded into the `.rknn` by the toolkit's `std_values`, so
a quantized build takes plain u8 bytes and a float build takes *float32 of the
same numbers*. Handing the float build 0..1 — the obvious guess — produces a
dark, low-contrast image that still looks like an upscale: **7.3 dB against the
reference, where the correct range gives 63.2 dB**. `SuperResolver` writes
whichever type the Engine reports, so this only bites callers driving the Engine
by hand.

**Judge it by edge energy, not PSNR.** Real-ESRGAN Compact is trained
perceptually: it invents plausible texture rather than the blur that minimises
squared error, so it scores *below* a bicubic resize on PSNR against the ground
truth while looking obviously sharper. Measured over six ×4 downscale/upscale
pairs:

| | vs the original (PSNR) | mean gradient magnitude |
|---|---|---|
| bicubic ×4 | 28.35 dB | 40.5 |
| **the model** | **24.55 dB** | **55.3** |
| the original | — | 62.3 |

So PSNR is the right tool for one job only — comparing a build against the float
model it came from — and that is where the two builds differ:

| build | agreement with the float ONNX | edge energy | one 128×128 tile |
|---|---|---|---|
| **fp16** (default) | **63.2 dB** | 55.3 (as the float model) | 70–82 ms |
| int8 | 31.5 dB | 66.4 — *past the original's 62.3* | 40–53 ms |

The int8 build's extra edge energy is not detail, it is quantization noise
landing on edges. It is registered because 1.6× matters for some uses, but the
default is fp16, and this is a per-model finding: XFeat's int8 build is
indistinguishable from its float one.

Tiling cost is linear and inference dominates: a 480×270 source is 15 tiles and
1237 ms in fp16 (796 ms int8) end to end.

**Optical flow needs an operator this runtime does not have, and that is worth
reading before converting any warping network.** NeuFlow v2 — like every
correlation-based flow model — warps features by the current estimate, which is
a `GridSample`. librknnrt 2.3.2 implements that op **nowhere**: not on the NPU,
and not on its own CPU fallback list. The toolkit lowers it to a generic CPU
node with one line in the conversion log, and the runtime then prints
`Unsupport CPU op: GridSample` and **segfaults inside `rknn_init`** — before a
caller has a handle to guard, so there is nothing to catch.

The supported route is a custom operator, and it is two halves:

* at conversion, the ONNX nodes are renamed to a `cst`-prefixed type and
  declared with the toolkit's `reg_custom_op` (which also wants a numpy
  implementation, for its own shape inference and simulation);
* at run time, `rknn_register_custom_ops` supplies the real kernel — which RCDL
  does for you in `backend/custom_ops.h`, unconditionally, because registering a
  type the model does not use costs nothing.

With the declaration the same graph loads cleanly; without the registration it
loads and fails at `rknn_run` with `has no custom delegate`, which is at least
an error. Two details cost an afternoon between them and are in the code: the
runtime returns operator attributes as **quoted text** (`"bilinear"`, and an int
attribute can arrive the same way), and it labels the sampling grid NCHW while
laying it out as ONNX defines it — so the shape, not the `fmt` field, says
whether the coordinate pairs are interleaved.

**Then the accuracy, which is where the simulator is not the board.** Flow can be
scored against manufactured ground truth: move a window across a photograph and
the answer is that constant; rotate the frame and the answer is a field that
varies across it but is still known at every pixel.

| | static pair | 2–16 px shifts | 3° rotation |
|---|---|---|---|
| ONNX float (CPU) | 0.026 px | 0.03–0.10 px | 0.145 px |
| RKNN fp16, toolkit simulator | 0.031 px | 0.04–0.08 px | 0.145 px |
| **RKNN fp16, on the NPU** | **0.105 px** | **0.10–0.13 px** | **0.751 px** |

The kernel is not the difference: dumped from a real call, RCDL's C++ GridSample
reproduces a numpy reference to 4.6e-06. The simulator simply does not model the
NPU's fp16 arithmetic, and a correlation-softmax over hundreds of candidates is
where that shows. A uniform shift barely notices — every pixel is displaced
alike — which is exactly why the rotation is in the test suite.

**It is correct, not fast: about 1.4 s a frame.** Each of the nine GridSample
calls sits between two NPU subgraphs, so the whole correlation volume (37 MB at
the 1/8 scale) crosses the CPU boundary every time. That is the price of the
capability on this runtime, not a property of the network.

**Promptable segmentation is two models, and the split is the API.** EdgeSAM's
encoder runs once per frame (~350 ms at 1024×1024) and every prompt after it is
a decoder pass (~140 ms) against the same embedding. A caller that re-encodes
per prompt pays three times over for nothing.

**Both halves are float, on measurement.** The int8 encoder converts and runs and
looks fine on large objects, and is not usable:

| prompt | float mask | int8 mask | IoU vs float |
|---|---|---|---|
| box around the bus | 28.7% of the frame | 15.8% | 0.53 |
| box around a person | 2.36% | 2.50% | 0.72 |
| box around another person | 5.23% | 4.67% | 0.82 |
| **a click** | **3.50%** | **0.07%** | **0.02** |

The click is the one that settles it: the mask does not degrade, it disappears.
An embedding feeding a second network is not a classifier — nothing downstream
re-normalises what quantization moved — so the int8 build stays recipe-only
rather than shipping as a trap. The fp16 encoder reproduces the float ONNX at
0.997–0.9997 mask IoU.

Three details the code carries so a caller does not have to:

* **The embedding has to be transposed between the two models.** The encoder
  emits `[1,256,64,64]` NCHW and the decoder's embedding input is reported NHWC.
  Passing the planar buffer straight across satisfies every shape check and
  produces masks that are simply wrong.
* **The prompt convention is SAM's**: a box is two points labelled 2 and 3, a
  click is one point labelled 1 (or 0 for background) padded with a `(0,0)` point
  labelled −1. That padding is why one fixed-shape export serves both.
* **The padding value is not arbitrary.** SAM pads with zero *after* its own
  normalisation, which corresponds to a pixel value at the channel mean (~124);
  RCDL's default 114 lands within a fifth of a standard deviation of that, so the
  band reads as neutral either way. RCDL also centres the image where SAM pins it
  top-left — the geometry goes through `LetterboxInfo` in both directions, so
  what changes is where the padding sits, not what the model sees of the object.

The check that makes the masks mean something is a **second model**: prompted
with yolov8n's bus box, EdgeSAM's mask agrees with yolov8n-seg's bus mask at
**IoU 0.944**, while filling only 74% of the prompt box — so it is segmenting the
bus, not returning the rectangle it was given.

**Whole-body pose is top-down, and its two conventions are both invisible in the
tensor.** RTMW is handed one person's box and returns the COCO-WholeBody 133 (17
body, 6 feet, 68 face, 21 per hand) as a SimCC pair — two 1-D distributions per
joint rather than a heatmap.

* **The crop** is the box padded by **1.25** and then grown to the model's 3:4
  aspect on whichever axis is short — never cropped to fit, because that is
  exactly where the hands and feet are. Measured on the same person, dropping the
  padding to 1.0 moves the median joint 1.7 px and costs score (0.794 → 0.777).
* **The split ratio is 2**: SimCC bins are two per input pixel. Decoding as one
  puts the whole skeleton at twice its offset inside the crop — still a person,
  smaller and in the corner.

**Float again, and for a third distinct reason.** The int8 build agrees with the
float one on easy crops (median 1.1–1.2 px) and collapses on hard ones: a
backlit person keeps **22 of 133** joints above threshold instead of 133, and a
partly-occluded one 65 of 131, with 95th-percentile errors of 10–70 px. A 1-D
argmax over 384 bins is a *decision*, and quantization noise moves decisions —
the same shape of failure as the semantic-segmentation argmax, not the graceful
blur that quantization gives a regression head.

The check that makes 133 points mean something is the plain pose head: the first
17 of the whole-body layout are the COCO body joints in the same order, so
yolov8n-pose — bottom-up, separately trained — answers the same question.
Measured on the largest person in `bus.jpg`: **median 5.2 px over 16 shared
joints on a 541 px person** (~1% of the diagonal). The other 116 are checked
structurally: the 68 face landmarks cluster within 4 px of the nose and each hand
cluster sits at its own wrist, which is what catches a mis-sliced layout that a
body-only comparison cannot see.

### Open-vocabulary detection is a conversion-time vocabulary

YOLOE is open-vocabulary because its classification branch compares an image
embedding against a CLIP **text** embedding of each prompt. That comparison is
the part that does not belong on an NPU: it needs a 600 MB text encoder and a
tensor of words. So the text encoder runs once on the conversion host and its
output is folded into the classification convolution. What reaches the board is
an ordinary anchor-free head with one class channel per word — **`resolveYoloHead`
reports the same `DFL reg_max=16` layout as yolov8n and the decoder needs no new
code at all.**

The consequence is that **the vocabulary is a conversion-time parameter, not a
runtime one**: different words mean a different `.rknn`. All that survives into
the runtime is the class_id → prompt mapping, which is `rcdl::LabelMap` (one
prompt per line, kept beside the model as `<model>.labels.txt`).

Because a stale labels file moves no box and changes no score — it only *renames*
every result, and a build with one word removed shifts every name after it by one
— `LabelMap::requireSize()` checks the table against the class count the model
declares. Use it; the failure it catches is otherwise invisible.

Two builds ship, from the same checkpoint:

| build | vocabulary | on `bus.jpg` |
|---|---|---|
| `..._coco80` | the COCO-80 names | 1 bus + 4 people, bus box within **IoU 0.906** of yolov8n's |
| `..._streetwear` | `sneakers`, `jeans`, `hoodie`, `sunglasses`, `license plate`, `street lamp` | **4 pairs of sneakers** (0.51–0.77), each at the feet of a person yolov8n found |

The second build is the one that shows the vocabulary is load-bearing rather
than decorative: `sneakers` is not a COCO class and no class id of the first
build can express it. Not every prompt works — `wheel`, `window`, `door` and
`tree` returned nothing at all on the same frame, so a vocabulary is worth
measuring on real frames before it is shipped.

int8 costs nothing measurable here. Against the float ONNX on `bus.jpg`, both
builds reproduce **every** detection with the same word at IoU > 0.7 (6/6 and
5/5); the int8 COCO build adds one spurious 6×20 px `tie` at 0.295.

### YOLOP: the published export compiles into a model that finds nothing

The published ONNX ends the detection branch with the decode itself — grid
arithmetic assembled out of `ScatterND` writes, producing one ready-made
`[1, 25200, 6]` tensor. It compiles without a single error, and what comes back
has its **objectness and class columns never written**: only the coordinate
columns carry data, so a detector reading it finds zero objects at every
threshold down to 0.01. Nothing reports a problem.

The usable build (`_cut`) stops the graph at the three head convolutions and
does the arithmetic on the CPU (`rcdl::decodeYoloV5Anchor`). It is cheap — three
small tensors, no dependency on how many objects are in the frame — and it was
checked against the published export's own in-graph decode at conversion time:
**max absolute difference 6.1e-05 over all 25200 candidates.**

The cut also has to happen before the *reshape* inside `Detect.forward`, which
permutes each head to `[1, na, H, W, no]`. The convolution's own output,
`[1, na*(5+nc), H, W]`, is what the decoder reads.

**This is the only anchor-BASED head in the library.** Everything else decodes
anchor-free LTRB, where a cell predicts distances to the four box edges. Here a
cell carries one prediction per prior box, and the box is an offset from the cell
times a multiplier on that prior — so **the priors are part of the model, not a
tuning knob**. Decoding the same tensors with unit priors on the board gives 128
boxes of median area 12 px² where the correct set gives 18 of median 3554 px²,
and none of them lands on a vehicle. The prior set is:

```
stride  8 : (3,9)   (5,11)  (4,20)
stride 16 : (7,18)  (6,39)  (12,31)
stride 32 : (19,50) (38,81) (68,157)      # model-input PIXELS, not stride units
```

The two segmentation outputs are ordinary 2-class logit volumes: decode them
with `Segmenter` bound to outputs 3 and 4, off the same inference.

**Quantization: the lane mask is what to measure, and the usual metric cannot
see it.** Running both shipped builds on the board over the same letterboxed
canvas — int8 through the normal path, fp16 fed a float tensor by hand:

| head | int8 vs fp16 agreement | int8 vs fp16 IoU |
|---|---|---|
| drivable area | 99.73% | 0.975 |
| lane lines | **99.76%** | **0.762** |

Those two lane numbers are the same mask. Pixel accuracy says the builds agree
almost perfectly; IoU says a quarter of the structure is somewhere else. Lane
lines are 1% of the frame, so agreement is dominated by the 99% that is
correctly *not* a lane. **On a sparse structure, score IoU, not agreement.**
Detection is insensitive by comparison: both builds find 18 vehicles, and 7 of 7
of yolov8n's vehicles on the same frame are matched at IoU 0.65–0.95.

**`quantized_algorithm="mmse"` is load-bearing for this model, and only the lane
head shows it.** Against the float ONNX on `cityscapes.png` — held out of the
calibration set, which is six dashcam frames plus brightness, mirror and zoom
variants:

| int8 build | drivable IoU | lane IoU | lane pixels (float: 4386) | components (float: 88) |
|---|---|---|---|---|
| default (`normal`) | 0.905 | **0.610** | 5327 | 68 |
| `mmse` | 0.982 | **0.814** | 4245 | 50 |

The default build's lane mask is not jittering at the boundary — dilating both
masks by a pixel only moves 0.610 to 0.639 — it paints 21% more lane pixels in
fewer connected components: thicker, blobbier, fewer distinct lines. `mmse`
costs about an hour of conversion time on this graph and lands within 3% of the
float model's lane pixel count. Detection and drivable area barely move either
way, so a model scored on boxes alone would have shipped the worse build.

(The simulator put the `mmse` lane IoU at 0.814 where the board measures 0.762
against fp16. Different comparison and different arithmetic — the board is the
number that counts. Same lesson as the PP-OCRv5 recogniser below.)

int8 is what the tests and benchmarks use, for a hardware reason rather than an
accuracy one: **the fp16 build's input is float32, so it cannot be letterboxed
straight into the NPU tensor** — `engineInputView` takes u8 image bytes, which is
the whole zero-copy VPU → RGA → NPU path. The fp16 build does run on the board,
fed a float32 tensor still in 0..255, and that is how the table above was
measured. If lane geometry is what a deployment needs rather than drivable area
and vehicles, take it and pay for the CPU input pass.

### Face recognition: the crop is the model contract

An identity embedding is not computed on the detector's box. Every ArcFace-family
model is trained on a crop in a fixed canonical pose — the five landmarks mapped
onto a template so the eyes land on the same pixels for every face the network
ever sees. Hand it the bounding box instead and it still returns a 512-d unit
vector that still compares. It is just a worse one, and nothing says so.

That is now measured rather than asserted. Taking one face from `zidane.jpg`:

| what it is compared against | cosine |
|---|---|
| itself, rotated ±8°, scaled 0.85, dimmed, blurred, JPEG q40 | **0.980 – 0.999** |
| itself, **box-cropped instead of five-point aligned** | **0.493** |
| the three other (different) people in the samples | −0.10 – +0.08 |

The box crop keeps enough to look plausible and loses half the identity. That is
why `FaceRecognizer` owns the warp instead of handing the caller a matrix: the
template, the output size, the channel order and whether the model wants bytes
or floats are all model contract, and a caller-side warp is where they get lost.
`faceAlignTransform()` is still public for a caller that wants to warp with
something else — `cv2.warpAffine` with that matrix agrees with the internal
resampler to cosine 0.982–0.999.

**Which is the more interesting number here, because it is larger than the
quantization error.** On an identical crop the fp16 build reproduces the fp32
ONNX to **cosine 1.00000** on all four faces. So on this network the choice of
*resampler* moves the embedding more than fp16 does — and it moves it most on
the small, low-resolution faces (0.982 on the `bus.jpg` pair against 0.9988 on
the large `zidane.jpg` ones), which is where a recognition system is already
weakest.

**Float for a data reason, not an accuracy one.** int8 would need a calibration
set of aligned 112×112 crops, and bcdl's experience is that this exact network
calibrated on centre crops is a different model whose published accuracy does
not apply. This repo carries four faces, all different people — not a
calibration set. fp16 needs none, so it is the build that can be made and
measured here; int8 stays open until there is licensed aligned-crop data.

**What is NOT established here.** The 0.980–0.999 figures come from *nuisance
transforms of one photograph*, which is the same identity by construction. They
are not two genuinely different pictures of one person — different pose,
lighting, expression, age — where published ArcFace cosines are far lower and
where an operating threshold would actually have to be chosen. This repo has no
such pair and cannot acquire one without redistributing photographs of an
identifiable person. What the numbers above do establish is that the embedding
separates identities, is stable under nuisance, and depends on the alignment.
Stage your own pairs (gitignored, never committed) to pick a threshold.

### A second semantic model, and what a calibration domain is worth

YOLO26n-sem predicts the **same 19 Cityscapes classes** as PP-LiteSeg, which is
the point: two independently trained models over one taxonomy can each check the
other, which no single model's "that looks like a street" ever can. On
`cityscapes.png` they label **92.1% of pixels identically**, road at IoU 0.967,
and their class mixes agree to within a point (road 41% both, vegetation 30%
against 31%). It is also **2.3x faster** — 21 ms of NPU against 45.

It is exported as LOGITS at `[1,19,80,80]`. Ultralytics' ONNX export bakes the
class reduction in and returns a `[1,H,W]` map, which is a sensible default and
the wrong one here: `Segmenter` already argmaxes a logit volume, so logits mean
PP-LiteSeg and this decode through the same path with no new code — and an
argmax inside the graph cannot be scored against anything. The usual argument for
baking it (an 80x smaller device-to-host copy) also disappears once the 8x
upsample is dropped: 19x80x80 is 121k values, smaller than the baked
full-resolution map, and `segToSource` does the upscale on the label map exactly
as it does for PP-LiteSeg.

**The calibration set is dashcam frames, not the COCO subset the other YOLO
builds use, and that choice is most of the accuracy.** bcdl measured an int8
build of this network agreeing with its float model on 61% of pixel decisions
while passing every per-tensor cosine gate — a dense 19-way argmax is a decision,
and one aggregate cosine cannot see a decision boundary move. Their fix was
calibration domain rather than bit width. Calibrated here on street scenes shot
from a car (`cityscapes.png` held out of that set):

| build | pixel agreement vs float | mIoU vs float | classes found |
|---|---|---|---|
| int8, dashcam calibration | **91.8%** | 0.746 | 10 of 11 |
| fp16 | 99.6% | 0.987 | 11 of 11 |

Read the mIoU next to the agreement: 91.8% of pixels is dominated by road and
building, and 0.746 mIoU is where the cost actually sits — on the thin and rare
classes, one of which int8 stops finding altogether. Worth putting beside the
cross-model number: int8 differs from its OWN float model by about as much
(8.2% of pixels) as PP-LiteSeg differs from it (7.9%). The quantization error on
this head is the size of the difference between two different networks.

int8 is what ships and what the tests use, because its u8 input is the zero-copy
VPU → RGA → NPU path and the fp16 build's float32 input is not — the same trade
as the YOLOP builds above, and `Segmenter` refuses the float build at
construction rather than quantizing the caller's frame behind their back.

### The open-vocabulary mask branch needed no decoder either

Exporting YOLOE's mask branch alongside the detection head gives 13 outputs,
which is exactly the yolov8n-seg layout, so `InstanceSegmenter` reads it with no
changes — the same result the detection build gave, one head further on. Checked
against a separately trained closed-vocabulary model on the same frame: all 5 of
yolov8n-seg's instances matched at **IoU 0.76–0.98**, with masks covering 32–86%
of their boxes.

### PP-OCRv5 recognition: the second build that this runtime will not run

`docs/MODELS.md` already recorded that the v5 **mobile** recogniser reproduces
its Paddle reference value-for-value in the toolkit simulator and then reads 1 of
16 lines correctly on the board. The **server** variant — a different, larger
network, and the one bcdl ships — now fails the same way on the same page:

| recogniser | non-empty lines | mean score | strings identical to v4 |
|---|---|---|---|
| PP-OCRv4 (shipped) | 15 of 16 | 0.661 | — |
| PP-OCRv5 server | 15 of 16 | **0.085** | **1 of 16** |

The failure has a shape: every line starts correctly and then stops — `纯臻` where
v4 reads `纯臻营养护发素`. The sequence collapses to the CTC blank after a few
steps, which is what the earlier per-tensor comparison showed too. Two
independent v5 recognisers failing identically makes this a property of the
**18385-class softmax on this runtime's f16 path**, not of one export. Neither
ships; the registry stays on v4 rec, and the v5 **detector** — which is
measurably equivalent to v4's — remains registered.

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
| XFeat 640×480, frame → 4096 features + descriptors | 56–85 ms (35 ms NPU); matching a 4096-pair is another 213–243 ms |
| Real-ESRGAN Compact ×4, one 128×128 tile → 512×512 | 70–82 ms fp16, 40–53 ms int8; a 480×270 → 1920×1080 upscale is 15 tiles |
| NeuFlow v2 512×384, two frames → a dense field | ~1.4 s, nearly all of it the nine CPU GridSample round trips |
| EdgeSAM 1024×1024, encode a frame | 350–450 ms |
| EdgeSAM, one prompt against that embedding | 140–165 ms |
| RTMW whole-body, one person (crop + 133 keypoints) | 23–47 ms |
| ArcFace R50, one face (five-point warp + 512-d embedding) | 32 ms (31.5 ms NPU) |
| YOLO26n-sem 640, frame → full-res label map | 37 ms (21 ms NPU) — PP-LiteSeg is 86 ms (45 ms NPU) |
| YOLOE-11s seg, frame → instances + masks | 34 ms NPU, masks on top |
| YOLOE-11s 640 open-vocab, frame → detections | 62 ms (49 ms NPU) — the vocabulary size does not change it |
| YOLOP 640, one inference → 18 boxes + two full-frame masks | 160 ms preprocess+infer (126 ms NPU) + 20 ms for the anchor decode and the second mask |

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
