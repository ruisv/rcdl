# RCDL Roadmap — adapting the BCDL architecture to Rockchip RKNPU2

RCDL is the Rockchip counterpart of [BCDL](https://github.com/ruisv/bcdl) (RDK
BPU): the same "one `Engine` + per-task classes + zero-copy media pipeline"
mental model, re-based on Rockchip's hardware stack — **RKNPU2** (NPU),
**RGA** (2-D engine) and **MPP** (VPU codecs). Target boards: RK3588 / RK3588S
first (three NPU cores, RGA3, 8K VPU), then RK3576 and RK356x.

Design rule: **the NPU runs the model, RGA does every resize / colour-space /
letterbox, the VPU does every decode / encode. The CPU only runs post-processing
(NMS, DFL, CTC …) and is a guarded fallback, never the default path.**

## 1. Stack mapping (BCDL → RCDL)

| Layer | BCDL (D-Robotics hobot) | RCDL (Rockchip) | Notes |
|---|---|---|---|
| Shared buffer | `hbUCPSysMem` (phy + vir) | **dma-buf fd** from a dma-heap (`rcdl::DmaBuf`) | The one handle every unit imports: `rknn_create_mem_from_fd`, RGA `importbuffer_fd`, MPP external `MppBuffer` |
| Cache discipline | `hbUCPMemFlush` CLEAN / INVALIDATE | `DMA_BUF_IOCTL_SYNC` start/end (`DmaBuf::syncStart/End`); the RKNN runtime flushes its own I/O tensors around `rknn_run` | Cached heaps by default; uncached heaps available |
| Task/queue | `hbUCPTaskHandle_t` (one scheduler) | none — `rknn_run` sync/async, MPP async queues, RGA sync; RCDL's pipelines own the threading | Same pipeline shapes as BCDL (bounded channels, FIFO) |
| Inference | `hbDNN*` on `.hbm` | `rknn_*` on `.rknn` (`rcdl::Engine`) | Zero-copy I/O via `rknn_set_io_mem`; `rknn_dup_context` + core masks for 3-core throughput |
| Quantized I/O | dequant from `hbDNNTensorProperties` | int8 affine (zp/scale) / DFP / fp16 from `rknn_tensor_attr` | `outputAsFloat()`; decoders may read int8 directly |
| Preproc | VP / GDC (hardware letterbox) + CPU fallback | **RGA im2d**: `imresize` / `imcvtcolor` / `improcess` (crop+scale+cvt+fill in one op) + CPU/OpenCV fallback | NV12 → RGB888 letterbox straight into the NPU input dma-buf |
| JPEG | JPU via hb_vp | MPP `MPP_VIDEO_CodingMJPEG` enc/dec | Hardware JPEG on RK3588 (8K dec / 4K enc) |
| Video | VPU via media_codec | MPP `rk_mpi` H.264 / H.265 / VP9 / AV1 dec, H.264 / H.265 enc | External buffer group → frames land in our dma-bufs |
| Camera | VIN/ISP (planned in BCDL) | V4L2 (USB/MIPI via rkaiq) → dma-buf import | later |
| Post-processing | CPU/NEON ports from ccdl | **same code**, ported from BCDL (Engine-free pure functions) | det LTRB/DFL · cls · pose · seg · obb · ocr · depth … |
| Tracking | ByteTrack | same | port |
| Python | nanobind, GIL released in infer | same | `rcdl_py` + `rcdl` wrapper |
| Packaging | `libbcdl` + `bcdl` conda pkgs over `hobot-*` pkgs | `librcdl` + `rcdl` over `librknnrt` / `librga` / `rockchip-mpp` pkgs | linux-aarch64 channel; kernel drivers stay on the board image |
| Model conversion | OpenExplorer `hb_compile` on an x86 host (separate project) | **rknn-toolkit2** on an x86 host (separate `rcdl-model-zoo` project) | PTQ, `accuracy_analysis` per-layer cosine, hybrid quant |

## 2. Repository layout

```
include/rcdl/, src/
  core/      Status · DmaBuf                               (dma-heap / dma-buf UAPI)
  backend/   Engine · output readers                       (librknnrt)
  preproc/   RGA letterbox / resize / cvtColor · CPU fallback (librga im2d, OpenCV)
  media/     VideoDecoder · VideoEncoder · JpegCodec       (librockchip_mpp)
  tasks/     det · cls · pose · instance-seg · obb · semseg · depth · ocr · embed · …
  tracks/    ByteTrack · ReID
  pipeline/  DetectionPipeline (sync) · AsyncDetectionPipeline · TrackingPipeline
             · AsyncVideoDetectionPipeline (VPU → RGA → NPU → CPU/VPU)
python/      nanobind module + pure-Python wrapper
examples/    standalone C++ programs (model_info, npu_bench, dma_buf_probe, demos)
tests/       pytest: static + numpy decode (anywhere) · board end-to-end (.rknn)
scripts/     sync / board_build / bootstrap / fetch_sdk / fetch_models / leak-scan
docs/        ROADMAP (this) · API.md / CPP_API.md / MODELS.md / RGA.md as they land
```

## 3. Milestones

Each milestone ends with a board-verified result and a pinned test.

- **M0 — skeleton + Engine** ✅
  `DmaBuf`, `Engine` (zero-copy I/O, dequant, core masks, dup contexts),
  `model_info` / `npu_bench` / `dma_buf_probe`, Python `Engine`, workstation →
  board workflow, conda env, leak-scan hook.
  *Verified on RK3588 (librknnrt 2.3.2 / driver 0.9.8): ResNet18 int8 4.0 ms on
  one core; 3 pinned contexts 707 fps aggregate.*

- **M1 — detection end-to-end (NPU + RGA)** ✅
  `preproc/rga`: `RgaLetterbox` (NV12/BGR → RGB888 letterbox into the NPU input
  dma-buf in one `improcess`), `RgaResize`, `RgaCvtColor`, CPU/OpenCV fallback
  behind `RCDL_HAVE_RGA`. `tasks/detection`: LTRB multi-scale + DFL decode +
  per-class NMS ported from BCDL; `DetectionPipeline` (sync, buffer reuse).
  *Verified: YOLOv8n and YOLO11n independently find 1 bus + 4 people on
  `bus.jpg`. RGA and the CPU reference agree to ±1 LSB on a linear ramp at every
  downscale factor, and to ≤5 LSB on band-limited content — but NOT on content
  that aliases, because RGA pre-filters when it shrinks and bilinear does not.
  End to end that means the two backends find the same objects and agree to
  ~1.5 px on confident boxes, while marginal detections move materially. All of
  that is measured in `docs/RGA.md` rather than assumed.*

- **M2 — media: MPP codecs, zero-copy** ✅
  `media/VideoDecoder` (H.264/H.265 → NV12 dma-bufs, external buffer group,
  reorder-correct), `VideoEncoder` (NV12/RGB dma-buf → H.264/H.265), `JpegCodec`
  (MJPEG enc/dec). *Verified: 1080p H.264 decode 324 fps, 4K H.265 244 fps,
  1080p H.264 encode 197 fps reading the decoder's dma-bufs in place — every
  frame carries an fd, so decode → RGA → NPU → encode never copies. Round-trip
  luma PSNR 47.2 dB. Hardware JPEG matches libjpeg to 0.02 mean luma.*

- **M3 — pipelines + multi-core** ✅ (async detection, `EnginePool`, ByteTrack)
  `AsyncDetectionPipeline` (preproc ‖ infer ‖ decode, bounded channels, FIFO),
  `EnginePool` (N dup'd contexts on cores 0/1/2, round-robin), `TrackingPipeline`
  (ByteTrack), `AsyncVideoDetectionPipeline` (VPU → RGA → NPU, all overlapped).
  *Verified: the async pipeline returns results field-identical to the
  synchronous one, in submission order, at 481 fps against 99 fps — a 4.86x
  speed-up on three pinned contexts. `TrackingPipeline` holds stable ids across
  a panning sequence — 4 objects, 4 distinct ids over 14 frames, three of them
  alive for all 14 — in BOTH its geometry-only and its ReID mode, and takes a
  decoded `VideoFrame` without copying it. `AsyncVideoDetectionPipeline` runs
  1080p H.264 → YOLOv8n at 72–97 fps against 23–28 fps for the same stream
  through the synchronous pipeline — a 3.1x to 3.7x speed-up over six runs —
  returning detections identical to it frame by frame.* `video_det_demo` remains the synchronous baseline and the only place
  the overlay → VPU encode tail is demonstrated.

- **M4 — task breadth** ✅
  Port the BCDL task heads that map cleanly: classification, pose, instance
  seg, OBB, semantic seg, depth, OCR (DBNet + CTC), embedding/ReID, face. Each:
  pure-function decoder + numpy test + board test + one `.rknn` in the model
  registry. *Verified against real models, and every one of these is now an
  assertion in the suite rather than a note: instance seg agrees with detection
  on `bus.jpg` (1 bus + 4 people) with plausible mask coverage; PP-LiteSeg
  resolves a Cityscapes street into 11 classes; every pose keypoint the model is
  confident about lands inside its own person's box; OBB finds 4 planes, 26 large
  vehicles and 3 ships, with the parked vehicles sharing an orientation
  (concentration 0.93); RetinaFace returns both faces with landmarks ordered
  eyes → nose → mouth; PP-OCRv4 reads all 15 text lines exactly.*
  *The native NC1HWC2 output path was benchmarked and **rejected**: binding
  outputs with `RKNN_QUERY_NATIVE_OUTPUT_ATTR` instead of the standard attrs is
  within noise on YOLOv8n and ResNet-18 (−1%) and **23.6% slower** on
  YOLOv8n-seg, while producing larger buffers (NC1HWC2 padding) and requiring a
  de-swizzle in every decoder. The runtime's own conversion is not the
  bottleneck, so RCDL keeps the standard NCHW binding.*

  Dynamic-shape inputs (`rknn_set_input_shapes`) are **not implemented, and not
  for lack of trying**: whether a model accepts them is fixed at conversion
  time, and querying `RKNN_QUERY_INPUT_DYNAMIC_RANGE` on every model in the
  registry returns `-6` — none was built with `dynamic_input`.
  Adding the API without a model that exercises it would ship untested code, so
  it waits for the model-zoo side to produce one.

  Worth knowing: the deployed **pose and OBB exports fuse box and class into one
  tensor per scale and leave the class sigmoid on the CPU**, unlike the detection
  export. The decoders read the model's own signature and handle both shapes —
  see `docs/MODELS.md`.

  Two heads shipped a decoder, a numpy oracle and **no model to run** for longer
  than they should have — monocular depth and embedding/ReID. Both now have one
  (Depth-Anything-V2-Small and OSNet x0.25), each checked against its untouched
  fp32 ONNX on the identical input rather than merely against itself, so all ten
  heads in `src/tasks/` have a model *and* an on-board assertion. The lesson is
  recorded because it is the reusable part: **audit `src/tasks/` against the
  registry head by head — a hand-kept model list will not show you its own
  holes.**

- **M5 — Python surface + docs** ✅
  nanobind coverage of the engine, preprocessing, codecs, tracking (including
  `TrackingPipeline` itself, via `Engine.tracker(reid=...)`) and the task heads
  (GIL released around anything that touches hardware);
  [`API.md`](API.md), [`CPP_API.md`](CPP_API.md), [`MODELS.md`](MODELS.md),
  [`RGA.md`](RGA.md).

- **M6 — packaging + release**
  conda recipes (`librknnrt`, `librga`, `rockchip-mpp`, `librcdl` + `rcdl`) for
  linux-aarch64 across Python 3.9–3.14, board validation of the packages, pip
  wheel, tagged 0.1.0. Benchmarks page with the annotated-figure gallery.
  *The pip wheel is done and validated on the board: the installed package —
  not the build tree — reports RGA and MPP available and runs detection end to
  end. The conda recipes live in a separate feedstock and are still to do.*

### Model and task breadth — where RCDL stands against BCDL

M0–M6 ported the *architecture*. They did not port the *catalogue*, and the gap
is worth stating plainly rather than leaving a reader to discover it: BCDL grew
its task surface over a dozen further milestones, and RCDL is at the end of the
equivalent of its M5.

| | BCDL | RCDL |
|---|---|---|
| Task heads in `src/tasks/` | 22 | **17** |
| Models in the registry | ~38 | **39** |
| Pipeline classes | 5 | 3 |

The core CV set — detection, classification, pose, instance and semantic
segmentation, oriented boxes, OCR, face detection, monocular depth,
embedding/ReID — was there from M4, each with a model and a board test. M7–M10
added the textline angle classifier, sparse features, super-resolution, optical
flow, promptable segmentation, whole-body pose, open-vocabulary detection,
panoptic driving and face recognition. What is still missing splits into two kinds, and they are not
equally hard.

**Same head, older or thinner model.** Cheap: the decoder already exists.
* OCR is at parity for detection and recognition — v4, v5 and v6 detectors, v4,
  v5 and v6 recognisers — with the two newer recognisers needing their softmax
  taken out of the graph to run at all, and the v6 detector needing the ONNX
  upstream publishes rather than its Paddle export; see `docs/MODELS.md`. The
  direction classifier is one generation behind (`ch_ppocr_mobile_v2.0_cls`,
  which reads 16 of 16 lines here; upstream ships none for v6).
* Semantic segmentation has two models where BCDL has three.
* Detection, classification, pose, instance seg and OBB sit a YOLO generation
  behind.

**Heads RCDL does not have.** Each is a decoder plus a model plus tests:
anomaly detection, RGB-D depth refinement, end-to-end driving, lidar 3-D
detection, monocular 3-D — plus stereo disparity, which in BCDL is a pipeline
rather than a head. Most of what remains is blocked on *data* rather than on
code: MVTec parts, a stereo pair, calibrated driving frames. None of it is
carried here.

Milestones follow that split, cheapest and most-broken first. Each still ends
with a board-verified result and a pinned test.

- **M7 — OCR stack refresh** ✅
  `TextAngleClassifier` came first because its absence was not a missing feature
  but a wrong answer, and the measurement says so: of the 16 lines on the sample
  page, every one rotated 180° comes back from the recogniser as the empty
  string or a single stray bracket — the content gone, with nothing in the
  result to say so. With the classifier in front, all of them read exactly as
  they do upright. *Verified: 16/16 upright lines labelled upright and 16/16
  rotated lines labelled rotated, at confidence 1.000 for every line but the
  vertical strip, which both orientations score ~0.6 and the 0.9 flip gate
  therefore leaves alone. The int8 build agrees with the Paddle fp32 original on
  100% of labels on a held-out split of the calibration lines.*
  Its preprocessing is **not** the letterbox the rest of the library uses — PP-OCR
  fits the crop to the model's height and pads to the right, and a centred
  letterbox costs 16/16 → 9/16 on the same lines (see
  [`MODELS.md`](MODELS.md)). The **recogniser** is fitted the same way, which is
  the half that had been missed: for a line wider than the input's aspect ratio
  the fit and a plain stretch are the same operation, so a page of long lines
  cannot tell them apart. Cut those lines to their leading 30% and PP-OCRv4 reads
  15 of 15 correctly under the reference fit against 10 of 15 under a stretch, so
  `fit="pad"` is now the default.

  PP-OCRv5 and v6 were then converted and measured. Both **detectors** ship as
  non-regressions — same regions, same strings as v4 on the sample page, pinned
  by parity tests — with one thing learned from v6: its DB thresholds ship *with
  the weights* and are not the library's defaults, and decoding it with the
  wrong ones changes a character. The v6 detector also settled a blocker: it
  comes from the ONNX PaddleOCR **publishes** for v6, so the `paddle2onnx`
  SIGABRT that stopped the first attempt was avoidable rather than fatal.

  Both newer **recognisers** at first appeared undeployable — the v5 build reads
  1 line in 16 on the NPU while the toolkit simulator reproduces its Paddle
  original exactly. The correction is worth more than the model: the failing
  piece is the **18385-way softmax op** in this runtime's float16 path, not the
  network, and CTC decoding only needs the argmax. Re-export to logits and both
  v5 and v6 run — 15 of 16 lines at mean confidence 0.928 and 0.930 against v4's
  0.661 — with the softmax applied on the CPU. PP-OCRv4 stays the default on
  size and latency, not accuracy. The numbers are in [`MODELS.md`](MODELS.md).

- **M8 — face recognition** ✅ (threshold still needs data)
  The geometry half is done and is the half that is easy to get subtly wrong:
  `similarityTransform` / `faceAlignTransform` map a detection's five landmarks
  onto the ArcFace template, in the closed form that **cannot produce a
  reflection** — a mirrored "alignment" fits five points just as well as a
  rotation and would hand the identity model a face that is not the one in the
  picture. *Verified on all four faces in this project's images: the transform's
  own residual is 1.3–7.3 px mean (five points, four degrees of freedom — the
  worst is the turned head), RetinaFace re-finds every aligned crop at 0.99–1.00,
  and the re-detected eyes and mouths land within a few pixels of the template's
  rows.*

  **The identity half is now there too**, and the route past the data problem was
  to stop needing calibration data: an fp16 ArcFace R50 needs none, where an int8
  one would need a set of ALIGNED crops (a same-shaped model calibrated on centre
  crops is a different model whose published figures do not transfer). It
  reproduces the fp32 ONNX to **cosine 1.00000** on an identical crop, at 32 ms
  per face.

  `FaceRecognizer` owns the warp rather than handing the caller a matrix, and
  the reason is the measurement: the same face **box-cropped instead of
  five-point aligned scores 0.493 against its aligned self**, while rotation,
  scale, brightness, blur and JPEG q40 all stay between 0.980 and 0.999, and the
  four different people in the samples sit between −0.10 and +0.08. A box crop
  returns a well-formed unit vector with half the identity gone and nothing in
  the output says so. A second number makes the same point about care: the
  internal resampler and `cv2.warpAffine` differ by 0.982–0.999 — **more than
  fp16 costs** — and differ most on the small faces.

  What is still missing is an **operating threshold**, and that genuinely needs
  data: nuisance transforms of one photograph are the same identity by
  construction, not two different pictures of a person. This repository cannot
  acquire such a pair without redistributing photographs of an identifiable
  person, so the pattern is to stage your own (gitignored, never committed) and
  pick the threshold there.

- **M9 — YOLO generation refresh** (detection ✅, the other four heads to do)
  The premise held exactly: **YOLO26n detection required no code at all.** Its
  head has no DFL, so the box branch is 4 channels instead of 64, and
  `resolveYoloHead()` reads that off the model's own signature and switches to
  the plain-LTRB path by itself. *Verified: all three generations return the
  same 1 bus + 4 people on `bus.jpg`; postprocessing drops from 13 ms (v8n) and
  13–18 ms (11n) to 6–8 ms because there is no DFL to reduce — while inference
  does not improve, 36–38 ms against 26 ms for v8n on this NPU.* The export has
  two traps, both of which yield a plausible model: an NMS-free head carries a
  second set of branches and only the `one2one_*` pair is the deployed one, and
  an ultralytics too old to know the architecture loads the checkpoint anyway
  and predicts confident nonsense. Both are written up in
  [`MODELS.md`](MODELS.md).

  Instance segmentation and pose followed. Segmentation needed no code either.
  **Pose needed one decoder addition**, and it is the sharpest trap of the three:
  YOLOv8/YOLO11 predict the keypoint offset in half cells and YOLO26 dropped the
  doubling, with nothing in the tensor to say which — so `KeypointDecode` gained
  `kCellRelativeWhole`. *Verified by running both formulas over the same model:
  16 of 16 confident joints land inside their own person's box with the YOLO26
  formula, 5 of 16 with the older one — a skeleton that still draws and is still
  wrong.*

  Oriented boxes and classification complete the family, each with a convention
  that changed silently: YOLO26 regresses the OBB angle in **radians** where v8
  emitted a fraction of a half-turn through a sigmoid (so `ObbConfig` gained
  `angle_scale`), and its classifier has the **softmax in the graph** where
  ResNet-18 emits logits. *Verified against the framework's own outputs: all
  four aircraft in `obb.jpg` within 0.12 rad with the right convention and the
  largest 0.35 rad out with the wrong one; class 812 at 0.939 respecting the
  in-graph softmax and 0.003 — same argmax, meaningless confidence — with a
  second one applied.* All five heads of the generation are now in the registry.

- **M10 and beyond — new task heads** (sparse features ✅, super-resolution ✅,
  optical flow ✅, promptable segmentation ✅, whole-body pose ✅,
  open-vocabulary detection ✅, panoptic driving ✅)
  Ported in BCDL's order where the maths carries over, each as decoder + numpy
  oracle + model + board test. What is left — anomaly detection, stereo
  disparity, and the sensor-fusion heads (lidar 3-D, mono3d, end-to-end
  driving) — is blocked on data rather than code: MVTec parts, a rectified
  stereo pair, calibrated driving frames, none of which this project carries.

  **Sparse local features (XFeat) landed first**, and it is the first head here
  that answers a question about *two* frames rather than one: repeatable points
  plus 64-d descriptors, mutual-nearest-neighbour matching, and therefore
  homographies, stitching and a SLAM front end. It is also the only head whose
  ground truth can be manufactured — rotate a photograph by a known amount and
  every correspondence has an exact right answer — so the board test asserts
  *agreement with the geometry* rather than "some matches were found". *Verified:
  76.6% of 2033 matches within 3 px of where a 12°/0.85 warp puts them, median
  1.36 px, against 77.4% and 1.35 px for the float ONNX on CPU; the same
  extractor on two different scenes finds 69 matches where the warped pair finds
  2033.*

  It also cost one **Engine** change, and the reason generalises. XFeat's input
  is not image bytes but an InstanceNorm output — kept on the CPU precisely
  because a per-image statistic quantizes badly — and a quantized RKNN input is
  presented to the runtime as u8, a path with no negative range at all. Half of
  a mean-zero map would clip to the zero point and the head would still return
  keypoints that look like keypoints. `EngineOptions::float_inputs` names such
  inputs, and `FeatureExtractor` refuses to construct without it rather than
  running.

  **Super-resolution followed**, and it is the first head whose output is an
  image rather than a description: a fixed 128×128 tile upscaled ×4, with an
  arbitrary frame cut into overlapping tiles and cross-faded back together.
  *Verified: butt-jointed tiles leave a 2.1× jump at the seam columns and the
  cross-fade brings it to 1.05×, indistinguishable from the picture's own
  variation.* Its lesson is about **how to judge a model at all**. This family is
  trained perceptually, so it scores BELOW a bicubic resize on PSNR against the
  ground truth (24.6 dB vs 28.4 dB) while looking obviously sharper — it invents
  plausible texture rather than the blur that minimises squared error. The board
  test therefore asserts recovered high-frequency energy, with bicubic as the
  floor and the original as the target, and PSNR is used only where it is valid:
  comparing one build against the float model it came from. That is also how the
  registry's default was chosen — fp16 reproduces the float model at 63.2 dB
  where int8 manages 31.5 and over-sharpens past the original, for 1.6× the
  speed. XFeat's int8 build, measured the same way, is indistinguishable from
  its float one; neither result is a rule.

  **Open-vocabulary detection (YOLOE) added no decoder at all**, and that is the
  finding rather than a shortcut. YOLOE is open-vocabulary because its
  classification branch compares an image embedding against a CLIP *text*
  embedding of each prompt — a 600 MB text encoder and a tensor of words, none of
  which belongs on an NPU. Run the text encoder once on the conversion host,
  fold its output into the classification convolution, and what reaches the board
  is an ordinary anchor-free head with one class channel per word: `resolveYoloHead`
  reports the same layout as yolov8n and `DetectionPipeline` reads it unchanged.
  The vocabulary becomes a **conversion-time** parameter, and the only runtime
  state is the class_id → prompt table (`rcdl::LabelMap`).
  *Verified two ways, because "it found five things" proves nothing here: the
  COCO-80 build agrees with yolov8n about where the bus is (IoU 0.906), and a
  six-prompt build finds four pairs of `sneakers` — a word COCO has no class for
  — each at the feet of a person yolov8n found.* Not every prompt works;
  `wheel`, `window` and `tree` returned nothing on the same frame, so a
  vocabulary is worth measuring before it ships. `LabelMap::requireSize()` is a
  hard check because a labels file from another build moves no box and changes no
  score — it only renames every result.

  **Panoptic driving (YOLOP)** is the opposite: one inference, three decoders,
  and the first ANCHOR-BASED head in the library. Everything else here decodes
  anchor-free LTRB, where a cell predicts distances to the box edges; this one
  predicts, per prior box, an offset from the cell and a multiplier on that
  prior's size — so the priors are part of the model, not a tuning knob.
  Decoding the same tensors with unit priors gives 128 boxes of median area
  12 px² where the correct set gives 18 of ~3000 px², and none on a vehicle.
  *Verified cross-model: all 7 of yolov8n's vehicles on a street frame matched,
  best IoUs 0.65–0.95, and the drivable-area mask covers 22% of the frame
  entirely below the horizon.*

  Its model name carries a warning. The published export bakes the anchor decode
  into the graph out of `ScatterND` writes; that compiles without a single error
  into a model whose objectness and class columns are **never written**, so a
  detector reading it finds zero objects at any threshold. The usable build cuts
  the graph at the three head convolutions — checked against the reference
  decode to 6.1e-05 over all 25200 candidates — and does the arithmetic on the
  CPU. Its quantization result adds a fifth distinct story, and the ordinary
  metric cannot see it: on the board the int8 and fp16 builds agree on **99.76%
  of lane pixels at an IoU of 0.762**, because lane lines are 1% of the frame
  and agreement is dominated by the 99% that is correctly not a lane. On a
  sparse structure, score IoU, not agreement. And only that number moves —
  switching the quantizer to `mmse` lifts the lane IoU against the float ONNX
  from 0.610 to 0.814 while detection and drivable area barely shift, so a build
  scored on boxes alone would have shipped the worse one.

  **Whole-body pose (RTMW)** is the same task as `tasks/pose.h` with the cost
  model inverted: top-down, one inference per person (~25 ms), 133 keypoints
  instead of 17 — the 68 face landmarks and 21 points per hand that sign
  language, gesture and "what are the fingers doing" need and a body skeleton
  cannot answer. *Verified against the plain pose head, which is the useful
  control here because the first 17 of the whole-body layout ARE the COCO body
  joints in order: median 5.2 px over 16 shared joints on a 541 px person, with
  the other 116 checked structurally — the face cluster within 4 px of the nose,
  each hand cluster at its own wrist.* Its int8 build is the third distinct
  quantization story in this milestone: fine on easy crops, and on a hard one it
  keeps 22 of 133 joints instead of 133. A 1-D argmax over 384 bins is a
  decision, and quantization noise moves decisions.

  **Promptable segmentation (EdgeSAM)** is the first head with no classes at
  all: it takes a click or a box and returns the mask of whatever is there, which
  is what an annotation tool, a "cut this out", or any detector wanting
  silhouettes instead of rectangles actually needs. It is also the first head
  that is two models, and the split is deliberately visible in the API — the
  encoder costs ~350 ms per frame and each prompt after it ~140 ms, so a caller
  that cannot see the difference will write the slow loop. *Verified against a
  second model rather than against its own prompt: given yolov8n's bus box,
  EdgeSAM's mask agrees with yolov8n-seg's bus mask at IoU 0.944 while filling
  only 74% of the box it was given — segmenting the bus, not returning the
  rectangle.*

  Its quantization result is the mirror of XFeat's. The int8 encoder converts,
  runs, and keeps the shape of large objects (IoU 0.53–0.82), but a CLICK that
  returns 3.5% of the frame in float returns 0.07% in int8 — the mask does not
  degrade, it disappears. An embedding feeding a second network has nothing
  downstream to re-normalise what quantization moved, so both halves ship float
  and the head refuses an int8 encoder rather than running one.

  **Dense optical flow came last and needed infrastructure, not a decoder.**
  NeuFlow v2 converts cleanly and the toolkit's simulator reproduces the float
  ONNX — but its correlation lookups are `GridSample`, which librknnrt 2.3.2
  implements on the NPU *and* on its CPU fallback path exactly nowhere. The
  toolkit lowers it to a generic CPU node without complaint and the runtime then
  **segfaults inside `rknn_init`**, which no caller can guard against. So RCDL
  gained a custom-operator layer: the model is converted with that node declared
  (`cstGridSample`) and `backend/custom_ops.h` registers a CPU kernel on every
  Engine. *Verified against manufactured ground truth — 0.10–0.13 px endpoint
  error on known shifts, and the kernel itself reproduces a numpy reference to
  4.6e-06 on a dumped real call.*

  The measurement worth carrying forward is the **3° rotation**: 0.145 px for the
  float model, 0.145 px in the toolkit's simulator, **0.751 px on the NPU**. The
  simulator does not model fp16 arithmetic, and a uniform shift cannot see the
  difference because every pixel is displaced alike. This is the same lesson the
  PP-OCRv5 recogniser taught in a louder voice — *matching the simulator is not
  matching the board* — and it is why the suite scores a rotation and not only a
  shift. The head is correct, not fast: each of the nine boundary crossings
  moves the whole correlation volume, so a frame is about 1.4 s.

- **Pipeline debt** ✅ — `AsyncVideoDetectionPipeline` is a class, built on the
  `acquireSlot` / `letterboxIntoSlot` / `commitSlot` split that exists precisely
  so a video stage need not hold a recycled frame while waiting for capacity,
  and bound in Python as `Engine.video_detector()`. Writing it turned up a
  silent hardware bug it is worth reading [`RGA.md`](RGA.md) §3.1 for: the
  letterbox's **CPU-painted border was not reproducible**, so the same clip gave
  different boxes on different runs, in the synchronous pipeline too. The border
  is now blitted by RGA. What found it was a test that compares two runs of one
  clip frame by frame — nothing errored, and every box looked reasonable.

- **Later** — camera (V4L2/rkaiq) source, RK3576 / RK356x validation, LLM/VLM
  via `rknn-llm` (separate project, as BLLM is to BCDL), ROS 2 nodes.

## 4. Hardware notes that shape the code (RK3588)

- **NPU**: 3 cores @ 1 GHz (6 TOPS int8). A small model does not scale across
  cores with a combined mask — run one context per core instead (`Engine::dup`
  + `NpuCore::Core0/1/2`). Combined masks help large models' latency.
- **Runtime ↔ driver**: librknnrt 2.3.x needs RKNPU driver ≥ 0.9.x; a model
  converted with toolkit X.Y runs on runtime ≥ X.Y. `model_info` prints both.
- **Zero-copy**: allocate I/O with `rknn_create_mem` (cached) or import a
  dma-buf with `rknn_create_mem_from_fd`; bind once with `rknn_set_io_mem`.
  Inputs are UINT8 NHWC for quantized models (mean/std folded by the toolkit).
  `w_stride` may exceed width — honor it when writing the input directly.
- **dma-heap**: `/dev/dma_heap/system*` works unprivileged when the user is in
  the `video` group (or via a udev rule); the `cma` heap can be empty on an
  image — RK3588's NPU/RGA3/VPU all sit behind IOMMUs, so the system heap is
  the default. RK356x (RGA2 without IOMMU) needs CMA.
- **RGA**: RGA3 (RK3588) handles NV12 ↔ RGB888, scaling 1/16–16×, min 68×2 for
  scaled ops, 16-byte stride alignment for YUV; `imcheck` before `improcess`.
  `improcess(src, dst, pat, srect, drect, prect, usage)` does crop + scale +
  cvtcolor + fill in one pass — that is the letterbox.
- **MPP**: decoder with an external buffer group so output frames live in our
  dma-bufs; `MPP_DEC_SET_OUTPUT_FORMAT` for NV12; encoder reads an `MppBuffer`
  wrapped around a dma-buf. JPEG goes through the same `rk_mpi` API
  (`MPP_VIDEO_CodingMJPEG`).

## 5. Model conversion (out of scope here)

ONNX → `.rknn` (rknn-toolkit2 PTQ, `accuracy_analysis`, hybrid quantization)
runs on an x86 host and is owned by the separate model-zoo project; this repo
consumes finished `.rknn` files (`scripts/fetch_models.sh`). Rules of thumb
baked into the decoders: keep DFL / sigmoid out of the graph when they hurt
int8 accuracy (the toolkit's YOLO recipes do), input `rgb`/`bgr` order is a
model property recorded in the registry, and every model's build carries the
toolkit version it was made with.
