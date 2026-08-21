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
  1080p H.264 → YOLOv8n at 96.7 fps against 27.7 fps for the same stream through
  the synchronous pipeline (3.49x), returning detections identical to it frame by
  frame.* `video_det_demo` remains the synchronous baseline and the only place
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
| Task heads in `src/tasks/` | 22 | **10** |
| Models in the registry | ~38 | **13** |
| Pipeline classes | 5 | 3 |

The ten RCDL has are the core CV set — detection, classification, pose, instance
and semantic segmentation, oriented boxes, OCR, face detection, monocular depth,
embedding/ReID — each with a model and a board test. What is missing splits into
two kinds, and they are not equally hard.

**Same head, older or thinner model.** Cheap: the decoder already exists.
* OCR is PP-OCRv4 det + rec against BCDL's v5/v6 stacks, and has **no textline
  angle classifier at all** — so rotated text decodes wrong today, silently.
* Face is detection only. BCDL pairs SCRFD with ArcFace for identity.
* Embedding is person ReID only; there is no image-text tower.
* Semantic segmentation has one model where BCDL has three.
* Detection, classification, pose, instance seg and OBB sit a YOLO generation
  behind.

**Heads RCDL does not have.** Each is a decoder plus a model plus tests:
anomaly detection, RGB-D depth refinement, end-to-end driving, sparse local
features, lidar 3-D detection, monocular 3-D, open-vocabulary detection, optical
flow, panoptic driving, promptable segmentation, super-resolution, whole-body
pose — plus stereo disparity, which in BCDL is a pipeline rather than a head.

Milestones follow that split, cheapest and most-broken first. Each still ends
with a board-verified result and a pinned test.

- **M7 — OCR stack refresh** (angle classifier ✅, v5/v6 det+rec to do)
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
  [`MODELS.md`](MODELS.md)).

  PP-OCRv5 was then converted and measured. The **detector** ships as a
  non-regression — same regions, same text, same confidence as v4 on the sample
  page, pinned by a parity test — while the **recogniser cannot be deployed on
  this board at all**: correct in the toolkit's simulator (0.9997 mean winning
  probability), 1 line in 16 correct on the NPU in float16, worse in int8, and
  int16 fails to submit its first Conv on this driver. PP-OCRv4 therefore stays
  the deployed recogniser, and the search for a newer one that survives float16
  moves to the model-zoo side. The numbers are in [`MODELS.md`](MODELS.md).

- **M8 — face recognition**
  An ArcFace-style identity embedding on top of the existing detector: the
  5-point similarity transform that produces an aligned 112×112 crop, then the
  same embed-and-compare path ReID already uses. Note that the aligned-crop
  calibration is what the accuracy figures depend on — a same-shaped model
  calibrated on centre crops is a different model.

- **M9 — YOLO generation refresh**
  A current YOLO family across detection, classification, pose, instance seg and
  OBB, replacing the v8/11 builds. Contingent on the toolkit converting those
  exports cleanly; the decoders already resolve head layout from the model, so
  the work is conversion and re-measurement rather than new code.

- **M10 and beyond — new task heads**
  Ported in BCDL's order where the maths carries over, each as decoder + numpy
  oracle + model + board test: optical flow, super-resolution, open-vocabulary
  detection, promptable segmentation, whole-body pose, anomaly detection, stereo
  disparity. The sensor-fusion and driving heads (lidar 3-D, mono3d, panoptic
  and end-to-end driving) are a separate question — they need calibration data
  and sample frames this project does not currently carry.

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
