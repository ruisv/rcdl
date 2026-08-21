# Changelog

All notable changes to RCDL are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project skeleton: CMake build (`librcdl.so`, examples, nanobind module,
  `find_package(rcdl)` export), conda env spec, workstation → board workflow
  scripts, leak-scan pre-commit hook.
- `rcdl::Engine` — loads an `.rknn`, binds zero-copy I/O tensors
  (`rknn_create_mem` + `rknn_set_io_mem`), runs inference, dequantizes outputs
  (int8 affine / DFP / fp16 / fp32), pins contexts to NPU cores, duplicates
  contexts (`rknn_dup_context`) for multi-core throughput.
- `rcdl::DmaBuf` — RAII dma-heap buffer with explicit CPU cache sync; the
  shared buffer type for NPU / RGA / VPU zero-copy.
- Examples: `model_info`, `npu_bench`, `dma_buf_probe`.
- Python: `rcdl.Engine` (numpy in/out), `rcdl.DmaBuf`, `rcdl.dequantize`.
- `rcdl::Image` / `rcdl::ImageView` — the dma-buf-backed image descriptor every
  hardware unit consumes by fd, plus `engineInputView()` so a preproc op can
  write straight into the NPU's bound input tensor.
- `preproc/rga` — RGA (im2d) hardware preprocessing: `rgaLetterbox` (crop +
  scale + colour convert + border fill in one `improcess`), `rgaResize`,
  `rgaCvtColor`, `rgaCropResize`, `rgaCopy`, `rgaFill`, `rgaDrawRect`, with
  `rgaCanHandle()` exposing the RGA3 constraints (scale range, minimum size,
  stride alignment) without throwing.
- `preproc/letterbox_cpu` — the guarded CPU fallback (bilinear with the OpenCV
  pixel-center convention, BT.601 conversions, OpenMP row loop), and
  `preproc/letterbox` — the backend-agnostic `letterbox` / `resize` /
  `cvtColor` that use RGA when the hardware accepts the request and the CPU
  otherwise.
- `tasks/detection` — per-class NMS, the fused single-tensor YOLO decoder, and
  the anchor-free LTRB multi-scale decoder with CPU-side DFL reduction. Handles
  both deployed export conventions (NCHW vs NHWC branch tensors, sigmoid inside
  or outside the graph) and resolves the head layout — grids, class count,
  `reg_max`, strides, channel order — from the model's own output signature
  (`resolveYoloHead`), so a mismatched model fails loudly instead of decoding
  garbage.
- `pipeline/DetectionPipeline` — synchronous streaming detection with zero
  per-frame allocation: RGA letterboxes the source frame directly into the NPU
  input dma-buf, then infer, then decode. Per-stage profiling.
- Example `det_demo` — detect on an image, draw boxes, report the per-stage
  latency breakdown.
- Python: `rcdl.Engine.detector()`, `rcdl.DetectionPipeline`, `rcdl.detect`,
  `rcdl.letterbox`, `rcdl.cvt_color`, `rcdl.decode`, `rcdl.decode_yolo_ltrb`,
  `rcdl.nms`, `rcdl.compute_letterbox`, `rcdl.rga_available`, COCO class names.
- Tests: numpy decode/NMS oracles (`test_detection.py`), preprocessing
  reference + RGA/CPU agreement (`test_letterbox.py`), and the on-board
  end-to-end detection check (`test_detection_board_py.py`).
- `media/VideoDecoder` — hardware H.264 / H.265 / VP9 / AV1 decode on the VPU
  through MPP, decoupled feed/drain with reorder support. Frames are decoded
  into an **external buffer group of RCDL-allocated dma-bufs**, so
  `VideoFrame::view()` carries an fd that RGA and the NPU import directly:
  decode → letterbox → infer with no copy anywhere. Handles the info-change
  renegotiation (the stream's real size and stride are only known after the
  first frames) and falls back to MPP's internal pool if an external group
  cannot be committed. *Measured on RK3588: 1080p H.264 at 324 fps and 4K
  H.265 at 244 fps, every frame carrying a dma-buf fd.*
- `media/VideoEncoder` — H.264 / H.265 / MJPEG encode, CBR/VBR/FixQP rate
  control, SPS/PPS extradata, and zero-copy input: an `ImageView` with a
  dma-buf fd is wrapped and read in place. *Measured: 1080p H.264 encode at
  197 fps straight from decoded frames, decode→encode→decode luma PSNR 47.2 dB.*
- `media/VideoFrame` — RAII handle over a decoded frame that returns its buffer
  to the decoder's pool on destruction, with an explicit CPU-access window
  (`beginCpuRead`/`endCpuRead`) for the cases that must read the pixels.
- `media/JpegCodec` — hardware JPEG encode/decode over the same `rk_mpi` path.
- `backend/EnginePool` — N `rknn_dup_context` duplicates sharing one model's
  weights, one per NPU core, leased out under RAII.
- `pipeline/AsyncDetectionPipeline` — overlaps preprocessing, inference and
  decoding across the three NPU cores and returns results in submission order.
  The letterbox runs on the calling thread, straight into the leased context's
  own input dma-buf, so the source frame is fully consumed when `submit()`
  returns and needs no lifetime rule. *Measured on ResNet-18: 481 fps vs 99 fps
  synchronous — a 4.86x speed-up — with results byte-identical to the
  synchronous pipeline and in order.*
- The LTRB detection decoder now uses the export's **score-sum branch** as a
  pre-filter. Every class score is non-negative, so the per-cell sum bounds the
  maximum from above and a cell below the confidence threshold can be skipped
  without reading its class channels — which is the whole cost of the decode
  (8400 cells x 80 classes, strided by H*W). *Measured on YOLOv8n at 640:
  post-processing 29.9 → 4.1 ms/frame, and the full VPU → RGA → NPU → overlay →
  VPU pipeline 14.4 → 37.2 fps, with byte-identical detections.*
- `pipeline/TrackingPipeline` — detect-and-track in one call, with an optional
  second Engine for ReID-gated association. Takes a decoded `VideoFrame`
  directly (no copy on the geometry-only path; a CPU window is opened only when
  appearance crops must be read). *Verified: 3 objects keep 3 stable ids across
  a 24-frame panning sequence.*
- `tracks/ByteTracker` — ByteTrack with a constant-velocity Kalman filter,
  Hungarian association, the low-score rescue pass, BoT-SORT appearance fusion
  and BoostTrack++ similarity, plus `tracks/reid` embedding primitives.
- Examples: `video_decode`, `video_roundtrip`, `jpeg_roundtrip`,
  `video_det_demo` (VPU → RGA → NPU → overlay → VPU), `async_bench`,
  `async_check`.
- Python: `rcdl.VideoDecoder` / `VideoEncoder` / `VideoFrame` / `JpegEncoder` /
  `JpegDecoder` / `decode_video`, `rcdl.ByteTracker`, `reid_preprocess`, and
  `DetectionPipeline.process_frame()` for the zero-copy path from a decoded
  frame.
- `docs/RGA.md` — the measured RGA constraints, including two that are not in
  the vendor documentation (see below).

- Task heads (M4), each a pure decode function plus a thin `Engine`-bound class
  that resolves the model's real output layout at construction and throws with
  the full signature printed when it does not match:
  `tasks/classification` (stable softmax, top-k, torchvision centre-crop
  geometry), `tasks/embedding` (L2 / cosine / euclidean, `EmbeddingBank`),
  `tasks/pose` (COCO-17 keypoints + skeleton), `tasks/instance_seg` (prototype
  masks, cropped and un-letterboxed to source pixels), `tasks/segmentation`
  (argmax label map, optional confidence, palette), `tasks/obb` (rotated boxes,
  rotated IoU and NMS), `tasks/depth` (affine / inverse-depth handling,
  colorization).
  *Verified against real models on RK3588*: instance seg finds the same
  1 bus + 4 people as detection with 32–63% mask coverage per box; PP-LiteSeg
  resolves a Cityscapes street scene into 11 classes (road 41%, vegetation 31%);
  pose puts all 17 keypoints on the right body parts; OBB finds 4 planes and
  ~24 vehicles at a consistent angle; YOLOv8n and YOLO11n independently agree on
  `bus.jpg`.
- The pose and OBB decoders handle **both** deployed export shapes: the
  separate `(box, cls[, sum])` branches per scale, and the **fused**
  `[1, 4*reg_max + nc, H, W]` tensor the vendor's own pose/OBB exports produce
  (with the class sigmoid left on the CPU, unlike the detection export).
- Python bindings for the whole task layer — classification, embeddings +
  `EmbeddingBank`, instance seg, semantic seg, depth — with the Engine-bound
  classes exposed as `Engine.classifier()` / `.embedder()` /
  `.instance_segmenter()` / `.segmenter()` / `.depth_estimator()` and numpy
  helpers `classify` / `embed` / `segment_instances` / `segment` /
  `estimate_depth` alongside `detect`. Masks, label maps and depth maps come
  back as correctly-shaped numpy arrays.
- Examples: `cls_demo`, `seg_demo` (picks the instance or semantic path from the
  model's output signature), `depth_demo`.
- `docs/API.md`, `docs/CPP_API.md`, `docs/MODELS.md`.
- pip wheel via scikit-build-core, validated on the board: the installed
  package (not the build tree) reports RGA and MPP available and runs detection
  end to end.
- **Monocular depth now has a model.** `DepthEstimator` had a decoder, a numpy
  oracle and no network to run: Depth-Anything-V2-Small (ViT-S, DPT head) is
  converted to int8 at 308×308 and 518×518 and registered in
  `scripts/fetch_models.sh`. Verified against the untouched fp32 ONNX on the
  identical letterboxed input — cosine 0.9894, Spearman 0.9951 — and the
  308×308 build turns out to be both 2.9× faster and *more* accurate than the
  network's native 518×518 resolution. See `docs/MODELS.md` for the numbers,
  the disparity-not-depth convention, and why the `align_corners` upsamples in
  the DPT head stay on the CPU.
- **`TrackingPipeline` is now bound to Python** — `Engine.tracker(reid=...)`,
  `rcdl.track()`, `process_frame()` for the zero-copy video path, plus
  `has_reid` / `last_embed_count` / `letterbox` / `profile`. It was the one
  pipeline class the module did not expose, which is also why its ReID half had
  never been exercised: `ByteTracker` alone left the caller to write the crop,
  embed and association loop by hand.
- **Appearance embeddings now have a model too.** `ImageEmbedder`,
  `EmbeddingBank` and the ReID side of the tracker had the same problem depth
  did — a decoder, a numpy oracle and nothing to run. OSNet x0.25 (MSMT17
  person ReID, 512-d) is converted to int8, calibrated on person crops cut from
  COCO images by the project's own detector, and registered. On `bus.jpg` it
  scores 0.960 / 0.993 for the same person re-cropped and rescaled against
  0.38–0.47 for different people, and matches the fp32 ONNX at cosine 0.9922.
- **On-board assertions for pose, OBB, OCR, faces, depth and ReID**
  (`tests/test_tasks_board_py.py`, 23 end-to-end tests). These five heads
  previously had numpy oracles only, which pin the decode maths but cannot see
  a changed *model contract* — a fused output layout, a channel order, an
  activation applied in the wrong place. Every one of this project's silent
  failures so far has been in that gap. The new tests pin what the hardware
  actually returns: pose keypoints must fall inside their own person's box,
  parked vehicles in the DOTA scene must share an orientation and rotated NMS
  must leave no duplicates, RetinaFace landmarks must be ordered eyes → nose →
  mouth, the OCR lines must match exact strings, and the depth map must put the
  road nearer than the sky and stay consistent when the letterbox padding
  changes. ReID is checked on the property a tracker depends on — the margin
  between same-person and different-person similarity — and through
  `EmbeddingBank` retrieval end to end. Tracking is checked on a synthetic pan,
  where the right answer is knowable — every object is present in every frame,
  so any id beyond the object count is association failing rather than the
  scene changing — in both the geometry-only and ReID modes.
- `rcdl::AsyncVideoDetectionPipeline` — compressed video in, detections out,
  with VPU decode, RGA letterbox and N-context NPU inference each on their own
  thread and results in decode order. `submit()` takes arbitrary chunks of an
  elementary stream (MPP's parser splits them) and reports back-pressure as a
  return value rather than a wait, which is what keeps a single-threaded driver
  from deadlocking against its own bounded queues. Bound in Python as
  `Engine.video_detector()`, with the GIL released around every stage, so a
  Python caller that only pumps bytes gets the C++ throughput: **72–97 fps on
  1080p H.264 → YOLOv8n against 23–28 fps frame-at-a-time, a 3.1–3.7x speed-up
  over six runs**, and per-frame detections identical to the synchronous path. Example:
  `video_det_async` (`--sync` measures both in one run).
- `rcdl::TextAngleClassifier` + `decodeTextOrientation()` — PP-OCR's 0°/180°
  text-line direction head, the piece whose absence made rotated text decode
  *wrong* rather than merely unsupported: measured on the sample page, all 16
  lines rotated 180° come back from the recogniser as the empty string or a
  single stray bracket, with nothing in the result to say the content was lost.
  With the classifier in front they read exactly as they do upright. The flip
  gate is asymmetric on purpose (only class 1 above the threshold flips),
  because flipping an upright line is the one outcome worse than not
  classifying it. Bound in Python as `Engine.text_angle_classifier()`, with
  `ocrLineFitWidth()` exposing the reference preprocessing — fit the crop to the
  model's HEIGHT, cap the width, anchor top-left, pad the rest, which is NOT the
  centred letterbox the rest of the library uses and is worth 16/16 orientations
  right against 9/16. Model: `ppocr_cls_rk3588.rknn` (`ch_ppocr_mobile_v2.0_cls`,
  192×48 BGR, int8), calibrated on line crops cut by RCDL's own detector and
  checked against the Paddle fp32 original — 100% label agreement on a held-out
  split, probability MAE 0.0000.
- `ppocrv5_det_rk3588.rknn` — the PP-OCRv5 mobile detector, kept as the newer
  build with a parity test rather than a claimed improvement: on the sample page
  it finds the same 16 regions as the v4 detector and they yield the same 15
  strings. Its recogniser counterpart is deliberately **not** in the registry —
  correct in the toolkit's simulator and wrong on the NPU (1/16 lines exact in
  float16, 0/16 in int8, int16 refuses to run), so PP-OCRv4 stays the deployed
  recogniser. See `docs/MODELS.md` for the table.
- `Engine` now presents an **int16** quantized input as float32 rather than
  handing the caller raw int16, so a 16-bit model's scale and zero-point do not
  leak into every call site. (The only int16 model built so far cannot run on
  this driver, which is how the mapping came to be needed and why it is not
  demonstrated end to end.)
- `similarityTransform()` / `faceAlignTransform()` / `arcFaceTemplate()` — the
  5-point face alignment an identity embedding is computed on. Written as a
  single complex quotient rather than a Procrustes SVD, because that form
  **cannot represent a reflection**: a mirrored fit matches five landmarks as
  well as the correct one and would silently embed the wrong face. Bound in
  Python (`face_align_transform`, feed the (2,3) result to `cv2.warpAffine`).
  *Verified on the four faces in the sample images: the fit's own residual is
  1.3–7.3 px mean, the detector re-finds every aligned crop at 0.99–1.00, and
  the eyes and mouth corners land on the template's rows.* The identity model
  itself is not in the registry — see ROADMAP M8 for the data it needs first.
- `yolo26n_rk3588.rknn` — a current-generation detector, added with **no
  decoder changes**, which is what the head-resolution design was for: YOLO26's
  box branch has 4 channels instead of 64 (no DFL), and `resolveYoloHead()`
  switches to the plain-LTRB path from the model's signature alone. All three
  generations now return the same 1 bus + 4 people on `bus.jpg`, pinned by a
  cross-generation test that also asserts the resolved layout, and
  postprocessing drops to 6–8 ms against 13 (v8n) and 13–18 (11n) — inference
  does not improve, at 36–38 ms against 26.
- `yolo26n-seg_rk3588.rknn` and `yolo26n-pose_rk3588.rknn`, and with them
  `KeypointDecode::kCellRelativeWhole` — the one piece of the YOLO26 family that
  did need code. YOLOv8/YOLO11 predict a keypoint offset in HALF cells and
  YOLO26 dropped the doubling; nothing in the tensor distinguishes them, and the
  older formula puts every joint at twice its distance from the cell centre —
  still a skeleton, still drawable, still wrong. Measured over one model with
  both formulas: 16/16 confident joints inside their own person's box with the
  new mode, 5/16 with the old one, which is what the board test asserts.
- `yolo26n-obb_rk3588.rknn` and `yolo26n-cls_rk3588.rknn` complete the YOLO26
  family, with `ObbConfig::angle_scale` for the one convention change that
  needed code: YOLO26 regresses the oriented-box angle in **radians**, where
  YOLOv8 emitted a fraction of a half-turn through an in-graph sigmoid. Decoded
  with the wrong one the boxes still sit on the objects, just rotated — and
  rotated NMS then merges a different set, so the count changes too. The
  classifier needs no code but does need `apply_softmax=False`: its softmax is
  in the graph, and a second one leaves the argmax intact while flattening the
  score from 0.939 to 0.003. Both are pinned against the framework's own
  outputs.
- `tasks/features` — XFeat sparse local features: `xfeatPreprocess` (channel-mean
  grey + resize + InstanceNorm), `decodeXfeat` (softmax over the 65 logits with
  the 65th as the reject bin, scatter to full resolution, square-window NMS,
  top-k, **bicubic** descriptor sampling with the dense map normalised before and
  each descriptor after), `FeatureExtractor`, and `matchFeatures` — mutual
  nearest neighbour with a cosine floor, OpenMP over one pass that keeps both
  directions instead of materialising the similarity matrix. With
  `xfeat_640x480_i8_rk3588.rknn`, `xfeat_demo` and Python bindings
  (`Engine.feature_extractor()`, `extract_features`, `match_features`,
  `decode_xfeat`, `xfeat_preprocess`).
  This is the first head that answers a question about *two* frames, and the only
  one whose ground truth can be manufactured: rotate a photograph by a known
  amount and every correspondence has an exact right answer. *Verified on the
  board: 76.6% of 2033 matches within 3 px of where a 12°/0.85 warp puts them
  (median 1.36 px), against 77.4% / 1.35 px for the float ONNX on CPU — and 69
  matches, not 2033, between two different scenes.*
- `EngineOptions::float_inputs` (Python: `Engine(..., float_inputs=[0])`) — name
  the inputs whose tensor is a normalised **map** rather than image bytes, and
  they are presented to the runtime as float32 instead of u8. XFeat is the case:
  its input is an InstanceNorm output, roughly ±3, and the u8 path a quantized
  model normally takes has no negative range at all, so half of it would clip to
  the zero point while the head went on returning keypoints that look like
  keypoints. `FeatureExtractor` checks `inputType()` and refuses to construct on
  an Engine that was opened the ordinary way.
- `tasks/superres` — ×4 super-resolution by tiling a fixed-size upscaler:
  `planTiles` (coverage with the last tile flush against the far edge rather
  than overhanging), `tileWeight` (a cross-fade that is never zero, so the
  normalized blend needs no border case) and `SuperResolver`, which reads the
  scale factor and tile size off the model's own shapes. With
  `realesr_general_x4v3_128_fp16_rk3588.rknn` and its int8 sibling, `sr_demo`
  and Python bindings (`Engine.upscaler()`, `upscale`, `plan_tiles`,
  `tile_weight`).
  Two things here produce a plausible picture when they are wrong, so both are
  pinned by tests: the tensor wants **0..255 in both builds** — bytes for int8,
  floats still in 0..255 for fp16, because the ÷255 is inside the model, and
  feeding 0..1 measures 7.3 dB against the reference instead of 63.2 — and the
  model is RGB where every image entry point here is BGR.
  *The registry keeps fp16 as the default on measurement: it reproduces the
  float model at 63.2 dB where int8 manages 31.5 and over-sharpens past the
  original (edge energy 66.4 against the original's 62.3), for 1.6× the speed.
  Judge this family by edge energy, not PSNR — it is trained perceptually and
  scores below a bicubic resize on PSNR while looking obviously sharper.*
- `backend/custom_ops` — CPU kernels for operators librknnrt does not implement,
  registered on every Engine (`EngineOptions::custom_ops`, on by default;
  `rknn_register_custom_ops` needs a live context, so it happens during
  construction and again for every `dup()`).
  The op that forced this is `GridSample`, which librknnrt 2.3.2 has neither on
  the NPU nor on its own CPU fallback list: the toolkit lowers such a node to a
  generic CPU node with one line in the log, and the runtime then **segfaults
  inside `rknn_init`** — nothing a caller can catch. Converted with the node
  declared as a custom operator instead, the same graph loads, and a missing
  registration downgrades to an error at `rknn_run`.
  Two undocumented details are handled in the kernel because both silently
  produce a plausible field: the runtime returns operator attributes as **quoted
  text** (an int attribute can arrive as `"1"`), and it labels the sampling grid
  NCHW while laying it out as ONNX defines it, so the shape and not the format
  field decides whether coordinate pairs are interleaved.
- `tasks/optical_flow` — dense two-frame flow: `decodeFlow` (planar or
  interleaved, with the model->source pixel scale that is otherwise applied
  silently), `flowColorize` (Middlebury wheel, normalised by the field's own 99th
  percentile so one fast object cannot flatten the frame), `flowEndpointError`,
  `flowPreprocess` and `OpticalFlowEstimator`. With
  `neuflow_v2_512x384_fp16_rk3588.rknn`, `flow_demo` and Python bindings
  (`Engine.flow_estimator()`, `estimate_flow`, `decode_flow`, `flow_colorize`,
  `flow_endpoint_error`, `flow_preprocess`).
  *Verified against manufactured ground truth — move a window across a
  photograph and the correct field is that constant: 0.10-0.13 px endpoint error
  on 0-16 px shifts. The suite also scores a 3 degree ROTATION, where the true
  field varies across the frame: 0.751 px on the NPU against 0.145 px for both
  the float ONNX and the toolkit's simulator. The kernel is not the gap — dumped
  from a real call it matches a numpy reference to 4.6e-06 — the simulator simply
  does not model fp16 arithmetic, and a uniform shift cannot see the difference.*
  Each GridSample sits between two NPU subgraphs, so a frame costs about 1.4 s:
  this head is correct, not fast.
- `rcdl::floatToHalf` beside the existing `halfToFloat` (Python:
  `rcdl.float_to_half`) — the direction a CPU kernel writes in when the graph
  around it carries fp16. It sits on a hardware path, so it is checked against
  numpy's own conversion over 8k values including subnormals, the largest
  normal, and overflow to infinity, rather than a handful of spot values.

### Fixed

- `VideoDecoder` treated `MPP_NOK` from `decode_get_frame` as fatal. MPP returns
  it with a null frame to mean "nothing ready yet", so the frame pointer — not
  the return code — is the signal. The old rule happened to work at 1080p and
  killed every 4K H.265 stream.
- Rotated IoU ran its Sutherland–Hodgman clip in float32. When two boxes share
  an edge, every vertex on it has a signed distance of zero and rounding decides
  which side it falls on, so the polygon self-crossed and its area became
  unrelated to either box: `rotatedIoU(box, box)` returned **−39.6** for a
  rotated box instead of 1.0. Boxes in general position were unaffected, which
  is why a randomized cross-check would not have found it. The clip now runs in
  double.
- `scripts/sync.sh` never copied the root Markdown files to the board. They are
  whitelisted in `.gitignore` with `!/README*.md`, but rsync's gitignore filter
  does not honour those negations the way git does — which stayed invisible
  until a board-side `pip install .` failed with "Readme file not found".
- `PreprocBackend::Auto` fell through to an exception when RGA accepted a
  request at `imcheck` time and then refused it at submit. It now falls back to
  the CPU on a run-time refusal as well as a pre-flight one.
- The RGA letterbox painted its border with the **CPU**, and no ordering of that
  write is safe. Before the blit, librga's destination import discards it and
  the bands come back with 64-, 128- or 192-byte runs of stale content; after
  the blit, the band edge lands mid-cache-line and the fill's read-modify-write
  puts stale bytes back over pixels the hardware just wrote. Because the
  destination is normally the NPU's input tensor, reused every frame, both
  variants feed the model the previous frame's pixels — silently, since the
  boxes stay plausible. Measured on a 16-frame clip letterboxed 816x1088 →
  640x640, three runs of the same bytes disagreed on **7 to 16 of 16 frames**
  (boxes ~1 px, scores ~0.005); the same clip on a square canvas, which needs no
  border, was identical every time. The border is now **blitted by RGA** from a
  flat-grey source allocated once per format/pad/size: every write to the
  destination comes from one engine, in order, and the same three runs are now
  identical on every frame. Costs one extra RGA op (about +0.4 ms/frame at
  640x640: 2.39 ms to 2.8-2.9 ms of letterbox per frame).
  The CPU band fill remains the fallback for a destination RGA will not take.
  The hardware colour fill is still probed once on a private scratch buffer
  rather than by attempting it on a live destination (a rejected fill damages
  what it was pointed at).
- The RGA letterbox left a one-pixel band of the canvas unwritten when
  `dst - scaled == 1`, and did not even-align its destination rectangle for
  4:2:0 destinations. Since the destination is normally the NPU's input tensor,
  reused every frame, that band fed the model the previous frame's pixels.

### Known hardware limitations (RK3588, measured)

- **RGA colour fill is unusable when the board has more than 4 GB of RAM.** The
  driver routes `imfill` to the RGA2 core, which has no IOMMU and cannot map
  pages above 4 GB. `rgaFill()` falls back to a CPU fill after the first
  failure, and the letterbox paints only its border bands. See `docs/RGA.md`.
- **RGA3 scales within [1/8, 8], not the [1/16, 16] `imcheck` accepts** — the
  wider range belongs to RGA2, so a ratio outside the narrower window is routed
  there and fails at submit. `PreprocBackend::Auto` rejects it up front and also
  catches a run-time refusal.
- **Row strides must be aligned for RGA**: 16 pixels for 3-byte packed RGB and
  for YUV luma, 4 for 4-byte packed RGB. Images RCDL allocates always are; a
  foreign buffer (a `cv::Mat` over an 810-pixel-wide JPEG) may not be and takes
  the CPU path.
- **GRAY8 is refused by RGA on this board** in every direction, and falls back
  to the CPU.
- **MPP does not decode MJPEG through its streaming API.** JPEG is forced onto
  MPP's "advanced" path, where the caller must attach the destination frame to
  the packet's metadata (`KEY_OUTPUT_FRAME`) and the input packet must itself be
  backed by an `MppBuffer`. Without both, `decode_get_frame` returns
  `MPP_NOK` with a null frame forever and no info-change is ever raised.
  `JpegDecoder` therefore parses the JFIF `SOFn` header itself to size the
  output buffer before decoding — which also lets it reject progressive,
  arithmetic and lossless JPEG with an accurate reason (`lastError()`) instead
  of a guess. Baseline 4:2:0, 4:2:2, 4:4:4 and grayscale all decode and come
  back as NV12.
