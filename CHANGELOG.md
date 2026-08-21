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
- The RGA letterbox wrote its border bands **before** the blit, and librga's
  destination import discards CPU writes made beforehand — so the bands came
  back with 64-, 128- or 192-byte runs of stale content on most frames. Because
  the destination is normally the NPU's input tensor, reused every frame, that
  fed the model the previous frame's pixels in the padding, silently. The border
  is now written after the blit, and the hardware fill is probed once on a
  private scratch buffer rather than by attempting it on a live destination
  (a rejected fill damages what it was pointed at).
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
