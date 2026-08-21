# C++ API

```cmake
find_package(rcdl REQUIRED)
target_link_libraries(app PRIVATE rcdl::rcdl)
```

```c++
#include "rcdl/rcdl.h"      // or the individual layer headers
```

Every header carries the reasoning in its comments; this page is the map and the
handful of rules that are easy to get wrong.

---

## The one idea

**A dma-buf fd is the only buffer.** The NPU imports it
(`rknn_create_mem_from_fd`), RGA imports it (`importbuffer_fd`), the VPU decodes
into it and encodes out of it (external `MppBuffer`). So a frame moves from the
decoder to the 2-D engine to the model without ever being copied, and the CPU
only sees it if you ask.

That is why the central type is a *descriptor*, not an owner:

```c++
struct ImageView {           // preproc/image.h
  void* data;                //   CPU mapping, or null — fd-only is normal
  int fd;                    //   dma-buf fd, or -1 for a host buffer
  int width, height;
  int wstride, hstride;      //   PIXELS and ROWS, not bytes; often > width/height
  PixelFormat format;
  std::size_t size;
};
```

`rcdl::Image` owns one (`Image::alloc`), `VideoFrame::view()` describes a
decoded frame, `engineInputView(engine, 0, fmt)` describes the NPU's own input
tensor, and `hostView()` wraps a `cv::Mat`. Anything that takes an `ImageView`
takes all four.

**Always read strides through `effWStride()` / `effHStride()` / `rowBytes()` /
`uvOffset()`.** An NPU input tensor's row stride is aligned up, a VPU frame's is
aligned up, and `strideAlign()` aligns what RCDL allocates — assuming
`stride == width` is the single most common way to get a sheared image.

---

## Layers

| Header | What is in it |
|---|---|
| `core/status.h` | `rcdl::Error`, `RCDL_CHECK`, `RCDL_REQUIRE` |
| `core/dma_buf.h` | `DmaBuf` (dma-heap alloc, `syncStart`/`syncEnd`), `dmaBufSyncStart/End(fd, ...)` |
| `backend/engine.h` | `Engine`, `EngineOptions`, `NpuCore` |
| `backend/engine_pool.h` | `EnginePool`, `EnginePool::Lease` |
| `backend/output_reader.h` | `outputAsFloat`, `dequantizeToFloat`, `sigmoid`, `quantParams` |
| `preproc/image.h` | `PixelFormat`, `ImageView`, `Image`, `engineInputView`, `strideAlign` |
| `preproc/geometry.h` | `LetterboxInfo`, `computeLetterbox`, `YuvRange` |
| `preproc/letterbox.h` | `letterbox` / `resize` / `cvtColor` + `PreprocBackend` |
| `preproc/rga.h`, `preproc/letterbox_cpu.h` | the two backends, if you want one specifically |
| `media/video_codec.h`, `media/video_frame.h`, `media/jpeg_codec.h` | VPU codecs |
| `tasks/*.h` | detection, classification, pose, instance seg, semantic seg, OBB, depth, OCR, embedding |
| `tracks/byte_tracker.h`, `tracks/reid.h` | ByteTrack + appearance |
| `pipeline/detection_pipeline.h`, `pipeline/async_detection_pipeline.h` | streaming |

---

## The shape of a task head

Every head is two things, and you can use either:

```c++
// 1. a pure function over float buffers — no Engine, no hardware, testable anywhere
std::vector<Detection> decode(const float* data, const std::vector<int>& shape,
                              const DetectConfig& cfg, const LetterboxInfo& lb);

// 2. an Engine-bound class that reads the outputs and calls it
YoloLtrbDetector det(engine);
std::vector<Detection> dets = det.postprocess(lb);
```

The bound class resolves the model's layout **at construction** and throws with
the full output signature printed if the model is not the head it was asked for.
Structure (grids, class count, `reg_max`, channel order, strides) always comes
from the model; the config supplies only thresholds and the two things the file
cannot tell you — channel order and where the activation lives (see
`docs/MODELS.md`).

Reading outputs goes through `outputAsFloat(engine, i, scratch, shape)`: a
packed f32 tensor is returned zero-copy, anything else (the usual int8-affine
RKNN output, fp16, a stride-padded layout) is gathered into `scratch`.

---

## Detection, end to end

```c++
rcdl::Engine engine("yolov8n_rk3588.rknn");
rcdl::PipelineConfig cfg;
cfg.model_input = rcdl::PixelFormat::RGB888;     // the export's channel order
rcdl::DetectionPipeline pipe(engine, cfg);

for (;;) {
  auto dets = pipe.process(frame.view());        // RGA -> NPU input tensor -> decode
  ...
}
```

`process()` allocates nothing per frame. The letterbox destination *is* the
NPU's input tensor, so one `improcess` crops, scales, converts NV12→RGB888 and
paints the border straight into the buffer the NPU is about to read.

Three cores, results in submission order:

```c++
rcdl::AsyncDetectionPipeline pipe(engine, cfg, {.workers = 3});
pipe.submit(frame.view());                       // letterbox on this thread, then hand off
std::vector<rcdl::Detection> dets;
while (pipe.next(dets)) { ... }
pipe.finish();
```

The letterbox runs on the **calling** thread deliberately: the source frame is
fully consumed when `submit()` returns, so a decoded frame can go back to the
VPU immediately and there is no lifetime rule to get wrong. If you must not hold
a frame while waiting for capacity, use the split form —
`acquireSlot()` / `letterboxIntoSlot()` / `commitSlot()` — which waits for a slot
before you take a frame.

---

## Compressed video in, detections out

`AsyncVideoDetectionPipeline` owns the whole path — VPU decode, RGA letterbox,
NPU inference on N contexts — so the caller only pumps bytes:

```c++
rcdl::VideoDecConfig dec{.codec = rcdl::VideoCodec::H264};
rcdl::AsyncVideoDetectionPipeline pipe(engine, cfg, dec);

std::vector<rcdl::Detection> dets;
while (std::size_t n = fread(buf, 1, sizeof buf, fp)) {
  while (!pipe.submit(buf, n) && !pipe.finished())
    while (pipe.tryNext(dets)) { /* make room, then retry */ }
  while (pipe.tryNext(dets)) { /* ... */ }
}
pipe.finish();
while (pipe.next(dets)) { /* the reorder tail */ }
```

Chunks may be any size — MPP's parser finds the access units — and results come
back in decode order with `lastPtsUs()` / `lastFrameIndex()` beside them, which
is all that ties a result to its frame once the decoded buffer has gone back to
the pool.

**`submit()` returning `false` is back-pressure, and it must not be a wait.**
Every queue in the pipeline is bounded and the last one is emptied only by
`next()`, so a single-threaded caller that blocked inside `submit()` would
deadlock against itself: no input accepted until frames move, no frames until a
context frees, no context until results are drained, and the only thread that
can drain is the one parked in `submit()`. Hence the retry loop above.
`finished()` distinguishes back-pressure from a closed pipeline; a caller that
drains on another thread can pass a negative timeout and block instead.

Measured on the board (1080p H.264 → YOLOv8n, three pinned contexts), over six
runs of a 300-frame clip: **72–97 fps against 23–28 fps for the same stream
through the synchronous pipeline, a 3.1x to 3.7x speed-up**, with the per-frame
detections identical to the synchronous path. The spread is the board's, not the
pipeline's — both figures move together run to run, which is why a range is
quoted rather than the best single number.

---

## Media

```c++
rcdl::VideoDecoder dec({.codec = rcdl::VideoCodec::H264});
dec.feed(bytes.data(), bytes.size());            // false == back-pressure, drain and retry
rcdl::VideoFrame frame;
while (dec.receive(frame, 0)) {
  auto dets = pipe.process(frame.view());        // no copy: RGA reads the VPU's buffer
  enc.feed(frame.view());                        // no copy: the VPU reads it back
}
```

**Release frames promptly.** Every frame you hold is one buffer the decoder
cannot reuse; holding the whole pool stalls decoding. `VideoFrame` is move-only
and returns its buffer in the destructor — let it go out of scope, or
`reset()` it.

Reading a decoded frame with the CPU needs a coherency window, because the VPU
wrote it through an IOMMU without touching the caches:

```c++
const std::uint8_t* y = frame.beginCpuRead();
...
frame.endCpuRead();
```

---

## Cache discipline, in one rule

- CPU wrote it, hardware will read it → **flush** before the op.
- Hardware wrote it, CPU will read it → **invalidate** before reading.
- It only ever moves between hardware units → **neither**, and skipping it is
  the whole point.

`DmaBuf::syncStart/syncEnd` and the free `dmaBufSyncStart/End(fd, ...)` do it.
The RKNN runtime flushes its own I/O tensors around `rknn_run`, and the preproc
wrappers leave fd-backed buffers alone for exactly this reason.

---

## Errors

`rcdl::Error` derives from `std::runtime_error` and carries the vendor return
code (`.code()`). `RCDL_CHECK(expr)` wraps a call returning 0-on-success;
`RCDL_REQUIRE(cond, "msg")` is a precondition. librga inverts the convention
(1 == success), so its wrappers normalise before throwing.

---

See `docs/API.md` for Python, `docs/RGA.md` for what the 2-D engine will and
will not do, and `docs/MODELS.md` for the per-model contract.
