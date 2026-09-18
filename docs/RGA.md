# RGA — the 2-D engine, and what it will and will not do

RCDL's rule is that **RGA does every resize, colour-space conversion and
letterbox**, and the CPU paths in `preproc/letterbox_cpu.h` exist only as a
guarded fallback. This page is what we measured on RK3588 while making that
true: the constraints that decide when RGA is used, one hardware limitation
that forced a design change, and the accuracy relationship between the two
backends.

Everything here was measured on an RK3588S board — Linux 6.1, librga 2.2.0
(`RGA_api v1.10.4`, hardware `RGA_2_Enhance RGA_3`), 16 GB of RAM.

## 1. The one call that matters

```c++
rcdl::LetterboxInfo lb = rcdl::letterbox(dst, src, /*pad=*/114);
```

`dst` is normally not a scratch canvas at all — it is the NPU's input tensor:

```c++
rcdl::ImageView in = rcdl::engineInputView(engine, 0, rcdl::PixelFormat::RGB888);
rcdl::letterbox(in, decoded_frame.view());   // NV12 dma-buf -> RGB888 tensor
engine.infer();
```

`Engine` allocated that tensor with `rknn_create_mem` and bound it once with
`rknn_set_io_mem`, so it is a dma-buf. `engineInputView()` hands RGA its fd, and
a single `improcess()` crops, scales, converts NV12 → RGB888 and lands the
result in the buffer the NPU is about to read. **No intermediate canvas, no CPU
copy, no per-frame allocation.** That is the whole reason the preproc layer is
shaped the way it is.

`improcess(src, dst, pat, srect, drect, prect, ...)` is what makes it one pass:
`srect` is the crop, `drect` is where it lands (so scaling and centring are the
same operation), and the format difference between the two buffers is the colour
conversion.

## 2. When RGA is used, and when it is not

`PreprocBackend::Auto` (the default) asks `rgaCanHandle()` — which runs librga's
own `imcheck` — and falls back to the CPU when the answer is no. The RGA3
constraints that trigger that:

| Constraint | Value | What happens outside it |
|---|---|---|
| Scale factor | 1/8 .. 8 (RGA3; 1/16 .. 16 on RGA2 with low buffers, §3.1) | rejected → CPU |
| Minimum source for a scaled op | 68 × 2 (documented) | **not** rejected — see below |
| Maximum dimension | 8192 | rejected → CPU |
| YUV row stride | 16-byte aligned | rejected → CPU |
| YUV width/height | even | rejected → CPU |

`rcdl::Image::alloc()` and `strideAlign()` apply the 16-pixel YUV alignment for
you, so buffers RCDL allocates are always acceptable; a buffer from somewhere
else may not be.

**The minimum-size row is the one `imcheck` does not enforce**, and it is worth
knowing before trusting `rgaCanHandle()` on small rectangles. Measured on this
board with a scaled RGB888 blit, a destination rectangle 32, 48, 64 or 80 px
wide is *accepted by the check* and then fails at SUBMIT — `Failed to call
RockChipRga interface`, an ioctl and a page of driver log per call — while 96 px
wide works. The op still produces the right answer, because `PreprocBackend::Auto`
falls back to the CPU after the failure; what it costs is the wasted submit and
the noise. OCR hits this on every short line crop, so the two line-crop paths in
the Python layer go straight to the CPU below 96 px rather than asking.

The rule was not pushed into `rgaCanHandle()` because the measured boundary is
not a simple minimum: a 320×80 source scaled *up* into a 320×96 destination
works, while an 80×128 source scaled into the same destination does not. Encoding
half a rule there would trade a loud failure for a silent slowdown on shapes that
do work.

Ask before you commit, without an exception:

```c++
std::string why;
if (!rcdl::rgaCanHandle(dst, src, &why)) { /* why says which rule it broke */ }
```

To know which backend actually ran on a given frame — the thing to check when a
frame is unexpectedly slow — pass the out-parameter, or read
`DetectionPipeline::lastBackend()`:

```c++
rcdl::PreprocBackend used;
rcdl::letterbox(dst, src, 114, rcdl::PreprocBackend::Auto, range, &used);
```

## 3. Two generations of core, and the 4 GB line

RK3588 has **two RGA3 cores and one RGA2 core**, and they are not
interchangeable. Rockchip's own numbers, confirmed by the driver's
`/sys/kernel/debug/rkrga/hardware` on this board:

| | RGA3 (×2) | RGA2 (×1) |
|---|---|---|
| MMU | 40-bit IOMMU — all memory | **32-bit — nothing above 4 GB physical** |
| Input / output | 68×2 .. 8176×8176 / 8128×8128 | 2×2 .. 8192×8192 / **4096×4096** |
| Scale | 1/8 .. 8 | 1/16 .. 16 |
| Row stride | 16-byte aligned | 4-byte aligned |
| Only here | AFBC | **colour fill**, ROP, palette, GRAY8 (YCbCr400), YUV planar, mosaic |

Two consequences shape everything below.

### 3.1 What the 4 GB line does

**Symptom.** Every `imfill` fails, on any pixel format, from malloc or from
the `system` dma-heap:

```
IM_STATUS 0: Failed to call RockChipRga interface
rga: RGA_MMU unsupported memory larger than 4G!
rga: scheduler core[4] unsupported mm_flag[0x0]!
```

Colour fill is RGA2's feature, RGA2 cannot map a page above 4 GB, and on a
16 GB board that is where a `system` allocation lands. The same wall stops
GRAY8 and any scale ratio beyond 8×. There is no bounce buffer, kernel option
or device-tree property that changes this; Rockchip's FAQ (Q4.5) says exactly
that colour fill and YUV planar "must allocate memory within 4G", from the
`system-dma32` dma-heap.

**What RCDL does.** The buffer carries the fact:

- `DmaBuf::Heap::SystemDma32` allocates from `/dev/dma_heap/system-dma32`,
  and a `DmaBuf`, `Image` or `ImageView` from it says so (`below4G()` /
  `ImageView::below4g`). `VideoDecConfig::pool_heap = SystemDma32` puts the
  whole decoder pool there, and every frame it hands out is flagged (Python:
  `VideoDecoder(pool_heap="system-dma32")`, `VideoFrame.below_4g`).
- An op that needs RGA2 runs on the hardware **only when both buffers are
  flagged** (or the board has no memory above 4 GB at all — the one case where
  the ordinary heap is probed, once, on a private scratch buffer). Otherwise
  `rgaCanHandle()` answers no up front and the `Auto` backend takes the CPU
  path, without a failed ioctl and a page of kernel log per frame. The
  hardware fill goes the same way: `rgaHwFillUsable(dst)` is per buffer, and
  `rgaFill()` writes with the CPU when it is false.
- The NPU's input tensors are allocated by the RKNN runtime and are never
  low, so nothing that targets them can use the fill — which is why the
  letterbox border is a **blit** (§3.3), not a fill.

The dma32 heap is the low quarter of a 16 GB board and shared with the kernel
and CMA: it is the heap for the few buffers that need it, not a default.

### 3.2 The driver load-balances, so RCDL pins the core

Left to itself the driver gives a job both cores can do to whichever is free —
and the two **do not resample the same way** (Rockchip FAQ Q2.22: the sampling
phase differs, the picture shifts). Measured here, a 1080p NV12 frame scaled
to 640×360 on RGA3 and again on RGA2 differed in **78% of its bytes, by up to
147 LSB**, while two RGA3 runs were identical. A pipeline whose frames landed
on either core would therefore not be reproducible — the same non-determinism
§3.3 fought, from the other side.

So every op in `preproc/rga.cc` pins the core it wants right before it
submits: RGA3 for resize / convert / letterbox / copy, RGA2 for fill, GRAY8 and
the wide ratios. `imconfig(IM_CONFIG_SCHEDULER_CORE, …)` is **thread-local**
(measured: a fill from a fresh thread ran while the main thread was pinned to
RGA3), so the async pipeline's workers do not interfere. A board without one
family (RK356x has only RGA2) gets the one it has.

This is also what makes RCDL work on a stock kernel: without the pin, the
driver can hand a copy or a 1/4 downscale to RGA2 because RGA2 was idle,
discover at map time that the buffer is above 4 GB, and fail — a kernel patch
that re-routes such jobs is one answer, the pin is the one that ships with the
library.

### 3.3 The letterbox border is a blit, because a CPU fill is not reproducible

With the hardware fill out of reach for NPU tensors, the obvious substitute is
to paint the letterbox border with the CPU. **Do not**, and this is the
sharpest measurement in this document, because the failure is invisible.

Setup: a 16-frame 816x1088 H.264 clip, decoded on the VPU, letterboxed by RGA
into a 640x640 NPU input tensor (so padX = 80, padY = 0), YOLOv8n. The same
clip, the same bytes, the same process, run three times:

| border | frames whose detections changed between runs |
|---|---|
| CPU fill after the blit | **7 to 16 of 16** (boxes moving ~1 px, scores ~0.005) |
| CPU fill before the blit | **16 of 16** |
| no border fill at all | 0 of 16 |
| square canvas, so no border at all | 0 of 16 |
| **grey source blitted by RGA** | **0 of 16** |

The decoded frames are bit-identical across runs (checked), RGA's letterbox is
bit-identical when repeated on one frame, and `rknn_run` is bit-identical on one
input. Only the CPU border is not, and it does not merely corrupt the border: a
band edge lands mid-cache-line, so filling it is a **read-modify-write of a
line the hardware just wrote**, and whichever of the two writebacks lands second
wins. Filling before the blit loses the band instead — librga's cache
maintenance on import discards the CPU's writes — which is the same effect from
the other side. There is no order that is safe.

So `rgaLetterbox()` paints the border with **RGA**: a small flat-grey source
(allocated once per format/pad/size, CPU-written once, thereafter read-only) is
stretched over the whole canvas, and the image blit lands on top of it. Every
write to the destination then comes from one engine, in order. It costs one
extra RGA op — letterboxing 1080p into a 640x640 canvas went from 2.39 ms/frame
to 2.8-2.9 ms across four runs, so about **+0.4 ms/frame** — and buys a pipeline
whose output does not depend on timing.
The CPU band fill remains as the fallback for a destination RGA will not take as
a blit target, which is not the NPU-input path.

Worth stating plainly, because it is the reusable lesson: **the bug was not that
results were wrong, it was that they were not the same twice.** Nothing failed,
no ioctl returned an error, and every box looked reasonable. It surfaced only
when a new test compared two runs of the same clip frame by frame.

### 3.4 The box overlay is CPU work, and that is the measurement talking

Drawing detection outlines on the decoded frame *can* go to the hardware now
(`rgaDrawRects(..., PreprocBackend::Rga)`, on a dma32 pool), and it is the
slower choice:

| 1080p NV12, outlines 2 px | 5 boxes | 20 boxes |
|---|---|---|
| CPU — one `mmap`, one cache sync, all boxes | 0.25–0.45 ms | **0.3 ms** |
| RGA2 — `imrectangleArray`, one submit per colour run | 0.8–1.8 ms | 3.2–13.7 ms |

Each outline is four fill jobs on the one RGA2 core, at roughly 0.13 ms of
fixed cost per job. The CPU cost is almost all the cache sync of the frame,
which is paid once — the earlier implementation mapped the frame and synced
the whole buffer *per band*, forty times for ten boxes, and that is what made
the CPU look slow.

Writing the overlay with the CPU is not the §3.3 hazard: nothing else writes
the frame after the decoder, the sync window invalidates the lines before the
CPU touches them and flushes them before the encoder reads. Measured, the same
clip encoded twice with CPU overlays gives the same bitstream.

Both backends produce the **same bytes**: the rectangle is clipped to the frame
first, snapped outward to even pixels on 4:2:0 with the thickness rounded up
to even, drawn solid when too small for an outline, painted in the order
given, and the colour is converted with the BT.601 studio-range integer
matrix the hardware uses (pure red is Y=82 Cb=90 Cr=240).
`tests/test_rga_overlay_py.py` pins this against a numpy reference on both
backends.

### 3.5 A note on the clock

A synchronous pipeline spends most of a frame waiting on the NPU, and under
the `schedutil` governor the CPU cores clock down while it waits — so the CPU
stages that follow start slow. On this board, `video_det_demo` at 816×1088:

| governor | postproc | overlay (CPU) | end to end |
|---|---|---|---|
| schedutil | 12.4 ms | 2.7 ms | 53 ms |
| performance | 1.4 ms | 0.26 ms | 21.5 ms |

The overlay did not get faster; the clock did. Before optimising a CPU stage
that runs right after a hardware wait, check the governor.

### 3.6 Alternatives measured, and why they are not the default

Rockchip's guide and FAQ suggest three more levers. Each was measured here
before deciding against it; the numbers are the reason, and they are the ones
to re-check on a different kernel or librga.

| lever | what the docs say | measured on this board | verdict |
|---|---|---|---|
| **Import once, wrap by handle** (`importbuffer_fd` + `wrapbuffer_handle`, FAQ Q1.10–11) | per-call import is "time-consuming"; keep a buffer pool | 1080p NV12 → 640×640 blit: 2.07 ms by fd, 2.12 ms by handle; 8 outlines: 5.8 vs 5.4 ms | no gain — driver 1.3.x already caches the mapping; per-call fd stays, and no handle lifetime to get wrong |
| Skipping `imcheck` before the op | — | 2.11 vs 2.12 ms | free; keep the check, its error names the offending view |
| Partial cache sync (`DMA_BUF_IOCTL_SYNC_PARTIAL`, a Rockchip kernel extension) | sync only the rows an overlay touched | `ENOTTY` on this kernel | not available; a whole-frame sync is ~0.2 ms anyway |
| **Encoder OSD** (`MPP_ENC_SET_OSD_DATA_CFG`) | the VPU blends up to 8 regions at encode time, frame untouched | 8 regions, 16-pixel granularity, palette-indexed bitmaps the CPU has to write per box (a 400×300 box is 120 KB against ~6 KB of outline bands) | wrong tool for per-frame boxes; right for a static logo or timestamp |
| Splitting a blit across both RGA3 cores | — | one blit is hardware-bound at ~2 ms per 1080p frame | halves latency, not throughput: the async pipeline already keeps both cores busy with different frames |

### 3.7 `rga_probe`

`./build/rga_probe` runs the measurements this section rests on — which heap
the fill reaches, whether the core mask is honoured and per thread, the fill
colour on NV12, `imrectangleArray` cost by count, RGA3 vs RGA2 resampling,
GRAY8 and 12× scaling on dma32 buffers — and prints them. Run it on a new
board or kernel before trusting any of the above there.

## 4. RGA and the CPU fallback do not resample identically

They agree to **±1 LSB on band-limited content and disagree substantially on
content that aliases**, because they are different filters:

- **RGA pre-filters when it shrinks** — it averages the pixels it is about to
  discard, like `INTER_AREA`.
- **The CPU path is bilinear point-sampling**, matching
  `cv2.resize(..., INTER_LINEAR)`.

Measured, 1280×720 source, `max` / `mean` absolute difference per channel:

| Content | → 640×640 | → 416×416 | → 320×320 | → 224×224 |
|---|---|---|---|---|
| Linear ramp | 1 / 0.12 | 1 / 0.09 | 1 / 0.12 | 1 / 0.17 |
| Band-limited (smooth) | 2 / 0.31 | 3 / 0.36 | 3 / 0.50 | 5 / 0.67 |
| Aliasing (32-px checkerboard) | 128 / 2.24 | 219 / 4.74 | 187 / 6.52 | 255 / 10.69 |

Neither is wrong. RGA's answer is arguably the better *picture*; the CPU's is
the one that reproduces the preprocessing the models were **quantization-
calibrated** with, which is why it stays the reference and the numpy oracle in
`tests/test_letterbox.py`.

The practical consequence for detection, measured end to end on `bus.jpg` with
YOLOv8n (`tests/test_detection_board_py.py` pins this):

- **The same objects are found either way** — one bus and four people, same
  classes, same count.
- **Well-conditioned boxes agree to ~1.5 px** and their scores to ~0.02.
- **Marginal detections do not.** A person cut off at the frame edge scores
  0.28 through RGA against 0.47 through the CPU, and an occluded person's top
  edge differs by ~70 px even at 0.73 confidence. Those are exactly the cases
  where the model is already unsure of an object's extent, so a small change in
  input pixels has leverage.

So: trust the object set, and the boxes of things the detector is confident and
unambiguous about. If your task turns on the precise extent of partly-visible
objects — or on a few LSB of preprocessing generally — force
`PreprocBackend::Cpu`, which reproduces the calibration-time preprocessing, and
pay the CPU cost knowingly.

## 5. Studio vs full range

Video is **studio-swing** NV12 (Y in [16,235]); models are calibrated on
full-range pixels. Getting this wrong costs about 14% of contrast, which is the
kind of error that shows up as slightly-low confidence scores rather than as an
obvious failure.

`YuvRange::kStudioToFull` (the default) selects limited range on the RGA path
(`IM_YUV_TO_RGB_BT601_LIMIT`) and the expanding matrix on the CPU path.
`YuvRange::kAsIs` treats the source as already full-range — correct for a frame
RCDL itself produced, and what makes an NV12 → RGB → NV12 round-trip
self-consistent.

### The colour matrix

The range is half of it; the other half is the matrix. SD video and JPEG are
**BT.601**; HD video (H.264 / H.265 at 720p and above) normally signals
**BT.709**. Reading a BT.709 frame with the BT.601 matrix does not change
contrast — it shifts hue and saturation, by up to ~30 LSB on saturated colour.

Both halves travel together as `YuvColorSpace { YuvRange range; YuvMatrix
matrix; }`, which every preproc entry point takes. A bare `YuvRange` converts to
it and means BT.601, so existing code is unchanged. The task configs carry a
`yuv_matrix` field next to `yuv_range`; in Python it is
`matrix="bt601" | "bt709"` next to `studio_range=`.

RGA does not cover every combination. Measured on RK3588 (librga 1.10.4), each
supported mode within ±1 LSB of the float reference:

| | BT.601 limited | BT.601 full | BT.709 limited | BT.709 full |
|---|---|---|---|---|
| YUV → RGB | RGA | RGA | RGA | CPU — librga has no mode |
| RGB → YUV | RGA | RGA | CPU — no core accepts the job | CPU |

`rgaCanHandle()` answers no for the CPU cells, so `PreprocBackend::Auto` takes
the CPU path up front instead of paying a failed ioctl; `PreprocBackend::Rga`
throws. The CPU path implements all four in both directions. For the
detection hot path (decoded HD NV12 → RGB888 at studio swing) BT.709 stays on
the hardware.

## 6. Strides are in pixels, and they are not the width

`ImageView::wstride` is a **pixel** stride (librga's convention, not a byte
stride) and `hstride` a row count. Both default to width/height, and both are
routinely larger than that in practice:

- an NPU input tensor's `w_stride` is aligned up (read it via
  `Engine::inputWidthStride`, which `engineInputView` does for you);
- a VPU-decoded frame's `hor_stride`/`ver_stride` are aligned up from the display
  size — reading its rows at `width` instead of `hor_stride` is the classic way
  to get a sheared picture.

Always go through `effWStride()` / `effHStride()` / `rowBytes()` / `uvOffset()`
rather than recomputing from `width`.

## 7. Cache discipline

RGA reads and writes DRAM through an IOMMU without snooping the CPU caches.

- A buffer the **CPU wrote** must be flushed before an RGA op reads it.
- A buffer an **RGA op wrote** must be invalidated before the CPU reads it.
- A buffer that only ever moves between hardware units (VPU → RGA → NPU) needs
  **neither**, and paying for it per frame is exactly the cost worth avoiding.

The RGA wrappers therefore do not touch caches for fd-backed views — the caller
owns that, via `DmaBuf::syncStart()/syncEnd()` or the free functions
`dmaBufSyncStart(fd, ...)` / `dmaBufSyncEnd(fd, ...)`. The one exception is the
CPU fill fallback in §3, which flushes what it wrote before the blit that
follows. The RKNN runtime flushes its own I/O tensors around `rknn_run`.

## 8. Format map

| `rcdl::PixelFormat` | RGA format | Notes |
|---|---|---|
| `RGB888` | `RK_FORMAT_RGB_888` | what an `--input-order rgb` model wants |
| `BGR888` | `RK_FORMAT_BGR_888` | what `cv::imread` gives you |
| `RGBA8888` / `BGRA8888` | `RK_FORMAT_RGBA_8888` / `RK_FORMAT_BGRA_8888` | |
| `GRAY8` | `RK_FORMAT_YCbCr_400` | 8-bit luma; **not** `RK_FORMAT_Y4`, which is 4 bits. RGA2 only — needs both buffers below 4 GB (§3.1) |
| `NV12` | `RK_FORMAT_YCbCr_420_SP` | the VPU's native output |
| `NV21` | `RK_FORMAT_YCrCb_420_SP` | |
| `YUV420P` | `RK_FORMAT_YCbCr_420_P` | |

## 9. Reproducing these numbers

```bash
python - <<'PY'
import numpy as np, rcdl
img = ...                                   # HxWx3 uint8 BGR
hw, lb, _ = rcdl.letterbox(img, 640, 640, src_fmt="bgr888", dst_fmt="rgb888", backend="rga")
sw, _,  _ = rcdl.letterbox(img, 640, 640, src_fmt="bgr888", dst_fmt="rgb888", backend="cpu")
d = np.abs(hw.astype(np.int16) - sw.astype(np.int16))
print(rcdl.rga_version(), d.max(), d.mean())
PY
```

`pytest tests/test_letterbox.py -s` prints the same comparison across scales and
content types, and skips cleanly on a board without RGA.
