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
| Scale factor | 1/16 .. 16 | rejected → CPU |
| Minimum source for a scaled op | 68 × 2 | rejected → CPU |
| Maximum dimension | 8192 | rejected → CPU |
| YUV row stride | 16-byte aligned | rejected → CPU |
| YUV width/height | even | rejected → CPU |

`rcdl::Image::alloc()` and `strideAlign()` apply the 16-pixel YUV alignment for
you, so buffers RCDL allocates are always acceptable; a buffer from somewhere
else may not be.

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

## 3. Colour fill does not work on this board

**Symptom.** Every `imfill` fails, regardless of pixel format (RGB888, BGR888,
RGBA8888, BGRA8888, RGB565) and regardless of whether the buffer came from
`malloc` or a dma-heap:

```
IM_STATUS 0: Failed to call RockChipRga interface
```

**Cause**, from `dmesg`:

```
rga: RGA_MMU unsupported memory larger than 4G!
rga: scheduler core[4] unsupported mm_flag[0x0]!
rga: dst channel map job buffer failed!
```

The driver routes colour fill to the **RGA2 core**, and RGA2 has no IOMMU — its
`RGA_MMU` cannot map physical pages above 4 GB. On a 16 GB board essentially
every `system` dma-heap allocation lands above that line, so the fill can never
be mapped. The scale/convert path is unaffected because it runs on an **RGA3**
core, which does have an IOMMU; that path works with both dma-buf fds and plain
virtual addresses.

**What RCDL does about it.** `rgaFill()` tries the hardware once; the first
failure switches the process to a CPU `memset` permanently (retrying per frame
would cost one failed ioctl and a page of kernel log every frame for nothing).
`rgaLetterbox()` also paints **only the border bands** rather than the whole
canvas, and skips the fill entirely when the aspect ratios already match — so
the fallback touches a few hundred KB, not the full tensor, and costs tens of
microseconds.

A board where the hardware fill does work keeps using it; nothing is disabled at
compile time — but the decision is made **once, on a private scratch buffer**,
never by attempting a fill on a real destination. A rejected fill does not leave
its target untouched: measured here, a band that took a failed attempt came back
with 64–192 bytes of pre-fill content still in it, at cache-line granularity, on
most runs.

**Write the border after the blit, not before.** For the same reason: whatever
cache maintenance librga performs on the destination when it imports it for
`improcess` discards CPU writes made beforehand. Filling the bands first left
64-byte runs of stale content in them on 8 runs out of 10; filling them after
the hardware is finished with the buffer is reliable, and the two regions are
disjoint so the order does not affect the picture. Since the destination is
normally the NPU's input tensor reused every frame, getting this wrong feeds the
model the previous frame's pixels in the pad bands — silently, since the boxes
stay plausible.

The proper fix would be a dma-heap backed by memory below 4 GB (a CMA heap).
The image this was measured on has an empty `cma` heap, and RK3588's NPU, RGA3
and VPU all sit behind IOMMUs, so `system` is the right default anyway.

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

`YuvRange::kStudioToFull` (the default) selects BT.601 limited-range on the RGA
path (`IM_YUV_TO_RGB_BT601_LIMIT`) and the expanding matrix on the CPU path.
`YuvRange::kAsIs` treats the source as already full-range — correct for a frame
RCDL itself produced, and what makes an NV12 → RGB → NV12 round-trip
self-consistent.

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
| `GRAY8` | `RK_FORMAT_YCbCr_400` | 8-bit luma; **not** `RK_FORMAT_Y4`, which is 4 bits |
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
