#pragma once

#include <cstdint>
#include <string>

#include "rcdl/preproc/geometry.h"
#include "rcdl/preproc/image.h"

namespace rcdl {

/// RGA (Raster Graphic Acceleration) — Rockchip's 2-D engine, reached through
/// librga's im2d API. This is where EVERY resize / colour-space conversion /
/// letterbox in RCDL is supposed to happen: the CPU paths in letterbox_cpu.h are
/// a guarded fallback, not the default.
///
/// RK3588 has two RGA3 cores plus one RGA2 core; the driver picks. The RGA3
/// constraints that shape the calls below (checked by imcheck before every op,
/// and by rgaCanHandle() when you want to decide without throwing):
///   - scaling factor within [1/16, 16]
///   - source and destination at least 68x2 for a scaled op
///   - YUV row stride 16-byte aligned, even width/height
///   - dimensions up to 8192
/// Anything outside that must go to the CPU fallback — hence the `Auto` backend
/// in preproc/letterbox.h, which asks rgaCanHandle() first.
///
/// All wrappers are SYNCHRONOUS (im2d `sync = 1`): the call returns after the
/// hardware is done, so the destination is immediately readable by the next
/// stage. Buffers are addressed by dma-buf fd when the ImageView has one (true
/// zero-copy: VPU frame -> RGA -> NPU input tensor) and by virtual address
/// otherwise (librga then does its own import, which costs a page-table walk).
///
/// Cache discipline: RGA reads and writes DRAM through an IOMMU without
/// snooping the CPU caches, so a buffer the CPU wrote must be flushed before an
/// RGA op reads it and invalidated after an RGA op wrote it. When an ImageView
/// carries a dma-buf fd these wrappers do NOT touch its cache — the caller owns
/// that (DmaBuf::syncStart / syncEnd), because in the hardware-only path there
/// is nothing to flush and paying for it every frame is the whole cost. When an
/// ImageView is host-only (fd < 0), librga's own import handles coherency.

/// Is librga present and usable (built with RCDL_HAVE_RGA and the driver
/// responds to a version query)? Cached after the first call; never throws.
bool rgaAvailable() noexcept;

/// librga version + RGA hardware version string, or "" when unavailable.
std::string rgaVersion();

/// The RK_FORMAT_* value for a PixelFormat, or -1 when RGA has no equivalent.
int toRgaFormat(PixelFormat f) noexcept;

/// Would RGA accept this (src -> dst) pair? Runs the same imcheck the ops run,
/// without throwing; `why` receives librga's explanation when it says no.
/// Returns false (with a reason) when RGA is not available at all.
bool rgaCanHandle(const ImageView& dst, const ImageView& src, std::string* why = nullptr) noexcept;

/// Aspect-preserving letterbox of `src` into the pre-allocated `dst`, in ONE
/// `improcess` call that crops, scales, converts colour space and fills the
/// border. This is the M1 hot path: an NV12 frame straight from the VPU becomes
/// the RGB888 contents of the NPU's input tensor with no CPU touch and no copy.
///
/// The destination is first filled with `pad` (a single `imfill` of the whole
/// canvas with grey `pad,pad,pad`), then the scaled image is written into the
/// centered destination rectangle. Both the rectangle and the returned geometry
/// use INTEGER pixel bounds — computeLetterbox()'s float geometry is rounded to
/// what the hardware actually did, so the inverse map matches the pixels.
///
/// `range` selects RGA's YUV->RGB matrix when the conversion happens:
/// kStudioToFull uses BT.601 limited-range (what a video decoder emits),
/// kAsIs uses full-range. Ignored when neither side is YUV.
///
/// Throws rcdl::Error when RGA is unavailable or imcheck rejects the pair — use
/// rcdl::letterbox() (preproc/letterbox.h) for the fallback-aware version.
LetterboxInfo rgaLetterbox(const ImageView& dst, const ImageView& src, std::uint8_t pad = 114,
                           YuvRange range = YuvRange::kStudioToFull);

/// Stretch `src` to fill `dst` (NO aspect preservation, no padding), converting
/// the colour format if they differ. Returns the geometry with the X scale and
/// zero padding; its uniform-scale inverse is only exact when the aspect ratios
/// match. Prefer rgaLetterbox() for detection.
LetterboxInfo rgaResize(const ImageView& dst, const ImageView& src,
                        YuvRange range = YuvRange::kStudioToFull);

/// Colour-space conversion only; `dst` and `src` must have the same width and
/// height (e.g. NV12 1920x1080 -> RGB888 1920x1080).
void rgaCvtColor(const ImageView& dst, const ImageView& src,
                 YuvRange range = YuvRange::kStudioToFull);

/// Copy the `(x, y, w, h)` rectangle of `src` into `dst`, scaling it to dst's
/// full extent (crop + resize in one op). `dst` may differ in format.
void rgaCropResize(const ImageView& dst, const ImageView& src, int x, int y, int w, int h,
                   YuvRange range = YuvRange::kStudioToFull);

/// Straight blit — same size, same format, honouring strides.
void rgaCopy(const ImageView& dst, const ImageView& src);

/// Fill the `(x, y, w, h)` rectangle of `dst` with an ABGR-packed colour
/// (0xAABBGGRR, the im2d convention). Used by the overlay stage in M3.
void rgaFill(const ImageView& dst, int x, int y, int w, int h, std::uint32_t abgr);

/// Draw a hollow rectangle outline `thickness` pixels wide, as four fills.
/// Clipped to `dst`; a rectangle entirely outside it is a no-op.
void rgaDrawRect(const ImageView& dst, int x, int y, int w, int h, std::uint32_t abgr,
                 int thickness = 2);

}  // namespace rcdl
