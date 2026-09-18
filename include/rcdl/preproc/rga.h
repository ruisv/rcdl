#pragma once

#include <cstdint>
#include <string>

#include "rcdl/preproc/geometry.h"
#include "rcdl/preproc/image.h"
#include "rcdl/preproc/letterbox.h"

namespace rcdl {

/// RGA (Raster Graphic Acceleration) — Rockchip's 2-D engine, reached through
/// librga's im2d API. This is where EVERY resize / colour-space conversion /
/// letterbox in RCDL is supposed to happen: the CPU paths in letterbox_cpu.h are
/// a guarded fallback, not the default.
///
/// RK3588 has two RGA3 cores plus one RGA2 core, and the wrappers below pick
/// the core themselves instead of letting the driver load-balance: the two
/// generations resample differently (a scaled frame comes out with most of its
/// bytes different, see docs/RGA.md), so a pipeline whose frames bounced
/// between them would not be reproducible. Resize / convert / letterbox go to
/// the RGA3 cores; colour fill and the formats or ratios only RGA2 has go to
/// RGA2. The RGA3 constraints that shape the calls (checked by imcheck before
/// every op, and by rgaCanHandle() when you want to decide without throwing):
///   - scaling factor within [1/8, 8]
///   - source and destination at least 68x2 for a scaled op
///   - YUV row stride 16-byte aligned, even width/height
///   - dimensions up to 8176
///
/// RGA2 has a 32-bit MMU: it can only touch memory below 4 GB physical. On a
/// board with more RAM than that, an op that needs RGA2 runs only when BOTH
/// buffers are known to be below the line (ImageView::below4g, i.e. allocated
/// from the `system-dma32` dma-heap); otherwise rgaCanHandle() says no and the
/// `Auto` backend in preproc/letterbox.h takes the CPU path.
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
///
/// `yuv` is the colour space the op would be given. RGA converts YUV -> RGB in
/// BT.601 limited / full and BT.709 limited, and RGB -> YUV in BT.601 only:
/// librga has no BT.709 full-range mode, and no core on RK3588 accepts an
/// RGB -> YUV BT.709 job. Anything else that crosses between RGB and YUV is a
/// "no" here, and so is an RGA2-only op (GRAY8, a scale ratio beyond 8x) on
/// buffers RGA2 cannot reach — see the header comment.
bool rgaCanHandle(const ImageView& dst, const ImageView& src, std::string* why = nullptr,
                  YuvColorSpace yuv = {}) noexcept;

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
/// `yuv` selects RGA's colour-space mode when a YUV <-> RGB conversion happens:
/// its range picks limited (kStudioToFull, what a video decoder emits) or full
/// (kAsIs), its matrix BT.601 or BT.709. Ignored when neither side is YUV.
///
/// Throws rcdl::Error when RGA is unavailable, imcheck rejects the pair, or the
/// colour space is one RGA cannot convert (see rgaCanHandle()) — use
/// rcdl::letterbox() (preproc/letterbox.h) for the fallback-aware version.
LetterboxInfo rgaLetterbox(const ImageView& dst, const ImageView& src, std::uint8_t pad = 114,
                           YuvColorSpace yuv = {});

/// Stretch `src` to fill `dst` (NO aspect preservation, no padding), converting
/// the colour format if they differ. Returns the geometry with the X scale and
/// zero padding; its uniform-scale inverse is only exact when the aspect ratios
/// match. Prefer rgaLetterbox() for detection.
LetterboxInfo rgaResize(const ImageView& dst, const ImageView& src, YuvColorSpace yuv = {});

/// Colour-space conversion only; `dst` and `src` must have the same width and
/// height (e.g. NV12 1920x1080 -> RGB888 1920x1080).
void rgaCvtColor(const ImageView& dst, const ImageView& src, YuvColorSpace yuv = {});

/// Copy the `(x, y, w, h)` rectangle of `src` into `dst`, scaling it to dst's
/// full extent (crop + resize in one op). `dst` may differ in format.
void rgaCropResize(const ImageView& dst, const ImageView& src, int x, int y, int w, int h,
                   YuvColorSpace yuv = {});

/// Straight blit — same size, same format, honouring strides.
void rgaCopy(const ImageView& dst, const ImageView& src);

/// Can the hardware colour fill reach `dst`? Colour fill is an RGA2-only
/// feature, and RGA2 can only address memory below 4 GB: the answer is yes
/// for a buffer flagged `below4g` (or any buffer on a board with at most
/// 4 GB), decided once per heap on a private scratch buffer, never by trying
/// a fill on a real destination. When this is false the fill ops below write
/// with the CPU instead.
bool rgaHwFillUsable(const ImageView& dst) noexcept;

/// Fill the `(x, y, w, h)` rectangle of `dst` with an ABGR-packed colour
/// (0xAABBGGRR, the im2d convention). Hardware when rgaHwFillUsable(dst), CPU
/// otherwise. On a 4:2:0 destination the colour is converted with the BT.601
/// studio-range matrix, which is what the hardware fill does too.
void rgaFill(const ImageView& dst, int x, int y, int w, int h, std::uint32_t abgr);

/// One rectangle outline for rgaDrawRects(). `abgr` is 0xAABBGGRR.
struct RectSpec {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  std::uint32_t abgr = 0xFFFFFFFFu;
  int thickness = 2;
};

/// Draw hollow rectangle outlines — the box overlay. Each is clipped to `dst`;
/// on a 4:2:0 destination edges and thickness are snapped outward to even
/// pixels, because a 2x2 chroma sample cannot be split.
///
/// `Auto` draws with the CPU: the frame is mapped once, the cache is
/// synchronised once, and every box is painted in that window. Measured on
/// RK3588 that is the fast path — the hardware needs four fill jobs per box
/// on the single RGA2 core, about half a millisecond each, so twenty boxes
/// cost over 13 ms there against well under a millisecond on the CPU. `Rga`
/// forces the hardware (one submit per colour, on RGA2) and throws when the
/// destination is not one it can reach (see rgaHwFillUsable()); it exists for
/// a pipeline that would rather spend RGA2 time than CPU time. `used` reports
/// which one ran.
///
/// Writing an overlay with the CPU is safe where a CPU letterbox border was
/// not (docs/RGA.md §3.1): nothing else writes the frame after the decoder,
/// and the sync window flushes the CPU's lines before the encoder reads them.
void rgaDrawRects(const ImageView& dst, const RectSpec* rects, std::size_t count,
                  PreprocBackend backend = PreprocBackend::Auto, PreprocBackend* used = nullptr);

/// One outline; see rgaDrawRects().
void rgaDrawRect(const ImageView& dst, int x, int y, int w, int h, std::uint32_t abgr,
                 int thickness = 2, PreprocBackend backend = PreprocBackend::Auto);

}  // namespace rcdl
