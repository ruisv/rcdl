#pragma once

#include <cstdint>

#include "rcdl/preproc/geometry.h"
#include "rcdl/preproc/image.h"

namespace rcdl {

/// Pure-CPU preprocessing — the guarded fallback for when RGA is unavailable or
/// refuses a request (see rgaCanHandle(): scale outside [1/16,16], a source
/// below 68x2, an unaligned YUV stride, an exotic format).
///
/// These are NOT the default path. RGA does the same work while the CPU is free
/// to run post-processing; these exist so a build without librga, an RK356x
/// without an RGA3, or an out-of-range request still produces the same pixels.
///
/// Geometry is computed with computeLetterbox() exactly like the RGA path and
/// then rounded the same way, so the LetterboxInfo (and therefore the inverse
/// map post-processing uses) is the same contract either way.
///
/// Sampling uses the OpenCV pixel-center convention
///     srcX = (dstX - padX + 0.5) / scale - 0.5
///     srcY = (dstY - padY + 0.5) / scale - 0.5
/// (clamped to the valid source range), so output matches cv::resize with
/// INTER_LINEAR and the numpy reference in tests/. OpenMP parallelises the row
/// loop when the build found it.
///
/// Cache discipline: these functions write with the CPU. When `dst` carries a
/// dma-buf fd the caller must have it in a CPU access window
/// (DmaBuf::syncStart/syncEnd) — these wrappers do not sync, because they are
/// also used on plain host buffers and on Engine tensors the runtime flushes
/// itself.

/// Aspect-preserving letterbox of `src` into the pre-allocated `dst`, entirely
/// on the CPU. `dst` is filled with `pad` (and neutral 128 chroma for YUV
/// destinations) and the bilinearly-resampled source is written into the
/// centered rectangle.
///
/// Formats: RGB888 / BGR888 / RGBA8888 / BGRA8888 / GRAY8 / NV12 / NV21 /
/// YUV420P on both sides, in any combination; a differing pair implies the
/// colour conversion cvtColorCpu() documents below (e.g. NV12 src -> RGB888
/// dst, the video path) and allocates one temporary buffer, sized to whichever
/// of the two sides is smaller. The same-format case allocates nothing.
///
/// Requires a CPU pointer on both views (ImageView::data); throws rcdl::Error
/// with the offending view's describe() otherwise.
LetterboxInfo letterboxCpu(const ImageView& dst, const ImageView& src, std::uint8_t pad = 114,
                           YuvColorSpace yuv = {});

/// Stretch-resize `src` into `dst` (no padding, aspect NOT preserved), with an
/// implied colour conversion when the formats differ. Same conventions as above.
LetterboxInfo resizeCpu(const ImageView& dst, const ImageView& src, YuvColorSpace yuv = {});

/// Colour-space conversion at identical width/height.
///
/// Supported: NV12/NV21/YUV420P -> RGB888/BGR888/RGBA8888/BGRA8888/GRAY8,
/// RGB888/BGR888/RGBA8888/BGRA8888 -> NV12/NV21/YUV420P/GRAY8, any YUV -> YUV
/// layout change, and any packed-RGB permutation (channel swap / alpha add or
/// drop).
///
/// COLOUR CONVENTION. `yuv` describes the YUV SIDE of the conversion — its
/// levels (`yuv.range`) and its colour matrix (`yuv.matrix`) — and is not a
/// one-way instruction. Both directions therefore agree with each other, and
/// with the RGA path, which selects its colour-space mode from the same value;
/// an asymmetric reading is how two backends end up writing ~14% different luma
/// for one and the same call. GRAY8 counts as the YUV side: it is a luma plane
/// (RK_FORMAT_YCbCr_400 to RGA).
///
/// Levels:
///  - `kStudioToFull` (default) — the YUV side is studio-swing, as a video
///    decoder produces. YUV -> RGB EXPANDS Y in [16,235] to [0,255]; with the
///    BT.601 matrix that is
///        R = 1.164*(Y-16) + 1.596*(V-128)
///        G = 1.164*(Y-16) - 0.813*(V-128) - 0.391*(U-128)
///        B = 1.164*(Y-16) + 2.018*(U-128)
///    which is exactly `cv::cvtColor(COLOR_YUV2BGR_NV12)`. RGB -> YUV COMPRESSES
///    back into that swing (RGA's IM_RGB_TO_YUV_BT601_LIMIT):
///        Y = 16  + (219/255) * (0.299R + 0.587G + 0.114B)
///        U = 128 + (224/255) * (-0.169R - 0.331G + 0.500B)
///        V = 128 + (224/255) * ( 0.500R - 0.419G - 0.081B)
///  - `kAsIs` — the YUV side is already full-range, so only the colour matrix is
///    applied, in either direction: R = Y + 1.402*(V-128), ... coming out, and
///    Y = 0.299R + 0.587G + 0.114B (cv2's COLOR_BGR2YUV_I420) going in.
///
/// Matrices (the full-range form; kStudioToFull scales it exactly as above):
///  - `kBt601` (default) — Kr = 0.299, Kb = 0.114, the coefficients above.
///  - `kBt709` — Kr = 0.2126, Kb = 0.0722:
///        R = Y + 1.5748*(V-128)
///        G = Y - 0.1873*(U-128) - 0.4681*(V-128)
///        B = Y + 1.8556*(U-128)
///        Y = 0.2126R + 0.7152G + 0.0722B
///        U = 128 - 0.1146R - 0.3854G + 0.5000B
///        V = 128 + 0.5000R - 0.4542G - 0.0458B
///    and studio swing gives 1.164 / 1.793 / 0.213 / 0.533 / 2.112, which is
///    what RGA's IM_YUV_TO_RGB_BT709_LIMIT applies (measured, ±1 LSB).
///
/// Each direction is the inverse of the other for the same `yuv`, so an
/// NV12 -> RGB -> NV12 round-trip is self-consistent under any value as long as
/// the same one is used throughout.
///
/// YUV -> YUV NEVER touches levels or matrix: both sides are the same YUV side,
/// so only the plane layout changes (NV12 <-> NV21 <-> YUV420P, and anything to
/// or from GRAY8 while the other side is YUV too). RGA agrees — with no RGB
/// side it uses IM_COLOR_SPACE_DEFAULT.
///
/// Chroma is subsampled by averaging each 2x2 RGB block before computing U/V.
void cvtColorCpu(const ImageView& dst, const ImageView& src, YuvColorSpace yuv = {});

/// Fill `dst` with a solid grey `value` (and neutral 128 chroma for YUV).
void fillCpu(const ImageView& dst, std::uint8_t value);

/// Fill the `(x, y, w, h)` rectangle of `dst` with a solid grey `value` (and
/// neutral 128 chroma for YUV). Clipped to `dst`; a rectangle entirely outside
/// it is a no-op. For a 4:2:0 destination the rectangle is snapped OUTWARD to
/// even bounds so no chroma sample is split across the edge.
///
/// This is what paints a letterbox border on the RGA path too: RK3588 routes
/// im2d's colour fill to the RGA2 core, which has no IOMMU and cannot reach
/// pages above 4 GB, so on a board with more memory than that the hardware fill
/// fails and this runs instead. It is only the border bands, so the cost is a
/// few tens of microseconds — see rgaLetterbox().
void fillRectCpu(const ImageView& dst, int x, int y, int w, int h, std::uint8_t value);

}  // namespace rcdl
