#pragma once

// Internal helpers shared by the MPP-backed media classes. NOT installed — the
// public headers (media/video_codec.h, media/jpeg_codec.h) deliberately keep
// MPP's types out of the API so a consumer never needs rockchip/rk_mpi.h.

#include <cstddef>
#include <cstdint>
#include <string>

#include "rcdl/media/video_codec.h"
#include "rcdl/preproc/image.h"

#if RCDL_HAVE_MPP

#include <rockchip/rk_mpi.h>

namespace rcdl {
namespace mpp {

/// Throw rcdl::Error unless `ret == MPP_OK`, naming the call and the MPP code.
void check(MPP_RET ret, const char* what);
/// Human-readable MPP_RET name ("MPP_ERR_NULL_PTR", ...), or the number.
const char* retName(MPP_RET ret) noexcept;

/// RCDL codec <-> MPP coding type. Throws for a codec MPP has no id for.
MppCodingType codingType(VideoCodec c);

/// RCDL pixel format <-> MPP frame format. `frameFormat` throws for a format
/// MPP cannot produce or consume; `pixelFormat` returns PixelFormat::Unknown
/// for an MPP format RCDL has no name for (10-bit, tiled, ...).
MppFrameFormat frameFormat(PixelFormat f);
PixelFormat pixelFormat(MppFrameFormat f) noexcept;

/// Describe an MppFrame as an RCDL image: the dma-buf fd of its MppBuffer, the
/// display width/height, and the hor_stride / ver_stride the VPU actually wrote
/// with (which are aligned up from the display size — reading rows at `width`
/// instead of `hor_stride` is the classic way to get a sheared picture).
///
/// `data` is left null: mapping the buffer for the CPU costs a page walk that
/// the hardware path never needs. Use `mapFrame()` when the CPU must read it.
ImageView viewOfFrame(MppFrame frame) noexcept;

/// CPU pointer for a frame's buffer (mmap'd by MPP on demand), or nullptr.
std::uint8_t* mapFrame(MppFrame frame) noexcept;

/// Bytes one frame of (format, hor_stride, ver_stride) occupies — what an
/// external buffer group must commit per slot.
std::size_t frameBufferBytes(MppFrameFormat fmt, int hor_stride, int ver_stride) noexcept;

}  // namespace mpp
}  // namespace rcdl

#endif  // RCDL_HAVE_MPP

namespace rcdl {
namespace mpp {
/// Message used by every media entry point when the build has no MPP, so the
/// failure names the cause instead of surfacing as a null dereference.
constexpr const char* kUnavailable =
    "RCDL was built without MPP (RCDL_HAVE_MPP off) — no hardware video or JPEG codec";
}  // namespace mpp
}  // namespace rcdl
