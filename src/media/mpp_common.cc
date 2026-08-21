// MODULE_TAG names this translation unit in MPP's own log lines ("mpp_buffer:
// rcdl_media ..."), which is how a commit / group failure gets attributed to us
// rather than to whatever library called MPP last. rk_type.h defaults it to
// NULL behind an #ifndef, so it has to be defined BEFORE any MPP header — i.e.
// before mpp_common.h, which pulls in rockchip/rk_mpi.h.
#ifndef MODULE_TAG
#define MODULE_TAG "rcdl_media"
#endif

#include "mpp_common.h"

#include <cstdio>
#include <string>

#include "rcdl/core/status.h"

#if RCDL_HAVE_MPP

namespace rcdl {
namespace mpp {

const char* retName(MPP_RET ret) noexcept {
  switch (ret) {
    // MPP_SUCCESS shares MPP_OK's value, so it must not appear as a second case.
    case MPP_OK: return "MPP_OK";
    case MPP_NOK: return "MPP_NOK";
    case MPP_ERR_UNKNOW: return "MPP_ERR_UNKNOW";
    case MPP_ERR_NULL_PTR: return "MPP_ERR_NULL_PTR";
    case MPP_ERR_MALLOC: return "MPP_ERR_MALLOC";
    case MPP_ERR_OPEN_FILE: return "MPP_ERR_OPEN_FILE";
    case MPP_ERR_VALUE: return "MPP_ERR_VALUE";
    case MPP_ERR_READ_BIT: return "MPP_ERR_READ_BIT";
    case MPP_ERR_TIMEOUT: return "MPP_ERR_TIMEOUT";
    case MPP_ERR_PERM: return "MPP_ERR_PERM";
    case MPP_ERR_BASE: return "MPP_ERR_BASE";
    case MPP_ERR_LIST_STREAM: return "MPP_ERR_LIST_STREAM";
    case MPP_ERR_INIT: return "MPP_ERR_INIT";
    case MPP_ERR_VPU_CODEC_INIT: return "MPP_ERR_VPU_CODEC_INIT";
    case MPP_ERR_STREAM: return "MPP_ERR_STREAM";
    case MPP_ERR_FATAL_THREAD: return "MPP_ERR_FATAL_THREAD";
    case MPP_ERR_NOMEM: return "MPP_ERR_NOMEM";
    case MPP_ERR_PROTOL: return "MPP_ERR_PROTOL";
    case MPP_FAIL_SPLIT_FRAME: return "MPP_FAIL_SPLIT_FRAME";
    case MPP_ERR_VPUHW: return "MPP_ERR_VPUHW";
    case MPP_EOS_STREAM_REACHED: return "MPP_EOS_STREAM_REACHED";
    case MPP_ERR_BUFFER_FULL: return "MPP_ERR_BUFFER_FULL";
    case MPP_ERR_DISPLAY_FULL: return "MPP_ERR_DISPLAY_FULL";
  }
  // Not an MPP_RET we know: several rk_mpi entry points document "0 and
  // positive for success", so an unnamed value is not necessarily a failure.
  // thread_local because two decode threads may format a code concurrently.
  static thread_local char buf[32];
  std::snprintf(buf, sizeof(buf), "MPP_RET(%d)", static_cast<int>(ret));
  return buf;
}

void check(MPP_RET ret, const char* what) {
  if (ret == MPP_OK) return;
  std::string msg = std::string("RCDL: ") + what + " failed: " + retName(ret) + " (" +
                    std::to_string(static_cast<int>(ret)) + ")";
  throw Error(static_cast<int>(ret), msg);
}

MppCodingType codingType(VideoCodec c) {
  switch (c) {
    case VideoCodec::H264: return MPP_VIDEO_CodingAVC;
    case VideoCodec::H265: return MPP_VIDEO_CodingHEVC;
    case VideoCodec::VP8: return MPP_VIDEO_CodingVP8;
    case VideoCodec::VP9: return MPP_VIDEO_CodingVP9;
    case VideoCodec::AV1: return MPP_VIDEO_CodingAV1;
    case VideoCodec::MJPEG: return MPP_VIDEO_CodingMJPEG;
  }
  throw Error(-1, std::string("RCDL: no MPP coding type for codec ") +
                      std::to_string(static_cast<int>(c)));
}

MppFrameFormat frameFormat(PixelFormat f) {
  switch (f) {
    // The VPU's native 4:2:0 output. NV21 is the same layout with the chroma
    // pair swapped, which MPP does in the post-processor, not for free.
    case PixelFormat::NV12: return MPP_FMT_YUV420SP;
    case PixelFormat::NV21: return MPP_FMT_YUV420SP_VU;
    case PixelFormat::YUV420P: return MPP_FMT_YUV420P;
    // Monochrome: MPP models "luma only" as 4:0:0, which is byte-identical to
    // an 8-bit gray image.
    case PixelFormat::GRAY8: return MPP_FMT_YUV400;
    // Packed RGB is only reachable on the encoder / JPEG side (the video
    // decoder never emits it). MPP names these after the byte order in memory,
    // same convention as RCDL and RGA.
    case PixelFormat::RGB888: return MPP_FMT_RGB888;
    case PixelFormat::BGR888: return MPP_FMT_BGR888;
    case PixelFormat::RGBA8888: return MPP_FMT_RGBA8888;
    case PixelFormat::BGRA8888: return MPP_FMT_BGRA8888;
    case PixelFormat::Unknown: break;
  }
  throw Error(-1, std::string("RCDL: MPP has no frame format for PixelFormat ") +
                      formatName(f));
}

PixelFormat pixelFormat(MppFrameFormat f) noexcept {
  // The property bits (FBC version, tile, HDR, endianness) live above the
  // format value. A compressed (AFBC/RKFBC) or tiled buffer is NOT a plain
  // planar image — RGA cannot letterbox it through the normal path and the CPU
  // certainly cannot read it — so report Unknown rather than lie about layout.
  if (MPP_FRAME_FMT_IS_FBC(f) || MPP_FRAME_FMT_IS_TILE(f)) return PixelFormat::Unknown;

  switch (static_cast<int>(f) & MPP_FRAME_FMT_MASK) {
    case MPP_FMT_YUV420SP: return PixelFormat::NV12;
    case MPP_FMT_YUV420SP_VU: return PixelFormat::NV21;
    case MPP_FMT_YUV420P: return PixelFormat::YUV420P;
    case MPP_FMT_YUV400: return PixelFormat::GRAY8;
    case MPP_FMT_RGB888: return PixelFormat::RGB888;
    case MPP_FMT_BGR888: return PixelFormat::BGR888;
    case MPP_FMT_RGBA8888: return PixelFormat::RGBA8888;
    case MPP_FMT_BGRA8888: return PixelFormat::BGRA8888;
    default: break;
  }
  // 10-bit 4:2:0, 4:2:2, 4:4:4, packed YUYV, ... — real MPP formats RCDL has no
  // PixelFormat for. Unknown makes ImageView::valid() false, so a caller cannot
  // hand one to RGA by accident.
  return PixelFormat::Unknown;
}

std::size_t frameBufferBytes(MppFrameFormat fmt, int hor_stride, int ver_stride) noexcept {
  if (hor_stride <= 0 || ver_stride <= 0) return 0;
  const std::size_t plane =
      static_cast<std::size_t>(hor_stride) * static_cast<std::size_t>(ver_stride);

  // hor_stride is the BYTE stride of the first plane, so the per-format factor
  // below only has to account for the extra chroma planes. That is why the
  // packed formats (YUYV, RGB, 10-bit semi-planar) come out as 1x: their extra
  // bytes per pixel are already inside hor_stride.
  if (MPP_FRAME_FMT_IS_FBC(fmt)) {
    // Frame Buffer Compression stores a header block plus a payload that is at
    // worst the uncompressed size. Rockchip's own decoders size an AFBC
    // allocation at 2x the luma plane, which covers header + 4:2:0 payload.
    return plane * 2;
  }

  switch (static_cast<int>(fmt) & MPP_FRAME_FMT_MASK) {
    // 4:2:0 — one interleaved (or two quarter-size) chroma planes = half of luma.
    case MPP_FMT_YUV420SP:
    case MPP_FMT_YUV420SP_10BIT:
    case MPP_FMT_YUV420SP_VU:
    case MPP_FMT_YUV420P:
    case MPP_FMT_YUV411SP:
      return plane + plane / 2;
    // 4:2:2 semi-planar / planar — chroma is a full extra plane's worth.
    case MPP_FMT_YUV422SP:
    case MPP_FMT_YUV422SP_10BIT:
    case MPP_FMT_YUV422SP_VU:
    case MPP_FMT_YUV422P:
    case MPP_FMT_YUV440SP:
      return plane * 2;
    // 4:4:4 — two more full planes.
    case MPP_FMT_YUV444SP:
    case MPP_FMT_YUV444SP_10BIT:
    case MPP_FMT_YUV444P:
      return plane * 3;
    // Luma only, packed YUV 4:2:2, and every RGB format: one plane.
    case MPP_FMT_YUV400:
    case MPP_FMT_YUV422_YUYV:
    case MPP_FMT_YUV422_YVYU:
    case MPP_FMT_YUV422_UYVY:
    case MPP_FMT_YUV422_VYUY:
      return plane;
    default: break;
  }
  if (MPP_FRAME_FMT_IS_RGB(fmt)) return plane;
  // Unknown layout: 4:4:4 is the largest non-FBC frame MPP produces, so sizing
  // for it is the safe guess — an under-sized committed buffer is a hardware
  // overrun, an over-sized one only wastes memory.
  return plane * 3;
}

ImageView viewOfFrame(MppFrame frame) noexcept {
  ImageView v;
  if (frame == nullptr) return v;

  v.width = static_cast<int>(mpp_frame_get_width(frame));
  v.height = static_cast<int>(mpp_frame_get_height(frame));
  v.format = pixelFormat(mpp_frame_get_fmt(frame));

  // hor_stride is a BYTE stride; ImageView::wstride is in PIXELS (RGA's
  // convention). They coincide for 8-bit YUV — one byte of luma per pixel —
  // but not for packed RGB, so divide there. MPP also exposes the pixel stride
  // directly on newer frames; prefer it when the decoder filled it in.
  const int hor_stride = static_cast<int>(mpp_frame_get_hor_stride(frame));
  const int bpp = bytesPerPixel(v.format);
  if (bpp > 1) {
    const int pixel_stride = static_cast<int>(mpp_frame_get_hor_stride_pixel(frame));
    v.wstride = pixel_stride > 0 ? pixel_stride : hor_stride / bpp;
  } else {
    v.wstride = hor_stride;
  }
  v.hstride = static_cast<int>(mpp_frame_get_ver_stride(frame));

  // The buffer is the whole point: its fd is what RGA imports and what makes
  // the decode -> letterbox -> NPU path copy-free. A frame can legitimately
  // arrive without one (the empty end-of-stream marker), hence the null check.
  MppBuffer buf = mpp_frame_get_buffer(frame);
  if (buf != nullptr) {
    v.fd = mpp_buffer_get_fd(buf);
    v.size = mpp_buffer_get_size(buf);
  } else {
    v.size = mpp_frame_get_buf_size(frame);
  }
  // data stays null on purpose — see the header. mapFrame() pays for the mmap
  // only when the CPU actually has to look at the pixels.
  return v;
}

std::uint8_t* mapFrame(MppFrame frame) noexcept {
  if (frame == nullptr) return nullptr;
  MppBuffer buf = mpp_frame_get_buffer(frame);
  if (buf == nullptr) return nullptr;
  // MPP mmaps the dma-buf on the first get_ptr and caches the mapping in the
  // MppBuffer, so repeated calls on the same frame are cheap.
  return static_cast<std::uint8_t*>(mpp_buffer_get_ptr(buf));
}

}  // namespace mpp
}  // namespace rcdl

#endif  // RCDL_HAVE_MPP
