#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "rcdl/core/dma_buf.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Pixel formats RCDL passes between the VPU, RGA and the NPU.
///
/// Byte order follows the RGA / OpenCV convention: `RGB888` is R,G,B in memory
/// (this is what an RKNN model built with `rgb` input order expects), `BGR888`
/// is B,G,R (what `cv::imread` produces). `NV12` is a Y plane followed by an
/// interleaved U,V plane at half resolution — the VPU's native output.
enum class PixelFormat {
  Unknown,
  RGB888,    ///< 3 bytes/pixel, R,G,B
  BGR888,    ///< 3 bytes/pixel, B,G,R
  RGBA8888,  ///< 4 bytes/pixel, R,G,B,A
  BGRA8888,  ///< 4 bytes/pixel, B,G,R,A
  GRAY8,     ///< 1 byte/pixel luma
  NV12,      ///< Y plane + interleaved UV (U first), 4:2:0
  NV21,      ///< Y plane + interleaved VU (V first), 4:2:0
  YUV420P,   ///< Y plane + U plane + V plane, 4:2:0 (I420)
};

const char* formatName(PixelFormat f) noexcept;
/// Bytes per pixel for packed formats (3, 4, 1). 0 for planar YUV.
int bytesPerPixel(PixelFormat f) noexcept;
bool isPlanarYuv(PixelFormat f) noexcept;
/// Total bytes of an image with the given strides (Y plane + chroma for YUV).
std::size_t imageBytes(PixelFormat f, int wstride, int hstride) noexcept;
/// Row stride alignment RGA wants for this format: 16 pixels for YUV (RGA3
/// requires a 16-byte-aligned Y stride), 1 otherwise. `alignUp(w, a)` helper.
int strideAlign(PixelFormat f) noexcept;
inline int alignUp(int v, int a) noexcept { return a > 1 ? ((v + a - 1) / a) * a : v; }

/// Non-owning description of an image buffer, valid for the CPU (`data`) and/or
/// a hardware unit (`fd`). At least one of the two must be set.
///
/// `wstride` is in PIXELS (RGA's convention: `wrapbuffer_*` takes a pixel width
/// stride, not a byte stride) and `hstride` in ROWS. Both default to
/// width/height when left at 0 — always read them through effWStride() /
/// effHStride() rather than the raw fields.
///
/// For NV12/NV21 the chroma plane starts at `wstride * hstride` bytes into the
/// buffer and uses the same `wstride`; for YUV420P the U and V planes follow at
/// half stride. uvOffset() computes it.
///
/// This is a plain descriptor: a `const ImageView&` destination parameter means
/// the DESCRIPTOR is const, not the pixels it points at. Destination parameters
/// come first in every preproc function (`letterbox(dst, src, ...)`).
struct ImageView {
  void* data = nullptr;  ///< CPU mapping, or nullptr when the buffer is fd-only
  int fd = -1;           ///< dma-buf fd, or -1 when the buffer is host-only
  int width = 0;
  int height = 0;
  int wstride = 0;  ///< row stride in PIXELS (0 => width)
  int hstride = 0;  ///< plane height in ROWS  (0 => height)
  PixelFormat format = PixelFormat::Unknown;
  std::size_t size = 0;  ///< allocated bytes (0 => imageBytes(format, wstride, hstride))

  int effWStride() const noexcept { return wstride > 0 ? wstride : width; }
  int effHStride() const noexcept { return hstride > 0 ? hstride : height; }
  std::size_t bytes() const noexcept {
    return size > 0 ? size : imageBytes(format, effWStride(), effHStride());
  }
  /// Byte stride of the primary (Y / packed) plane.
  std::size_t rowBytes() const noexcept {
    const int bpp = bytesPerPixel(format);
    return static_cast<std::size_t>(effWStride()) * (bpp > 0 ? bpp : 1);
  }
  /// Byte offset of the interleaved UV (NV12/NV21) or U (YUV420P) plane.
  std::size_t uvOffset() const noexcept {
    return static_cast<std::size_t>(effWStride()) * effHStride();
  }
  bool valid() const noexcept {
    return format != PixelFormat::Unknown && width > 0 && height > 0 &&
           (data != nullptr || fd >= 0);
  }
  std::uint8_t* bytePtr() const noexcept { return static_cast<std::uint8_t*>(data); }
  /// Human-readable "640x640 RGB888 ws=640 hs=640 fd=7" for error messages.
  std::string describe() const;
};

/// Describe a host (CPU) buffer. Strides default to packed.
ImageView hostView(void* data, int width, int height, PixelFormat format,
                   int wstride = 0, int hstride = 0);

/// Describe an Engine input tensor as an image so RGA can write straight into
/// it. Uses the tensor's own `w_stride` (which may exceed width) and both the
/// virtual address and the dma-buf fd the runtime allocated, so the same view
/// works for the RGA path and the CPU fallback.
///
/// Throws rcdl::Error unless input `i` is a 4-D UINT8 NHWC tensor with 1, 3 or
/// 4 channels; `format` says how to interpret those channels (a model built with
/// `rgb` input order takes PixelFormat::RGB888).
ImageView engineInputView(Engine& engine, int i, PixelFormat format);

/// An image that owns its pixels in a dma-buf, so every unit (NPU, RGA, VPU)
/// can consume it by fd and the CPU by pointer.
///
/// Strides are aligned up to strideAlign(format) unless given explicitly, which
/// is what keeps RGA happy with YUV sources. Move-only, like DmaBuf.
class Image {
 public:
  Image() = default;

  /// Allocate (width, height) in `format` from `heap`. `wstride`/`hstride` <= 0
  /// are derived (wstride = alignUp(width, strideAlign(format)), hstride =
  /// height rounded up to 2 for YUV).
  static Image alloc(int width, int height, PixelFormat format, int wstride = 0,
                     int hstride = 0, DmaBuf::Heap heap = DmaBuf::Heap::System);

  Image(const Image&) = delete;
  Image& operator=(const Image&) = delete;
  Image(Image&&) noexcept = default;
  Image& operator=(Image&&) noexcept = default;

  bool valid() const noexcept { return buf_.valid(); }
  int width() const noexcept { return view_.width; }
  int height() const noexcept { return view_.height; }
  PixelFormat format() const noexcept { return view_.format; }
  int wstride() const noexcept { return view_.effWStride(); }
  int hstride() const noexcept { return view_.effHStride(); }
  std::size_t bytes() const noexcept { return view_.bytes(); }

  /// Descriptor for the preproc / media functions. The CPU pointer is mapped
  /// lazily on the first call.
  ImageView view();
  /// Descriptor without touching the CPU mapping (fd only) — for the pure
  /// hardware path, where mmap'ing would be wasted work.
  ImageView deviceView() const noexcept { return view_; }

  DmaBuf& buffer() noexcept { return buf_; }
  const DmaBuf& buffer() const noexcept { return buf_; }
  int fd() const noexcept { return buf_.fd(); }

  /// Cache maintenance around CPU access to this image (cached heaps).
  void syncStart(bool read = true, bool write = true) const { buf_.syncStart(read, write); }
  void syncEnd(bool read = true, bool write = true) const { buf_.syncEnd(read, write); }

 private:
  DmaBuf buf_;
  ImageView view_;
};

}  // namespace rcdl
