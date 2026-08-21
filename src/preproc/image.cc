#include "rcdl/preproc/image.h"

#include <sstream>
#include <string>

#include "rcdl/backend/engine.h"
#include "rcdl/core/status.h"

namespace rcdl {

namespace {

// Channels a PixelFormat presents to a NHWC tensor. 0 for the planar YUV
// formats, which never describe an Engine input (the NPU takes interleaved
// pixels; the YUV -> RGB step belongs to RGA or the CPU fallback).
int tensorChannels(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::GRAY8: return 1;
    case PixelFormat::RGB888:
    case PixelFormat::BGR888: return 3;
    case PixelFormat::RGBA8888:
    case PixelFormat::BGRA8888: return 4;
    default: return 0;
  }
}

}  // namespace

const char* formatName(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::RGB888: return "RGB888";
    case PixelFormat::BGR888: return "BGR888";
    case PixelFormat::RGBA8888: return "RGBA8888";
    case PixelFormat::BGRA8888: return "BGRA8888";
    case PixelFormat::GRAY8: return "GRAY8";
    case PixelFormat::NV12: return "NV12";
    case PixelFormat::NV21: return "NV21";
    case PixelFormat::YUV420P: return "YUV420P";
    case PixelFormat::Unknown: break;
  }
  return "Unknown";
}

int bytesPerPixel(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::RGB888:
    case PixelFormat::BGR888: return 3;
    case PixelFormat::RGBA8888:
    case PixelFormat::BGRA8888: return 4;
    case PixelFormat::GRAY8: return 1;
    default: return 0;  // planar YUV has no single bytes-per-pixel; Unknown has none
  }
}

bool isPlanarYuv(PixelFormat f) noexcept {
  return f == PixelFormat::NV12 || f == PixelFormat::NV21 || f == PixelFormat::YUV420P;
}

std::size_t imageBytes(PixelFormat f, int wstride, int hstride) noexcept {
  if (wstride <= 0 || hstride <= 0) return 0;
  const std::size_t plane = static_cast<std::size_t>(wstride) * static_cast<std::size_t>(hstride);
  const int bpp = bytesPerPixel(f);
  if (bpp > 0) return plane * static_cast<std::size_t>(bpp);
  if (isPlanarYuv(f)) {
    // 4:2:0: the chroma half-plane is exactly half the luma plane, whether it is
    // one interleaved UV plane (NV12/NV21) or two quarter planes (YUV420P).
    // Image::alloc rounds hstride up to an even number so this division is exact.
    return plane + plane / 2;
  }
  return 0;
}

int strideAlign(PixelFormat f) noexcept {
  // What RGA3 demands of a row stride, measured on the hardware rather than
  // taken from the documentation (which only mentions the YUV case):
  //
  //   BGR888 / RGB888   16 pixels  ("bgr888 width stride should be 16 aligned!")
  //   RGBA / BGRA       4 pixels
  //   NV12 / NV21 / I420  16 pixels on the luma plane
  //   GRAY8             16, for symmetry — RGA refuses it on this board anyway
  //
  // The field is in PIXELS, so for the 1-byte-per-pixel formats 16 pixels is
  // the 16 bytes the hardware wants. Aligning here is what makes every image
  // RCDL allocates acceptable to RGA; a foreign buffer (a cv::Mat over an
  // 810-pixel-wide JPEG, say) will not be, and takes the CPU path.
  switch (f) {
    case PixelFormat::RGBA8888:
    case PixelFormat::BGRA8888:
      return 4;
    case PixelFormat::Unknown:
      return 1;
    default:
      return 16;
  }
}

std::string ImageView::describe() const {
  // Exactly the form documented in the header — error messages elsewhere embed
  // it verbatim. `fd=-1` already says "host-only"; the CPU pointer is not
  // printed because an address means nothing to whoever reads the message.
  std::ostringstream os;
  os << width << "x" << height << " " << formatName(format) << " ws=" << effWStride()
     << " hs=" << effHStride() << " fd=" << fd;
  return os.str();
}

ImageView hostView(void* data, int width, int height, PixelFormat format, int wstride,
                   int hstride) {
  RCDL_REQUIRE(data != nullptr, "hostView: null data pointer");
  RCDL_REQUIRE(width > 0 && height > 0, "hostView: dimensions must be positive");
  RCDL_REQUIRE(format != PixelFormat::Unknown, "hostView: unknown pixel format");
  RCDL_REQUIRE(wstride <= 0 || wstride >= width, "hostView: wstride is smaller than width");
  RCDL_REQUIRE(hstride <= 0 || hstride >= height, "hostView: hstride is smaller than height");

  ImageView v;
  v.data = data;
  v.fd = -1;  // host-only: RGA will import the mapping itself if it is handed one
  v.width = width;
  v.height = height;
  v.wstride = wstride > 0 ? wstride : 0;
  v.hstride = hstride > 0 ? hstride : 0;
  v.format = format;
  v.size = 0;  // derived from the strides by ImageView::bytes()
  return v;
}

ImageView engineInputView(Engine& engine, int i, PixelFormat format) {
  const rknn_tensor_attr& a = engine.inputAttr(i);  // range-checked by the Engine

  std::ostringstream where;
  where << "engineInputView: input " << i << " (" << a.name << ")";
  const std::string prefix = where.str();

  RCDL_REQUIRE(a.n_dims == 4u, (prefix + " is not a 4-D tensor").c_str());
  RCDL_REQUIRE(a.fmt == RKNN_TENSOR_NHWC,
               (prefix + " is not NHWC — an image input must be NHWC for RGA to write it "
                         "row by row").c_str());
  // What matters is the encoding the runtime expects from US (io_attr), not the
  // model's internal one: a quantized model stores INT8 but takes UINT8 image
  // bytes and requantizes on the way in.
  RCDL_REQUIRE(engine.inputType(i) == RKNN_TENSOR_UINT8,
               (prefix + " does not take UINT8 image bytes").c_str());

  const int height = static_cast<int>(a.dims[1]);
  const int width = static_cast<int>(a.dims[2]);
  const int channels = static_cast<int>(a.dims[3]);
  RCDL_REQUIRE(width > 0 && height > 0, (prefix + " has a zero spatial dimension").c_str());
  RCDL_REQUIRE(channels == 1 || channels == 3 || channels == 4,
               (prefix + " has " + std::to_string(channels) +
                " channels; expected 1 (gray), 3 (RGB/BGR) or 4 (RGBA/BGRA)").c_str());

  const int want = tensorChannels(format);
  RCDL_REQUIRE(want != 0, (prefix + ": " + formatName(format) +
                           " is not a packed pixel format an NPU input can hold").c_str());
  RCDL_REQUIRE(want == channels,
               (prefix + ": tensor has " + std::to_string(channels) + " channels but " +
                formatName(format) + " carries " + std::to_string(want)).c_str());

  ImageView v;
  v.data = engine.inputData(i);
  v.fd = engine.inputFd(i);  // -1 when the runtime did not export one; RGA then uses data
  v.width = width;
  v.height = height;
  // The runtime may pad rows (w_stride > width) so each row starts on the
  // alignment the NPU's DMA wants — RGA must honour it or every row after the
  // first lands skewed.
  v.wstride = engine.inputWidthStride(i);
  v.hstride = height;
  v.format = format;
  v.size = engine.inputBytes(i);
  return v;
}

Image Image::alloc(int width, int height, PixelFormat format, int wstride, int hstride,
                   DmaBuf::Heap heap) {
  RCDL_REQUIRE(width > 0 && height > 0, "Image::alloc: dimensions must be positive");
  RCDL_REQUIRE(format != PixelFormat::Unknown, "Image::alloc: unknown pixel format");
  if (isPlanarYuv(format)) {
    // 4:2:0 chroma is subsampled by two on both axes; an odd extent has no
    // well-defined chroma sample and RGA rejects it outright.
    RCDL_REQUIRE(width % 2 == 0 && height % 2 == 0,
                 "Image::alloc: 4:2:0 formats need even width and height");
  }

  const int ws = wstride > 0 ? wstride : alignUp(width, strideAlign(format));
  const int hs = hstride > 0 ? hstride : (isPlanarYuv(format) ? alignUp(height, 2) : height);
  RCDL_REQUIRE(ws >= width, "Image::alloc: wstride is smaller than width");
  RCDL_REQUIRE(hs >= height, "Image::alloc: hstride is smaller than height");
  if (isPlanarYuv(format)) {
    RCDL_REQUIRE(hs % 2 == 0, "Image::alloc: 4:2:0 hstride must be even");
  }

  const std::size_t bytes = imageBytes(format, ws, hs);
  RCDL_REQUIRE(bytes > 0, "Image::alloc: cannot size this format");

  Image img;
  img.buf_ = DmaBuf::alloc(bytes, heap);
  img.view_.data = nullptr;  // mapped lazily by view(); the hardware path never needs it
  img.view_.fd = img.buf_.fd();
  img.view_.width = width;
  img.view_.height = height;
  img.view_.wstride = ws;
  img.view_.hstride = hs;
  img.view_.format = format;
  img.view_.size = bytes;
  return img;
}

ImageView Image::view() {
  RCDL_REQUIRE(buf_.valid(), "Image::view on an empty image");
  // First call mmaps the dma-buf; afterwards this is just a field read. Callers
  // that only feed hardware should use deviceView() and never pay for the map.
  view_.data = buf_.data();
  return view_;
}

}  // namespace rcdl
