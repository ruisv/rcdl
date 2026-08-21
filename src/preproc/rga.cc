#include "rcdl/preproc/rga.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <sys/mman.h>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>

#include "rcdl/core/dma_buf.h"
#include "rcdl/core/status.h"
#include "rcdl/preproc/letterbox_cpu.h"

#if RCDL_HAVE_RGA
// im2d's public headers define imcheck()/wrapbuffer_*()/imStrError() as
// statement-expression macros that call memset() and printf() unqualified, so
// the C headers they need must be in scope. The _t() entry points used below
// avoid the macros, but a caller of this header might not.
#include <stdio.h>
#include <string.h>

#include <rga/im2d.h>
#endif

namespace rcdl {

namespace {

#if RCDL_HAVE_RGA
constexpr const char* kUnavailable =
    "RGA is unavailable: librga is linked but the driver answered no version query "
    "(is /dev/rga present and readable?)";
#else
constexpr const char* kUnavailable =
    "RCDL was built without librga (RCDL_HAVE_RGA off) — use the CPU preproc path";
#endif

}  // namespace

#if RCDL_HAVE_RGA

namespace {

// im2d reports success as IM_STATUS_SUCCESS (1) or IM_STATUS_NOERROR (2) and
// failure as a negative IM_ERROR_* — the inverse of RCDL_CHECK's "0 is fine"
// convention, so every im2d call goes through this instead.
bool imOk(IM_STATUS status) noexcept {
  return status == IM_STATUS_SUCCESS || status == IM_STATUS_NOERROR;
}

void checkIm(IM_STATUS status, const char* what, const ImageView* dst = nullptr,
             const ImageView* src = nullptr) {
  if (imOk(status)) return;
  std::ostringstream os;
  os << "RCDL: RGA " << what << " failed: " << imStrError_t(status) << " (IM_STATUS "
     << static_cast<int>(status) << ")";
  if (src != nullptr) os << "\n  src " << src->describe();
  if (dst != nullptr) os << "\n  dst " << dst->describe();
  throw Error(static_cast<int>(status), os.str());
}

void requireRga() { RCDL_REQUIRE(rgaAvailable(), kUnavailable); }

// Does this format sit on the YUV side of a colour-space conversion? GRAY8 maps
// to RK_FORMAT_YCbCr_400, i.e. a luma-only YUV surface, so RGA applies the same
// matrix when converting it to or from RGB.
bool isYuvSide(PixelFormat f) noexcept { return isPlanarYuv(f) || f == PixelFormat::GRAY8; }

// Describe an ImageView to RGA. A dma-buf fd is the fast path: the driver maps
// the pages through the IOMMU directly. A bare virtual address makes librga
// import the userspace mapping first, which costs a page-table walk per call.
rga_buffer_t wrap(const ImageView& v, const char* which) {
  RCDL_REQUIRE(v.valid(),
               (std::string("RGA: ") + which + " view is not usable: " + v.describe()).c_str());
  const int fmt = toRgaFormat(v.format);
  RCDL_REQUIRE(fmt >= 0, (std::string("RGA: ") + which + " format " + formatName(v.format) +
                          " has no RK_FORMAT_* equivalent").c_str());
  if (v.fd >= 0) {
    return wrapbuffer_fd_t(v.fd, v.width, v.height, v.effWStride(), v.effHStride(), fmt);
  }
  return wrapbuffer_virtualaddr_t(v.data, v.width, v.height, v.effWStride(), v.effHStride(), fmt);
}

// RGA's colour-space matrix for the conversion (src -> dst) implies, or
// IM_COLOR_SPACE_DEFAULT when both sides live in the same space and no matrix
// is applied at all.
int cscMode(const ImageView& src, const ImageView& dst, YuvRange range) noexcept {
  const bool src_yuv = isYuvSide(src.format);
  const bool dst_yuv = isYuvSide(dst.format);
  if (src_yuv == dst_yuv) return IM_COLOR_SPACE_DEFAULT;
  if (src_yuv) {
    // A decoded frame carries BT.601 studio swing (Y in [16,235]); _LIMIT is the
    // matrix that expands it to full-range RGB, which is what the models are
    // calibrated on. _FULL treats the levels as already full-range.
    return (range == YuvRange::kStudioToFull) ? IM_YUV_TO_RGB_BT601_LIMIT
                                              : IM_YUV_TO_RGB_BT601_FULL;
  }
  // Going the other way, kStudioToFull means "produce what a video encoder
  // expects", i.e. compress full-range RGB into studio-swing YUV.
  return (range == YuvRange::kStudioToFull) ? IM_RGB_TO_YUV_BT601_LIMIT
                                            : IM_RGB_TO_YUV_BT601_FULL;
}

// The mode is carried on the buffers rather than as a call argument for
// improcess(). Which channel librga reads it from depends on the direction, and
// the two fields are combined, so writing the same value on both is both safe
// and direction-independent.
void applyCsc(rga_buffer_t* s, rga_buffer_t* d, int mode) {
  if (mode == IM_COLOR_SPACE_DEFAULT) return;
  imsetColorSpace(s, static_cast<IM_COLOR_SPACE_MODE>(mode));
  imsetColorSpace(d, static_cast<IM_COLOR_SPACE_MODE>(mode));
}

// imcheck_t is the same validation improcess() runs internally; running it up
// front turns "the blit returned -2" into a message naming the offending view.
void checkPair(const rga_buffer_t& s, const rga_buffer_t& d, const im_rect& srect,
               const im_rect& drect, const ImageView& src, const ImageView& dst,
               const char* what) {
  rga_buffer_t pat{};
  im_rect prect{};
  checkIm(imcheck_t(s, d, pat, srect, drect, prect, 0), what, &dst, &src);
}

// One synchronous crop + scale + colour convert. `drect` may be a sub-rectangle
// of the destination (that is how the letterbox writes inside its border).
void process(rga_buffer_t s, rga_buffer_t d, const im_rect& srect, const im_rect& drect,
             const ImageView& src, const ImageView& dst, const char* what) {
  rga_buffer_t pat{};
  im_rect prect{};
  checkIm(improcess(s, d, pat, srect, drect, prect, /*acquire_fence_fd=*/-1,
                    /*release_fence_fd=*/nullptr, /*opt_ptr=*/nullptr, IM_SYNC),
          what, &dst, &src);
}

// Intersect (x,y,w,h) with the destination extent. Returns false when nothing
// is left to draw.
bool clipRect(const ImageView& dst, int x, int y, int w, int h, im_rect* out) noexcept {
  const int x0 = std::max(x, 0);
  const int y0 = std::max(y, 0);
  const int x1 = std::min(x + w, dst.width);
  const int y1 = std::min(y + h, dst.height);
  if (x1 <= x0 || y1 <= y0) return false;
  out->x = x0;
  out->y = y0;
  out->width = x1 - x0;
  out->height = y1 - y0;
  return true;
}

// RGA3's own scaling limit. im2d's documented range is [1/16, 16], but that
// upper half belongs to RGA2; RGA3 does [1/8, 8]. On this board only RGA3 is
// usable (see the fill comment below), so this is the real limit.
constexpr double kRga3MaxScale = 8.0;

// --- colour fill ---------------------------------------------------------------
//
// imcheck_t validates a src -> dst pair; a fill has no source channel, so the
// check that matters is the one improcess() runs internally on the dst (it is
// on unless imconfig(IM_CONFIG_CHECK, ...) turned it off).
//
// RK3588 routes im2d's colour fill to the RGA2 core, which has no IOMMU: its
// RGA_MMU cannot map pages above 4 GB, so on a board with more memory than that
// every fill fails with "job buffer map failed" no matter the pixel format or
// whether the buffer came from malloc or a dma-heap. (The scale/convert path is
// unaffected — that runs on an RGA3 core, which does have an IOMMU.)
//
// So the first failure switches this process to the CPU fill permanently: the
// alternative is one failed ioctl plus a page of driver log per frame, and the
// rectangles involved are letterbox borders and overlay outlines, which the CPU
// covers in tens of microseconds. A board where the hardware fill does work
// keeps using it.
// Whether the hardware fill works, decided ONCE on a private scratch buffer.
//
// The obvious design — try imfill on the real destination and fall back on
// failure — is wrong, and measurably so. A failed fill does not leave the
// destination untouched: letterboxing a 1280x720 frame into a 640x640 NPU input
// tensor, the band that took the failed attempt came back with 64, 128 or 192
// bytes of PRE-FILL content still in it, at cache-line granularity, on 8 runs
// out of 10 — while the second band, filled after the attempt had been given
// up on, was always correct. Whatever the driver does while tearing down the
// rejected job disturbs the cache state of the buffer it was pointed at.
//
// Since the destination here is normally the NPU's input tensor, reused every
// frame, that is a stale band fed to the model — a silent accuracy bug that
// only shows up as slightly-wrong results. So the probe runs against a scratch
// dma-buf at first use and the real destination is only ever handed a fill that
// is already known to work.
//
// (On RK3588 with more than 4 GB of RAM the probe always fails: the driver
// routes colour fill to the RGA2 core, which has no IOMMU and cannot map pages
// above 4 GB. See docs/RGA.md.)
bool hwFillUsable() noexcept {
  static const bool ok = []() noexcept {
    try {
      constexpr int kW = 64, kH = 64;
      DmaBuf scratch = DmaBuf::alloc(static_cast<std::size_t>(kW) * kH * 4);
      rga_buffer_t d = wrapbuffer_fd_t(scratch.fd(), kW, kH, kW, kH, RK_FORMAT_RGBA_8888);
      const im_rect r{0, 0, kW, kH};
      return imOk(imfill_t(d, r, 0, /*sync=*/1));
    } catch (...) {
      return false;
    }
  }();
  return ok;
}

// Try the hardware fill. False means it is unusable on this board, decided by
// the scratch probe above rather than by damaging a real destination.
bool tryHwFill(rga_buffer_t d, const im_rect& rect, std::uint32_t abgr) noexcept {
  if (!hwFillUsable()) return false;
  return imOk(imfill_t(d, rect, static_cast<int>(abgr), /*sync=*/1));
}

// im2d packs a fill colour as ABGR (0xAABBGGRR), so R is the LOW byte.
struct FillColor {
  std::uint8_t r, g, b, a;
};

FillColor unpackAbgr(std::uint32_t abgr) noexcept {
  return FillColor{static_cast<std::uint8_t>(abgr & 0xFFu),
                   static_cast<std::uint8_t>((abgr >> 8) & 0xFFu),
                   static_cast<std::uint8_t>((abgr >> 16) & 0xFFu),
                   static_cast<std::uint8_t>((abgr >> 24) & 0xFFu)};
}

std::uint32_t greyAbgr(std::uint8_t v) noexcept {
  // R == G == B, so the channel order does not matter for a grey.
  return 0xFF000000u | (static_cast<std::uint32_t>(v) << 16) |
         (static_cast<std::uint32_t>(v) << 8) | static_cast<std::uint32_t>(v);
}

std::uint8_t clampU8(float v) noexcept {
  const long i = std::lround(v);
  return static_cast<std::uint8_t>(i < 0 ? 0 : (i > 255 ? 255 : i));
}

// Colour-correct CPU fill of `rect` in `dst`. This is the fallback for the
// public rgaFill() / rgaDrawRect(), which take an arbitrary colour and are the
// overlay path — collapsing that colour to a single grey level is not an option
// there, and on this board the hardware fill never runs at all.
//
// It duplicates a little of letterbox_cpu.cc's plane arithmetic on purpose:
// that file's fillRectCpu() is the single-level (grey Y + neutral chroma) fill
// the letterbox border wants and stays the reference for it, so the border
// comes out byte-identical on both backends.
void fillRectCpuColor(const ImageView& dst, const im_rect& rect, std::uint32_t abgr) {
  RCDL_REQUIRE(dst.data != nullptr, "CPU colour fill: destination has no CPU mapping");
  const FillColor c = unpackAbgr(abgr);
  const bool yuv = isPlanarYuv(dst.format);

  int x0 = std::max(0, rect.x);
  int y0 = std::max(0, rect.y);
  int x1 = std::min(dst.width, rect.x + rect.width);
  int y1 = std::min(dst.height, rect.y + rect.height);
  if (yuv) {
    // Snap OUTWARD to even bounds: a 2x2 chroma sample must be wholly inside or
    // wholly outside the filled region, never split. Same rule as the CPU path.
    x0 &= ~1;
    y0 &= ~1;
    x1 = std::min(dst.width, (x1 + 1) & ~1);
    y1 = std::min(dst.height, (y1 + 1) & ~1);
  }
  if (x1 <= x0 || y1 <= y0) return;

  std::uint8_t* base = dst.bytePtr();
  const std::size_t stride = dst.rowBytes();  // packed-pixel / luma row bytes

  if (!yuv) {
    // One pixel as bytes, in the destination's own channel order.
    std::uint8_t px[4] = {0, 0, 0, 0};
    switch (dst.format) {
      case PixelFormat::RGB888:
        px[0] = c.r; px[1] = c.g; px[2] = c.b;
        break;
      case PixelFormat::BGR888:
        px[0] = c.b; px[1] = c.g; px[2] = c.r;
        break;
      case PixelFormat::RGBA8888:
        px[0] = c.r; px[1] = c.g; px[2] = c.b; px[3] = c.a;
        break;
      case PixelFormat::BGRA8888:
        px[0] = c.b; px[1] = c.g; px[2] = c.r; px[3] = c.a;
        break;
      case PixelFormat::GRAY8:
        px[0] = clampU8(0.299f * c.r + 0.587f * c.g + 0.114f * c.b);
        break;
      default:
        RCDL_REQUIRE(false, "CPU colour fill: unsupported destination format");
    }
    const int bpp = bytesPerPixel(dst.format);
    bool uniform = true;
    for (int i = 1; i < bpp; ++i) uniform = uniform && px[i] == px[0];
    const std::size_t run = static_cast<std::size_t>(x1 - x0) * static_cast<std::size_t>(bpp);
    for (int y = y0; y < y1; ++y) {
      std::uint8_t* row =
          base + static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x0) * bpp;
      if (uniform) {
        std::memset(row, px[0], run);
      } else {
        for (std::size_t o = 0; o < run; o += static_cast<std::size_t>(bpp)) {
          std::memcpy(row + o, px, static_cast<std::size_t>(bpp));
        }
      }
    }
    return;
  }

  // 4:2:0. Full-range BT.601 — the same matrix letterbox_cpu.cc uses in the
  // RGB -> YUV direction, so an overlay drawn by either backend is the same
  // colour, and a grey (R == G == B) still lands on neutral 128 chroma.
  const float r = static_cast<float>(c.r);
  const float g = static_cast<float>(c.g);
  const float b = static_cast<float>(c.b);
  const std::uint8_t yv = clampU8(0.299f * r + 0.587f * g + 0.114f * b);
  const std::uint8_t cb = clampU8(-0.169f * r - 0.331f * g + 0.500f * b + 128.0f);
  const std::uint8_t cr = clampU8(0.500f * r - 0.419f * g - 0.081f * b + 128.0f);

  const std::size_t run = static_cast<std::size_t>(x1 - x0);
  for (int y = y0; y < y1; ++y) {
    std::memset(base + static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x0), yv,
                run);
  }
  std::uint8_t* uv = base + dst.uvOffset();
  const int cy0 = y0 / 2, cy1 = y1 / 2;
  if (dst.format == PixelFormat::YUV420P) {
    // Two separate planes, each at half stride and half height.
    const std::size_t cstride = stride / 2;
    const std::size_t plane = cstride * static_cast<std::size_t>(dst.effHStride() / 2);
    const std::size_t crun = run / 2;
    for (int cy = cy0; cy < cy1; ++cy) {
      const std::size_t off =
          static_cast<std::size_t>(cy) * cstride + static_cast<std::size_t>(x0) / 2;
      std::memset(uv + off, cb, crun);
      std::memset(uv + plane + off, cr, crun);
    }
    return;
  }
  // NV12 / NV21: one interleaved plane at the luma row stride, half the rows.
  // NV12 carries Cb first, NV21 Cr.
  const std::uint8_t first = dst.format == PixelFormat::NV12 ? cb : cr;
  const std::uint8_t second = dst.format == PixelFormat::NV12 ? cr : cb;
  for (int cy = cy0; cy < cy1; ++cy) {
    std::uint8_t* row =
        uv + static_cast<std::size_t>(cy) * stride + static_cast<std::size_t>(x0);
    for (std::size_t o = 0; o + 1 < run; o += 2) {
      row[o] = first;
      row[o + 1] = second;
    }
  }
}

// A CPU view of the destination for the fill fallback.
//
// The zero-copy case has no CPU mapping at all: an `rcdl::Image` handed around
// as `deviceView()`, or a decoded frame, is an fd and nothing else — which is
// the point, since mapping it would cost a page walk the hardware path never
// needs. But RGA's colour fill does not work on this board (RGA2 has no IOMMU;
// see the comment on tryHwFill), so the border has to be painted by the CPU,
// and the CPU needs an address.
//
// So map it here, for the duration of the fill, and unmap after. That is a few
// tens of microseconds on the border bands, and it only happens on the fallback
// path — an Engine input tensor already carries the runtime's virtual address,
// so the detection hot path never takes it.
class CpuFillView {
 public:
  explicit CpuFillView(const ImageView& dst) : view_(dst) {
    if (view_.data != nullptr) return;
    RCDL_REQUIRE(dst.fd >= 0,
                 "RGA colour fill is unavailable on this board (RGA2 has no IOMMU) and the "
                 "destination has neither a CPU mapping nor a dma-buf fd to map");
    bytes_ = dst.bytes();
    map_ = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, dst.fd, 0);
    RCDL_REQUIRE(map_ != MAP_FAILED, "RGA fill fallback: could not mmap the destination dma-buf");
    view_.data = map_;
  }
  ~CpuFillView() {
    if (map_ != nullptr && map_ != MAP_FAILED) ::munmap(map_, bytes_);
  }
  CpuFillView(const CpuFillView&) = delete;
  CpuFillView& operator=(const CpuFillView&) = delete;
  const ImageView& get() const noexcept { return view_; }

 private:
  ImageView view_;
  void* map_ = nullptr;
  std::size_t bytes_ = 0;
};

// Fill an already-clipped rectangle with an arbitrary ABGR colour.
void fillRect(rga_buffer_t d, const im_rect& rect, std::uint32_t abgr, const ImageView& dst) {
  if (tryHwFill(d, rect, abgr)) return;
  CpuFillView cpu(dst);
  // The CPU writes; flush those lines to DRAM before RGA reads or writes the
  // same buffer, or the blit that follows could be overwritten at the seam.
  dmaBufSyncStart(dst.fd, false, true);
  fillRectCpuColor(cpu.get(), rect, abgr);
  dmaBufSyncEnd(dst.fd, false, true);
}

// Same, for the letterbox border: one grey level with neutral chroma. The
// fallback goes through letterbox_cpu.cc's fillRectCpu() so the border is
// byte-identical to the one letterboxCpu() paints.
void fillRectGrey(rga_buffer_t d, const im_rect& rect, std::uint8_t value, const ImageView& dst) {
  if (tryHwFill(d, rect, greyAbgr(value))) return;
  CpuFillView cpu(dst);
  dmaBufSyncStart(dst.fd, false, true);
  fillRectCpu(cpu.get(), rect.x, rect.y, rect.width, rect.height, value);
  dmaBufSyncEnd(dst.fd, false, true);
}

}  // namespace

bool rgaAvailable() noexcept {
  // One driver probe for the process. querystring() opens /dev/rga on its first
  // call, so a missing device or a denied permission shows up here rather than
  // as a mystery -2 from the first blit of the first frame.
  static const bool ok = []() noexcept {
    try {
      const char* v = querystring(RGA_VERSION);
      return v != nullptr && v[0] != '\0';
    } catch (...) {
      return false;
    }
  }();
  return ok;
}

std::string rgaVersion() {
  if (!rgaAvailable()) return "";
  const char* v = querystring(RGA_VERSION);
  return v != nullptr ? std::string(v) : std::string();
}

int toRgaFormat(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::RGB888: return RK_FORMAT_RGB_888;
    case PixelFormat::BGR888: return RK_FORMAT_BGR_888;
    case PixelFormat::RGBA8888: return RK_FORMAT_RGBA_8888;
    case PixelFormat::BGRA8888: return RK_FORMAT_BGRA_8888;
    // Y-only 8-bit. NOT RK_FORMAT_Y4, which is 4 bits per sample.
    case PixelFormat::GRAY8: return RK_FORMAT_YCbCr_400;
    case PixelFormat::NV12: return RK_FORMAT_YCbCr_420_SP;  // Y + interleaved Cb,Cr
    case PixelFormat::NV21: return RK_FORMAT_YCrCb_420_SP;  // Y + interleaved Cr,Cb
    case PixelFormat::YUV420P: return RK_FORMAT_YCbCr_420_P;
    case PixelFormat::Unknown: break;
  }
  return -1;
}

bool rgaCanHandle(const ImageView& dst, const ImageView& src, std::string* why) noexcept {
  try {
    if (!rgaAvailable()) {
      if (why != nullptr) *why = kUnavailable;
      return false;
    }
    if (!src.valid() || !dst.valid()) {
      if (why != nullptr) *why = "src or dst view is not usable";
      return false;
    }
    if (toRgaFormat(src.format) < 0 || toRgaFormat(dst.format) < 0) {
      if (why != nullptr) {
        *why = std::string("no RK_FORMAT_* for ") + formatName(src.format) + " -> " +
               formatName(dst.format);
      }
      return false;
    }
    // GRAY8 (RK_FORMAT_YCbCr_400) fails at SUBMIT in every direction on this
    // board — as a source, as a destination, and gray-to-gray — with the same
    // "job buffer map failed" the colour fill gives, i.e. the driver routes it
    // to the IOMMU-less RGA2 core. imcheck does not know that, so reject it
    // here rather than pay a failed ioctl and a page of kernel log per frame.
    if (src.format == PixelFormat::GRAY8 || dst.format == PixelFormat::GRAY8) {
      if (why != nullptr) {
        *why = "RGA cannot handle GRAY8 on this board (routed to RGA2, which "
               "cannot address memory above 4 GB)";
      }
      return false;
    }
    // Row strides: measured requirements, tighter than the documented YUV-only
    // rule. See strideAlign() in preproc/image.cc for the table.
    for (const auto* v : {&src, &dst}) {
      const int a = strideAlign(v->format);
      if (a > 1 && v->effWStride() % a != 0) {
        if (why != nullptr) {
          *why = std::string(formatName(v->format)) + " width stride " +
                 std::to_string(v->effWStride()) + " is not " + std::to_string(a) + "-aligned";
        }
        return false;
      }
    }
    // imcheck accepts the [1/16, 16] range because RGA2 covers it — but on a
    // board with more than 4 GB of RAM RGA2 is unusable (no IOMMU; its RGA_MMU
    // cannot map pages above 4 GB, which is where the system dma-heap allocates)
    // and the driver silently routes any ratio outside RGA3's own [1/8, 8] to
    // it. The op then fails at submit with "job buffer map failed". Rejecting
    // the ratio here keeps that off the steady-state path — one check instead of
    // a failed ioctl and a page of kernel log every frame. See docs/RGA.md.
    const double sx = static_cast<double>(dst.width) / src.width;
    const double sy = static_cast<double>(dst.height) / src.height;
    const double lo = std::min(sx, sy), hi = std::max(sx, sy);
    if (lo < 1.0 / kRga3MaxScale || hi > kRga3MaxScale) {
      if (why != nullptr) {
        *why = "scale " + std::to_string(lo) + ".." + std::to_string(hi) +
               " is outside the RGA3 range [1/8, 8]; wider ratios need RGA2, "
               "which cannot address memory above 4 GB on this board";
      }
      return false;
    }
    const rga_buffer_t s = wrap(src, "src");
    const rga_buffer_t d = wrap(dst, "dst");
    rga_buffer_t pat{};
    // Zeroed rectangles mean "the whole image" to im2d. That checks the scale
    // ratio against the full destination extent; a letterbox writes into a
    // slightly smaller rectangle, so this is marginally optimistic at the
    // extremes of the range — the op itself still checks exactly.
    im_rect srect{};
    im_rect drect{};
    im_rect prect{};
    const IM_STATUS st = imcheck_t(s, d, pat, srect, drect, prect, 0);
    if (imOk(st)) return true;
    if (why != nullptr) *why = imStrError_t(st);
    return false;
  } catch (const std::exception& e) {
    if (why != nullptr) *why = e.what();
    return false;
  } catch (...) {
    if (why != nullptr) *why = "unknown error while checking RGA support";
    return false;
  }
}

LetterboxInfo rgaLetterbox(const ImageView& dst, const ImageView& src, std::uint8_t pad,
                           YuvRange range) {
  requireRga();
  LetterboxInfo lb = computeLetterbox(src.width, src.height, dst.width, dst.height);

  // The hardware only writes integer rectangles. Round the scaled extent first,
  // then centre THAT — deriving the padding from the rounded extent rather than
  // rounding the float padding is what keeps this identical to letterboxCpu(),
  // which does the four steps below in exactly this order. `scale` stays the
  // float min-ratio on both paths.
  const bool dst_yuv = isPlanarYuv(dst.format);
  int newW = std::min(static_cast<int>(std::lround(src.width * lb.scale)), dst.width);
  int newH = std::min(static_cast<int>(std::lround(src.height * lb.scale)), dst.height);
  if (dst_yuv) {
    // A 4:2:0 destination shares one chroma sample per 2x2 luma block, so an odd
    // extent or offset would split a sample across the border seam — improcess
    // rejects such a rectangle outright, where the CPU path would have produced
    // a frame. Masking down keeps padX + newW inside the canvas.
    newW &= ~1;
    newH &= ~1;
  }
  int padX = (dst.width - newW) / 2;
  int padY = (dst.height - newH) / 2;
  if (dst_yuv) {
    padX &= ~1;
    padY &= ~1;
  }
  RCDL_REQUIRE(padX >= 0 && padY >= 0 && padX + newW <= dst.width && padY + newH <= dst.height,
               "rgaLetterbox: scaled image does not fit the canvas");
  // Report the integers the hardware was actually given: post-processing's
  // inverse map has to land on the pixels that reached the NPU.
  lb.padX = static_cast<float>(padX);
  lb.padY = static_cast<float>(padY);

  rga_buffer_t s = wrap(src, "src");
  rga_buffer_t d = wrap(dst, "dst");
  const im_rect srect{0, 0, src.width, src.height};
  const im_rect drect{padX, padY, newW, newH};
  if (newW <= 0 || newH <= 0) {
    // The source scales below one pixel (or below two on a 4:2:0 destination):
    // there is no rectangle to blit, only the border. letterboxCpu() returns the
    // same all-pad canvas and the same geometry.
    fillRectGrey(d, im_rect{0, 0, dst.width, dst.height}, pad, dst);
    return lb;
  }
  checkPair(s, d, srect, drect, src, dst, "letterbox");

  // 1. Border — only the bands the blit will not cover. An aspect-matched source
  //    pads on neither axis and skips the fill entirely; the usual
  //    16:9-into-square case costs two bands instead of a whole canvas.
  //
  //    Each band is tested on ITS OWN extent, never on `pad > 0`: when the
  //    leftover is a single pixel the leading pad rounds down to 0 and only the
  //    trailing band exists, and skipping it would leave that row (or column)
  //    holding whatever was there before — which, since the destination is
  //    normally the NPU input tensor reused every frame, means the PREVIOUS
  //    frame's pixels.
  //
  //    The fill is issued before the colour-space mode is set on the buffers, so
  //    a YUV->RGB matrix meant for the source cannot also touch the border. On a
  //    board where the hardware fill works, a YUV destination is at the mercy of
  //    whether RGA runs the fill colour through its RGB->YUV matrix; the CPU
  //    fallback (which is what runs here) always writes Y=pad with neutral 128
  //    chroma, exactly like letterboxCpu().
  const int right = padX + newW;
  const int bottom = padY + newH;

  // 2. Crop, scale and colour-convert into the centered rectangle in one pass.
  //    This runs BEFORE the border, not after. The border and the blit target
  //    disjoint rectangles, so the order does not matter to the picture — but it
  //    matters to the cache. When the CPU wrote the border first, the band came
  //    back with 64-, 128- or 192-byte runs of pre-fill content (cache-line
  //    granularity) on most runs: whatever cache maintenance librga performs on
  //    the destination when it imports it for the blit discards CPU writes made
  //    beforehand. Writing the border last, after the hardware is finished with
  //    the buffer, is reliable — and since the destination is normally the NPU's
  //    input tensor reused every frame, the alternative is a stale band fed to
  //    the model, which shows up as slightly-wrong results rather than an error.
  applyCsc(&s, &d, cscMode(src, dst, range));
  process(s, d, srect, drect, src, dst, "letterbox blit");

  // 3. Border, into the bands the blit did not cover.
  if (padY > 0) {
    fillRectGrey(d, im_rect{0, 0, dst.width, padY}, pad, dst);
  }
  if (bottom < dst.height) {
    fillRectGrey(d, im_rect{0, bottom, dst.width, dst.height - bottom}, pad, dst);
  }
  if (padX > 0) {
    fillRectGrey(d, im_rect{0, padY, padX, newH}, pad, dst);
  }
  if (right < dst.width) {
    fillRectGrey(d, im_rect{right, padY, dst.width - right, newH}, pad, dst);
  }
  return lb;
}

LetterboxInfo rgaResize(const ImageView& dst, const ImageView& src, YuvRange range) {
  requireRga();
  RCDL_REQUIRE(src.valid() && dst.valid(), "rgaResize: src or dst view is not usable");

  rga_buffer_t s = wrap(src, "src");
  rga_buffer_t d = wrap(dst, "dst");
  const im_rect srect{0, 0, src.width, src.height};
  const im_rect drect{0, 0, dst.width, dst.height};
  checkPair(s, d, srect, drect, src, dst, "resize");
  applyCsc(&s, &d, cscMode(src, dst, range));
  process(s, d, srect, drect, src, dst, "resize");

  LetterboxInfo lb;
  lb.srcW = src.width;
  lb.srcH = src.height;
  lb.dstW = dst.width;
  lb.dstH = dst.height;
  // A stretch has two scales; LetterboxInfo carries one. Reporting the X scale
  // keeps the inverse map exact whenever the aspect ratios match (the only case
  // where a single-scale inverse can be exact at all).
  lb.scale = static_cast<float>(dst.width) / static_cast<float>(src.width);
  lb.padX = 0.0f;
  lb.padY = 0.0f;
  return lb;
}

void rgaCvtColor(const ImageView& dst, const ImageView& src, YuvRange range) {
  requireRga();
  RCDL_REQUIRE(src.width == dst.width && src.height == dst.height,
               "rgaCvtColor: src and dst must have the same width and height");

  rga_buffer_t s = wrap(src, "src");
  rga_buffer_t d = wrap(dst, "dst");
  const im_rect srect{0, 0, src.width, src.height};
  const im_rect drect{0, 0, dst.width, dst.height};
  checkPair(s, d, srect, drect, src, dst, "cvtColor");
  // imcvtcolor takes the matrix as an argument rather than off the buffers.
  checkIm(imcvtcolor(s, d, s.format, d.format, cscMode(src, dst, range), /*sync=*/1), "cvtColor",
          &dst, &src);
}

void rgaCropResize(const ImageView& dst, const ImageView& src, int x, int y, int w, int h,
                   YuvRange range) {
  requireRga();
  RCDL_REQUIRE(w > 0 && h > 0, "rgaCropResize: crop rectangle is empty");
  RCDL_REQUIRE(x >= 0 && y >= 0 && x + w <= src.width && y + h <= src.height,
               "rgaCropResize: crop rectangle is outside the source image");
  RCDL_REQUIRE(!isPlanarYuv(src.format) || (x % 2 == 0 && y % 2 == 0 && w % 2 == 0 && h % 2 == 0),
               "rgaCropResize: a 4:2:0 source needs an even crop origin and extent — a 2x2 "
               "chroma sample cannot be split");

  rga_buffer_t s = wrap(src, "src");
  rga_buffer_t d = wrap(dst, "dst");
  const im_rect srect{x, y, w, h};
  const im_rect drect{0, 0, dst.width, dst.height};
  checkPair(s, d, srect, drect, src, dst, "cropResize");
  applyCsc(&s, &d, cscMode(src, dst, range));
  process(s, d, srect, drect, src, dst, "cropResize");
}

void rgaCopy(const ImageView& dst, const ImageView& src) {
  requireRga();
  RCDL_REQUIRE(src.width == dst.width && src.height == dst.height,
               "rgaCopy: src and dst must have the same size");
  RCDL_REQUIRE(src.format == dst.format,
               "rgaCopy: src and dst must have the same format (use rgaCvtColor to convert)");

  const rga_buffer_t s = wrap(src, "src");
  const rga_buffer_t d = wrap(dst, "dst");
  const im_rect srect{0, 0, src.width, src.height};
  const im_rect drect{0, 0, dst.width, dst.height};
  checkPair(s, d, srect, drect, src, dst, "copy");
  checkIm(imcopy(s, d, /*sync=*/1), "copy", &dst, &src);
}

void rgaFill(const ImageView& dst, int x, int y, int w, int h, std::uint32_t abgr) {
  requireRga();
  im_rect rect{};
  if (!clipRect(dst, x, y, w, h, &rect)) return;  // nothing of it lands on the canvas
  const rga_buffer_t d = wrap(dst, "dst");
  fillRect(d, rect, abgr, dst);
}

void rgaDrawRect(const ImageView& dst, int x, int y, int w, int h, std::uint32_t abgr,
                 int thickness) {
  requireRga();
  if (thickness <= 0 || w <= 0 || h <= 0) return;

  // A border thicker than half the box would draw its two opposite edges over
  // each other; clamp so the four fills stay disjoint.
  const int t = std::min(thickness, std::min(w, h) / 2 > 0 ? std::min(w, h) / 2 : 1);
  const int inner_h = h - 2 * t;  // may be <= 0 for a very thin box; clipped away below

  rgaFill(dst, x, y, w, t, abgr);                    // top
  rgaFill(dst, x, y + h - t, w, t, abgr);            // bottom
  if (inner_h > 0) {
    rgaFill(dst, x, y + t, t, inner_h, abgr);        // left
    rgaFill(dst, x + w - t, y + t, t, inner_h, abgr);  // right
  }
}

#else  // ---------------------------------------------------------------------
// No librga: the declarations still exist so callers compile, and every op
// fails loudly with the same message rgaCanHandle() hands back. preproc's
// PreprocBackend::Auto asks rgaCanHandle() first and never gets here.

namespace {
[[noreturn]] void noRga() { throw Error(-1, std::string("RCDL: ") + kUnavailable); }
}  // namespace

bool rgaAvailable() noexcept { return false; }

std::string rgaVersion() { return ""; }

int toRgaFormat(PixelFormat) noexcept { return -1; }

bool rgaCanHandle(const ImageView&, const ImageView&, std::string* why) noexcept {
  if (why != nullptr) *why = kUnavailable;
  return false;
}

LetterboxInfo rgaLetterbox(const ImageView&, const ImageView&, std::uint8_t, YuvRange) { noRga(); }

LetterboxInfo rgaResize(const ImageView&, const ImageView&, YuvRange) { noRga(); }

void rgaCvtColor(const ImageView&, const ImageView&, YuvRange) { noRga(); }

void rgaCropResize(const ImageView&, const ImageView&, int, int, int, int, YuvRange) { noRga(); }

void rgaCopy(const ImageView&, const ImageView&) { noRga(); }

void rgaFill(const ImageView&, int, int, int, int, std::uint32_t) { noRga(); }

void rgaDrawRect(const ImageView&, int, int, int, int, std::uint32_t, int) { noRga(); }

#endif  // RCDL_HAVE_RGA

}  // namespace rcdl
