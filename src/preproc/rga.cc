#include "rcdl/preproc/rga.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <sys/mman.h>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sys/sysinfo.h>

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

// --- which core ------------------------------------------------------------------
//
// RK3588 carries two RGA3 cores and one RGA2 core, and they are not
// interchangeable: RGA3 has a 40-bit IOMMU and does resize / convert / blend;
// RGA2 has a 32-bit MMU (nothing above 4 GB physical), and is the only one with
// colour fill, GRAY8 (YCbCr400), YUV planar and scale ratios beyond 8x. Left to
// itself the driver load-balances any job both cores can do — and the two
// resample differently enough that a 1080p -> 640x360 blit comes out with 78%
// of its bytes different (max 147 LSB) depending on which core drew it, so a
// pipeline whose frames land on either would not be reproducible. Every op
// here therefore pins the core it wants, per thread (imconfig's scheduler
// setting is thread-local, measured), right before it submits.
//
// A board without one of the families (RK356x has RGA2 only) gets whichever it
// has: the mask is derived from the driver's version string once.
constexpr int kMaskRga3 = IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1;
constexpr int kMaskRga2 = IM_SCHEDULER_RGA2_CORE0;

struct CoreMasks {
  int rga3 = 0;  ///< 0 when the board has no RGA3
  int rga2 = 0;  ///< 0 when the board has no RGA2
};

const CoreMasks& coreMasks() noexcept {
  static const CoreMasks m = []() noexcept {
    CoreMasks c;
    try {
      const std::string v = rgaVersion();  // "... RGA version : RGA_2_Enhance RGA_3"
      if (v.find("RGA_3") != std::string::npos) c.rga3 = kMaskRga3;
      if (v.find("RGA_2") != std::string::npos) c.rga2 = kMaskRga2;
    } catch (...) {
    }
    return c;
  }();
  return m;
}

// Pin this thread's next submits to `mask`. A family the board lacks falls
// back to the other one, and a board that reports neither is left to the
// driver (imconfig rejects an empty mask, so it is simply not called).
void pinCore(int mask) noexcept {
  const CoreMasks& m = coreMasks();
  int use = mask;
  if (mask == kMaskRga3 && m.rga3 == 0) use = m.rga2;
  if (mask == kMaskRga2 && m.rga2 == 0) use = m.rga3;
  if (use != 0) imconfig(IM_CONFIG_SCHEDULER_CORE, static_cast<std::uint64_t>(use));
}

// cscMode()'s answer for a conversion RGA cannot perform.
constexpr int kCscUnsupported = -1;

// RGA's colour-space matrix for the conversion (src -> dst) implies,
// IM_COLOR_SPACE_DEFAULT when both sides live in the same space and no matrix
// is applied at all, or kCscUnsupported.
//
// What is supported is measured, not read off im2d_type.h (librga 1.10.4,
// RK3588, against float references):
//   YUV -> RGB  BT.601 limited / full, BT.709 limited   all within ±1 LSB
//   YUV -> RGB  BT.709 full    librga refuses: "Not support full csc mode"
//   RGB -> YUV  BT.601 limited / full                   within ±1 LSB
//   RGB -> YUV  BT.709 (either range)  "no core match" from the driver whether
//               the job is pinned to RGA3 or left to the scheduler; RGA2 with
//               low buffers is untested
int cscMode(const ImageView& src, const ImageView& dst, YuvColorSpace yuv) noexcept {
  const bool src_yuv = isYuvSide(src.format);
  const bool dst_yuv = isYuvSide(dst.format);
  if (src_yuv == dst_yuv) return IM_COLOR_SPACE_DEFAULT;
  const bool studio = yuv.range == YuvRange::kStudioToFull;
  if (src_yuv) {
    // A decoded frame carries studio swing (Y in [16,235]); _LIMIT is the
    // matrix that expands it to full-range RGB, which is what the models are
    // calibrated on. _FULL treats the levels as already full-range.
    if (yuv.matrix == YuvMatrix::kBt709) {
      return studio ? IM_YUV_TO_RGB_BT709_LIMIT : kCscUnsupported;
    }
    return studio ? IM_YUV_TO_RGB_BT601_LIMIT : IM_YUV_TO_RGB_BT601_FULL;
  }
  // Going the other way, kStudioToFull means "produce what a video encoder
  // expects", i.e. compress full-range RGB into studio-swing YUV.
  if (yuv.matrix == YuvMatrix::kBt709) return kCscUnsupported;
  return studio ? IM_RGB_TO_YUV_BT601_LIMIT : IM_RGB_TO_YUV_BT601_FULL;
}

const char* matrixName(YuvMatrix m) noexcept {
  return m == YuvMatrix::kBt709 ? "BT.709" : "BT.601";
}

std::string cscUnsupportedWhy(const ImageView& src, const ImageView& dst, YuvColorSpace yuv) {
  return std::string("RGA cannot convert ") + formatName(src.format) + " -> " +
         formatName(dst.format) + " as " + matrixName(yuv.matrix) +
         (yuv.range == YuvRange::kStudioToFull ? " limited" : " full") +
         " range (supported: YUV -> RGB in BT.601 limited/full and BT.709 limited, "
         "RGB -> YUV in BT.601)";
}

// The mode for an op that is about to run; throws for a combination RGA cannot
// do, which PreprocBackend::Auto turns into the CPU path.
int requireCsc(const ImageView& src, const ImageView& dst, YuvColorSpace yuv) {
  const int mode = cscMode(src, dst, yuv);
  if (mode == kCscUnsupported) {
    throw Error(-1, "RCDL: " + cscUnsupportedWhy(src, dst, yuv));
  }
  return mode;
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

// One synchronous crop + scale + colour convert on the given core. `drect` may
// be a sub-rectangle of the destination (that is how the letterbox writes
// inside its border).
void process(rga_buffer_t s, rga_buffer_t d, const im_rect& srect, const im_rect& drect,
             const ImageView& src, const ImageView& dst, const char* what, int core) {
  rga_buffer_t pat{};
  im_rect prect{};
  pinCore(core);
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

// RGA3's own scaling limit. im2d's documented range is [1/16, 16], but the
// outer half belongs to RGA2; RGA3 does [1/8, 8].
constexpr double kRga3MaxScale = 8.0;

// --- what RGA2 can reach ------------------------------------------------------
//
// RGA2's MMU is 32-bit: it maps nothing above 4 GB physical. Whether that
// matters depends on the board and on the buffer. On a board with at most 4 GB
// every buffer is fine; on a 16 GB board a `system` dma-heap allocation is
// almost always above the line and only a `system-dma32` one (ImageView::
// below4g) is usable. The board-level fact is measured ONCE, per heap, with a
// colour fill on a private 64x64 scratch buffer — never by trying an op on a
// real destination, because a rejected job does not leave its target alone:
// measured here, a letterbox band that took a failed fill came back with
// 64-192 bytes of pre-fill content at cache-line granularity on 8 runs of 10,
// and since that target is normally the NPU's input tensor, that is a stale
// band fed to the model. Colour fill is also RGA2's own feature, so the probe
// answers "can RGA2 fill this heap" and "can RGA2 reach this heap" at once.
bool fillProbe(DmaBuf::Heap heap) noexcept {
  static std::mutex mu;
  static std::map<int, bool> cache;
  std::lock_guard<std::mutex> lock(mu);
  const auto it = cache.find(static_cast<int>(heap));
  if (it != cache.end()) return it->second;
  bool ok = false;
  try {
    constexpr int kW = 64, kH = 64;
    DmaBuf scratch = DmaBuf::alloc(static_cast<std::size_t>(kW) * kH * 4, heap);
    rga_buffer_t d = wrapbuffer_fd_t(scratch.fd(), kW, kH, kW, kH, RK_FORMAT_RGBA_8888);
    const im_rect r{0, 0, kW, kH};
    pinCore(kMaskRga2);
    ok = imOk(imfill_t(d, r, 0, /*sync=*/1));
  } catch (...) {
    ok = false;  // no such heap, or no access to it
  }
  cache.emplace(static_cast<int>(heap), ok);
  return ok;
}

// Can RGA2 address this buffer? Known-low pages, or a board where the ordinary
// heap is reachable — one with no memory above 4 GB at all. A board with more
// is not probed for the ordinary heap: a `system` allocation there may land
// anywhere, so "unreachable" is the only answer that is never wrong, and it
// spares the process the failed job and the page of kernel log the probe
// would cost.
bool rga2Reaches(const ImageView& v) noexcept {
  if (v.below4g) return fillProbe(DmaBuf::Heap::SystemDma32);
  static const bool small_board = []() noexcept {
    struct sysinfo si {};
    if (::sysinfo(&si) != 0) return false;
    const unsigned long long total = static_cast<unsigned long long>(si.totalram) * si.mem_unit;
    return total <= (4ull << 30);
  }();
  return small_board && fillProbe(DmaBuf::Heap::System);
}

// Does this (src -> dst) op need the RGA2 core? `why` names the feature.
bool needsRga2(const ImageView& src, const ImageView& dst, std::string* why) noexcept {
  if (src.format == PixelFormat::GRAY8 || dst.format == PixelFormat::GRAY8) {
    if (why != nullptr) *why = "GRAY8 (YCbCr400) is an RGA2-only format";
    return true;
  }
  const double sx = static_cast<double>(dst.width) / src.width;
  const double sy = static_cast<double>(dst.height) / src.height;
  const double lo = std::min(sx, sy), hi = std::max(sx, sy);
  if (lo < 1.0 / kRga3MaxScale || hi > kRga3MaxScale) {
    if (why != nullptr) {
      *why = "scale " + std::to_string(lo) + ".." + std::to_string(hi) +
             " is outside the RGA3 range [1/8, 8]";
    }
    return true;
  }
  return false;
}

// The core an op runs on, or a reason it cannot run at all (false).
bool coreFor(const ImageView& src, const ImageView& dst, int* core, std::string* why) noexcept {
  std::string feature;
  if (!needsRga2(src, dst, &feature)) {
    *core = kMaskRga3;
    return true;
  }
  *core = kMaskRga2;
  if (rga2Reaches(src) && rga2Reaches(dst)) return true;
  if (why != nullptr) {
    *why = feature + ", and RGA2 cannot address " +
           (rga2Reaches(src) ? "the destination" : rga2Reaches(dst) ? "the source" : "either buffer") +
           " (its MMU is 32-bit; allocate from the system-dma32 heap, see docs/RGA.md)";
  }
  return false;
}

int requireCore(const ImageView& src, const ImageView& dst) {
  int core = 0;
  std::string why;
  if (!coreFor(src, dst, &core, &why)) throw Error(-1, "RCDL: RGA: " + why);
  return core;
}

// Whether the hardware colour fill can be handed `dst`.
bool hwFillUsable(const ImageView& dst) noexcept { return rga2Reaches(dst); }

// --- the border, painted by the hardware --------------------------------------
//
// The CPU must not write into a buffer RGA is blitting into, in either order,
// and that is a measurement rather than a principle. Letterboxing a decoded
// 816x1088 stream into a 640x640 NPU input tensor, with the border painted by
// the CPU (RGA's own colour fill being unusable here — see tryHwFill), the same
// 16-frame clip run three times gave DIFFERENT detections on 7 to 16 of its
// frames: boxes moving about a pixel, scores by ~0.005. The same clip on a
// square canvas, where the letterbox needs no border at all, was identical on
// every frame of every run, and so was the padded clip with the border fill
// disabled. Painting the border before the blit instead of after made it worse
// (16 of 16), which is the other half of the same effect: the blit's cache
// maintenance on the destination discards CPU writes made just before it, and a
// CPU write just after it merges stale bytes back over the seam, because the
// band edge lands mid-cache-line and filling it is a read-modify-write of a
// line the hardware just wrote.
//
// So: no CPU writes. The border is a BLIT from a small grey source, allocated
// once per (format, pad, size) and thereafter read-only. Every write to the
// destination then comes from the same engine, in order. The CPU fill stays as
// the fallback for a destination RGA cannot take.
int greySide(const ImageView& dst) noexcept {
  // RGA3 scales by at most 8x; a quarter of the destination leaves margin, and
  // 68 is the engine's minimum source width.
  const int need = std::max({(dst.width + 3) / 4, (dst.height + 3) / 4, 68});
  return (need + 15) & ~15;  // 16-aligned, which is what packed RGB strides want
}

// One flat grey image, ready for RGA to stretch over a border band. Returns
// nullptr when it cannot be built, which sends the caller to the CPU fallback.
const ImageView* greySource(PixelFormat fmt, std::uint8_t pad, int side) noexcept {
  struct Grey {
    DmaBuf buf;
    ImageView view;
  };
  static std::mutex mu;
  static std::map<std::uint64_t, std::unique_ptr<Grey>> cache;

  const std::uint64_t key = (static_cast<std::uint64_t>(fmt) << 40) |
                            (static_cast<std::uint64_t>(pad) << 32) |
                            static_cast<std::uint64_t>(static_cast<std::uint32_t>(side));
  std::lock_guard<std::mutex> lock(mu);
  const auto it = cache.find(key);
  if (it != cache.end()) return it->second ? &it->second->view : nullptr;

  std::unique_ptr<Grey> g;
  try {
    g = std::make_unique<Grey>();
    g->buf = DmaBuf::alloc(imageBytes(fmt, side, side));
    g->view.data = g->buf.data();
    g->view.fd = g->buf.fd();
    g->view.width = g->view.wstride = side;
    g->view.height = g->view.hstride = side;
    g->view.format = fmt;
    g->view.size = g->buf.size();
    // The one CPU write this buffer ever sees, on a buffer no hardware is
    // touching yet, flushed before anything reads it.
    g->buf.syncStart(/*read=*/false, /*write=*/true);
    fillRectCpu(g->view, 0, 0, side, side, pad);
    g->buf.syncEnd(/*read=*/false, /*write=*/true);
  } catch (...) {
    cache.emplace(key, nullptr);
    return nullptr;
  }
  const ImageView* view = &g->view;
  cache.emplace(key, std::move(g));
  return view;
}

// Stretch the grey source over `rect`. Nearest or bilinear makes no difference
// to a constant image, so the band comes out exactly `pad` — byte-identical to
// what letterboxCpu() paints.
bool tryGreyBlit(const ImageView& dst, const im_rect& rect, std::uint8_t pad) noexcept {
  try {
    const ImageView* grey = greySource(dst.format, pad, greySide(dst));
    if (grey == nullptr) return false;
    int core = 0;
    if (!coreFor(*grey, dst, &core, nullptr)) return false;
    rga_buffer_t s = wrap(*grey, "grey");
    rga_buffer_t d = wrap(dst, "dst");
    const im_rect srect{0, 0, grey->width, grey->height};
    rga_buffer_t pat{};
    im_rect prect{};
    if (!imOk(imcheck_t(s, d, pat, srect, rect, prect, 0))) return false;
    pinCore(core);
    return imOk(improcess(s, d, pat, srect, rect, prect, /*acquire_fence_fd=*/-1,
                          /*release_fence_fd=*/nullptr, /*opt_ptr=*/nullptr, IM_SYNC));
  } catch (...) {
    return false;
  }
}

// Try the hardware fill. False means RGA2 cannot reach this destination,
// decided by the scratch probe above rather than by damaging a real one.
bool tryHwFill(rga_buffer_t d, const im_rect& rect, std::uint32_t abgr,
               const ImageView& dst) noexcept {
  if (!hwFillUsable(dst)) return false;
  pinCore(kMaskRga2);
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

// Colour-correct CPU fill of `rect` in `dst`. This is the CPU side of the
// public rgaFill() / rgaDrawRects(), which take an arbitrary colour and are the
// overlay path — collapsing that colour to a single grey level is not an option
// there.
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

  // 4:2:0. BT.601 STUDIO range, with the classic 8-bit fixed-point
  // coefficients: that is byte-for-byte what the hardware fill writes for the
  // same colour (measured on RK3588: red -> Y=82 Cb=90 Cr=240, green -> Y=144),
  // so a box drawn by either backend is the same bytes — and it is the right
  // range for the frame, which is studio-swing video on its way to an encoder.
  // A grey (R == G == B) still lands on neutral 128 chroma.
  const int r = c.r, g = c.g, b = c.b;
  const std::uint8_t yv = clampU8(static_cast<float>(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16));
  const std::uint8_t cb = clampU8(static_cast<float>(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128));
  const std::uint8_t cr = clampU8(static_cast<float>(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128));

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
// So map it here, ONCE per call (a whole frame's worth of boxes shares the
// mapping), and unmap after. The map costs tens of microseconds; what used to
// cost milliseconds was doing it — and a whole-buffer cache sync — once per
// band. An Engine input tensor already carries the runtime's virtual address,
// so the detection hot path never takes it.
class CpuFillView {
 public:
  explicit CpuFillView(const ImageView& dst) : view_(dst) {
    if (view_.data != nullptr) return;
    RCDL_REQUIRE(dst.fd >= 0,
                 "CPU fill: the destination has neither a CPU mapping nor a dma-buf fd to map");
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
  if (tryHwFill(d, rect, abgr, dst)) return;
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
  if (tryHwFill(d, rect, greyAbgr(value), dst)) return;
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

bool rgaCanHandle(const ImageView& dst, const ImageView& src, std::string* why,
                  YuvColorSpace yuv) noexcept {
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
    // A colour space RGA has no mode for (or one the driver can only run on
    // RGA2) — see cscMode(). imcheck does not look at the mode at all.
    if (cscMode(src, dst, yuv) == kCscUnsupported) {
      if (why != nullptr) *why = cscUnsupportedWhy(src, dst, yuv);
      return false;
    }
    // An RGA2-only op (GRAY8, a ratio beyond 8x) runs only where RGA2 can reach
    // both buffers; imcheck accepts it regardless, and the op would fail at
    // submit with a page of kernel log per frame. See coreFor().
    {
      int core = 0;
      if (!coreFor(src, dst, &core, why)) return false;
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
                           YuvColorSpace yuv) {
  requireRga();
  // Before anything touches the destination: a colour space RGA cannot convert,
  // or a core that cannot reach the buffers, has to fail with the canvas
  // untouched, so the CPU fallback starts clean.
  const int csc = requireCsc(src, dst, yuv);
  const int core = requireCore(src, dst);
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
    const im_rect whole{0, 0, dst.width, dst.height};
    if (!tryGreyBlit(dst, whole, pad)) fillRectGrey(d, whole, pad, dst);
    return lb;
  }
  checkPair(s, d, srect, drect, src, dst, "letterbox");

  // The border is needed unless the source's aspect matches the canvas exactly.
  // Each edge is tested on ITS OWN extent, never on `pad > 0`: when the leftover
  // is a single pixel the leading pad rounds down to 0 and only the trailing
  // band exists, and skipping it would leave that row (or column) holding
  // whatever was there before — which, since the destination is normally the
  // NPU input tensor reused every frame, means the PREVIOUS frame's pixels.
  //
  // Both the border blit and the CPU fallback run before the colour-space mode
  // is set on the buffers, so a YUV->RGB matrix meant for the source cannot also
  // touch the border: the grey source is in the destination's own format, and
  // the CPU fill writes Y=pad with neutral 128 chroma, exactly like
  // letterboxCpu().
  const int right = padX + newW;
  const int bottom = padY + newH;
  const bool has_border =
      padY > 0 || padX > 0 || bottom < dst.height || right < dst.width;

  // 1. Border, by hardware, over the WHOLE canvas and before the blit. One RGA
  //    op instead of up to four CPU-filled bands, and — the reason it is done
  //    this way — no CPU write into a buffer the blit is about to write. The
  //    bytes under the image rectangle are painted twice, which costs a little
  //    bandwidth and buys an ordering that is simply not racy: same engine,
  //    same queue, in order. See the comment on greySource() for what the CPU
  //    fill measured before this replaced it.
  const bool hw_border = has_border && tryGreyBlit(dst, im_rect{0, 0, dst.width, dst.height}, pad);

  // 2. Crop, scale and colour-convert into the centred rectangle in one pass,
  //    on top of the grey the step above laid down.
  applyCsc(&s, &d, csc);
  process(s, d, srect, drect, src, dst, "letterbox blit", core);

  // 3. Border, the fallback: the CPU paints only the bands the blit did not
  //    cover, and only AFTER it. That order is the lesser evil — filling first
  //    loses the band outright (the blit's cache maintenance discards the CPU's
  //    writes), filling afterwards keeps it but can disturb the seam. A
  //    destination that gets here is one RGA would not take as a blit target,
  //    so it is not the NPU-input hot path.
  if (has_border && !hw_border) {
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
  }
  return lb;
}

LetterboxInfo rgaResize(const ImageView& dst, const ImageView& src, YuvColorSpace yuv) {
  requireRga();
  RCDL_REQUIRE(src.valid() && dst.valid(), "rgaResize: src or dst view is not usable");

  rga_buffer_t s = wrap(src, "src");
  rga_buffer_t d = wrap(dst, "dst");
  const im_rect srect{0, 0, src.width, src.height};
  const im_rect drect{0, 0, dst.width, dst.height};
  checkPair(s, d, srect, drect, src, dst, "resize");
  applyCsc(&s, &d, requireCsc(src, dst, yuv));
  process(s, d, srect, drect, src, dst, "resize", requireCore(src, dst));

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

void rgaCvtColor(const ImageView& dst, const ImageView& src, YuvColorSpace yuv) {
  requireRga();
  RCDL_REQUIRE(src.width == dst.width && src.height == dst.height,
               "rgaCvtColor: src and dst must have the same width and height");

  rga_buffer_t s = wrap(src, "src");
  rga_buffer_t d = wrap(dst, "dst");
  const im_rect srect{0, 0, src.width, src.height};
  const im_rect drect{0, 0, dst.width, dst.height};
  checkPair(s, d, srect, drect, src, dst, "cvtColor");
  // imcvtcolor takes the matrix as an argument rather than off the buffers.
  const int mode = requireCsc(src, dst, yuv);
  pinCore(requireCore(src, dst));
  checkIm(imcvtcolor(s, d, s.format, d.format, mode, /*sync=*/1), "cvtColor", &dst, &src);
}

void rgaCropResize(const ImageView& dst, const ImageView& src, int x, int y, int w, int h,
                   YuvColorSpace yuv) {
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
  applyCsc(&s, &d, requireCsc(src, dst, yuv));
  // The ratio is that of the crop, not of the whole source.
  ImageView crop = src;
  crop.width = w;
  crop.height = h;
  process(s, d, srect, drect, src, dst, "cropResize", requireCore(crop, dst));
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
  pinCore(requireCore(src, dst));
  checkIm(imcopy(s, d, /*sync=*/1), "copy", &dst, &src);
}

bool rgaHwFillUsable(const ImageView& dst) noexcept {
  if (!rgaAvailable() || !dst.valid()) return false;
  return hwFillUsable(dst);
}

void rgaFill(const ImageView& dst, int x, int y, int w, int h, std::uint32_t abgr) {
  requireRga();
  im_rect rect{};
  if (!clipRect(dst, x, y, w, h, &rect)) return;  // nothing of it lands on the canvas
  const rga_buffer_t d = wrap(dst, "dst");
  fillRect(d, rect, abgr, dst);
}

namespace {

// One outline, normalised the same way for both backends: clipped to the
// canvas FIRST (so a box that runs off the frame is closed at the frame edge,
// which is what a detector overlay wants), snapped OUTWARD to even pixels on
// 4:2:0 with the thickness rounded up to even (a chroma sample cannot be
// split, and RGA2 wants every edge 2-aligned), and a rectangle too small to
// have an inside is drawn solid. Returns false when nothing is left to draw.
struct Outline {
  im_rect rect{};
  int thickness = 0;  ///< <= 0 means solid
};

bool outlineOf(const ImageView& dst, const RectSpec& r, Outline* out) noexcept {
  if (r.thickness <= 0 || r.w <= 0 || r.h <= 0) return false;
  const bool yuv = isPlanarYuv(dst.format);
  int x0 = std::max(r.x, 0), y0 = std::max(r.y, 0);
  int x1 = std::min(r.x + r.w, dst.width), y1 = std::min(r.y + r.h, dst.height);
  int t = r.thickness;
  if (yuv) {
    x0 &= ~1;
    y0 &= ~1;
    x1 = std::min(dst.width, (x1 + 1) & ~1);
    y1 = std::min(dst.height, (y1 + 1) & ~1);
    t = (t + 1) & ~1;
  }
  if (x1 <= x0 || y1 <= y0) return false;
  out->rect = im_rect{x0, y0, x1 - x0, y1 - y0};
  const int w = x1 - x0, h = y1 - y0;
  out->thickness = (w > 2 * t && h > 2 * t) ? t : 0;
  return true;
}

// The bands an outline is made of (four, or one solid block).
int outlineBands(const Outline& o, im_rect out[4]) noexcept {
  const im_rect& r = o.rect;
  const int t = o.thickness;
  if (t <= 0) {
    out[0] = r;
    return 1;
  }
  out[0] = im_rect{r.x, r.y, r.width, t};                          // top
  out[1] = im_rect{r.x, r.y + r.height - t, r.width, t};           // bottom
  out[2] = im_rect{r.x, r.y + t, t, r.height - 2 * t};             // left
  out[3] = im_rect{r.x + r.width - t, r.y + t, t, r.height - 2 * t};  // right
  return 4;
}

// Hardware outlines on RGA2, with imrectangleArray: one submit per RUN of
// consecutive rectangles sharing a colour and thickness. Runs rather than
// groups so that boxes are painted in the order given, exactly like the CPU
// path — where two boxes of different colours overlap, the later one is on
// top on both backends, byte for byte. The solid ones (too small for an
// outline) go with thickness -1, im2d's filled rectangle.
void drawRectsHw(const ImageView& dst, const RectSpec* rects, std::size_t count) {
  RCDL_REQUIRE(hwFillUsable(dst),
               "RGA: the hardware colour fill cannot reach this destination (RGA2 has a "
               "32-bit MMU; allocate it from the system-dma32 heap, see docs/RGA.md)");
  const rga_buffer_t d = wrap(dst, "dst");
  struct Item {
    Outline o;
    std::uint32_t abgr;
  };
  std::vector<Item> items;
  items.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    Item it{};
    it.abgr = rects[i].abgr;
    if (outlineOf(dst, rects[i], &it.o)) items.push_back(it);
  }
  for (std::size_t i = 0; i < items.size();) {
    const std::uint32_t colour = items[i].abgr;
    const int thick = items[i].o.thickness;
    std::vector<im_rect> batch;
    for (; i < items.size() && items[i].abgr == colour && items[i].o.thickness == thick; ++i) {
      batch.push_back(items[i].o.rect);
    }
    pinCore(kMaskRga2);
    checkIm(imrectangleArray(d, batch.data(), static_cast<int>(batch.size()), colour,
                             thick > 0 ? thick : -1, /*sync=*/1, /*release_fence_fd=*/nullptr),
            "rectangle overlay", &dst);
  }
}

// CPU outlines: map once, one coherency window for every box.
void drawRectsCpu(const ImageView& dst, const RectSpec* rects, std::size_t count) {
  CpuFillView cpu(dst);
  // READ as well as write: a band edge shares its cache line with pixels the
  // decoder wrote, so the line has to come in fresh before the CPU modifies it.
  dmaBufSyncStart(dst.fd, /*read=*/true, /*write=*/true);
  for (std::size_t i = 0; i < count; ++i) {
    Outline o;
    if (!outlineOf(dst, rects[i], &o)) continue;
    im_rect bands[4];
    const int n = outlineBands(o, bands);
    for (int k = 0; k < n; ++k) fillRectCpuColor(cpu.get(), bands[k], rects[i].abgr);
  }
  dmaBufSyncEnd(dst.fd, /*read=*/true, /*write=*/true);
}

}  // namespace

void rgaDrawRects(const ImageView& dst, const RectSpec* rects, std::size_t count,
                  PreprocBackend backend, PreprocBackend* used) {
  requireRga();
  RCDL_REQUIRE(dst.valid(), "rgaDrawRects: destination view is not usable");
  if (count == 0) return;
  RCDL_REQUIRE(rects != nullptr, "rgaDrawRects: null rectangle array");
  if (backend == PreprocBackend::Rga) {
    drawRectsHw(dst, rects, count);
    if (used != nullptr) *used = PreprocBackend::Rga;
    return;
  }
  // Auto and Cpu both draw with the CPU (measured faster by an order of
  // magnitude, see rga.h); Auto keeps the hardware as the way out for a
  // destination the CPU cannot map at all.
  try {
    drawRectsCpu(dst, rects, count);
    if (used != nullptr) *used = PreprocBackend::Cpu;
    return;
  } catch (const Error&) {
    if (backend != PreprocBackend::Auto || !hwFillUsable(dst)) throw;
  }
  drawRectsHw(dst, rects, count);
  if (used != nullptr) *used = PreprocBackend::Rga;
}

void rgaDrawRect(const ImageView& dst, int x, int y, int w, int h, std::uint32_t abgr,
                 int thickness, PreprocBackend backend) {
  const RectSpec r{x, y, w, h, abgr, thickness};
  rgaDrawRects(dst, &r, 1, backend, nullptr);
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

bool rgaCanHandle(const ImageView&, const ImageView&, std::string* why, YuvColorSpace) noexcept {
  if (why != nullptr) *why = kUnavailable;
  return false;
}

LetterboxInfo rgaLetterbox(const ImageView&, const ImageView&, std::uint8_t, YuvColorSpace) {
  noRga();
}

LetterboxInfo rgaResize(const ImageView&, const ImageView&, YuvColorSpace) { noRga(); }

void rgaCvtColor(const ImageView&, const ImageView&, YuvColorSpace) { noRga(); }

void rgaCropResize(const ImageView&, const ImageView&, int, int, int, int, YuvColorSpace) {
  noRga();
}

void rgaCopy(const ImageView&, const ImageView&) { noRga(); }

bool rgaHwFillUsable(const ImageView&) noexcept { return false; }

void rgaFill(const ImageView&, int, int, int, int, std::uint32_t) { noRga(); }

void rgaDrawRects(const ImageView&, const RectSpec*, std::size_t, PreprocBackend,
                  PreprocBackend*) {
  noRga();
}

void rgaDrawRect(const ImageView&, int, int, int, int, std::uint32_t, int, PreprocBackend) {
  noRga();
}

#endif  // RCDL_HAVE_RGA

}  // namespace rcdl
