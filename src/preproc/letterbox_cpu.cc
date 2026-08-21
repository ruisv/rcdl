#include "rcdl/preproc/letterbox_cpu.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "rcdl/core/status.h"

namespace rcdl {
namespace {

/// Round-half-away-from-zero to a saturated u8 — the rounding the numpy
/// reference in tests/ agrees with. Everything here accumulates in float and
/// converts to u8 once, at the end.
inline std::uint8_t clampU8(float v) {
  const int i = static_cast<int>(std::lround(v));
  return static_cast<std::uint8_t>(i < 0 ? 0 : (i > 255 ? 255 : i));
}

/// Byte layout of the packed formats: where R, G and B sit inside a pixel,
/// where alpha sits (-1 when the format has none) and whether the format is
/// single-channel luma. `bpp == 0` means "not packed" (a 4:2:0 YUV, or
/// Unknown) — that is how every dispatch below tells the two families apart.
///
/// `gray` is a colour-space flag as much as a layout one: GRAY8 is the luma
/// plane of a YUV image (RGA calls it RK_FORMAT_YCbCr_400), so it sits on the
/// YUV SIDE of a conversion. RGB -> GRAY8 therefore changes levels exactly as
/// RGB -> NV12 does, and NV12 -> GRAY8 is a YUV -> YUV move that must not.
struct Packed {
  int bpp;
  int r, g, b, a;
  bool gray;
};

Packed packedOf(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::RGB888:
      return {3, 0, 1, 2, -1, false};
    case PixelFormat::BGR888:
      return {3, 2, 1, 0, -1, false};
    case PixelFormat::RGBA8888:
      return {4, 0, 1, 2, 3, false};
    case PixelFormat::BGRA8888:
      return {4, 2, 1, 0, 3, false};
    case PixelFormat::GRAY8:
      return {1, 0, 0, 0, -1, true};
    default:
      return {0, -1, -1, -1, -1, false};
  }
}

/// All three YUV formats RCDL carries are 4:2:0 with an 8-bit Y plane; they
/// differ only in how the two chroma channels are laid out after it.
bool isYuv420(PixelFormat f) noexcept {
  switch (f) {
    case PixelFormat::NV12:
    case PixelFormat::NV21:
    case PixelFormat::YUV420P:
      return true;
    default:
      return false;
  }
}

/// The three chroma layouts reduced to one addressing scheme: a base pointer
/// per channel, a row stride and the byte distance between two successive
/// samples of the SAME channel (2 for the interleaved NV formats, 1 for the
/// fully planar one). Every loop below then treats U and V uniformly and does
/// not care which of the three formats it is looking at.
///
/// ImageView::data is a plain `void*` even behind a `const ImageView&` (the
/// descriptor is const, not the pixels), so one mutable struct serves both the
/// source and the destination side.
struct Yuv {
  std::uint8_t* y = nullptr;
  std::uint8_t* u = nullptr;
  std::uint8_t* v = nullptr;
  std::size_t yStride = 0;
  std::size_t cStride = 0;
  int cStep = 1;
};

Yuv yuvPlanes(const ImageView& im) {
  Yuv p;
  std::uint8_t* base = im.bytePtr();
  p.y = base;
  p.yStride = im.rowBytes();  // == wstride bytes for every 4:2:0 layout
  std::uint8_t* c = base + im.uvOffset();
  switch (im.format) {
    case PixelFormat::NV12:
      p.u = c;
      p.v = c + 1;
      p.cStride = im.rowBytes();
      p.cStep = 2;
      break;
    case PixelFormat::NV21:
      p.v = c;
      p.u = c + 1;
      p.cStride = im.rowBytes();
      p.cStep = 2;
      break;
    default: {  // YUV420P: a full U plane then a full V plane, both half stride
      const std::size_t half = static_cast<std::size_t>(im.effWStride() / 2);
      p.u = c;
      p.v = c + half * static_cast<std::size_t>(im.effHStride() / 2);
      p.cStride = half;
      p.cStep = 1;
      break;
    }
  }
  return p;
}

/// RCDL_REQUIRE with the offending view spelled out — the caller of a preproc
/// function usually has several views in flight, so the message has to say
/// which one was wrong.
void requireView(bool cond, const std::string& what, const ImageView& v) {
  if (!cond) {
    const std::string msg = what + ": " + v.describe();
    RCDL_REQUIRE(false, msg.c_str());
  }
}

/// Preconditions shared by every entry point: a CPU mapping (these functions
/// dereference `data`, an fd-only view belongs on the RGA path), a format this
/// file knows, and — for 4:2:0 — even extents, because an odd width, height or
/// stride would split a chroma sample.
void validate(const ImageView& v, const char* fn, const char* role) {
  const std::string where = std::string(fn) + ": " + role;
  requireView(v.valid(), where + " is not a valid ImageView", v);
  requireView(v.data != nullptr, where + " needs a CPU pointer (ImageView::data)", v);
  requireView(packedOf(v.format).bpp > 0 || isYuv420(v.format),
              where + " has a format the CPU path does not support", v);
  requireView(v.effWStride() >= v.width && v.effHStride() >= v.height,
              where + " has a stride smaller than its extent", v);
  if (isYuv420(v.format)) {
    requireView((v.width % 2) == 0 && (v.height % 2) == 0 && (v.effWStride() % 2) == 0 &&
                    (v.effHStride() % 2) == 0,
                where + " is 4:2:0 and needs even width/height and even strides", v);
  }
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Bilinear resample of `chans` u8 channels, OpenCV's pixel-center convention
///     srcX = (dstX + 0.5) * invSx - 0.5,  clamped to [0, srcW-1]
/// where invSx is srcW/outW — the effective scale of the integer rectangle the
/// caller decided on, i.e. exactly what cv::resize(INTER_LINEAR) to that size
/// does. Both pointers address the ORIGIN OF THE REGION, so a
/// letterbox just passes a pre-offset destination pointer and the padded border
/// is never touched.
///
/// `srcStep` / `dstStep` are the byte distance between two successive pixels of
/// the plane: `chans` for a packed image, but 2 when resampling one channel of
/// an interleaved NV12 chroma plane into another. Keeping them independent is
/// what lets a single function serve packed RGB, luma and chroma.
void resample(std::uint8_t* dstBase, std::size_t dstStride, int dstStep,
              const std::uint8_t* srcBase, std::size_t srcStride, int srcStep, int chans,
              int outW, int outH, int srcW, int srcH, float invSx, float invSy) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int dy = 0; dy < outH; ++dy) {
    float fy = (dy + 0.5f) * invSy - 0.5f;
    if (fy < 0.0f) fy = 0.0f;
    if (fy > static_cast<float>(srcH - 1)) fy = static_cast<float>(srcH - 1);
    const int y0 = static_cast<int>(fy);
    const int y1 = std::min(y0 + 1, srcH - 1);
    const float wy = fy - static_cast<float>(y0);

    const std::uint8_t* row0 = srcBase + static_cast<std::size_t>(y0) * srcStride;
    const std::uint8_t* row1 = srcBase + static_cast<std::size_t>(y1) * srcStride;
    std::uint8_t* out = dstBase + static_cast<std::size_t>(dy) * dstStride;

    for (int dx = 0; dx < outW; ++dx) {
      float fx = (dx + 0.5f) * invSx - 0.5f;
      if (fx < 0.0f) fx = 0.0f;
      if (fx > static_cast<float>(srcW - 1)) fx = static_cast<float>(srcW - 1);
      const int x0 = static_cast<int>(fx);
      const int x1 = std::min(x0 + 1, srcW - 1);
      const float wx = fx - static_cast<float>(x0);

      const int o0 = x0 * srcStep;
      const int o1 = x1 * srcStep;
      std::uint8_t* p = out + static_cast<std::size_t>(dx) * static_cast<std::size_t>(dstStep);
      for (int c = 0; c < chans; ++c) {
        const float v00 = row0[o0 + c];
        const float v01 = row0[o1 + c];
        const float v10 = row1[o0 + c];
        const float v11 = row1[o1 + c];
        const float top = v00 + (v01 - v00) * wx;
        const float bot = v10 + (v11 - v10) * wx;
        p[c] = clampU8(top + (bot - top) * wy);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Colour conversion
// ---------------------------------------------------------------------------

/// BT.601 full-range luma — the same matrix row the RGB -> YUV direction uses,
/// so RGB -> GRAY8 and RGB -> NV12's Y plane agree to the last bit.
inline float luma601(float r, float g, float b) { return 0.299f * r + 0.587f * g + 0.114f * b; }

/// The two YUV -> RGB matrices from letterbox_cpu.h, picked by `range`. Both
/// fold the level handling of the YUV side into the matrix, so this is the only
/// place the YUV -> RGB direction needs to know about swing.
struct YuvMatrix {
  float ky, kr, kgu, kgv, kb, yoff;
};

YuvMatrix yuvMatrix(YuvRange range) {
  // kStudioToFull is exactly cv::cvtColor(COLOR_YUV2BGR_NV12): the YUV side is
  // studio-swing, so Y in [16,235] expands to [0,255]. kAsIs is the plain
  // matrix. Each is the exact inverse of rgbLevels() for the same `range`.
  if (range == YuvRange::kStudioToFull) return {1.164f, 1.596f, 0.391f, 0.813f, 2.018f, 16.0f};
  return {1.0f, 1.402f, 0.344f, 0.714f, 1.772f, 0.0f};
}

/// The RGB -> YUV direction of the same contract. `range` describes the YUV
/// SIDE, not a one-way instruction, so kStudioToFull here means "compress into
/// studio swing" — the exact inverse of the expansion yuvMatrix() folds into
/// the other direction, and what RGA selects for the same call
/// (IM_RGB_TO_YUV_BT601_LIMIT):
///     Y = 16 + (219/255) * Yfull,   C = 128 + (224/255) * (Cfull - 128)
/// Getting this asymmetric is how the two backends end up writing ~14%
/// different luma for one and the same cvtColor() call.
struct RgbLevels {
  float ky, yoff, kc;
};

RgbLevels rgbLevels(YuvRange range) {
  if (range == YuvRange::kStudioToFull) return {219.0f / 255.0f, 16.0f, 224.0f / 255.0f};
  return {1.0f, 0.0f, 1.0f};
}

// YUV -> YUV needs no table at all: both sides are the same YUV side, so under
// either `range` the levels are already what they should be and only the plane
// layout changes.

/// Packed -> packed: a channel permutation, an alpha added (opaque) or dropped,
/// or a luma reduction. GRAY8 is the YUV side of the pair, so RGB <-> GRAY8
/// carries the same level change RGB <-> NV12 does, while RGB -> RGB and
/// GRAY8 -> GRAY8 carry none.
///
/// A permutation is per-pixel and independent of the neighbours, so the callers
/// may run it before or after the resample; a LEVEL change saturates and
/// therefore does not commute with interpolation, which is why warp() pins the
/// order in that case.
void packedToPacked(std::uint8_t* dst, std::size_t dstStride, PixelFormat dstFmt,
                    const std::uint8_t* src, std::size_t srcStride, PixelFormat srcFmt, int w,
                    int h, YuvRange range) {
  const Packed S = packedOf(srcFmt);
  const Packed D = packedOf(dstFmt);
  const RgbLevels lv = rgbLevels(range);       // RGB -> luma: compress
  const YuvMatrix m = yuvMatrix(range);        // luma -> RGB: expand
  const bool fromLuma = S.gray && !D.gray;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < h; ++y) {
    const std::uint8_t* srow = src + static_cast<std::size_t>(y) * srcStride;
    std::uint8_t* drow = dst + static_cast<std::size_t>(y) * dstStride;
    for (int x = 0; x < w; ++x) {
      const std::uint8_t* sp = srow + static_cast<std::size_t>(x) * S.bpp;
      std::uint8_t* dp = drow + static_cast<std::size_t>(x) * D.bpp;
      const std::uint8_t r = sp[S.r];
      const std::uint8_t g = S.gray ? r : sp[S.g];
      const std::uint8_t b = S.gray ? r : sp[S.b];
      if (D.gray) {
        // GRAY8 -> GRAY8 is YUV -> YUV: copy. RGB -> GRAY8 compresses into the
        // YUV side's levels, exactly as RGB -> NV12's Y plane does.
        dp[0] = S.gray ? r : clampU8(lv.yoff + lv.ky * luma601(r, g, b));
      } else if (fromLuma) {
        // GRAY8 -> RGB expands the YUV side's levels; with U = V = 128 the
        // colour matrix contributes nothing, so all three channels come out
        // equal.
        const std::uint8_t e = clampU8(m.ky * (static_cast<float>(r) - m.yoff));
        dp[D.r] = e;
        dp[D.g] = e;
        dp[D.b] = e;
        if (D.a >= 0) dp[D.a] = 255;
      } else {
        dp[D.r] = r;
        dp[D.g] = g;
        dp[D.b] = b;
        // An added alpha channel is opaque; a source alpha is carried through.
        if (D.a >= 0) dp[D.a] = (S.a >= 0) ? sp[S.a] : 255;
      }
    }
  }
}

/// 4:2:0 -> packed RGB / luma at identical size. Chroma is nearest-neighbour
/// (each 2x2 luma block shares one sample), matching cv::cvtColor.
void yuvToPacked(std::uint8_t* dst, std::size_t dstStride, PixelFormat dstFmt,
                 const ImageView& src, YuvRange range) {
  const Packed D = packedOf(dstFmt);
  const Yuv s = yuvPlanes(src);
  const YuvMatrix m = yuvMatrix(range);
  const int w = src.width;
  const int h = src.height;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < h; ++y) {
    const std::uint8_t* yrow = s.y + static_cast<std::size_t>(y) * s.yStride;
    const std::uint8_t* urow = s.u + static_cast<std::size_t>(y / 2) * s.cStride;
    const std::uint8_t* vrow = s.v + static_cast<std::size_t>(y / 2) * s.cStride;
    std::uint8_t* drow = dst + static_cast<std::size_t>(y) * dstStride;
    for (int x = 0; x < w; ++x) {
      if (D.gray) {
        // YUV -> GRAY8 is a YUV -> YUV move: the destination IS the luma plane,
        // so it keeps the source's levels whatever `range` says. Expanding here
        // would make NV12 -> GRAY8 -> RGB disagree with NV12 -> RGB.
        drow[x] = yrow[x];
        continue;
      }
      const float Y = m.ky * (static_cast<float>(yrow[x]) - m.yoff);
      const std::size_t co = static_cast<std::size_t>(x / 2) * static_cast<std::size_t>(s.cStep);
      const float U = static_cast<float>(urow[co]) - 128.0f;
      const float V = static_cast<float>(vrow[co]) - 128.0f;
      std::uint8_t* p = drow + static_cast<std::size_t>(x) * D.bpp;
      p[D.r] = clampU8(Y + m.kr * V);
      p[D.g] = clampU8(Y - m.kgu * U - m.kgv * V);
      p[D.b] = clampU8(Y + m.kb * U);
      if (D.a >= 0) p[D.a] = 255;
    }
  }
}

/// Packed RGB / luma -> the (x0,y0,w,h) region of a 4:2:0 destination. The
/// region must be even-aligned in both offset and extent so the 2x2 chroma
/// blocks line up with the destination's own grid.
///
/// BT.601 throughout, with the levels of the YUV side selected by `range`:
/// kAsIs writes full range (Y = 0.299R+0.587G+0.114B, no offset — cv2's
/// COLOR_BGR2YUV_I420, the exact inverse of yuvMatrix(kAsIs)) and
/// kStudioToFull compresses into [16,235] / [16,240], the exact inverse of the
/// expansion yuvMatrix(kStudioToFull) applies coming back.
///
/// A GRAY8 source is the YUV side already, so it is copied, not converted.
void packedToYuvRegion(const ImageView& dst, int x0, int y0, int w, int h,
                       const std::uint8_t* src, std::size_t srcStride, PixelFormat srcFmt,
                       YuvRange range) {
  const Packed S = packedOf(srcFmt);
  const Yuv d = yuvPlanes(dst);
  const RgbLevels lv = rgbLevels(range);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < h; ++y) {
    const std::uint8_t* srow = src + static_cast<std::size_t>(y) * srcStride;
    std::uint8_t* yrow = d.y + static_cast<std::size_t>(y0 + y) * d.yStride + x0;
    for (int x = 0; x < w; ++x) {
      const std::uint8_t* p = srow + static_cast<std::size_t>(x) * S.bpp;
      // GRAY8 -> YUV is YUV -> YUV: the luma plane moves across untouched.
      yrow[x] = S.gray ? p[0] : clampU8(lv.yoff + lv.ky * luma601(p[S.r], p[S.g], p[S.b]));
    }
  }

  // Chroma: average each 2x2 RGB block, then convert once (4:2:0 subsampling).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int cy = 0; cy < h / 2; ++cy) {
    const std::uint8_t* srow0 = src + static_cast<std::size_t>(cy * 2) * srcStride;
    const std::uint8_t* srow1 = src + static_cast<std::size_t>(cy * 2 + 1) * srcStride;
    const std::size_t crow = static_cast<std::size_t>(y0 / 2 + cy) * d.cStride +
                             static_cast<std::size_t>(x0 / 2) * static_cast<std::size_t>(d.cStep);
    std::uint8_t* urow = d.u + crow;
    std::uint8_t* vrow = d.v + crow;
    for (int cx = 0; cx < w / 2; ++cx) {
      const std::size_t co = static_cast<std::size_t>(cx) * static_cast<std::size_t>(d.cStep);
      if (S.gray) {
        // A luma-only source has no colour: neutral chroma, no levels involved.
        urow[co] = 128;
        vrow[co] = 128;
        continue;
      }
      const std::uint8_t* a = srow0 + static_cast<std::size_t>(cx * 2) * S.bpp;
      const std::uint8_t* b = srow0 + static_cast<std::size_t>(cx * 2 + 1) * S.bpp;
      const std::uint8_t* c = srow1 + static_cast<std::size_t>(cx * 2) * S.bpp;
      const std::uint8_t* e = srow1 + static_cast<std::size_t>(cx * 2 + 1) * S.bpp;
      const float R = 0.25f * (a[S.r] + b[S.r] + c[S.r] + e[S.r]);
      const float G = 0.25f * (a[S.g] + b[S.g] + c[S.g] + e[S.g]);
      const float B = 0.25f * (a[S.b] + b[S.b] + c[S.b] + e[S.b]);
      urow[co] = clampU8(128.0f + lv.kc * (-0.169f * R - 0.331f * G + 0.500f * B));  // Cb
      vrow[co] = clampU8(128.0f + lv.kc * (0.500f * R - 0.419f * G - 0.081f * B));   // Cr
    }
  }
}

/// 4:2:0 -> 4:2:0 at identical size: a per-plane copy that re-orders the chroma
/// (NV12 <-> NV21 <-> YUV420P is nothing but a layout change).
///
/// No level conversion in either direction: `range` describes the YUV side, and
/// here BOTH sides are that same YUV side, so there is nothing to convert. RGA
/// makes the same call (IM_COLOR_SPACE_DEFAULT when neither side is RGB).
void yuvToYuv(const ImageView& dst, const ImageView& src) {
  const Yuv s = yuvPlanes(src);
  const Yuv d = yuvPlanes(dst);
  const int w = src.width;
  const int h = src.height;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int y = 0; y < h; ++y) {
    const std::uint8_t* srow = s.y + static_cast<std::size_t>(y) * s.yStride;
    std::uint8_t* drow = d.y + static_cast<std::size_t>(y) * d.yStride;
    std::memcpy(drow, srow, static_cast<std::size_t>(w));
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int cy = 0; cy < h / 2; ++cy) {
    const std::uint8_t* su = s.u + static_cast<std::size_t>(cy) * s.cStride;
    const std::uint8_t* sv = s.v + static_cast<std::size_t>(cy) * s.cStride;
    std::uint8_t* du = d.u + static_cast<std::size_t>(cy) * d.cStride;
    std::uint8_t* dv = d.v + static_cast<std::size_t>(cy) * d.cStride;
    for (int cx = 0; cx < w / 2; ++cx) {
      du[static_cast<std::size_t>(cx) * d.cStep] = su[static_cast<std::size_t>(cx) * s.cStep];
      dv[static_cast<std::size_t>(cx) * d.cStep] = sv[static_cast<std::size_t>(cx) * s.cStep];
    }
  }
}

/// 4:2:0 -> 4:2:0 resample into an even-aligned destination region. The chroma
/// planes are half resolution on BOTH sides, so the very same scale applies to
/// them; U and V are resampled as separate channels, which makes any
/// interleaved/planar combination fall out for free. Levels are left alone —
/// both sides are the same YUV side (see yuvToYuv()).
void yuvWarpRegion(const ImageView& dst, int x0, int y0, int w, int h, const ImageView& src,
                   float invSx, float invSy) {
  const Yuv s = yuvPlanes(src);
  const Yuv d = yuvPlanes(dst);
  const int sw = src.width;
  const int sh = src.height;

  resample(d.y + static_cast<std::size_t>(y0) * d.yStride + x0, d.yStride, 1, s.y, s.yStride, 1, 1,
           w, h, sw, sh, invSx, invSy);
  const std::size_t croff = static_cast<std::size_t>(y0 / 2) * d.cStride +
                            static_cast<std::size_t>(x0 / 2) * static_cast<std::size_t>(d.cStep);
  resample(d.u + croff, d.cStride, d.cStep, s.u, s.cStride, s.cStep, 1, w / 2, h / 2, sw / 2,
           sh / 2, invSx, invSy);
  resample(d.v + croff, d.cStride, d.cStep, s.v, s.cStride, s.cStep, 1, w / 2, h / 2, sw / 2,
           sh / 2, invSx, invSy);
}

// ---------------------------------------------------------------------------
// Letterbox / resize
// ---------------------------------------------------------------------------

/// The body shared by letterboxCpu() and resizeCpu(): they differ only in the
/// geometry (uniform scale + centred padding vs. per-axis stretch) — the
/// format dispatch below is identical.
LetterboxInfo warp(const ImageView& dst, const ImageView& src, std::uint8_t pad, YuvRange range,
                   bool stretch) {
  const char* fn = stretch ? "resizeCpu" : "letterboxCpu";
  validate(dst, fn, "dst");
  validate(src, fn, "src");

  const int sw = src.width;
  const int sh = src.height;
  const int dw = dst.width;
  const int dh = dst.height;

  LetterboxInfo lb;
  int px = 0, py = 0, ow = dw, oh = dh;
  float invSx = 1.0f, invSy = 1.0f;
  if (stretch) {
    lb.srcW = sw;
    lb.srcH = sh;
    lb.dstW = dw;
    lb.dstH = dh;
    // Like rgaResize(): report the X scale with no padding. The uniform-scale
    // inverse map is only exact when the aspect ratios already match.
    lb.scale = static_cast<float>(dw) / static_cast<float>(sw);
    invSx = static_cast<float>(sw) / static_cast<float>(dw);
    invSy = static_cast<float>(sh) / static_cast<float>(dh);
  } else {
    lb = computeLetterbox(sw, sh, dw, dh);
    // The content rectangle is INTEGER: round the scaled extent first, then
    // centre that. Deriving the padding from the rounded extent (rather than
    // rounding the float padding) is what keeps this identical to the RGA path
    // and to the numpy reference, which differ from each other by a pixel when
    // (dst - round(src*scale)) is odd.
    ow = std::min(static_cast<int>(std::lround(sw * lb.scale)), dw);
    oh = std::min(static_cast<int>(std::lround(sh * lb.scale)), dh);
  }

  const bool srcYuv = isYuv420(src.format);
  const bool dstYuv = isYuv420(dst.format);
  if (!stretch) {
    // 4:2:0 chroma is 2x2 subsampled: an odd offset or extent would split a
    // chroma sample across the border.
    if (dstYuv) {
      ow &= ~1;
      oh &= ~1;
    }
    px = (dw - ow) / 2;
    py = (dh - oh) / 2;
    if (dstYuv) {
      px &= ~1;
      py &= ~1;
    }
    // Per-axis, from the INTEGER extent actually written: sampling at
    // srcW/outW is exactly a cv::resize to (ow,oh) pasted into the canvas.
    // It equals 1/scale whenever the scaled extent was already integral, and
    // is the fractional-pixel-correct answer when it was not.
    invSx = (ow > 0) ? static_cast<float>(sw) / static_cast<float>(ow) : 0.0f;
    invSy = (oh > 0) ? static_cast<float>(sh) / static_cast<float>(oh) : 0.0f;
  }
  // The hardware path writes integer destination rectangles and reflects that
  // rounding back into the geometry it returns; do the same here so the inverse
  // map post-processing uses matches the pixels whichever backend ran.
  lb.padX = static_cast<float>(px);
  lb.padY = static_cast<float>(py);

  if (!stretch) fillCpu(dst, pad);  // border first; the content is written over it
  if (ow <= 0 || oh <= 0) return lb;

  const Packed S = packedOf(src.format);
  const Packed D = packedOf(dst.format);
  const std::uint8_t* sBase = src.bytePtr();
  const std::size_t sStride = src.rowBytes();
  const std::size_t dStride = dst.rowBytes();

  if (!srcYuv && !dstYuv) {
    std::uint8_t* dRoi = dst.bytePtr() + static_cast<std::size_t>(py) * dStride +
                         static_cast<std::size_t>(px) * D.bpp;
    // RGB <-> GRAY8 crosses between the RGB and the YUV side, so it carries a
    // level change under kStudioToFull. That transform saturates, and a
    // saturating transform does NOT commute with interpolation (a source value
    // below 16 clips to 0 either before or after the blend, with different
    // results), so pin the order there: convert first, always.
    const bool levelChange = (S.gray != D.gray) && range == YuvRange::kStudioToFull;
    if (src.format == dst.format) {
      // Same format: straight into the destination, nothing allocated.
      resample(dRoi, dStride, D.bpp, sBase, sStride, S.bpp, D.bpp, ow, oh, sw, sh, invSx, invSy);
    } else if (!levelChange && static_cast<std::size_t>(ow) * oh <=
                                   static_cast<std::size_t>(sw) * sh) {
      // A pure channel shuffle is per-pixel, so it commutes with bilinear
      // interpolation and only the temporary's size decides the order.
      // Downscale (the usual case): resample first, so the temporary is the
      // small destination-sized one.
      std::vector<std::uint8_t> tmp(static_cast<std::size_t>(ow) * oh * S.bpp);
      const std::size_t tStride = static_cast<std::size_t>(ow) * S.bpp;
      resample(tmp.data(), tStride, S.bpp, sBase, sStride, S.bpp, S.bpp, ow, oh, sw, sh, invSx,
               invSy);
      packedToPacked(dRoi, dStride, dst.format, tmp.data(), tStride, src.format, ow, oh, range);
    } else {
      // Upscale, or a level change: convert first, at the source size.
      std::vector<std::uint8_t> tmp(static_cast<std::size_t>(sw) * sh * D.bpp);
      const std::size_t tStride = static_cast<std::size_t>(sw) * D.bpp;
      packedToPacked(tmp.data(), tStride, dst.format, sBase, sStride, src.format, sw, sh, range);
      resample(dRoi, dStride, D.bpp, tmp.data(), tStride, D.bpp, D.bpp, ow, oh, sw, sh, invSx,
               invSy);
    }
  } else if (srcYuv && !dstYuv) {
    // YUV source (the video path): convert at FULL source resolution first, then
    // resample. Resampling the 2x2 subsampled chroma before the matrix would
    // interpolate colour at half resolution and would not match a
    // cvtColor-then-resize reference. Temporary: srcW*srcH*dstBpp.
    std::vector<std::uint8_t> tmp(static_cast<std::size_t>(sw) * sh * D.bpp);
    const std::size_t tStride = static_cast<std::size_t>(sw) * D.bpp;
    yuvToPacked(tmp.data(), tStride, dst.format, src, range);
    std::uint8_t* dRoi = dst.bytePtr() + static_cast<std::size_t>(py) * dStride +
                         static_cast<std::size_t>(px) * D.bpp;
    resample(dRoi, dStride, D.bpp, tmp.data(), tStride, D.bpp, D.bpp, ow, oh, sw, sh, invSx, invSy);
  } else if (!srcYuv && dstYuv) {
    // YUV destination (the encode path): resample first, in the source's packed
    // format and at the destination size, then subsample. That temporary
    // (outW*outH*srcBpp) is the smaller one for any downscale AND leaves the
    // 4:2:0 averaging as the very last step, where it loses the least.
    std::vector<std::uint8_t> tmp(static_cast<std::size_t>(ow) * oh * S.bpp);
    const std::size_t tStride = static_cast<std::size_t>(ow) * S.bpp;
    resample(tmp.data(), tStride, S.bpp, sBase, sStride, S.bpp, S.bpp, ow, oh, sw, sh, invSx,
             invSy);
    packedToYuvRegion(dst, px, py, ow, oh, tmp.data(), tStride, src.format, range);
  } else {
    // YUV -> YUV: plane-wise, straight into the destination region. Nothing
    // allocated, and no colour matrix is involved at all.
    yuvWarpRegion(dst, px, py, ow, oh, src, invSx, invSy);
  }
  return lb;
}

}  // namespace

LetterboxInfo letterboxCpu(const ImageView& dst, const ImageView& src, std::uint8_t pad,
                           YuvRange range) {
  return warp(dst, src, pad, range, /*stretch=*/false);
}

LetterboxInfo resizeCpu(const ImageView& dst, const ImageView& src, YuvRange range) {
  return warp(dst, src, 0, range, /*stretch=*/true);
}

void cvtColorCpu(const ImageView& dst, const ImageView& src, YuvRange range) {
  validate(dst, "cvtColorCpu", "dst");
  validate(src, "cvtColorCpu", "src");
  requireView(dst.width == src.width && dst.height == src.height,
              "cvtColorCpu: dst and src must have the same width and height, src is " +
                  src.describe() + ", dst",
              dst);

  const bool srcYuv = isYuv420(src.format);
  const bool dstYuv = isYuv420(dst.format);
  if (srcYuv && dstYuv) {
    yuvToYuv(dst, src);
  } else if (srcYuv) {
    yuvToPacked(dst.bytePtr(), dst.rowBytes(), dst.format, src, range);
  } else if (dstYuv) {
    packedToYuvRegion(dst, 0, 0, src.width, src.height, src.bytePtr(), src.rowBytes(), src.format,
                      range);
  } else {
    // Identical packed formats land here too and come out as a plain copy.
    packedToPacked(dst.bytePtr(), dst.rowBytes(), dst.format, src.bytePtr(), src.rowBytes(),
                   src.format, src.width, src.height, range);
  }
}

void fillCpu(const ImageView& dst, std::uint8_t value) {
  validate(dst, "fillCpu", "dst");
  std::uint8_t* base = dst.bytePtr();
  const std::size_t rows = static_cast<std::size_t>(dst.effHStride());
  // Fill through the stride padding as well: it costs one memset either way and
  // leaves no undefined bytes for a later hardware stage to read.
  std::memset(base, value, dst.rowBytes() * rows);
  if (isYuv420(dst.format)) {
    // Neutral chroma, so a grey Y really is grey. Both chroma planes of
    // YUV420P are contiguous, so one memset covers every layout.
    std::memset(base + dst.uvOffset(), 128, dst.rowBytes() * (rows / 2));
  }
}

void fillRectCpu(const ImageView& dst, int x, int y, int w, int h, std::uint8_t value) {
  validate(dst, "fillRectCpu", "dst");
  // Clip to the canvas; an empty or fully-outside rectangle is a no-op.
  int x0 = std::max(0, x), y0 = std::max(0, y);
  int x1 = std::min(dst.width, x + w), y1 = std::min(dst.height, y + h);
  const bool yuv = isYuv420(dst.format);
  if (yuv) {
    // Snap OUTWARD to even bounds: a 2x2 chroma sample must be wholly inside or
    // wholly outside the filled region, never split.
    x0 &= ~1;
    y0 &= ~1;
    x1 = std::min(dst.width, (x1 + 1) & ~1);
    y1 = std::min(dst.height, (y1 + 1) & ~1);
  }
  if (x1 <= x0 || y1 <= y0) return;

  std::uint8_t* base = dst.bytePtr();
  const std::size_t stride = dst.rowBytes();
  const int bpp = yuv ? 1 : bytesPerPixel(dst.format);
  const std::size_t run = static_cast<std::size_t>(x1 - x0) * static_cast<std::size_t>(bpp);
  for (int r = y0; r < y1; ++r) {
    std::memset(base + static_cast<std::size_t>(r) * stride +
                    static_cast<std::size_t>(x0) * bpp,
                value, run);
  }
  if (!yuv) return;
  // Chroma: neutral 128 over the corresponding half-resolution rectangle.
  std::uint8_t* uv = base + dst.uvOffset();
  const int cy0 = y0 / 2, cy1 = y1 / 2;
  if (dst.format == PixelFormat::YUV420P) {
    // Two separate planes, each at half stride and half height.
    const std::size_t cstride = stride / 2;
    const std::size_t plane = cstride * static_cast<std::size_t>(dst.effHStride() / 2);
    const std::size_t crun = static_cast<std::size_t>(x1 - x0) / 2;
    for (int r = cy0; r < cy1; ++r) {
      const std::size_t off = static_cast<std::size_t>(r) * cstride +
                              static_cast<std::size_t>(x0) / 2;
      std::memset(uv + off, 128, crun);
      std::memset(uv + plane + off, 128, crun);
    }
    return;
  }
  // NV12 / NV21: one interleaved plane at the luma row stride, half the rows.
  const std::size_t uv_run = static_cast<std::size_t>(x1 - x0);
  for (int r = cy0; r < cy1; ++r) {
    std::memset(uv + static_cast<std::size_t>(r) * stride + static_cast<std::size_t>(x0), 128,
                uv_run);
  }
}

}  // namespace rcdl
