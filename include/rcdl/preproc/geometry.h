#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rcdl {

/// The levels of the YUV SIDE of a conversion — not a one-way instruction.
///
/// Video carries **studio-swing** YUV (Y in [16,235], chroma in [16,240]); models
/// are calibrated on full-range pixels. This enum says which of the two a YUV
/// buffer holds, and every conversion reads it the same way, so the two
/// directions — and the two backends, since RGA picks its colour-space mode from
/// this same value — agree:
///
///   YUV -> RGB   expands under kStudioToFull (= cv::cvtColor(COLOR_YUV2BGR_NV12),
///                RGA's IM_YUV_TO_RGB_BT601_LIMIT)
///   RGB -> YUV   compresses back into that swing (RGA's IM_RGB_TO_YUV_BT601_LIMIT)
///   YUV -> YUV   never touches the levels — both sides are the same YUV side,
///                so RGA uses IM_COLOR_SPACE_DEFAULT and the CPU path copies
///
/// GRAY8 counts as the YUV side: it is a luma plane (RK_FORMAT_YCbCr_400).
/// Handing the NPU an unexpanded studio-swing frame costs the model ~14% of its
/// contrast against calibration, which is why kStudioToFull is the default.
///
/// The levels are only half of what a YUV buffer needs to say; the colour
/// matrix is the other half — see YuvMatrix and YuvColorSpace below.
enum class YuvRange {
  kAsIs,          ///< the YUV side is full-range: colour matrix only, no level shift
  kStudioToFull,  ///< the YUV side is studio-swing: expand coming out, compress going in
};

/// The colour matrix of the YUV side: which weights of R, G and B its Y, Cb and
/// Cr carry. Independent of the levels — a stream can be BT.709 at either swing.
///
/// Decoding with the wrong one does not change contrast the way a wrong range
/// does; it shifts hue and saturation — by up to ~30 LSB per channel on
/// saturated colour — which is just as invisible to a model.
enum class YuvMatrix {
  kBt601,  ///< SD video, JPEG/JFIF, and what cv::cvtColor's YUV codes assume
  kBt709,  ///< HD video: H.264 / H.265 at 720p and above normally signal this
};

/// Everything a conversion needs to know about its YUV side: levels and matrix.
///
/// Implicitly constructible from a bare YuvRange, which means BT.601 — so a call
/// written as `letterbox(dst, src, pad, backend, YuvRange::kAsIs)` keeps both
/// compiling and its meaning.
///
/// Not every combination runs on RGA: the hardware converts YUV -> RGB in
/// BT.601 limited / full and BT.709 limited, and RGB -> YUV in BT.601 only.
/// rgaCanHandle() says no to the rest, so PreprocBackend::Auto routes them to
/// the CPU path, which implements all four in both directions.
struct YuvColorSpace {
  YuvRange range = YuvRange::kStudioToFull;
  YuvMatrix matrix = YuvMatrix::kBt601;

  constexpr YuvColorSpace() noexcept = default;
  constexpr YuvColorSpace(YuvRange r, YuvMatrix m = YuvMatrix::kBt601) noexcept  // NOLINT: implicit
      : range(r), matrix(m) {}
};

/// Result of an aspect-preserving "letterbox" fit of a source image of size
/// (srcW, srcH) into a model input canvas of size (dstW, dstH).
///
/// A single uniform `scale` is applied to both axes; the scaled image is then
/// centered in the canvas with `padX`/`padY` pixels of border on each leading
/// edge. This is the standard YOLO preprocessing geometry — the inverse map
/// below is what post-processing uses to project boxes from model-input
/// coordinates back to original-image pixels.
struct LetterboxInfo {
  float scale = 1.0f;  ///< srcPix * scale -> dstPix (same on x and y)
  float padX = 0.0f;   ///< left padding added in the canvas (model coords)
  float padY = 0.0f;   ///< top padding added in the canvas (model coords)
  int srcW = 0;
  int srcH = 0;
  int dstW = 0;
  int dstH = 0;

  /// Forward map: original-image pixel -> model-input pixel.
  float fwdX(float x) const { return x * scale + padX; }
  float fwdY(float y) const { return y * scale + padY; }

  /// Inverse map: model-input pixel -> original-image pixel (un-letterbox).
  float invX(float x) const { return (x - padX) / scale; }
  float invY(float y) const { return (y - padY) / scale; }

  /// Clamp an original-image x/y to the valid source range.
  float clampX(float x) const { return std::min(std::max(x, 0.0f), static_cast<float>(srcW)); }
  float clampY(float y) const { return std::min(std::max(y, 0.0f), static_cast<float>(srcH)); }
};

/// Compute the letterbox geometry to fit (srcW,srcH) into (dstW,dstH).
///
/// `centerPad` centers the scaled image (YOLOv5/v8 default); when false the
/// image is pinned to the top-left (padX=padY=0), matching some exporters.
///
/// RGA NOTE: the hardware writes integer destination rectangles, so the RGA
/// letterbox rounds (padX,padY,newW,newH) to integers. Rounding is done by the
/// RGA wrapper and reflected back into the returned LetterboxInfo, so the
/// inverse map post-processing uses always matches the pixels the NPU saw.
inline LetterboxInfo computeLetterbox(int srcW, int srcH, int dstW, int dstH,
                                      bool centerPad = true) {
  LetterboxInfo lb;
  lb.srcW = srcW;
  lb.srcH = srcH;
  lb.dstW = dstW;
  lb.dstH = dstH;
  if (srcW <= 0 || srcH <= 0) return lb;
  lb.scale = std::min(static_cast<float>(dstW) / srcW, static_cast<float>(dstH) / srcH);
  const float newW = srcW * lb.scale;
  const float newH = srcH * lb.scale;
  if (centerPad) {
    lb.padX = (dstW - newW) * 0.5f;
    lb.padY = (dstH - newH) * 0.5f;
  }
  return lb;
}

}  // namespace rcdl
