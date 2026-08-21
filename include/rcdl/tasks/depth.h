#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rcdl/preproc/geometry.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Monocular depth
// ===========================================================================
//
// A monocular-depth head is one dense single-channel map, and it comes with
// three variables a decoder has to be told about rather than guess:
//
//   * LAYOUT. The map arrives as [1,1,H,W] (NCHW) or [1,H,W,1] (NHWC) — and with
//     a single channel those two are byte-for-byte the SAME row-major H*W plane,
//     so unlike a seg logit volume this one needs no channel-order flag. Shapes
//     of {1,H,W} and {H,W} occur too and mean the same thing.
//
//   * RESOLUTION. The map is usually smaller than the model input (a 518-input
//     transformer head often emits 1/14 of that), so it has to be resized to the
//     source frame — through the letterbox, if the input was letterboxed.
//
//   * UNITS. Relative-depth models (the common self-supervised / distilled
//     family) emit INVERSE depth, i.e. disparity: big = near. Their scale is
//     arbitrary, which is why the usual presentation is a min-max normalise to
//     [0,1]. Metric models emit real distances, often in millimetres, in which
//     case normalising throws the information away — turn it off and use
//     `scale`/`shift` to land in the units you want.
//
// Precision: these heads are frequently exported as float (the toolkit leaves a
// depth head in fp16/fp32 far more often than a classifier), but the quantized
// case is normal too; outputAsFloat() covers both, so nothing here cares.

/// Post-processing parameters for a single-channel dense depth / disparity head.
struct DepthConfig {
  int width = 0;   ///< output map width  (0 => infer from the tensor shape)
  int height = 0;  ///< output map height (0 => infer)

  /// Raw -> physical units, applied FIRST:  v = raw * scale + shift.
  /// Identity by default. A metric head reporting millimetres becomes metres
  /// with scale = 0.001f.
  float scale = 1.0f;
  float shift = 0.0f;

  /// The head emits INVERSE depth (disparity, big = near). true => take
  /// v = 1 / max(v, inverse_eps) after the affine above, turning the map into a
  /// depth (big = far) before clipping and normalising. Leave false to keep a
  /// relative-depth map as the disparity it is — which is what visualisations
  /// want, since disparity is what the model was trained to be smooth in.
  bool inverse = false;
  float inverse_eps = 1e-6f;

  /// Optional clip applied after the affine/inverse and BEFORE vmin/vmax are
  /// observed. Active only when clip_hi > clip_lo; leave both 0 to disable.
  /// Its main job is keeping a handful of wild pixels from flattening the whole
  /// min-max normalisation below.
  float clip_lo = 0.0f;
  float clip_hi = 0.0f;

  /// Min-max normalise the map into [0,1] using the values actually observed in
  /// THIS map. Right for a relative-depth head, wrong for a metric one (it is
  /// per-frame, so the mapping changes from frame to frame). A flat map yields
  /// all zeros rather than a division by zero.
  bool normalize = true;
};

/// A decoded depth map, row-major height*width.
///
/// `data` is normalised to [0,1] when DepthConfig::normalize, otherwise it holds
/// the processed values as they are. `vmin`/`vmax` are the raw range actually
/// observed (after clip, before normalise), which is what a caller needs to undo
/// the normalisation or to colourise consistently across frames.
struct DepthMap {
  int width = 0;
  int height = 0;
  std::vector<float> data;
  float vmin = 0.0f;
  float vmax = 0.0f;

  float at(int x, int y) const {
    if (x < 0 || y < 0 || x >= width || y >= height) return 0.0f;
    return data[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                static_cast<std::size_t>(x)];
  }
};

/// Decode a single-channel float tensor into a DepthMap.
///
/// `data`  : product(shape) floats, row-major.
/// `shape` : {1,1,H,W}, {1,H,W,1}, {1,H,W} or {H,W} — all the same H*W plane.
/// Pipeline: affine (scale/shift) -> optional inverse -> optional clip ->
/// observe vmin/vmax -> optional min-max normalise.
DepthMap decodeDepth(const float* data, const std::vector<int>& shape, const DepthConfig& cfg);

/// Bilinear resample of a depth map to (dst_w, dst_h).
///
/// Bilinear, not nearest: depth is a continuous quantity, so interpolating
/// between two samples is meaningful (the opposite of a label map). Uses the
/// pixel-centre map src = (dst + 0.5) * src_n/dst_n - 0.5 clamped to the source,
/// matching the mask resizes in the instance-seg decoder. vmin/vmax carry over
/// unchanged — interpolation cannot leave the convex hull of the samples.
DepthMap depthResize(const DepthMap& m, int dst_w, int dst_h);

/// Project a depth map produced from a LETTERBOXED input back onto the original
/// frame: strips the padding and resamples to (lb.srcW, lb.srcH).
///
/// `m` is assumed to cover the whole model-input canvas (lb.dstW x lb.dstH),
/// whatever its own resolution. Each SOURCE pixel centre is mapped forward
/// through the letterbox and sampled back out of the map, so the padding never
/// contributes and no intermediate image is built:
///
///     mx = lb.fwdX(x + 0.5) * m.width  / lb.dstW - 0.5      (then bilinear)
///     my = lb.fwdY(y + 0.5) * m.height / lb.dstH - 0.5
DepthMap depthToSource(const DepthMap& m, const LetterboxInfo& lb);

/// Min-max stretch of a depth map to uint8 [0,255], row-major height*width.
/// Uses the map's OWN observed range, so it renders sensibly whether `data` is
/// already normalised or still in raw units. A flat map comes out all zeros.
std::vector<std::uint8_t> depthToGray8(const DepthMap& m);

/// Turbo colourmap -> BGR, height*width*3 bytes (OpenCV channel order). The
/// curve is a compact polynomial approximation of Google's Turbo palette: good
/// enough to look at, not colorimetrically exact.
std::vector<std::uint8_t> depthColorize(const DepthMap& m);

/// Engine-bound depth estimator.
///
/// The caller preprocesses + infer()s; postprocess() reads the selected output
/// (zero-copy for a packed float head, dequantized otherwise) and runs
/// decodeDepth().
class DepthEstimator {
 public:
  explicit DepthEstimator(Engine& engine, DepthConfig cfg = DepthConfig(), int output_index = 0);

  /// Depth map at the model's own output resolution.
  DepthMap postprocess() const;
  /// Depth map projected back onto the original frame (postprocess + depthToSource).
  DepthMap postprocess(const LetterboxInfo& lb) const;

  const DepthConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  DepthConfig cfg_;
  int out_idx_;
};

}  // namespace rcdl
