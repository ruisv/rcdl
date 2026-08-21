#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rcdl/preproc/geometry.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// What, if anything, to report alongside each pixel's class id.
enum class SegScore {
  /// No per-pixel confidence — argmax only. Cheapest, and the default: a label
  /// map needs one comparison per channel and no exp() at all.
  kNone,
  /// softmax(logits)[winner] in [0,1]. The right choice for an export that emits
  /// raw LOGITS (the common case: the softmax is a no-op for argmax, so
  /// exporters usually strip it).
  kSoftmax,
  /// The winning channel's value as-is. The right choice when the export already
  /// has the softmax in the graph, so the values are probabilities. Tell the two
  /// apart by dumping one pixel: probabilities are non-negative and sum to 1
  /// across channels, logits are unbounded and do not.
  kMax,
};

/// Post-processing parameters for a dense semantic-segmentation head.
struct SegConfig {
  /// Channel count C. 0 => take it from the tensor shape, which is what you
  /// want; a non-zero value only ever NARROWS the argmax (it is clamped to what
  /// the buffer actually holds, never widened).
  int num_classes = 0;
  /// Logit layout: [1,C,H,W] (true) vs [1,H,W,C] (false). Segmenter takes this
  /// from the output's rknn_tensor_attr.fmt rather than guessing — a
  /// [1,21,513,513] tensor and a [1,513,513,21] one are the same bytes in a
  /// different order, and only fmt says which.
  bool channels_first = true;
  /// Set when the model already emits per-pixel class ids ([1,H,W] / [H,W],
  /// integer or float). decodeSeg then rounds to int instead of argmaxing.
  bool argmaxed = false;
  SegScore score = SegScore::kNone;
};

/// A decoded label map. `labels` is row-major height*width class ids;
/// `confidence` is the matching per-pixel score, empty unless SegConfig::score
/// asked for one.
struct SegMask {
  int width = 0;
  int height = 0;
  int num_classes = 0;
  std::vector<std::int32_t> labels;
  std::vector<float> confidence;

  std::int32_t labelAt(int x, int y) const {
    if (x < 0 || y < 0 || x >= width || y >= height) return -1;
    return labels[static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                  static_cast<std::size_t>(x)];
  }
};

/// Argmax a logit volume (or pass through an id map) into a SegMask.
///
/// `data`  : product(shape) floats in logical row-major order.
/// `shape` : {1,C,H,W} / {C,H,W} / {1,H,W,C} / {H,W,C} logits, or {1,H,W} /
///           {H,W} class ids when cfg.argmaxed.
///
/// TIE-BREAKING: the scan keeps a channel only on a STRICTLY greater value, so
/// the LOWEST channel index wins a tie. That matters more than it looks — an
/// int8-quantized head saturates whole regions to the same code, so ties are
/// common rather than exotic, and numpy's argmax breaks them the same way.
SegMask decodeSeg(const float* data, const std::vector<int>& shape, const SegConfig& cfg);

/// Nearest-neighbour resample of a label map to (dst_w, dst_h).
///
/// Nearest, never bilinear: class ids are nominal, and interpolating id 3 and
/// id 7 into 5 invents a class that was never predicted. `confidence` (when
/// present) is resampled with the SAME map so a pixel's label and score keep
/// belonging to each other.
///
/// The map is the pixel-CENTRE one, src_x = floor((dst_x + 0.5) * src_w/dst_w),
/// which is symmetric under an exact integer up- or down-scale — unlike the
/// truncating floor(dst_x * src_w/dst_w) rule, which biases every pixel towards
/// the top-left by half a destination pixel.
SegMask segResize(const SegMask& m, int dst_w, int dst_h);

/// Project a label map produced from a LETTERBOXED input back onto the original
/// frame: strips the padding and resamples to (lb.srcW, lb.srcH).
///
/// `m` is assumed to cover the whole model-input canvas (lb.dstW x lb.dstH),
/// whatever its own resolution. Rather than crop-then-resize, each SOURCE pixel
/// centre is mapped forward through the letterbox and read back from the mask —
/// one nearest lookup, no intermediate image, and correct even when the head's
/// output resolution is not a divisor of the canvas:
///
///     mx = round( lb.fwdX(x + 0.5) * m.width  / lb.dstW - 0.5 )
///     my = round( lb.fwdY(y + 0.5) * m.height / lb.dstH - 0.5 )
///
/// (the -0.5/+0.5 pair is the same pixel-centre convention as segResize; padding
/// falls outside the source extent by construction, so it never contributes).
SegMask segToSource(const SegMask& m, const LetterboxInfo& lb);

/// Colour a label map with a fixed deterministic palette -> BGR, height*width*3
/// bytes, row-major (OpenCV's channel order, so it can be written or blended
/// straight into a cv::Mat / an RGA buffer).
///
/// Ids 0..20 use the classic PASCAL VOC palette (id 0 = black background), which
/// is what every VOC-trained segmentation demo renders; higher ids fall back to
/// a golden-ratio hue walk so neighbouring ids stay visually distinct.
std::vector<std::uint8_t> segColorize(const SegMask& m);

/// The 21 PASCAL VOC class names in label order (0 = "background") — the labels
/// a VOC-trained head's ids refer to. `vocClassName` returns "class <id>" for an
/// out-of-range id rather than throwing.
const std::vector<std::string>& vocClassNames();
const char* vocClassName(int class_id);

/// Engine-bound semantic segmenter.
///
/// The caller preprocesses + infer()s; postprocess() reads the selected output
/// and runs decodeSeg(). Channel order and class count come from the output's
/// own attributes, so `cfg`'s only job is the score mode (and, if you insist, a
/// narrowing num_classes).
///
/// It also takes a quantized fast path — see the note in the .cc — that argmaxes
/// int8/uint8 affine output codes directly, skipping the dequant of a volume
/// that can be tens of millions of floats.
class Segmenter {
 public:
  explicit Segmenter(Engine& engine, SegConfig cfg = SegConfig(), int output_index = 0);

  /// Label map at the model's own output resolution.
  SegMask postprocess() const;
  /// Label map projected back onto the original frame (postprocess + segToSource).
  SegMask postprocess(const LetterboxInfo& lb) const;

  const SegConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  SegConfig cfg_;
  int out_idx_;
};

}  // namespace rcdl
