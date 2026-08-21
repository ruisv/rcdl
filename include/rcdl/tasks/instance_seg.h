#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rcdl/preproc/geometry.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// One instance-segmentation result. The box is in ORIGINAL-image pixels (already
/// un-letterboxed, exactly like rcdl::Detection) and so is the mask: `mask` covers
/// the window [mask_x0, mask_x0+mask_w) x [mask_y0, mask_y0+mask_h) of the SOURCE
/// image, row-major, one byte per pixel, 0 or 1.
///
/// The window is the whole source frame by default; set
/// InstanceSegConfig::full_frame_masks to false to get the instance's bounding
/// rectangle instead, which is all that is ever non-zero (the decoder crops each
/// mask to its own box) and is ~an order of magnitude less memory at 1080p.
struct InstanceMask {
  float x1;      ///< top-left x  (original-image pixels)
  float y1;      ///< top-left y
  float x2;      ///< bottom-right x
  float y2;      ///< bottom-right y
  float score;   ///< confidence in [0,1]
  int class_id;  ///< argmax class index
  int mask_x0;   ///< mask window origin in original-image pixels
  int mask_y0;
  int mask_w;    ///< mask window size; mask.size() == mask_w * mask_h
  int mask_h;
  std::vector<std::uint8_t> mask;  ///< 0/1 per pixel; empty when masks are off

  /// Full-frame view: the mask value at original-image pixel (x, y), 0 outside
  /// the stored window. Lets callers ignore the windowing when they do not care.
  std::uint8_t at(int x, int y) const {
    const int lx = x - mask_x0;
    const int ly = y - mask_y0;
    if (lx < 0 || ly < 0 || lx >= mask_w || ly >= mask_h) return 0;
    const std::size_t i = static_cast<std::size_t>(ly) * static_cast<std::size_t>(mask_w) +
                          static_cast<std::size_t>(lx);
    return i < mask.size() ? mask[i] : static_cast<std::uint8_t>(0);
  }
};

// ===========================================================================
// The head
// ===========================================================================
//
// A YOLO instance-seg export is the anchor-free LTRB detection head (see
// detection.h — same cell-centred decode, same DFL reduction, same per-class
// NMS) with two additions:
//
//   - one extra per-cell branch `mc` of `num_coef` MASK COEFFICIENTS, and
//   - one whole-image PROTOTYPE tensor of `num_coef` planes at 1/4 of the model
//     input resolution (160x160 for a 640 input).
//
// An instance's mask is the coefficient-weighted sum of the prototypes:
//
//     mask(y,x) = sigmoid( Σ_c coef[c] * proto[c](y,x) )       (prototype grid)
//
// upsampled to the model-input canvas, zeroed outside the instance's box, then
// stripped of the letterbox padding and resized to the source frame.
//
// The deployed export emits, per scale, four tensors: box (4 or 4*reg_max ch),
// cls (nc ch), a 1-channel per-cell score-sum used only as a cheap pre-filter
// (ignored here, as in the detector), and mc (num_coef ch); plus the single
// prototype tensor. Output ORDER is not relied on — resolveInstanceSegHead()
// works it out from the shapes.
//
// Two vendor-dependent conventions, both explicit config flags:
//
//   - CHANNEL ORDER. The model-zoo export is NCHW: the per-scale branches are
//     [1,C,H,W] and the prototype is [1,32,160,160]. Other exporters emit NHWC
//     ([1,H,W,C] / [1,160,160,32]). Take the answer from each output's
//     rknn_tensor_attr.fmt — that is what resolveInstanceSegHead() does — and
//     never from the dimension values, because a 160x160x32 tensor and a
//     32x160x160 one are distinguishable only by fmt.
//
//   - CLASS ACTIVATION. The model-zoo ONNX has the sigmoid on `cls` INSIDE the
//     graph, so `cls` is already a probability and apply_sigmoid stays false
//     (the default, matching YoloLtrbConfig). Raw-logit exports set it. Tell
//     them apart by dumping one cls tensor: probabilities live in [0,1] and a
//     background cell sits near 0, logits are unbounded and background cells sit
//     around -6..-10. The MASK COEFFICIENTS are always raw (linear) — the
//     sigmoid in the formula above is applied by this decoder, never by the
//     graph, so there is no flag for it.

/// Post-processing parameters for a YOLO instance-segmentation head.
struct InstanceSegConfig {
  int num_classes = 0;  ///< 0 => take it from the model (InstanceSegmenter only)
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  /// Instance-seg masks are expensive, so the default cap is lower than the
  /// detector's 300 — every kept instance costs one mask assembly.
  int max_dets = 100;
  /// One stride per scale, ordered to match the per-scale tensor groups.
  std::vector<int> strides = {8, 16, 32};
  /// DFL box head: 0 => plain LTRB (4 ch/cell), >0 => 4*reg_max ch/cell reduced
  /// by Σ b·softmax(b) per side. InstanceSegmenter derives this from the box
  /// tensor's channel count; the raw decodeInstanceSeg() entry point trusts it.
  int reg_max = 0;
  /// Channel order of the per-scale cls/box/mc buffers: true => [C,H,W].
  bool channels_first = true;
  /// Channel order of the PROTOTYPE buffer: true => [num_coef,H,W] (the
  /// model-zoo NCHW export), false => [H,W,num_coef]. Kept separate from
  /// `channels_first` because the two are resolved from different tensors'
  /// fmt fields and an export is free to disagree with itself.
  bool proto_channels_first = true;
  /// Apply sigmoid to the class scores. false when the export already did.
  bool apply_sigmoid = false;
  /// Binarisation threshold on the assembled (sigmoid) mask.
  float mask_thresh = 0.5f;
  /// Skip mask assembly entirely — boxes/scores/classes only, `mask` empty.
  /// Useful when the seg model is being used as a plain detector.
  bool compute_masks = true;
  /// true  => `mask` covers the whole source frame (mask_x0/y0 = 0);
  /// false => `mask` covers only the instance's clipped integer bounding box.
  bool full_frame_masks = true;
};

/// Decode parallel per-scale cls/box/mc buffers plus a prototype tensor into
/// instance masks (Engine-free; the reference the numpy tests mirror).
///
/// `cls[i]`     : row-major class scores  — [nc,H,W]  / [H,W,nc]
/// `box[i]`     : row-major LTRB          — [4,H,W]   / [H,W,4]   (or 4*reg_max)
/// `mc[i]`      : row-major mask coefs    — [np,H,W]  / [H,W,np]
/// `grid_hw[i]` : the (H_i, W_i) of scale i.
/// `num_classes`: trailing/leading class-channel count of every cls buffer.
/// `proto`      : row-major prototype — [np,proto_h,proto_w] when
///                cfg.proto_channels_first, else [proto_h,proto_w,np].
/// `proto_c`    : np — must equal the mask-coefficient count.
/// `lb`         : the letterbox geometry used at preprocess time. It carries the
///                source extent (srcW/srcH), so boxes AND masks come back in
///                original-image pixels without a second size argument.
///
/// Runs threshold -> per-class NMS (on model-input boxes, where IoU is identical
/// because the letterbox map is a uniform scale + translation) -> per-instance
/// mask assembly. cls/box/mc/grid_hw/cfg.strides must share one length.
std::vector<InstanceMask> decodeInstanceSeg(const std::vector<const float*>& cls,
                                            const std::vector<const float*>& box,
                                            const std::vector<const float*>& mc,
                                            const std::vector<std::pair<int, int>>& grid_hw,
                                            int num_classes, const float* proto, int proto_h,
                                            int proto_w, int proto_c,
                                            const InstanceSegConfig& cfg, const LetterboxInfo& lb);

/// Which Engine outputs hold one scale's cls / box / mc (/ optional score-sum).
struct InstanceSegScaleOutputs {
  int cls_index = -1;
  int box_index = -1;
  int mc_index = -1;
  int sum_index = -1;  ///< -1 when the export has no score-sum branch
  int grid_h = 0;
  int grid_w = 0;
  int num_classes = 0;
  int num_coef = 0;
  int box_channels = 0;  ///< 4 (plain LTRB) or 4*reg_max (DFL)
};

/// Resolved description of a whole instance-seg head, as read from an Engine.
struct InstanceSegHeadLayout {
  std::vector<InstanceSegScaleOutputs> scales;  ///< ordered by DESCENDING grid (stride 8 first)
  std::vector<int> strides;                     ///< derived as input_h / grid_h per scale
  int proto_index = -1;
  int proto_h = 0;
  int proto_w = 0;
  int num_classes = 0;
  int num_coef = 0;
  int reg_max = 0;  ///< 0 => plain LTRB
  bool channels_first = true;
  bool proto_channels_first = true;
  bool has_score_sum = false;
  std::string describe() const;
};

/// Work out an instance-seg head's layout from the Engine's output signature.
///
/// The prototype is the one output that does not share its (H,W) with any other
/// — every detection branch comes in a group of three or four. Its channel count
/// then identifies the `mc` branch inside each group (mask coefficients must
/// match the prototype planes for the matmul to exist at all), the 1-channel
/// tensor is the score-sum, the 4 / 4*reg_max one is the box head, and what is
/// left is cls. Pass `num_classes` to break the tie when the class count happens
/// to equal the coefficient count (a 32-class model on a 32-plane prototype).
///
/// Throws rcdl::Error with the full output signature when the outputs do not
/// form a consistent seg head, so a mismatched model fails at construction
/// instead of decoding garbage.
InstanceSegHeadLayout resolveInstanceSegHead(const Engine& engine, int num_classes = 0);

/// Engine-bound instance segmenter.
///
/// Construction resolves the head from the Engine, so grid sizes, class and
/// coefficient counts, DFL reg_max, both channel orders and the strides all come
/// from the model rather than from `cfg` — a mis-configured num_classes cannot
/// make the decoder index past a buffer. `cfg` supplies the thresholds, the
/// class-activation convention and the mask options.
class InstanceSegmenter {
 public:
  explicit InstanceSegmenter(Engine& engine, InstanceSegConfig cfg = InstanceSegConfig());

  std::vector<InstanceMask> postprocess(const LetterboxInfo& lb) const;

  const InstanceSegConfig& config() const { return cfg_; }
  const InstanceSegHeadLayout& layout() const { return layout_; }

 private:
  Engine& engine_;
  InstanceSegConfig cfg_;
  InstanceSegHeadLayout layout_;
};

}  // namespace rcdl
