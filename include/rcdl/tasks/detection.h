#pragma once

#include <string>
#include <utility>
#include <vector>

#include "rcdl/preproc/geometry.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// One detected object, expressed in ORIGINAL-image pixel coordinates
/// (i.e. already un-letterboxed back from the model-input canvas).
struct Detection {
  float x1;      ///< top-left x
  float y1;      ///< top-left y
  float x2;      ///< bottom-right x
  float y2;      ///< bottom-right y
  float score;   ///< confidence in [0,1]
  int class_id;  ///< argmax class index
};

/// Intersection-over-union of two axis-aligned boxes.
float iou(const Detection& a, const Detection& b);

/// Classic greedy per-class NMS. Sorts `dets` by score descending, suppresses
/// any lower-scoring box of the SAME class whose IoU with a kept box exceeds
/// `iou_thresh`, and returns the kept indices (into `dets`) truncated to
/// `max_dets`.
std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh, int max_dets);

// ===========================================================================
// Fused single-tensor head  ([1, 4+nc, N] / [1, N, 4+nc])
// ===========================================================================

/// Raw-tensor memory layout / decode family for the fused head.
enum class DecodeLayout {
  /// Anchor-free YOLOv8/v11 head exported WITH the concat+DFL in the graph.
  /// Output is [1, 4+nc, N] (channels_first) or [1, N, 4+nc]. Each candidate is
  /// (cx,cy,w,h) in model-input pixels followed by `nc` class scores (already
  /// activated, or logits when `apply_sigmoid` is set). class_id = argmax.
  kYoloV8,
  /// YOLOv5-style head with objectness: (cx,cy,w,h, obj, nc class scores).
  /// Final score = obj * max(class).
  kYoloV5,
};

/// Post-processing parameters for a fused single-tensor detection head.
struct DetectConfig {
  int input_w = 640;  ///< model input width  (letterbox canvas)
  int input_h = 640;  ///< model input height (letterbox canvas)
  int num_classes = 80;
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  int max_dets = 300;
  DecodeLayout layout = DecodeLayout::kYoloV8;
  /// true => [1, attrs, N], false => [1, N, attrs].
  bool channels_first = true;
  /// Apply sigmoid to class (and objectness) scores before thresholding. Set
  /// when the exported head emits logits rather than probabilities.
  bool apply_sigmoid = false;
};

/// Decode a contiguous (row-major) float output tensor into final Detections.
///
/// `data`  : product(shape) floats in logical row-major order.
/// `shape` : tensor shape, e.g. {1, 4+nc, N} or {1, N, 4+nc}.
/// `lb`    : letterbox geometry used at preprocess time; boxes are mapped back
///           to original-image pixels via lb.invX/invY and clamped to the source
///           extent. Runs threshold -> per-class NMS -> max_dets truncation.
std::vector<Detection> decode(const float* data, const std::vector<int>& shape,
                              const DetectConfig& cfg, const LetterboxInfo& lb);

/// Convenience wrapper binding an Engine output to a DetectConfig. The caller
/// owns preprocessing + infer; postprocess() reads the selected output through
/// outputAsFloat() (zero-copy for packed f32, dequant-into-scratch otherwise)
/// and runs decode().
class Detector {
 public:
  Detector(Engine& engine, DetectConfig cfg, int output_index = 0);

  std::vector<Detection> postprocess(const LetterboxInfo& lb) const;

  const DetectConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  DetectConfig cfg_;
  int out_idx_;
};

// ===========================================================================
// Anchor-free LTRB multi-scale head — the deployed RKNPU2 / RDK YOLO export
// ===========================================================================
//
// Both vendors' YOLO exports keep the per-scale branches SEPARATE rather than
// fusing them into one [1,4+nc,N] tensor, because the concat + DFL reduction
// quantizes badly. For each stride s (default {8,16,32}, grids {80,40,20} at
// 640) the graph emits:
//   - box : LTRB distances, 4 channels/cell, or 4*reg_max DFL logits (64 for
//           the ultralytics reg_max=16 head, reduced here on the CPU)
//   - cls : nc per-cell class scores
//   - sum : OPTIONAL 1-channel per-cell score sum, present in the rknn_model_zoo
//           YOLOv8/YOLO11 export as a cheap pre-filter — and used as one here,
//           since it bounds the class maximum from above (see decodeYoloLtrb).
//
// Boxes are anchor-free, decoded relative to the cell-center grid:
//   cx = gx + 0.5,  cy = gy + 0.5                (grid units)
//   x1 = (cx - left) * s,   y1 = (cy - top)    * s
//   x2 = (cx + right) * s,  y2 = (cy + bottom) * s     (model-input pixels)
//
// The two vendors differ in TWO ways, both handled here:
//   - channel order: rknn_model_zoo emits NCHW ([1,C,H,W]); the RDK export
//     emits NHWC ([1,H,W,C]).  -> YoloLtrbConfig::channels_first
//   - class activation: the rknn_model_zoo ONNX has the sigmoid INSIDE the
//     graph (cls is already a probability); the RDK export emits raw logits.
//     -> YoloLtrbConfig::apply_sigmoid

/// Post-processing parameters for the LTRB multi-scale head.
struct YoloLtrbConfig {
  int num_classes = 80;
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  int max_dets = 300;
  /// One stride per scale, ordered to match the per-scale output groups. The
  /// default {8,16,32} is the standard P3/P4/P5 head; grid sizes come from each
  /// output's own shape, so only the stride is needed here.
  std::vector<int> strides = {8, 16, 32};
  /// DFL box head: 0 => plain LTRB (box tensor has 4 channels, distances read
  /// directly). >0 => the box tensor has `4*reg_max` channels; each side's
  /// `reg_max` raw logits are softmaxed and reduced by Σ b·softmax(b) into one
  /// distance. YoloLtrbDetector auto-detects this from the box channel count.
  int reg_max = 0;
  /// Channel order of the cls/box buffers: true => [C,H,W] (rknn_model_zoo),
  /// false => [H,W,C] (RDK / channels-last exports).
  bool channels_first = true;
  /// Apply sigmoid to the class scores. false when the export already did
  /// (rknn_model_zoo YOLOv8/YOLO11), true for raw-logit exports.
  bool apply_sigmoid = false;
};

/// Decode parallel per-scale float cls/box buffers into final Detections.
///
/// `cls[i]`     : row-major class scores for scale i — [nc,H,W] when
///                cfg.channels_first, else [H,W,nc].
/// `box[i]`     : row-major LTRB — [4,H,W] / [4*reg_max,H,W] when
///                cfg.channels_first, else [H,W,4] / [H,W,4*reg_max]. For the
///                DFL layout the 4*reg_max channels are side-major
///                (side 0's reg_max bins, then side 1's, ...), which is what
///                both ultralytics and the rknn_model_zoo export produce.
/// `grid_hw[i]` : the (H_i, W_i) of scale i.
/// `lb`         : letterbox geometry; boxes are mapped back to original-image
///                pixels and clamped. Runs threshold -> per-class NMS ->
///                max_dets truncation across ALL scales.
/// The three input vectors plus cfg.strides must share the same length.
/// `score_sum[i]` is OPTIONAL: the export's per-cell score-sum branch, one value
/// per cell, which the rknn_model_zoo YOLO models emit purely as a cheap
/// pre-filter. Because every class score is non-negative, the sum bounds the
/// maximum from above — so a cell whose sum falls below `conf_thresh` cannot
/// contain a detection, and its `num_classes`-wide argmax can be skipped
/// outright. That argmax is the whole cost of this function: 8400 cells x 80
/// classes with a stride of H*W between channels, which measured at 30 ms per
/// frame, as expensive as the inference it follows. Pass the branch and it
/// drops by roughly an order of magnitude. Leave it empty and every cell is
/// scanned, which is what a head without the branch requires.
std::vector<Detection> decodeYoloLtrb(const std::vector<const float*>& cls,
                                      const std::vector<const float*>& box,
                                      const std::vector<std::pair<int, int>>& grid_hw,
                                      const YoloLtrbConfig& cfg, const LetterboxInfo& lb,
                                      const std::vector<const float*>& score_sum = {});

/// Which Engine outputs hold one scale's cls / box (/ optional score-sum)
/// tensors, plus that scale's grid.
struct YoloScaleOutputs {
  int cls_index = -1;
  int box_index = -1;
  int sum_index = -1;  ///< -1 when the export has no score-sum branch
  int grid_h = 0;
  int grid_w = 0;
  int num_classes = 0;
  int box_channels = 0;  ///< 4 (plain LTRB) or 4*reg_max (DFL)
};

/// Resolved description of a whole LTRB head, as read from an Engine.
struct YoloHeadLayout {
  std::vector<YoloScaleOutputs> scales;  ///< ordered by DESCENDING grid size (stride 8 first)
  std::vector<int> strides;              ///< derived as input_h / grid_h per scale
  int num_classes = 0;
  int reg_max = 0;         ///< 0 => plain LTRB
  bool channels_first = true;
  bool has_score_sum = false;
  std::string describe() const;
};

/// Work out an LTRB head's layout from the Engine's output signature alone.
///
/// Outputs are grouped by their (H,W) grid; within a group the tensors are told
/// apart by channel count: 1 => score-sum, 4 or a multiple of 4 that is not the
/// class count => box, everything else => cls. When `num_classes` is > 0 it
/// disambiguates the box/cls pair directly (a 4-class model whose box head is
/// also 4 channels is otherwise ambiguous); pass 0 to rely on the heuristic.
/// Scales are ordered by descending grid size and strides derived from the
/// model's input height. Channel order comes from the outputs' rknn fmt.
///
/// Throws rcdl::Error with a description of what it saw when the outputs do not
/// form a consistent LTRB head — so a mismatched model fails loudly at
/// construction instead of decoding garbage.
YoloHeadLayout resolveYoloHead(const Engine& engine, int num_classes = 0);

/// Engine-bound LTRB multi-scale detector.
///
/// Construction resolves the head layout from the Engine (resolveYoloHead), so
/// the grid sizes, class count, DFL reg_max, channel order and strides all come
/// from the model rather than from `cfg` — a mis-configured num_classes cannot
/// make the decoder index past a buffer. `cfg` supplies the thresholds; its
/// `strides` are used only when the model's input height does not divide evenly
/// into the grids, and `apply_sigmoid` is honoured as given.
class YoloLtrbDetector {
 public:
  explicit YoloLtrbDetector(Engine& engine, YoloLtrbConfig cfg = YoloLtrbConfig());

  std::vector<Detection> postprocess(const LetterboxInfo& lb) const;

  const YoloLtrbConfig& config() const { return cfg_; }
  const YoloHeadLayout& layout() const { return layout_; }

 private:
  Engine& engine_;
  YoloLtrbConfig cfg_;
  YoloHeadLayout layout_;
};

/// The 80 COCO class names, in the order every YOLO export uses. `cocoClassName`
/// returns "class <id>" for an out-of-range id rather than throwing.
const std::vector<std::string>& cocoClassNames();
const char* cocoClassName(int class_id);

}  // namespace rcdl
