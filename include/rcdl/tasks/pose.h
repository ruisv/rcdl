#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rcdl/preproc/geometry.h"
#include "rcdl/tasks/detection.h"  // Detection / iou / nms — a pose box IS a Detection

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// One skeleton joint, in ORIGINAL-image pixel coordinates (i.e. already
/// un-letterboxed back from the model-input canvas).
struct Keypoint {
  float x;
  float y;
  float score;  ///< visibility / confidence in [0,1]
};

/// One detected pose.
///
/// The person box is a plain `Detection`, not a copy of its four fields: the
/// pose head is the detection head plus a keypoint branch, so everything that
/// already consumes a Detection (iou(), nms(), the trackers, a drawing helper)
/// works on `box` unchanged.
struct PoseDetection {
  Detection box;                    ///< person box, original-image pixels
  std::vector<Keypoint> keypoints;  ///< num_keypoints joints, original-image pixels
};

// ===========================================================================
// Anchor-free LTRB multi-scale pose head
// ===========================================================================
//
// A pose export is the LTRB detection head of detection.h with one extra branch.
// Per feature scale (strides {8,16,32}, grids {80,40,20} at 640) it emits the
// familiar box (4 or 4*reg_max channels) and cls (nc channels, nc == 1 for the
// single "person" class), plus a keypoint branch of num_keypoints*3 values per
// cell laid out (x, y, visibility) per joint.
//
// TWO HEAD SHAPES, both supported and both resolved from the model:
//   FUSED (what the deployed .rknn actually is, and the default the config
//   describes) — box and cls concatenated into ONE tensor per scale, box
//   channels first, and the keypoints as a single tensor covering every anchor:
//       [1, 65, 80, 80] [1, 65, 40, 40] [1, 65, 20, 20]   65 = 64 DFL + 1 class
//       [1, 17,  3, 8400]                                 8400 = 80²+40²+20²
//   PER-BRANCH — box and cls as separate tensors per scale (what the
//   split-branch detection export of detection.h looks like), optionally with
//   per-scale keypoint tensors instead of the shared one.
// Note the fused export does NOT fold the class sigmoid into the graph the way
// the split-branch detection export does, hence apply_sigmoid defaulting to true.
//
// Box decode is bit-for-bit the detection head's:
//   cx = gx + 0.5, cy = gy + 0.5                        (grid units)
//   x1 = (cx - left) * s   ...   y2 = (cy + bottom) * s (model-input pixels)
//
// The keypoint branch is where exports differ, hence KeypointDecode below.
// WHERE THE ACTIVATIONS LIVE (verified against the vendor model-zoo reference
// decoder for the deployed export):
//   - class score : raw logit, sigmoid applied HERE (`apply_sigmoid`, default
//     true). Note this is the opposite of the split-branch detection export,
//     which bakes the class sigmoid into the graph.
//   - keypoint xy : decoded INSIDE the graph for the deployed export — the
//     tensor already holds model-input pixels (kModelPixels). A raw head emits
//     cell-relative values instead (kCellRelative).
//   - keypoint visibility : sigmoid INSIDE the graph for the deployed export
//     (`kpt_apply_sigmoid`, default false); a raw head emits the logit.

/// How the keypoint branch encodes a joint's position.
enum class KeypointDecode {
  /// The graph already did the decode: the two channels are model-input pixels
  /// and are used as-is. This is what the deployed export emits — the exporter
  /// keeps the cheap per-anchor arithmetic on the NPU and only the DFL reduction
  /// (which quantizes badly) on the CPU.
  kModelPixels,
  /// Raw head: the two channels are cell-relative and are decoded here as
  ///   kx = (2*raw_x + gx) * stride,  ky = (2*raw_y + gy) * stride
  /// which is `(raw*2 + (anchor - 0.5)) * stride` for the cell-center anchor
  /// (anchor = gx + 0.5). Note the doubling, and note that the keypoint ADDS the
  /// grid where the box SUBTRACTS a distance from it.
  kCellRelative,
};

/// Post-processing parameters for the LTRB multi-scale pose head.
struct PoseConfig {
  /// Pose exports are single-class ("person"); kept general because the decode
  /// is an argmax either way.
  int num_classes = 1;
  /// Joints per pose (17 for COCO). PoseEstimator overrides this from the
  /// keypoint tensor's own channel count; it is the fallback for the raw
  /// decodePose() entry point.
  int num_keypoints = 17;
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  int max_dets = 300;
  /// One stride per scale, ordered to match the per-scale buffers. Grid sizes
  /// come from `grid_hw`, so only the stride is needed here.
  std::vector<int> strides = {8, 16, 32};
  /// 0 => plain LTRB (4 box channels). >0 => 4*reg_max DFL logits per cell,
  /// side-major, reduced by Σ b·softmax(b) exactly as decodeYoloLtrb does.
  int reg_max = 0;
  /// Channel order of the cls/box/kpt buffers: true => [C,H,W], false => [H,W,C].
  /// Taken from each output's rknn fmt by PoseEstimator, never assumed.
  bool channels_first = true;
  /// Apply sigmoid to the class score. True for the deployed fused export, whose
  /// reference decoder thresholds against unsigmoid(conf) and then sigmoids.
  bool apply_sigmoid = true;
  KeypointDecode kpt_decode = KeypointDecode::kModelPixels;
  /// Apply sigmoid to the keypoint visibility channel. False for the deployed
  /// export (the graph already did it), true for a raw head.
  bool kpt_apply_sigmoid = false;
};

/// Where one scale's keypoint values live inside the keypoint tensor.
///
/// The real layouts differ only in two strides, so the decoder takes them rather
/// than a bool that cannot express all of them:
///   shared over ALL anchors           : cell_step 1,     chan_step A
///     (`data` points at this scale's first anchor, i.e. base + anchor_offset)
///   per-scale [K*3,H,W]               : cell_step 1,     chan_step H*W
///   per-scale [H,W,K*3]               : cell_step K*3,   chan_step 1
/// Channels are joint-major: channel 3*j+0/+1/+2 is joint j's x / y / visibility.
///
/// The shared case is spelled [1,K*3,A] by some exports and [1,K,3,A] by others
/// (the deployed one) — same buffer, same joint-major order, same anchor-last
/// axis, so the same two strides serve both. resolvePoseHead therefore reads
/// that branch by ELEMENT COUNT (K*3*A) rather than trusting the axis split.
struct KeypointPlane {
  const float* data = nullptr;
  std::int64_t cell_step = 0;  ///< elements between consecutive cells (row-major H*W)
  std::int64_t chan_step = 0;  ///< elements between consecutive channels of one cell
};

/// Decode parallel per-scale cls/box buffers plus keypoint planes into poses.
///
/// `cls[i]`     : row-major class scores for scale i — [nc,H,W] when
///                cfg.channels_first, else [H,W,nc].
/// `box[i]`     : row-major LTRB — [4,H,W] / [4*reg_max,H,W], or the channels-last
///                transposes. DFL channels are side-major.
/// `kpt[i]`     : scale i's view of the keypoint tensor (see KeypointPlane).
/// `grid_hw[i]` : the (H_i, W_i) of scale i.
/// `lb`         : letterbox geometry; box corners AND keypoints are mapped back
///                to original-image pixels via lb.invX/invY and clamped to the
///                source extent — a joint the model places outside the frame is
///                pinned to the border, and its `score` is the signal for
///                whether it is really there. Runs threshold -> box NMS (the
///                shared rcdl::nms) -> max_dets truncation across ALL scales.
/// The four input vectors plus cfg.strides must share the same length.
std::vector<PoseDetection> decodePose(const std::vector<const float*>& cls,
                                      const std::vector<const float*>& box,
                                      const std::vector<KeypointPlane>& kpt,
                                      const std::vector<std::pair<int, int>>& grid_hw,
                                      const PoseConfig& cfg, const LetterboxInfo& lb);

/// Convenience overload for the per-scale keypoint layout: `kpt[i]` is one
/// scale's own [K*3,H,W] (cfg.channels_first) or [H,W,K*3] buffer, and the two
/// strides are derived from cfg.num_keypoints and the grid.
std::vector<PoseDetection> decodePose(const std::vector<const float*>& cls,
                                      const std::vector<const float*>& box,
                                      const std::vector<const float*>& kpt,
                                      const std::vector<std::pair<int, int>>& grid_hw,
                                      const PoseConfig& cfg, const LetterboxInfo& lb);

/// Which Engine outputs hold one scale's tensors, plus that scale's grid.
struct PoseScaleOutputs {
  /// cls_index == box_index when the export fuses both into one tensor; the box
  /// channels come first, so `box_channels` also says where the class block
  /// starts.
  int cls_index = -1;
  int box_index = -1;
  int kpt_index = -1;  ///< -1 when one keypoint tensor is shared by all scales
  int grid_h = 0;
  int grid_w = 0;
  int num_classes = 0;
  int box_channels = 0;   ///< 4 (plain LTRB) or 4*reg_max (DFL)
  int kpt_channels = 0;   ///< num_keypoints * 3
  int anchor_offset = 0;  ///< this scale's first cell inside a shared keypoint tensor
};

/// Resolved description of a whole pose head, as read from an Engine.
struct PoseHeadLayout {
  std::vector<PoseScaleOutputs> scales;  ///< ordered by DESCENDING grid size (stride 8 first)
  std::vector<int> strides;              ///< derived as input_h / grid_h per scale
  int num_classes = 0;
  int num_keypoints = 0;
  int reg_max = 0;  ///< 0 => plain LTRB
  bool channels_first = true;
  bool fused_box_cls = false;   ///< one [1, 4*reg_max+nc, H, W] tensor per scale
  int shared_kpt_index = -1;    ///< >= 0 when one [1,K*3,A] tensor covers every scale
  int total_anchors = 0;        ///< Σ H*W over the scales
  std::string describe() const;
};

/// Work out a pose head's layout from the Engine's output signature alone.
///
/// This is resolveYoloHead()'s idea extended for the extra branch, and the extra
/// branch is exactly what makes plain grouping-by-grid insufficient: the shared
/// keypoint tensor of the deployed export covers ALL anchors at once, so it has
/// no grid of its own and would either form a bogus scale group or be mistaken
/// for a wide class branch. The discriminator is the model input: a tensor is a
/// feature-scale branch only if (H,W) is the input divided by one common integer
/// stride. Whatever is left over is the keypoint branch, and it is shared when
/// its per-channel extent equals the total anchor count.
///
/// Within a scale group the branches are told apart by channel count — a
/// multiple of 3 that is >= 9 is the keypoint branch (a box head is 4 or 4*reg_max
/// and a pose class head is 1 channel, so neither collides), a single tensor is
/// the fused box+cls export (its channel count minus `num_classes` is the box
/// width), and otherwise the wider-than-`num_classes` one is the box. `num_keypoints` may be passed to pin K when a model is unusual;
/// pass 0 to take it from the tensor.
///
/// Throws rcdl::Error with the model's full output signature when the outputs do
/// not form a pose head, so a mismatched model fails at construction instead of
/// decoding garbage.
PoseHeadLayout resolvePoseHead(const Engine& engine, int num_classes = 1,
                               int num_keypoints = 0);

/// Engine-bound LTRB multi-scale pose estimator.
///
/// Construction resolves the head from the Engine, so grids, class count,
/// keypoint count, DFL reg_max, channel order and strides all come from the
/// model; `cfg` supplies the thresholds and the activation conventions.
class PoseEstimator {
 public:
  explicit PoseEstimator(Engine& engine, PoseConfig cfg = PoseConfig());

  std::vector<PoseDetection> postprocess(const LetterboxInfo& lb) const;

  const PoseConfig& config() const { return cfg_; }
  const PoseHeadLayout& layout() const { return layout_; }

 private:
  Engine& engine_;
  PoseConfig cfg_;
  PoseHeadLayout layout_;
};

/// The 17 COCO keypoint names, in the order every pose export uses.
/// `cocoKeypointName` returns "keypoint <id>" for an out-of-range id.
const std::vector<std::string>& cocoKeypointNames();
const char* cocoKeypointName(int keypoint_id);

/// COCO-17 skeleton as 0-based (joint, joint) edges — the bone list a demo draws
/// so it does not have to carry its own copy of the topology.
const std::vector<std::pair<int, int>>& cocoSkeleton();

}  // namespace rcdl
