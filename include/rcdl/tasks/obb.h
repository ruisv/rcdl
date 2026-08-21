#pragma once

#include <string>
#include <utility>
#include <vector>

#include "rcdl/preproc/geometry.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// A rotated rectangle. (cx,cy) is the centre, (w,h) the side lengths along the
/// box's own axes, `angle` the rotation in RADIANS (positive = image x toward
/// image y, i.e. clockwise in the usual y-down image frame).
///
/// After decode these are ORIGINAL-image pixels: the centre is un-letterboxed
/// through lb.invX/invY and the sides are divided by the letterbox scale. The
/// angle needs no correction — the letterbox map is a uniform scale plus a
/// translation, which preserves angles.
struct RotatedBox {
  float cx;
  float cy;
  float w;
  float h;
  float angle;  ///< radians
};

/// One oriented detection, in original-image pixel coordinates.
struct ObbDetection {
  RotatedBox rrect;
  float score;   ///< class confidence in [0,1]
  int class_id;  ///< argmax class index
};

/// The box's 4 corners as x0,y0, x1,y1, x2,y2, x3,y3 — consecutive corners share
/// an edge, so the array draws directly as a closed polyline.
///
///   local corners (-dx,-dy) (dx,-dy) (dx,dy) (-dx,dy),  dx = w/2, dy = h/2
///   world  x = cx + lx*cos(a) - ly*sin(a),  y = cy + lx*sin(a) + ly*cos(a)
///
/// Corners of a decoded ObbDetection are therefore already in original-image
/// pixels, like every other coordinate this library hands back.
void rotatedBoxCorners(const RotatedBox& r, float out[8]);

/// Intersection-over-union of two rotated rectangles.
///
/// This is the one piece of OBB post-processing that axis-aligned detection has
/// no answer for: `iou()` from detection.h compares min/max extents, which is
/// simply wrong once the boxes are rotated. Instead we expand both boxes into
/// their 4 corners, clip one polygon against the other (Sutherland-Hodgman —
/// both are convex, so the intersection is a convex polygon of at most 8
/// vertices), take its shoelace area and divide by the union. Returns 0 for a
/// degenerate box or an empty intersection.
///
/// The clip runs at DOUBLE precision even though the boxes are float, and that
/// is load-bearing, not incidental: two boxes sharing an edge — a box against
/// itself during NMS, or two cells predicting one object at one angle — give
/// every vertex of that edge a signed distance of zero, and in float the
/// rounding is the same size as the quantity whose sign decides which side the
/// vertex falls on. Vertices then drop out, the clip closes into a
/// self-crossing polygon, and its shoelace area bears no relation to either box
/// (measured: 1.0 became -39.6). Boxes in general position are unaffected,
/// which is exactly why this has to be stated rather than discovered.
float rotatedIoU(const RotatedBox& a, const RotatedBox& b);

/// Greedy per-class rotated NMS — nms() from detection.h with rotatedIoU() in
/// place of iou(). Sorts `dets` by score descending, suppresses any lower-scoring
/// box of the SAME class whose rotated IoU with a kept box exceeds `iou_thresh`,
/// and returns the kept indices (into `dets`) truncated to `max_dets`.
std::vector<int> rotatedNms(const std::vector<ObbDetection>& dets, float iou_thresh,
                            int max_dets);

// ===========================================================================
// Anchor-free LTRB multi-scale OBB head
// ===========================================================================
//
// An OBB export is the LTRB detection head of detection.h with one extra branch.
// Per feature scale (strides {8,16,32}, grids {80,40,20} at 640) it emits box (4
// or 4*reg_max channels) and cls (nc channels), plus one angle value per cell.
//
// TWO HEAD SHAPES, both supported and both resolved from the model:
//   FUSED (what the deployed .rknn actually is, and the default the config
//   describes) — box and cls concatenated into ONE tensor per scale, box
//   channels first, and the angle as a single tensor covering every anchor:
//       [1, 79, 80, 80] [1, 79, 40, 40] [1, 79, 20, 20]   79 = 64 DFL + 15 classes
//       [1,  1, 8400]                                     8400 = 80²+40²+20²
//   PER-BRANCH — box, cls and angle as separate tensors per scale.
// Note the fused export does NOT fold the class sigmoid into the graph the way
// the split-branch detection export does, hence apply_sigmoid defaulting to true;
// the ANGLE, by contrast, is sigmoided in-graph (the branch is literally the
// graph's Sigmoid output), hence apply_angle_sigmoid defaulting to false.
//
// Decode (anchor-free about the cell-centre grid gx+0.5, gy+0.5):
//   l,t,r,b = |LTRB distances|          (a DFL reduction is already >= 0; the
//                                        absolute value only guards a plain head)
//   a  = (angle_value - angle_bias) * pi
//   xf = (r - l)/2,  yf = (b - t)/2     offset of the box centre from the cell,
//                                       expressed in the BOX's own frame
//   cx = (gx + 0.5 + xf*cos a - yf*sin a) * stride    <- rotate that offset into
//   cy = (gy + 0.5 + xf*sin a + yf*cos a) * stride       the image frame
//   w  = (l + r) * stride,   h = (t + b) * stride
//   if regularize and w < h: swap(w,h), a += pi/2     (canonicalise w >= h)
// then rotated per-class NMS, then un-letterbox.
//
// WHERE THE ACTIVATIONS LIVE (verified against the vendor model-zoo reference
// decoder for the deployed export):
//   - class score : raw logit, sigmoid applied HERE (`apply_sigmoid`, default
//     true) — the reference thresholds against unsigmoid(conf), then sigmoids.
//   - angle : the sigmoid is INSIDE the graph, so the tensor already holds a
//     value in [0,1] and the CPU only applies the affine map
//     (v - 0.25) * pi -> [-pi/4, 3pi/4] (`apply_angle_sigmoid` false,
//     `angle_bias` 0.25). A raw head that emits the logit sets
//     apply_angle_sigmoid; a head using the symmetric [-pi/2, pi/2]
//     parameterisation sets angle_bias to 0.5.

/// Post-processing parameters for the LTRB multi-scale OBB head.
struct ObbConfig {
  int num_classes = 15;  ///< DOTA-15; ObbDetector takes the real count from the model
  float conf_thresh = 0.25f;
  /// Rotated-IoU threshold. Lower than the axis-aligned default because two
  /// rotated boxes of the same object overlap more tightly than their extents do.
  float iou_thresh = 0.4f;
  int max_dets = 300;
  /// One stride per scale, ordered to match the per-scale buffers.
  std::vector<int> strides = {8, 16, 32};
  /// 0 => plain LTRB (4 box channels). >0 => 4*reg_max DFL logits per cell,
  /// side-major, reduced by Σ b·softmax(b) exactly as decodeYoloLtrb does.
  int reg_max = 0;
  /// Channel order of the cls/box buffers: true => [C,H,W], false => [H,W,C].
  /// Taken from each output's rknn fmt by ObbDetector, never assumed.
  bool channels_first = true;
  /// Apply sigmoid to the class scores (see the activation note above).
  bool apply_sigmoid = true;
  /// Apply sigmoid to the angle channel. False for the deployed export, whose
  /// graph already did it.
  bool apply_angle_sigmoid = false;
  /// angle = (value - angle_bias) * pi. 0.25 gives the ultralytics
  /// [-pi/4, 3pi/4] range; 0.5 gives a symmetric [-pi/2, pi/2].
  float angle_bias = 0.25f;
  /// Canonicalise every box to w >= h by swapping the sides and adding pi/2.
  /// Without it the same physical box has two encodings and rotated NMS sees
  /// them as different shapes.
  bool regularize = true;
};

/// Decode parallel per-scale float cls/box/angle buffers into final detections.
///
/// `cls[i]`     : row-major class scores for scale i — [nc,H,W] when
///                cfg.channels_first, else [H,W,nc].
/// `box[i]`     : row-major LTRB — [4,H,W] / [4*reg_max,H,W], or the channels-last
///                transposes. DFL channels are side-major.
/// `angle[i]`   : one value per cell in row-major H*W order. A single channel is
///                contiguous per cell in EVERY layout, so this needs no channel
///                order and no stride — for the shared [1,1,A] tensor of the
///                fused export it is simply the base plus that scale's anchor
///                offset.
/// `grid_hw[i]` : the (H_i, W_i) of scale i.
/// `lb`         : letterbox geometry; centres map back through lb.invX/invY and
///                sides divide by lb.scale. Runs threshold -> rotated per-class
///                NMS -> max_dets truncation across ALL scales.
/// The four input vectors plus cfg.strides must share the same length.
std::vector<ObbDetection> decodeObb(const std::vector<const float*>& cls,
                                    const std::vector<const float*>& box,
                                    const std::vector<const float*>& angle,
                                    const std::vector<std::pair<int, int>>& grid_hw,
                                    const ObbConfig& cfg, const LetterboxInfo& lb);

/// Which Engine outputs hold one scale's tensors, plus that scale's grid.
struct ObbScaleOutputs {
  /// cls_index == box_index when the export fuses both into one tensor; the box
  /// channels come first, so `box_channels` also says where the class block starts.
  int cls_index = -1;
  int box_index = -1;
  int angle_index = -1;   ///< -1 when one angle tensor is shared by all scales
  int grid_h = 0;
  int grid_w = 0;
  int num_classes = 0;
  int box_channels = 0;   ///< 4 (plain LTRB) or 4*reg_max (DFL)
  int anchor_offset = 0;  ///< this scale's first cell inside a shared angle tensor
};

/// Resolved description of a whole OBB head, as read from an Engine.
struct ObbHeadLayout {
  std::vector<ObbScaleOutputs> scales;  ///< ordered by DESCENDING grid size (stride 8 first)
  std::vector<int> strides;             ///< derived as input_h / grid_h per scale
  int num_classes = 0;
  int reg_max = 0;  ///< 0 => plain LTRB
  bool channels_first = true;
  bool fused_box_cls = false;   ///< one [1, 4*reg_max+nc, H, W] tensor per scale
  int shared_angle_index = -1;  ///< >= 0 when one [1,1,A] tensor covers every scale
  int total_anchors = 0;        ///< Σ H*W over the scales
  std::string describe() const;
};

/// Work out an OBB head's layout from the Engine's output signature alone.
///
/// Same idea as resolveYoloHead(), extended for the angle branch — and the angle
/// branch is what makes plain grouping-by-grid insufficient: in the deployed
/// export it is one tensor covering ALL anchors, so it has no grid of its own.
/// The discriminator is the model input: a tensor is a feature-scale branch only
/// if (H,W) is the input divided by one common integer stride; whatever is left
/// over is the angle branch, and it must cover exactly the total anchor count.
///
/// Within a scale group, a 1-channel branch is the angle (an OBB export has no
/// score-sum branch, so unlike resolveYoloHead a lone 1-channel tensor is never
/// a pre-filter here), a single remaining tensor is the fused box+cls export, and
/// otherwise the branch matching `num_classes` is cls and the 4/4*reg_max one is
/// box.
///
/// Throws rcdl::Error with the model's full output signature when the outputs do
/// not form an OBB head, so a mismatched model fails at construction instead of
/// decoding garbage.
ObbHeadLayout resolveObbHead(const Engine& engine, int num_classes = 15);

/// Engine-bound LTRB multi-scale OBB detector.
///
/// Construction resolves the head from the Engine, so grids, class count, DFL
/// reg_max, channel order and strides all come from the model; `cfg` supplies the
/// thresholds and the angle/regularise conventions.
class ObbDetector {
 public:
  explicit ObbDetector(Engine& engine, ObbConfig cfg = ObbConfig());

  std::vector<ObbDetection> postprocess(const LetterboxInfo& lb) const;

  const ObbConfig& config() const { return cfg_; }
  const ObbHeadLayout& layout() const { return layout_; }

 private:
  Engine& engine_;
  ObbConfig cfg_;
  ObbHeadLayout layout_;
};

/// The 15 DOTA-v1 class names, in the order every OBB export uses.
/// `dotaClassName` returns "class <id>" for an out-of-range id rather than throwing.
const std::vector<std::string>& dotaClassNames();
const char* dotaClassName(int class_id);

}  // namespace rcdl
