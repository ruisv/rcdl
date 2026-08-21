#pragma once

#include <string>
#include <utility>
#include <vector>

#include "rcdl/preproc/geometry.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// RetinaFace — anchor-based ("prior box") face detection with 5 landmarks
// ===========================================================================
//
// Unlike every YOLO head in this repo, RetinaFace is an SSD-family detector: it
// does NOT predict boxes in image coordinates. The graph emits, for each of a
// fixed set of PRIOR boxes laid out over the feature maps, a small *delta*
// against that prior. So the decoder is only half of the story — the other half
// is regenerating the exact same priors the model was trained and exported
// with. Get the prior layout wrong (a step, a min_size, or the nesting order of
// the generation loops) and nothing errors: boxes just land slightly, or wildly,
// off, and the landmarks scatter. generatePriors() therefore mirrors the
// reference generator exactly, and FaceDetector cross-checks its prior count
// against the model's own anchor count before it decodes anything.
//
// THE THREE OUTPUTS, per prior:
//   loc   : [N, 4]   box delta (dcx, dcy, dw, dh)
//   conf  : [N, 2]   class scores, SOFTMAXED IN THE GRAPH — [background, face]
//   landm : [N, 10]  five landmark deltas (dx, dy), same encoding as the centre
//
// DECODE (all in NORMALIZED [0,1] coordinates, then scaled by the model input):
//   cx = p.cx + dcx * var_center * p.w      w = p.w * exp(dw * var_size)
//   cy = p.cy + dcy * var_center * p.h      h = p.h * exp(dh * var_size)
//   lx = p.cx + dx  * var_center * p.w      ly = p.cy + dy  * var_center * p.h
//
// Landmarks use var_center on BOTH axes: they are centre-like offsets, not
// sizes, so the size variance never appears in them. Swapping the two variances
// is the classic RetinaFace porting bug — it shows up as landmarks drifting off
// the face while the box still looks roughly right.
//
// The two variances are part of the training recipe, not free parameters: they
// rescale the regression targets so the network learns comparable magnitudes for
// centre and size. They must match the weights, which is why they live in the
// config with the reference values rather than being derived.
//
// PREPROCESSING NOTE (not this header's job, but it decides whether any of the
// above produces faces): RetinaFace is trained with BGR channel order and BGR
// mean subtraction, so a converted model usually wants a BGR tensor — unlike the
// YOLO exports in this repo, which take RGB. Feeding the wrong order does not
// error and does not scatter the landmarks; it quietly costs the model most of
// its recall, leaving only the easiest face in a frame. Letterbox into the
// Engine input with PixelFormat::BGR888 unless the model card says otherwise.

/// One detected face in ORIGINAL-image pixels (already un-letterboxed), with
/// its five landmarks in the canonical RetinaFace order: left eye, right eye,
/// nose, left mouth corner, right mouth corner. "Left" is the viewer's left,
/// i.e. the subject's right.
struct FaceDetection {
  float x1;     ///< top-left x
  float y1;     ///< top-left y
  float x2;     ///< bottom-right x
  float y2;     ///< bottom-right y
  float score;  ///< face-class probability in [0,1]
  /// 5 landmarks as (x, y) pairs, original-image pixels, clamped to the source
  /// extent like the box is.
  std::vector<std::pair<float, float>> landmarks;
};

/// One prior ("anchor") box in NORMALIZED coordinates: centre and size are each
/// a fraction of the model input, so the same prior set is valid whatever the
/// canvas is scaled to. Sizes come straight from min_sizes / input; centres from
/// the feature-map cell grid.
struct PriorBox {
  float cx;
  float cy;
  float w;
  float h;
};

/// Post-processing parameters for the RetinaFace head.
///
/// The prior-generation fields (`steps`, `min_sizes`, `clip`) plus the input
/// size fully determine the anchor set; the defaults are the reference
/// RetinaFace (mobilenet/resnet backbone) configuration that the public export
/// uses, and produce 4200 priors at 320x320 and 16800 at 640x640.
struct FaceConfig {
  int input_w = 320;  ///< model input width  (letterbox canvas)
  int input_h = 320;  ///< model input height (letterbox canvas)

  float conf_thresh = 0.5f;
  float iou_thresh = 0.4f;
  int max_faces = 100;

  /// Variance the box/landmark regression targets were encoded with.
  /// `var_center` scales centre and landmark offsets, `var_size` the log-size
  /// deltas. These belong to the trained weights — change them only to match a
  /// differently-trained export.
  float var_center = 0.1f;
  float var_size = 0.2f;

  /// Feature-map strides, one per scale, in model-input pixels. The grid for
  /// scale k is ceil(input_h / steps[k]) x ceil(input_w / steps[k]).
  std::vector<int> steps = {8, 16, 32};
  /// Prior side lengths in model-input pixels, per scale. Every entry adds one
  /// prior per cell of that scale, so the anchor count is
  /// Σ_k grid_h(k) * grid_w(k) * min_sizes[k].size().
  std::vector<std::vector<int>> min_sizes = {{16, 32}, {64, 128}, {256, 512}};
  /// Clamp priors into [0,1]. The reference generator leaves them unclamped
  /// (large priors at the border legitimately hang off the canvas), so this is
  /// off by default; it exists because some forks enable it and a mismatch
  /// shifts every border box.
  bool clip = false;

  /// Column of the conf tensor holding the face probability. The exported head
  /// is [background, face], so 1.
  int face_class = 1;
  /// Softmax the conf channels before thresholding. The deployed export already
  /// has the softmax in the graph, so this is off; set it for an export that
  /// emits raw logits.
  bool apply_softmax = false;
};

/// Generate the prior boxes for `cfg`, in the exact order the network's output
/// rows are in.
///
/// The nesting is load-bearing and mirrors the reference implementation:
///
///     for each scale k (in cfg.steps order)
///       for each row i, column j of that scale's grid   (row-major)
///         for each min_size of that scale                (varies FASTEST)
///
/// so prior index = ((scale_base + i * grid_w + j) * n_min_sizes) + m. Any other
/// nesting produces the same COUNT with a permuted layout — which is why the
/// count check in FaceDetector is necessary but not sufficient, and why this
/// order is spelled out rather than left to the reader.
///
/// Throws rcdl::Error when the input size is non-positive, `steps` and
/// `min_sizes` disagree in length, or a step / min_size is not positive.
std::vector<PriorBox> generatePriors(const FaceConfig& cfg);

/// Decode the three RetinaFace output tensors into faces.
///
/// `loc`   / `loc_shape`   : [1,N,4]  or [1,4,N]  box deltas.
/// `conf`  / `conf_shape`  : [1,N,2]  or [1,2,N]  class scores.
/// `landm` / `landm_shape` : [1,N,10] or [1,10,N] landmark deltas.
/// `priors` : generatePriors(cfg) — must hold exactly N entries.
/// `lb`     : letterbox geometry used at preprocess time; boxes AND landmarks
///            are mapped back to original-image pixels via lb.invX/invY and
///            clamped to the source extent.
///
/// Each tensor's axis order is read from its own shape rather than assumed: the
/// channel axis is the one matching that tensor's known channel count (4 / 2 /
/// 10), which is unambiguous because N is in the thousands. Leading unit axes
/// are ignored, so [1,N,C], [N,C] and [1,1,N,C] all decode alike.
///
/// Runs threshold -> NMS -> max_faces truncation. Throws rcdl::Error when a
/// shape cannot be reconciled with the prior count.
std::vector<FaceDetection> decodeFaces(const float* loc, const std::vector<int>& loc_shape,
                                       const float* conf, const std::vector<int>& conf_shape,
                                       const float* landm, const std::vector<int>& landm_shape,
                                       const std::vector<PriorBox>& priors,
                                       const FaceConfig& cfg, const LetterboxInfo& lb);

/// Which Engine output holds which branch of the head, plus the anchor count the
/// model itself declares. Resolved from the output signature, never assumed.
struct FaceHeadLayout {
  int loc_index = -1;    ///< [.., N, 4]
  int conf_index = -1;   ///< [.., N, 2]
  int landm_index = -1;  ///< [.., N, 10]
  int num_priors = 0;    ///< N, as read from the model
  int input_w = 0;       ///< model input width  (from input 0)
  int input_h = 0;       ///< model input height
  std::string describe() const;
};

/// Work out the head layout from the Engine's signature alone.
///
/// The three branches are told apart by their channel count (4 / 2 / 10) — they
/// are distinct and fixed for this head, so no index convention is needed and a
/// re-exported model with a different output ORDER still resolves. All three
/// must report the same N. Throws rcdl::Error describing what it saw when the
/// outputs are not a RetinaFace head, so a mismatched model fails at
/// construction instead of decoding garbage.
FaceHeadLayout resolveFaceHead(const Engine& engine);

/// Engine-bound RetinaFace detector.
///
/// Construction resolves the head (resolveFaceHead), takes the canvas size from
/// the model's own input tensor — overriding `cfg.input_w/h` — and generates the
/// priors once. The prior count is then checked against the model's anchor
/// count: a mismatch means the configured steps / min_sizes are not the ones the
/// model was exported with, so it throws rather than decode boxes that would be
/// subtly and silently wrong.
class FaceDetector {
 public:
  explicit FaceDetector(Engine& engine, FaceConfig cfg = FaceConfig());

  std::vector<FaceDetection> postprocess(const LetterboxInfo& lb) const;

  const FaceConfig& config() const { return cfg_; }
  const FaceHeadLayout& layout() const { return layout_; }
  /// The generated prior set, in output-row order.
  const std::vector<PriorBox>& priors() const { return priors_; }

 private:
  Engine& engine_;
  FaceConfig cfg_;
  FaceHeadLayout layout_;
  std::vector<PriorBox> priors_;
};

}  // namespace rcdl
