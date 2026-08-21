#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rcdl/preproc/image.h"
#include "rcdl/preproc/letterbox.h"
#include "rcdl/tasks/pose.h"  // Keypoint

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Whole-body pose: 133 keypoints per person, top-down
// ===========================================================================
//
// The pose head in tasks/pose.h is BOTTOM-UP and gives 17 body joints for every
// person in one pass. This one is TOP-DOWN: it is handed one person's box and
// returns the COCO-WholeBody 133 — the 17 body joints, 6 feet, 68 face and 2x21
// hand points — which is what sign language, gesture, and any "what are the
// fingers doing" application needs and a 17-point skeleton cannot answer.
//
// The cost model is the opposite of the bottom-up head's: one inference PER
// PERSON (~47 ms), so a crowded frame is linear in people, and a detector has to
// run first. Use tasks/pose.h when 17 joints are enough.
//
// TWO CONVENTIONS DECIDE WHETHER THE OUTPUT IS RIGHT, and neither is in the
// tensor:
//
//   * THE CROP. The box is padded by 1.25 and then forced to the model's 3:4
//     aspect — grown on whichever axis is short, never cropped. Feeding a
//     tight box, or a squashed one, moves every joint and reports nothing.
//   * THE SPLIT RATIO. The head is SimCC: each joint is a pair of 1-D
//     distributions, at `split_ratio` bins PER INPUT PIXEL. Forgetting the
//     division puts the whole skeleton at twice its coordinates inside the crop,
//     which still looks like a person — a small one in the top-left corner.
//
// Precision: this model ships FLOAT on measurement. Its int8 build agrees with
// the float one on easy crops (median 1.1-1.2 px) and falls apart on hard ones —
// a backlit person keeps 22 of 133 joints above threshold instead of 133, with
// 95th-percentile errors of 10-70 px. A 1-D argmax over 384 bins is a
// decision, and quantization noise moves decisions.

/// The five regions of the COCO-WholeBody 133, in index order.
enum class BodyPart { Body, Foot, Face, LeftHand, RightHand };

/// Which region keypoint `i` belongs to (0-16 body, 17-22 feet, 23-90 face,
/// 91-111 left hand, 112-132 right hand).
BodyPart bodyPart(int i) noexcept;
const char* bodyPartName(BodyPart p) noexcept;
/// [begin, end) index range of a region, for slicing a result.
void bodyPartRange(BodyPart p, int* begin, int* end) noexcept;

/// The rectangle of the SOURCE image the model is actually shown: the box
/// padded and then grown to the model's aspect. Everything maps back through it.
struct CropRect {
  float cx = 0.0f, cy = 0.0f, w = 0.0f, h = 0.0f;

  float x0() const { return cx - w * 0.5f; }
  float y0() const { return cy - h * 0.5f; }
};

struct WholeBodyConfig {
  /// Joints below this score come back with score as measured but position
  /// (-1,-1): a low-confidence SimCC argmax is a peak in noise, not a guess
  /// worth drawing.
  float kpt_thresh = 0.3f;
  /// Box padding before the aspect fix. 1.25 is the value the model was
  /// trained and exported with.
  float padding = 1.25f;
  /// SimCC bins per input pixel.
  float split_ratio = 2.0f;
  /// Fill for the parts of the crop that fall outside the image.
  std::uint8_t pad = 114;
};

/// The crop rect for a person box. Pure function.
CropRect cropGeometry(float x1, float y1, float x2, float y2, int in_w, int in_h,
                      float padding = 1.25f);

/// Decode a SimCC pair into keypoints in SOURCE pixels. Pure function: no
/// Engine, numpy-testable.
///
/// `simcc_x` is row-major [K, bins_x], `simcc_y` [K, bins_y]. Each joint's
/// position is the argmax of each axis divided by `cfg.split_ratio` (model-input
/// pixels), then mapped out through `crop`; its score is the mean of the two
/// peak values.
std::vector<Keypoint> decodeSimcc(const float* simcc_x, const float* simcc_y, int num_kpts,
                                  int bins_x, int bins_y, const CropRect& crop, int in_w,
                                  int in_h, const WholeBodyConfig& cfg = {});

/// Engine-bound whole-body head: a person box in, 133 keypoints out.
class WholeBodyEstimator {
 public:
  explicit WholeBodyEstimator(Engine& engine, WholeBodyConfig cfg = {});

  /// Crop `box` out of `src`, run, and decode into SOURCE-image pixels.
  std::vector<Keypoint> estimate(const ImageView& src, float x1, float y1, float x2, float y2);

  /// Decode the Engine's current outputs for a crop you set up yourself.
  std::vector<Keypoint> postprocess(const CropRect& crop) const;

  /// The crop the last estimate() used.
  const CropRect& lastCrop() const { return last_crop_; }
  int inputWidth() const { return in_w_; }
  int inputHeight() const { return in_h_; }
  int numKeypoints() const { return num_kpts_; }
  const WholeBodyConfig& config() const { return cfg_; }

 private:
  Engine& engine_;
  WholeBodyConfig cfg_;
  int in_w_ = 0, in_h_ = 0;
  int num_kpts_ = 0, bins_x_ = 0, bins_y_ = 0;
  CropRect last_crop_;
  std::vector<float> input_;
};

}  // namespace rcdl
