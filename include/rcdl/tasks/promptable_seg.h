#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rcdl/preproc/geometry.h"
#include "rcdl/preproc/image.h"
#include "rcdl/preproc/letterbox.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Promptable segmentation (SAM-family): point at a thing, get its mask
// ===========================================================================
//
// Every other segmentation head here decides FOR you what the classes are. This
// one has no classes at all: it takes a prompt — a click or a box — and returns
// the mask of whatever is there. That makes it the head for annotation tools,
// for "cut this out", and for turning any detector's box into a silhouette.
//
// It is TWO models, and the split is the whole point:
//
//   * the ENCODER runs once per frame (~300 ms at 1024x1024) and produces a
//     256x64x64 embedding;
//   * the DECODER runs once per PROMPT (~140 ms) against that embedding.
//
// So `setImage()` then many `box()` / `point()` calls is the shape of every
// sensible use. Calling setImage per prompt is not wrong, just several times the
// cost.
//
// THE PROMPT CONVENTION IS SAM'S, and it is why the exported decoder can take a
// fixed two points: a BOX is two points labelled 2 (top-left) and 3
// (bottom-right); a single CLICK is one point labelled 1 (foreground) or 0
// (background), padded with a (0,0) point labelled -1. Coordinates are in the
// model's 1024-pixel canvas, so a prompt in source pixels goes forward through
// the letterbox exactly as a decoded box comes back through it.
//
// FOUR MASKS COME BACK, not one. SAM's decoder answers an ambiguous prompt with
// several nestings — a click on a shirt can mean the shirt, the person, or the
// crowd — each with a predicted IoU. `multimask` picks the highest-scoring one;
// `masks()` hands over all four when the caller wants to choose.
//
// Precision: the encoder is FLOAT here on measurement, not preference. Its int8
// build keeps the shape of large objects (IoU 0.53-0.82 against the float model)
// and loses small ones completely — a click that returns 3.5% of the frame in
// float returns 0.07% in int8. An embedding feeding a second network is not a
// classifier; nothing downstream re-normalises what quantization moved.

/// One mask, in SOURCE-image pixels: row-major `width * height` bytes, 0 or 1.
struct PromptMask {
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> data;
  /// The decoder's own predicted IoU for this mask — SAM's quality estimate,
  /// and the number `multimask` selects on. Not a class confidence.
  float score = 0.0f;
  /// Tight bounding box of the set pixels; all zero when the mask is empty.
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;

  bool empty() const { return x1 <= x0 || y1 <= y0; }
  /// Fraction of the frame the mask covers.
  float area() const;
};

struct PromptConfig {
  /// Masks come back as LOGITS; > 0 is the model's own decision boundary.
  /// Raising it shrinks every mask, which is occasionally what a caller wants
  /// on a soft edge.
  float mask_thresh = 0.0f;
  /// Return the highest-scoring of the decoder's four masks. With false, the
  /// first (SAM's "single mask" slot) is returned instead.
  bool multimask = true;
  /// Letterbox padding. 114 is not arbitrary: SAM pads with ZERO after its own
  /// normalisation, which corresponds to a pixel value at the channel mean
  /// (~124). 114 lands within a fifth of a standard deviation of that, so the
  /// padding reads as neutral to the encoder either way.
  std::uint8_t pad = 114;
};

/// Encode a prompt into the decoder's two-point form.
///
/// `coords` receives 4 floats (x0,y0,x1,y1) and `labels` 2, both in the model's
/// canvas. Pure function: no Engine, numpy-testable. A box uses labels {2,3}; a
/// click uses {1,-1} for foreground or {0,-1} for background.
void encodeBoxPrompt(float x1, float y1, float x2, float y2, const LetterboxInfo& lb,
                     float* coords, float* labels);
void encodePointPrompt(float x, float y, bool positive, const LetterboxInfo& lb, float* coords,
                       float* labels);

/// Project one decoder logit map onto the source frame and threshold it.
///
/// `logits` is row-major `h * w` covering the whole model canvas (whatever its
/// own resolution — SAM's is 256x256 for a 1024 input), `lb` the letterbox that
/// produced the embedding. Pure function.
PromptMask maskFromLogits(const float* logits, int w, int h, const LetterboxInfo& lb,
                          float thresh = 0.0f, float score = 0.0f);

/// Encoder + decoder, with the image encoded once and prompted many times.
class PromptableSegmenter {
 public:
  PromptableSegmenter(Engine& encoder, Engine& decoder, PromptConfig cfg = {});

  /// Letterbox `src` into the encoder's input and run it. Everything after this
  /// is per-prompt work against the resulting embedding.
  void setImage(const ImageView& src, PreprocBackend backend = PreprocBackend::Auto);

  /// Prompts, in SOURCE-image pixels. Both require setImage() first.
  PromptMask box(float x1, float y1, float x2, float y2);
  PromptMask point(float x, float y, bool positive = true);

  /// All four masks for the LAST prompt, best-scoring first — SAM's answer to an
  /// ambiguous click, where the right nesting is the caller's to choose.
  std::vector<PromptMask> masks() const;

  const LetterboxInfo& letterbox() const { return lb_; }
  PreprocBackend lastBackend() const { return last_backend_; }
  int inputWidth() const { return in_w_; }
  int inputHeight() const { return in_h_; }
  const PromptConfig& config() const { return cfg_; }

 private:
  void runDecoder(const float* coords, const float* labels);
  PromptMask best() const;

  Engine& encoder_;
  Engine& decoder_;
  PromptConfig cfg_;
  int in_w_ = 0, in_h_ = 0;      ///< encoder canvas
  int mask_w_ = 0, mask_h_ = 0;  ///< decoder mask resolution
  int num_masks_ = 0;
  bool have_image_ = false;
  LetterboxInfo lb_;
  Image stage_;                  ///< u8 canvas the hardware letterbox writes
  std::vector<float> input_;     ///< the encoder's float tensor
  std::vector<float> embedding_; ///< transposed for the decoder, see the .cc
  std::vector<float> logits_;    ///< the last prompt's masks
  std::vector<float> scores_;
  PreprocBackend last_backend_ = PreprocBackend::Auto;
};

}  // namespace rcdl
