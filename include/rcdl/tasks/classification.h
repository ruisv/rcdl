#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rcdl/preproc/geometry.h"
#include "rcdl/preproc/image.h"
#include "rcdl/preproc/letterbox.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Top-K image classification (ImageNet-style single-label heads)
// ===========================================================================
//
// The simplest head there is: one output tensor holding `nc` per-class scores,
// no geometry to invert. Three RKNPU2 realities shape the code below.
//
// SHAPE. The class axis is the only axis with more than one element, but WHICH
// axis it is depends on the export: a converted ONNX classifier reports
// [1,1000], [1,1000,1,1] (NCHW with the spatial dims kept) or [1,1,1,1000]
// (NHWC after a global pool). All three describe the SAME 1000 contiguous
// values, so the decoder needs the class COUNT, not the layout — and the count
// is found by looking for the single non-unit dimension (classCountFromShape),
// which also rejects a tensor that is not a classifier head at all instead of
// silently decoding it.
//
// QUANTIZATION. A quantized `.rknn` emits INT8 affine logits, not float; the
// dequant happens in outputAsFloat(), so nothing here may assume the fast path.
// One consequence worth knowing at bring-up: int8 logits are coarse (one step
// is `scale`), so two near-tied classes can come back with exactly equal
// scores. decodeClassification breaks such ties by ascending class id, so the
// result is at least reproducible run to run.
//
// ACTIVATION PLACEMENT. Some exports fold the final softmax into the graph and
// some stop at the logits — the classification examples in the
// rknn_model_zoo (MobileNet, ResNet) stop at the logits and softmax on the CPU,
// which is why ClsConfig::apply_softmax defaults to true. Guessing per-model is
// not possible from the tensor metadata, so it is a flag; looksLikeProbabilities()
// answers it empirically from one real output (see below).

/// Post-processing parameters for an image-classification head.
struct ClsConfig {
  /// Number of top-scoring classes to return, sorted by score descending. A
  /// value <= 0 (or one exceeding the class count) returns ALL classes sorted.
  int top_k = 5;
  /// true  => scores are softmax probabilities over the `num_classes` logits.
  /// false => scores are passed through unchanged (raw logits, or the
  ///          probabilities of an export that already has the softmax inside
  ///          the graph).
  ///
  /// Softmax is monotonic, so this flag never changes the ranking or the class
  /// ids — only the score magnitudes. Set it wrong in the "already softmaxed"
  /// direction and the reported confidences are squashed towards 1/nc (a
  /// softmax of probabilities); set it wrong the other way and they are raw
  /// logits that can be negative or above 1.
  bool apply_softmax = true;
};

/// One Top-K classification entry.
struct ClsResult {
  int class_id;  ///< class index (0..999 for ImageNet-1k)
  float score;   ///< softmax probability in [0,1], or the unmodified output
                 ///< value when ClsConfig::apply_softmax is false.
};

/// Number of classes described by an output shape: its single non-unit
/// dimension ([1,1000] / [1,1000,1,1] / [1,1,1,1000] all give 1000, and an
/// all-ones shape gives 1).
///
/// Throws rcdl::Error naming the shape when TWO or more dimensions exceed 1 —
/// that tensor is a feature map or a multi-head output, not a classifier's
/// score vector, and flattening it by product would decode nonsense silently.
int classCountFromShape(const std::vector<int>& shape);

/// Decode `num_classes` contiguous scores into Top-K classification results.
///
/// `logits`      : pointer to `num_classes` floats in row-major order.
/// `num_classes` : number of class scores (e.g. 1000 for ImageNet-1k).
/// `cfg`         : top_k + softmax toggle.
///
/// With `cfg.apply_softmax` the scores are a numerically-stable softmax — the
/// max logit is subtracted before exponentiating, so an int8 model whose
/// dequantized logits reach the tens never overflows exp() — and the returned
/// score is a probability in [0,1]. Otherwise the value is returned as it came.
/// Results are sorted by score descending, ties broken by ascending class id,
/// and truncated to `cfg.top_k` (all classes when top_k <= 0 or > num_classes).
/// Returns empty if `logits` is null or `num_classes <= 0`.
std::vector<ClsResult> decodeClassification(const float* logits, int num_classes,
                                            const ClsConfig& cfg);

/// Shape-taking overload: resolves the class count with classCountFromShape()
/// (and therefore throws on a shape that is not a classifier head).
std::vector<ClsResult> decodeClassification(const float* logits,
                                            const std::vector<int>& shape,
                                            const ClsConfig& cfg);

/// Do these `n` values already look like a probability distribution — all in
/// [0,1] and summing to 1 within `tol`?
///
/// This is the empirical answer to "does this export have the softmax inside
/// the graph?": run one image, read the raw output, call this. true => leave
/// ClsConfig::apply_softmax off; false (logits, typically spanning negative to
/// positive) => leave it on. It is a bring-up diagnostic, not something to
/// branch on per frame — a confident softmaxed output looks like (1, 0, 0, ...)
/// which a one-hot logit vector also could, so the decision belongs in the
/// model's registry entry once, not in the hot path.
bool looksLikeProbabilities(const float* data, int n, float tol = 1e-2f);

// ---------------------------------------------------------------------------
// Preprocessing geometry: resize-then-center-crop, NOT letterbox
// ---------------------------------------------------------------------------

/// A rectangle in ORIGINAL-image pixels.
struct CropBox {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

/// The source rectangle implementing the standard ImageNet EVAL transform,
/// `Resize(shorter side -> S)` followed by `CenterCrop(out_h, out_w)` with
/// `S = round(out / crop_ratio)` — torchvision's 256/224 recipe at the default
/// `crop_ratio` of 0.875.
///
/// Doing it as "crop this rectangle, then squash it to (out_w, out_h)" is
/// algebraically the same transform as "scale the whole image, then crop", and
/// it is one hardware op instead of two: RGA crops and scales in a single
/// improcess, and nothing outside the crop is ever resampled.
///
/// It is deliberately NOT a letterbox. Classifiers are trained and evaluated on
/// centre crops; padding bars are pixels the model has never seen, and they cost
/// accuracy. Nothing here needs inverting afterwards — a class id has no
/// coordinates — which is why this returns a plain rectangle rather than a
/// LetterboxInfo.
///
/// `crop_ratio` >= 1 means "no crop margin": the largest centred rectangle with
/// the output's aspect ratio, i.e. a pure aspect-preserving centre crop. Values
/// <= 0 are treated as the 0.875 default. The result is clamped to the source
/// and never empty for a positive source size.
CropBox centerCropBox(int src_w, int src_h, int out_w, int out_h, float crop_ratio = 0.875f);

/// Preprocessing parameters for Classifier::classify().
struct ClsPreproc {
  /// The channel order the model was built with — RGB888 for the
  /// rknn_model_zoo exports, BGR888 for a model calibrated on OpenCV-native
  /// images.
  /// Getting it wrong costs accuracy silently, so it is explicit here and
  /// recorded per model in the model registry.
  PixelFormat model_input = PixelFormat::RGB888;
  /// Centre-crop margin, see centerCropBox(). 0.875 is the ImageNet eval
  /// convention; use 1.0 for a model evaluated without a crop margin, and note
  /// that some vendor demos just squash the whole frame to the input size —
  /// pass a crop_ratio of 1.0 and a square input to reproduce a plain resize of
  /// an already-square image.
  float crop_ratio = 0.875f;
  PreprocBackend backend = PreprocBackend::Auto;  ///< RGA, CPU, or pick
  YuvRange yuv_range = YuvRange::kStudioToFull;   ///< NV12 sources: level handling
};

/// Engine-bound classifier.
///
/// Two ways in, sharing one postprocess:
///   - `postprocess()`  — the caller did preprocessing + infer() itself (the
///     Python layer, or a pipeline that already owns the input tensor);
///   - `classify(src)` — crop + resize straight into the NPU's input tensor,
///     infer, decode. Needs a uint8 NHWC image-input model (every quantized
///     `.rknn` classifier is one, with mean/std folded into the graph by the
///     conversion toolkit); a float-input model is rejected there — not at
///     construction, so such a model can still be driven through postprocess()
///     — rather than being fed image bytes it would read as garbage floats.
///
/// The class count is resolved from the Engine's output shape at construction,
/// so a model whose selected output is not a score vector fails there instead of
/// decoding nonsense per frame.
class Classifier {
 public:
  explicit Classifier(Engine& engine, ClsConfig cfg = ClsConfig(),
                      ClsPreproc pre = ClsPreproc(), int output_index = 0);

  /// Decode the CURRENT contents of the bound output (i.e. after an infer()).
  std::vector<ClsResult> postprocess() const;

  /// Preprocess `src`, infer, and decode. `src` may be anything the preproc
  /// layer accepts — an NV12 dma-buf frame from the VPU (cropped and scaled by
  /// RGA with no CPU touch) or a host BGR888 buffer.
  std::vector<ClsResult> classify(const ImageView& src);
  /// Convenience overload for an interleaved, row-contiguous host BGR image.
  std::vector<ClsResult> classify(const std::uint8_t* bgr, int width, int height);

  /// The source rectangle classify() would crop out of a (src_w, src_h) frame.
  /// Exposed so a caller can draw or log it without re-deriving the convention.
  CropBox cropFor(int src_w, int src_h) const;

  int numClasses() const noexcept { return num_classes_; }
  int inputWidth() const noexcept { return input_w_; }
  int inputHeight() const noexcept { return input_h_; }
  /// Backend that ran the most recent classify() preproc (tells you whether RGA
  /// fell back to the CPU).
  PreprocBackend lastBackend() const noexcept { return last_backend_; }

  const ClsConfig& config() const noexcept { return cfg_; }
  const ClsPreproc& preproc() const noexcept { return pre_; }

 private:
  Engine& engine_;
  ClsConfig cfg_;
  ClsPreproc pre_;
  int out_idx_;
  int num_classes_ = 0;
  int input_w_ = 0;
  int input_h_ = 0;
  PreprocBackend last_backend_ = PreprocBackend::Auto;
};

// ---------------------------------------------------------------------------
// Class labels
// ---------------------------------------------------------------------------

/// Load a class-name table from a text file, one name per line, in class-index
/// order.
///
/// This is the file classification models ship with rather than a table baked
/// into the library: ImageNet-1k's `synset.txt` is 1000 lines of
/// `n01440764 tench, Tinca tinca` — a WordNet id, a space, then the comma-
/// separated synonyms. With `strip_wnid` (the default) the leading `nXXXXXXXX `
/// is dropped, so line 0 becomes `tench, Tinca tinca`; a plain label file
/// without wnids is unaffected, because the prefix is only removed when it
/// really is `n` followed by 8 digits. Trailing CR (a file authored on Windows)
/// and surrounding blanks are stripped either way.
///
/// Throws rcdl::Error when the file cannot be opened — a silently empty label
/// table would print every prediction as `class 285`.
std::vector<std::string> loadClassLabels(const std::string& path, bool strip_wnid = true);

/// `labels[class_id]`, or "class <id>" when the id is outside the table (a
/// mismatched label file, or a model with more classes than the file has lines).
/// Returns by value so there is no lifetime question about the fallback.
std::string classLabel(const std::vector<std::string>& labels, int class_id);

}  // namespace rcdl
