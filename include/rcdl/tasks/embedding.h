#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "rcdl/preproc/geometry.h"
#include "rcdl/preproc/image.h"
#include "rcdl/preproc/letterbox.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

// ===========================================================================
// Image embeddings — ReID appearance vectors, retrieval, zero-shot matching
// ===========================================================================
//
// An embedding model maps an image (usually a CROP: one person, one vehicle,
// one object) to a fixed-length vector, so that semantic similarity becomes
// geometric closeness. Once every vector is L2-normalized, cosine similarity is
// a plain dot product, which is all the matching in this header is.
//
// The two uses on this hardware:
//   tracking   — a ReID head (OSNet and friends) embeds each detection crop;
//                the tracker associates a detection with a track by cosine
//                similarity to the track's smoothed vector. The tracking-side
//                helpers live in tracks/reid.h and consume what this produces.
//   retrieval  — embed a gallery once into an EmbeddingBank, then embed a query
//                and take the best dot products. Zero-shot classification with a
//                dual-encoder model is the same operation against a table of
//                text vectors computed offline on a host, which is why
//                EmbeddingBank does not care where its vectors came from.
//
// OUTPUT SHAPE. Like a classifier head, the embedding axis is the only non-unit
// axis, but which axis it is depends on the export: [1,512], [1,512,1,1] (NCHW
// with the pooled spatial dims kept) or [1,1,1,512] (NHWC). All three are the
// same 512 contiguous values — embeddingDimFromShape() finds the width and
// rejects anything that is not a single vector, which is what catches a model
// whose selected output is the per-patch feature GRID ([1,N,D]) rather than the
// pooled vector.
//
// QUANTIZATION. A quantized `.rknn` emits INT8 affine values, dequantized by
// outputAsFloat(). Nothing here may assume the float fast path, and one thing
// follows from it: an int8 embedding is coarse, so cosine similarities of two
// crops of the same object land a little lower than the float model's. Compare
// against thresholds measured with the SAME quantized model.
//
// PREPROCESSING IS NOT A LETTERBOX. ReID and retrieval towers are trained on
// crops squashed to a fixed shape (128x256 for person ReID, a square for the
// retrieval towers), so preserving aspect would feed them padding bars they have
// never seen. An embedding has no coordinates, so unlike detection there is no
// geometry to invert afterwards — which is why nothing here returns a
// LetterboxInfo.

/// Post-processing parameters for an embedding head.
struct EmbedConfig {
  /// L2-normalize the vector on read-out. Leave this on: every similarity here
  /// assumes unit vectors so that cosine reduces to a dot product. Turning it
  /// off is for callers that want the raw pooled activations (e.g. to average
  /// several crops before normalizing once).
  bool l2_normalize = true;
};

/// One entry of a similarity search, most-similar first.
struct EmbedMatch {
  int index;          ///< position of the entry in the bank
  float score;        ///< cosine similarity in [-1,1] (dot of unit vectors)
  std::string label;  ///< the entry's label, empty if it was added without one
};

/// Embedding width described by an output shape: its single non-unit dimension
/// ([1,512] / [1,512,1,1] / [1,1,1,512] all give 512).
///
/// Throws rcdl::Error naming the shape when two or more dimensions exceed 1 —
/// that output is a feature grid, not one pooled vector, and flattening it by
/// product would silently produce an N*D-long "embedding" that matches nothing.
int embeddingDimFromShape(const std::vector<int>& shape);

/// L2-normalize `dim` floats in place.
///
/// A zero (or denormal-magnitude) vector is left alone: dividing by ~0 would
/// poison every later dot product with NaNs, whereas a zero row simply scores 0
/// against everything, which is the sane answer for "no appearance information".
///
/// Named for the pointer form on purpose — the vector-level `l2Normalize()` /
/// `cosineSimilarity()` used by the tracker live in tracks/reid.h, and both
/// headers must be includable in one translation unit.
void l2NormalizeInPlace(float* v, int dim);

/// Read `dim` floats into an embedding vector, optionally L2-normalized.
///
/// `data` : pointer to `dim` contiguous floats (the pooled head's output).
/// `dim`  : embedding width (128 / 512 / 768 ... depending on the model).
/// Returns empty if `data` is null or `dim <= 0`.
std::vector<float> decodeEmbedding(const float* data, int dim, const EmbedConfig& cfg);

/// Shape-taking overload: resolves the width with embeddingDimFromShape().
std::vector<float> decodeEmbedding(const float* data, const std::vector<int>& shape,
                                   const EmbedConfig& cfg);

// ---------------------------------------------------------------------------
// Similarity / distance
// ---------------------------------------------------------------------------
//
// All three take (pointer, pointer, dim) rather than vectors: it lets them work
// on a row of an EmbeddingBank without copying it out, and it keeps them
// overloads of — rather than redefinitions of — the vector-level helpers in
// tracks/reid.h.

/// Cosine similarity in [-1,1]. The inputs need NOT be pre-normalized (the
/// norms are divided out here); returns 0 when either vector is zero, so a
/// missing embedding scores as "no evidence" rather than NaN.
float cosineSimilarity(const float* a, const float* b, int dim);

/// 1 - cosineSimilarity: a distance in [0,2], which is the form an association
/// cost matrix wants (0 == identical appearance).
float cosineDistance(const float* a, const float* b, int dim);

/// Plain L2 distance. On UNIT vectors it carries the same ordering as cosine
/// (‖a-b‖² = 2 - 2·cos), so it is interchangeable there; on raw vectors it is
/// not, because it also sees the magnitudes. Provided for models whose published
/// thresholds are euclidean, and for gallery code that never normalizes.
float euclideanDistance(const float* a, const float* b, int dim);

// ---------------------------------------------------------------------------
// Bank
// ---------------------------------------------------------------------------

/// A searchable table of equal-length embeddings.
///
/// The CPU half of both retrieval and zero-shot classification: fill it with
/// gallery vectors (or with text vectors computed offline on a host), then
/// search a freshly embedded image against it. Vectors are normalized on
/// insert, so search() is one dot product per entry — a linear scan, which is
/// the right structure at the sizes that fit on a board (thousands of entries;
/// a 512-d bank of 10k entries is 20 MB and scans in a few ms).
///
/// The first add() fixes `dim`; a later add() of a different length throws
/// rcdl::Error, because a silent dimension mismatch surfaces only as
/// meaningless similarity scores.
class EmbeddingBank {
 public:
  EmbeddingBank() = default;

  /// Append one entry. `vec` is L2-normalized INTO the bank (the caller's copy
  /// is untouched). `label` is free-form — a class name for zero-shot use, an
  /// image or track id for retrieval, or empty.
  void add(const std::vector<float>& vec, const std::string& label = "");

  /// Search `query` against every entry and return the `k` best by cosine
  /// similarity, descending, ties broken by ascending index so the order is
  /// reproducible. `query` is normalized internally, so it may be raw. `k <= 0`
  /// or beyond the entry count returns all entries, sorted. Returns empty for an
  /// empty bank; throws rcdl::Error if `query`'s length disagrees with dim().
  std::vector<EmbedMatch> search(const std::vector<float>& query, int k = 5) const;

  /// Pointer to entry `i`'s normalized row (dim() floats), or nullptr when `i`
  /// is out of range. Valid until the next add().
  const float* row(int i) const noexcept;

  void clear() noexcept;

  int size() const noexcept { return static_cast<int>(labels_.size()); }
  int dim() const noexcept { return dim_; }
  const std::string& label(int i) const { return labels_.at(static_cast<std::size_t>(i)); }

 private:
  int dim_ = 0;
  std::vector<float> data_;  ///< size() * dim_, row-major, unit-norm rows
  std::vector<std::string> labels_;
};

// ---------------------------------------------------------------------------
// Engine-bound embedder
// ---------------------------------------------------------------------------

/// Preprocessing parameters for ImageEmbedder::embed().
struct EmbedPreproc {
  /// The channel order the model was built with — RGB888 for the torchreid /
  /// model-zoo exports, BGR888 for a model calibrated on OpenCV-native images.
  /// Wrong here costs accuracy silently, so it is explicit and recorded per
  /// model in the model registry.
  PixelFormat model_input = PixelFormat::RGB888;
  /// Scale factor applied to the box around its centre before cropping. 1.0 is
  /// the plain detection box, which is what person-ReID models are trained on;
  /// >1 adds context, which some vehicle / object models prefer. The expanded
  /// box is clipped to the frame.
  float box_expand = 1.0f;
  PreprocBackend backend = PreprocBackend::Auto;  ///< RGA, CPU, or pick
  YuvRange yuv_range = YuvRange::kStudioToFull;   ///< NV12 sources: level handling
};

/// Engine-bound image embedder: one crop in, one vector out.
///
/// Two ways in, sharing one postprocess:
///   - `postprocess()` — the caller did preprocessing + infer() itself;
///   - `embed(src, x1,y1,x2,y2)` — crop the box out of the frame, squash it
///     into the NPU's input tensor (RGA does crop + scale + colour conversion in
///     one op, so a VPU frame is never touched by the CPU), infer, decode.
///
/// The embedding width is resolved from the Engine's output shape at
/// construction, so a model whose selected output is a patch-feature grid fails
/// there rather than returning a vector that matches nothing. Check dim()
/// against the model's documented width once at startup.
///
/// Not thread-safe, because the bound Engine is not: run one embedder per
/// Engine (see backend/engine_pool.h to fan crops across the NPU cores).
class ImageEmbedder {
 public:
  explicit ImageEmbedder(Engine& engine, EmbedConfig cfg = EmbedConfig(),
                         EmbedPreproc pre = EmbedPreproc(), int output_index = 0);

  /// Decode the CURRENT contents of the bound output (i.e. after an infer()).
  std::vector<float> postprocess() const;

  /// Crop the box (in ORIGINAL-image pixels) out of `src`, run the model, and
  /// return one vector. The box is expanded by `EmbedPreproc::box_expand` and
  /// clipped to the frame, so a detection hanging off the edge is fine.
  /// Throws rcdl::Error when the box is empty after clipping.
  std::vector<float> embed(const ImageView& src, float x1, float y1, float x2, float y2);
  /// Whole-frame overload, for a source that is already a crop.
  std::vector<float> embed(const ImageView& src);
  /// Convenience overload for an interleaved, row-contiguous host BGR image.
  std::vector<float> embed(const std::uint8_t* bgr, int width, int height, float x1, float y1,
                           float x2, float y2);

  /// Embedding width, from the bound output's shape.
  int dim() const noexcept { return dim_; }
  int inputWidth() const noexcept { return input_w_; }
  int inputHeight() const noexcept { return input_h_; }
  /// Backend that ran the most recent embed() preproc (tells you whether RGA
  /// fell back to the CPU).
  PreprocBackend lastBackend() const noexcept { return last_backend_; }

  const EmbedConfig& config() const noexcept { return cfg_; }
  const EmbedPreproc& preproc() const noexcept { return pre_; }

 private:
  Engine& engine_;
  /// Host staging for a FLOAT-input model, where the crop cannot be written
  /// into the NPU tensor directly. Unused (and unallocated) for a quantized one.
  std::vector<std::uint8_t> host_;
  std::vector<float> input_;
  EmbedConfig cfg_;
  EmbedPreproc pre_;
  int out_idx_;
  int dim_ = 0;
  int input_w_ = 0;
  int input_h_ = 0;
  PreprocBackend last_backend_ = PreprocBackend::Auto;
};

}  // namespace rcdl
