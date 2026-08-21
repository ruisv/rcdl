#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "rcdl/preproc/image.h"

namespace rcdl {

/// Crop preprocessing parameters for a ReID model. Defaults are the torchreid
/// convention (RGB, /255, ImageNet mean/std) that OSNet and its relatives were
/// trained with; a model trained otherwise needs its own numbers.
struct ReidConfig {
  float mean[3] = {0.485f, 0.456f, 0.406f};
  float std[3] = {0.229f, 0.224f, 0.225f};
};

/// Cut a detection box out of a BGR frame and turn it into the model's input:
/// `[1,3,in_h,in_w]` float32 NCHW, RGB, normalized, written to `out`.
///
/// **This is a squashing resize, deliberately NOT a letterbox.** Person-ReID
/// models are trained on crops squashed to a fixed 2:1 shape from whatever
/// aspect the box happened to have, so preserving aspect would feed them
/// padding bars they have never seen. An embedding has no coordinates, so
/// unlike the detection path there is no geometry to invert afterwards — which
/// is why this returns nothing.
///
/// `bgr`     : interleaved HxWx3 uint8, `stride` bytes per row.
/// `bx*/by*` : the detection box in ORIGINAL-image pixels; it is clipped to the
///             frame, so a box hanging off the edge is fine.
/// Sampling uses the OpenCV pixel-center convention, matching
/// cv::resize(INTER_LINEAR) and the Python `rcdl.reid_preprocess`.
///
/// This is a CPU path on purpose: the crops are small (a 128x256 person box is
/// ~100k output floats) and the result is NCHW float32, which RGA cannot
/// produce. Use it when the ReID model is a separate float input; the detector's
/// own letterbox stays on RGA.
///
/// Throws Error(-1) on bad dimensions or a box that is empty after clipping.
void reidPreprocess(const uint8_t* bgr, int width, int height, int stride,
                    float bx1, float by1, float bx2, float by2, int in_w, int in_h,
                    const ReidConfig& cfg, std::vector<float>& out);

/// ImageView overload of the above, for the common case where the frame is
/// already described as an RCDL image (a `hostView()` over a cv::Mat, or a
/// CPU-mapped `Image`). `src` must be a packed 3-byte format — BGR888 or
/// RGB888; the channel order is read from `src.format`, so an RGB source is not
/// swapped twice. The maths is identical to the raw-pointer form.
///
/// Throws Error(-1) when `src` has no CPU mapping or is not BGR888 / RGB888
/// (a planar YUV frame must be converted first — RGA or the media path does
/// that on the way out of the decoder).
void reidPreprocess(const ImageView& src, float bx1, float by1, float bx2, float by2,
                    int in_w, int in_h, const ReidConfig& cfg, std::vector<float>& out);

// ===========================================================================
// ReID appearance embeddings (BoT-SORT / StrongSORT association)
// ===========================================================================
//
// A lightweight ReID head (OSNet / fast-reid distillation) turns each detection
// crop into a D-dim appearance vector. For appearance-gated tracking the vector
// is L2-normalized so that cosine similarity is a plain dot product; matching a
// track's smoothed embedding to a detection is then `1 - dot`. These are the
// pure, board-independent primitives the tracker calls — the full BoT-SORT
// association (IoU + appearance cost matrix, EMA-smoothed gallery) is layered on
// top in tracks/byte_tracker, and switches on as soon as a ReID `.rknn` is
// available to fill the embeddings in.

/// L2-normalize an embedding in place (no-op for a zero vector). After this,
/// cosineSimilarity() with another normalized vector is a dot product.
inline void l2Normalize(std::vector<float>& v) {
  double sum = 0.0;
  for (float x : v) sum += static_cast<double>(x) * x;
  const double norm = std::sqrt(sum);
  if (norm <= 0.0) return;
  const float inv = static_cast<float>(1.0 / norm);
  for (float& x : v) x *= inv;
}

/// Return an L2-normalized copy of a raw embedding (e.g. straight off the
/// engine output).
inline std::vector<float> normalizeEmbedding(const float* data, int dim) {
  std::vector<float> v;
  if (data == nullptr || dim <= 0) return v;
  v.assign(data, data + dim);
  l2Normalize(v);
  return v;
}

/// Cosine similarity of two equal-length embeddings in [-1, 1]. Computes the
/// normalized dot product, so the inputs need NOT be pre-normalized; returns 0
/// on length mismatch or a zero vector.
inline float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0f;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

}  // namespace rcdl
