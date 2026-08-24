#include "rcdl/tasks/embedding.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"
#include "rcdl/preproc/rga.h"

namespace rcdl {

namespace {

/// "[1,1,1,512]" — every error in this file names the shape it choked on, so a
/// model with an unexpected head is diagnosable from the message alone.
std::string describeShape(const std::vector<int>& shape) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) os << ",";
    os << shape[i];
  }
  os << "]";
  return os.str();
}

/// Crop (x,y,w,h) out of `src` and squash-resize it into `dst`, whatever the
/// two formats are.
///
/// Two routes, and the choice is about WHO can express the offset:
///   - a source with a dma-buf fd goes through RGA's own crop rectangle, so a
///     VPU frame is cropped, scaled and colour-converted in one improcess with
///     no CPU touch and no intermediate buffer;
///   - a host-only source is re-based instead: an ImageView is only a
///     descriptor, so pointing `data` at the box's first pixel and shrinking
///     width/height (the row stride stays the SOURCE's) describes the crop
///     exactly, and the ordinary backend-agnostic resize() takes it from there.
/// The sub-view trick needs a packed format and a CPU pointer; a planar-YUV
/// host frame has a second plane at its own offset, so it goes to RGA too.
void cropResizeInto(const ImageView& dst, const ImageView& src, int x, int y, int w, int h,
                    PreprocBackend backend, YuvRange range, PreprocBackend* used) {
  // Whole-frame box: no cropping to express, so this is a plain resize and
  // every backend/format combination the preproc layer supports works.
  if (x == 0 && y == 0 && w == src.width && h == src.height) {
    resize(dst, src, backend, range, used);
    return;
  }

  const int bpp = bytesPerPixel(src.format);
  const bool can_subview = src.data != nullptr && bpp > 0;
  if (src.fd >= 0 && backend != PreprocBackend::Cpu && rgaAvailable()) {
    // RGA reads chroma at half resolution, so an odd crop origin on a planar
    // format would sample the wrong chroma pair (and the driver rejects it
    // outright). Round the origin down and the extent down to even — a
    // one-pixel shift of the crop, below the resampling error either way.
    if (bpp <= 0) {
      x &= ~1;
      y &= ~1;
      w &= ~1;
      h &= ~1;
    }
    RCDL_REQUIRE(w > 0 && h > 0, "RCDL cropResize: empty crop rectangle");
    if (!can_subview) {
      // No CPU-side fallback exists for this source, so let RGA's own error
      // (out-of-range scale, unaligned stride, ...) reach the caller.
      rgaCropResize(dst, src, x, y, w, h, range);
      if (used) *used = PreprocBackend::Rga;
      return;
    }
    try {
      rgaCropResize(dst, src, x, y, w, h, range);
      if (used) *used = PreprocBackend::Rga;
      return;
    } catch (const Error&) {
      // RGA refused this pair — a crop below its 68x2 minimum (small detection
      // boxes hit this constantly) or a scale outside [1/16,16]. The sub-view
      // path below does the same work on the CPU; a throw here is the
      // hardware's "not this one", not an error to propagate.
    }
  }

  RCDL_REQUIRE(can_subview,
               ("RCDL cropResize: cropping " + src.describe() +
                " needs either a dma-buf fd with RGA available or a CPU-mapped packed "
                "format; convert a planar YUV host frame first")
                   .c_str());
  ImageView sub = src;
  sub.fd = -1;  // the fd addresses the WHOLE buffer; the offset lives in `data`
  sub.data = src.bytePtr() + static_cast<std::size_t>(y) * src.rowBytes() +
             static_cast<std::size_t>(x) * static_cast<std::size_t>(bpp);
  sub.width = w;
  sub.height = h;
  sub.wstride = src.effWStride();  // rows still stride by the SOURCE's pitch
  sub.hstride = h;
  sub.size = 0;  // recomputed from the sub-extent by ImageView::bytes()
  resize(dst, sub, backend, range, used);
}

}  // namespace

// ---------------------------------------------------------------------------
// Decode + similarity
// ---------------------------------------------------------------------------

int embeddingDimFromShape(const std::vector<int>& shape) {
  int dim = 1;
  int non_unit = 0;
  for (int d : shape) {
    const int v = d > 0 ? d : 1;
    if (v > 1) {
      ++non_unit;
      dim = v;
    }
  }
  if (non_unit > 1) {
    throw Error(-1, "RCDL embedding: output shape " + describeShape(shape) +
                        " has more than one non-unit dimension, so it is a feature grid "
                        "rather than one pooled embedding");
  }
  return dim;
}

void l2NormalizeInPlace(float* v, int dim) {
  if (v == nullptr || dim <= 0) return;
  // Accumulate in double: a 1024-d int8-dequantized vector sums a lot of small
  // squares, and the norm is then divided into every element.
  double sum = 0.0;
  for (int i = 0; i < dim; ++i) sum += static_cast<double>(v[i]) * v[i];
  const double norm = std::sqrt(sum);
  if (!(norm > 1e-12)) return;  // zero / denormal / NaN: leave it alone
  const float inv = static_cast<float>(1.0 / norm);
  for (int i = 0; i < dim; ++i) v[i] *= inv;
}

std::vector<float> decodeEmbedding(const float* data, int dim, const EmbedConfig& cfg) {
  if (data == nullptr || dim <= 0) return {};
  std::vector<float> out(data, data + dim);
  if (cfg.l2_normalize) l2NormalizeInPlace(out.data(), dim);
  return out;
}

std::vector<float> decodeEmbedding(const float* data, const std::vector<int>& shape,
                                   const EmbedConfig& cfg) {
  return decodeEmbedding(data, embeddingDimFromShape(shape), cfg);
}

float cosineSimilarity(const float* a, const float* b, int dim) {
  if (a == nullptr || b == nullptr || dim <= 0) return 0.0f;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (int i = 0; i < dim; ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  // A zero vector has no direction; scoring it 0 keeps a missing embedding out
  // of the association rather than turning the whole cost matrix into NaNs.
  if (na <= 0.0 || nb <= 0.0) return 0.0f;
  return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
}

float cosineDistance(const float* a, const float* b, int dim) {
  return 1.0f - cosineSimilarity(a, b, dim);
}

float euclideanDistance(const float* a, const float* b, int dim) {
  if (a == nullptr || b == nullptr || dim <= 0) return 0.0f;
  double sum = 0.0;
  for (int i = 0; i < dim; ++i) {
    const double d = static_cast<double>(a[i]) - b[i];
    sum += d * d;
  }
  return static_cast<float>(std::sqrt(sum));
}

// ---------------------------------------------------------------------------
// EmbeddingBank
// ---------------------------------------------------------------------------

void EmbeddingBank::add(const std::vector<float>& vec, const std::string& label) {
  const int d = static_cast<int>(vec.size());
  RCDL_REQUIRE(d > 0, "RCDL embedding: cannot add an empty vector to a bank");
  if (dim_ == 0) {
    dim_ = d;
  } else if (d != dim_) {
    throw Error(-1, "RCDL embedding: bank dimension mismatch (bank " + std::to_string(dim_) +
                        ", added " + std::to_string(d) + ")");
  }
  data_.insert(data_.end(), vec.begin(), vec.end());
  // Normalize the row IN the bank, so search() is a bare dot product per entry
  // and the caller's vector is left as it was.
  l2NormalizeInPlace(data_.data() + data_.size() - static_cast<std::size_t>(dim_), dim_);
  labels_.push_back(label);
}

const float* EmbeddingBank::row(int i) const noexcept {
  if (i < 0 || i >= size()) return nullptr;
  return data_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(dim_);
}

void EmbeddingBank::clear() noexcept {
  dim_ = 0;
  data_.clear();
  labels_.clear();
}

std::vector<EmbedMatch> EmbeddingBank::search(const std::vector<float>& query, int k) const {
  const int n = size();
  if (n == 0) return {};
  if (static_cast<int>(query.size()) != dim_) {
    throw Error(-1, "RCDL embedding: query dimension mismatch (bank " + std::to_string(dim_) +
                        ", query " + std::to_string(query.size()) + ")");
  }

  // Normalize the query ONCE rather than once per entry: with unit rows in the
  // bank, each score is then a plain dot product.
  std::vector<float> q(query);
  l2NormalizeInPlace(q.data(), dim_);

  std::vector<EmbedMatch> all(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const float* r = data_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(dim_);
    double dot = 0.0;
    for (int c = 0; c < dim_; ++c) dot += static_cast<double>(r[c]) * q[c];
    all[static_cast<std::size_t>(i)] =
        EmbedMatch{i, static_cast<float>(dot), labels_[static_cast<std::size_t>(i)]};
  }

  const auto by_score_desc = [](const EmbedMatch& a, const EmbedMatch& b) {
    // Tie-break on index so equal scores come back in a stable, reproducible
    // order (an int8 head produces exact ties more often than a float one).
    if (a.score != b.score) return a.score > b.score;
    return a.index < b.index;
  };

  const int take = (k <= 0 || k > n) ? n : k;
  if (take < n) {
    std::partial_sort(all.begin(), all.begin() + take, all.end(), by_score_desc);
    all.resize(static_cast<std::size_t>(take));
  } else {
    std::sort(all.begin(), all.end(), by_score_desc);
  }
  return all;
}

// ---------------------------------------------------------------------------
// ImageEmbedder
// ---------------------------------------------------------------------------

ImageEmbedder::ImageEmbedder(Engine& engine, EmbedConfig cfg, EmbedPreproc pre, int output_index)
    : engine_(engine), cfg_(cfg), pre_(pre), out_idx_(output_index) {
  RCDL_REQUIRE(out_idx_ >= 0 && out_idx_ < engine_.numOutputs(),
               "RCDL embedding: output index out of range");
  // Resolve the width ONCE, from the model: a patch-feature output fails here
  // instead of yielding a vector that silently matches nothing.
  dim_ = embeddingDimFromShape(engine_.outputShape(out_idx_));

  // Input canvas from the tensor's declared layout, not guessed from dim
  // magnitudes: a 3-class NCHW input is [1,3,H,W] and a 3-channel NHWC input is
  // [1,H,W,3], which shape alone cannot tell apart.
  if (engine_.numInputs() > 0) {
    const rknn_tensor_attr& a = engine_.inputAttr(0);
    if (a.n_dims == 4) {
      const bool nchw = a.fmt == RKNN_TENSOR_NCHW;
      input_h_ = static_cast<int>(nchw ? a.dims[2] : a.dims[1]);
      input_w_ = static_cast<int>(nchw ? a.dims[3] : a.dims[2]);
    }
  }
}

std::vector<float> ImageEmbedder::postprocess() const {
  // Zero-copy for packed f32, dequant-into-scratch for the usual int8-affine
  // output. `scratch` must outlive `data`, so it stays in this scope.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);

  // Belt and braces: decodeEmbedding reads dim_ floats, so make sure the buffer
  // really holds them even if the runtime shape disagrees with construction.
  std::int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  RCDL_REQUIRE(total >= dim_,
               ("RCDL embedding: output " + describeShape(shape) + " is smaller than the " +
                std::to_string(dim_) + "-d embedding resolved at construction")
                   .c_str());
  return decodeEmbedding(data, dim_, cfg_);
}

std::vector<float> ImageEmbedder::embed(const ImageView& src, float x1, float y1, float x2,
                                        float y2) {
  RCDL_REQUIRE(src.valid(),
               ("RCDL ImageEmbedder::embed: invalid source view: " + src.describe()).c_str());
  // A quantized model takes the crop as raw bytes written straight into its
  // input tensor. A float model cannot: the bytes would be reinterpreted as
  // float32 and it would infer on garbage rather than fail. It gets the crop
  // staged through a host buffer and converted instead — which is what lets one
  // class serve both, and matters because the models that most need a float
  // build (image-text towers, ViTs) are exactly the ones int8 ruins.
  RCDL_REQUIRE(engine_.numInputs() == 1,
               "RCDL ImageEmbedder::embed: needs a single image input");
  const rknn_tensor_type in_type = engine_.inputType(0);
  RCDL_REQUIRE(in_type == RKNN_TENSOR_UINT8 || in_type == RKNN_TENSOR_FLOAT32,
               "RCDL ImageEmbedder::embed: input 0 takes neither u8 image bytes nor float32; "
               "preprocess externally and call postprocess()");

  // Optional context margin, then clip to the frame so a detection hanging off
  // the edge is croppable rather than an error.
  if (pre_.box_expand > 0.0f && pre_.box_expand != 1.0f) {
    const float cx = (x1 + x2) * 0.5f;
    const float cy = (y1 + y2) * 0.5f;
    const float hw = (x2 - x1) * 0.5f * pre_.box_expand;
    const float hh = (y2 - y1) * 0.5f * pre_.box_expand;
    x1 = cx - hw;
    x2 = cx + hw;
    y1 = cy - hh;
    y2 = cy + hh;
  }
  const int ix1 = std::max(0, static_cast<int>(std::floor(std::min(x1, x2))));
  const int iy1 = std::max(0, static_cast<int>(std::floor(std::min(y1, y2))));
  const int ix2 = std::min(src.width, static_cast<int>(std::ceil(std::max(x1, x2))));
  const int iy2 = std::min(src.height, static_cast<int>(std::ceil(std::max(y1, y2))));
  RCDL_REQUIRE(ix2 > ix1 && iy2 > iy1,
               "RCDL ImageEmbedder::embed: box is empty after clipping to the frame");

  if (in_type == RKNN_TENSOR_UINT8) {
    // The crop destination IS the NPU's input tensor: engineInputView() hands
    // its fd + virtual address + row stride to the preproc layer, so on the
    // hardware path RGA writes the model's input directly with no intermediate
    // canvas.
    const ImageView dst = engineInputView(engine_, 0, pre_.model_input);
    cropResizeInto(dst, src, ix1, iy1, ix2 - ix1, iy2 - iy1, pre_.backend, pre_.yuv_range,
                   &last_backend_);
  } else {
    // Float build: crop into a host buffer at the model's size, then widen. The
    // values stay 0..255 — a float export's mean/std is folded into the .rknn
    // exactly as a quantized one's is, and dividing here would hand it a picture
    // 255x too dark that still returns a well-formed unit vector.
    const int w = inputWidth();
    const int h = inputHeight();
    const int bpp = bytesPerPixel(pre_.model_input);
    host_.resize(static_cast<std::size_t>(w) * h * bpp);
    ImageView dst = hostView(host_.data(), w, h, pre_.model_input, w, h);
    cropResizeInto(dst, src, ix1, iy1, ix2 - ix1, iy2 - iy1, pre_.backend, pre_.yuv_range,
                   &last_backend_);
    input_.resize(host_.size());
    for (std::size_t i = 0; i < host_.size(); ++i) {
      input_[i] = static_cast<float>(host_[i]);
    }
    engine_.setInput(0, input_.data(), input_.size() * sizeof(float));
  }

  engine_.infer();
  return postprocess();
}

std::vector<float> ImageEmbedder::embed(const ImageView& src) {
  return embed(src, 0.0f, 0.0f, static_cast<float>(src.width), static_cast<float>(src.height));
}

std::vector<float> ImageEmbedder::embed(const std::uint8_t* bgr, int width, int height, float x1,
                                        float y1, float x2, float y2) {
  RCDL_REQUIRE(bgr != nullptr && width > 0 && height > 0,
               "RCDL ImageEmbedder::embed: invalid BGR frame");
  // ImageView is a non-owning descriptor and preproc only READS the source, so
  // this const_cast is a description-level cast, not a write.
  return embed(hostView(const_cast<std::uint8_t*>(bgr), width, height, PixelFormat::BGR888), x1,
               y1, x2, y2);
}

}  // namespace rcdl
