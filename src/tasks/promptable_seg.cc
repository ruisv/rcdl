#include "rcdl/tasks/promptable_seg.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"
#include "rcdl/preproc/letterbox.h"

namespace rcdl {

namespace {

/// Bilinear sample of a row-major h*w float map, clamped at the edges.
float sampleMap(const float* m, int w, int h, float x, float y) {
  x = std::clamp(x, 0.0f, static_cast<float>(w - 1));
  y = std::clamp(y, 0.0f, static_cast<float>(h - 1));
  const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
  const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
  const float fx = x - x0, fy = y - y0;
  const float a = m[static_cast<std::size_t>(y0) * w + x0];
  const float b = m[static_cast<std::size_t>(y0) * w + x1];
  const float c = m[static_cast<std::size_t>(y1) * w + x0];
  const float d = m[static_cast<std::size_t>(y1) * w + x1];
  return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
}

}  // namespace

float PromptMask::area() const {
  if (data.empty()) return 0.0f;
  std::size_t on = 0;
  for (std::uint8_t v : data) on += v ? 1u : 0u;
  return static_cast<float>(on) / static_cast<float>(data.size());
}

void encodeBoxPrompt(float x1, float y1, float x2, float y2, const LetterboxInfo& lb,
                     float* coords, float* labels) {
  RCDL_REQUIRE(coords != nullptr && labels != nullptr, "encodeBoxPrompt: null output");
  coords[0] = lb.fwdX(x1);
  coords[1] = lb.fwdY(y1);
  coords[2] = lb.fwdX(x2);
  coords[3] = lb.fwdY(y2);
  labels[0] = 2.0f;  // SAM: box top-left
  labels[1] = 3.0f;  // SAM: box bottom-right
}

void encodePointPrompt(float x, float y, bool positive, const LetterboxInfo& lb, float* coords,
                       float* labels) {
  RCDL_REQUIRE(coords != nullptr && labels != nullptr, "encodePointPrompt: null output");
  coords[0] = lb.fwdX(x);
  coords[1] = lb.fwdY(y);
  // The padding point is what lets one exported graph serve both prompt kinds:
  // its label -1 tells the decoder to ignore it.
  coords[2] = 0.0f;
  coords[3] = 0.0f;
  labels[0] = positive ? 1.0f : 0.0f;
  labels[1] = -1.0f;
}

PromptMask maskFromLogits(const float* logits, int w, int h, const LetterboxInfo& lb,
                          float thresh, float score) {
  RCDL_REQUIRE(logits != nullptr && w > 0 && h > 0, "maskFromLogits: empty logit map");
  RCDL_REQUIRE(lb.srcW > 0 && lb.srcH > 0 && lb.dstW > 0 && lb.dstH > 0,
               "maskFromLogits: the letterbox has no geometry — call setImage() first");

  PromptMask out;
  out.width = lb.srcW;
  out.height = lb.srcH;
  out.score = score;
  out.data.assign(static_cast<std::size_t>(lb.srcW) * lb.srcH, 0);

  // Every SOURCE pixel centre is pushed forward through the letterbox and
  // sampled back out of the map, so the padding never contributes and no
  // intermediate image is built — the same projection depthToSource() uses for
  // a depth map, which this is: a float map over the model canvas.
  const float mx = static_cast<float>(w) / static_cast<float>(lb.dstW);
  const float my = static_cast<float>(h) / static_cast<float>(lb.dstH);
  int x0 = lb.srcW, y0 = lb.srcH, x1 = 0, y1 = 0;
  for (int y = 0; y < lb.srcH; ++y) {
    const float sy = lb.fwdY(y + 0.5f) * my - 0.5f;
    std::uint8_t* row = out.data.data() + static_cast<std::size_t>(y) * lb.srcW;
    for (int x = 0; x < lb.srcW; ++x) {
      const float sx = lb.fwdX(x + 0.5f) * mx - 0.5f;
      if (sampleMap(logits, w, h, sx, sy) > thresh) {
        row[x] = 1;
        x0 = std::min(x0, x);
        y0 = std::min(y0, y);
        x1 = std::max(x1, x + 1);
        y1 = std::max(y1, y + 1);
      }
    }
  }
  if (x1 > x0 && y1 > y0) {
    out.x0 = x0;
    out.y0 = y0;
    out.x1 = x1;
    out.y1 = y1;
  }
  return out;
}

PromptableSegmenter::PromptableSegmenter(Engine& encoder, Engine& decoder, PromptConfig cfg)
    : encoder_(encoder), decoder_(decoder), cfg_(cfg) {
  const std::vector<int> es = encoder.inputShape(0);
  RCDL_REQUIRE(es.size() == 4, "PromptableSegmenter: the encoder wants a 4-D image tensor");
  const bool nhwc = encoder.inputFormat(0) == RKNN_TENSOR_NHWC;
  in_h_ = nhwc ? es[1] : es[2];
  in_w_ = nhwc ? es[2] : es[3];
  RCDL_REQUIRE(in_w_ > 0 && in_h_ > 0, "PromptableSegmenter: empty encoder input size");

  RCDL_REQUIRE(decoder.numInputs() >= 3,
               "PromptableSegmenter: the decoder takes (embedding, point_coords, point_labels)");
  RCDL_REQUIRE(decoder.numOutputs() >= 2,
               "PromptableSegmenter: the decoder returns (scores, masks)");
  const std::vector<int> ms = decoder.outputShape(1);
  RCDL_REQUIRE(ms.size() == 4, "PromptableSegmenter: expected [1,N,H,W] masks");
  num_masks_ = ms[1];
  mask_h_ = ms[2];
  mask_w_ = ms[3];
  RCDL_REQUIRE(num_masks_ >= 1 && mask_w_ > 0 && mask_h_ > 0,
               "PromptableSegmenter: the decoder's mask output has no geometry");

  // Both models are float here (see the header: the int8 encoder loses small
  // objects entirely), and the prompt tensors could not be image bytes in any
  // case — they are coordinates in a 1024-pixel canvas.
  RCDL_REQUIRE(encoder.inputType(0) == RKNN_TENSOR_FLOAT32,
               "PromptableSegmenter: this encoder's input is not float32 — an int8 build "
               "would take image bytes, and its embeddings are not usable (docs/MODELS.md)");
  for (int i = 1; i <= 2; ++i) {
    RCDL_REQUIRE(decoder.inputType(i) == RKNN_TENSOR_FLOAT32,
                 "PromptableSegmenter: the decoder's prompt inputs must be float32");
  }
}

void PromptableSegmenter::setImage(const ImageView& src, PreprocBackend backend) {
  RCDL_REQUIRE(src.width > 0 && src.height > 0, "setImage: empty source image");
  if (!stage_.valid() || stage_.width() != in_w_ || stage_.height() != in_h_) {
    // A dma-buf canvas so RGA can do the resize; the CPU then widens it into the
    // float tensor. The encoder is a FLOAT model, so its input cannot be written
    // by the hardware directly the way a quantized model's can.
    stage_ = Image::alloc(in_w_, in_h_, PixelFormat::RGB888);
    RCDL_REQUIRE(stage_.valid(), "setImage: could not allocate the encoder canvas");
  }
  const ImageView dst = stage_.view();
  lb_ = rcdl::letterbox(dst, src, cfg_.pad, backend, YuvRange::kStudioToFull, &last_backend_);

  stage_.syncStart(/*read=*/true, /*write=*/false);
  input_.resize(static_cast<std::size_t>(in_w_) * in_h_ * 3);
  const std::uint8_t* base = static_cast<const std::uint8_t*>(dst.data);
  const std::size_t stride = dst.rowBytes();
  for (int y = 0; y < in_h_; ++y) {
    const std::uint8_t* row = base + static_cast<std::size_t>(y) * stride;
    float* out = input_.data() + static_cast<std::size_t>(y) * in_w_ * 3;
    // 0..255, not 0..1: the mean/std normalisation is folded into the .rknn, so
    // the tensor wants the pixel values themselves.
    for (int i = 0; i < in_w_ * 3; ++i) out[i] = row[i];
  }
  stage_.syncEnd(/*read=*/true, /*write=*/false);

  encoder_.setInput(0, input_.data(), input_.size() * sizeof(float));
  encoder_.infer();

  // The encoder emits [1,256,64,64] and the decoder's embedding input is
  // reported NHWC, so the planes have to be interleaved on the way across.
  // Handing the planar buffer over unchanged is accepted by every shape check
  // and produces masks that are simply wrong.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* emb = outputAsFloat(encoder_, 0, scratch, shape);
  RCDL_REQUIRE(shape.size() == 4, "setImage: expected a [1,C,H,W] embedding");
  const int c = shape[1], h = shape[2], w = shape[3];
  const std::size_t plane = static_cast<std::size_t>(h) * w;
  const bool dec_nhwc = decoder_.inputFormat(0) == RKNN_TENSOR_NHWC;
  embedding_.resize(plane * c);
  if (dec_nhwc) {
    for (std::size_t p = 0; p < plane; ++p) {
      for (int ch = 0; ch < c; ++ch) embedding_[p * c + ch] = emb[ch * plane + p];
    }
  } else {
    std::copy(emb, emb + plane * c, embedding_.begin());
  }
  have_image_ = true;
}

void PromptableSegmenter::runDecoder(const float* coords, const float* labels) {
  RCDL_REQUIRE(have_image_, "PromptableSegmenter: call setImage() before prompting");
  decoder_.setInput(0, embedding_.data(), embedding_.size() * sizeof(float));
  decoder_.setInput(1, coords, 4 * sizeof(float));
  decoder_.setInput(2, labels, 2 * sizeof(float));
  decoder_.infer();

  std::vector<float> s_scratch, m_scratch;
  std::vector<int> s_shape, m_shape;
  const float* scores = outputAsFloat(decoder_, 0, s_scratch, s_shape);
  const float* masks = outputAsFloat(decoder_, 1, m_scratch, m_shape);
  scores_.assign(scores, scores + num_masks_);
  logits_.assign(masks, masks + static_cast<std::size_t>(num_masks_) * mask_w_ * mask_h_);
}

PromptMask PromptableSegmenter::box(float x1, float y1, float x2, float y2) {
  float coords[4], labels[2];
  encodeBoxPrompt(x1, y1, x2, y2, lb_, coords, labels);
  runDecoder(coords, labels);
  return best();
}

PromptMask PromptableSegmenter::point(float x, float y, bool positive) {
  float coords[4], labels[2];
  encodePointPrompt(x, y, positive, lb_, coords, labels);
  runDecoder(coords, labels);
  return best();
}

PromptMask PromptableSegmenter::best() const {
  // Only the selected mask is projected. Each projection walks the whole source
  // frame, so building all four to return one costs four times as much CPU as
  // the answer needs — and on a 1080p frame that is not small.
  int pick = 0;
  if (cfg_.multimask) {
    for (int i = 1; i < num_masks_; ++i) {
      if (scores_[i] > scores_[pick]) pick = i;
    }
  }
  const std::size_t plane = static_cast<std::size_t>(mask_w_) * mask_h_;
  return maskFromLogits(logits_.data() + pick * plane, mask_w_, mask_h_, lb_, cfg_.mask_thresh,
                        scores_[pick]);
}

std::vector<PromptMask> PromptableSegmenter::masks() const {
  RCDL_REQUIRE(!logits_.empty(), "PromptableSegmenter::masks: no prompt has been run");
  const std::size_t plane = static_cast<std::size_t>(mask_w_) * mask_h_;
  std::vector<int> order(num_masks_);
  for (int i = 0; i < num_masks_; ++i) order[i] = i;
  if (cfg_.multimask) {
    std::sort(order.begin(), order.end(),
              [this](int a, int b) { return scores_[a] > scores_[b]; });
  }
  std::vector<PromptMask> out;
  out.reserve(order.size());
  for (int i : order) {
    out.push_back(maskFromLogits(logits_.data() + i * plane, mask_w_, mask_h_, lb_,
                                 cfg_.mask_thresh, scores_[i]));
  }
  return out;
}

}  // namespace rcdl
