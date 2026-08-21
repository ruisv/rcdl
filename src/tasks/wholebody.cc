#include "rcdl/tasks/wholebody.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

namespace {

// COCO-WholeBody boundaries, in index order.
constexpr int kBodyEnd = 17;
constexpr int kFootEnd = 23;
constexpr int kFaceEnd = 91;
constexpr int kLeftHandEnd = 112;
constexpr int kRightHandEnd = 133;

/// Bilinear sample of one channel of an interleaved 8-bit image, with a
/// constant outside — the crop legitimately hangs off the frame when a person
/// stands at the edge, and clamping the edge pixel instead would smear a
/// column of skin across the padding.
inline float sampleChannel(const std::uint8_t* base, int w, int h, std::size_t stride, int bpp,
                           float x, float y, int c, std::uint8_t pad) {
  const int x0 = static_cast<int>(std::floor(x)), y0 = static_cast<int>(std::floor(y));
  const float fx = x - x0, fy = y - y0;
  float acc = 0.0f;
  for (int j = 0; j < 2; ++j) {
    const int sy = y0 + j;
    const float wy = j ? fy : 1.0f - fy;
    for (int i = 0; i < 2; ++i) {
      const int sx = x0 + i;
      const float wx = i ? fx : 1.0f - fx;
      const float wgt = wy * wx;
      if (wgt == 0.0f) continue;
      const bool inside = sx >= 0 && sx < w && sy >= 0 && sy < h;
      acc += wgt * (inside ? base[static_cast<std::size_t>(sy) * stride + sx * bpp + c] : pad);
    }
  }
  return acc;
}

}  // namespace

BodyPart bodyPart(int i) noexcept {
  if (i < kBodyEnd) return BodyPart::Body;
  if (i < kFootEnd) return BodyPart::Foot;
  if (i < kFaceEnd) return BodyPart::Face;
  if (i < kLeftHandEnd) return BodyPart::LeftHand;
  return BodyPart::RightHand;
}

const char* bodyPartName(BodyPart p) noexcept {
  switch (p) {
    case BodyPart::Body: return "body";
    case BodyPart::Foot: return "foot";
    case BodyPart::Face: return "face";
    case BodyPart::LeftHand: return "left_hand";
    case BodyPart::RightHand: return "right_hand";
  }
  return "unknown";
}

void bodyPartRange(BodyPart p, int* begin, int* end) noexcept {
  int b = 0, e = 0;
  switch (p) {
    case BodyPart::Body: b = 0; e = kBodyEnd; break;
    case BodyPart::Foot: b = kBodyEnd; e = kFootEnd; break;
    case BodyPart::Face: b = kFootEnd; e = kFaceEnd; break;
    case BodyPart::LeftHand: b = kFaceEnd; e = kLeftHandEnd; break;
    case BodyPart::RightHand: b = kLeftHandEnd; e = kRightHandEnd; break;
  }
  if (begin) *begin = b;
  if (end) *end = e;
}

CropRect cropGeometry(float x1, float y1, float x2, float y2, int in_w, int in_h,
                      float padding) {
  RCDL_REQUIRE(in_w > 0 && in_h > 0, "cropGeometry: the model input has no size");
  RCDL_REQUIRE(padding > 0.0f, "cropGeometry: padding must be positive");
  CropRect r;
  r.cx = (x1 + x2) * 0.5f;
  r.cy = (y1 + y2) * 0.5f;
  float w = std::abs(x2 - x1) * padding;
  float h = std::abs(y2 - y1) * padding;
  // Grow the short axis, never shrink the long one: cropping to fit would cut
  // off the hands and feet this head exists to find.
  const float aspect = static_cast<float>(in_w) / static_cast<float>(in_h);
  if (w > h * aspect) {
    h = w / aspect;
  } else {
    w = h * aspect;
  }
  r.w = std::max(w, 1.0f);
  r.h = std::max(h, 1.0f);
  return r;
}

std::vector<Keypoint> decodeSimcc(const float* simcc_x, const float* simcc_y, int num_kpts,
                                  int bins_x, int bins_y, const CropRect& crop, int in_w,
                                  int in_h, const WholeBodyConfig& cfg) {
  RCDL_REQUIRE(simcc_x != nullptr && simcc_y != nullptr, "decodeSimcc: null tensor");
  RCDL_REQUIRE(num_kpts > 0 && bins_x > 0 && bins_y > 0, "decodeSimcc: empty SimCC tensors");
  RCDL_REQUIRE(cfg.split_ratio > 0.0f, "decodeSimcc: split_ratio must be positive");
  RCDL_REQUIRE(in_w > 0 && in_h > 0, "decodeSimcc: the model input has no size");

  std::vector<Keypoint> out;
  out.reserve(num_kpts);
  const float sx = crop.w / static_cast<float>(in_w);
  const float sy = crop.h / static_cast<float>(in_h);
  for (int k = 0; k < num_kpts; ++k) {
    const float* rx = simcc_x + static_cast<std::size_t>(k) * bins_x;
    const float* ry = simcc_y + static_cast<std::size_t>(k) * bins_y;
    int bx = 0, by = 0;
    for (int i = 1; i < bins_x; ++i) {
      if (rx[i] > rx[bx]) bx = i;
    }
    for (int i = 1; i < bins_y; ++i) {
      if (ry[i] > ry[by]) by = i;
    }
    // Bins -> model-input pixels -> source pixels.
    const float mx = bx / cfg.split_ratio;
    const float my = by / cfg.split_ratio;
    const float score = 0.5f * (rx[bx] + ry[by]);
    Keypoint kp{crop.x0() + mx * sx, crop.y0() + my * sy, score};
    if (score < cfg.kpt_thresh) {
      kp.x = -1.0f;
      kp.y = -1.0f;
    }
    out.push_back(kp);
  }
  return out;
}

WholeBodyEstimator::WholeBodyEstimator(Engine& engine, WholeBodyConfig cfg)
    : engine_(engine), cfg_(cfg) {
  const std::vector<int> is = engine.inputShape(0);
  RCDL_REQUIRE(is.size() == 4, "WholeBodyEstimator: expected a 4-D image input");
  const bool nhwc = engine.inputFormat(0) == RKNN_TENSOR_NHWC;
  in_h_ = nhwc ? is[1] : is[2];
  in_w_ = nhwc ? is[2] : is[3];
  RCDL_REQUIRE(in_w_ > 0 && in_h_ > 0, "WholeBodyEstimator: empty model input size");
  RCDL_REQUIRE(engine.numOutputs() >= 2,
               "WholeBodyEstimator: a SimCC head has two outputs (simcc_x, simcc_y)");

  const std::vector<int> xs = engine.outputShape(0);
  const std::vector<int> ys = engine.outputShape(1);
  RCDL_REQUIRE(xs.size() == 3 && ys.size() == 3 && xs[1] == ys[1],
               "WholeBodyEstimator: expected [1,K,bins] SimCC outputs sharing K");
  num_kpts_ = xs[1];
  bins_x_ = xs[2];
  bins_y_ = ys[2];
  // The x axis is the narrow one. Reversed outputs decode into a person rotated
  // into a corner, which is obvious on a picture and invisible in a shape check.
  RCDL_REQUIRE(bins_x_ < bins_y_,
               "WholeBodyEstimator: simcc_x should have fewer bins than simcc_y — are the "
               "two outputs the other way round?");
  RCDL_REQUIRE(std::abs(bins_x_ / cfg_.split_ratio - in_w_) < 1.0f &&
                   std::abs(bins_y_ / cfg_.split_ratio - in_h_) < 1.0f,
               "WholeBodyEstimator: the SimCC bin counts do not match the input size at this "
               "split_ratio");
}

std::vector<Keypoint> WholeBodyEstimator::estimate(const ImageView& src, float x1, float y1,
                                                   float x2, float y2) {
  RCDL_REQUIRE(src.data != nullptr && src.width > 0 && src.height > 0,
               "WholeBodyEstimator: the source needs a CPU mapping");
  const int bpp = bytesPerPixel(src.format);
  RCDL_REQUIRE(bpp == 3 || bpp == 4,
               "WholeBodyEstimator: expected a packed RGB/BGR source image");
  const bool src_bgr =
      src.format == PixelFormat::BGR888 || src.format == PixelFormat::BGRA8888;

  last_crop_ = cropGeometry(x1, y1, x2, y2, in_w_, in_h_, cfg_.padding);

  // The transform is a crop that may hang off the frame on any side, which the
  // hardware letterbox cannot express (it centres a whole image), so the resample
  // is done here. It is 192x256 samples — irrelevant next to the inference.
  input_.resize(static_cast<std::size_t>(in_w_) * in_h_ * 3);
  const std::uint8_t* base = static_cast<const std::uint8_t*>(src.data);
  const std::size_t stride = src.rowBytes();
  const float sx = last_crop_.w / static_cast<float>(in_w_);
  const float sy = last_crop_.h / static_cast<float>(in_h_);
  for (int y = 0; y < in_h_; ++y) {
    const float syy = last_crop_.y0() + (y + 0.5f) * sy - 0.5f;
    float* row = input_.data() + static_cast<std::size_t>(y) * in_w_ * 3;
    for (int x = 0; x < in_w_; ++x) {
      const float sxx = last_crop_.x0() + (x + 0.5f) * sx - 0.5f;
      // The model takes RGB, and 0..255: its mean/std are folded into the .rknn.
      for (int c = 0; c < 3; ++c) {
        const int chan = src_bgr ? 2 - c : c;
        row[x * 3 + c] =
            sampleChannel(base, src.width, src.height, stride, bpp, sxx, syy, chan, cfg_.pad);
      }
    }
  }
  engine_.setInput(0, input_.data(), input_.size() * sizeof(float));
  engine_.infer();
  return postprocess(last_crop_);
}

std::vector<Keypoint> WholeBodyEstimator::postprocess(const CropRect& crop) const {
  std::vector<float> sx_scratch, sy_scratch;
  std::vector<int> sx_shape, sy_shape;
  const float* sx = outputAsFloat(engine_, 0, sx_scratch, sx_shape);
  const float* sy = outputAsFloat(engine_, 1, sy_scratch, sy_shape);
  return decodeSimcc(sx, sy, num_kpts_, bins_x_, bins_y_, crop, in_w_, in_h_, cfg_);
}

}  // namespace rcdl
