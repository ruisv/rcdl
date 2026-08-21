#include "rcdl/tasks/superres.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

std::vector<TilePlacement> planTiles(int width, int height, int tile_w, int tile_h,
                                     int overlap) {
  RCDL_REQUIRE(tile_w > 0 && tile_h > 0, "planTiles: tile must be positive");
  RCDL_REQUIRE(overlap >= 0 && overlap < std::min(tile_w, tile_h),
               "planTiles: overlap must be in [0, min(tile_w, tile_h))");

  auto axis = [&](int extent, int tile) {
    std::vector<int> origins;
    if (extent <= tile) {
      origins.push_back(0);
      return origins;
    }
    const int step = tile - overlap;
    for (int p = 0; p + tile < extent; p += step) origins.push_back(p);
    origins.push_back(extent - tile);  // flush against the far edge
    return origins;
  };

  const std::vector<int> xs = axis(width, tile_w);
  const std::vector<int> ys = axis(height, tile_h);
  std::vector<TilePlacement> out;
  out.reserve(xs.size() * ys.size());
  for (int y : ys) {
    for (int x : xs) out.push_back({x, y});
  }
  return out;
}

float tileWeight(int i, int len, int ramp) {
  if (ramp <= 0) return 1.0f;
  const int d = std::min(i + 1, len - i);  // distance to the nearer end, 1-based
  if (d >= ramp) return 1.0f;
  // Strictly positive at the very edge so the normalized blend stays defined.
  return std::max(static_cast<float>(d) / static_cast<float>(ramp), 1e-3f);
}

SuperResolver::SuperResolver(Engine& engine, SuperResConfig cfg, int input_index,
                             int output_index)
    : engine_(engine), cfg_(cfg), in_idx_(input_index), out_idx_(output_index) {
  RCDL_REQUIRE(input_index >= 0 && input_index < engine.numInputs() && output_index >= 0 &&
                   output_index < engine.numOutputs(),
               "SuperResolver: tensor index out of range");
  const std::vector<int> is = engine.inputShape(input_index);
  const std::vector<int> os = engine.outputShape(output_index);
  RCDL_REQUIRE(is.size() == 4 && os.size() == 4,
               "SuperResolver: expected 4-D image tensors in and out");

  // The toolkit reports a quantized image input as NHWC even when the ONNX was
  // NCHW, so read the spatial dims through the format. The OUTPUT keeps the
  // model's own NCHW planar layout, which is what the read-out below assumes.
  const bool in_nhwc = engine.inputFormat(input_index) == RKNN_TENSOR_NHWC;
  tile_h_ = in_nhwc ? is[1] : is[2];
  tile_w_ = in_nhwc ? is[2] : is[3];
  const int in_c = in_nhwc ? is[3] : is[1];
  RCDL_REQUIRE(engine.outputAttr(output_index).fmt == RKNN_TENSOR_NCHW,
               "SuperResolver: the upscaled tile is read as planar NCHW; this model's "
               "output is not");
  RCDL_REQUIRE(in_c == 3 && os[1] == 3, "SuperResolver: expected 3-channel tensors");
  RCDL_REQUIRE(tile_w_ > 0 && tile_h_ > 0, "SuperResolver: model input has an empty tile size");
  RCDL_REQUIRE(os[2] % tile_h_ == 0 && os[3] % tile_w_ == 0 &&
                   os[2] / tile_h_ == os[3] / tile_w_,
               "SuperResolver: output is not an integer multiple of the input");
  scale_ = os[2] / tile_h_;
  RCDL_REQUIRE(scale_ >= 1, "SuperResolver: bad scale factor");
  RCDL_REQUIRE(cfg_.overlap >= 0 && cfg_.overlap < std::min(tile_w_, tile_h_),
               "SuperResolver: overlap must be in [0, tile)");

  // A quantized build takes image bytes; a float build takes float32 — of the
  // SAME 0..255 values, because the /255 is folded into the model. Both are
  // written by writeTile(); the type is the only difference.
  const rknn_tensor_type t = engine.inputType(input_index);
  float_input_ = t == RKNN_TENSOR_FLOAT32;
  RCDL_REQUIRE(float_input_ || t == RKNN_TENSOR_UINT8,
               "SuperResolver: model input is neither u8 image bytes nor float32");
}

void SuperResolver::writeTile(const std::uint8_t* src, int src_stride, int tx, int ty) {
  const std::size_t n = static_cast<std::size_t>(tile_w_) * tile_h_ * 3;
  if (float_input_) {
    input_f32_.resize(n);
  } else {
    input_u8_.resize(n);
  }
  for (int y = 0; y < tile_h_; ++y) {
    const std::uint8_t* row = src + static_cast<std::size_t>(ty + y) * src_stride + tx * 3;
    const std::size_t o = static_cast<std::size_t>(y) * tile_w_ * 3;
    for (int x = 0; x < tile_w_; ++x) {
      // BGR -> RGB, interleaved: the tensor layout is NHWC on both builds.
      const std::uint8_t r = row[x * 3 + 2], g = row[x * 3 + 1], b = row[x * 3 + 0];
      if (float_input_) {
        input_f32_[o + x * 3 + 0] = r;
        input_f32_[o + x * 3 + 1] = g;
        input_f32_[o + x * 3 + 2] = b;
      } else {
        input_u8_[o + x * 3 + 0] = r;
        input_u8_[o + x * 3 + 1] = g;
        input_u8_[o + x * 3 + 2] = b;
      }
    }
  }
  if (float_input_) {
    engine_.setInput(in_idx_, input_f32_.data(), input_f32_.size() * sizeof(float));
  } else {
    engine_.setInput(in_idx_, input_u8_.data(), input_u8_.size());
  }
}

SrImage SuperResolver::upscale(const std::uint8_t* bgr, int width, int height, int stride) {
  RCDL_REQUIRE(bgr != nullptr && width > 0 && height > 0,
               "SuperResolver::upscale: bad image dimensions");
  if (stride <= 0) stride = width * 3;

  // An image smaller than one tile is edge-replicated up to tile size; the
  // padding is cropped off the result, so it only ever feeds the receptive
  // field. Zero padding would put a hard black edge inside it.
  const int src_w = std::max(width, tile_w_);
  const int src_h = std::max(height, tile_h_);
  std::vector<std::uint8_t> padded;
  const std::uint8_t* src = bgr;
  int src_stride = stride;
  if (src_w != width || src_h != height) {
    padded.resize(static_cast<std::size_t>(src_w) * src_h * 3);
    for (int y = 0; y < src_h; ++y) {
      const int sy = std::min(y, height - 1);
      for (int x = 0; x < src_w; ++x) {
        const int sx = std::min(x, width - 1);
        const std::uint8_t* p = bgr + static_cast<std::size_t>(sy) * stride + sx * 3;
        std::uint8_t* q = padded.data() + (static_cast<std::size_t>(y) * src_w + x) * 3;
        q[0] = p[0];
        q[1] = p[1];
        q[2] = p[2];
      }
    }
    src = padded.data();
    src_stride = src_w * 3;
  }

  const int out_w = width * scale_;
  const int out_h = height * scale_;
  const int pad_out_w = src_w * scale_;
  const int pad_out_h = src_h * scale_;

  std::vector<float> acc(static_cast<std::size_t>(pad_out_w) * pad_out_h * 3, 0.0f);
  std::vector<float> wsum(static_cast<std::size_t>(pad_out_w) * pad_out_h, 0.0f);

  const std::vector<TilePlacement> plan =
      planTiles(src_w, src_h, tile_w_, tile_h_, cfg_.overlap);
  last_tiles_ = static_cast<int>(plan.size());

  const int ramp = cfg_.overlap * scale_;
  std::vector<float> scratch;
  std::vector<int> oshape;
  for (const TilePlacement& t : plan) {
    writeTile(src, src_stride, t.x, t.y);
    engine_.infer();
    const float* sr = outputAsFloat(engine_, out_idx_, scratch, oshape);

    const int th = tile_h_ * scale_, tw = tile_w_ * scale_;
    const std::size_t opx = static_cast<std::size_t>(th) * tw;
    const int ox = t.x * scale_, oy = t.y * scale_;
    for (int y = 0; y < th; ++y) {
      const float wy = tileWeight(y, th, ramp);
      for (int x = 0; x < tw; ++x) {
        const float w = wy * tileWeight(x, tw, ramp);
        const std::size_t o = static_cast<std::size_t>(y) * tw + x;
        const std::size_t d = static_cast<std::size_t>(oy + y) * pad_out_w + (ox + x);
        acc[d * 3 + 0] += w * sr[2 * opx + o];   // B
        acc[d * 3 + 1] += w * sr[opx + o];       // G
        acc[d * 3 + 2] += w * sr[o];             // R
        wsum[d] += w;
      }
    }
  }

  SrImage out;
  out.width = out_w;
  out.height = out_h;
  out.data.resize(static_cast<std::size_t>(out_w) * out_h * 3);
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const std::size_t s = static_cast<std::size_t>(y) * pad_out_w + x;
      const std::size_t d = static_cast<std::size_t>(y) * out_w + x;
      const float inv = wsum[s] > 0.0f ? 1.0f / wsum[s] : 0.0f;
      for (int c = 0; c < 3; ++c) {
        out.data[d * 3 + c] = static_cast<std::uint8_t>(
            std::lround(std::clamp(acc[s * 3 + c] * inv, 0.0f, 1.0f) * 255.0f));
      }
    }
  }
  return out;
}

}  // namespace rcdl
