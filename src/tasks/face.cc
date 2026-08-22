#include "rcdl/tasks/face.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"
#include "rcdl/tasks/detection.h"
#include "rcdl/tracks/reid.h"

namespace rcdl {

namespace {

/// Channels each branch of the head carries, per prior. These are what
/// resolveFaceHead() identifies the outputs BY, and what decodeFaces() strides
/// its buffers with, so they are named once and shared.
constexpr int kLocChannels = 4;
constexpr int kConfChannels = 2;
constexpr int kLandmChannels = 10;
constexpr int kNumLandmarks = kLandmChannels / 2;

/// How a [.., N, C] / [.., C, N] tensor is laid out in the flat row-major buffer.
struct AxisOrder {
  std::int64_t n = 0;       ///< number of priors
  bool channels_first = false;
};

/// Reduce a tensor shape to (N, channel-axis position) given the branch's known
/// channel count.
///
/// Leading unit axes carry no information, so they are dropped first — that
/// makes [1,N,C], [N,C] and [1,1,N,C] all behave the same. What is left must be
/// 2-D with exactly one axis equal to `channels`; N comes from the other. The
/// ambiguous case N == channels cannot arise for a real head (N is in the
/// thousands, channels is 2/4/10) but is resolved in favour of channels-last,
/// the layout the export actually uses.
AxisOrder axisOrder(const std::vector<int>& shape, int channels, const char* what,
                    const char* fn) {
  std::vector<std::int64_t> dims;
  for (int d : shape) {
    if (dims.empty() && d == 1) continue;  // drop leading batch / unit axes
    dims.push_back(d);
  }
  auto fail = [&]() {
    std::ostringstream os;
    os << "RCDL " << fn << ": " << what << " tensor shape [";
    for (std::size_t i = 0; i < shape.size(); ++i) os << (i ? "," : "") << shape[i];
    os << "] has no axis of " << channels << " channels";
    throw Error(-1, os.str());
  };
  if (dims.size() != 2 || dims[0] <= 0 || dims[1] <= 0) fail();

  AxisOrder ax;
  if (dims[1] == channels) {
    ax.channels_first = false;  // [N, C]
    ax.n = dims[0];
  } else if (dims[0] == channels) {
    ax.channels_first = true;  // [C, N]
    ax.n = dims[1];
  } else {
    fail();
  }
  return ax;
}

/// Element (prior `i`, channel `c`) of a branch buffer.
inline float at(const float* data, const AxisOrder& ax, int channels, std::int64_t i, int c) {
  const std::int64_t off = ax.channels_first ? (static_cast<std::int64_t>(c) * ax.n + i)
                                             : (i * channels + c);
  return data[off];
}

/// Model input height/width, or (0,0) when input 0 is not a 4-D image tensor.
std::pair<int, int> modelInputHw(const Engine& engine) {
  if (engine.numInputs() < 1) return {0, 0};
  const rknn_tensor_attr& in = engine.inputAttr(0);
  if (in.n_dims != 4) return {0, 0};
  if (in.fmt == RKNN_TENSOR_NHWC) {
    return {static_cast<int>(in.dims[1]), static_cast<int>(in.dims[2])};
  }
  return {static_cast<int>(in.dims[2]), static_cast<int>(in.dims[3])};
}

/// The model's output signature, for error messages that say what was actually
/// seen rather than only what was expected.
std::string describeOutputs(const Engine& engine) {
  std::ostringstream os;
  for (int i = 0; i < engine.numOutputs(); ++i) {
    const rknn_tensor_attr& a = engine.outputAttr(i);
    os << "\n  out[" << i << "] " << a.name << " shape=[";
    for (std::uint32_t d = 0; d < a.n_dims; ++d) os << (d ? "," : "") << a.dims[d];
    os << "]";
  }
  return os.str();
}

[[noreturn]] void failHead(const Engine& engine, const std::string& why) {
  throw Error(-1, "RCDL resolveFaceHead: " + why + ". Model outputs:" + describeOutputs(engine));
}

}  // namespace

// ---------------------------------------------------------------------------
// Prior boxes
// ---------------------------------------------------------------------------

std::vector<PriorBox> generatePriors(const FaceConfig& cfg) {
  RCDL_REQUIRE(cfg.input_w > 0 && cfg.input_h > 0,
               "RCDL generatePriors: model input size must be positive");
  RCDL_REQUIRE(cfg.steps.size() == cfg.min_sizes.size(),
               "RCDL generatePriors: steps and min_sizes must have one entry per scale");
  RCDL_REQUIRE(!cfg.steps.empty(), "RCDL generatePriors: no scales configured");

  const float iw = static_cast<float>(cfg.input_w);
  const float ih = static_cast<float>(cfg.input_h);

  std::vector<PriorBox> priors;
  for (std::size_t k = 0; k < cfg.steps.size(); ++k) {
    const int step = cfg.steps[k];
    RCDL_REQUIRE(step > 0, "RCDL generatePriors: every step must be positive");
    RCDL_REQUIRE(!cfg.min_sizes[k].empty(),
                 "RCDL generatePriors: every scale needs at least one min_size");
    // ceil, not floor: the last (partial) cell still carries priors, and this is
    // where an input size that is not a multiple of the stride gets its extra
    // row/column — dropping it would silently shorten the anchor set.
    const int grid_h = (cfg.input_h + step - 1) / step;
    const int grid_w = (cfg.input_w + step - 1) / step;
    priors.reserve(priors.size() +
                   static_cast<std::size_t>(grid_h) * grid_w * cfg.min_sizes[k].size());

    for (int i = 0; i < grid_h; ++i) {
      for (int j = 0; j < grid_w; ++j) {
        // Cell CENTRE, normalized. The +0.5 puts the prior in the middle of the
        // cell; note it is scaled by `step`, not by the grid size, so a grid
        // rounded up by ceil() keeps its cells aligned to the stride.
        const float cx = (static_cast<float>(j) + 0.5f) * static_cast<float>(step) / iw;
        const float cy = (static_cast<float>(i) + 0.5f) * static_cast<float>(step) / ih;
        for (int min_size : cfg.min_sizes[k]) {
          RCDL_REQUIRE(min_size > 0, "RCDL generatePriors: every min_size must be positive");
          PriorBox p;
          p.cx = cx;
          p.cy = cy;
          p.w = static_cast<float>(min_size) / iw;
          p.h = static_cast<float>(min_size) / ih;
          if (cfg.clip) {
            p.cx = std::min(std::max(p.cx, 0.0f), 1.0f);
            p.cy = std::min(std::max(p.cy, 0.0f), 1.0f);
            p.w = std::min(std::max(p.w, 0.0f), 1.0f);
            p.h = std::min(std::max(p.h, 0.0f), 1.0f);
          }
          priors.push_back(p);
        }
      }
    }
  }
  return priors;
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

std::vector<FaceDetection> decodeFaces(const float* loc, const std::vector<int>& loc_shape,
                                       const float* conf, const std::vector<int>& conf_shape,
                                       const float* landm, const std::vector<int>& landm_shape,
                                       const std::vector<PriorBox>& priors,
                                       const FaceConfig& cfg, const LetterboxInfo& lb) {
  if (loc == nullptr || conf == nullptr || landm == nullptr || priors.empty()) return {};
  RCDL_REQUIRE(cfg.face_class >= 0 && cfg.face_class < kConfChannels,
               "RCDL decodeFaces: face_class must index the 2-channel conf tensor");

  const char* kFn = "decodeFaces";
  const AxisOrder loc_ax = axisOrder(loc_shape, kLocChannels, "loc", kFn);
  const AxisOrder conf_ax = axisOrder(conf_shape, kConfChannels, "conf", kFn);
  const AxisOrder landm_ax = axisOrder(landm_shape, kLandmChannels, "landmark", kFn);

  // The prior set is the authority on N: every read below is indexed by prior,
  // so a tensor holding fewer rows than there are priors would be read past its
  // end. Requiring exact agreement (rather than min()) also catches a
  // mis-configured prior layout that happens to be a prefix of the real one.
  const std::int64_t n = static_cast<std::int64_t>(priors.size());
  if (loc_ax.n != n || conf_ax.n != n || landm_ax.n != n) {
    throw Error(-1, "RCDL decodeFaces: " + std::to_string(priors.size()) +
                        " priors but the loc/conf/landmark tensors hold " +
                        std::to_string(loc_ax.n) + "/" + std::to_string(conf_ax.n) + "/" +
                        std::to_string(landm_ax.n) +
                        " rows — the prior configuration does not match this model");
  }

  const float vc = cfg.var_center;
  const float vs = cfg.var_size;
  const float mw = static_cast<float>(cfg.input_w);
  const float mh = static_cast<float>(cfg.input_h);
  const int bg_class = kConfChannels - 1 - cfg.face_class;

  // NMS runs through the shared routine, so candidates are staged as Detection
  // and their landmarks carried alongside by index.
  std::vector<Detection> boxes;
  std::vector<std::vector<std::pair<float, float>>> marks;

  for (std::int64_t i = 0; i < n; ++i) {
    float score = at(conf, conf_ax, kConfChannels, i, cfg.face_class);
    if (cfg.apply_softmax) {
      // Two-class softmax, shifted by the max for stability. Only needed for an
      // export that left the softmax out of the graph.
      const float other = at(conf, conf_ax, kConfChannels, i, bg_class);
      const float m = std::max(score, other);
      const float e0 = std::exp(score - m);
      const float e1 = std::exp(other - m);
      score = e0 / (e0 + e1);
    }
    if (score < cfg.conf_thresh) continue;

    const PriorBox& p = priors[static_cast<std::size_t>(i)];

    // Centre/size deltas -> a normalized box, then to model-input pixels.
    const float cx = p.cx + at(loc, loc_ax, kLocChannels, i, 0) * vc * p.w;
    const float cy = p.cy + at(loc, loc_ax, kLocChannels, i, 1) * vc * p.h;
    const float w = p.w * std::exp(at(loc, loc_ax, kLocChannels, i, 2) * vs);
    const float h = p.h * std::exp(at(loc, loc_ax, kLocChannels, i, 3) * vs);

    Detection d;
    d.x1 = lb.clampX(lb.invX((cx - w * 0.5f) * mw));
    d.y1 = lb.clampY(lb.invY((cy - h * 0.5f) * mh));
    d.x2 = lb.clampX(lb.invX((cx + w * 0.5f) * mw));
    d.y2 = lb.clampY(lb.invY((cy + h * 0.5f) * mh));
    d.score = score;
    d.class_id = 0;  // single class: nms() is per-class, and there is only one

    std::vector<std::pair<float, float>> pts;
    pts.reserve(kNumLandmarks);
    for (int t = 0; t < kNumLandmarks; ++t) {
      // Same centre encoding as the box centre — var_center on both axes.
      const float lx = p.cx + at(landm, landm_ax, kLandmChannels, i, 2 * t) * vc * p.w;
      const float ly = p.cy + at(landm, landm_ax, kLandmChannels, i, 2 * t + 1) * vc * p.h;
      pts.emplace_back(lb.clampX(lb.invX(lx * mw)), lb.clampY(lb.invY(ly * mh)));
    }

    boxes.push_back(d);
    marks.push_back(std::move(pts));
  }

  const std::vector<int> keep = nms(boxes, cfg.iou_thresh, cfg.max_faces);
  std::vector<FaceDetection> out;
  out.reserve(keep.size());
  for (int idx : keep) {
    const Detection& d = boxes[static_cast<std::size_t>(idx)];
    FaceDetection f;
    f.x1 = d.x1;
    f.y1 = d.y1;
    f.x2 = d.x2;
    f.y2 = d.y2;
    f.score = d.score;
    f.landmarks = std::move(marks[static_cast<std::size_t>(idx)]);
    out.push_back(std::move(f));
  }
  return out;
}

// ---------------------------------------------------------------------------
// Engine binding
// ---------------------------------------------------------------------------

std::string FaceHeadLayout::describe() const {
  std::ostringstream os;
  os << "RetinaFace head: " << num_priors << " priors, input " << input_w << "x" << input_h
     << ", loc=out[" << loc_index << "] conf=out[" << conf_index << "] landmark=out["
     << landm_index << "]";
  return os.str();
}

FaceHeadLayout resolveFaceHead(const Engine& engine) {
  if (engine.numOutputs() < 3) {
    failHead(engine, "a RetinaFace head has 3 outputs, this model has " +
                         std::to_string(engine.numOutputs()));
  }

  FaceHeadLayout layout;
  std::int64_t n_loc = 0, n_conf = 0, n_landm = 0;

  for (int i = 0; i < engine.numOutputs(); ++i) {
    // Leading unit axes carry no information; what remains must be (N, C) or
    // (C, N) for a branch of this head.
    std::vector<int> dims;
    for (int d : engine.outputShape(i)) {
      if (dims.empty() && d == 1) continue;
      dims.push_back(d);
    }
    if (dims.size() != 2) continue;

    // Branches are identified by channel count alone: 4 / 2 / 10 are distinct
    // and fixed for this head, so an export that reorders its outputs still
    // resolves. A shape that matches none is simply skipped — a model with an
    // extra auxiliary output stays usable.
    struct Candidate {
      int channels;
      int* index;
      std::int64_t* count;
      const char* what;
    };
    const Candidate cands[] = {{kLocChannels, &layout.loc_index, &n_loc, "loc"},
                               {kConfChannels, &layout.conf_index, &n_conf, "conf"},
                               {kLandmChannels, &layout.landm_index, &n_landm, "landmark"}};
    for (const Candidate& c : cands) {
      if (dims[0] != c.channels && dims[1] != c.channels) continue;
      if (*c.index >= 0) {
        failHead(engine, std::string("two outputs look like the ") + c.what +
                             " branch (" + std::to_string(c.channels) + " channels): out[" +
                             std::to_string(*c.index) + "] and out[" + std::to_string(i) + "]");
      }
      *c.index = i;
      *c.count = dims[1] == c.channels ? dims[0] : dims[1];
      break;
    }
  }

  if (layout.loc_index < 0 || layout.conf_index < 0 || layout.landm_index < 0) {
    failHead(engine,
             "could not find all three branches — expected 2-D outputs with 4 (box), 2 (score) "
             "and 10 (landmark) channels");
  }
  if (n_loc != n_conf || n_loc != n_landm) {
    failHead(engine, "the three branches disagree on the anchor count: loc " +
                         std::to_string(n_loc) + ", conf " + std::to_string(n_conf) +
                         ", landmark " + std::to_string(n_landm));
  }
  layout.num_priors = static_cast<int>(n_loc);

  const std::pair<int, int> hw = modelInputHw(engine);
  if (hw.first <= 0 || hw.second <= 0) {
    failHead(engine, "input 0 is not a 4-D image tensor, cannot size the prior grid");
  }
  layout.input_h = hw.first;
  layout.input_w = hw.second;
  return layout;
}

FaceDetector::FaceDetector(Engine& engine, FaceConfig cfg)
    : engine_(engine), cfg_(std::move(cfg)), layout_(resolveFaceHead(engine)) {
  // The canvas comes from the model, not from the caller: the prior grid is
  // derived from it, and a config that disagrees with the loaded model would
  // generate priors for a different network.
  cfg_.input_w = layout_.input_w;
  cfg_.input_h = layout_.input_h;
  priors_ = generatePriors(cfg_);

  if (static_cast<int>(priors_.size()) != layout_.num_priors) {
    // Free, decisive check: the anchor count is a fingerprint of the prior
    // configuration. If it does not match, the steps / min_sizes in cfg are not
    // the ones this model was exported with, and every decoded box would be
    // wrong in a way nothing downstream could detect.
    std::ostringstream os;
    os << "RCDL FaceDetector: prior configuration produces " << priors_.size()
       << " priors but the model declares " << layout_.num_priors << " at " << cfg_.input_w
       << "x" << cfg_.input_h << " (steps";
    for (int s : cfg_.steps) os << " " << s;
    os << ", min_sizes";
    for (const std::vector<int>& ms : cfg_.min_sizes) {
      os << " [";
      for (std::size_t i = 0; i < ms.size(); ++i) os << (i ? "," : "") << ms[i];
      os << "]";
    }
    os << ") — check FaceConfig against the model's export settings";
    throw Error(-1, os.str());
  }
}

std::vector<FaceDetection> FaceDetector::postprocess(const LetterboxInfo& lb) const {
  // Scratch must outlive the pointers handed to the decoder: outputAsFloat is
  // zero-copy for packed f32 (nothing lands in scratch) and dequant-into-scratch
  // for the int8 / fp16 branches this head actually emits.
  std::vector<float> loc_buf, conf_buf, landm_buf;
  std::vector<int> loc_shape, conf_shape, landm_shape;

  const float* loc = outputAsFloat(engine_, layout_.loc_index, loc_buf, loc_shape);
  const float* conf = outputAsFloat(engine_, layout_.conf_index, conf_buf, conf_shape);
  const float* landm = outputAsFloat(engine_, layout_.landm_index, landm_buf, landm_shape);

  return decodeFaces(loc, loc_shape, conf, conf_shape, landm, landm_shape, priors_, cfg_, lb);
}

// ===========================================================================
// Face alignment — the 5-point similarity transform
// ===========================================================================

namespace {

// insightface's arcface_dst, the template every ArcFace-family model in common
// use was trained against, for a 112x112 crop: left eye, right eye, nose, left
// mouth corner, right mouth corner.
constexpr float kArcFaceDst112[10] = {
    38.2946f, 51.6963f,   // left eye
    73.5318f, 51.5014f,   // right eye
    56.0252f, 71.7366f,   // nose
    41.5493f, 92.3655f,   // left mouth corner
    70.7299f, 92.2041f,   // right mouth corner
};

}  // namespace

void arcFaceTemplate(float out[10], int out_w, int out_h) {
  RCDL_REQUIRE(out != nullptr, "arcFaceTemplate: null output");
  RCDL_REQUIRE(out_w > 0 && out_h > 0, "arcFaceTemplate: output size must be positive");
  const float sx = static_cast<float>(out_w) / 112.0f;
  const float sy = static_cast<float>(out_h) / 112.0f;
  for (int i = 0; i < 5; ++i) {
    out[2 * i] = kArcFaceDst112[2 * i] * sx;
    out[2 * i + 1] = kArcFaceDst112[2 * i + 1] * sy;
  }
}

void similarityTransform(const float src[10], const float dst[10], float m[6]) {
  RCDL_REQUIRE(src != nullptr && dst != nullptr && m != nullptr,
               "similarityTransform: null argument");
  double sx = 0, sy = 0, dx = 0, dy = 0;
  for (int i = 0; i < 5; ++i) {
    sx += src[2 * i];
    sy += src[2 * i + 1];
    dx += dst[2 * i];
    dy += dst[2 * i + 1];
  }
  sx /= 5.0;
  sy /= 5.0;
  dx /= 5.0;
  dy /= 5.0;

  // One complex multiply IS a 2-D similarity, so the least-squares fit is a
  // single quotient: c = Σ w_i·conj(z_i) / Σ|z_i|², with z the centred source
  // and w the centred target. No SVD, and — the reason to prefer this form —
  // no reflection is representable, so an alignment can never come back
  // mirrored the way an unguarded Procrustes solution can.
  double num_r = 0, num_i = 0, den = 0;
  for (int i = 0; i < 5; ++i) {
    const double zx = src[2 * i] - sx;
    const double zy = src[2 * i + 1] - sy;
    const double wx = dst[2 * i] - dx;
    const double wy = dst[2 * i + 1] - dy;
    num_r += wx * zx + wy * zy;   // Re(w · conj(z))
    num_i += wy * zx - wx * zy;   // Im(w · conj(z))
    den += zx * zx + zy * zy;
  }
  // All five points coincident: there is no scale or rotation to recover, so
  // fall back to the translation that maps one centroid onto the other.
  double a = 1.0, b = 0.0;
  if (den > 1e-12) {
    a = num_r / den;
    b = num_i / den;
  }
  m[0] = static_cast<float>(a);
  m[1] = static_cast<float>(-b);
  m[2] = static_cast<float>(dx - (a * sx - b * sy));
  m[3] = static_cast<float>(b);
  m[4] = static_cast<float>(a);
  m[5] = static_cast<float>(dy - (b * sx + a * sy));
}

void faceAlignTransform(const float landmarks[10], int out_w, int out_h, float m[6]) {
  float tpl[10];
  arcFaceTemplate(tpl, out_w, out_h);
  similarityTransform(landmarks, tpl, m);
}

void faceLandmarkArray(const FaceDetection& face, float out[10]) {
  RCDL_REQUIRE(out != nullptr, "faceLandmarkArray: null output");
  RCDL_REQUIRE(face.landmarks.size() == 5,
               "faceLandmarkArray: alignment needs exactly five landmarks");
  for (int i = 0; i < 5; ++i) {
    out[2 * i] = face.landmarks[static_cast<std::size_t>(i)].first;
    out[2 * i + 1] = face.landmarks[static_cast<std::size_t>(i)].second;
  }
}

// ===========================================================================
// FaceRecognizer
// ===========================================================================

namespace {

/// Bilinear sample of one channel, with `pad` outside the frame. Same sampler
/// the whole-body crop uses: a face at the edge of the frame is normal, and the
/// template reaches past it, so the outside has to be a defined value rather
/// than a clamped column of skin smeared along the border.
float sampleFaceChannel(const std::uint8_t* base, int w, int h, std::size_t stride, int bpp,
                        float x, float y, int c, std::uint8_t pad) {
  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const float fx = x - static_cast<float>(x0);
  const float fy = y - static_cast<float>(y0);
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
      acc += wgt * static_cast<float>(
                       inside ? base[static_cast<std::size_t>(sy) * stride +
                                     static_cast<std::size_t>(sx) * bpp + c]
                              : pad);
    }
  }
  return acc;
}

/// Invert a 2x3 similarity/affine. The forward matrix maps SOURCE -> template;
/// resampling walks the template and needs the other direction.
bool invertAffine(const float m[6], float inv[6]) {
  const float det = m[0] * m[4] - m[1] * m[3];
  if (std::abs(det) < 1e-12f) return false;
  const float id = 1.0f / det;
  inv[0] = m[4] * id;
  inv[1] = -m[1] * id;
  inv[3] = -m[3] * id;
  inv[4] = m[0] * id;
  inv[2] = -(inv[0] * m[2] + inv[1] * m[5]);
  inv[5] = -(inv[3] * m[2] + inv[4] * m[5]);
  return true;
}

}  // namespace

FaceRecognizer::FaceRecognizer(Engine& engine, FaceRecogConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {
  RCDL_REQUIRE(engine_.numInputs() == 1,
               "FaceRecognizer: an identity model takes exactly one input");
  RCDL_REQUIRE(out_idx_ >= 0 && out_idx_ < engine_.numOutputs(),
               "FaceRecognizer: output index out of range");

  const rknn_tensor_attr& in = engine_.inputAttr(0);
  RCDL_REQUIRE(in.n_dims == 4, "FaceRecognizer: input 0 is not a 4-D image tensor");
  if (in.fmt == RKNN_TENSOR_NHWC) {
    in_h_ = static_cast<int>(in.dims[1]);
    in_w_ = static_cast<int>(in.dims[2]);
  } else {
    in_h_ = static_cast<int>(in.dims[2]);
    in_w_ = static_cast<int>(in.dims[3]);
  }
  RCDL_REQUIRE(in_w_ > 0 && in_h_ > 0, "FaceRecognizer: could not read the input size");

  const rknn_tensor_type it = engine_.inputType(0);
  RCDL_REQUIRE(it == RKNN_TENSOR_UINT8 || it == RKNN_TENSOR_FLOAT32,
               "FaceRecognizer: the model takes neither u8 image bytes nor float32");
  float_input_ = it == RKNN_TENSOR_FLOAT32;

  const std::vector<int> os = engine_.outputShape(out_idx_);
  dim_ = os.empty() ? 0 : os.back();
  if (dim_ <= 1) {
    throw Error(-1, "FaceRecognizer: output " + std::to_string(out_idx_) +
                        " is not an embedding vector");
  }
}

void FaceRecognizer::feed(const ImageView& src, const float m[6]) {
  RCDL_REQUIRE(src.data != nullptr && src.width > 0 && src.height > 0,
               "FaceRecognizer: the source needs a CPU mapping");
  const int bpp = bytesPerPixel(src.format);
  RCDL_REQUIRE(bpp == 3 || bpp == 4, "FaceRecognizer: expected a packed RGB/BGR source");
  const bool src_bgr =
      src.format == PixelFormat::BGR888 || src.format == PixelFormat::BGRA8888;
  const bool want_bgr =
      cfg_.model_input == PixelFormat::BGR888 || cfg_.model_input == PixelFormat::BGRA8888;
  const bool swap = src_bgr != want_bgr;

  float inv[6];
  RCDL_REQUIRE(invertAffine(m, inv),
               "FaceRecognizer: the alignment transform is degenerate (are the five landmarks "
               "distinct?)");

  const std::uint8_t* base = static_cast<const std::uint8_t*>(src.data);
  const std::size_t stride = src.rowBytes();

  // The transform is a rotation + uniform scale about an arbitrary centre, which
  // the hardware letterbox cannot express (it centres a whole image), so the
  // resample runs here. 112x112 samples is nothing next to a 50-layer network.
  if (float_input_) {
    scratch_.resize(static_cast<std::size_t>(in_w_) * in_h_ * 3);
    for (int y = 0; y < in_h_; ++y) {
      float* row = scratch_.data() + static_cast<std::size_t>(y) * in_w_ * 3;
      for (int x = 0; x < in_w_; ++x) {
        const float fx = static_cast<float>(x) + 0.5f;
        const float fy = static_cast<float>(y) + 0.5f;
        const float sx = inv[0] * fx + inv[1] * fy + inv[2] - 0.5f;
        const float sy = inv[3] * fx + inv[4] * fy + inv[5] - 0.5f;
        for (int c = 0; c < 3; ++c) {
          const int chan = swap ? 2 - c : c;
          // 0..255, NOT 0..1: the model's own (x-127.5)/127.5 is folded into
          // the .rknn, so scaling here would darken the face by 255x and still
          // return a well-formed unit vector.
          row[x * 3 + c] =
              sampleFaceChannel(base, src.width, src.height, stride, bpp, sx, sy, chan, cfg_.pad);
        }
      }
    }
    engine_.setInput(0, scratch_.data(), scratch_.size() * sizeof(float));
    return;
  }

  // Quantized build: the destination IS the NPU's input tensor.
  const ImageView dst = engineInputView(engine_, 0, cfg_.model_input);
  std::uint8_t* out = static_cast<std::uint8_t*>(dst.data);
  const std::size_t dstride = dst.rowBytes();
  const int dbpp = bytesPerPixel(dst.format);
  for (int y = 0; y < in_h_; ++y) {
    std::uint8_t* row = out + static_cast<std::size_t>(y) * dstride;
    for (int x = 0; x < in_w_; ++x) {
      const float fx = static_cast<float>(x) + 0.5f;
      const float fy = static_cast<float>(y) + 0.5f;
      const float sx = inv[0] * fx + inv[1] * fy + inv[2] - 0.5f;
      const float sy = inv[3] * fx + inv[4] * fy + inv[5] - 0.5f;
      for (int c = 0; c < 3; ++c) {
        const int chan = swap ? 2 - c : c;
        const float v =
            sampleFaceChannel(base, src.width, src.height, stride, bpp, sx, sy, chan, cfg_.pad);
        row[static_cast<std::size_t>(x) * dbpp + c] =
            static_cast<std::uint8_t>(std::lround(std::min(255.0f, std::max(0.0f, v))));
      }
    }
  }
}

std::vector<float> FaceRecognizer::embed(const ImageView& src, const float landmarks[10]) {
  RCDL_REQUIRE(landmarks != nullptr, "FaceRecognizer: null landmarks");
  float m[6];
  faceAlignTransform(landmarks, in_w_, in_h_, m);
  std::copy(m, m + 6, last_m_.begin());
  feed(src, m);
  engine_.infer();
  return postprocess();
}

std::vector<float> FaceRecognizer::embedAligned(const ImageView& aligned) {
  if (aligned.width != in_w_ || aligned.height != in_h_) {
    throw Error(-1, "FaceRecognizer: an already-aligned crop must be " + std::to_string(in_w_) +
                        "x" + std::to_string(in_h_) + ", got " + std::to_string(aligned.width) +
                        "x" + std::to_string(aligned.height));
  }
  const float identity[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  std::copy(identity, identity + 6, last_m_.begin());
  feed(aligned, identity);
  engine_.infer();
  return postprocess();
}

std::vector<float> FaceRecognizer::postprocess() const {
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* v = outputAsFloat(engine_, out_idx_, scratch, shape);
  if (!cfg_.normalize) return std::vector<float>(v, v + dim_);
  return normalizeEmbedding(v, dim_);
}

}  // namespace rcdl
