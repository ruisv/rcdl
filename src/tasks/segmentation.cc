#include "rcdl/tasks/segmentation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

namespace {

/// Resolve (C,H,W) of a logit volume from its logical shape. Tolerates a missing
/// batch dim ({C,H,W} / {H,W,C}) as well as the usual {1,...} forms, because a
/// decoder should not care whether the exporter kept the leading 1.
void resolveCHW(const std::vector<int>& shape, bool channels_first, int& C, int& H, int& W) {
  C = H = W = 0;
  const std::size_t n = shape.size();
  if (n >= 4) {
    if (channels_first) {
      C = shape[n - 3];
      H = shape[n - 2];
      W = shape[n - 1];
    } else {
      H = shape[n - 3];
      W = shape[n - 2];
      C = shape[n - 1];
    }
  } else if (n == 3) {
    if (channels_first) {
      C = shape[0];
      H = shape[1];
      W = shape[2];
    } else {
      H = shape[0];
      W = shape[1];
      C = shape[2];
    }
  }
}

/// Resolve (H,W) of an already-argmaxed id map by dropping unit dims.
void resolveHW(const std::vector<int>& shape, int& H, int& W) {
  std::vector<int> dims;
  for (int d : shape) {
    if (d > 1) dims.push_back(d);
  }
  if (dims.size() >= 2) {
    H = dims[dims.size() - 2];
    W = dims[dims.size() - 1];
    return;
  }
  const int n = static_cast<int>(shape.size());
  H = n >= 2 ? shape[n - 2] : (n >= 1 ? shape[n - 1] : 0);
  W = n >= 1 ? shape[n - 1] : 0;
}

/// HSV (all components in [0,1]) -> RGB bytes.
void hsv2rgb(float h, float s, float v, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b) {
  const float hh = h * 6.0f;
  const int i = static_cast<int>(std::floor(hh)) % 6;
  const float f = hh - std::floor(hh);
  const float p = v * (1.0f - s);
  const float q = v * (1.0f - s * f);
  const float t = v * (1.0f - s * (1.0f - f));
  float fr, fg, fb;
  switch (i) {
    case 0: fr = v; fg = t; fb = p; break;
    case 1: fr = q; fg = v; fb = p; break;
    case 2: fr = p; fg = v; fb = t; break;
    case 3: fr = p; fg = q; fb = v; break;
    case 4: fr = t; fg = p; fb = v; break;
    default: fr = v; fg = p; fb = q; break;
  }
  r = static_cast<std::uint8_t>(std::lround(fr * 255.0f));
  g = static_cast<std::uint8_t>(std::lround(fg * 255.0f));
  b = static_cast<std::uint8_t>(std::lround(fb * 255.0f));
}

/// PASCAL VOC palette for ids 0..20 (the bit-interleaving generator every VOC
/// toolkit uses, so RCDL's overlays match published reference images), then a
/// golden-ratio hue walk for anything above it.
void paletteColor(std::int32_t id, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b) {
  if (id <= 0) {  // id 0 is "background" in VOC and comes out black anyway
    r = g = b = 0;
    return;
  }
  if (id < 21) {
    int c = id;
    int rr = 0, gg = 0, bb = 0;
    for (int i = 0; i < 8; ++i) {
      rr |= ((c >> 0) & 1) << (7 - i);
      gg |= ((c >> 1) & 1) << (7 - i);
      bb |= ((c >> 2) & 1) << (7 - i);
      c >>= 3;
    }
    r = static_cast<std::uint8_t>(rr);
    g = static_cast<std::uint8_t>(gg);
    b = static_cast<std::uint8_t>(bb);
    return;
  }
  // Successive ids land ~137.5 degrees apart on the hue circle, which is about
  // as far from their neighbours as a 1-D generator can put them.
  const float golden = 0.6180339887498949f;
  const float h = std::fmod(static_cast<float>(id) * golden, 1.0f);
  hsv2rgb(h, 0.7f, 0.95f, r, g, b);
}

/// Nearest source index for destination index `d` under the pixel-centre map.
int nearestIndex(int d, int dst_n, int src_n) noexcept {
  const float f = (static_cast<float>(d) + 0.5f) * static_cast<float>(src_n) /
                  static_cast<float>(dst_n);
  int i = static_cast<int>(f);  // f >= 0, so the truncation is a floor
  if (i < 0) i = 0;
  if (i > src_n - 1) i = src_n - 1;
  return i;
}

/// Width (innermost spatial dim) of a tensor in its own layout — the axis
/// w_stride pads. Mirrors the reader's packed-rows test so the raw fast path
/// below only fires when the buffer really is packed row-major.
int widthOf(const rknn_tensor_attr& a) noexcept {
  if (a.n_dims == 4) {
    if (a.fmt == RKNN_TENSOR_NHWC) return static_cast<int>(a.dims[2]);
    return static_cast<int>(a.dims[3]);
  }
  return a.n_dims > 0 ? static_cast<int>(a.dims[a.n_dims - 1]) : 1;
}

/// Per-pixel argmax straight over affine-quantized CODES.
///
/// WHY THIS IS EXACT, not an approximation: an RKNN affine-asymmetric output
/// carries ONE (scale, zero_point) for the whole tensor, so dequantization is
/// the single map q -> (q - zp) * scale applied identically to every channel.
/// With scale > 0 that map is strictly increasing, hence order-preserving AND
/// tie-preserving; argmax over q therefore selects the same channel as argmax
/// over the dequantized floats, tie-break included. The caller rejects
/// scale <= 0 (which would reverse the order) and any non-affine encoding, and
/// the float path in decodeSeg() stays the reference.
///
/// The point is the volume: a 19-class 1024x2048 head is 40M values, so
/// dequantizing it to float costs an allocation plus 160 MB of traffic before
/// the argmax even starts.
template <typename Q>
void argmaxQuantized(const Q* src, int C, int H, int W, bool channels_first,
                     std::int32_t* out) {
  const std::size_t npix = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);
  const std::size_t step = channels_first ? npix : 1;
  for (std::size_t p = 0; p < npix; ++p) {
    const Q* v = channels_first ? (src + p) : (src + p * static_cast<std::size_t>(C));
    Q best = v[0];
    int best_c = 0;
    for (int c = 1; c < C; ++c) {
      const Q q = v[static_cast<std::size_t>(c) * step];
      if (q > best) {  // strict: ties keep the lowest channel, as in decodeSeg
        best = q;
        best_c = c;
      }
    }
    out[p] = best_c;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

SegMask decodeSeg(const float* data, const std::vector<int>& shape, const SegConfig& cfg) {
  SegMask m;
  std::int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? static_cast<std::int64_t>(d) : 0);
  if (data == nullptr) total = 0;

  if (cfg.argmaxed) {
    // Pass-through id map: the graph already argmaxed, we only have to round the
    // float codes back to integers.
    int H = 0, W = 0;
    resolveHW(shape, H, W);
    m.width = W;
    m.height = H;
    m.num_classes = cfg.num_classes;
    const std::int64_t n = std::max<std::int64_t>(0, static_cast<std::int64_t>(H) * W);
    m.labels.assign(static_cast<std::size_t>(n), 0);
    const std::int64_t readable = std::min(n, total);
    std::int32_t max_label = -1;
    for (std::int64_t i = 0; i < readable; ++i) {
      const auto lbl = static_cast<std::int32_t>(std::lround(data[i]));
      m.labels[static_cast<std::size_t>(i)] = lbl;
      if (lbl > max_label) max_label = lbl;
    }
    if (m.num_classes <= 0) m.num_classes = max_label + 1;
    return m;
  }

  int C = 0, H = 0, W = 0;
  resolveCHW(shape, cfg.channels_first, C, H, W);
  if (cfg.num_classes > 0) C = std::min(C, cfg.num_classes);  // narrow only

  const std::int64_t npix = std::max<std::int64_t>(0, static_cast<std::int64_t>(H) * W);
  // Memory-safety bound: the scan reads C channels per pixel out of a buffer
  // holding product(shape) elements. Clamp C so a shape/channel mismatch can
  // never walk off the end.
  if (npix > 0 && static_cast<std::int64_t>(C) * npix > total) {
    C = static_cast<int>(total / npix);
  }
  m.width = W;
  m.height = H;
  m.num_classes = C;
  m.labels.assign(static_cast<std::size_t>(npix), 0);
  if (cfg.score != SegScore::kNone) m.confidence.assign(static_cast<std::size_t>(npix), 0.0f);
  if (npix <= 0 || C <= 0) return m;

  // Logical row-major addressing:
  //   channels_first ([C,H,W]) : value(c,p) = data[c*npix + p]
  //   channels_last  ([H,W,C]) : value(c,p) = data[p*C + c]
  const std::int64_t step = cfg.channels_first ? npix : 1;
  for (std::int64_t p = 0; p < npix; ++p) {
    const float* v = data + (cfg.channels_first ? p : p * C);
    float best = v[0];
    int best_c = 0;
    for (int c = 1; c < C; ++c) {
      const float x = v[static_cast<std::int64_t>(c) * step];
      if (x > best) {  // strict: ties keep the lowest channel index
        best = x;
        best_c = c;
      }
    }
    m.labels[static_cast<std::size_t>(p)] = best_c;

    if (cfg.score == SegScore::kMax) {
      m.confidence[static_cast<std::size_t>(p)] = best;
    } else if (cfg.score == SegScore::kSoftmax) {
      // softmax[winner] = 1 / Σ_c exp(v_c - v_winner). Shifting by the max (the
      // winner, by construction) keeps every exponent <= 0, so this cannot
      // overflow and the denominator is >= 1.
      float sum = 0.0f;
      for (int c = 0; c < C; ++c) sum += std::exp(v[static_cast<std::int64_t>(c) * step] - best);
      m.confidence[static_cast<std::size_t>(p)] = (sum > 0.0f) ? 1.0f / sum : 0.0f;
    }
  }
  return m;
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

SegMask segResize(const SegMask& m, int dst_w, int dst_h) {
  SegMask o;
  o.num_classes = m.num_classes;
  o.width = std::max(0, dst_w);
  o.height = std::max(0, dst_h);
  const std::size_t n = static_cast<std::size_t>(o.width) * static_cast<std::size_t>(o.height);
  if (n == 0 || m.width <= 0 || m.height <= 0) return o;

  const bool with_conf = m.confidence.size() == static_cast<std::size_t>(m.width) *
                                                    static_cast<std::size_t>(m.height);
  o.labels.assign(n, 0);
  if (with_conf) o.confidence.assign(n, 0.0f);

  // The x map does not depend on y, so pay for it once.
  std::vector<int> sx(static_cast<std::size_t>(o.width));
  for (int x = 0; x < o.width; ++x) {
    sx[static_cast<std::size_t>(x)] = nearestIndex(x, o.width, m.width);
  }
  for (int y = 0; y < o.height; ++y) {
    const std::size_t srow =
        static_cast<std::size_t>(nearestIndex(y, o.height, m.height)) *
        static_cast<std::size_t>(m.width);
    const std::size_t drow = static_cast<std::size_t>(y) * static_cast<std::size_t>(o.width);
    for (int x = 0; x < o.width; ++x) {
      const std::size_t s = srow + static_cast<std::size_t>(sx[static_cast<std::size_t>(x)]);
      o.labels[drow + static_cast<std::size_t>(x)] = m.labels[s];
      if (with_conf) o.confidence[drow + static_cast<std::size_t>(x)] = m.confidence[s];
    }
  }
  return o;
}

SegMask segToSource(const SegMask& m, const LetterboxInfo& lb) {
  RCDL_REQUIRE(lb.srcW > 0 && lb.srcH > 0 && lb.dstW > 0 && lb.dstH > 0,
               "RCDL segToSource: letterbox has a non-positive extent");
  RCDL_REQUIRE(lb.scale > 0.0f, "RCDL segToSource: letterbox scale must be > 0");

  SegMask o;
  o.num_classes = m.num_classes;
  o.width = lb.srcW;
  o.height = lb.srcH;
  const std::size_t n = static_cast<std::size_t>(o.width) * static_cast<std::size_t>(o.height);
  if (n == 0 || m.width <= 0 || m.height <= 0) return o;

  const bool with_conf = m.confidence.size() == static_cast<std::size_t>(m.width) *
                                                    static_cast<std::size_t>(m.height);
  o.labels.assign(n, 0);
  if (with_conf) o.confidence.assign(n, 0.0f);

  // Source pixel centre -> canvas pixel -> mask pixel. Going forward like this
  // (rather than cropping the padding and then resizing) needs no intermediate
  // image and stays correct when the head's output grid is not a divisor of the
  // canvas, e.g. a 65x65 logit map for a 513x513 input.
  const float mx_ratio = static_cast<float>(m.width) / static_cast<float>(lb.dstW);
  const float my_ratio = static_cast<float>(m.height) / static_cast<float>(lb.dstH);
  const auto pick = [](float f, int hi) {
    int i = static_cast<int>(std::lround(f));
    if (i < 0) i = 0;
    if (i > hi) i = hi;
    return i;
  };

  std::vector<int> sx(static_cast<std::size_t>(o.width));
  for (int x = 0; x < o.width; ++x) {
    sx[static_cast<std::size_t>(x)] =
        pick(lb.fwdX(static_cast<float>(x) + 0.5f) * mx_ratio - 0.5f, m.width - 1);
  }
  for (int y = 0; y < o.height; ++y) {
    const int my = pick(lb.fwdY(static_cast<float>(y) + 0.5f) * my_ratio - 0.5f, m.height - 1);
    const std::size_t srow = static_cast<std::size_t>(my) * static_cast<std::size_t>(m.width);
    const std::size_t drow = static_cast<std::size_t>(y) * static_cast<std::size_t>(o.width);
    for (int x = 0; x < o.width; ++x) {
      const std::size_t s = srow + static_cast<std::size_t>(sx[static_cast<std::size_t>(x)]);
      o.labels[drow + static_cast<std::size_t>(x)] = m.labels[s];
      if (with_conf) o.confidence[drow + static_cast<std::size_t>(x)] = m.confidence[s];
    }
  }
  return o;
}

// ---------------------------------------------------------------------------
// Visualisation
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> segColorize(const SegMask& m) {
  const std::size_t n = std::min(m.labels.size(),
                                 static_cast<std::size_t>(std::max(0, m.width)) *
                                     static_cast<std::size_t>(std::max(0, m.height)));
  std::vector<std::uint8_t> out(n * 3);

  // Ids repeat heavily across a frame and the palette is deterministic, so build
  // the low ids once into a lookup table; the (rare) higher ids are computed on
  // the spot rather than growing an unbounded cache from an untrusted label.
  static const std::vector<std::uint8_t> kLut = [] {
    std::vector<std::uint8_t> lut(256 * 3);
    for (int id = 0; id < 256; ++id) {
      std::uint8_t r, g, b;
      paletteColor(id, r, g, b);
      lut[static_cast<std::size_t>(id) * 3 + 0] = b;  // BGR, OpenCV order
      lut[static_cast<std::size_t>(id) * 3 + 1] = g;
      lut[static_cast<std::size_t>(id) * 3 + 2] = r;
    }
    return lut;
  }();

  for (std::size_t i = 0; i < n; ++i) {
    const std::int32_t id = m.labels[i];
    if (id >= 0 && id < 256) {
      const std::size_t k = static_cast<std::size_t>(id) * 3;
      out[i * 3 + 0] = kLut[k + 0];
      out[i * 3 + 1] = kLut[k + 1];
      out[i * 3 + 2] = kLut[k + 2];
    } else {
      std::uint8_t r, g, b;
      paletteColor(id, r, g, b);
      out[i * 3 + 0] = b;
      out[i * 3 + 1] = g;
      out[i * 3 + 2] = r;
    }
  }
  return out;
}

const std::vector<std::string>& vocClassNames() {
  // Function-local static: built once, never copied.
  static const std::vector<std::string> kNames = {
      "background", "aeroplane",   "bicycle", "bird",  "boat",      "bottle", "bus",
      "car",        "cat",         "chair",   "cow",   "diningtable", "dog",  "horse",
      "motorbike",  "person",      "pottedplant", "sheep", "sofa",  "train",  "tvmonitor"};
  return kNames;
}

const char* vocClassName(int class_id) {
  const std::vector<std::string>& names = vocClassNames();
  if (class_id >= 0 && class_id < static_cast<int>(names.size())) {
    return names[static_cast<std::size_t>(class_id)].c_str();
  }
  // Out-of-range ids come from non-VOC models, so name them rather than throw.
  // A small rotating cache keeps the returned pointer valid for the caller.
  static thread_local std::string fallback[4];
  static thread_local int next = 0;
  std::string& slot = fallback[next];
  next = (next + 1) % 4;
  slot = "class " + std::to_string(class_id);
  return slot.c_str();
}

// ---------------------------------------------------------------------------
// Engine-bound segmenter
// ---------------------------------------------------------------------------

Segmenter::Segmenter(Engine& engine, SegConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "RCDL Segmenter: output index out of range");
  }
  // Channel order is a property of the TENSOR, never of the caller's guess: an
  // NHWC logit volume and an NCHW one are the same bytes in a different order,
  // and only fmt tells them apart. UNDEFINED is what the runtime reports for the
  // plain NCHW logical layout, so anything that is not explicitly NHWC is NCHW.
  const rknn_tensor_attr& attr = engine_.outputAttr(out_idx_);
  cfg_.channels_first = attr.fmt != RKNN_TENSOR_NHWC;
}

SegMask Segmenter::postprocess() const {
  const rknn_tensor_attr& attr = engine_.outputAttr(out_idx_);

  // Quantized fast path: argmax the int8/uint8 codes in place. See
  // argmaxQuantized() for why this gives bit-identical labels to the float path.
  // It only applies to the plain label map — a confidence value has to be a real
  // number, so kSoftmax / kMax fall through to the dequantizing path below.
  const QuantParams q = quantParams(attr);
  const bool packed_rows =
      attr.w_stride == 0 || static_cast<int>(attr.w_stride) == widthOf(attr);
  if (!cfg_.argmaxed && cfg_.score == SegScore::kNone && q.is_affine && q.scale > 0.0f &&
      packed_rows && attr.n_dims == 4 && attr.dims[0] == 1) {
    const std::vector<int> shape = engine_.outputShape(out_idx_);
    int C = 0, H = 0, W = 0;
    resolveCHW(shape, cfg_.channels_first, C, H, W);
    if (cfg_.num_classes > 0) C = std::min(C, cfg_.num_classes);
    const std::int64_t npix = static_cast<std::int64_t>(H) * W;
    if (C > 0 && npix > 0 &&
        static_cast<std::int64_t>(C) * npix <= static_cast<std::int64_t>(attr.n_elems)) {
      SegMask m;
      m.width = W;
      m.height = H;
      m.num_classes = C;
      m.labels.assign(static_cast<std::size_t>(npix), 0);
      const void* base = engine_.outputData(out_idx_);
      if (attr.type == RKNN_TENSOR_INT8) {
        argmaxQuantized(static_cast<const std::int8_t*>(base), C, H, W, cfg_.channels_first,
                        m.labels.data());
        return m;
      }
      if (attr.type == RKNN_TENSOR_UINT8) {
        argmaxQuantized(static_cast<const std::uint8_t*>(base), C, H, W, cfg_.channels_first,
                        m.labels.data());
        return m;
      }
    }
  }

  // Reference path: zero-copy for packed F32, dequant-into-scratch otherwise.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);
  return decodeSeg(data, shape, cfg_);
}

SegMask Segmenter::postprocess(const LetterboxInfo& lb) const {
  return segToSource(postprocess(), lb);
}

}  // namespace rcdl
