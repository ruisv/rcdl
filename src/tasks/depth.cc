#include "rcdl/tasks/depth.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

namespace {

/// Resolve (H,W) of a single-channel dense map by dropping the unit dims. This
/// is why depth needs no channel-order flag: {1,1,H,W} and {1,H,W,1} both reduce
/// to the same H*W plane, and with C == 1 they are also the same bytes.
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
  // Degenerate maps ({1,1}, {H,1}, ...): fall back to the raw trailing dims.
  const int n = static_cast<int>(shape.size());
  H = n >= 2 ? shape[n - 2] : (n >= 1 ? shape[n - 1] : 0);
  W = n >= 1 ? shape[n - 1] : 0;
}

/// One axis of a bilinear resample: the two source indices and the blend weight.
struct Lerp {
  int i0 = 0;
  int i1 = 0;
  float w = 0.0f;
};

/// Split a (possibly fractional) source coordinate into a Lerp, clamped to
/// [0, src_n-1] so the edges replicate instead of reading out of bounds.
Lerp lerpAt(float f, int src_n) noexcept {
  const float hi = static_cast<float>(src_n - 1);
  if (!(f > 0.0f)) f = 0.0f;  // false for NaN too
  if (f > hi) f = hi;
  Lerp l;
  l.i0 = static_cast<int>(f);
  l.i1 = std::min(l.i0 + 1, src_n - 1);
  l.w = f - static_cast<float>(l.i0);
  return l;
}

/// Pixel-centre map, destination index -> source coordinate.
float centreMap(int d, int dst_n, int src_n) noexcept {
  return (static_cast<float>(d) + 0.5f) * static_cast<float>(src_n) / static_cast<float>(dst_n) -
         0.5f;
}

/// Shared body of depthResize / depthToSource: `sx`/`sy` give, per destination
/// pixel, where to sample the source map.
DepthMap resampleBilinear(const DepthMap& m, int dst_w, int dst_h, const std::vector<Lerp>& sx,
                          const std::vector<Lerp>& sy) {
  DepthMap o;
  o.width = dst_w;
  o.height = dst_h;
  o.vmin = m.vmin;  // bilinear stays inside the convex hull of the samples
  o.vmax = m.vmax;
  const std::size_t n = static_cast<std::size_t>(dst_w) * static_cast<std::size_t>(dst_h);
  o.data.assign(n, 0.0f);
  if (n == 0) return o;

  const std::size_t sw = static_cast<std::size_t>(m.width);
  for (int y = 0; y < dst_h; ++y) {
    const Lerp& ly = sy[static_cast<std::size_t>(y)];
    const float* r0 = m.data.data() + static_cast<std::size_t>(ly.i0) * sw;
    const float* r1 = m.data.data() + static_cast<std::size_t>(ly.i1) * sw;
    float* drow = o.data.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(dst_w);
    for (int x = 0; x < dst_w; ++x) {
      const Lerp& lx = sx[static_cast<std::size_t>(x)];
      const float top = r0[lx.i0] * (1.0f - lx.w) + r0[lx.i1] * lx.w;
      const float bot = r1[lx.i0] * (1.0f - lx.w) + r1[lx.i1] * lx.w;
      drow[x] = top * (1.0f - ly.w) + bot * ly.w;
    }
  }
  return o;
}

/// Turbo colourmap approximation (a degree-5 polynomial fit of Google's Turbo).
/// Input clamped to [0,1]; RGB out. Good enough to look at, not exact.
void turbo(float t, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b) noexcept {
  t = std::min(1.0f, std::max(0.0f, t));
  const float t2 = t * t;
  const float t3 = t2 * t;
  const float t4 = t3 * t;
  const float t5 = t4 * t;
  float fr = 0.13572138f + 4.61539260f * t - 42.66032258f * t2 + 132.13108234f * t3 -
             152.94239396f * t4 + 59.28637943f * t5;
  float fg = 0.09140261f + 2.19418839f * t + 4.84296658f * t2 - 14.18503333f * t3 +
             4.27729857f * t4 + 2.82956604f * t5;
  float fb = 0.10667330f + 12.64194608f * t - 60.58204836f * t2 + 110.36276771f * t3 -
             89.90310912f * t4 + 27.34824973f * t5;
  fr = std::min(1.0f, std::max(0.0f, fr));
  fg = std::min(1.0f, std::max(0.0f, fg));
  fb = std::min(1.0f, std::max(0.0f, fb));
  r = static_cast<std::uint8_t>(std::lround(fr * 255.0f));
  g = static_cast<std::uint8_t>(std::lround(fg * 255.0f));
  b = static_cast<std::uint8_t>(std::lround(fb * 255.0f));
}

}  // namespace

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

DepthMap decodeDepth(const float* data, const std::vector<int>& shape, const DepthConfig& cfg) {
  int H = 0, W = 0;
  resolveHW(shape, H, W);
  if (cfg.height > 0) H = cfg.height;
  if (cfg.width > 0) W = cfg.width;

  DepthMap m;
  m.width = std::max(0, W);
  m.height = std::max(0, H);
  const std::int64_t n = static_cast<std::int64_t>(m.width) * m.height;
  if (n <= 0 || data == nullptr) return m;

  // Memory-safety bound: `data` holds product(shape) elements. A width/height
  // override (or a shape mismatch) that asks for more than that reads only what
  // exists and leaves the tail at zero, instead of walking off the end.
  std::int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? static_cast<std::int64_t>(d) : 0);
  const std::int64_t readable = std::min(n, total);

  m.data.assign(static_cast<std::size_t>(n), 0.0f);
  const bool do_clip = cfg.clip_hi > cfg.clip_lo;
  float vmin = std::numeric_limits<float>::infinity();
  float vmax = -std::numeric_limits<float>::infinity();

  for (std::int64_t i = 0; i < readable; ++i) {
    // Units first: raw -> physical, then disparity -> depth if asked. Both have
    // to happen before the clip and the min/max, or the thresholds and the
    // reported range would be in whatever arbitrary units the head emits.
    float v = data[i] * cfg.scale + cfg.shift;
    if (cfg.inverse) v = 1.0f / std::max(v, cfg.inverse_eps);
    if (do_clip) v = std::min(cfg.clip_hi, std::max(cfg.clip_lo, v));
    m.data[static_cast<std::size_t>(i)] = v;
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }
  if (readable > 0) {
    m.vmin = vmin;
    m.vmax = vmax;
  }

  if (cfg.normalize) {
    const float range = m.vmax - m.vmin;
    if (range > 0.0f) {
      const float inv = 1.0f / range;
      const float lo = m.vmin;
      for (std::int64_t i = 0; i < n; ++i) {
        m.data[static_cast<std::size_t>(i)] = (m.data[static_cast<std::size_t>(i)] - lo) * inv;
      }
    } else {
      // Flat map: no range to stretch. All-zeros beats a division by zero, and
      // vmin/vmax still tell the caller what the constant was.
      std::fill(m.data.begin(), m.data.end(), 0.0f);
    }
  }
  return m;
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

DepthMap depthResize(const DepthMap& m, int dst_w, int dst_h) {
  const int dw = std::max(0, dst_w);
  const int dh = std::max(0, dst_h);
  if (m.width <= 0 || m.height <= 0 ||
      m.data.size() < static_cast<std::size_t>(m.width) * static_cast<std::size_t>(m.height)) {
    DepthMap o;
    o.width = dw;
    o.height = dh;
    o.data.assign(static_cast<std::size_t>(dw) * static_cast<std::size_t>(dh), 0.0f);
    o.vmin = m.vmin;
    o.vmax = m.vmax;
    return o;
  }
  std::vector<Lerp> sx(static_cast<std::size_t>(dw)), sy(static_cast<std::size_t>(dh));
  for (int x = 0; x < dw; ++x) sx[static_cast<std::size_t>(x)] = lerpAt(centreMap(x, dw, m.width), m.width);
  for (int y = 0; y < dh; ++y) sy[static_cast<std::size_t>(y)] = lerpAt(centreMap(y, dh, m.height), m.height);
  return resampleBilinear(m, dw, dh, sx, sy);
}

DepthMap depthToSource(const DepthMap& m, const LetterboxInfo& lb) {
  RCDL_REQUIRE(lb.srcW > 0 && lb.srcH > 0 && lb.dstW > 0 && lb.dstH > 0,
               "RCDL depthToSource: letterbox has a non-positive extent");
  RCDL_REQUIRE(lb.scale > 0.0f, "RCDL depthToSource: letterbox scale must be > 0");
  if (m.width <= 0 || m.height <= 0 ||
      m.data.size() < static_cast<std::size_t>(m.width) * static_cast<std::size_t>(m.height)) {
    DepthMap o;
    o.width = lb.srcW;
    o.height = lb.srcH;
    o.data.assign(static_cast<std::size_t>(lb.srcW) * static_cast<std::size_t>(lb.srcH), 0.0f);
    o.vmin = m.vmin;
    o.vmax = m.vmax;
    return o;
  }

  // Source pixel centre -> canvas pixel -> map pixel. Mapping forward like this
  // (instead of cropping the padded border off and then resizing) needs no
  // intermediate image and stays exact when the head's grid is not a divisor of
  // the canvas — which is the normal case for a depth head.
  const float mx_ratio = static_cast<float>(m.width) / static_cast<float>(lb.dstW);
  const float my_ratio = static_cast<float>(m.height) / static_cast<float>(lb.dstH);
  std::vector<Lerp> sx(static_cast<std::size_t>(lb.srcW)), sy(static_cast<std::size_t>(lb.srcH));
  for (int x = 0; x < lb.srcW; ++x) {
    sx[static_cast<std::size_t>(x)] =
        lerpAt(lb.fwdX(static_cast<float>(x) + 0.5f) * mx_ratio - 0.5f, m.width);
  }
  for (int y = 0; y < lb.srcH; ++y) {
    sy[static_cast<std::size_t>(y)] =
        lerpAt(lb.fwdY(static_cast<float>(y) + 0.5f) * my_ratio - 0.5f, m.height);
  }
  return resampleBilinear(m, lb.srcW, lb.srcH, sx, sy);
}

// ---------------------------------------------------------------------------
// Visualisation
// ---------------------------------------------------------------------------

std::vector<std::uint8_t> depthToGray8(const DepthMap& m) {
  const std::size_t n = std::min(m.data.size(), static_cast<std::size_t>(std::max(0, m.width)) *
                                                    static_cast<std::size_t>(std::max(0, m.height)));
  std::vector<std::uint8_t> out(n, 0);
  if (n == 0) return out;

  // Stretch by the map's OWN range, decided once over the whole map, so this
  // renders correctly whether `data` is raw or already normalised to [0,1].
  float lo = m.data[0], hi = m.data[0];
  for (std::size_t i = 1; i < n; ++i) {
    lo = std::min(lo, m.data[i]);
    hi = std::max(hi, m.data[i]);
  }
  const float range = hi - lo;
  if (!(range > 0.0f)) return out;  // flat (or NaN) map -> all zeros
  const float inv = 1.0f / range;
  for (std::size_t i = 0; i < n; ++i) {
    const float t = std::min(1.0f, std::max(0.0f, (m.data[i] - lo) * inv));
    out[i] = static_cast<std::uint8_t>(std::lround(t * 255.0f));
  }
  return out;
}

std::vector<std::uint8_t> depthColorize(const DepthMap& m) {
  const std::vector<std::uint8_t> gray = depthToGray8(m);
  // 256 possible inputs, so the colourmap is a lookup table rather than a
  // polynomial evaluated per pixel.
  static const std::vector<std::uint8_t> kLut = [] {
    std::vector<std::uint8_t> lut(256 * 3);
    for (int i = 0; i < 256; ++i) {
      std::uint8_t r, g, b;
      turbo(static_cast<float>(i) / 255.0f, r, g, b);
      lut[static_cast<std::size_t>(i) * 3 + 0] = b;  // BGR, OpenCV order
      lut[static_cast<std::size_t>(i) * 3 + 1] = g;
      lut[static_cast<std::size_t>(i) * 3 + 2] = r;
    }
    return lut;
  }();

  std::vector<std::uint8_t> out(gray.size() * 3);
  for (std::size_t i = 0; i < gray.size(); ++i) {
    const std::size_t k = static_cast<std::size_t>(gray[i]) * 3;
    out[i * 3 + 0] = kLut[k + 0];
    out[i * 3 + 1] = kLut[k + 1];
    out[i * 3 + 2] = kLut[k + 2];
  }
  return out;
}

// ---------------------------------------------------------------------------
// Engine-bound estimator
// ---------------------------------------------------------------------------

DepthEstimator::DepthEstimator(Engine& engine, DepthConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "RCDL DepthEstimator: output index out of range");
  }
}

DepthMap DepthEstimator::postprocess() const {
  // Zero-copy when the head is packed F32 (which depth heads often are),
  // dequant-into-scratch for the fp16 / int8-affine cases.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);
  return decodeDepth(data, shape, cfg_);
}

DepthMap DepthEstimator::postprocess(const LetterboxInfo& lb) const {
  return depthToSource(postprocess(), lb);
}

}  // namespace rcdl
