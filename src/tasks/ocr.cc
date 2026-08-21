#include "rcdl/tasks/ocr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

namespace {

struct Pt {
  float x;
  float y;
};

/// Signed area of the triangle (o,a,b), doubled. > 0 for a left turn in the
/// standard orientation, which is a RIGHT turn on screen (image y grows down) —
/// the hull code below only ever compares it against 0, so the flip is harmless.
float cross(const Pt& o, const Pt& a, const Pt& b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

/// Convex hull by Andrew's monotone chain. Returns the hull vertices in order,
/// without the closing repeat. Collinear points are dropped (the <= 0 test), so
/// a degenerate point set can come back with 1 or 2 vertices — every caller
/// below handles that rather than assuming a polygon.
std::vector<Pt> convexHull(std::vector<Pt> p) {
  const std::size_t n = p.size();
  if (n < 3) return p;
  std::sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
    return a.x != b.x ? a.x < b.x : a.y < b.y;
  });
  std::vector<Pt> h(2 * n);
  std::size_t k = 0;
  for (std::size_t i = 0; i < n; ++i) {  // lower chain
    while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0.0f) --k;
    h[k++] = p[i];
  }
  const std::size_t lower = k + 1;
  for (std::size_t i = n - 1; i-- > 0;) {  // upper chain
    while (k >= lower && cross(h[k - 2], h[k - 1], p[i]) <= 0.0f) --k;
    h[k++] = p[i];
  }
  h.resize(k > 0 ? k - 1 : 0);  // the last point repeats the first
  return h;
}

/// Reorder four corners into the reference decoder's TL, TR, BR, BL.
///
/// Sort by x (STABLY — the reference's Python sort is stable, and a square's two
/// pairs of equal x must keep their relative order for the y test below to pick
/// the same corners), then the upper of the left pair is TL and the upper of the
/// right pair is TR.
void orderQuad(const Pt in[4], float out[8]) {
  Pt s[4] = {in[0], in[1], in[2], in[3]};
  std::stable_sort(s, s + 4, [](const Pt& a, const Pt& b) { return a.x < b.x; });
  const int i0 = (s[1].y > s[0].y) ? 0 : 1;  // top-left
  const int i3 = (s[1].y > s[0].y) ? 1 : 0;  // bottom-left
  const int i1 = (s[3].y > s[2].y) ? 2 : 3;  // top-right
  const int i2 = (s[3].y > s[2].y) ? 3 : 2;  // bottom-right
  const int order[4] = {i0, i1, i2, i3};
  for (int k = 0; k < 4; ++k) {
    out[2 * k] = s[order[k]].x;
    out[2 * k + 1] = s[order[k]].y;
  }
}

/// Mean probability inside a quadrilateral (the reference's box_score_fast).
///
/// Scanline even-odd fill over the quad's clipped bounding rows, sampling pixel
/// centres at integer coordinates — the same convention the reference's polygon
/// rasteriser uses, so the two masks agree to within a boundary pixel.
///
/// A quad too thin to contain any pixel centre would score 0 and be dropped, so
/// the fallback samples the pixel nearest its centroid instead: a sliver is the
/// min_size filter's business, not the score's.
float boxScoreFast(const float* prob, int H, int W, const float quad[8]) {
  float minx = quad[0], maxx = quad[0], miny = quad[1], maxy = quad[1];
  for (int k = 1; k < 4; ++k) {
    minx = std::min(minx, quad[2 * k]);
    maxx = std::max(maxx, quad[2 * k]);
    miny = std::min(miny, quad[2 * k + 1]);
    maxy = std::max(maxy, quad[2 * k + 1]);
  }
  const int x0 = std::max(0, std::min(W - 1, static_cast<int>(std::floor(minx))));
  const int x1 = std::max(0, std::min(W - 1, static_cast<int>(std::ceil(maxx))));
  const int y0 = std::max(0, std::min(H - 1, static_cast<int>(std::floor(miny))));
  const int y1 = std::max(0, std::min(H - 1, static_cast<int>(std::ceil(maxy))));

  double sum = 0.0;
  std::int64_t count = 0;
  float xs[4];
  for (int y = y0; y <= y1; ++y) {
    const float fy = static_cast<float>(y);
    int nxs = 0;
    for (int k = 0; k < 4; ++k) {
      const float ay = quad[2 * k + 1], by = quad[2 * ((k + 1) & 3) + 1];
      // Half-open edge test: a vertex on the scanline is counted by exactly one
      // of its two edges, so a corner never produces a doubled crossing.
      if ((ay <= fy) == (by <= fy)) continue;
      const float ax = quad[2 * k], bx = quad[2 * ((k + 1) & 3)];
      xs[nxs++] = ax + (fy - ay) * (bx - ax) / (by - ay);
    }
    // Insertion sort of at most four crossings. std::sort on a 4-element stack
    // array trips gcc's -Warray-bounds on its own small-range threshold path,
    // and four values are not worth a call anyway.
    for (int a = 1; a < nxs; ++a) {
      const float key = xs[a];
      int b = a - 1;
      while (b >= 0 && xs[b] > key) {
        xs[b + 1] = xs[b];
        --b;
      }
      xs[b + 1] = key;
    }
    for (int s = 0; s + 1 < nxs; s += 2) {
      const int xa = std::max(x0, static_cast<int>(std::ceil(xs[s])));
      const int xb = std::min(x1, static_cast<int>(std::floor(xs[s + 1])));
      const float* row = prob + static_cast<std::ptrdiff_t>(y) * W;
      for (int x = xa; x <= xb; ++x) {
        sum += row[x];
        ++count;
      }
    }
  }
  if (count > 0) return static_cast<float>(sum / static_cast<double>(count));

  const int cx = std::max(0, std::min(W - 1, static_cast<int>((minx + maxx) * 0.5f)));
  const int cy = std::max(0, std::min(H - 1, static_cast<int>((miny + maxy) * 0.5f)));
  return prob[static_cast<std::ptrdiff_t>(cy) * W + cx];
}

/// Probability-map pixel -> original-image pixel.
///
/// Two hops, because they are two different things: (sx,sy) rescales a possibly
/// down-sampled head to the model-input canvas, then the letterbox inverse undoes
/// the padding and the fit. Identity when `lb` was never filled in.
class ProbToImage {
 public:
  ProbToImage(int H, int W, const LetterboxInfo& lb)
      : sx_(lb.dstW > 0 ? static_cast<float>(lb.dstW) / static_cast<float>(W) : 1.0f),
        sy_(lb.dstH > 0 ? static_cast<float>(lb.dstH) / static_cast<float>(H) : 1.0f),
        has_lb_(lb.dstW > 0 && lb.dstH > 0),
        lb_(lb) {}

  void operator()(float px, float py, float& ox, float& oy) const {
    if (!has_lb_) {
      ox = px;
      oy = py;
      return;
    }
    ox = lb_.clampX(lb_.invX(px * sx_));
    oy = lb_.clampY(lb_.invY(py * sy_));
  }

 private:
  float sx_;
  float sy_;
  bool has_lb_;
  const LetterboxInfo& lb_;
};

/// Fill a TextBox from four mapped corners: `pts` verbatim, `x1..y2` their extent.
TextBox makeBox(const float pts[8], float score) {
  TextBox b;
  b.score = score;
  b.x1 = b.x2 = pts[0];
  b.y1 = b.y2 = pts[1];
  for (int k = 0; k < 4; ++k) {
    b.pts[2 * k] = pts[2 * k];
    b.pts[2 * k + 1] = pts[2 * k + 1];
    b.x1 = std::min(b.x1, pts[2 * k]);
    b.x2 = std::max(b.x2, pts[2 * k]);
    b.y1 = std::min(b.y1, pts[2 * k + 1]);
    b.y2 = std::max(b.y2, pts[2 * k + 1]);
  }
  return b;
}

/// (H, W, C) of an output tensor, read from its OWN dims + format.
///
/// The last three dims are (H,W,C) under NHWC and (C,H,W) under anything else —
/// the runtime reports UNDEFINED for the plain NCHW logical layout, so only an
/// explicit NHWC means channels-last. A 2-D tensor is a bare (H,W).
/// Returns false when the tensor has too few dims to be a map at all.
bool resolveMapHWC(const rknn_tensor_attr& a, int& H, int& W, int& C) {
  const int n = static_cast<int>(a.n_dims);
  if (n >= 3) {
    const int d0 = static_cast<int>(a.dims[n - 3]);
    const int d1 = static_cast<int>(a.dims[n - 2]);
    const int d2 = static_cast<int>(a.dims[n - 1]);
    if (a.fmt == RKNN_TENSOR_NHWC) {
      H = d0;
      W = d1;
      C = d2;
    } else {
      C = d0;
      H = d1;
      W = d2;
    }
    return true;
  }
  if (n == 2) {
    H = static_cast<int>(a.dims[0]);
    W = static_cast<int>(a.dims[1]);
    C = 1;
    return true;
  }
  return false;
}

std::string shapeString(const rknn_tensor_attr& a) {
  std::ostringstream os;
  os << '[';
  for (std::uint32_t i = 0; i < a.n_dims; ++i) {
    if (i != 0) os << ',';
    os << a.dims[i];
  }
  os << ']';
  return os.str();
}

}  // namespace

// ---------------------------------------------------------------------------
// Quadrilateral geometry
// ---------------------------------------------------------------------------

void minAreaQuad(const float* xy, int n, float out[8], float* short_side) {
  for (int k = 0; k < 8; ++k) out[k] = 0.0f;
  if (short_side != nullptr) *short_side = 0.0f;
  if (xy == nullptr || n <= 0) return;

  std::vector<Pt> pts;
  pts.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) pts.push_back({xy[2 * i], xy[2 * i + 1]});
  const std::vector<Pt> hull = convexHull(std::move(pts));
  if (hull.empty()) return;

  // Rotating calipers: one side of the minimum-area rectangle is flush with a
  // hull edge, so the loop over edges is exhaustive rather than heuristic.
  float best_area = -1.0f;
  float best_w = 0.0f, best_h = 0.0f;
  Pt best_u{1.0f, 0.0f}, best_v{0.0f, 1.0f};
  float best_umin = 0.0f, best_vmin = 0.0f;
  const std::size_t m = hull.size();
  for (std::size_t i = 0; i < m; ++i) {
    const Pt& a = hull[i];
    const Pt& b = hull[(i + 1) % m];
    const float dx = b.x - a.x, dy = b.y - a.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len <= 0.0f) continue;
    const Pt u{dx / len, dy / len};
    const Pt v{-u.y, u.x};
    float umin = hull[0].x * u.x + hull[0].y * u.y, umax = umin;
    float vmin = hull[0].x * v.x + hull[0].y * v.y, vmax = vmin;
    for (std::size_t j = 1; j < m; ++j) {
      const float pu = hull[j].x * u.x + hull[j].y * u.y;
      const float pv = hull[j].x * v.x + hull[j].y * v.y;
      umin = std::min(umin, pu);
      umax = std::max(umax, pu);
      vmin = std::min(vmin, pv);
      vmax = std::max(vmax, pv);
    }
    const float w = umax - umin, h = vmax - vmin;
    const float area = w * h;
    if (best_area < 0.0f || area < best_area) {
      best_area = area;
      best_w = w;
      best_h = h;
      best_u = u;
      best_v = v;
      best_umin = umin;
      best_vmin = vmin;
    }
  }

  if (best_area < 0.0f) {  // every hull point identical — a single pixel
    Pt c[4] = {hull[0], hull[0], hull[0], hull[0]};
    orderQuad(c, out);
    return;
  }

  const float umax = best_umin + best_w, vmax = best_vmin + best_h;
  const auto corner = [&](float pu, float pv) {
    return Pt{pu * best_u.x + pv * best_v.x, pu * best_u.y + pv * best_v.y};
  };
  const Pt c[4] = {corner(best_umin, best_vmin), corner(umax, best_vmin),
                   corner(umax, vmax), corner(best_umin, vmax)};
  orderQuad(c, out);
  if (short_side != nullptr) *short_side = std::min(best_w, best_h);
}

void unclipQuad(const float in[8], float ratio, float out[8]) {
  // Shoelace area and perimeter of the quad — the two numbers the DB offset
  // distance is defined from.
  float area2 = 0.0f, perim = 0.0f;
  for (int k = 0; k < 4; ++k) {
    const int j = (k + 1) & 3;
    area2 += in[2 * k] * in[2 * j + 1] - in[2 * j] * in[2 * k + 1];
    const float ex = in[2 * j] - in[2 * k], ey = in[2 * j + 1] - in[2 * k + 1];
    perim += std::sqrt(ex * ex + ey * ey);
  }
  const float area = std::fabs(area2) * 0.5f;
  const float d = (perim > 0.0f) ? area * ratio / perim : 0.0f;
  // Sign of the shoelace tells the winding, and the winding tells which side of
  // an edge is OUTSIDE. Getting this backwards would shrink the box instead.
  const float wind = (area2 >= 0.0f) ? 1.0f : -1.0f;

  float res[8];
  for (int k = 0; k < 4; ++k) {
    const int prev = (k + 3) & 3;
    const int next = (k + 1) & 3;
    // Outward unit normals of the two edges meeting at corner k.
    const auto normal = [&](int from, int to, float& nx, float& ny) {
      const float ex = in[2 * to] - in[2 * from], ey = in[2 * to + 1] - in[2 * from + 1];
      const float len = std::sqrt(ex * ex + ey * ey);
      if (len <= 0.0f) {
        nx = 0.0f;
        ny = 0.0f;
        return;
      }
      nx = wind * ey / len;
      ny = -wind * ex / len;
    };
    float n0x, n0y, n1x, n1y;
    normal(prev, k, n0x, n0y);
    normal(k, next, n1x, n1y);
    // Miter: moving the corner by d*(n0+n1)/(1+n0.n1) pushes BOTH edges out by
    // exactly d. The denominator collapses only when the two edges double back
    // on each other, where a plain normal offset is the sane answer.
    const float dot = n0x * n1x + n0y * n1y;
    const float den = 1.0f + dot;
    float mx, my;
    if (den > 1e-6f) {
      mx = d * (n0x + n1x) / den;
      my = d * (n0y + n1y) / den;
    } else {
      mx = d * n1x;
      my = d * n1y;
    }
    res[2 * k] = in[2 * k] + mx;
    res[2 * k + 1] = in[2 * k + 1] + my;
  }
  for (int k = 0; k < 8; ++k) out[k] = res[k];
}

// ---------------------------------------------------------------------------
// A. Detection (DBNet)
// ---------------------------------------------------------------------------

std::vector<TextBox> decodeTextBoxes(const float* prob, int H, int W, const OcrDetConfig& cfg,
                                     const LetterboxInfo& lb) {
  std::vector<TextBox> boxes;
  if (prob == nullptr || H <= 0 || W <= 0) return boxes;
  const std::size_t n_pix = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);

  // A logit head needs one pass to become a probability map, because the score
  // that survives to TextBox::score has to be a probability, not a logit. The
  // common (already-sigmoided) export pays nothing for this.
  std::vector<float> activated;
  const float* p = prob;
  if (cfg.apply_sigmoid) {
    activated.resize(n_pix);
    for (std::size_t i = 0; i < n_pix; ++i) activated[i] = sigmoid(prob[i]);
    p = activated.data();
  }

  const ProbToImage to_image(H, W, lb);
  const int nneigh = (cfg.connectivity == 4) ? 4 : 8;
  static const int kDx[8] = {1, -1, 0, 0, 1, 1, -1, -1};
  static const int kDy[8] = {0, 0, 1, -1, 1, -1, 1, -1};

  std::vector<std::uint8_t> visited(n_pix, 0);
  std::vector<std::int32_t> stack;
  std::vector<float> border;  // interleaved x,y of the region's boundary pixels
  int candidates = 0;

  for (int sy = 0; sy < H && candidates < cfg.max_candidates; ++sy) {
    for (int sx = 0; sx < W && candidates < cfg.max_candidates; ++sx) {
      const std::int32_t seed = static_cast<std::int32_t>(sy) * W + sx;
      if (visited[static_cast<std::size_t>(seed)]) continue;
      if (!(p[seed] > cfg.bin_thresh)) continue;

      // Flood the region, keeping only its BOUNDARY pixels: the minimum-area
      // rectangle is a property of the convex hull, and the hull of a filled
      // region is the hull of its border, so a 200k-pixel blob contributes a few
      // hundred points instead of all of them.
      border.clear();
      stack.clear();
      stack.push_back(seed);
      visited[static_cast<std::size_t>(seed)] = 1;
      while (!stack.empty()) {
        const std::int32_t cur = stack.back();
        stack.pop_back();
        const int cx = cur % W;
        const int cy = cur / W;
        bool is_border = false;
        for (int k = 0; k < nneigh; ++k) {
          const int nx = cx + kDx[k], ny = cy + kDy[k];
          if (nx < 0 || nx >= W || ny < 0 || ny >= H) {
            if (k < 4) is_border = true;  // the frame edge bounds the region too
            continue;
          }
          const std::int32_t nidx = static_cast<std::int32_t>(ny) * W + nx;
          const bool fg = p[nidx] > cfg.bin_thresh;
          if (!fg) {
            if (k < 4) is_border = true;  // 4-connected background => on the border
            continue;
          }
          if (!visited[static_cast<std::size_t>(nidx)]) {
            visited[static_cast<std::size_t>(nidx)] = 1;
            stack.push_back(nidx);
          }
        }
        if (is_border) {
          border.push_back(static_cast<float>(cx));
          border.push_back(static_cast<float>(cy));
        }
      }
      ++candidates;
      if (border.size() < 2) continue;

      float quad[8], sside = 0.0f;
      minAreaQuad(border.data(), static_cast<int>(border.size() / 2), quad, &sside);
      if (sside < static_cast<float>(cfg.min_size)) continue;

      const float score = boxScoreFast(p, H, W, quad);
      if (score < cfg.box_thresh) continue;

      float grown[8], refit[8], sside2 = 0.0f;
      unclipQuad(quad, cfg.unclip_ratio, grown);
      // Re-fit after unclipping, as the reference does. For the rectangle this
      // always is, the fit is a no-op that re-orders the corners; the filter it
      // feeds (min_size + 2) is the point.
      minAreaQuad(grown, 4, refit, &sside2);
      if (sside2 < static_cast<float>(cfg.min_size + 2)) continue;

      float mapped[8];
      for (int k = 0; k < 4; ++k) {
        to_image(refit[2 * k], refit[2 * k + 1], mapped[2 * k], mapped[2 * k + 1]);
      }
      // Final filter in ORIGINAL pixels: a box can survive every check on the
      // map and still be a sliver once a big downscale is undone. Truncating to
      // int before comparing matches the reference's `int(norm(...)) <= 3`.
      const float ex = mapped[2] - mapped[0], ey = mapped[3] - mapped[1];
      const float fx = mapped[6] - mapped[0], fy = mapped[7] - mapped[1];
      const int bw = static_cast<int>(std::sqrt(ex * ex + ey * ey));
      const int bh = static_cast<int>(std::sqrt(fx * fx + fy * fy));
      if (bw <= cfg.min_box_side || bh <= cfg.min_box_side) continue;

      boxes.push_back(makeBox(mapped, score));
    }
  }
  return boxes;
}

void sortTextBoxes(std::vector<TextBox>& boxes, float row_tol) {
  std::stable_sort(boxes.begin(), boxes.end(), [](const TextBox& a, const TextBox& b) {
    if (a.pts[1] != b.pts[1]) return a.pts[1] < b.pts[1];
    return a.pts[0] < b.pts[0];
  });
  // Insertion pass over the (y,x) order: neighbours on the same visual row that
  // came out right-to-left are swapped back, and the scan stops at the first
  // neighbour that is NOT on the same row, so a row never steals from the next.
  for (std::size_t i = 0; i + 1 < boxes.size(); ++i) {
    for (std::ptrdiff_t j = static_cast<std::ptrdiff_t>(i); j >= 0; --j) {
      const TextBox& lo = boxes[static_cast<std::size_t>(j)];
      const TextBox& hi = boxes[static_cast<std::size_t>(j) + 1];
      if (std::fabs(hi.pts[1] - lo.pts[1]) < row_tol && hi.pts[0] < lo.pts[0]) {
        std::swap(boxes[static_cast<std::size_t>(j)], boxes[static_cast<std::size_t>(j) + 1]);
      } else {
        break;
      }
    }
  }
}

TextDetector::TextDetector(Engine& engine, OcrDetConfig cfg, int output_index)
    : engine_(engine), cfg_(cfg), out_idx_(output_index) {
  RCDL_REQUIRE(out_idx_ >= 0 && out_idx_ < engine_.numOutputs(),
               "RCDL TextDetector: output index out of range");
  const rknn_tensor_attr& attr = engine_.outputAttr(out_idx_);
  int c = 0;
  if (!resolveMapHWC(attr, map_h_, map_w_, c)) {
    throw Error(-1, "RCDL TextDetector: output " + shapeString(attr) +
                        " has too few dims to be a probability map");
  }
  // The DB head emits ONE channel, and with C == 1 an NHWC and an NCHW tensor
  // hold the same bytes in the same order — only the dims ORDER differs, which
  // is exactly why H/W come from the format above instead of from a guess. A
  // multi-channel output means this is not a DB probability map at all.
  if (c != 1 || map_h_ <= 0 || map_w_ <= 0) {
    std::ostringstream os;
    os << "RCDL TextDetector: output " << shapeString(attr) << " fmt="
       << (attr.fmt == RKNN_TENSOR_NHWC ? "NHWC" : "NCHW")
       << " is not a single-channel probability map (resolved H=" << map_h_
       << " W=" << map_w_ << " C=" << c << ")";
    throw Error(-1, os.str());
  }
}

std::vector<TextBox> TextDetector::postprocess(const LetterboxInfo& lb) const {
  // scratch must outlive `data`: the dequantizing path returns a pointer into it.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);
  if (data == nullptr) return {};
  return decodeTextBoxes(data, map_h_, map_w_, cfg_, lb);
}

std::string TextDetector::describe() const {
  const rknn_tensor_attr& attr = engine_.outputAttr(out_idx_);
  const QuantParams q = quantParams(attr);
  std::ostringstream os;
  os << map_w_ << 'x' << map_h_ << " map, " << dtypeName(attr.type);
  if (q.is_affine) os << " affine scale=" << q.scale << " zp=" << q.zero_point;
  os << ", sigmoid " << (cfg_.apply_sigmoid ? "on the CPU" : "in the graph");
  return os.str();
}

// ---------------------------------------------------------------------------
// B. Recognition (CRNN + CTC)
// ---------------------------------------------------------------------------

std::vector<std::string> loadCharDict(const std::string& path, bool paddle_special) {
  std::ifstream in(path);
  // An unreadable dictionary would otherwise decode every crop to the empty
  // string and look like a bad model, so it fails here instead.
  if (!in.is_open()) {
    throw Error(-1, "RCDL loadCharDict: cannot open the character dictionary: " + path);
  }
  std::vector<std::string> dict;
  if (paddle_special) dict.push_back("blank");  // CTC blank owns class 0
  std::string line;
  while (std::getline(in, line)) {
    // Trailing CR from a CRLF-terminated file, stripped so a dictionary edited on
    // another platform still matches the model's classes byte for byte.
    if (!line.empty() && line.back() == '\r') line.pop_back();
    dict.push_back(line);
  }
  if (paddle_special) dict.push_back(" ");  // use_space_char: the last class
  return dict;
}

TextLine ctcGreedyDecode(const float* logits, int num_steps, int num_classes,
                         const std::vector<std::string>& dict, const OcrRecConfig& cfg) {
  TextLine out;
  out.score = 0.0f;
  if (logits == nullptr || num_steps <= 0 || num_classes <= 0) return out;

  int prev = -1;  // no previous step yet, so step 0 can always emit
  double score_sum = 0.0;
  int emitted = 0;

  for (int t = 0; t < num_steps; ++t) {
    const float* row = logits + static_cast<std::ptrdiff_t>(t) * num_classes;
    int best = 0;
    float best_v = row[0];
    for (int c = 1; c < num_classes; ++c) {
      if (row[c] > best_v) {  // strict: a tie keeps the lowest class, as argmax does
        best_v = row[c];
        best = c;
      }
    }
    // Emit on a non-blank that DIFFERS from the previous step's class. The blank
    // is what separates a doubled character: "a a" is one 'a', "a _ a" is two.
    if (best != cfg.blank_index && best != prev) {
      if (best >= 0 && best < static_cast<int>(dict.size())) {
        out.text += dict[static_cast<std::size_t>(best)];
        float conf = best_v;
        if (cfg.apply_softmax) {
          // softmax(row)[best] with the max factored out: exp(0) / sum(exp(x-max)).
          double sum = 0.0;
          for (int c = 0; c < num_classes; ++c) sum += std::exp(row[c] - best_v);
          conf = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;
        }
        score_sum += conf;
        ++emitted;
      }
    }
    prev = best;
  }

  if (emitted > 0) out.score = static_cast<float>(score_sum / emitted);
  return out;
}

TextRecognizer::TextRecognizer(Engine& engine, const std::string& dict_path, OcrRecConfig cfg,
                               int output_index)
    : engine_(engine), dict_(loadCharDict(dict_path)), cfg_(cfg), out_idx_(output_index) {
  resolveAxes();
}

TextRecognizer::TextRecognizer(Engine& engine, std::vector<std::string> dict, OcrRecConfig cfg,
                               int output_index)
    : engine_(engine), dict_(std::move(dict)), cfg_(cfg), out_idx_(output_index) {
  resolveAxes();
}

void TextRecognizer::resolveAxes() {
  RCDL_REQUIRE(out_idx_ >= 0 && out_idx_ < engine_.numOutputs(),
               "RCDL TextRecognizer: output index out of range");
  const rknn_tensor_attr& attr = engine_.outputAttr(out_idx_);
  const int n = static_cast<int>(attr.n_dims);
  if (n < 2) {
    throw Error(-1, "RCDL TextRecognizer: output " + shapeString(attr) +
                        " is not a [1,T,C] / [1,C,T] sequence tensor");
  }
  // The two INNERMOST dims carry the sequence; anything before them is the batch.
  const int d0 = static_cast<int>(attr.dims[n - 2]);
  const int d1 = static_cast<int>(attr.dims[n - 1]);
  const int dict_n = static_cast<int>(dict_.size());
  // The dictionary is the discriminator: C is the class count by definition, and
  // a 3-D tensor's format is reported UNDEFINED so it cannot tell us. Only an
  // UNAMBIGUOUS match decides — a square output leaves cfg.time_major in charge.
  if (dict_n > 0 && d1 == dict_n && d0 != dict_n) {
    cfg_.time_major = true;
  } else if (dict_n > 0 && d0 == dict_n && d1 != dict_n) {
    cfg_.time_major = false;
  }
  num_steps_ = cfg_.time_major ? d0 : d1;
  num_classes_ = cfg_.time_major ? d1 : d0;
  if (num_steps_ <= 0 || num_classes_ <= 0) {
    throw Error(-1, "RCDL TextRecognizer: output " + shapeString(attr) +
                        " has a zero-length sequence or vocabulary");
  }
}

TextLine TextRecognizer::postprocess() const {
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);
  if (data == nullptr) return TextLine{std::string(), 0.0f};

  std::int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 0);
  const std::int64_t need = static_cast<std::int64_t>(num_steps_) * num_classes_;
  RCDL_REQUIRE(total >= need, "RCDL TextRecognizer: output smaller than T*C");

  if (cfg_.time_major) return ctcGreedyDecode(data, num_steps_, num_classes_, dict_, cfg_);

  // [1,C,T]: gather into a [T,C] scratch so the decoder stays a plain row-major
  // function. T is a few dozen, so this copy is negligible next to the argmax
  // that follows it — and a strided decoder would be one more thing to test.
  std::vector<float> tm(static_cast<std::size_t>(need));
  for (int t = 0; t < num_steps_; ++t) {
    for (int c = 0; c < num_classes_; ++c) {
      tm[static_cast<std::size_t>(t) * num_classes_ + c] =
          data[static_cast<std::ptrdiff_t>(c) * num_steps_ + t];
    }
  }
  return ctcGreedyDecode(tm.data(), num_steps_, num_classes_, dict_, cfg_);
}

std::string TextRecognizer::describe() const {
  std::ostringstream os;
  os << "T=" << num_steps_ << " C=" << num_classes_ << ' '
     << (cfg_.time_major ? "[1,T,C]" : "[1,C,T]") << ", dict " << dict_.size()
     << (cfg_.apply_softmax ? ", softmax on the CPU" : ", softmax in the graph");
  return os.str();
}

// ===========================================================================
// Text-line orientation (0 deg / 180 deg)
// ===========================================================================

int ocrLineFitWidth(int src_w, int src_h, int dst_w, int dst_h) {
  if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) return dst_w > 0 ? dst_w : 1;
  // ceil, as the reference does: a crop whose scaled width lands a fraction over
  // the input is capped rather than rounded down, so a wide line always fills
  // the width instead of leaving a one-pixel strip of padding.
  const double scaled = std::ceil(static_cast<double>(dst_h) * src_w / src_h);
  const int w = static_cast<int>(scaled);
  if (w >= dst_w) return dst_w;
  return w > 0 ? w : 1;
}

TextOrientation decodeTextOrientation(const float* scores, int n, float thresh) {
  TextOrientation out;
  if (scores == nullptr || n <= 0) return out;
  // Argmax over every element rather than over exactly two, so a head that
  // emits [1,N] (or carries a stray unit axis) still yields a sane label
  // instead of reading past the end of a two-element assumption.
  int best = 0;
  float best_v = scores[0];
  for (int i = 1; i < n; ++i) {
    if (scores[i] > best_v) {
      best_v = scores[i];
      best = i;
    }
  }
  out.label = best;
  out.score = best_v;
  out.flip180 = (best == 1 && best_v > thresh);
  return out;
}

TextAngleClassifier::TextAngleClassifier(Engine& engine, float thresh, int output_index)
    : engine_(engine), thresh_(thresh), out_idx_(output_index) {
  RCDL_REQUIRE(out_idx_ >= 0 && out_idx_ < engine.numOutputs(),
               "RCDL TextAngleClassifier: output index is out of range");
}

TextOrientation TextAngleClassifier::postprocess() const {
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);
  std::int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 0);
  RCDL_REQUIRE(data != nullptr && total > 0, "RCDL TextAngleClassifier: the output is empty");
  return decodeTextOrientation(data, static_cast<int>(total), thresh_);
}

std::string TextAngleClassifier::describe() const {
  std::ostringstream os;
  const std::vector<int> shape = engine_.outputShape(out_idx_);
  std::int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 0);
  os << total << " classes, flip above " << std::fixed << std::setprecision(2) << thresh_;
  return os.str();
}

}  // namespace rcdl
