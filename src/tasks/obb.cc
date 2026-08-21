#include "rcdl/tasks/obb.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

namespace {

constexpr float kPi = 3.14159265358979323846f;

/// A 2-D point for the polygon-clipping intersection routine.
///
/// The whole clip runs in DOUBLE even though the boxes are float: two boxes that
/// share an angle (the common case — a real object lights up several cells of the
/// same head) have exactly parallel edges, and in float the rounding on a corner
/// is the same order as the distance being tested, so a vertex that lies on a
/// clip line lands on the wrong side and the clipper emits a self-crossing
/// polygon whose shoelace area is meaningless. Two boxes are a few hundred per
/// frame, so the wider arithmetic costs nothing.
struct Pt {
  double x;
  double y;
};

/// Shoelace area of a polygon; absolute value, so the winding does not matter.
double polygonArea(const std::vector<Pt>& p) {
  if (p.size() < 3) return 0.0;
  double acc = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const Pt& a = p[i];
    const Pt& b = p[(i + 1) % p.size()];
    acc += a.x * b.y - b.x * a.y;
  }
  return std::fabs(acc) * 0.5;
}

/// Twice the signed area of the triangle a-b-p: positive when p is to the left
/// of the directed edge a->b. Divided by the edge length this is p's distance
/// from the line, which is how the caller turns it into a tolerance.
double cross(const Pt& a, const Pt& b, const Pt& p) {
  return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
}

/// Squared length of the edge a->b, the natural scale for that edge's tolerance.
double edgeLenSq(const Pt& a, const Pt& b) {
  const double dx = b.x - a.x;
  const double dy = b.y - a.y;
  return dx * dx + dy * dy;
}

/// A point within this many edge-lengths of a clip line counts as ON it, and so
/// as inside. Belt and braces rather than the fix: at double precision the
/// coincident-edge cases already come out right without it (it is the float
/// arithmetic above that they could not survive), but a shared edge leaves every
/// vertex's side decided by the last few bits, and this keeps that decision from
/// depending on them at all.
constexpr double kOnLineTol = 1e-9;

/// Intersection of segment p1->p2 with the infinite line a->b. When the segment
/// is (near-)parallel to the line there is no meaningful crossing point, so fall
/// back to whichever endpoint already lies closest to the line: returning the
/// far endpoint instead would push a vertex outside the clip region and inflate
/// the intersection area.
Pt lineIntersect(const Pt& p1, const Pt& p2, const Pt& a, const Pt& b) {
  const double a1 = b.y - a.y;
  const double b1 = a.x - b.x;
  const double c1 = a1 * a.x + b1 * a.y;
  const double a2 = p2.y - p1.y;
  const double b2 = p1.x - p2.x;
  const double c2 = a2 * p1.x + b2 * p1.y;
  const double det = a1 * b2 - a2 * b1;
  if (std::fabs(det) <= 1e-12 * (std::fabs(a1 * b2) + std::fabs(a2 * b1) + 1.0)) {
    return std::fabs(cross(a, b, p1)) <= std::fabs(cross(a, b, p2)) ? p1 : p2;
  }
  return Pt{(b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det};
}

/// Sutherland-Hodgman: clip the convex `subject` polygon against the convex
/// `clip` quad, returning the intersection polygon. The "inside" half-plane sign
/// is read from the clip quad's own winding rather than assumed, so the routine
/// does not care whether the corners came out clockwise or counter-clockwise
/// (which depends on the sign of the box angle).
std::vector<Pt> clipPolygon(const std::vector<Pt>& subject, const Pt clip[4]) {
  const double wind = cross(clip[0], clip[1], clip[2]);
  const double inside_sign = wind >= 0.0 ? 1.0 : -1.0;

  std::vector<Pt> output = subject;
  for (int e = 0; e < 4 && !output.empty(); ++e) {
    const Pt& a = clip[e];
    const Pt& b = clip[(e + 1) % 4];
    const double tol = kOnLineTol * edgeLenSq(a, b);
    const std::vector<Pt> input = output;
    output.clear();
    const std::size_t n = input.size();
    for (std::size_t i = 0; i < n; ++i) {
      const Pt& cur = input[i];
      const Pt& prv = input[(i + n - 1) % n];
      const bool cur_in = cross(a, b, cur) * inside_sign >= -tol;
      const bool prv_in = cross(a, b, prv) * inside_sign >= -tol;
      if (cur_in) {
        if (!prv_in) output.push_back(lineIntersect(prv, cur, a, b));
        output.push_back(cur);
      } else if (prv_in) {
        output.push_back(lineIntersect(prv, cur, a, b));
      }
    }
  }
  return output;
}

/// Σ b·softmax(b) over one side's `reg` DFL bins, walked with an arbitrary
/// element stride so the same code serves [C,H,W] (step = H*W) and [H,W,C]
/// (step = 1). Same reduction as decodeYoloLtrb's — the box branch of an OBB
/// export is the detection box branch.
float dflReduce(const float* bins, int reg, std::int64_t step) {
  float maxv = bins[0];
  for (int b = 1; b < reg; ++b) {
    maxv = std::max(maxv, bins[static_cast<std::int64_t>(b) * step]);
  }
  float sum = 0.0f;
  float acc = 0.0f;
  for (int b = 0; b < reg; ++b) {
    const float e = std::exp(bins[static_cast<std::int64_t>(b) * step] - maxv);
    sum += e;
    acc += e * static_cast<float>(b);
  }
  return (sum > 0.0f) ? acc / sum : 0.0f;
}

}  // namespace

void rotatedBoxCorners(const RotatedBox& r, float out[8]) {
  const float c = std::cos(r.angle);
  const float s = std::sin(r.angle);
  const float dx = r.w * 0.5f;
  const float dy = r.h * 0.5f;
  const float lx[4] = {-dx, dx, dx, -dx};
  const float ly[4] = {-dy, -dy, dy, dy};
  for (int i = 0; i < 4; ++i) {
    out[2 * i] = r.cx + lx[i] * c - ly[i] * s;
    out[2 * i + 1] = r.cy + lx[i] * s + ly[i] * c;
  }
}

float rotatedIoU(const RotatedBox& a, const RotatedBox& b) {
  const double area_a = static_cast<double>(a.w) * a.h;
  const double area_b = static_cast<double>(b.w) * b.h;
  if (area_a <= 0.0 || area_b <= 0.0) return 0.0f;

  float ca[8], cb[8];
  rotatedBoxCorners(a, ca);
  rotatedBoxCorners(b, cb);

  std::vector<Pt> subject(4);
  Pt clip[4];
  for (int i = 0; i < 4; ++i) {
    subject[static_cast<std::size_t>(i)] = Pt{ca[2 * i], ca[2 * i + 1]};
    clip[i] = Pt{cb[2 * i], cb[2 * i + 1]};
  }

  const double inter = polygonArea(clipPolygon(subject, clip));
  if (inter <= 0.0) return 0.0f;
  const double uni = area_a + area_b - inter;
  return uni > 0.0 ? static_cast<float>(inter / uni) : 0.0f;
}

std::vector<int> rotatedNms(const std::vector<ObbDetection>& dets, float iou_thresh,
                            int max_dets) {
  std::vector<int> order(dets.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return dets[static_cast<std::size_t>(a)].score > dets[static_cast<std::size_t>(b)].score;
  });

  std::vector<int> keep;
  std::vector<char> suppressed(dets.size(), 0);
  for (int oi = 0; oi < static_cast<int>(order.size()); ++oi) {
    const int i = order[static_cast<std::size_t>(oi)];
    if (suppressed[static_cast<std::size_t>(i)]) continue;
    keep.push_back(i);
    if (max_dets > 0 && static_cast<int>(keep.size()) >= max_dets) break;
    for (int oj = oi + 1; oj < static_cast<int>(order.size()); ++oj) {
      const int j = order[static_cast<std::size_t>(oj)];
      if (suppressed[static_cast<std::size_t>(j)]) continue;
      const ObbDetection& di = dets[static_cast<std::size_t>(i)];
      const ObbDetection& dj = dets[static_cast<std::size_t>(j)];
      if (dj.class_id != di.class_id) continue;  // per-class
      if (rotatedIoU(di.rrect, dj.rrect) > iou_thresh) suppressed[static_cast<std::size_t>(j)] = 1;
    }
  }
  return keep;
}

std::vector<ObbDetection> decodeObb(const std::vector<const float*>& cls,
                                    const std::vector<const float*>& box,
                                    const std::vector<const float*>& angle,
                                    const std::vector<std::pair<int, int>>& grid_hw,
                                    const ObbConfig& cfg, const LetterboxInfo& lb) {
  const int nc = cfg.num_classes;
  const std::size_t scales = grid_hw.size();
  RCDL_REQUIRE(cls.size() == scales && box.size() == scales && angle.size() == scales &&
                   cfg.strides.size() == scales,
               "RCDL decodeObb: cls/box/angle/grid/strides length mismatch");
  RCDL_REQUIRE(nc > 0, "RCDL decodeObb: num_classes must be > 0");
  const int reg = cfg.reg_max;
  RCDL_REQUIRE(reg >= 0, "RCDL decodeObb: reg_max must be >= 0");
  const int box_ch = (reg > 0) ? 4 * reg : 4;

  std::vector<ObbDetection> dets;
  for (std::size_t s = 0; s < scales; ++s) {
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    const float stride = static_cast<float>(cfg.strides[s]);
    const float* cp = cls[s];
    const float* bp = box[s];
    const float* ap = angle[s];
    if (cp == nullptr || bp == nullptr || ap == nullptr || H <= 0 || W <= 0) continue;

    // Channel-order abstraction hoisted out of the cell loop, as in
    // decodeYoloLtrb: channel c of a cell is at c*H*W + cell (channels-first) or
    // cell*C + c (channels-last).
    const std::int64_t plane = static_cast<std::int64_t>(H) * W;
    const std::int64_t chan_step = cfg.channels_first ? plane : 1;

    for (int gy = 0; gy < H; ++gy) {
      for (int gx = 0; gx < W; ++gx) {
        const std::int64_t cell = static_cast<std::int64_t>(gy) * W + gx;
        const float* logits = cp + (cfg.channels_first ? cell : cell * nc);

        // argmax over classes on the RAW values: sigmoid is monotonic, so the
        // winner is the same and we pay at most one exp() per surviving cell.
        int best_k = 0;
        float best_raw = logits[0];
        for (int k = 1; k < nc; ++k) {
          const float v = logits[static_cast<std::int64_t>(k) * chan_step];
          if (v > best_raw) {
            best_raw = v;
            best_k = k;
          }
        }
        const float score = cfg.apply_sigmoid ? sigmoid(best_raw) : best_raw;
        if (score < cfg.conf_thresh) continue;

        const float* bb = bp + (cfg.channels_first ? cell : cell * box_ch);
        float d[4];
        if (reg > 0) {
          for (int side = 0; side < 4; ++side) {
            d[side] = dflReduce(bb + static_cast<std::int64_t>(side) * reg * chan_step, reg,
                                chan_step);
          }
        } else {
          for (int side = 0; side < 4; ++side) {
            d[side] = bb[static_cast<std::int64_t>(side) * chan_step];
          }
        }
        // A DFL reduction is already >= 0; the absolute value only guards a
        // plain-LTRB head, whose distances must not fold the box inside out.
        const float l = std::fabs(d[0]);
        const float t = std::fabs(d[1]);
        const float r = std::fabs(d[2]);
        const float bot = std::fabs(d[3]);

        const float raw_a = ap[cell];
        const float act_a = cfg.apply_angle_sigmoid ? sigmoid(raw_a) : raw_a;
        const float a_rad = (act_a - cfg.angle_bias) * kPi;

        // The LTRB distances are measured in the BOX's frame, so the centre
        // offset they imply has to be rotated into the image frame before it is
        // added to the cell centre.
        const float grid_x = static_cast<float>(gx) + 0.5f;
        const float grid_y = static_cast<float>(gy) + 0.5f;
        const float xf = (r - l) * 0.5f;
        const float yf = (bot - t) * 0.5f;
        const float ca = std::cos(a_rad);
        const float sa = std::sin(a_rad);

        ObbDetection det;
        det.rrect.cx = (grid_x + xf * ca - yf * sa) * stride;
        det.rrect.cy = (grid_y + xf * sa + yf * ca) * stride;
        det.rrect.w = (l + r) * stride;
        det.rrect.h = (t + bot) * stride;
        det.rrect.angle = a_rad;
        if (cfg.regularize && det.rrect.w < det.rrect.h) {
          std::swap(det.rrect.w, det.rrect.h);
          det.rrect.angle += kPi * 0.5f;
        }
        det.score = score;
        det.class_id = best_k;
        dets.push_back(det);
      }
    }
  }

  // NMS runs in MODEL-INPUT pixels: the letterbox map is a uniform scale, so it
  // changes every area by the same factor and cannot change which box wins.
  const std::vector<int> keep = rotatedNms(dets, cfg.iou_thresh, cfg.max_dets);
  const float inv_scale = lb.scale > 0.0f ? 1.0f / lb.scale : 1.0f;
  std::vector<ObbDetection> out;
  out.reserve(keep.size());
  for (int idx : keep) {
    ObbDetection det = dets[static_cast<std::size_t>(idx)];
    // Un-letterbox. The centre maps back affinely and the sides divide by the
    // uniform scale; the angle is invariant under that map. Note the centre is
    // NOT clamped to the source extent the way an axis-aligned box is — clamping
    // a centre would silently translate the whole rectangle.
    det.rrect.cx = lb.invX(det.rrect.cx);
    det.rrect.cy = lb.invY(det.rrect.cy);
    det.rrect.w *= inv_scale;
    det.rrect.h *= inv_scale;
    out.push_back(det);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Head resolution from the Engine's output signature
// ---------------------------------------------------------------------------

namespace {

const char* formatName(rknn_tensor_format fmt) noexcept {
  switch (fmt) {
    case RKNN_TENSOR_NCHW: return "NCHW";
    case RKNN_TENSOR_NHWC: return "NHWC";
    case RKNN_TENSOR_NC1HWC2: return "NC1HWC2";
    default: return "UNDEFINED";
  }
}

/// Every output as "[i] 'name' [d0,d1,...] fmt DTYPE" — appended to every error
/// thrown here, because "resolve failed" without the signature is useless when a
/// model turns out to have a head RCDL does not know.
std::string describeOutputs(const Engine& engine) {
  std::ostringstream os;
  for (int i = 0; i < engine.numOutputs(); ++i) {
    const rknn_tensor_attr& a = engine.outputAttr(i);
    os << "\n  [" << i << "] '" << engine.outputName(i) << "' [";
    for (std::uint32_t d = 0; d < a.n_dims; ++d) {
      if (d) os << ",";
      os << a.dims[d];
    }
    os << "] fmt " << formatName(a.fmt) << " " << dtypeName(a.type);
  }
  return os.str();
}

[[noreturn]] void failObb(const Engine& engine, const std::string& why) {
  throw Error(-1, "RCDL resolveObbHead: " + why + ". Model outputs:" + describeOutputs(engine));
}

/// One output reduced to what head resolution cares about: how many channels it
/// has per cell, and how many cells it covers.
struct BranchInfo {
  int index = -1;
  int c = 0;               ///< channels per cell (grid branches)
  std::int64_t cells = 0;  ///< cells covered (H*W for a grid branch)
  std::int64_t elems = 0;  ///< product of every dim — the only reading a fused
                           ///< angle tensor's axis split cannot invalidate
  int h = 0;               ///< 0 for a branch that is not on a feature scale
  int w = 0;
  bool channels_first = true;
  bool is_grid = false;  ///< (H,W) is the model input divided by one common stride
  int stride = 0;
};

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

/// Reduce every output to a BranchInfo and mark the ones that sit on a feature
/// scale. The rknn fmt says which axis is which: NHWC => [N,H,W,C], everything
/// else (NCHW, and the UNDEFINED the runtime reports for a plain NCHW logical
/// layout) => [N,C,H,W]; a 3-D output is the same minus the spatial split.
std::vector<BranchInfo> readBranches(const Engine& engine, int input_h, int input_w) {
  std::vector<BranchInfo> outs;
  outs.reserve(static_cast<std::size_t>(engine.numOutputs()));
  for (int i = 0; i < engine.numOutputs(); ++i) {
    const rknn_tensor_attr& a = engine.outputAttr(i);
    if (a.n_dims < 2 || a.n_dims > 4) {
      failObb(engine, "output " + std::to_string(i) + " is " + std::to_string(a.n_dims) +
                          "-D, a head branch is 2-D to 4-D");
    }
    if (a.dims[0] != 1) {
      failObb(engine, "output " + std::to_string(i) + " has batch " +
                          std::to_string(a.dims[0]) + ", only batch 1 is supported");
    }
    BranchInfo o;
    o.index = i;
    o.channels_first = a.fmt != RKNN_TENSOR_NHWC;
    o.elems = 1;
    for (std::uint32_t d = 0; d < a.n_dims; ++d) {
      o.elems *= static_cast<std::int64_t>(a.dims[d]);
    }
    if (a.n_dims == 4) {
      if (o.channels_first) {
        o.c = static_cast<int>(a.dims[1]);
        o.h = static_cast<int>(a.dims[2]);
        o.w = static_cast<int>(a.dims[3]);
      } else {
        o.h = static_cast<int>(a.dims[1]);
        o.w = static_cast<int>(a.dims[2]);
        o.c = static_cast<int>(a.dims[3]);
      }
      o.cells = static_cast<std::int64_t>(o.h) * o.w;
    } else if (a.n_dims == 3) {
      if (o.channels_first) {
        o.c = static_cast<int>(a.dims[1]);
        o.cells = static_cast<std::int64_t>(a.dims[2]);
      } else {
        o.cells = static_cast<std::int64_t>(a.dims[1]);
        o.c = static_cast<int>(a.dims[2]);
      }
    } else {  // 2-D [1, N]: only ever a fused branch, read through `elems`
      o.c = 1;
      o.cells = static_cast<std::int64_t>(a.dims[1]);
    }
    if (o.c <= 0 || o.cells <= 0 || o.elems <= 0) {
      failObb(engine, "output " + std::to_string(i) + " has a non-positive channel or cell count");
    }
    // The one discriminator that does not need channel counts: a feature-scale
    // branch has the model input's aspect divided by a single integer stride.
    if (o.h > 0 && o.w > 0 && input_h > 0 && input_w > 0 && input_h % o.h == 0 &&
        input_w % o.w == 0 && input_h / o.h == input_w / o.w) {
      o.is_grid = true;
      o.stride = input_h / o.h;
    }
    outs.push_back(o);
  }
  return outs;
}

/// Group the feature-scale branches by (H,W), largest grid (finest stride) first.
std::vector<std::vector<BranchInfo>> groupByGrid(const std::vector<BranchInfo>& outs) {
  std::vector<std::vector<BranchInfo>> groups;
  for (const BranchInfo& o : outs) {
    if (!o.is_grid) continue;
    auto it = std::find_if(groups.begin(), groups.end(),
                           [&](const std::vector<BranchInfo>& g) {
                             return g.front().h == o.h && g.front().w == o.w;
                           });
    if (it == groups.end()) {
      groups.push_back({o});
    } else {
      it->push_back(o);
    }
  }
  std::sort(groups.begin(), groups.end(),
            [](const std::vector<BranchInfo>& x, const std::vector<BranchInfo>& y) {
              return x.front().cells > y.front().cells;
            });
  return groups;
}

}  // namespace

ObbHeadLayout resolveObbHead(const Engine& engine, int num_classes) {
  if (num_classes <= 0) num_classes = 15;
  const std::pair<int, int> in_hw = modelInputHw(engine);
  if (in_hw.first <= 0 || in_hw.second <= 0) {
    failObb(engine, "input 0 is not a 4-D image tensor, cannot tell scale branches apart");
  }
  const std::vector<BranchInfo> outs = readBranches(engine, in_hw.first, in_hw.second);
  const std::vector<std::vector<BranchInfo>> groups = groupByGrid(outs);
  if (groups.empty()) {
    failObb(engine, "no output has a grid that is the model input divided by a stride");
  }

  std::int64_t total_anchors = 0;
  for (const std::vector<BranchInfo>& g : groups) total_anchors += g.front().cells;

  // 1. The shared angle branch, if any: the one output that is not on a feature
  //    scale. It must cover every anchor and carry exactly one channel.
  ObbHeadLayout layout;
  layout.total_anchors = static_cast<int>(total_anchors);
  {
    std::vector<const BranchInfo*> loose;
    for (const BranchInfo& o : outs) {
      if (!o.is_grid) loose.push_back(&o);
    }
    if (loose.size() > 1) {
      failObb(engine, "more than one output sits outside the feature scales; "
                      "an OBB head has at most one shared angle tensor");
    }
    if (loose.size() == 1) {
      // Read it through the ELEMENT COUNT, not the axis split: the runtime spells
      // this one buffer [1,1,A], [1,A] or [1,1,A,1] depending on the export, and
      // all that matters is that it holds exactly one angle per anchor.
      const BranchInfo& k = *loose.front();
      if (k.elems != total_anchors) {
        failObb(engine, "angle output " + std::to_string(k.index) + " holds " +
                            std::to_string(k.elems) + " elements but the scales define " +
                            std::to_string(total_anchors) + " anchors (one angle each)");
      }
      layout.shared_angle_index = k.index;
    }
  }

  // 2. Resolve each scale group into (cls, box) + optional per-scale angle.
  int anchor_offset = 0;
  for (const std::vector<BranchInfo>& g : groups) {
    const std::string where = "grid " + std::to_string(g.front().h) + "x" +
                              std::to_string(g.front().w);
    std::vector<BranchInfo> rest = g;

    // 2a. Pull out this scale's angle branch when there is no shared one. A
    //     1-channel branch is unambiguous here: an OBB export has no score-sum
    //     pre-filter branch, and a 1-class OBB model does not exist.
    int angle_index = -1;
    if (layout.shared_angle_index < 0) {
      std::vector<std::size_t> cand;
      for (std::size_t i = 0; i < rest.size(); ++i) {
        if (rest[i].c == 1) cand.push_back(i);
      }
      if (cand.size() != 1) {
        failObb(engine, where + ": " + (cand.empty() ? "no branch" : "more than one branch") +
                            " is the 1-channel angle branch");
      }
      angle_index = rest[cand.front()].index;
      rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(cand.front()));
    }

    // 2b. What remains is either one fused [4*reg_max+nc] tensor or a cls/box pair.
    ObbScaleOutputs sc;
    sc.grid_h = g.front().h;
    sc.grid_w = g.front().w;
    sc.angle_index = angle_index;
    sc.anchor_offset = anchor_offset;
    sc.num_classes = num_classes;
    if (rest.size() == 1) {
      layout.fused_box_cls = true;
      const int box_ch = rest[0].c - num_classes;
      if (box_ch < 4 || box_ch % 4 != 0) {
        failObb(engine, where + ": a fused branch of " + std::to_string(rest[0].c) +
                            " channels minus " + std::to_string(num_classes) +
                            " classes leaves " + std::to_string(box_ch) +
                            " box channels, expected 4 or 4*reg_max");
      }
      sc.cls_index = rest[0].index;
      sc.box_index = rest[0].index;
      sc.box_channels = box_ch;
    } else if (rest.size() == 2) {
      const BranchInfo& a = rest[0];
      const BranchInfo& b = rest[1];
      bool a_is_cls;
      if ((a.c == num_classes) != (b.c == num_classes)) {
        a_is_cls = a.c == num_classes;
      } else if ((a.c % 4 == 0 && a.c >= 4) != (b.c % 4 == 0 && b.c >= 4)) {
        a_is_cls = !(a.c % 4 == 0 && a.c >= 4);
      } else {
        failObb(engine, where + ": cannot tell the cls branch (" + std::to_string(a.c) +
                            " ch) from the box branch (" + std::to_string(b.c) +
                            " ch) — pass num_classes to disambiguate");
      }
      const BranchInfo& cls = a_is_cls ? a : b;
      const BranchInfo& box = a_is_cls ? b : a;
      if (box.c < 4 || box.c % 4 != 0) {
        failObb(engine, where + ": box branch has " + std::to_string(box.c) +
                            " channels, expected 4 (plain LTRB) or 4*reg_max");
      }
      sc.cls_index = cls.index;
      sc.box_index = box.index;
      sc.box_channels = box.c;
      sc.num_classes = cls.c;
    } else {
      failObb(engine, where + " has " + std::to_string(rest.size()) +
                          " tensors left after the angle branch; an OBB scale is one fused "
                          "box+cls tensor or a cls/box pair");
    }
    layout.scales.push_back(sc);
    anchor_offset += static_cast<int>(g.front().cells);
  }

  // 3. Cross-scale consistency + the derived quantities the decoder needs.
  layout.channels_first = outs.front().channels_first;
  for (const BranchInfo& o : outs) {
    if (o.channels_first != layout.channels_first) {
      failObb(engine, "outputs mix NCHW and NHWC channel order");
    }
  }
  layout.num_classes = layout.scales.front().num_classes;
  const int box_channels = layout.scales.front().box_channels;
  for (const ObbScaleOutputs& sc : layout.scales) {
    if (sc.num_classes != layout.num_classes) {
      failObb(engine, "class count differs across scales");
    }
    if (sc.box_channels != box_channels) {
      failObb(engine, "box channel count differs across scales");
    }
  }
  // 64 box channels => reg_max 16 (the ultralytics DFL head); exactly 4 => plain.
  layout.reg_max = (box_channels != 4) ? box_channels / 4 : 0;
  layout.strides.reserve(layout.scales.size());
  for (const std::vector<BranchInfo>& g : groups) layout.strides.push_back(g.front().stride);
  return layout;
}

std::string ObbHeadLayout::describe() const {
  std::ostringstream os;
  os << "YOLO OBB head: " << scales.size() << " scale(s), " << num_classes << " classes, "
     << (reg_max > 0 ? "DFL reg_max=" + std::to_string(reg_max) : std::string("plain LTRB"))
     << ", " << (channels_first ? "NCHW" : "NHWC")
     << (fused_box_cls ? ", fused box+cls" : "");
  if (shared_angle_index >= 0) {
    os << ", shared angle=out[" << shared_angle_index << "] over " << total_anchors
       << " anchors";
  }
  for (std::size_t i = 0; i < scales.size(); ++i) {
    const ObbScaleOutputs& sc = scales[i];
    os << "\n  scale " << i << ": grid " << sc.grid_h << "x" << sc.grid_w;
    if (i < strides.size()) os << " stride " << strides[i];
    os << "  cls=out[" << sc.cls_index << "](" << sc.num_classes << "ch)"
       << " box=out[" << sc.box_index << "](" << sc.box_channels << "ch)";
    if (sc.angle_index >= 0) os << " angle=out[" << sc.angle_index << "]";
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// Engine-bound detector
// ---------------------------------------------------------------------------

ObbDetector::ObbDetector(Engine& engine, ObbConfig cfg)
    : engine_(engine), cfg_(std::move(cfg)), layout_(resolveObbHead(engine, cfg_.num_classes)) {}

std::vector<ObbDetection> ObbDetector::postprocess(const LetterboxInfo& lb) const {
  const std::size_t scales = layout_.scales.size();
  const int nc = layout_.num_classes;

  const auto elems = [](const std::vector<int>& shape) {
    std::int64_t n = 1;
    for (int d : shape) n *= (d > 0 ? d : 1);
    return n;
  };

  // The scratch buffers below must outlive the decode, which reads through the
  // pointers taken from them.
  std::vector<std::vector<float>> main_buf(scales), box_buf(scales), ang_buf(scales);
  std::vector<std::vector<float>> cls_scratch(scales);
  std::vector<const float*> cls_ptr(scales), box_ptr(scales), ang_ptr(scales);
  std::vector<std::pair<int, int>> grid_hw(scales);

  // The shared angle tensor is one buffer for the whole model, so read it once
  // and hand each scale a window into it.
  std::vector<float> shared_ang_buf;
  const float* shared_ang = nullptr;
  if (layout_.shared_angle_index >= 0) {
    std::vector<int> shape;
    shared_ang = outputAsFloat(engine_, layout_.shared_angle_index, shared_ang_buf, shape);
    if (elems(shape) < layout_.total_anchors) {
      throw Error(-1, "RCDL ObbDetector: angle output smaller than the resolved layout:\n" +
                          layout_.describe());
    }
  }

  for (std::size_t s = 0; s < scales; ++s) {
    const ObbScaleOutputs& sc = layout_.scales[s];
    const std::int64_t cells = static_cast<std::int64_t>(sc.grid_h) * sc.grid_w;
    grid_hw[s] = {sc.grid_h, sc.grid_w};

    std::vector<int> main_shape;
    const float* main = outputAsFloat(engine_, sc.cls_index, main_buf[s], main_shape);

    if (layout_.fused_box_cls) {
      if (elems(main_shape) < cells * (sc.box_channels + nc)) {
        throw Error(-1, "RCDL ObbDetector: output smaller than the resolved layout:\n" +
                            layout_.describe());
      }
      if (layout_.channels_first) {
        // [C,H,W]: channel c of a cell is c*H*W + cell, so the class block simply
        // starts box_channels planes in — two pointers into one buffer, no copy.
        box_ptr[s] = main;
        cls_ptr[s] = main + static_cast<std::int64_t>(sc.box_channels) * cells;
      } else {
        // [H,W,C]: box and class channels interleave per cell, and the decoder
        // walks each buffer with its own per-cell stride, so split them out.
        const int total_ch = sc.box_channels + nc;
        box_buf[s].resize(static_cast<std::size_t>(cells) * sc.box_channels);
        cls_scratch[s].resize(static_cast<std::size_t>(cells) * nc);
        for (std::int64_t cell = 0; cell < cells; ++cell) {
          const float* src = main + cell * total_ch;
          std::copy(src, src + sc.box_channels, box_buf[s].begin() + cell * sc.box_channels);
          std::copy(src + sc.box_channels, src + total_ch, cls_scratch[s].begin() + cell * nc);
        }
        box_ptr[s] = box_buf[s].data();
        cls_ptr[s] = cls_scratch[s].data();
      }
    } else {
      std::vector<int> box_shape;
      box_ptr[s] = outputAsFloat(engine_, sc.box_index, box_buf[s], box_shape);
      cls_ptr[s] = main;
      if (elems(main_shape) < cells * nc || elems(box_shape) < cells * sc.box_channels) {
        throw Error(-1, "RCDL ObbDetector: output smaller than the resolved layout:\n" +
                            layout_.describe());
      }
    }

    if (shared_ang != nullptr) {
      ang_ptr[s] = shared_ang + sc.anchor_offset;
    } else {
      std::vector<int> ang_shape;
      ang_ptr[s] = outputAsFloat(engine_, sc.angle_index, ang_buf[s], ang_shape);
      if (elems(ang_shape) < cells) {
        throw Error(-1, "RCDL ObbDetector: angle output smaller than the resolved layout:\n" +
                            layout_.describe());
      }
    }
  }

  // Everything structural comes from the layout (read from the model), so a
  // mis-configured cfg cannot make the decoder index past a buffer; cfg supplies
  // only the thresholds and the angle conventions.
  ObbConfig eff = cfg_;
  eff.num_classes = nc;
  eff.reg_max = layout_.reg_max;
  eff.channels_first = layout_.channels_first;
  if (!layout_.strides.empty()) eff.strides = layout_.strides;
  RCDL_REQUIRE(eff.strides.size() == scales,
               "RCDL ObbDetector: stride count does not match the head's scale count");
  return decodeObb(cls_ptr, box_ptr, ang_ptr, grid_hw, eff, lb);
}

// ---------------------------------------------------------------------------
// DOTA class names
// ---------------------------------------------------------------------------

const std::vector<std::string>& dotaClassNames() {
  // DOTA-v1's 15 categories in the order every OBB export's class indices refer
  // to. Function-local static: built once, never copied.
  static const std::vector<std::string> kNames = {
      "plane",       "ship",           "storage tank",       "baseball diamond",
      "tennis court", "basketball court", "ground track field", "harbor",
      "bridge",      "large vehicle",  "small vehicle",      "helicopter",
      "roundabout",  "soccer ball field", "swimming pool"};
  return kNames;
}

const char* dotaClassName(int class_id) {
  const std::vector<std::string>& names = dotaClassNames();
  if (class_id >= 0 && class_id < static_cast<int>(names.size())) {
    return names[static_cast<std::size_t>(class_id)].c_str();
  }
  // Out-of-range ids come from non-DOTA models, so name them rather than throw.
  // A small rotating cache keeps the returned pointer valid for the caller.
  static thread_local std::string fallback[4];
  static thread_local int next = 0;
  std::string& slot = fallback[next];
  next = (next + 1) % 4;
  slot = "class " + std::to_string(class_id);
  return slot.c_str();
}

}  // namespace rcdl
