#include "rcdl/tasks/pose.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

namespace {

/// Σ b·softmax(b) over one side's `reg` DFL bins. `step` is the element distance
/// between consecutive bins, so the same code serves [C,H,W] (step = H*W) and
/// [H,W,C] (step = 1). Identical reduction to decodeYoloLtrb's — the box branch
/// of a pose export is the detection box branch.
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

std::vector<PoseDetection> decodePose(const std::vector<const float*>& cls,
                                      const std::vector<const float*>& box,
                                      const std::vector<KeypointPlane>& kpt,
                                      const std::vector<std::pair<int, int>>& grid_hw,
                                      const PoseConfig& cfg, const LetterboxInfo& lb) {
  const int nc = cfg.num_classes;
  const int nk = cfg.num_keypoints;
  const std::size_t scales = grid_hw.size();
  RCDL_REQUIRE(cls.size() == scales && box.size() == scales && kpt.size() == scales &&
                   cfg.strides.size() == scales,
               "RCDL decodePose: cls/box/kpt/grid/strides length mismatch");
  RCDL_REQUIRE(nc > 0, "RCDL decodePose: num_classes must be > 0");
  RCDL_REQUIRE(nk >= 0, "RCDL decodePose: num_keypoints must be >= 0");
  const int reg = cfg.reg_max;
  RCDL_REQUIRE(reg >= 0, "RCDL decodePose: reg_max must be >= 0");
  const int box_ch = (reg > 0) ? 4 * reg : 4;

  std::vector<PoseDetection> dets;
  for (std::size_t s = 0; s < scales; ++s) {
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    const float stride = static_cast<float>(cfg.strides[s]);
    const float* cp = cls[s];
    const float* bp = box[s];
    const KeypointPlane& kplane = kpt[s];
    if (cp == nullptr || bp == nullptr || H <= 0 || W <= 0) continue;
    if (nk > 0 && kplane.data == nullptr) continue;

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

        const float cx = static_cast<float>(gx) + 0.5f;
        const float cy = static_cast<float>(gy) + 0.5f;

        PoseDetection det;
        det.box.x1 = lb.clampX(lb.invX((cx - d[0]) * stride));
        det.box.y1 = lb.clampY(lb.invY((cy - d[1]) * stride));
        det.box.x2 = lb.clampX(lb.invX((cx + d[2]) * stride));
        det.box.y2 = lb.clampY(lb.invY((cy + d[3]) * stride));
        det.box.score = score;
        det.box.class_id = best_k;

        // Keypoints. Channels are joint-major (3*j + {x,y,visibility}); the two
        // plane strides say where this cell's channels are, which is what lets
        // one loop read a shared [1,K*3,A] tensor and a per-scale [H,W,K*3] one.
        const float* kb = kplane.data + cell * kplane.cell_step;
        det.keypoints.reserve(static_cast<std::size_t>(nk));
        for (int k = 0; k < nk; ++k) {
          const std::int64_t base = static_cast<std::int64_t>(3 * k) * kplane.chan_step;
          const float raw_x = kb[base];
          const float raw_y = kb[base + kplane.chan_step];
          const float raw_s = kb[base + 2 * kplane.chan_step];

          float mkx = raw_x;
          float mky = raw_y;
          if (cfg.kpt_decode == KeypointDecode::kCellRelative) {
            // (raw*2 + (anchor - 0.5)) * stride with anchor = grid + 0.5. The
            // keypoint ADDS the grid; the box above SUBTRACTS a distance from it.
            mkx = (2.0f * raw_x + static_cast<float>(gx)) * stride;
            mky = (2.0f * raw_y + static_cast<float>(gy)) * stride;
          }

          Keypoint joint;
          joint.x = lb.clampX(lb.invX(mkx));
          joint.y = lb.clampY(lb.invY(mky));
          joint.score = cfg.kpt_apply_sigmoid ? sigmoid(raw_s) : raw_s;
          det.keypoints.push_back(joint);
        }

        dets.push_back(std::move(det));
      }
    }
  }

  // NMS on the person boxes with the shared rcdl::nms — a pose candidate is a
  // Detection plus joints, so there is nothing pose-specific to suppress on.
  std::vector<Detection> boxes;
  boxes.reserve(dets.size());
  for (const PoseDetection& p : dets) boxes.push_back(p.box);

  const std::vector<int> keep = nms(boxes, cfg.iou_thresh, cfg.max_dets);
  std::vector<PoseDetection> out;
  out.reserve(keep.size());
  for (int idx : keep) out.push_back(std::move(dets[static_cast<std::size_t>(idx)]));
  return out;
}

std::vector<PoseDetection> decodePose(const std::vector<const float*>& cls,
                                      const std::vector<const float*>& box,
                                      const std::vector<const float*>& kpt,
                                      const std::vector<std::pair<int, int>>& grid_hw,
                                      const PoseConfig& cfg, const LetterboxInfo& lb) {
  RCDL_REQUIRE(kpt.size() == grid_hw.size(),
               "RCDL decodePose: kpt/grid length mismatch");
  const std::int64_t kch = static_cast<std::int64_t>(cfg.num_keypoints) * 3;
  std::vector<KeypointPlane> planes(kpt.size());
  for (std::size_t s = 0; s < kpt.size(); ++s) {
    const std::int64_t plane =
        static_cast<std::int64_t>(grid_hw[s].first) * grid_hw[s].second;
    planes[s].data = kpt[s];
    planes[s].cell_step = cfg.channels_first ? 1 : kch;
    planes[s].chan_step = cfg.channels_first ? plane : 1;
  }
  return decodePose(cls, box, planes, grid_hw, cfg, lb);
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

/// Every resolver failure carries the model's whole output signature.
[[noreturn]] void failHead(const Engine& engine, const char* fn, const std::string& why) {
  throw Error(-1, std::string("RCDL ") + fn + ": " + why + ". Model outputs:" +
                      describeOutputs(engine));
}

[[noreturn]] void failPose(const Engine& engine, const std::string& why) {
  failHead(engine, "resolvePoseHead", why);
}

/// One output reduced to what head resolution cares about: how many channels it
/// has per cell, and how many cells it covers.
struct BranchInfo {
  int index = -1;
  int c = 0;                 ///< channels per cell (grid branches)
  std::int64_t cells = 0;    ///< cells covered (H*W for a grid branch)
  std::int64_t elems = 0;    ///< product of every dim — the only reading a fused
                             ///< keypoint tensor's axis split cannot invalidate
  int h = 0;                 ///< 0 for a branch that is not on a feature scale
  int w = 0;
  bool channels_first = true;
  bool is_grid = false;      ///< (H,W) is the model input divided by one common stride
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
std::vector<BranchInfo> readBranches(const Engine& engine, int input_h, int input_w,
                                     const char* fn) {
  std::vector<BranchInfo> outs;
  outs.reserve(static_cast<std::size_t>(engine.numOutputs()));
  for (int i = 0; i < engine.numOutputs(); ++i) {
    const rknn_tensor_attr& a = engine.outputAttr(i);
    if (a.n_dims < 2 || a.n_dims > 4) {
      failHead(engine, fn, "output " + std::to_string(i) + " is " + std::to_string(a.n_dims) +
                               "-D, a head branch is 2-D to 4-D");
    }
    if (a.dims[0] != 1) {
      failHead(engine, fn, "output " + std::to_string(i) + " has batch " +
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
      failHead(engine, fn,
               "output " + std::to_string(i) + " has a non-positive channel or cell count");
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

PoseHeadLayout resolvePoseHead(const Engine& engine, int num_classes, int num_keypoints) {
  if (num_classes <= 0) num_classes = 1;
  const std::pair<int, int> in_hw = modelInputHw(engine);
  if (in_hw.first <= 0 || in_hw.second <= 0) {
    failPose(engine, "input 0 is not a 4-D image tensor, cannot tell scale branches apart");
  }
  const std::vector<BranchInfo> outs =
      readBranches(engine, in_hw.first, in_hw.second, "resolvePoseHead");
  std::vector<std::vector<BranchInfo>> groups = groupByGrid(outs);
  if (groups.empty()) {
    failPose(engine, "no output has a grid that is the model input divided by a stride");
  }

  std::int64_t total_anchors = 0;
  for (const std::vector<BranchInfo>& g : groups) total_anchors += g.front().cells;

  // 1. The shared keypoint branch, if any: the one output that is not on a
  //    feature scale. It must cover every anchor and carry 3 channels per joint.
  PoseHeadLayout layout;
  layout.total_anchors = static_cast<int>(total_anchors);
  int shared_kpt_ch = 0;
  {
    std::vector<const BranchInfo*> loose;
    for (const BranchInfo& o : outs) {
      if (!o.is_grid) loose.push_back(&o);
    }
    if (loose.size() > 1) {
      failPose(engine, "more than one output sits outside the feature scales; "
                       "a pose head has at most one shared keypoint tensor");
    }
    if (loose.size() == 1) {
      // Read it through the ELEMENT COUNT, not the axis split. Real exports name
      // the same buffer [1,K*3,A] and [1,K,3,A] (and the runtime may report a
      // trailing 1), all of them joint-major with the anchor axis last; the only
      // invariant is K*3*A elements, so that is what we key on. The decoder then
      // walks it with chan_step = A, cell_step = 1, which is right for every one
      // of those spellings.
      const BranchInfo& k = *loose.front();
      if (total_anchors <= 0 || k.elems % total_anchors != 0) {
        failPose(engine, "keypoint output " + std::to_string(k.index) + " holds " +
                             std::to_string(k.elems) +
                             " elements, not a whole multiple of the " +
                             std::to_string(total_anchors) + " anchors the scales define");
      }
      const std::int64_t ch = k.elems / total_anchors;
      if (ch < 3 || ch % 3 != 0 || ch > 4096) {
        failPose(engine, "keypoint output " + std::to_string(k.index) + " works out to " +
                             std::to_string(ch) +
                             " channels per anchor, expected 3 per joint (x, y, visibility)");
      }
      layout.shared_kpt_index = k.index;
      shared_kpt_ch = static_cast<int>(ch);
      layout.num_keypoints = shared_kpt_ch / 3;
    }
  }

  // 2. Resolve each scale group into (cls, box) + optional per-scale keypoints.
  int anchor_offset = 0;
  for (const std::vector<BranchInfo>& g : groups) {
    const std::string where = "grid " + std::to_string(g.front().h) + "x" +
                              std::to_string(g.front().w);
    std::vector<BranchInfo> rest = g;

    // 2a. Pull out this scale's keypoint branch when there is no shared one.
    //     A keypoint branch has 3 channels per joint; a box branch has 4 or
    //     4*reg_max and a pose class branch has `num_classes`, so requiring
    //     >= 9 channels and != num_classes keeps the three apart.
    int kpt_index = -1;
    int kpt_channels = shared_kpt_ch;
    if (layout.shared_kpt_index < 0) {
      std::vector<std::size_t> cand;
      for (std::size_t i = 0; i < rest.size(); ++i) {
        const int c = rest[i].c;
        const bool ok = (num_keypoints > 0) ? (c == 3 * num_keypoints)
                                            : (c >= 9 && c % 3 == 0 && c != num_classes);
        if (ok) cand.push_back(i);
      }
      if (cand.size() != 1) {
        failPose(engine, where + ": " + (cand.empty() ? "no branch" : "more than one branch") +
                             " looks like the keypoint branch (3 channels per joint)" +
                             (num_keypoints > 0 ? "" : " — pass num_keypoints to disambiguate"));
      }
      kpt_index = rest[cand.front()].index;
      kpt_channels = rest[cand.front()].c;
      rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(cand.front()));
      if (layout.num_keypoints == 0) {
        layout.num_keypoints = kpt_channels / 3;
      } else if (layout.num_keypoints != kpt_channels / 3) {
        failPose(engine, "keypoint count differs across scales");
      }
    }

    // 2b. What remains is either one fused [4*reg_max+nc] tensor or a cls/box pair.
    PoseScaleOutputs sc;
    sc.grid_h = g.front().h;
    sc.grid_w = g.front().w;
    sc.kpt_index = kpt_index;
    sc.kpt_channels = kpt_channels;
    sc.anchor_offset = anchor_offset;
    sc.num_classes = num_classes;
    if (rest.size() == 1) {
      layout.fused_box_cls = true;
      const int box_ch = rest[0].c - num_classes;
      if (box_ch < 4 || box_ch % 4 != 0) {
        failPose(engine, where + ": a fused branch of " + std::to_string(rest[0].c) +
                             " channels minus " + std::to_string(num_classes) +
                             " classes leaves " + std::to_string(box_ch) +
                             " box channels, expected 4 or 4*reg_max");
      }
      sc.cls_index = rest[0].index;
      sc.box_index = rest[0].index;
      sc.box_channels = box_ch;
    } else if (rest.size() == 2) {
      // The box branch is 4 or 4*reg_max channels; the other is cls. A pose class
      // head is 1 channel, so the two never tie in practice, but prefer an exact
      // num_classes match when it is unambiguous.
      const BranchInfo& a = rest[0];
      const BranchInfo& b = rest[1];
      bool a_is_cls;
      if ((a.c == num_classes) != (b.c == num_classes)) {
        a_is_cls = a.c == num_classes;
      } else if ((a.c % 4 == 0 && a.c >= 4) != (b.c % 4 == 0 && b.c >= 4)) {
        a_is_cls = !(a.c % 4 == 0 && a.c >= 4);
      } else {
        failPose(engine, where + ": cannot tell the cls branch (" + std::to_string(a.c) +
                             " ch) from the box branch (" + std::to_string(b.c) + " ch)");
      }
      const BranchInfo& cls = a_is_cls ? a : b;
      const BranchInfo& box = a_is_cls ? b : a;
      if (box.c < 4 || box.c % 4 != 0) {
        failPose(engine, where + ": box branch has " + std::to_string(box.c) +
                             " channels, expected 4 (plain LTRB) or 4*reg_max");
      }
      sc.cls_index = cls.index;
      sc.box_index = box.index;
      sc.box_channels = box.c;
      sc.num_classes = cls.c;
    } else {
      failPose(engine, where + " has " + std::to_string(rest.size()) +
                           " tensors left after the keypoint branch; a pose scale is one "
                           "fused box+cls tensor or a cls/box pair");
    }
    layout.scales.push_back(sc);
    anchor_offset += static_cast<int>(g.front().cells);
  }

  // 3. Cross-scale consistency + the derived quantities the decoder needs.
  layout.channels_first = outs.front().channels_first;
  for (const BranchInfo& o : outs) {
    if (o.channels_first != layout.channels_first) {
      failPose(engine, "outputs mix NCHW and NHWC channel order");
    }
  }
  layout.num_classes = layout.scales.front().num_classes;
  const int box_channels = layout.scales.front().box_channels;
  for (const PoseScaleOutputs& sc : layout.scales) {
    if (sc.num_classes != layout.num_classes) {
      failPose(engine, "class count differs across scales");
    }
    if (sc.box_channels != box_channels) {
      failPose(engine, "box channel count differs across scales");
    }
  }
  // 64 box channels => reg_max 16 (the ultralytics DFL head); exactly 4 => plain.
  layout.reg_max = (box_channels != 4) ? box_channels / 4 : 0;
  layout.strides.reserve(layout.scales.size());
  for (const std::vector<BranchInfo>& g : groups) layout.strides.push_back(g.front().stride);
  if (layout.num_keypoints <= 0) {
    failPose(engine, "could not determine the keypoint count");
  }
  return layout;
}

std::string PoseHeadLayout::describe() const {
  std::ostringstream os;
  os << "YOLO pose head: " << scales.size() << " scale(s), " << num_classes << " class(es), "
     << num_keypoints << " keypoints, "
     << (reg_max > 0 ? "DFL reg_max=" + std::to_string(reg_max) : std::string("plain LTRB"))
     << ", " << (channels_first ? "NCHW" : "NHWC")
     << (fused_box_cls ? ", fused box+cls" : "");
  if (shared_kpt_index >= 0) {
    os << ", shared kpt=out[" << shared_kpt_index << "] over " << total_anchors << " anchors";
  }
  for (std::size_t i = 0; i < scales.size(); ++i) {
    const PoseScaleOutputs& sc = scales[i];
    os << "\n  scale " << i << ": grid " << sc.grid_h << "x" << sc.grid_w;
    if (i < strides.size()) os << " stride " << strides[i];
    os << "  cls=out[" << sc.cls_index << "](" << sc.num_classes << "ch)"
       << " box=out[" << sc.box_index << "](" << sc.box_channels << "ch)";
    if (sc.kpt_index >= 0) os << " kpt=out[" << sc.kpt_index << "](" << sc.kpt_channels << "ch)";
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// Engine-bound estimator
// ---------------------------------------------------------------------------

PoseEstimator::PoseEstimator(Engine& engine, PoseConfig cfg)
    : engine_(engine),
      cfg_(std::move(cfg)),
      layout_(resolvePoseHead(engine, cfg_.num_classes, 0)) {}

std::vector<PoseDetection> PoseEstimator::postprocess(const LetterboxInfo& lb) const {
  const std::size_t scales = layout_.scales.size();
  const int nc = layout_.num_classes;
  const int nk = layout_.num_keypoints;
  const int kch = nk * 3;

  const auto elems = [](const std::vector<int>& shape) {
    std::int64_t n = 1;
    for (int d : shape) n *= (d > 0 ? d : 1);
    return n;
  };

  // The scratch buffers below must outlive the decode, which reads through the
  // pointers taken from them.
  std::vector<std::vector<float>> main_buf(scales), box_buf(scales), kpt_buf(scales);
  std::vector<std::vector<float>> cls_scratch(scales);
  std::vector<const float*> cls_ptr(scales), box_ptr(scales);
  std::vector<KeypointPlane> kpt(scales);
  std::vector<std::pair<int, int>> grid_hw(scales);

  // The shared keypoint tensor is one buffer for the whole model, so read it
  // once and hand each scale a window into it.
  std::vector<float> shared_kpt_buf;
  const float* shared_kpt = nullptr;
  if (layout_.shared_kpt_index >= 0) {
    std::vector<int> shape;
    shared_kpt = outputAsFloat(engine_, layout_.shared_kpt_index, shared_kpt_buf, shape);
    if (elems(shape) < static_cast<std::int64_t>(layout_.total_anchors) * kch) {
      throw Error(-1, "RCDL PoseEstimator: keypoint output smaller than the resolved "
                      "layout:\n" + layout_.describe());
    }
  }

  for (std::size_t s = 0; s < scales; ++s) {
    const PoseScaleOutputs& sc = layout_.scales[s];
    const std::int64_t cells = static_cast<std::int64_t>(sc.grid_h) * sc.grid_w;
    grid_hw[s] = {sc.grid_h, sc.grid_w};

    std::vector<int> main_shape;
    const float* main = outputAsFloat(engine_, sc.cls_index, main_buf[s], main_shape);

    if (layout_.fused_box_cls) {
      if (elems(main_shape) < cells * (sc.box_channels + nc)) {
        throw Error(-1, "RCDL PoseEstimator: output smaller than the resolved layout:\n" +
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
        throw Error(-1, "RCDL PoseEstimator: output smaller than the resolved layout:\n" +
                            layout_.describe());
      }
    }

    if (shared_kpt != nullptr) {
      // [1, K*3, A]: consecutive cells are adjacent, consecutive channels are a
      // whole anchor plane apart.
      kpt[s].data = shared_kpt + sc.anchor_offset;
      kpt[s].cell_step = 1;
      kpt[s].chan_step = layout_.total_anchors;
    } else {
      std::vector<int> kpt_shape;
      const float* kp = outputAsFloat(engine_, sc.kpt_index, kpt_buf[s], kpt_shape);
      if (elems(kpt_shape) < cells * kch) {
        throw Error(-1, "RCDL PoseEstimator: keypoint output smaller than the resolved "
                        "layout:\n" + layout_.describe());
      }
      kpt[s].data = kp;
      kpt[s].cell_step = layout_.channels_first ? 1 : kch;
      kpt[s].chan_step = layout_.channels_first ? cells : 1;
    }
  }

  // Everything structural comes from the layout (read from the model), so a
  // mis-configured cfg cannot make the decoder index past a buffer; cfg supplies
  // only the thresholds and the activation conventions.
  PoseConfig eff = cfg_;
  eff.num_classes = nc;
  eff.num_keypoints = nk;
  eff.reg_max = layout_.reg_max;
  eff.channels_first = layout_.channels_first;
  if (!layout_.strides.empty()) eff.strides = layout_.strides;
  RCDL_REQUIRE(eff.strides.size() == scales,
               "RCDL PoseEstimator: stride count does not match the head's scale count");
  return decodePose(cls_ptr, box_ptr, kpt, grid_hw, eff, lb);
}

// ---------------------------------------------------------------------------
// COCO-17 names + skeleton
// ---------------------------------------------------------------------------

const std::vector<std::string>& cocoKeypointNames() {
  // The COCO person keypoint order every YOLO pose export uses. Left/right are
  // the SUBJECT's left/right, not the viewer's.
  static const std::vector<std::string> kNames = {
      "nose",          "left_eye",      "right_eye",  "left_ear",    "right_ear",
      "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
      "right_wrist",   "left_hip",      "right_hip",  "left_knee",   "right_knee",
      "left_ankle",    "right_ankle"};
  return kNames;
}

const char* cocoKeypointName(int keypoint_id) {
  const std::vector<std::string>& names = cocoKeypointNames();
  if (keypoint_id >= 0 && keypoint_id < static_cast<int>(names.size())) {
    return names[static_cast<std::size_t>(keypoint_id)].c_str();
  }
  // Out-of-range ids come from non-COCO layouts, so name them rather than throw.
  // A small rotating cache keeps the returned pointer valid for the caller.
  static thread_local std::string fallback[4];
  static thread_local int next = 0;
  std::string& slot = fallback[next];
  next = (next + 1) % 4;
  slot = "keypoint " + std::to_string(keypoint_id);
  return slot.c_str();
}

const std::vector<std::pair<int, int>>& cocoSkeleton() {
  // The bone list the upstream demos draw, converted from their 1-based indices
  // to the 0-based joint ids of PoseDetection::keypoints.
  static const std::vector<std::pair<int, int>> kEdges = {
      {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
      {5, 6},   {5, 7},   {6, 8},   {7, 9},   {8, 10},  {1, 2},  {0, 1},
      {0, 2},   {1, 3},   {2, 4},   {3, 5},   {4, 6}};
  return kEdges;
}

}  // namespace rcdl
