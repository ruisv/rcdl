#include "rcdl/tasks/detection.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

float iou(const Detection& a, const Detection& b) {
  const float ix1 = std::max(a.x1, b.x1);
  const float iy1 = std::max(a.y1, b.y1);
  const float ix2 = std::min(a.x2, b.x2);
  const float iy2 = std::min(a.y2, b.y2);
  const float iw = std::max(0.0f, ix2 - ix1);
  const float ih = std::max(0.0f, iy2 - iy1);
  const float inter = iw * ih;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float uni = area_a + area_b - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<int> nms(const std::vector<Detection>& dets, float iou_thresh, int max_dets) {
  std::vector<int> order(dets.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return dets[a].score > dets[b].score;
  });

  std::vector<int> keep;
  std::vector<char> suppressed(dets.size(), 0);
  for (int oi = 0; oi < static_cast<int>(order.size()); ++oi) {
    const int i = order[oi];
    if (suppressed[i]) continue;
    keep.push_back(i);
    if (max_dets > 0 && static_cast<int>(keep.size()) >= max_dets) break;
    for (int oj = oi + 1; oj < static_cast<int>(order.size()); ++oj) {
      const int j = order[oj];
      if (suppressed[j]) continue;
      if (dets[j].class_id != dets[i].class_id) continue;  // per-class
      if (iou(dets[i], dets[j]) > iou_thresh) suppressed[j] = 1;
    }
  }
  return keep;
}

// ---------------------------------------------------------------------------
// Fused single-tensor head
// ---------------------------------------------------------------------------

std::vector<Detection> decode(const float* data, const std::vector<int>& shape,
                              const DetectConfig& cfg, const LetterboxInfo& lb) {
  const int nc = cfg.num_classes;
  const bool has_obj = cfg.layout == DecodeLayout::kYoloV5;
  const int attrs = (has_obj ? 5 : 4) + nc;

  std::int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  // nc <= 0 would make the class loop read past the box attributes.
  if (data == nullptr || nc <= 0 || attrs <= 0 || total < attrs) return {};
  const std::int64_t N = total / attrs;

  // at(a, j): attribute `a` of candidate `j` in the contiguous logical buffer.
  // channels_first  => [.., attrs, N]: offset = a*N + j
  // channels_last   => [.., N, attrs]: offset = j*attrs + a
  const auto at = [&](int a, std::int64_t j) -> float {
    const std::int64_t off =
        cfg.channels_first ? (static_cast<std::int64_t>(a) * N + j) : (j * attrs + a);
    return data[off];
  };

  const int cls_start = has_obj ? 5 : 4;
  std::vector<Detection> dets;
  for (std::int64_t j = 0; j < N; ++j) {
    // argmax over classes (sigmoid is monotonic, so argmax is invariant to it).
    int best_k = 0;
    float best_raw = at(cls_start, j);
    for (int k = 1; k < nc; ++k) {
      const float v = at(cls_start + k, j);
      if (v > best_raw) {
        best_raw = v;
        best_k = k;
      }
    }
    const float cls_score = cfg.apply_sigmoid ? sigmoid(best_raw) : best_raw;

    float score;
    if (has_obj) {
      float obj = at(4, j);
      if (cfg.apply_sigmoid) obj = sigmoid(obj);
      score = obj * cls_score;
    } else {
      score = cls_score;
    }
    if (score < cfg.conf_thresh) continue;

    const float cx = at(0, j);
    const float cy = at(1, j);
    const float w = at(2, j);
    const float h = at(3, j);
    // model-input pixel box -> original-image pixels (un-letterbox) + clamp.
    const float mx1 = cx - w * 0.5f;
    const float my1 = cy - h * 0.5f;
    const float mx2 = cx + w * 0.5f;
    const float my2 = cy + h * 0.5f;

    Detection det;
    det.x1 = lb.clampX(lb.invX(mx1));
    det.y1 = lb.clampY(lb.invY(my1));
    det.x2 = lb.clampX(lb.invX(mx2));
    det.y2 = lb.clampY(lb.invY(my2));
    det.score = score;
    det.class_id = best_k;
    dets.push_back(det);
  }

  const std::vector<int> keep = nms(dets, cfg.iou_thresh, cfg.max_dets);
  std::vector<Detection> out;
  out.reserve(keep.size());
  for (int idx : keep) out.push_back(dets[idx]);
  return out;
}

Detector::Detector(Engine& engine, DetectConfig cfg, int output_index)
    : engine_(engine), cfg_(std::move(cfg)), out_idx_(output_index) {}

std::vector<Detection> Detector::postprocess(const LetterboxInfo& lb) const {
  if (out_idx_ < 0 || out_idx_ >= engine_.numOutputs()) {
    throw Error(-1, "RCDL Detector: output index out of range");
  }
  // Zero-copy for packed F32, dequant-into-scratch otherwise, so decode() stays
  // layout/precision agnostic while the common case avoids the per-element walk.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);
  return decode(data, shape, cfg_, lb);
}

// ---------------------------------------------------------------------------
// Anchor-free LTRB multi-scale head
// ---------------------------------------------------------------------------

std::vector<Detection> decodeYoloLtrb(const std::vector<const float*>& cls,
                                      const std::vector<const float*>& box,
                                      const std::vector<std::pair<int, int>>& grid_hw,
                                      const YoloLtrbConfig& cfg, const LetterboxInfo& lb,
                                      const std::vector<const float*>& score_sum) {
  const int nc = cfg.num_classes;
  const std::size_t scales = grid_hw.size();
  if (cls.size() != scales || box.size() != scales || cfg.strides.size() != scales) {
    throw Error(-1, "RCDL decodeYoloLtrb: cls/box/grid/strides length mismatch");
  }
  if (nc <= 0) throw Error(-1, "RCDL decodeYoloLtrb: num_classes must be > 0");
  // Box head: plain LTRB (reg==0, 4 ch/cell) vs DFL (reg>0, 4*reg ch/cell).
  const int reg = cfg.reg_max;
  if (reg < 0) throw Error(-1, "RCDL decodeYoloLtrb: reg_max must be >= 0");
  const int box_ch = (reg > 0) ? 4 * reg : 4;  // box channels per cell

  std::vector<Detection> dets;
  for (std::size_t s = 0; s < scales; ++s) {
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    const float stride = static_cast<float>(cfg.strides[s]);
    const float* cp = cls[s];
    const float* bp = box[s];
    if (cp == nullptr || bp == nullptr || H <= 0 || W <= 0) continue;
    // Optional pre-filter (see the header): sum >= max over non-negative class
    // scores, so `sum < conf_thresh` proves the cell is empty without touching
    // its class channels. One contiguous read replaces `nc` strided ones.
    const float* sump = (s < score_sum.size()) ? score_sum[s] : nullptr;

    // Channel-order abstraction, hoisted out of the cell loop so the inner loops
    // stay a base + k*step walk in both layouts:
    //   channels_first ([C,H,W]) : channel c of cell is at c*H*W + cell
    //   channels_last  ([H,W,C]) : channel c of cell is at cell*C + c
    const std::int64_t plane = static_cast<std::int64_t>(H) * W;
    const std::int64_t cls_step = cfg.channels_first ? plane : 1;
    const std::int64_t box_step = cfg.channels_first ? plane : 1;

    for (int gy = 0; gy < H; ++gy) {
      for (int gx = 0; gx < W; ++gx) {
        const std::int64_t cell = static_cast<std::int64_t>(gy) * W + gx;
        if (sump != nullptr && sump[cell] < cfg.conf_thresh) continue;
        const float* logits = cp + (cfg.channels_first ? cell : cell * nc);

        // argmax over classes: sigmoid is monotonic, so the winner is the same
        // on raw logits and we pay at most one exp() per surviving cell.
        int best_k = 0;
        float best_raw = logits[0];
        for (int k = 1; k < nc; ++k) {
          const float v = logits[static_cast<std::int64_t>(k) * cls_step];
          if (v > best_raw) {
            best_raw = v;
            best_k = k;
          }
        }
        // rknn_model_zoo exports keep the sigmoid inside the graph (cls is
        // already a probability); raw-logit exports set apply_sigmoid.
        const float score = cfg.apply_sigmoid ? sigmoid(best_raw) : best_raw;
        if (score < cfg.conf_thresh) continue;

        // LTRB distances about the cell center. Plain head reads 4 values; DFL
        // head reduces each side's `reg` raw logits via softmax-weighted sum
        // Σ b·softmax(b) (ultralytics DFL: 64 ch = 4 sides × 16 bins, side-major).
        const float* bb = bp + (cfg.channels_first ? cell : cell * box_ch);
        float d[4];
        if (reg > 0) {
          for (int side = 0; side < 4; ++side) {
            const std::int64_t base = static_cast<std::int64_t>(side) * reg * box_step;
            float maxv = bb[base];
            for (int b = 1; b < reg; ++b) {
              maxv = std::max(maxv, bb[base + static_cast<std::int64_t>(b) * box_step]);
            }
            float sum = 0.0f, acc = 0.0f;
            for (int b = 0; b < reg; ++b) {
              const float e = std::exp(bb[base + static_cast<std::int64_t>(b) * box_step] - maxv);
              sum += e;
              acc += e * static_cast<float>(b);
            }
            d[side] = (sum > 0.0f) ? acc / sum : 0.0f;
          }
        } else {
          for (int side = 0; side < 4; ++side) {
            d[side] = bb[static_cast<std::int64_t>(side) * box_step];
          }
        }

        const float cx = static_cast<float>(gx) + 0.5f;
        const float cy = static_cast<float>(gy) + 0.5f;
        const float mx1 = (cx - d[0]) * stride;
        const float my1 = (cy - d[1]) * stride;
        const float mx2 = (cx + d[2]) * stride;
        const float my2 = (cy + d[3]) * stride;

        Detection det;
        det.x1 = lb.clampX(lb.invX(mx1));
        det.y1 = lb.clampY(lb.invY(my1));
        det.x2 = lb.clampX(lb.invX(mx2));
        det.y2 = lb.clampY(lb.invY(my2));
        det.score = score;
        det.class_id = best_k;
        dets.push_back(det);
      }
    }
  }

  const std::vector<int> keep = nms(dets, cfg.iou_thresh, cfg.max_dets);
  std::vector<Detection> out;
  out.reserve(keep.size());
  for (int idx : keep) out.push_back(dets[idx]);
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

/// One output tensor reduced to what head resolution cares about.
struct OutputInfo {
  int index = -1;
  int c = 0;
  int h = 0;
  int w = 0;
  bool channels_first = true;
};

/// Every output as "[i] 'name' [d0,d1,...] fmt DTYPE" — appended to every error
/// this file throws, because "resolve failed" without the signature is useless
/// when a model turns out to have a head RCDL does not know.
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

[[noreturn]] void failLayout(const Engine& engine, const std::string& why) {
  throw Error(-1, "RCDL resolveYoloHead: " + why + ". Model outputs:" + describeOutputs(engine));
}

/// How plausible a channel count is as a box branch — lower is better, -1 means
/// "not a box". A DFL box head has 4*reg_max channels (64 for the ultralytics
/// reg_max=16 head); a plain LTRB head has exactly 4. Class counts that happen
/// to be multiples of 4 (80 for COCO) also pass the coarse test, so the ranking
/// prefers the two shapes real exports actually use before falling back to
/// "smaller multiple of 4 wins".
int boxRank(int c) noexcept {
  if (c == 4) return 0;
  if (c == 64) return 1;
  if (c > 4 && c <= 128 && c % 4 == 0) return 2;
  return -1;
}

}  // namespace

YoloHeadLayout resolveYoloHead(const Engine& engine, int num_classes) {
  const int n_out = engine.numOutputs();
  if (n_out < 2) {
    failLayout(engine, "an LTRB head needs at least 2 outputs, this model has " +
                           std::to_string(n_out));
  }

  // 1. Reduce every output to (C,H,W) + channel order. The rknn fmt says which
  //    axis is which: NHWC => [N,H,W,C], everything else (NCHW, and UNDEFINED,
  //    which the runtime reports for the plain NCHW logical layout) => [N,C,H,W].
  std::vector<OutputInfo> outs;
  outs.reserve(static_cast<std::size_t>(n_out));
  for (int i = 0; i < n_out; ++i) {
    const rknn_tensor_attr& a = engine.outputAttr(i);
    if (a.n_dims != 4) {
      failLayout(engine, "output " + std::to_string(i) + " is " +
                             std::to_string(a.n_dims) + "-D, an LTRB head branch is 4-D");
    }
    if (a.dims[0] != 1) {
      failLayout(engine, "output " + std::to_string(i) + " has batch " +
                             std::to_string(a.dims[0]) + ", only batch 1 is supported");
    }
    OutputInfo o;
    o.index = i;
    o.channels_first = a.fmt != RKNN_TENSOR_NHWC;
    if (o.channels_first) {
      o.c = static_cast<int>(a.dims[1]);
      o.h = static_cast<int>(a.dims[2]);
      o.w = static_cast<int>(a.dims[3]);
    } else {
      o.h = static_cast<int>(a.dims[1]);
      o.w = static_cast<int>(a.dims[2]);
      o.c = static_cast<int>(a.dims[3]);
    }
    if (o.c <= 0 || o.h <= 0 || o.w <= 0) {
      failLayout(engine, "output " + std::to_string(i) + " has a non-positive C/H/W");
    }
    outs.push_back(o);
  }

  // 2. Group by (H,W): one group per scale, first-seen order (fixed up by the
  //    sort in step 4). A model whose branches share a grid across scales would
  //    be indistinguishable here, but no YOLO export does that.
  std::vector<std::vector<OutputInfo>> groups;
  for (const OutputInfo& o : outs) {
    auto it = std::find_if(groups.begin(), groups.end(), [&](const std::vector<OutputInfo>& g) {
      return g.front().h == o.h && g.front().w == o.w;
    });
    if (it == groups.end()) {
      groups.push_back({o});
    } else {
      it->push_back(o);
    }
  }

  // 3. Resolve each group into cls / box / optional score-sum.
  std::vector<YoloScaleOutputs> scales;
  scales.reserve(groups.size());
  for (const std::vector<OutputInfo>& g : groups) {
    const std::string where = "grid " + std::to_string(g.front().h) + "x" +
                              std::to_string(g.front().w);
    if (g.size() < 2 || g.size() > 3) {
      failLayout(engine, where + " has " + std::to_string(g.size()) +
                             " tensors; an LTRB scale is cls+box (+ optional score-sum)");
    }

    std::vector<OutputInfo> rest = g;
    int sum_index = -1;
    if (rest.size() == 3) {
      // The third branch of the rknn_model_zoo export is a 1-channel per-cell
      // score sum used as a cheap pre-filter. RCDL ignores its values but has to
      // recognise it so it is not mistaken for a 1-class cls head.
      auto it = std::find_if(rest.begin(), rest.end(),
                             [](const OutputInfo& o) { return o.c == 1; });
      if (it == rest.end()) {
        failLayout(engine, where + " has 3 tensors but none is the 1-channel score-sum");
      }
      sum_index = it->index;
      rest.erase(it);
    }

    const OutputInfo& a = rest[0];
    const OutputInfo& b = rest[1];
    const OutputInfo* cls = nullptr;
    const OutputInfo* box = nullptr;
    if (num_classes > 0) {
      // Caller told us the class count: the branch that matches it is cls, full
      // stop — this is the only way to disambiguate e.g. a 4-class model whose
      // box head is also 4 channels.
      const bool a_is_cls = a.c == num_classes;
      const bool b_is_cls = b.c == num_classes;
      if (a_is_cls == b_is_cls) {
        failLayout(engine, where + ": " + (a_is_cls ? "both branches have" : "no branch has") +
                               " the requested class count " + std::to_string(num_classes));
      }
      cls = a_is_cls ? &a : &b;
      box = a_is_cls ? &b : &a;
    } else {
      const int ra = boxRank(a.c);
      const int rb = boxRank(b.c);
      if (ra < 0 && rb < 0) {
        failLayout(engine, where + ": neither branch (" + std::to_string(a.c) + " and " +
                               std::to_string(b.c) +
                               " channels) looks like a box head (4 or 4*reg_max <= 128)");
      }
      bool a_is_box;
      if (ra < 0 || rb < 0) {
        a_is_box = rb < 0;
      } else if (ra != rb) {
        a_is_box = ra < rb;
      } else if (a.c != b.c) {
        a_is_box = a.c < b.c;  // both plausible: the box head is the narrower one
      } else {
        failLayout(engine, where + ": both branches have " + std::to_string(a.c) +
                               " channels — pass num_classes to disambiguate");
      }
      box = a_is_box ? &a : &b;
      cls = a_is_box ? &b : &a;
    }

    if (box->c < 4 || box->c % 4 != 0) {
      failLayout(engine, where + ": box branch has " + std::to_string(box->c) +
                             " channels, expected 4 (plain LTRB) or 4*reg_max");
    }
    YoloScaleOutputs sc;
    sc.cls_index = cls->index;
    sc.box_index = box->index;
    sc.sum_index = sum_index;
    sc.grid_h = cls->h;
    sc.grid_w = cls->w;
    sc.num_classes = cls->c;
    sc.box_channels = box->c;
    scales.push_back(sc);

    // Both branches of a scale must agree on channel order — the decoder walks
    // them with one `channels_first` flag.
    if (cls->channels_first != box->channels_first) {
      failLayout(engine, where + ": cls and box branches disagree on channel order");
    }
  }

  YoloHeadLayout layout;
  layout.channels_first = outs.front().channels_first;
  layout.num_classes = scales.front().num_classes;
  const int box_channels = scales.front().box_channels;
  layout.has_score_sum = true;
  for (const YoloScaleOutputs& sc : scales) {
    if (sc.num_classes != layout.num_classes) {
      failLayout(engine, "class count differs across scales (" +
                             std::to_string(layout.num_classes) + " vs " +
                             std::to_string(sc.num_classes) + ")");
    }
    if (sc.box_channels != box_channels) {
      failLayout(engine, "box channel count differs across scales (" +
                             std::to_string(box_channels) + " vs " +
                             std::to_string(sc.box_channels) + ")");
    }
    if (sc.sum_index < 0) layout.has_score_sum = false;
  }
  for (const OutputInfo& o : outs) {
    if (o.channels_first != layout.channels_first) {
      failLayout(engine, "outputs mix NCHW and NHWC channel order");
    }
  }
  // 64 box channels => reg_max 16 (ultralytics DFL); exactly 4 => plain LTRB.
  layout.reg_max = (box_channels != 4) ? box_channels / 4 : 0;

  // 4. Stride-8 branch first: the largest grid is the finest scale.
  std::sort(scales.begin(), scales.end(),
            [](const YoloScaleOutputs& x, const YoloScaleOutputs& y) {
              const std::int64_t ax = static_cast<std::int64_t>(x.grid_h) * x.grid_w;
              const std::int64_t ay = static_cast<std::int64_t>(y.grid_h) * y.grid_w;
              if (ax != ay) return ax > ay;
              return x.grid_h > y.grid_h;
            });
  layout.scales = scales;

  // 5. Strides come from the model itself: stride = input_h / grid_h. Deriving
  //    them beats trusting a config, and an inexact division means the outputs
  //    are not the head we think they are, so it is an error rather than a
  //    rounded guess.
  if (engine.numInputs() < 1) failLayout(engine, "model has no inputs to derive strides from");
  const rknn_tensor_attr& in = engine.inputAttr(0);
  int input_h = 0;
  if (in.n_dims == 4) {
    input_h = static_cast<int>(in.fmt == RKNN_TENSOR_NHWC ? in.dims[1] : in.dims[2]);
  }
  if (input_h <= 0) {
    failLayout(engine, "input 0 is not a 4-D image tensor, cannot derive strides");
  }
  layout.strides.reserve(layout.scales.size());
  for (const YoloScaleOutputs& sc : layout.scales) {
    if (sc.grid_h <= 0 || input_h % sc.grid_h != 0) {
      failLayout(engine, "input height " + std::to_string(input_h) +
                             " is not an exact multiple of grid height " +
                             std::to_string(sc.grid_h));
    }
    layout.strides.push_back(input_h / sc.grid_h);
  }
  return layout;
}

std::string YoloHeadLayout::describe() const {
  std::ostringstream os;
  os << "YOLO LTRB head: " << scales.size() << " scale(s), " << num_classes << " classes, "
     << (reg_max > 0 ? "DFL reg_max=" + std::to_string(reg_max) : std::string("plain LTRB"))
     << ", " << (channels_first ? "NCHW" : "NHWC")
     << (has_score_sum ? ", score-sum branch" : "");
  for (std::size_t i = 0; i < scales.size(); ++i) {
    const YoloScaleOutputs& sc = scales[i];
    os << "\n  scale " << i << ": grid " << sc.grid_h << "x" << sc.grid_w;
    if (i < strides.size()) os << " stride " << strides[i];
    os << "  cls=out[" << sc.cls_index << "](" << sc.num_classes << "ch)"
       << " box=out[" << sc.box_index << "](" << sc.box_channels << "ch)";
    if (sc.sum_index >= 0) os << " sum=out[" << sc.sum_index << "]";
  }
  return os.str();
}

YoloLtrbDetector::YoloLtrbDetector(Engine& engine, YoloLtrbConfig cfg)
    : engine_(engine), cfg_(std::move(cfg)), layout_(resolveYoloHead(engine, cfg_.num_classes)) {}

std::vector<Detection> YoloLtrbDetector::postprocess(const LetterboxInfo& lb) const {
  const std::size_t scales = layout_.scales.size();

  // View each (cls, box) pair as row-major floats — zero-copy for packed F32,
  // dequant-into-scratch for the usual int8-affine output. The scratch buffers
  // must outlive the decode below, which reads through the pointers.
  std::vector<std::vector<float>> cls_buf(scales), box_buf(scales), sum_buf(scales);
  std::vector<const float*> cls_ptr(scales), box_ptr(scales), sum_ptr(scales, nullptr);
  std::vector<std::pair<int, int>> grid_hw(scales);

  for (std::size_t s = 0; s < scales; ++s) {
    const YoloScaleOutputs& sc = layout_.scales[s];
    std::vector<int> cls_shape, box_shape;
    cls_ptr[s] = outputAsFloat(engine_, sc.cls_index, cls_buf[s], cls_shape);
    box_ptr[s] = outputAsFloat(engine_, sc.box_index, box_buf[s], box_shape);
    grid_hw[s] = {sc.grid_h, sc.grid_w};
    // The score-sum branch, when the export has one: a per-cell upper bound on
    // the class maximum that lets the decoder skip most cells without reading
    // their class channels at all. Measured on YOLOv8n at 640, this is the
    // difference between ~30 ms and ~3 ms of post-processing per frame.
    if (sc.sum_index >= 0) {
      std::vector<int> sum_shape;
      const float* sp = outputAsFloat(engine_, sc.sum_index, sum_buf[s], sum_shape);
      std::int64_t n = 1;
      for (int d : sum_shape) n *= (d > 0 ? d : 1);
      if (n >= static_cast<std::int64_t>(sc.grid_h) * sc.grid_w) sum_ptr[s] = sp;
    }

    // Belt and braces: the decoder indexes up to (channels-1)*H*W, so make sure
    // the buffers really hold what the resolved layout promised.
    const auto elems = [](const std::vector<int>& shape) {
      std::int64_t n = 1;
      for (int d : shape) n *= (d > 0 ? d : 1);
      return n;
    };
    const std::int64_t cells = static_cast<std::int64_t>(sc.grid_h) * sc.grid_w;
    if (elems(cls_shape) < cells * sc.num_classes || elems(box_shape) < cells * sc.box_channels) {
      throw Error(-1, "RCDL YoloLtrbDetector: output smaller than the resolved head layout:\n" +
                          layout_.describe());
    }
  }

  // Everything structural comes from the LAYOUT (read from the model), so a
  // mis-configured cfg cannot make the decoder index past a buffer; cfg only
  // supplies the thresholds and the class-activation convention.
  YoloLtrbConfig eff = cfg_;
  eff.num_classes = layout_.num_classes;
  eff.reg_max = layout_.reg_max;
  eff.channels_first = layout_.channels_first;
  if (!layout_.strides.empty()) eff.strides = layout_.strides;
  if (eff.strides.size() != scales) {
    throw Error(-1, "RCDL YoloLtrbDetector: stride count does not match the head's scale count");
  }
  return decodeYoloLtrb(cls_ptr, box_ptr, grid_hw, eff, lb, sum_ptr);
}

// ---------------------------------------------------------------------------
// COCO class names
// ---------------------------------------------------------------------------

const std::vector<std::string>& cocoClassNames() {
  // Ultralytics' COCO-80 order and spelling — the labels every YOLO export's
  // class indices refer to. Function-local static: built once, never copied.
  static const std::vector<std::string> kNames = {
      "person",        "bicycle",      "car",           "motorcycle",    "airplane",
      "bus",           "train",        "truck",         "boat",          "traffic light",
      "fire hydrant",  "stop sign",    "parking meter", "bench",         "bird",
      "cat",           "dog",          "horse",         "sheep",         "cow",
      "elephant",      "bear",         "zebra",         "giraffe",       "backpack",
      "umbrella",      "handbag",      "tie",           "suitcase",      "frisbee",
      "skis",          "snowboard",    "sports ball",   "kite",          "baseball bat",
      "baseball glove", "skateboard",  "surfboard",     "tennis racket", "bottle",
      "wine glass",    "cup",          "fork",          "knife",         "spoon",
      "bowl",          "banana",       "apple",         "sandwich",      "orange",
      "broccoli",      "carrot",       "hot dog",       "pizza",         "donut",
      "cake",          "chair",        "couch",         "potted plant",  "bed",
      "dining table",  "toilet",       "tv",            "laptop",        "mouse",
      "remote",        "keyboard",     "cell phone",    "microwave",     "oven",
      "toaster",       "sink",         "refrigerator",  "book",          "clock",
      "vase",          "scissors",     "teddy bear",    "hair drier",    "toothbrush"};
  return kNames;
}

const char* cocoClassName(int class_id) {
  const std::vector<std::string>& names = cocoClassNames();
  if (class_id >= 0 && class_id < static_cast<int>(names.size())) {
    return names[static_cast<std::size_t>(class_id)].c_str();
  }
  // Out-of-range ids come from non-COCO models, so name them rather than throw.
  // A small rotating cache keeps the returned pointer valid for the caller.
  static thread_local std::string fallback[4];
  static thread_local int next = 0;
  std::string& slot = fallback[next];
  next = (next + 1) % 4;
  slot = "class " + std::to_string(class_id);
  return slot.c_str();
}

}  // namespace rcdl
