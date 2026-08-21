#include "rcdl/tasks/panoptic_drive.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

namespace {

/// Model-input height, or 0 when there is nothing to read it from. Used to
/// check the CONFIGURED strides against the grids the model reports.
int inputHeight(const Engine& engine) {
  if (engine.numInputs() < 1) return 0;
  const rknn_tensor_attr& in = engine.inputAttr(0);
  if (in.n_dims != 4) return 0;
  return static_cast<int>(in.fmt == RKNN_TENSOR_NHWC ? in.dims[1] : in.dims[2]);
}

}  // namespace

std::vector<Detection> decodeYoloV5Anchor(const std::vector<const float*>& raw,
                                          const std::vector<std::pair<int, int>>& grid_hw,
                                          const AnchorDetectConfig& cfg, const LetterboxInfo& lb) {
  const std::size_t ns = raw.size();
  if (ns == 0 || grid_hw.size() != ns || cfg.strides.size() != ns || cfg.anchors.size() != ns) {
    throw Error(-1, "RCDL anchor detect: raw/grid/strides/anchors length mismatch");
  }
  const int nc = cfg.num_classes;
  if (nc <= 0) throw Error(-1, "RCDL anchor detect: num_classes must be positive");

  const std::size_t na = cfg.anchors[0].size();
  if (na == 0) throw Error(-1, "RCDL anchor detect: no anchors given");
  for (const auto& a : cfg.anchors) {
    if (a.size() != na) {
      throw Error(-1, "RCDL anchor detect: every scale must declare the same anchor count");
    }
  }

  const int no = 5 + nc;  // cx,cy,w,h,obj + classes
  std::vector<Detection> cands;

  for (std::size_t s = 0; s < ns; ++s) {
    const float* p = raw[s];
    if (p == nullptr) continue;
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    if (H <= 0 || W <= 0) continue;
    const float stride = static_cast<float>(cfg.strides[s]);
    const std::size_t plane = static_cast<std::size_t>(H) * static_cast<std::size_t>(W);

    for (std::size_t a = 0; a < na; ++a) {
      // Channels-first: attribute `k` of anchor `a` is the plane at channel
      // (a*no + k), so one grid cell's attributes are `plane` elements apart.
      const std::size_t base = (a * static_cast<std::size_t>(no)) * plane;
      const float aw = cfg.anchors[s][a].w;
      const float ah = cfg.anchors[s][a].h;

      for (int gy = 0; gy < H; ++gy) {
        for (int gx = 0; gx < W; ++gx) {
          const std::size_t off = base + static_cast<std::size_t>(gy) * static_cast<std::size_t>(W) +
                                  static_cast<std::size_t>(gx);

          // Objectness gates everything else: check it before paying for the
          // class scan, which is where most of the candidates die.
          const float obj = sigmoid(p[off + 4 * plane]);
          if (obj < cfg.conf_thresh) continue;

          int best_c = 0;
          float best_s = -1.0f;
          for (int c = 0; c < nc; ++c) {
            const float v = sigmoid(p[off + static_cast<std::size_t>(5 + c) * plane]);
            if (v > best_s) {
              best_s = v;
              best_c = c;
            }
          }
          const float score = obj * best_s;
          if (score < cfg.conf_thresh) continue;

          const float tx = sigmoid(p[off + 0 * plane]);
          const float ty = sigmoid(p[off + 1 * plane]);
          const float tw = sigmoid(p[off + 2 * plane]);
          const float th = sigmoid(p[off + 3 * plane]);

          const float cx = (tx * 2.0f - 0.5f + static_cast<float>(gx)) * stride;
          const float cy = (ty * 2.0f - 0.5f + static_cast<float>(gy)) * stride;
          const float bw = (tw * 2.0f) * (tw * 2.0f) * aw;
          const float bh = (th * 2.0f) * (th * 2.0f) * ah;

          Detection d;
          d.x1 = lb.clampX(lb.invX(cx - bw * 0.5f));
          d.y1 = lb.clampY(lb.invY(cy - bh * 0.5f));
          d.x2 = lb.clampX(lb.invX(cx + bw * 0.5f));
          d.y2 = lb.clampY(lb.invY(cy + bh * 0.5f));
          d.score = score;
          d.class_id = best_c;
          cands.push_back(d);
        }
      }
    }
  }

  const std::vector<int> keep = nms(cands, cfg.iou_thresh, cfg.max_dets);
  std::vector<Detection> out;
  out.reserve(keep.size());
  for (int i : keep) out.push_back(cands[static_cast<std::size_t>(i)]);
  return out;
}

AnchorDetector::AnchorDetector(Engine& engine, AnchorDetectConfig cfg, int output_base)
    : engine_(engine), cfg_(std::move(cfg)), out_base_(output_base) {
  if (cfg_.strides.empty() || cfg_.anchors.size() != cfg_.strides.size()) {
    throw Error(-1, "RCDL anchor detect: " + std::to_string(cfg_.strides.size()) +
                        " strides against " + std::to_string(cfg_.anchors.size()) +
                        " anchor sets — one prior set per scale is required");
  }
  for (const auto& a : cfg_.anchors) {
    if (a.empty()) throw Error(-1, "RCDL anchor detect: a scale declares no priors");
    // Checked here rather than left to the decoder: postprocess() takes `na`
    // from scale 0 and checks every scale's channel count against it, so a
    // ragged set would surface as a channel-count complaint blaming
    // num_classes instead of the priors.
    if (a.size() != cfg_.anchors.front().size()) {
      throw Error(-1, "RCDL anchor detect: every scale must declare the same anchor count");
    }
  }
  const int need = out_base_ + static_cast<int>(cfg_.strides.size());
  if (out_base_ < 0 || need > engine_.numOutputs()) {
    throw Error(-1, "RCDL anchor detect: output range [" + std::to_string(out_base_) + "," +
                        std::to_string(need) + ") exceeds the model's " +
                        std::to_string(engine_.numOutputs()) + " outputs");
  }
}

std::vector<Detection> AnchorDetector::postprocess(const LetterboxInfo& lb) const {
  const std::size_t ns = cfg_.strides.size();
  const std::size_t na = cfg_.anchors.empty() ? 0 : cfg_.anchors[0].size();
  const int no = 5 + cfg_.num_classes;

  // Scratch buffers must outlive the pointers handed to the decoder.
  std::vector<std::vector<float>> scratch(ns);
  std::vector<const float*> raw(ns, nullptr);
  std::vector<std::pair<int, int>> grid(ns, {0, 0});

  for (std::size_t s = 0; s < ns; ++s) {
    std::vector<int> shape;
    raw[s] = outputAsFloat(engine_, out_base_ + static_cast<int>(s), scratch[s], shape);

    // Expect [1, na*(5+nc), H, W] (channels-first head convolution output).
    const int idx = out_base_ + static_cast<int>(s);
    if (shape.size() < 3) {
      throw Error(-1, "RCDL anchor detect: output " + std::to_string(idx) +
                          " has too few dims for an anchor head");
    }
    // Unlike the LTRB head this decoder walks channel planes directly, so an
    // NHWC export is not the same bytes in a different order to it — say so
    // rather than let the channel check below blame num_classes.
    if (engine_.outputAttr(idx).fmt == RKNN_TENSOR_NHWC) {
      throw Error(-1, "RCDL anchor detect: output " + std::to_string(idx) +
                          " is NHWC; this decoder reads channels-first raw head tensors");
    }
    const std::size_t d = shape.size();
    const int C = shape[d - 3];
    grid[s] = {shape[d - 2], shape[d - 1]};

    // The strides are configured, but the grids come from the model, and a
    // model that enumerates its heads P5-first would pair them the wrong way
    // round: every box at the wrong position and 4x the wrong size, silently.
    // Cross-check them the way resolveYoloHead derives strides in the first
    // place. (0 means the input is not a 4-D image tensor — nothing to check
    // against, e.g. a model driven entirely by the caller.)
    if (const int in_h = inputHeight(engine_); in_h > 0) {
      if (grid[s].first <= 0 || grid[s].first * cfg_.strides[s] != in_h) {
        throw Error(-1, "RCDL anchor detect: output " + std::to_string(idx) + " has a " +
                            std::to_string(grid[s].first) + "-row grid, which stride " +
                            std::to_string(cfg_.strides[s]) + " does not divide the " +
                            std::to_string(in_h) +
                            "-pixel input into — the strides are in a different order "
                            "than the outputs");
      }
    }

    const int expect = static_cast<int>(na) * no;
    if (C != expect) {
      throw Error(-1, "RCDL anchor detect: output " + std::to_string(idx) + " has " +
                          std::to_string(C) + " channels, expected " +
                          std::to_string(expect) + " (= " + std::to_string(na) + " anchors x (5 + " +
                          std::to_string(cfg_.num_classes) +
                          " classes)) — check num_classes/anchors");
    }
  }
  return decodeYoloV5Anchor(raw, grid, cfg_, lb);
}

}  // namespace rcdl
