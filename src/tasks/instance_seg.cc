#include "rcdl/tasks/instance_seg.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"
#include "rcdl/tasks/detection.h"  // Detection + nms(): the box half of this head

namespace rcdl {

namespace {

int clampi(int v, int lo, int hi) noexcept { return v < lo ? lo : (v > hi ? hi : v); }

/// ceil(v) as an int, clamped into [lo,hi]. NaN lands on `lo`, and the clamp
/// happens in float so a wild value cannot overflow the cast.
int ceilClamped(float v, int lo, int hi) noexcept {
  const float c = std::ceil(v);
  if (!(c > static_cast<float>(lo))) return lo;  // false for NaN too
  if (c >= static_cast<float>(hi)) return hi;
  return static_cast<int>(c);
}

/// One axis of a bilinear resize: the two source indices and the blend weight
/// for one destination index. Precomputed per axis because every instance of a
/// frame shares the same resize geometry.
struct Lerp {
  int i0 = 0;
  int i1 = 0;
  float w = 0.0f;  ///< value = src[i0]*(1-w) + src[i1]*w
};

/// Pixel-CENTRE mapping, dst index -> src coordinate, clamped to the source:
///     f = (d + 0.5) * src_n / dst_n - 0.5
/// This is cv2.resize(INTER_LINEAR) and torch interpolate(align_corners=False),
/// which is what the reference post-process uses for both mask resizes. Getting
/// this wrong (e.g. the align_corners=True map d*(src_n-1)/(dst_n-1)) shifts
/// every mask by half a destination pixel at each of the two resizes.
std::vector<Lerp> axisLerp(int dst_n, int src_n) {
  std::vector<Lerp> out(static_cast<std::size_t>(std::max(0, dst_n)));
  if (dst_n <= 0 || src_n <= 0) return out;
  const float ratio = static_cast<float>(src_n) / static_cast<float>(dst_n);
  const float hi = static_cast<float>(src_n - 1);
  for (int d = 0; d < dst_n; ++d) {
    float f = (static_cast<float>(d) + 0.5f) * ratio - 0.5f;
    if (f < 0.0f) f = 0.0f;
    if (f > hi) f = hi;
    Lerp l;
    l.i0 = static_cast<int>(f);
    l.i1 = std::min(l.i0 + 1, src_n - 1);
    l.w = f - static_cast<float>(l.i0);
    out[static_cast<std::size_t>(d)] = l;
  }
  return out;
}

/// Turns one instance's mask coefficients into a binary mask in source pixels.
///
/// The chain is the reference one, with the two intermediate images fused where
/// that is exact rather than approximate:
///
///   1. mask = sigmoid(coef · proto)                        at the prototype grid
///   2. bilinear upsample to the model-input canvas         (dstH x dstW)
///   3. zero everything outside the instance's box          ("crop_mask")
///   4. drop the letterbox padding: keep the canvas window the scaled source
///      image actually occupies                             (nh x nw)
///   5. bilinear resize that window to the source frame     (srcH x srcW)
///   6. threshold
///
/// Steps 2 and 3 are fused: since 3 zeroes everything outside the box, step 2 is
/// only ever evaluated inside it. Steps 4 and 5 are fused too — the de-pad is a
/// sub-rectangle, so it is an index offset on the step-5 reads rather than a
/// copy. Nothing else is folded: bilinear-of-bilinear is NOT one bilinear, so
/// the two resizes stay two resizes and the result matches the reference
/// numerically, not just approximately.
///
/// Buffers and both axis mappings are built once and reused for every instance
/// of the frame; the canvas is cleared by rubbing out the PREVIOUS instance's
/// box, the only region that can be non-zero.
class MaskAssembler {
 public:
  MaskAssembler(const float* proto, int proto_h, int proto_w, int proto_c,
                const InstanceSegConfig& cfg, const LetterboxInfo& lb)
      : proto_(proto),
        ph_(proto_h),
        pw_(proto_w),
        pc_(proto_c),
        cfg_(cfg),
        lb_(lb) {
    RCDL_REQUIRE(proto_ != nullptr && ph_ > 0 && pw_ > 0 && pc_ > 0,
                 "RCDL instance seg: prototype tensor is missing or empty");
    RCDL_REQUIRE(lb_.dstW > 0 && lb_.dstH > 0 && lb_.srcW > 0 && lb_.srcH > 0,
                 "RCDL instance seg: letterbox has a non-positive extent");
    RCDL_REQUIRE(lb_.scale > 0.0f, "RCDL instance seg: letterbox scale must be > 0");

    // The scaled source image occupies [padX, padX + srcW*scale) x
    // [padY, padY + srcH*scale) of the canvas. The preprocessor wrote integer
    // rectangles (RGA blits to whole pixels and reflects the rounding back into
    // LetterboxInfo), so round the same way and clamp so the window fits inside
    // the canvas — this window is what the mask must be cut down to BEFORE it is
    // resized to the source frame.
    nw_ = clampi(static_cast<int>(std::lround(static_cast<float>(lb_.srcW) * lb_.scale)), 1,
                 lb_.dstW);
    nh_ = clampi(static_cast<int>(std::lround(static_cast<float>(lb_.srcH) * lb_.scale)), 1,
                 lb_.dstH);
    px_ = clampi(static_cast<int>(std::lround(lb_.padX)), 0, lb_.dstW - nw_);
    py_ = clampi(static_cast<int>(std::lround(lb_.padY)), 0, lb_.dstH - nh_);

    canvas_x_ = axisLerp(lb_.dstW, pw_);  // canvas <- prototype
    canvas_y_ = axisLerp(lb_.dstH, ph_);
    out_x_ = axisLerp(lb_.srcW, nw_);  // source frame <- de-padded window
    out_y_ = axisLerp(lb_.srcH, nh_);

    pm_.assign(static_cast<std::size_t>(ph_) * static_cast<std::size_t>(pw_), 0.0f);
    canvas_.assign(static_cast<std::size_t>(lb_.dstH) * static_cast<std::size_t>(lb_.dstW), 0.0f);
  }

  /// `box` is in MODEL-INPUT pixels (the crop happens on the canvas, before the
  /// un-letterbox). Fills out.mask / mask_x0 / mask_y0 / mask_w / mask_h.
  void assemble(const float* coef, const Detection& box, InstanceMask& out) {
    const int dstW = lb_.dstW;
    const int dstH = lb_.dstH;

    // 1. Prototype combination, then the sigmoid — in that order, at the
    //    prototype grid. sigmoid does not commute with interpolation, so
    //    activating after the upsample would give a different mask.
    const std::size_t plane = static_cast<std::size_t>(ph_) * static_cast<std::size_t>(pw_);
    for (std::size_t p = 0; p < plane; ++p) {
      float acc = 0.0f;
      if (cfg_.proto_channels_first) {
        const float* q = proto_ + p;  // [C,H,W]: channel c is one plane away
        for (int c = 0; c < pc_; ++c) acc += coef[c] * q[static_cast<std::size_t>(c) * plane];
      } else {
        const float* q = proto_ + p * static_cast<std::size_t>(pc_);  // [H,W,C]
        for (int c = 0; c < pc_; ++c) acc += coef[c] * q[c];
      }
      pm_[p] = sigmoid(acc);
    }

    // 2+3. Upsample to the canvas, but only inside the box: crop_mask keeps
    //      integer canvas pixel k iff k >= m1 && k < m2, i.e. k in
    //      [ceil(m1), ceil(m2)). Everything else stays exactly 0.
    const int cx0 = ceilClamped(box.x1, 0, dstW);
    const int cx1 = ceilClamped(box.x2, 0, dstW);
    const int cy0 = ceilClamped(box.y1, 0, dstH);
    const int cy1 = ceilClamped(box.y2, 0, dstH);

    clearPrevious();
    for (int y = cy0; y < cy1; ++y) {
      const Lerp& ly = canvas_y_[static_cast<std::size_t>(y)];
      const float* r0 = pm_.data() + static_cast<std::size_t>(ly.i0) * static_cast<std::size_t>(pw_);
      const float* r1 = pm_.data() + static_cast<std::size_t>(ly.i1) * static_cast<std::size_t>(pw_);
      float* dst = canvas_.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(dstW);
      for (int x = cx0; x < cx1; ++x) {
        const Lerp& lx = canvas_x_[static_cast<std::size_t>(x)];
        const float top = r0[lx.i0] * (1.0f - lx.w) + r0[lx.i1] * lx.w;
        const float bot = r1[lx.i0] * (1.0f - lx.w) + r1[lx.i1] * lx.w;
        dst[x] = top * (1.0f - ly.w) + bot * ly.w;
      }
    }
    prev_x0_ = cx0;
    prev_x1_ = cx1;
    prev_y0_ = cy0;
    prev_y1_ = cy1;

    // 4+5+6. Resize the de-padded canvas window to the source frame and
    //        threshold. `out_*_` map source pixels into that window, so px_/py_
    //        is the only trace the padding leaves — no intermediate copy.
    int wx0 = 0, wy0 = 0, wx1 = lb_.srcW, wy1 = lb_.srcH;
    if (!cfg_.full_frame_masks) {
      // Only the instance's own box can be non-zero, so the caller can ask for
      // just that rectangle. Grow it to whole pixels so nothing is cut off.
      wx0 = ceilClamped(std::floor(out.x1), 0, lb_.srcW);
      wy0 = ceilClamped(std::floor(out.y1), 0, lb_.srcH);
      wx1 = ceilClamped(out.x2, 0, lb_.srcW);
      wy1 = ceilClamped(out.y2, 0, lb_.srcH);
      if (wx1 < wx0) wx1 = wx0;
      if (wy1 < wy0) wy1 = wy0;
    }
    const int mw = wx1 - wx0;
    const int mh = wy1 - wy0;
    out.mask_x0 = wx0;
    out.mask_y0 = wy0;
    out.mask_w = mw;
    out.mask_h = mh;
    out.mask.assign(static_cast<std::size_t>(std::max(0, mw)) *
                        static_cast<std::size_t>(std::max(0, mh)),
                    static_cast<std::uint8_t>(0));
    if (mw <= 0 || mh <= 0) return;

    for (int dy = wy0; dy < wy1; ++dy) {
      const Lerp& ly = out_y_[static_cast<std::size_t>(dy)];
      const int r0 = py_ + ly.i0;
      const int r1 = py_ + ly.i1;
      // Both taps outside the box => both canvas values are 0 => the whole row
      // is 0 and is already zeroed. Pure speed; correctness is the zeroed canvas.
      if (r1 < cy0 || r0 >= cy1) continue;
      const float* c0 = canvas_.data() + static_cast<std::size_t>(r0) * static_cast<std::size_t>(dstW);
      const float* c1 = canvas_.data() + static_cast<std::size_t>(r1) * static_cast<std::size_t>(dstW);
      std::uint8_t* mrow = out.mask.data() + static_cast<std::size_t>(dy - wy0) *
                                                 static_cast<std::size_t>(mw);
      for (int dx = wx0; dx < wx1; ++dx) {
        const Lerp& lx = out_x_[static_cast<std::size_t>(dx)];
        const int a0 = px_ + lx.i0;
        const int a1 = px_ + lx.i1;
        if (a1 < cx0 || a0 >= cx1) continue;
        const float top = c0[a0] * (1.0f - lx.w) + c0[a1] * lx.w;
        const float bot = c1[a0] * (1.0f - lx.w) + c1[a1] * lx.w;
        const float v = top * (1.0f - ly.w) + bot * ly.w;
        if (v > cfg_.mask_thresh) mrow[dx - wx0] = 1;
      }
    }
  }

 private:
  void clearPrevious() {
    for (int y = prev_y0_; y < prev_y1_; ++y) {
      float* row = canvas_.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(lb_.dstW);
      std::fill(row + prev_x0_, row + prev_x1_, 0.0f);
    }
    prev_x0_ = prev_x1_ = prev_y0_ = prev_y1_ = 0;
  }

  const float* proto_;
  int ph_, pw_, pc_;
  InstanceSegConfig cfg_;
  LetterboxInfo lb_;
  int nw_ = 0, nh_ = 0, px_ = 0, py_ = 0;
  std::vector<Lerp> canvas_x_, canvas_y_, out_x_, out_y_;
  std::vector<float> pm_, canvas_;
  int prev_x0_ = 0, prev_x1_ = 0, prev_y0_ = 0, prev_y1_ = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

std::vector<InstanceMask> decodeInstanceSeg(const std::vector<const float*>& cls,
                                            const std::vector<const float*>& box,
                                            const std::vector<const float*>& mc,
                                            const std::vector<std::pair<int, int>>& grid_hw,
                                            int num_classes, const float* proto, int proto_h,
                                            int proto_w, int proto_c,
                                            const InstanceSegConfig& cfg,
                                            const LetterboxInfo& lb) {
  const std::size_t scales = grid_hw.size();
  if (cls.size() != scales || box.size() != scales || mc.size() != scales ||
      cfg.strides.size() != scales) {
    throw Error(-1, "RCDL decodeInstanceSeg: cls/box/mc/grid/strides length mismatch");
  }
  if (num_classes <= 0) throw Error(-1, "RCDL decodeInstanceSeg: num_classes must be > 0");
  const int reg = cfg.reg_max;
  if (reg < 0) throw Error(-1, "RCDL decodeInstanceSeg: reg_max must be >= 0");
  const int box_ch = (reg > 0) ? 4 * reg : 4;

  // Mask coefficients only have meaning against a prototype of the same width,
  // so require the pair up front instead of discovering it mid-matmul.
  const bool masks = cfg.compute_masks;
  if (masks && (proto == nullptr || proto_h <= 0 || proto_w <= 0 || proto_c <= 0)) {
    throw Error(-1, "RCDL decodeInstanceSeg: mask assembly needs a non-empty prototype");
  }
  const int num_coef = masks ? proto_c : 0;

  // Candidates carry their box in MODEL-INPUT pixels: the mask crop happens on
  // the model canvas, and NMS is unaffected because the letterbox map is a
  // uniform scale + translation (IoU is invariant under it).
  std::vector<Detection> cands;
  std::vector<std::vector<float>> coefs;

  for (std::size_t s = 0; s < scales; ++s) {
    const int H = grid_hw[s].first;
    const int W = grid_hw[s].second;
    const float* cp = cls[s];
    const float* bp = box[s];
    const float* mp = mc[s];
    if (cp == nullptr || bp == nullptr || H <= 0 || W <= 0) continue;
    if (masks && mp == nullptr) {
      throw Error(-1, "RCDL decodeInstanceSeg: mask coefficients missing for a scale");
    }
    const float stride = static_cast<float>(cfg.strides[s]);

    // Channel-order abstraction hoisted out of the cell loop, as in the
    // detector: channels_first ([C,H,W]) => channel c is one plane away;
    // channels_last ([H,W,C]) => channel c is one element away.
    const std::int64_t plane = static_cast<std::int64_t>(H) * W;
    const std::int64_t step = cfg.channels_first ? plane : 1;

    for (int gy = 0; gy < H; ++gy) {
      for (int gx = 0; gx < W; ++gx) {
        const std::int64_t cell = static_cast<std::int64_t>(gy) * W + gx;
        const float* logits = cp + (cfg.channels_first ? cell : cell * num_classes);

        // argmax first, activation second: sigmoid is monotonic, so the winner
        // is the same on raw logits and at most one exp() is paid per survivor.
        int best_k = 0;
        float best_raw = logits[0];
        for (int k = 1; k < num_classes; ++k) {
          const float v = logits[static_cast<std::int64_t>(k) * step];
          if (v > best_raw) {
            best_raw = v;
            best_k = k;
          }
        }
        const float score = cfg.apply_sigmoid ? sigmoid(best_raw) : best_raw;
        if (score < cfg.conf_thresh) continue;

        // LTRB about the cell centre; DFL reduces each side's `reg` logits by
        // the softmax-weighted expectation Σ b·softmax(b) (side-major bins).
        const float* bb = bp + (cfg.channels_first ? cell : cell * box_ch);
        float d[4];
        if (reg > 0) {
          for (int side = 0; side < 4; ++side) {
            const std::int64_t base = static_cast<std::int64_t>(side) * reg * step;
            float maxv = bb[base];
            for (int b = 1; b < reg; ++b) {
              maxv = std::max(maxv, bb[base + static_cast<std::int64_t>(b) * step]);
            }
            float sum = 0.0f, acc = 0.0f;
            for (int b = 0; b < reg; ++b) {
              const float e = std::exp(bb[base + static_cast<std::int64_t>(b) * step] - maxv);
              sum += e;
              acc += e * static_cast<float>(b);
            }
            d[side] = (sum > 0.0f) ? acc / sum : 0.0f;
          }
        } else {
          for (int side = 0; side < 4; ++side) d[side] = bb[static_cast<std::int64_t>(side) * step];
        }

        const float cx = static_cast<float>(gx) + 0.5f;
        const float cy = static_cast<float>(gy) + 0.5f;
        Detection det;
        det.x1 = (cx - d[0]) * stride;
        det.y1 = (cy - d[1]) * stride;
        det.x2 = (cx + d[2]) * stride;
        det.y2 = (cy + d[3]) * stride;
        det.score = score;
        det.class_id = best_k;
        cands.push_back(det);

        if (masks) {
          // Mask coefficients are LINEAR — no activation here, the sigmoid comes
          // after the prototype matmul.
          const float* mcp = mp + (cfg.channels_first ? cell : cell * num_coef);
          std::vector<float> c(static_cast<std::size_t>(num_coef));
          for (int k = 0; k < num_coef; ++k) {
            c[static_cast<std::size_t>(k)] = mcp[static_cast<std::int64_t>(k) * step];
          }
          coefs.push_back(std::move(c));
        }
      }
    }
  }

  const std::vector<int> keep = nms(cands, cfg.iou_thresh, cfg.max_dets);

  std::vector<InstanceMask> out;
  out.reserve(keep.size());
  // The assembler owns the per-frame scratch, so build it once for all kept
  // instances (and not at all when masks are off / nothing survived).
  std::unique_ptr<MaskAssembler> assembler;
  if (masks && !keep.empty()) {
    assembler.reset(new MaskAssembler(proto, proto_h, proto_w, proto_c, cfg, lb));
  }
  for (int idx : keep) {
    const Detection& d = cands[static_cast<std::size_t>(idx)];
    InstanceMask im;
    im.x1 = lb.clampX(lb.invX(d.x1));
    im.y1 = lb.clampY(lb.invY(d.y1));
    im.x2 = lb.clampX(lb.invX(d.x2));
    im.y2 = lb.clampY(lb.invY(d.y2));
    im.score = d.score;
    im.class_id = d.class_id;
    im.mask_x0 = 0;
    im.mask_y0 = 0;
    im.mask_w = 0;
    im.mask_h = 0;
    if (assembler) assembler->assemble(coefs[static_cast<std::size_t>(idx)].data(), d, im);
    out.push_back(std::move(im));
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

struct OutInfo {
  int index = -1;
  int c = 0;
  int h = 0;
  int w = 0;
  bool channels_first = true;
};

/// Every output as "[i] 'name' [dims] fmt DTYPE" — appended to every error this
/// file throws, because "resolve failed" without the signature is useless when a
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

[[noreturn]] void failSeg(const Engine& engine, const std::string& why) {
  throw Error(-1,
              "RCDL resolveInstanceSegHead: " + why + ". Model outputs:" + describeOutputs(engine));
}

}  // namespace

InstanceSegHeadLayout resolveInstanceSegHead(const Engine& engine, int num_classes) {
  const int n_out = engine.numOutputs();
  // Smallest possible seg head: one scale (cls+box+mc) plus the prototype.
  if (n_out < 4) {
    failSeg(engine, "an instance-seg head needs at least 4 outputs, this model has " +
                        std::to_string(n_out));
  }

  // 1. Reduce every output to (C,H,W) + channel order. The rknn fmt says which
  //    axis is which: NHWC => [N,H,W,C], everything else (NCHW, and UNDEFINED,
  //    which the runtime reports for the plain NCHW logical layout) => [N,C,H,W].
  //    This is the ONLY reliable discriminator — a [1,32,160,160] prototype and
  //    a [1,160,160,32] one hold the same numbers in a different order.
  std::vector<OutInfo> outs;
  outs.reserve(static_cast<std::size_t>(n_out));
  for (int i = 0; i < n_out; ++i) {
    const rknn_tensor_attr& a = engine.outputAttr(i);
    if (a.n_dims != 4) {
      failSeg(engine, "output " + std::to_string(i) + " is " + std::to_string(a.n_dims) +
                          "-D, every branch of a seg head is 4-D");
    }
    if (a.dims[0] != 1) {
      failSeg(engine, "output " + std::to_string(i) + " has batch " + std::to_string(a.dims[0]) +
                          ", only batch 1 is supported");
    }
    OutInfo o;
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
      failSeg(engine, "output " + std::to_string(i) + " has a non-positive C/H/W");
    }
    outs.push_back(o);
  }

  // 2. Group by (H,W). Detection branches always come in groups of 3 or 4; the
  //    prototype is alone on its own (much larger) grid.
  std::vector<std::vector<OutInfo>> groups;
  for (const OutInfo& o : outs) {
    auto it = std::find_if(groups.begin(), groups.end(), [&](const std::vector<OutInfo>& g) {
      return g.front().h == o.h && g.front().w == o.w;
    });
    if (it == groups.end()) {
      groups.push_back({o});
    } else {
      it->push_back(o);
    }
  }

  const OutInfo* proto = nullptr;
  std::vector<std::vector<OutInfo>> scale_groups;
  for (const std::vector<OutInfo>& g : groups) {
    if (g.size() == 1) {
      if (proto != nullptr) {
        failSeg(engine, "two outputs sit alone on their own grid — the prototype is ambiguous");
      }
      proto = &g.front();
    } else {
      scale_groups.push_back(g);
    }
  }
  if (proto == nullptr) failSeg(engine, "no output stands alone on its own grid (no prototype)");
  if (scale_groups.empty()) failSeg(engine, "no detection branches found alongside the prototype");

  InstanceSegHeadLayout layout;
  layout.proto_index = proto->index;
  layout.proto_h = proto->h;
  layout.proto_w = proto->w;
  layout.num_coef = proto->c;
  layout.proto_channels_first = proto->channels_first;

  // 3. Resolve each scale group into cls / box / mc (/ optional score-sum).
  std::vector<InstanceSegScaleOutputs> scales;
  scales.reserve(scale_groups.size());
  for (const std::vector<OutInfo>& g : scale_groups) {
    const std::string where = "grid " + std::to_string(g.front().h) + "x" +
                              std::to_string(g.front().w);
    if (g.size() < 3 || g.size() > 4) {
      failSeg(engine, where + " has " + std::to_string(g.size()) +
                          " tensors; a seg scale is cls+box+mc (+ optional score-sum)");
    }

    std::vector<OutInfo> rest = g;
    int sum_index = -1;
    if (rest.size() == 4) {
      // The extra branch is the 1-channel per-cell score sum the export adds as
      // a cheap pre-filter. RCDL ignores its values but has to recognise it so
      // it is not mistaken for a 1-class cls head.
      auto it = std::find_if(rest.begin(), rest.end(), [](const OutInfo& o) { return o.c == 1; });
      if (it == rest.end()) {
        failSeg(engine, where + " has 4 tensors but none is the 1-channel score-sum");
      }
      sum_index = it->index;
      rest.erase(it);
    }

    // mc is the branch whose channel count matches the prototype planes — the
    // matmul only type-checks for that one. When the class count collides with
    // it, the later output wins: the export emits ... cls ... mc within a group.
    auto mc_it = rest.end();
    for (auto it = rest.begin(); it != rest.end(); ++it) {
      if (it->c != layout.num_coef) continue;
      // A tie (class count == coefficient count, e.g. a 32-class model on a
      // 32-plane prototype) goes to the LATER output: within one grid group the
      // export emits the mask-coefficient branch after the class branch.
      if (mc_it == rest.end() || it->index > mc_it->index) mc_it = it;
    }
    if (mc_it == rest.end()) {
      failSeg(engine, where + ": no branch has the prototype's " +
                          std::to_string(layout.num_coef) + " channels (no mask-coef branch)");
    }
    const OutInfo mc = *mc_it;
    rest.erase(mc_it);

    // Of the two left, the box head is 4 or 4*reg_max channels. num_classes,
    // when given, settles it directly (a 4-class model whose box head is also
    // 4 channels is otherwise ambiguous).
    const OutInfo& a = rest[0];
    const OutInfo& b = rest[1];
    const OutInfo* cls = nullptr;
    const OutInfo* bx = nullptr;
    if (num_classes > 0) {
      const bool a_is_cls = a.c == num_classes;
      const bool b_is_cls = b.c == num_classes;
      if (a_is_cls == b_is_cls) {
        failSeg(engine, where + ": " + (a_is_cls ? "both branches have" : "no branch has") +
                            " the requested class count " + std::to_string(num_classes));
      }
      cls = a_is_cls ? &a : &b;
      bx = a_is_cls ? &b : &a;
    } else {
      const auto box_rank = [](int c) {
        if (c == 4) return 0;    // plain LTRB
        if (c == 64) return 1;   // the ultralytics reg_max=16 DFL head
        if (c > 4 && c <= 128 && c % 4 == 0) return 2;
        return -1;
      };
      const int ra = box_rank(a.c);
      const int rb = box_rank(b.c);
      if (ra < 0 && rb < 0) {
        failSeg(engine, where + ": neither remaining branch (" + std::to_string(a.c) + " and " +
                            std::to_string(b.c) + " channels) looks like a box head");
      }
      bool a_is_box;
      if (ra < 0 || rb < 0) {
        a_is_box = rb < 0;
      } else if (ra != rb) {
        a_is_box = ra < rb;
      } else if (a.c != b.c) {
        a_is_box = a.c < b.c;  // both plausible: the box head is the narrower one
      } else {
        failSeg(engine, where + ": both remaining branches have " + std::to_string(a.c) +
                            " channels — pass num_classes to disambiguate");
      }
      bx = a_is_box ? &a : &b;
      cls = a_is_box ? &b : &a;
    }

    if (bx->c < 4 || bx->c % 4 != 0) {
      failSeg(engine, where + ": box branch has " + std::to_string(bx->c) +
                          " channels, expected 4 (plain LTRB) or 4*reg_max");
    }
    if (cls->channels_first != bx->channels_first || cls->channels_first != mc.channels_first) {
      failSeg(engine, where + ": the branches of one scale disagree on channel order");
    }

    InstanceSegScaleOutputs sc;
    sc.cls_index = cls->index;
    sc.box_index = bx->index;
    sc.mc_index = mc.index;
    sc.sum_index = sum_index;
    sc.grid_h = cls->h;
    sc.grid_w = cls->w;
    sc.num_classes = cls->c;
    sc.num_coef = mc.c;
    sc.box_channels = bx->c;
    scales.push_back(sc);
  }

  // outs[] is built in output order, so an output index doubles as its slot.
  layout.channels_first = outs[static_cast<std::size_t>(scales.front().cls_index)].channels_first;
  layout.num_classes = scales.front().num_classes;
  const int box_channels = scales.front().box_channels;
  layout.has_score_sum = true;
  for (const InstanceSegScaleOutputs& sc : scales) {
    if (sc.num_classes != layout.num_classes) {
      failSeg(engine, "class count differs across scales (" + std::to_string(layout.num_classes) +
                          " vs " + std::to_string(sc.num_classes) + ")");
    }
    if (sc.num_coef != layout.num_coef) {
      failSeg(engine, "mask-coefficient count differs from the prototype's " +
                          std::to_string(layout.num_coef) + " planes");
    }
    if (sc.box_channels != box_channels) {
      failSeg(engine, "box channel count differs across scales (" + std::to_string(box_channels) +
                          " vs " + std::to_string(sc.box_channels) + ")");
    }
    if (sc.sum_index < 0) layout.has_score_sum = false;
    if (outs[static_cast<std::size_t>(sc.cls_index)].channels_first != layout.channels_first) {
      failSeg(engine, "the per-scale branches mix NCHW and NHWC channel order");
    }
  }
  // 64 box channels => reg_max 16 (ultralytics DFL); exactly 4 => plain LTRB.
  layout.reg_max = (box_channels != 4) ? box_channels / 4 : 0;

  // 4. Stride-8 branch first: the largest grid is the finest scale.
  std::sort(scales.begin(), scales.end(),
            [](const InstanceSegScaleOutputs& x, const InstanceSegScaleOutputs& y) {
              const std::int64_t ax = static_cast<std::int64_t>(x.grid_h) * x.grid_w;
              const std::int64_t ay = static_cast<std::int64_t>(y.grid_h) * y.grid_w;
              if (ax != ay) return ax > ay;
              return x.grid_h > y.grid_h;
            });
  layout.scales = scales;

  // 5. Strides come from the model: stride = input_h / grid_h. An inexact
  //    division means these outputs are not the head we think they are, so it
  //    is an error rather than a rounded guess.
  if (engine.numInputs() < 1) failSeg(engine, "model has no inputs to derive strides from");
  const rknn_tensor_attr& in = engine.inputAttr(0);
  int input_h = 0;
  if (in.n_dims == 4) {
    input_h = static_cast<int>(in.fmt == RKNN_TENSOR_NHWC ? in.dims[1] : in.dims[2]);
  }
  if (input_h <= 0) failSeg(engine, "input 0 is not a 4-D image tensor, cannot derive strides");
  layout.strides.reserve(layout.scales.size());
  for (const InstanceSegScaleOutputs& sc : layout.scales) {
    if (sc.grid_h <= 0 || input_h % sc.grid_h != 0) {
      failSeg(engine, "input height " + std::to_string(input_h) +
                          " is not an exact multiple of grid height " + std::to_string(sc.grid_h));
    }
    layout.strides.push_back(input_h / sc.grid_h);
  }
  return layout;
}

std::string InstanceSegHeadLayout::describe() const {
  std::ostringstream os;
  os << "YOLO instance-seg head: " << scales.size() << " scale(s), " << num_classes
     << " classes, " << num_coef << " mask coefs, "
     << (reg_max > 0 ? "DFL reg_max=" + std::to_string(reg_max) : std::string("plain LTRB"))
     << ", branches " << (channels_first ? "NCHW" : "NHWC")
     << (has_score_sum ? ", score-sum branch" : "");
  os << "\n  proto: out[" << proto_index << "] " << proto_h << "x" << proto_w << "x" << num_coef
     << " (" << (proto_channels_first ? "NCHW" : "NHWC") << ")";
  for (std::size_t i = 0; i < scales.size(); ++i) {
    const InstanceSegScaleOutputs& sc = scales[i];
    os << "\n  scale " << i << ": grid " << sc.grid_h << "x" << sc.grid_w;
    if (i < strides.size()) os << " stride " << strides[i];
    os << "  cls=out[" << sc.cls_index << "](" << sc.num_classes << "ch)"
       << " box=out[" << sc.box_index << "](" << sc.box_channels << "ch)"
       << " mc=out[" << sc.mc_index << "](" << sc.num_coef << "ch)";
    if (sc.sum_index >= 0) os << " sum=out[" << sc.sum_index << "]";
  }
  return os.str();
}

// ---------------------------------------------------------------------------
// Engine-bound segmenter
// ---------------------------------------------------------------------------

InstanceSegmenter::InstanceSegmenter(Engine& engine, InstanceSegConfig cfg)
    : engine_(engine),
      cfg_(std::move(cfg)),
      layout_(resolveInstanceSegHead(engine, cfg_.num_classes)) {}

std::vector<InstanceMask> InstanceSegmenter::postprocess(const LetterboxInfo& lb) const {
  const std::size_t scales = layout_.scales.size();

  // Zero-copy for packed F32, dequant-into-scratch for the usual int8-affine
  // output. The scratch buffers must outlive the decode below.
  std::vector<std::vector<float>> cls_buf(scales), box_buf(scales), mc_buf(scales);
  std::vector<const float*> cls_ptr(scales, nullptr), box_ptr(scales, nullptr),
      mc_ptr(scales, nullptr);
  std::vector<std::pair<int, int>> grid_hw(scales);

  const auto elems = [](const std::vector<int>& shape) {
    std::int64_t n = 1;
    for (int d : shape) n *= (d > 0 ? d : 1);
    return n;
  };

  for (std::size_t s = 0; s < scales; ++s) {
    const InstanceSegScaleOutputs& sc = layout_.scales[s];
    std::vector<int> cls_shape, box_shape, mc_shape;
    cls_ptr[s] = outputAsFloat(engine_, sc.cls_index, cls_buf[s], cls_shape);
    box_ptr[s] = outputAsFloat(engine_, sc.box_index, box_buf[s], box_shape);
    mc_ptr[s] = outputAsFloat(engine_, sc.mc_index, mc_buf[s], mc_shape);
    grid_hw[s] = {sc.grid_h, sc.grid_w};

    // Belt and braces: the decoder indexes up to (channels-1)*H*W, so check the
    // buffers really hold what the resolved layout promised.
    const std::int64_t cells = static_cast<std::int64_t>(sc.grid_h) * sc.grid_w;
    if (elems(cls_shape) < cells * sc.num_classes || elems(box_shape) < cells * sc.box_channels ||
        elems(mc_shape) < cells * sc.num_coef) {
      throw Error(-1, "RCDL InstanceSegmenter: an output is smaller than the resolved layout:\n" +
                          layout_.describe());
    }
  }

  std::vector<float> proto_buf;
  std::vector<int> proto_shape;
  const float* proto = outputAsFloat(engine_, layout_.proto_index, proto_buf, proto_shape);
  const std::int64_t proto_need = static_cast<std::int64_t>(layout_.proto_h) * layout_.proto_w *
                                  layout_.num_coef;
  if (elems(proto_shape) < proto_need) {
    throw Error(-1, "RCDL InstanceSegmenter: prototype smaller than the resolved layout:\n" +
                        layout_.describe());
  }

  // Everything structural comes from the LAYOUT (read from the model), so a
  // mis-configured cfg cannot make the decoder index past a buffer; cfg only
  // supplies thresholds, the class-activation convention and the mask options.
  InstanceSegConfig eff = cfg_;
  eff.num_classes = layout_.num_classes;
  eff.reg_max = layout_.reg_max;
  eff.channels_first = layout_.channels_first;
  eff.proto_channels_first = layout_.proto_channels_first;
  if (!layout_.strides.empty()) eff.strides = layout_.strides;
  if (eff.strides.size() != scales) {
    throw Error(-1, "RCDL InstanceSegmenter: stride count does not match the head's scale count");
  }

  return decodeInstanceSeg(cls_ptr, box_ptr, mc_ptr, grid_hw, layout_.num_classes, proto,
                           layout_.proto_h, layout_.proto_w, layout_.num_coef, eff, lb);
}

}  // namespace rcdl
