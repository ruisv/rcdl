#include "rcdl/tasks/classification.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"
#include "rcdl/preproc/rga.h"

namespace rcdl {

namespace {

/// "[1,1,1,1000]" — every error in this file names the shape it choked on,
/// because "not a classifier head" without the shape is useless when a model
/// turns out to have an output RCDL did not expect.
std::string describeShape(const std::vector<int>& shape) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i) os << ",";
    os << shape[i];
  }
  os << "]";
  return os.str();
}

/// Crop `box` out of `src` and squash-resize it into `dst`, whatever the two
/// formats are.
///
/// Two routes, and the choice is about WHO can express the offset:
///   - a source with a dma-buf fd goes through RGA's own crop rectangle, so a
///     VPU frame is cropped, scaled and colour-converted in one improcess with
///     no CPU touch and no intermediate buffer;
///   - a host-only source is re-based instead: an ImageView is only a
///     descriptor, so pointing `data` at the box's first pixel and shrinking
///     width/height (the row stride stays the SOURCE's) describes the crop
///     exactly, and the ordinary backend-agnostic resize() takes it from there.
/// The sub-view trick needs a packed format and a CPU pointer; a planar-YUV
/// host frame has a second plane at its own offset, so it goes to RGA too.
LetterboxInfo cropResizeInto(const ImageView& dst, const ImageView& src, CropBox box,
                             PreprocBackend backend, YuvRange range, PreprocBackend* used) {
  // Whole-frame box: no cropping to express, so this is a plain resize and
  // every backend/format combination the preproc layer supports works.
  if (box.x == 0 && box.y == 0 && box.w == src.width && box.h == src.height) {
    return resize(dst, src, backend, range, used);
  }

  const int bpp = bytesPerPixel(src.format);
  const bool can_subview = src.data != nullptr && bpp > 0;
  if (src.fd >= 0 && backend != PreprocBackend::Cpu && rgaAvailable()) {
    // RGA reads chroma at half resolution, so an odd crop origin on a planar
    // format would sample the wrong chroma pair (and the driver rejects it
    // outright). Round the origin down and the extent down to even — a
    // one-pixel shift of a centre crop, below the resampling error either way.
    if (bpp <= 0) {
      box.x &= ~1;
      box.y &= ~1;
      box.w &= ~1;
      box.h &= ~1;
    }
    RCDL_REQUIRE(box.w > 0 && box.h > 0, "RCDL cropResize: empty crop rectangle");
    LetterboxInfo lb;
    lb.srcW = box.w;
    lb.srcH = box.h;
    lb.dstW = dst.width;
    lb.dstH = dst.height;
    lb.scale = static_cast<float>(dst.width) / box.w;
    if (!can_subview) {
      // No CPU-side fallback exists for this source, so let RGA's own error
      // (out-of-range scale, unaligned stride, ...) reach the caller.
      rgaCropResize(dst, src, box.x, box.y, box.w, box.h, range);
      if (used) *used = PreprocBackend::Rga;
      return lb;
    }
    try {
      rgaCropResize(dst, src, box.x, box.y, box.w, box.h, range);
      if (used) *used = PreprocBackend::Rga;
      return lb;
    } catch (const Error&) {
      // RGA refused this pair — a crop below its 68x2 minimum or a scale
      // outside [1/16,16] (a 4K frame down to 224 is a 17x reduction). The
      // sub-view path below does the same work on the CPU; a throw here is the
      // hardware's "not this one", not an error to propagate.
    }
  }

  RCDL_REQUIRE(can_subview,
               ("RCDL cropResize: cropping " + src.describe() +
                " needs either a dma-buf fd with RGA available or a CPU-mapped packed "
                "format; convert a planar YUV host frame first")
                   .c_str());
  ImageView sub = src;
  sub.fd = -1;  // the fd addresses the WHOLE buffer; the offset lives in `data`
  sub.data = src.bytePtr() + static_cast<std::size_t>(box.y) * src.rowBytes() +
             static_cast<std::size_t>(box.x) * static_cast<std::size_t>(bpp);
  sub.width = box.w;
  sub.height = box.h;
  sub.wstride = src.effWStride();  // rows still stride by the SOURCE's pitch
  sub.hstride = box.h;
  sub.size = 0;  // recomputed from the sub-extent by ImageView::bytes()
  return resize(dst, sub, backend, range, used);
}

}  // namespace

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

int classCountFromShape(const std::vector<int>& shape) {
  // The class axis is whichever one is not 1; the rest are batch / degenerate
  // spatial dims that the export happened to keep. Two non-unit dims mean this
  // is not a score vector at all.
  int count = 1;
  int non_unit = 0;
  for (int d : shape) {
    const int v = d > 0 ? d : 1;
    if (v > 1) {
      ++non_unit;
      count = v;
    }
  }
  if (non_unit > 1) {
    throw Error(-1, "RCDL classification: output shape " + describeShape(shape) +
                        " has more than one non-unit dimension, so it is not a "
                        "single-label score vector");
  }
  return count;
}

std::vector<ClsResult> decodeClassification(const float* logits, int num_classes,
                                            const ClsConfig& cfg) {
  if (logits == nullptr || num_classes <= 0) return {};

  // Build (class_id, score) for every class: a softmax probability when asked
  // for, else the value as it came out of the model.
  std::vector<ClsResult> all(static_cast<std::size_t>(num_classes));
  if (cfg.apply_softmax) {
    // Max-subtraction: exp() of a dequantized int8 logit can reach e^30 and a
    // float head's can be far larger, so subtracting the max is what keeps the
    // sum finite. It cancels exactly in the ratio, so the result is unchanged.
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; ++i) {
      if (logits[i] > max_logit) max_logit = logits[i];
    }
    double sum = 0.0;  // accumulate in double; 1000 exp() terms add up
    for (int i = 0; i < num_classes; ++i) {
      const float e = std::exp(logits[i] - max_logit);
      all[static_cast<std::size_t>(i)].class_id = i;
      all[static_cast<std::size_t>(i)].score = e;  // unnormalised; divided below
      sum += e;
    }
    const float inv = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
    for (int i = 0; i < num_classes; ++i) all[static_cast<std::size_t>(i)].score *= inv;
  } else {
    for (int i = 0; i < num_classes; ++i) {
      all[static_cast<std::size_t>(i)].class_id = i;
      all[static_cast<std::size_t>(i)].score = logits[i];
    }
  }

  // How many to keep: all classes when top_k is unset (<=0) or oversized.
  int k = cfg.top_k;
  if (k <= 0 || k > num_classes) k = num_classes;

  // Ties are common on an int8 head (one quantization step apart is one step,
  // and equal scores happen), so order them by class id to keep the output
  // reproducible instead of leaving it to the sort's internals.
  const auto by_score_desc = [](const ClsResult& a, const ClsResult& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.class_id < b.class_id;
  };

  // Partial sort: only the top-k prefix is fully ordered, which is much cheaper
  // than a full sort when k << num_classes (top-5 of 1000).
  if (k < num_classes) {
    std::partial_sort(all.begin(), all.begin() + k, all.end(), by_score_desc);
    all.resize(static_cast<std::size_t>(k));
  } else {
    std::sort(all.begin(), all.end(), by_score_desc);
  }
  return all;
}

std::vector<ClsResult> decodeClassification(const float* logits,
                                            const std::vector<int>& shape,
                                            const ClsConfig& cfg) {
  return decodeClassification(logits, classCountFromShape(shape), cfg);
}

bool looksLikeProbabilities(const float* data, int n, float tol) {
  if (data == nullptr || n <= 0) return false;
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    const float v = data[i];
    if (!(v >= 0.0f) || v > 1.0f + tol) return false;  // !(>=) also catches NaN
    sum += v;
  }
  return std::fabs(sum - 1.0) <= static_cast<double>(tol);
}

// ---------------------------------------------------------------------------
// Preprocessing geometry
// ---------------------------------------------------------------------------

CropBox centerCropBox(int src_w, int src_h, int out_w, int out_h, float crop_ratio) {
  CropBox box;
  if (src_w <= 0 || src_h <= 0 || out_w <= 0 || out_h <= 0) return box;
  if (!(crop_ratio > 0.0f)) crop_ratio = 0.875f;  // also catches NaN
  if (crop_ratio > 1.0f) crop_ratio = 1.0f;       // >1 would crop outside the image

  // torchvision's eval transform is Resize(shorter side -> S) + CenterCrop(out),
  // with S = out / crop_ratio (224 / 0.875 = 256). Expressed in SOURCE pixels
  // the crop is out/scale wide, where scale = S / min(src_w, src_h) is the
  // resize factor — i.e. the crop grows as the margin does.
  const float shorter = static_cast<float>(std::min(src_w, src_h));
  const float scale_w = static_cast<float>(out_w) / crop_ratio / shorter;
  const float scale_h = static_cast<float>(out_h) / crop_ratio / shorter;
  // One resize factor for both axes (the resize is aspect-preserving); the
  // larger of the two keeps the crop inside the image for a non-square output.
  const float scale = std::max(scale_w, scale_h);

  int w = static_cast<int>(std::lround(out_w / scale));
  int h = static_cast<int>(std::lround(out_h / scale));
  // A source smaller than the crop, or a very lopsided aspect, can push one
  // side past the image; shrink BOTH sides together so the crop keeps the
  // output's aspect ratio rather than squashing differently per image.
  const float over = std::max(static_cast<float>(w) / src_w, static_cast<float>(h) / src_h);
  if (over > 1.0f) {
    w = static_cast<int>(w / over);
    h = static_cast<int>(h / over);
  }
  box.w = std::max(1, std::min(w, src_w));
  box.h = std::max(1, std::min(h, src_h));
  box.x = (src_w - box.w) / 2;
  box.y = (src_h - box.h) / 2;
  return box;
}

// ---------------------------------------------------------------------------
// Classifier
// ---------------------------------------------------------------------------

Classifier::Classifier(Engine& engine, ClsConfig cfg, ClsPreproc pre, int output_index)
    : engine_(engine), cfg_(cfg), pre_(pre), out_idx_(output_index) {
  RCDL_REQUIRE(out_idx_ >= 0 && out_idx_ < engine_.numOutputs(),
               "RCDL classification: output index out of range");
  // Resolve the class count ONCE, from the model. A model whose selected output
  // is a feature map rather than a score vector fails here, not per frame.
  num_classes_ = classCountFromShape(engine_.outputShape(out_idx_));

  // The input canvas, from the tensor's declared layout rather than guessed
  // from dim magnitudes: a 3-class NCHW input is [1,3,H,W] and a 3-channel NHWC
  // input is [1,H,W,3], which shape alone cannot tell apart.
  if (engine_.numInputs() > 0) {
    const rknn_tensor_attr& a = engine_.inputAttr(0);
    if (a.n_dims == 4) {
      const bool nchw = a.fmt == RKNN_TENSOR_NCHW;
      input_h_ = static_cast<int>(nchw ? a.dims[2] : a.dims[1]);
      input_w_ = static_cast<int>(nchw ? a.dims[3] : a.dims[2]);
    }
  }
}

std::vector<ClsResult> Classifier::postprocess() const {
  // Zero-copy for packed f32, dequant-into-scratch for the usual int8-affine
  // output. `scratch` must outlive `data`, so it stays in this scope.
  std::vector<float> scratch;
  std::vector<int> shape;
  const float* data = outputAsFloat(engine_, out_idx_, scratch, shape);

  // Belt and braces: the decoder reads num_classes_ floats, so make sure the
  // buffer really holds them (a model reloaded under the same Engine, or an
  // output whose runtime shape disagrees with the one seen at construction).
  std::int64_t total = 1;
  for (int d : shape) total *= (d > 0 ? d : 1);
  RCDL_REQUIRE(total >= num_classes_,
               ("RCDL classification: output " + describeShape(shape) + " is smaller than the " +
                std::to_string(num_classes_) + " classes resolved at construction")
                   .c_str());
  return decodeClassification(data, num_classes_, cfg_);
}

CropBox Classifier::cropFor(int src_w, int src_h) const {
  return centerCropBox(src_w, src_h, input_w_, input_h_, pre_.crop_ratio);
}

std::vector<ClsResult> Classifier::classify(const ImageView& src) {
  RCDL_REQUIRE(src.valid(),
               ("RCDL Classifier::classify: invalid source view: " + src.describe()).c_str());
  // classify() feeds raw image bytes into the input tensor. A float-input model
  // would reinterpret them as float32 and infer on garbage instead of failing,
  // so reject it here; such a model can still be driven through postprocess().
  RCDL_REQUIRE(engine_.numInputs() == 1 && engine_.inputType(0) == RKNN_TENSOR_UINT8,
               "RCDL Classifier::classify: needs a single uint8 image input; preprocess "
               "externally and call postprocess() for a float-input model");

  // The crop destination IS the NPU's input tensor: engineInputView() hands its
  // fd + virtual address + row stride to the preproc layer, so on the hardware
  // path RGA writes the model's input directly with no intermediate canvas.
  const ImageView dst = engineInputView(engine_, 0, pre_.model_input);
  const CropBox box = centerCropBox(src.width, src.height, dst.width, dst.height,
                                    pre_.crop_ratio);
  cropResizeInto(dst, src, box, pre_.backend, pre_.yuv_range, &last_backend_);

  engine_.infer();
  return postprocess();
}

std::vector<ClsResult> Classifier::classify(const std::uint8_t* bgr, int width, int height) {
  RCDL_REQUIRE(bgr != nullptr && width > 0 && height > 0,
               "RCDL Classifier::classify: invalid BGR frame");
  // ImageView is a non-owning descriptor and preproc only READS the source, so
  // this const_cast is a description-level cast, not a write.
  return classify(hostView(const_cast<std::uint8_t*>(bgr), width, height, PixelFormat::BGR888));
}

// ---------------------------------------------------------------------------
// Class labels
// ---------------------------------------------------------------------------

std::vector<std::string> loadClassLabels(const std::string& path, bool strip_wnid) {
  std::ifstream in(path);
  if (!in) {
    throw Error(-1, "RCDL classification: cannot open class label file '" + path + "'");
  }
  std::vector<std::string> labels;
  std::string line;
  while (std::getline(in, line)) {
    // Files authored on Windows leave a CR that would otherwise end up inside
    // the printed label.
    while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
      line.pop_back();
    }
    std::size_t begin = 0;
    if (strip_wnid && line.size() > 10 && line[0] == 'n' && line[9] == ' ') {
      // "n01440764 tench, Tinca tinca" -> "tench, Tinca tinca". Only stripped
      // when the prefix really is 'n' + 8 digits, so a plain label file (or a
      // label that merely starts with 'n') survives untouched.
      bool wnid = true;
      for (std::size_t i = 1; i < 9; ++i) {
        if (line[i] < '0' || line[i] > '9') {
          wnid = false;
          break;
        }
      }
      if (wnid) begin = 10;
    }
    while (begin < line.size() && (line[begin] == ' ' || line[begin] == '\t')) ++begin;
    labels.push_back(line.substr(begin));
  }
  return labels;
}

std::string classLabel(const std::vector<std::string>& labels, int class_id) {
  if (class_id >= 0 && class_id < static_cast<int>(labels.size())) {
    return labels[static_cast<std::size_t>(class_id)];
  }
  // An id past the table means a mismatched label file, not a fatal error —
  // name it so the caller can still see what the model said.
  return "class " + std::to_string(class_id);
}

}  // namespace rcdl
