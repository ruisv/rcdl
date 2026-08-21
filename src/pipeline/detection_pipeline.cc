#include "rcdl/pipeline/detection_pipeline.h"

#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"  // dtypeName(), for the input-type error message
#include "rcdl/core/status.h"

namespace rcdl {

const char* headName(DetectHead h) noexcept {
  switch (h) {
    case DetectHead::kAuto: return "auto";
    case DetectHead::kSingleTensor: return "single-tensor";
    case DetectHead::kYoloLtrb: return "yolo-ltrb";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Shared pipeline core
// ---------------------------------------------------------------------------

PipelineConfig resolveDetectionConfig(Engine& engine, PipelineConfig cfg) {
  if (cfg.input_w <= 0 || cfg.input_h <= 0) {
    // The letterbox canvas is the model's own spatial input size. Take it from
    // the tensor's declared layout rather than guessing from dim magnitudes:
    // a 3-class model's NCHW input is [1,3,H,W] and a 3-channel NHWC input is
    // [1,H,W,3], which are indistinguishable by shape alone.
    const rknn_tensor_attr& a = engine.inputAttr(0);
    RCDL_REQUIRE(a.n_dims == 4,
                 ("DetectionPipeline: cannot derive the input canvas from a " +
                  std::to_string(a.n_dims) + "-D input tensor; set PipelineConfig::input_w/h")
                     .c_str());
    const bool nchw = a.fmt == RKNN_TENSOR_NCHW;
    const int in_h = static_cast<int>(nchw ? a.dims[2] : a.dims[1]);
    const int in_w = static_cast<int>(nchw ? a.dims[3] : a.dims[2]);
    if (cfg.input_w <= 0) cfg.input_w = in_w;
    if (cfg.input_h <= 0) cfg.input_h = in_h;
  }
  // The single-tensor decoder scales its (cx,cy,w,h) outputs by the canvas it is
  // told about, so it must agree with the canvas we actually letterbox into.
  // (The LTRB decoder reads its grids from the model and needs no canvas here.)
  cfg.detect.input_w = cfg.input_w;
  cfg.detect.input_h = cfg.input_h;

  if (cfg.head == DetectHead::kAuto) {
    // A one-output model cannot be an LTRB head (that head emits at least a
    // cls+box pair per scale), so skip the probe and its exception for it.
    cfg.head = DetectHead::kSingleTensor;
    if (engine.numOutputs() > 1) {
      // Probed twice, and the second probe is the important one. When given a
      // num_classes > 0 resolveYoloHead() uses it as a HARD filter to tell the
      // cls tensor from the box tensor — so probing with cfg.ltrb.num_classes
      // (80 by default) throws on a perfectly good 1- or 4-class LTRB export,
      // whose branches carry no 80-channel tensor. Concluding kSingleTensor
      // there would hand the fused decoder that model's [1,64,80,80] box tensor
      // and yield garbage boxes with no error anywhere, so retry in heuristic
      // mode (num_classes == 0: box vs cls told apart by channel count) before
      // giving up on the LTRB head.
      const int probes[2] = {cfg.ltrb.num_classes, 0};
      const int n_probes = cfg.ltrb.num_classes > 0 ? 2 : 1;  // 0 twice is pointless
      for (int k = 0; k < n_probes; ++k) {
        try {
          // A throw is this probe's negative answer, not an error to propagate.
          const YoloHeadLayout layout = resolveYoloHead(engine, probes[k]);
          cfg.head = DetectHead::kYoloLtrb;
          // The layout, read from the model, is the authority — YoloLtrbDetector
          // re-resolves it and never indexes by cfg.ltrb.num_classes, so decoding
          // is safe either way. Sync it anyway: on the heuristic probe cfg still
          // carries the default 80, and anything downstream that reads the
          // resolved config (label tables, the Python bindings, logging) should
          // see the model's real class count rather than what we guessed with.
          cfg.ltrb.num_classes = layout.num_classes;
          break;
        } catch (const Error&) {
          cfg.head = DetectHead::kSingleTensor;  // not this shape; try the next probe
        }
      }
    }
  }
  return cfg;
}

void requireImageInputModel(Engine& engine) {
  // These pipelines hand the NPU raw image bytes (RGA writes them straight into
  // the input tensor). A float-input model would reinterpret those bytes as
  // float32 and infer on garbage instead of failing, so reject it loudly here.
  // inputType() is the type the runtime expects FROM US, which is what matters.
  const rknn_tensor_type it = engine.inputType(0);
  const rknn_tensor_attr& a = engine.inputAttr(0);
  RCDL_REQUIRE(it == RKNN_TENSOR_UINT8 && a.n_dims == 4,
               ("DetectionPipeline feeds uint8 image bytes, but model input 0 is " +
                std::to_string(a.n_dims) + "-D " + dtypeName(it) +
                "; preprocess externally and drive Engine + Detector directly")
                   .c_str());
}

HeadDecoder::HeadDecoder(Engine& engine, const PipelineConfig& resolved_cfg) {
  // kAuto is resolved by resolveDetectionConfig(); reaching here with it means
  // the caller skipped that step, and picking a head silently would hide it.
  RCDL_REQUIRE(resolved_cfg.head != DetectHead::kAuto,
               "HeadDecoder: pass a config resolved by resolveDetectionConfig()");
  if (resolved_cfg.head == DetectHead::kYoloLtrb) {
    // The detector re-reads grids / class count / reg_max / channel order from
    // the Engine, so only the thresholds in cfg.ltrb are actually honoured.
    ltrb_ = std::make_unique<YoloLtrbDetector>(engine, resolved_cfg.ltrb);
  } else {
    single_ = std::make_unique<Detector>(engine, resolved_cfg.detect, resolved_cfg.output_index);
  }
}

std::vector<Detection> HeadDecoder::postprocess(const LetterboxInfo& lb) const {
  return ltrb_ ? ltrb_->postprocess(lb) : single_->postprocess(lb);
}

// ---------------------------------------------------------------------------
// DetectionPipeline (synchronous)
// ---------------------------------------------------------------------------

namespace {

/// Validate the Engine and resolve the config in one expression, so both happen
/// BEFORE engineInputView() runs in the member-init list (members initialise in
/// declaration order, and cfg_ precedes input_view_).
PipelineConfig prepare(Engine& engine, PipelineConfig cfg) {
  RCDL_REQUIRE(engine.numInputs() == 1,
               ("DetectionPipeline: single-input models only, this one has " +
                std::to_string(engine.numInputs()) + " inputs")
                   .c_str());
  requireImageInputModel(engine);
  return resolveDetectionConfig(engine, std::move(cfg));
}

inline double msBetween(std::chrono::steady_clock::time_point a,
                        std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

DetectionPipeline::DetectionPipeline(Engine& engine, PipelineConfig cfg)
    : engine_(engine),
      cfg_(prepare(engine, std::move(cfg))),
      // The letterbox destination for every frame: the NPU's own input tensor,
      // resolved once here (fd + virtual address + w_stride) and never
      // re-derived, so process() touches no runtime query and allocates nothing.
      input_view_(engineInputView(engine, 0, cfg_.model_input)),
      decoder_(engine, cfg_),
      last_lb_() {
  // Sanity: the tensor we will write into must be the canvas the decoders were
  // configured for. They disagree only if the caller pinned input_w/h by hand.
  RCDL_REQUIRE(input_view_.width == cfg_.input_w && input_view_.height == cfg_.input_h,
               ("DetectionPipeline: configured canvas " + std::to_string(cfg_.input_w) + "x" +
                std::to_string(cfg_.input_h) + " does not match the model input tensor " +
                input_view_.describe())
                   .c_str());
}

std::vector<Detection> DetectionPipeline::process(const ImageView& src) {
  RCDL_REQUIRE(src.valid(),
               ("DetectionPipeline::process: invalid source view: " + src.describe()).c_str());

  using Clock = std::chrono::steady_clock;

  // 1. Preproc. One hardware op: RGA scales, converts to the model's channel
  //    order and paints the border DIRECTLY into the NPU input tensor. No
  //    intermediate canvas, no CPU copy — `used` tells us afterwards whether the
  //    hardware took it or the CPU fallback ran (a slow frame is then traceable).
  const auto t0 = Clock::now();
  last_lb_ = letterbox(input_view_, src, cfg_.pad_value, cfg_.backend, cfg_.yuv_range,
                       &last_backend_);

  // 2. Infer. I/O was bound once at Engine construction, so this is submit+wait;
  //    the runtime handles cache maintenance around its own tensors.
  const auto t1 = Clock::now();
  engine_.infer();

  // 3. Decode. Boxes come back in ORIGINAL-image pixels: last_lb_ carries the
  //    geometry RGA actually used (integer-rounded rectangle included), so the
  //    inverse map matches the pixels the NPU saw.
  const auto t2 = Clock::now();
  std::vector<Detection> dets = decoder_.postprocess(last_lb_);
  const auto t3 = Clock::now();

  prof_.preproc_ms += msBetween(t0, t1);
  prof_.infer_ms += msBetween(t1, t2);
  prof_.postproc_ms += msBetween(t2, t3);
  ++prof_.frames;
  return dets;
}

std::vector<Detection> DetectionPipeline::process(const std::uint8_t* bgr, int width, int height) {
  RCDL_REQUIRE(bgr != nullptr && width > 0 && height > 0,
               "DetectionPipeline::process: invalid BGR frame");
  // ImageView is a non-owning descriptor and the preproc path only READS the
  // source, so const_cast here is a description-level cast, not a write.
  return process(hostView(const_cast<std::uint8_t*>(bgr), width, height, PixelFormat::BGR888));
}

}  // namespace rcdl
