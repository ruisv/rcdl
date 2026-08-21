#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rcdl/preproc/geometry.h"
#include "rcdl/preproc/image.h"
#include "rcdl/preproc/letterbox.h"
#include "rcdl/tasks/detection.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Which detection-head decoder the pipeline drives.
enum class DetectHead {
  /// Pick from the Engine's output signature at construction: an output count
  /// that resolves through resolveYoloHead() => kYoloLtrb, otherwise
  /// kSingleTensor.
  kAuto,
  /// One fused tensor [1,4+nc,N] / [1,N,4+nc]. Uses PipelineConfig::detect.
  kSingleTensor,
  /// Anchor-free LTRB multi-scale head (rknn_model_zoo YOLOv8/YOLO11, RDK YOLO).
  /// Uses PipelineConfig::ltrb.
  kYoloLtrb,
};

const char* headName(DetectHead h) noexcept;

/// Tunables for a streaming DetectionPipeline.
///
/// `input_w`/`input_h` are the model's letterbox canvas (e.g. 640x640). Leave
/// them <= 0 to have the pipeline derive them from the Engine's input[0] shape.
/// `model_input` is the CHANNEL ORDER the model was built with — RGB888 for the
/// rknn_model_zoo exports (`--input-order rgb`), BGR888 for an OpenCV-native
/// build. Getting it wrong silently costs accuracy, so it is explicit here and
/// recorded per model in scripts/fetch_models.sh.
struct PipelineConfig {
  int input_w = 0;  ///< model input width  (0 => derive from Engine)
  int input_h = 0;  ///< model input height (0 => derive from Engine)
  PixelFormat model_input = PixelFormat::RGB888;  ///< channel order the model expects
  DetectHead head = DetectHead::kAuto;
  DetectConfig detect;   ///< kSingleTensor decode config
  YoloLtrbConfig ltrb;   ///< kYoloLtrb decode config
  int output_index = 0;  ///< kSingleTensor: which output holds the head
  std::uint8_t pad_value = 114;                          ///< letterbox border (YOLO default)
  PreprocBackend backend = PreprocBackend::Auto;         ///< RGA, CPU, or pick
  YuvRange yuv_range = YuvRange::kStudioToFull;          ///< NV12 sources: level handling
};

/// Per-stage timing accumulator (milliseconds, summed over all frames), shared
/// by the synchronous DetectionPipeline and the threaded async pipelines so both
/// report a comparable breakdown.
///
/// In the SYNC pipeline the stages run back-to-back, so their sum equals the
/// per-frame process() cost. In an ASYNC pipeline each stage is timed on its own
/// worker thread (SERVICE time), so the sum EXCEEDS wall time — that is the
/// point: it shows which stage bounds the overlapped pipeline.
struct StageProfile {
  double decode_ms = 0;    ///< VPU video decode (video pipelines only; 0 otherwise)
  double preproc_ms = 0;   ///< letterbox into the NPU input tensor (RGA or CPU)
  double infer_ms = 0;     ///< Engine::infer (NPU submit + wait)
  double postproc_ms = 0;  ///< dequant + decode + per-class NMS (CPU)
  std::uint64_t frames = 0;
  double totalMs() const { return decode_ms + preproc_ms + infer_ms + postproc_ms; }
  double decodePerFrame() const { return frames ? decode_ms / frames : 0.0; }
  double preprocPerFrame() const { return frames ? preproc_ms / frames : 0.0; }
  double inferPerFrame() const { return frames ? infer_ms / frames : 0.0; }
  double postprocPerFrame() const { return frames ? postproc_ms / frames : 0.0; }
};

// ---------------------------------------------------------------------------
// Shared pipeline core — reused by the synchronous DetectionPipeline and the
// threaded async pipelines so preproc / feed / decode logic lives in one place.
// ---------------------------------------------------------------------------

/// Fill in cfg.input_w/input_h from the Engine's input[0] when unset, sync the
/// decode configs to that canvas, and resolve DetectHead::kAuto to a concrete
/// head from the Engine's output signature. Pure; returns the resolved config.
PipelineConfig resolveDetectionConfig(Engine& engine, PipelineConfig cfg);

/// Throw rcdl::Error unless the model's input 0 is a UINT8 NHWC image tensor:
/// these pipelines feed image bytes, and a float-input model would silently
/// infer on garbage.
void requireImageInputModel(Engine& engine);

/// Owns the resolved detector (single-tensor or LTRB) for an already-resolved
/// PipelineConfig and dispatches postprocess to it. Holds an Engine&.
class HeadDecoder {
 public:
  HeadDecoder(Engine& engine, const PipelineConfig& resolved_cfg);
  std::vector<Detection> postprocess(const LetterboxInfo& lb) const;
  /// Non-null for kYoloLtrb — lets callers print the resolved head layout.
  const YoloLtrbDetector* ltrb() const noexcept { return ltrb_.get(); }

 private:
  std::unique_ptr<Detector> single_;
  std::unique_ptr<YoloLtrbDetector> ltrb_;
};

/// Synchronous, allocation-free-per-frame streaming object detector.
///
/// The whole point on Rockchip: the letterbox destination IS the NPU's input
/// tensor. Engine allocated that tensor as a dma-buf at construction and bound
/// it once with rknn_set_io_mem; engineInputView() hands its fd to RGA, so one
/// `improcess` crops, scales, converts NV12 -> RGB888 and paints the border
/// straight into the buffer the NPU will read. There is no intermediate canvas
/// and no CPU copy on the hardware path — per frame this pipeline allocates
/// nothing at all.
///
/// Per-frame data flow:
///   letterbox(engine input tensor <- source frame)   RGA (or CPU fallback)
///   engine.infer()                                   NPU
///   decoder.postprocess(last_lb_)                    CPU (dequant + NMS)
///
/// Cache discipline: on the RGA path neither side is CPU-touched, so nothing is
/// flushed; on the CPU fallback the input tensor is written through its mapping
/// and the RKNN runtime flushes its own I/O around rknn_run.
class DetectionPipeline {
 public:
  /// Build the pipeline; resolves the config and the head decoder once.
  DetectionPipeline(Engine& engine, PipelineConfig cfg);

  DetectionPipeline(const DetectionPipeline&) = delete;
  DetectionPipeline& operator=(const DetectionPipeline&) = delete;

  /// Run one frame end-to-end and return detections in ORIGINAL-image pixels.
  ///
  /// `src` may be any format the preproc layer accepts — an NV12 dma-buf from
  /// the VPU (zero copy), or a host BGR888 buffer from cv::imread. Nothing is
  /// heap-allocated in steady state.
  std::vector<Detection> process(const ImageView& src);

  /// Convenience overload for an interleaved, row-contiguous host BGR image.
  std::vector<Detection> process(const std::uint8_t* bgr, int width, int height);

  const PipelineConfig& config() const noexcept { return cfg_; }
  /// Letterbox geometry of the most recent process() call.
  const LetterboxInfo& lastLetterbox() const noexcept { return last_lb_; }
  /// Backend that ran the most recent preproc (tells you if RGA fell back).
  PreprocBackend lastBackend() const noexcept { return last_backend_; }
  DetectHead head() const noexcept { return cfg_.head; }
  const HeadDecoder& decoder() const noexcept { return decoder_; }

  /// Per-stage timing accumulated across every process() call.
  const StageProfile& profile() const noexcept { return prof_; }
  void resetProfile() noexcept { prof_ = StageProfile{}; }

 private:
  Engine& engine_;
  PipelineConfig cfg_;
  ImageView input_view_;  ///< the Engine input tensor, as a letterbox destination
  HeadDecoder decoder_;
  LetterboxInfo last_lb_;
  PreprocBackend last_backend_ = PreprocBackend::Auto;
  StageProfile prof_;
};

}  // namespace rcdl
