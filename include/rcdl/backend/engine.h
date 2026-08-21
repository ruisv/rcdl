#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rknn_api.h"

namespace rcdl {

/// Which NPU core(s) a context runs on. RK3588 has three cores; RK3576 two;
/// RK356x one (where only Auto / Core0 are meaningful). Combined masks split a
/// single inference across cores (helps large models); one context per core
/// run concurrently maximises throughput for small models.
enum class NpuCore : int {
  Auto = RKNN_NPU_CORE_AUTO,
  Core0 = RKNN_NPU_CORE_0,
  Core1 = RKNN_NPU_CORE_1,
  Core2 = RKNN_NPU_CORE_2,
  Core01 = RKNN_NPU_CORE_0_1,
  Core012 = RKNN_NPU_CORE_0_1_2,
  All = RKNN_NPU_CORE_ALL,
};

/// Loads a compiled `.rknn` model and runs NPU inference through librknnrt.
///
/// Tensor buffers (one rknn_tensor_mem per input/output, dma-buf backed) are
/// allocated once at construction from the model's tensor attributes and bound
/// with rknn_set_io_mem — the "zero-copy" path — then reused every infer():
/// no per-frame allocation, no per-frame copy inside the runtime. The runtime
/// handles cache coherency of these buffers around rknn_run.
///
/// Inputs are presented to the runtime as UINT8 NHWC for quantized models
/// (image bytes; mean/std normalisation is folded into the model by the
/// toolkit) and FLOAT32 for float models. Outputs stay in the model's own
/// encoding (INT8 affine / FP16 / FP32); outputAsFloat() dequantizes.
struct EngineOptions {
  NpuCore core = NpuCore::Auto;
  /// rknn_init flags (RKNN_FLAG_*). COLLECT_PERF_MASK enables perfDetail().
  std::uint32_t init_flags = 0;

  /// Input indices to present as FLOAT32 even though the model quantized them.
  ///
  /// The default (image bytes for a quantized input) is right for every model
  /// fed by a camera, and wrong for a model whose input is a computed MAP: the
  /// u8 path can only express the non-negative half of the tensor's affine
  /// range, so half of a mean-zero input clips to the zero point and nothing
  /// reports it. XFeat is the case in this repo — its input is an InstanceNorm
  /// output, roughly ±3 — and any two-frame or feature-space model is the same.
  /// Task heads that need it check inputType() and say so rather than running.
  std::vector<int> float_inputs;
};

class Engine {
 public:
  using Options = EngineOptions;

  /// Load a `.rknn` file.
  explicit Engine(const std::string& rknn_path, const Options& opts = Options());
  /// Load a `.rknn` image from memory.
  Engine(const void* model_data, std::size_t model_size, const Options& opts = Options());
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  /// A second context sharing this model's weights (rknn_dup_context), with its
  /// own I/O buffers, optionally pinned to another core. The usual way to run
  /// one model on all three RK3588 cores concurrently.
  std::unique_ptr<Engine> dup(NpuCore core = NpuCore::Auto) const;

  const std::string& path() const noexcept { return path_; }
  NpuCore core() const noexcept { return core_; }

  int numInputs() const noexcept { return static_cast<int>(inputs_.size()); }
  int numOutputs() const noexcept { return static_cast<int>(outputs_.size()); }

  /// Model-side tensor attributes as queried (RKNN_QUERY_INPUT/OUTPUT_ATTR).
  /// Every index below is range-checked (throws rcdl::Error).
  const rknn_tensor_attr& inputAttr(int i) const;
  const rknn_tensor_attr& outputAttr(int i) const;

  std::vector<int> inputShape(int i) const;   ///< dims in the model's fmt (NHWC for quantized image inputs)
  std::vector<int> outputShape(int i) const;
  std::string inputName(int i) const;
  std::string outputName(int i) const;
  rknn_tensor_type inputType(int i) const;    ///< the type the CALLER provides (UINT8 / FLOAT32)
  rknn_tensor_type outputType(int i) const;   ///< the model's output encoding
  rknn_tensor_format inputFormat(int i) const;

  /// Allocated byte size of input[i]'s device buffer (with stride).
  std::size_t inputBytes(int i) const;
  /// Byte size of input[i] as a PACKED row-major array in inputType(i).
  std::size_t inputPackedBytes(int i) const;
  /// Row stride in elements along W for NHWC inputs (== width when unpadded).
  int inputWidthStride(int i) const;

  /// Copy `bytes` of host data into input[i]'s device buffer.
  /// `bytes` must be inputPackedBytes(i) (packed rows are scattered into the
  /// strided layout) or inputBytes(i) (already in device layout). Anything else
  /// throws rather than silently copying a prefix.
  void setInput(int i, const void* data, std::size_t bytes);
  /// Direct writable view of input[i]'s device buffer (device layout, i.e. honor
  /// inputWidthStride). For hardware producers (RGA) and hand-rolled preproc.
  void* inputData(int i);
  /// The dma-buf fd backing input[i] — hand it to RGA / MPP to write the tensor
  /// directly. -1 if the runtime did not expose one.
  int inputFd(int i) const;

  /// Run one inference (blocking). Outputs are valid until the next infer().
  void infer();
  /// Run asynchronously (non-blocking rknn_run); pair with wait().
  void inferAsync();
  void wait(int timeout_ms = 0);

  /// Raw output buffer after infer(), in the model's encoding + NCHW layout.
  const void* outputData(int i) const;
  std::size_t outputBytes(int i) const;         ///< allocated (with stride)
  std::size_t outputPackedBytes(int i) const;   ///< n_elems * element size
  int outputFd(int i) const;
  /// Dequantize / convert output[i] into float32 (n_elems values).
  void outputAsFloat(int i, float* dst, std::size_t n_floats) const;
  std::vector<float> outputAsFloat(int i) const;

  /// NPU time of the last infer() in microseconds (RKNN_QUERY_PERF_RUN), or -1.
  long lastRunMicros() const;
  /// Per-layer report (requires RKNN_FLAG_COLLECT_PERF_MASK in Options::init_flags).
  std::string perfDetail() const;

  /// librknnrt API version + NPU driver version ("2.3.2 ..." / "0.9.8").
  std::string sdkVersion() const;
  std::string driverVersion() const;

  rknn_context handle() const noexcept { return ctx_; }

 private:
  struct Tensor {
    rknn_tensor_attr attr{};      ///< model-side attr as queried
    rknn_tensor_attr io_attr{};   ///< attr handed to rknn_set_io_mem (type/fmt we provide)
    rknn_tensor_mem* mem = nullptr;
    std::size_t packed_bytes = 0;
  };

  Engine(rknn_context dup_from, const std::string& path, NpuCore core,
         std::vector<int> float_inputs);
  void init(const void* model_data, std::size_t model_size, const Options& opts);
  void setupIo();
  void checkInput(int i) const;
  void checkOutput(int i) const;

  std::string path_;
  NpuCore core_ = NpuCore::Auto;
  rknn_context ctx_ = 0;
  std::vector<int> float_inputs_;  ///< see EngineOptions::float_inputs
  std::vector<Tensor> inputs_;
  std::vector<Tensor> outputs_;
  bool async_pending_ = false;
};

}  // namespace rcdl
