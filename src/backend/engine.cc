#include "rcdl/backend/engine.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {

namespace {

std::vector<std::uint8_t> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  RCDL_REQUIRE(f.good(), ("cannot open model file: " + path).c_str());
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), {});
}

// Width (innermost spatial dim) of a tensor in its own layout; 1 when the
// layout has no spatial meaning (1-D / 2-D tensors).
int widthOf(const rknn_tensor_attr& a) {
  if (a.n_dims == 4) {
    if (a.fmt == RKNN_TENSOR_NHWC) return static_cast<int>(a.dims[2]);
    return static_cast<int>(a.dims[3]);  // NCHW and friends
  }
  return a.n_dims > 0 ? static_cast<int>(a.dims[a.n_dims - 1]) : 1;
}

// Number of "rows" of width elements: product of every dim except the width
// dim (and, for NHWC, times channels folds into the row).
std::size_t rowCount(const rknn_tensor_attr& a) {
  if (a.n_dims == 4) {
    if (a.fmt == RKNN_TENSOR_NHWC) return static_cast<std::size_t>(a.dims[0]) * a.dims[1];
    return static_cast<std::size_t>(a.dims[0]) * a.dims[1] * a.dims[2];
  }
  std::size_t rows = 1;
  for (std::uint32_t d = 0; d + 1 < a.n_dims; ++d) rows *= a.dims[d];
  return rows;
}

// Elements per row in packed vs strided layout (NHWC rows carry C per pixel).
std::size_t rowElems(const rknn_tensor_attr& a, int width) {
  if (a.n_dims == 4 && a.fmt == RKNN_TENSOR_NHWC) return static_cast<std::size_t>(width) * a.dims[3];
  return static_cast<std::size_t>(width);
}

}  // namespace

// --- construction -------------------------------------------------------------

Engine::Engine(const std::string& rknn_path, const Options& opts) : path_(rknn_path) {
  auto blob = readFile(rknn_path);
  init(blob.data(), blob.size(), opts);
}

Engine::Engine(const void* model_data, std::size_t model_size, const Options& opts)
    : path_("<memory>") {
  init(model_data, model_size, opts);
}

Engine::Engine(rknn_context dup_from, const std::string& path, NpuCore core)
    : path_(path), core_(core) {
  rknn_context in = dup_from;
  RCDL_CHECK(rknn_dup_context(&in, &ctx_));
  if (core_ != NpuCore::Auto) {
    RCDL_CHECK(rknn_set_core_mask(ctx_, static_cast<rknn_core_mask>(core_)));
  }
  setupIo();
}

void Engine::init(const void* model_data, std::size_t model_size, const Options& opts) {
  RCDL_REQUIRE(model_data != nullptr && model_size > 0, "empty model image");
  core_ = opts.core;
  // rknn_init copies the model image (unless RKNN_FLAG_MODEL_BUFFER_ZERO_COPY),
  // so the caller's buffer may go away afterwards.
  RCDL_CHECK(rknn_init(&ctx_, const_cast<void*>(model_data),
                       static_cast<std::uint32_t>(model_size), opts.init_flags, nullptr));
  if (core_ != NpuCore::Auto) {
    RCDL_CHECK(rknn_set_core_mask(ctx_, static_cast<rknn_core_mask>(core_)));
  }
  setupIo();
}

void Engine::setupIo() {
  rknn_input_output_num io{};
  RCDL_CHECK(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)));
  inputs_.resize(io.n_input);
  outputs_.resize(io.n_output);

  for (std::uint32_t i = 0; i < io.n_input; ++i) {
    Tensor& t = inputs_[i];
    t.attr.index = i;
    RCDL_CHECK(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &t.attr, sizeof(t.attr)));
    t.io_attr = t.attr;
    // What the caller hands us: image bytes for quantized models, float32 for
    // float models. The runtime converts to the model's own encoding on the way
    // in (pass_through == 0).
    switch (t.attr.type) {
      case RKNN_TENSOR_INT8:
      case RKNN_TENSOR_UINT8:
        t.io_attr.type = RKNN_TENSOR_UINT8;
        break;
      case RKNN_TENSOR_FLOAT16:
      case RKNN_TENSOR_FLOAT32:
        t.io_attr.type = RKNN_TENSOR_FLOAT32;
        break;
      default:
        break;  // keep as is (int16/int32 inputs are rare; provide the model's type)
    }
    if (t.io_attr.fmt == RKNN_TENSOR_NC1HWC2 || t.io_attr.fmt == RKNN_TENSOR_UNDEFINED) {
      t.io_attr.fmt = RKNN_TENSOR_NHWC;
    }
    t.io_attr.pass_through = 0;

    const std::size_t model_elem = elementSize(t.attr.type);
    const std::size_t io_elem = elementSize(t.io_attr.type);
    RCDL_REQUIRE(model_elem > 0 && io_elem > 0, "unsupported input tensor type");
    std::size_t strided_elems =
        t.attr.size_with_stride ? t.attr.size_with_stride / model_elem : t.attr.n_elems;
    strided_elems = std::max<std::size_t>(strided_elems, t.attr.n_elems);
    t.packed_bytes = static_cast<std::size_t>(t.attr.n_elems) * io_elem;
    const std::size_t bytes = strided_elems * io_elem;

    t.mem = rknn_create_mem(ctx_, static_cast<std::uint32_t>(bytes));
    RCDL_REQUIRE(t.mem != nullptr, "rknn_create_mem (input) failed");
    RCDL_CHECK(rknn_set_io_mem(ctx_, t.mem, &t.io_attr));
  }

  for (std::uint32_t i = 0; i < io.n_output; ++i) {
    Tensor& t = outputs_[i];
    t.attr.index = i;
    RCDL_CHECK(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &t.attr, sizeof(t.attr)));
    // Keep the model's own encoding (int8 affine / fp16 / fp32) and the standard
    // NCHW layout; dequantisation is ours (outputAsFloat) and stays off the
    // critical path for decoders that can work on int8 directly.
    t.io_attr = t.attr;
    const std::size_t elem = elementSize(t.attr.type);
    RCDL_REQUIRE(elem > 0, "unsupported output tensor type");
    t.packed_bytes = static_cast<std::size_t>(t.attr.n_elems) * elem;
    const std::size_t bytes = std::max<std::size_t>(
        std::max<std::size_t>(t.attr.size_with_stride, t.attr.size), t.packed_bytes);
    t.mem = rknn_create_mem(ctx_, static_cast<std::uint32_t>(bytes));
    RCDL_REQUIRE(t.mem != nullptr, "rknn_create_mem (output) failed");
    RCDL_CHECK(rknn_set_io_mem(ctx_, t.mem, &t.io_attr));
  }
}

Engine::~Engine() {
  for (auto& t : inputs_) {
    if (t.mem) rknn_destroy_mem(ctx_, t.mem);
  }
  for (auto& t : outputs_) {
    if (t.mem) rknn_destroy_mem(ctx_, t.mem);
  }
  if (ctx_) rknn_destroy(ctx_);
}

std::unique_ptr<Engine> Engine::dup(NpuCore core) const {
  return std::unique_ptr<Engine>(new Engine(ctx_, path_, core));
}

// --- introspection ------------------------------------------------------------

void Engine::checkInput(int i) const {
  RCDL_REQUIRE(i >= 0 && i < numInputs(), "input index out of range");
}
void Engine::checkOutput(int i) const {
  RCDL_REQUIRE(i >= 0 && i < numOutputs(), "output index out of range");
}

const rknn_tensor_attr& Engine::inputAttr(int i) const {
  checkInput(i);
  return inputs_[i].attr;
}
const rknn_tensor_attr& Engine::outputAttr(int i) const {
  checkOutput(i);
  return outputs_[i].attr;
}

std::vector<int> Engine::inputShape(int i) const {
  const auto& a = inputAttr(i);
  return std::vector<int>(a.dims, a.dims + a.n_dims);
}
std::vector<int> Engine::outputShape(int i) const {
  const auto& a = outputAttr(i);
  return std::vector<int>(a.dims, a.dims + a.n_dims);
}
std::string Engine::inputName(int i) const { return inputAttr(i).name; }
std::string Engine::outputName(int i) const { return outputAttr(i).name; }
rknn_tensor_type Engine::inputType(int i) const {
  checkInput(i);
  return inputs_[i].io_attr.type;
}
rknn_tensor_type Engine::outputType(int i) const { return outputAttr(i).type; }
rknn_tensor_format Engine::inputFormat(int i) const {
  checkInput(i);
  return inputs_[i].io_attr.fmt;
}

std::size_t Engine::inputBytes(int i) const {
  checkInput(i);
  return inputs_[i].mem->size;
}
std::size_t Engine::inputPackedBytes(int i) const {
  checkInput(i);
  return inputs_[i].packed_bytes;
}
int Engine::inputWidthStride(int i) const {
  const auto& a = inputAttr(i);
  return a.w_stride ? static_cast<int>(a.w_stride) : widthOf(a);
}

// --- data path ------------------------------------------------------------------

void Engine::setInput(int i, const void* data, std::size_t bytes) {
  checkInput(i);
  Tensor& t = inputs_[i];
  RCDL_REQUIRE(data != nullptr, "setInput: null data");
  const std::size_t dev_bytes = t.mem->size;
  if (bytes == dev_bytes) {
    std::memcpy(t.mem->virt_addr, data, bytes);
    return;
  }
  RCDL_REQUIRE(bytes == t.packed_bytes,
               "setInput: byte count is neither inputPackedBytes(i) nor inputBytes(i)");
  const int width = widthOf(t.attr);
  const int stride = inputWidthStride(i);
  const std::size_t elem = elementSize(t.io_attr.type);
  const std::size_t src_row = rowElems(t.attr, width) * elem;
  const std::size_t dst_row = rowElems(t.attr, stride) * elem;
  if (src_row == dst_row) {
    std::memcpy(t.mem->virt_addr, data, bytes);
    return;
  }
  // Scatter packed rows into the padded device layout.
  const std::size_t rows = rowCount(t.attr);
  RCDL_REQUIRE(rows * dst_row <= dev_bytes, "setInput: strided layout exceeds device buffer");
  const auto* src = static_cast<const std::uint8_t*>(data);
  auto* dst = static_cast<std::uint8_t*>(t.mem->virt_addr);
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(dst + r * dst_row, src + r * src_row, src_row);
  }
}

void* Engine::inputData(int i) {
  checkInput(i);
  return inputs_[i].mem->virt_addr;
}
int Engine::inputFd(int i) const {
  checkInput(i);
  return inputs_[i].mem->fd;
}

void Engine::infer() {
  RCDL_CHECK(rknn_run(ctx_, nullptr));
  async_pending_ = false;
}

void Engine::inferAsync() {
  rknn_run_extend ext{};
  ext.non_block = 1;
  RCDL_CHECK(rknn_run(ctx_, &ext));
  async_pending_ = true;
}

void Engine::wait(int timeout_ms) {
  if (!async_pending_) return;
  rknn_run_extend ext{};
  ext.non_block = 0;
  ext.timeout_ms = timeout_ms;
  RCDL_CHECK(rknn_wait(ctx_, &ext));
  async_pending_ = false;
}

const void* Engine::outputData(int i) const {
  checkOutput(i);
  return outputs_[i].mem->virt_addr;
}
std::size_t Engine::outputBytes(int i) const {
  checkOutput(i);
  return outputs_[i].mem->size;
}
std::size_t Engine::outputPackedBytes(int i) const {
  checkOutput(i);
  return outputs_[i].packed_bytes;
}
int Engine::outputFd(int i) const {
  checkOutput(i);
  return outputs_[i].mem->fd;
}

void Engine::outputAsFloat(int i, float* dst, std::size_t n_floats) const {
  checkOutput(i);
  const Tensor& t = outputs_[i];
  RCDL_REQUIRE(n_floats == t.attr.n_elems, "outputAsFloat: destination must hold n_elems floats");
  const int width = widthOf(t.attr);
  const int stride = t.attr.w_stride ? static_cast<int>(t.attr.w_stride) : width;
  if (stride == width) {
    dequantizeToFloat(t.attr, t.mem->virt_addr, dst, n_floats);
    return;
  }
  // Padded rows: gather into a packed scratch buffer first.
  const std::size_t elem = elementSize(t.attr.type);
  const std::size_t src_row = rowElems(t.attr, stride) * elem;
  const std::size_t dst_row = rowElems(t.attr, width) * elem;
  const std::size_t rows = rowCount(t.attr);
  std::vector<std::uint8_t> packed(rows * dst_row);
  const auto* src = static_cast<const std::uint8_t*>(t.mem->virt_addr);
  for (std::size_t r = 0; r < rows; ++r) {
    std::memcpy(packed.data() + r * dst_row, src + r * src_row, dst_row);
  }
  dequantizeToFloat(t.attr, packed.data(), dst, n_floats);
}

std::vector<float> Engine::outputAsFloat(int i) const {
  checkOutput(i);
  std::vector<float> out(outputs_[i].attr.n_elems);
  outputAsFloat(i, out.data(), out.size());
  return out;
}

// --- diagnostics ----------------------------------------------------------------

long Engine::lastRunMicros() const {
  rknn_perf_run pr{};
  if (rknn_query(ctx_, RKNN_QUERY_PERF_RUN, &pr, sizeof(pr)) != RKNN_SUCC) return -1;
  return static_cast<long>(pr.run_duration);
}

std::string Engine::perfDetail() const {
  rknn_perf_detail pd{};
  if (rknn_query(ctx_, RKNN_QUERY_PERF_DETAIL, &pd, sizeof(pd)) != RKNN_SUCC || !pd.perf_data) {
    return "";
  }
  return std::string(pd.perf_data, pd.perf_data + pd.data_len);
}

std::string Engine::sdkVersion() const {
  rknn_sdk_version v{};
  RCDL_CHECK(rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &v, sizeof(v)));
  return v.api_version;
}

std::string Engine::driverVersion() const {
  rknn_sdk_version v{};
  RCDL_CHECK(rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &v, sizeof(v)));
  return v.drv_version;
}

}  // namespace rcdl
