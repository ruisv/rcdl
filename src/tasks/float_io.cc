#include "float_io.h"

#include <cmath>
#include <cstdint>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/status.h"

namespace rcdl {
namespace detail {

namespace {

bool nhwc4(const rknn_tensor_attr& a) { return a.n_dims == 4 && a.fmt == RKNN_TENSOR_NHWC; }

std::vector<int> logical(const rknn_tensor_attr& a) {
  std::vector<int> s(a.dims, a.dims + a.n_dims);
  if (nhwc4(a)) return {s[0], s[3], s[1], s[2]};
  return s;
}

std::size_t count(const std::vector<int>& s) {
  std::size_t n = 1;
  for (int d : s) n *= static_cast<std::size_t>(d);
  return n;
}

}  // namespace

std::vector<int> logicalInputShape(const Engine& engine, int i) {
  return logical(engine.inputAttr(i));
}

std::vector<int> logicalOutputShape(const Engine& engine, int i) {
  return logical(engine.outputAttr(i));
}

int inputIndex(const Engine& engine, const std::string& name) {
  for (int i = 0; i < engine.numInputs(); ++i)
    if (engine.inputName(i) == name) return i;
  return -1;
}

int outputIndex(const Engine& engine, const std::string& name) {
  for (int i = 0; i < engine.numOutputs(); ++i)
    if (engine.outputName(i) == name) return i;
  return -1;
}

void setInputFloat(Engine& engine, int i, const float* data, std::size_t n,
                   std::vector<float>& stage, const char* who) {
  const rknn_tensor_attr& a = engine.inputAttr(i);
  if (engine.inputType(i) == RKNN_TENSOR_INT32) {
    // An integer input (a length or a count): rounded, never reinterpreted.
    if (n != a.n_elems)
      throw Error(-1, std::string("RCDL ") + who + ": input '" + a.name + "' holds " +
                          std::to_string(a.n_elems) + " values, got " + std::to_string(n));
    std::vector<std::int32_t> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = static_cast<std::int32_t>(std::lround(data[k]));
    engine.setInput(i, v.data(), n * sizeof(std::int32_t));
    return;
  }
  if (engine.inputType(i) != RKNN_TENSOR_FLOAT32)
    throw Error(-1, std::string("RCDL ") + who + ": input '" + a.name +
                        "' is quantized and takes image bytes by default, but this is a "
                        "computed tensor — open the Engine with float_inputs=[" +
                        std::to_string(i) + "] (or use a float build)");
  if (n != a.n_elems)
    throw Error(-1, std::string("RCDL ") + who + ": input '" + a.name + "' holds " +
                        std::to_string(a.n_elems) + " values, got " + std::to_string(n));
  if (!nhwc4(a)) {
    engine.setInput(i, data, n * sizeof(float));
    return;
  }
  // NCHW host data into an NHWC device tensor.
  const int c = static_cast<int>(a.dims[3]);
  const std::size_t plane = static_cast<std::size_t>(a.dims[1]) * a.dims[2];
  const std::size_t batch = plane * c;
  stage.resize(n);
  for (std::size_t b = 0; b < static_cast<std::size_t>(a.dims[0]); ++b) {
    const float* src = data + b * batch;
    float* dst = stage.data() + b * batch;
    for (std::size_t p = 0; p < plane; ++p)
      for (int ch = 0; ch < c; ++ch) dst[p * c + ch] = src[ch * plane + p];
  }
  engine.setInput(i, stage.data(), n * sizeof(float));
}

const float* outputFloat(const Engine& engine, int i, std::vector<float>& scratch,
                         std::vector<int>& shape) {
  const rknn_tensor_attr& a = engine.outputAttr(i);
  if (!nhwc4(a)) {
    const float* p = outputAsFloat(engine, i, scratch, shape);
    shape = logical(a);
    return p;
  }
  std::vector<float> tmp;
  std::vector<int> raw;
  const float* src = outputAsFloat(engine, i, tmp, raw);
  shape = logical(a);
  const int h = static_cast<int>(a.dims[1]), w = static_cast<int>(a.dims[2]),
            c = static_cast<int>(a.dims[3]);
  const std::size_t plane = static_cast<std::size_t>(h) * w;
  const std::size_t batch = plane * c;
  scratch.resize(count(shape));
  for (std::size_t b = 0; b < static_cast<std::size_t>(a.dims[0]); ++b) {
    const float* s = src + b * batch;
    float* d = scratch.data() + b * batch;
    for (std::size_t p = 0; p < plane; ++p)
      for (int ch = 0; ch < c; ++ch) d[ch * plane + p] = s[p * c + ch];
  }
  return scratch.data();
}

}  // namespace detail
}  // namespace rcdl
