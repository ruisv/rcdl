// nanobind bindings for RCDL.
//
// The data path stays explicit: inputs come in as C-contiguous numpy arrays
// (raw bytes copied into the NPU input buffer), outputs go out as float32
// numpy arrays (dequantized) or raw bytes + dtype. The pure-python `rcdl`
// wrapper (python/rcdl/__init__.py) adds the numpy conveniences on top.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/core/dma_buf.h"
#include "rcdl/version.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Contig = nb::ndarray<nb::numpy, nb::c_contig, nb::device::cpu>;

void setInputFromArray(rcdl::Engine& e, int i, const Contig& arr) {
  std::size_t bytes = arr.itemsize();
  for (std::size_t d = 0; d < arr.ndim(); ++d) bytes *= arr.shape(d);
  e.setInput(i, arr.data(), bytes);
}

nb::ndarray<nb::numpy, float> outputFloat(const rcdl::Engine& e, int i) {
  const auto& a = e.outputAttr(i);
  std::vector<std::size_t> shape(a.dims, a.dims + a.n_dims);
  if (shape.empty()) shape.push_back(a.n_elems);
  float* buf = new float[a.n_elems];
  e.outputAsFloat(i, buf, a.n_elems);
  nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
  return nb::ndarray<nb::numpy, float>(buf, shape.size(), shape.data(), owner);
}

}  // namespace

NB_MODULE(rcdl_py, m) {
  m.doc() = "RCDL — RKNPU inference & media library (compiled core)";
  m.attr("__version__") = RCDL_VERSION_STRING;

  nb::enum_<rcdl::NpuCore>(m, "NpuCore")
      .value("AUTO", rcdl::NpuCore::Auto)
      .value("CORE_0", rcdl::NpuCore::Core0)
      .value("CORE_1", rcdl::NpuCore::Core1)
      .value("CORE_2", rcdl::NpuCore::Core2)
      .value("CORE_0_1", rcdl::NpuCore::Core01)
      .value("CORE_0_1_2", rcdl::NpuCore::Core012)
      .value("ALL", rcdl::NpuCore::All);

  nb::class_<rcdl::Engine>(m, "Engine")
      .def(
          "__init__",
          [](rcdl::Engine* self, const std::string& path, rcdl::NpuCore core,
             std::uint32_t init_flags) {
            rcdl::Engine::Options o;
            o.core = core;
            o.init_flags = init_flags;
            new (self) rcdl::Engine(path, o);
          },
          "path"_a, "core"_a = rcdl::NpuCore::Auto, "init_flags"_a = 0u)
      .def("dup", &rcdl::Engine::dup, "core"_a = rcdl::NpuCore::Auto)
      .def_prop_ro("path", &rcdl::Engine::path)
      .def_prop_ro("core", &rcdl::Engine::core)
      .def_prop_ro("num_inputs", &rcdl::Engine::numInputs)
      .def_prop_ro("num_outputs", &rcdl::Engine::numOutputs)
      .def("input_shape", &rcdl::Engine::inputShape, "i"_a)
      .def("output_shape", &rcdl::Engine::outputShape, "i"_a)
      .def("input_name", &rcdl::Engine::inputName, "i"_a)
      .def("output_name", &rcdl::Engine::outputName, "i"_a)
      .def(
          "input_dtype", [](const rcdl::Engine& e, int i) { return rcdl::dtypeName(e.inputType(i)); },
          "i"_a, "dtype the caller provides for input i ('u8' for quantized image models, 'f32' for float models)")
      .def(
          "output_dtype",
          [](const rcdl::Engine& e, int i) { return rcdl::dtypeName(e.outputType(i)); }, "i"_a)
      .def(
          "input_format",
          [](const rcdl::Engine& e, int i) { return get_format_string(e.inputFormat(i)); }, "i"_a)
      .def(
          "output_quant",
          [](const rcdl::Engine& e, int i) {
            const auto& a = e.outputAttr(i);
            return nb::make_tuple(static_cast<int>(a.qnt_type), a.zp, a.scale, static_cast<int>(a.fl));
          },
          "i"_a, "(qnt_type, zp, scale, fl) of output i")
      .def("input_bytes", &rcdl::Engine::inputBytes, "i"_a)
      .def("input_packed_bytes", &rcdl::Engine::inputPackedBytes, "i"_a)
      .def("input_width_stride", &rcdl::Engine::inputWidthStride, "i"_a)
      .def("input_fd", &rcdl::Engine::inputFd, "i"_a)
      .def("output_bytes", &rcdl::Engine::outputBytes, "i"_a)
      .def("output_packed_bytes", &rcdl::Engine::outputPackedBytes, "i"_a)
      .def("output_fd", &rcdl::Engine::outputFd, "i"_a)
      .def("set_input", &setInputFromArray, "i"_a, "array"_a,
           "Copy a C-contiguous array (packed or device-strided byte count) into input i")
      .def(
          "infer", [](rcdl::Engine& e) { nb::gil_scoped_release nogil; e.infer(); },
          "Run one inference (blocking, GIL released)")
      .def(
          "infer_async", [](rcdl::Engine& e) { nb::gil_scoped_release nogil; e.inferAsync(); })
      .def(
          "wait",
          [](rcdl::Engine& e, int timeout_ms) {
            nb::gil_scoped_release nogil;
            e.wait(timeout_ms);
          },
          "timeout_ms"_a = 0)
      .def("output_float", &outputFloat, "i"_a,
           "Output i dequantized to a float32 numpy array of the model's shape")
      .def(
          "output_raw",
          [](const rcdl::Engine& e, int i) {
            return nb::bytes(static_cast<const char*>(e.outputData(i)), e.outputBytes(i));
          },
          "i"_a, "Raw output buffer (model encoding, device layout)")
      .def("last_run_micros", &rcdl::Engine::lastRunMicros)
      .def("perf_detail", &rcdl::Engine::perfDetail)
      .def("sdk_version", &rcdl::Engine::sdkVersion)
      .def("driver_version", &rcdl::Engine::driverVersion);

  nb::enum_<rcdl::DmaBuf::Heap>(m, "DmaHeap")
      .value("SYSTEM", rcdl::DmaBuf::Heap::System)
      .value("SYSTEM_UNCACHED", rcdl::DmaBuf::Heap::SystemUncached)
      .value("CMA", rcdl::DmaBuf::Heap::Cma)
      .value("CMA_UNCACHED", rcdl::DmaBuf::Heap::CmaUncached);

  nb::class_<rcdl::DmaBuf>(m, "DmaBuf")
      .def_static(
          "alloc",
          [](std::size_t size, rcdl::DmaBuf::Heap heap) {
            return std::make_unique<rcdl::DmaBuf>(rcdl::DmaBuf::alloc(size, heap));
          },
          "size"_a, "heap"_a = rcdl::DmaBuf::Heap::System)
      .def_prop_ro("fd", &rcdl::DmaBuf::fd)
      .def_prop_ro("size", &rcdl::DmaBuf::size)
      .def("sync_start", &rcdl::DmaBuf::syncStart, "read"_a = true, "write"_a = true)
      .def("sync_end", &rcdl::DmaBuf::syncEnd, "read"_a = true, "write"_a = true)
      .def(
          "write",
          [](rcdl::DmaBuf& b, nb::bytes data, std::size_t offset) {
            if (offset + data.size() > b.size()) throw std::out_of_range("DmaBuf.write overflow");
            std::memcpy(static_cast<char*>(b.data()) + offset, data.c_str(), data.size());
          },
          "data"_a, "offset"_a = 0)
      .def(
          "read",
          [](rcdl::DmaBuf& b, std::size_t n, std::size_t offset) {
            if (offset + n > b.size()) throw std::out_of_range("DmaBuf.read overflow");
            return nb::bytes(static_cast<const char*>(b.data()) + offset, n);
          },
          "n"_a, "offset"_a = 0);

  m.def("dequantize", [](nb::bytes raw, const std::string& dtype, int qnt_type, int zp, float scale,
                         int fl) {
    rknn_tensor_attr a{};
    if (dtype == "f32") a.type = RKNN_TENSOR_FLOAT32;
    else if (dtype == "f16") a.type = RKNN_TENSOR_FLOAT16;
    else if (dtype == "i8") a.type = RKNN_TENSOR_INT8;
    else if (dtype == "u8") a.type = RKNN_TENSOR_UINT8;
    else if (dtype == "i16") a.type = RKNN_TENSOR_INT16;
    else if (dtype == "u16") a.type = RKNN_TENSOR_UINT16;
    else if (dtype == "i32") a.type = RKNN_TENSOR_INT32;
    else if (dtype == "u32") a.type = RKNN_TENSOR_UINT32;
    else if (dtype == "i64") a.type = RKNN_TENSOR_INT64;
    else if (dtype == "bool") a.type = RKNN_TENSOR_BOOL;
    else throw std::invalid_argument("dequantize: unknown dtype " + dtype);
    a.qnt_type = static_cast<rknn_tensor_qnt_type>(qnt_type);
    a.zp = zp;
    a.scale = scale;
    a.fl = static_cast<std::int8_t>(fl);
    const std::size_t elem = rcdl::elementSize(a.type);
    const std::size_t n = raw.size() / elem;
    float* buf = new float[n];
    rcdl::dequantizeToFloat(a, raw.c_str(), buf, n);
    nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<float*>(p); });
    std::size_t shape[1] = {n};
    return nb::ndarray<nb::numpy, float>(buf, 1, shape, owner);
  }, "raw"_a, "dtype"_a, "qnt_type"_a = 0, "zp"_a = 0, "scale"_a = 1.0f, "fl"_a = 0,
     "Dequantize raw tensor bytes the way Engine.output_float does (test hook; needs no NPU)");
}
