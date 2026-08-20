// Load an .rknn, print its I/O signature and the runtime/driver versions, run
// one inference on a zero-filled input, and report latency.
//
//   ./model_info model.rknn [core]      core: auto|0|1|2|01|012  (default auto)

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

namespace {

rcdl::NpuCore parseCore(const char* s) {
  std::string c = s ? s : "auto";
  if (c == "0") return rcdl::NpuCore::Core0;
  if (c == "1") return rcdl::NpuCore::Core1;
  if (c == "2") return rcdl::NpuCore::Core2;
  if (c == "01") return rcdl::NpuCore::Core01;
  if (c == "012") return rcdl::NpuCore::Core012;
  return rcdl::NpuCore::Auto;
}

void printAttr(const char* tag, int i, const rknn_tensor_attr& a, rknn_tensor_type io_type,
               std::size_t bytes) {
  std::printf("  %s[%d] %-24s shape=[", tag, i, a.name);
  for (std::uint32_t d = 0; d < a.n_dims; ++d) {
    std::printf("%u%s", a.dims[d], d + 1 < a.n_dims ? "," : "");
  }
  std::printf("] fmt=%s model=%s io=%s", get_format_string(a.fmt), rcdl::dtypeName(a.type),
              rcdl::dtypeName(io_type));
  if (a.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC) {
    std::printf(" zp=%d scale=%g", a.zp, a.scale);
  } else if (a.qnt_type == RKNN_TENSOR_QNT_DFP) {
    std::printf(" fl=%d", a.fl);
  }
  if (a.w_stride) std::printf(" w_stride=%u", a.w_stride);
  std::printf(" bytes=%zu\n", bytes);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s model.rknn [core: auto|0|1|2|01|012]\n", argv[0]);
    return 1;
  }
  try {
    rcdl::Engine::Options opts;
    opts.core = parseCore(argc > 2 ? argv[2] : nullptr);
    rcdl::Engine engine(argv[1], opts);
    std::printf("rcdl %s | librknnrt %s | driver %s\n", RCDL_VERSION_STRING,
                engine.sdkVersion().c_str(), engine.driverVersion().c_str());
    std::printf("model: %s\n", engine.path().c_str());
    for (int i = 0; i < engine.numInputs(); ++i) {
      printAttr("in ", i, engine.inputAttr(i), engine.inputType(i), engine.inputBytes(i));
    }
    for (int i = 0; i < engine.numOutputs(); ++i) {
      printAttr("out", i, engine.outputAttr(i), engine.outputType(i), engine.outputBytes(i));
    }

    for (int i = 0; i < engine.numInputs(); ++i) {
      std::vector<std::uint8_t> zeros(engine.inputPackedBytes(i), 0);
      engine.setInput(i, zeros.data(), zeros.size());
    }
    engine.infer();  // warm-up
    const int iters = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iters; ++k) engine.infer();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::printf("infer: %.3f ms/iter wall (%d iters), npu %.3f ms (RKNN_QUERY_PERF_RUN)\n", ms,
                iters, engine.lastRunMicros() / 1000.0);

    for (int i = 0; i < engine.numOutputs(); ++i) {
      std::vector<float> f = engine.outputAsFloat(i);
      float mn = f.empty() ? 0 : f[0], mx = mn;
      for (float v : f) {
        mn = v < mn ? v : mn;
        mx = v > mx ? v : mx;
      }
      std::printf("  out[%d] n=%zu min=%g max=%g\n", i, f.size(), mn, mx);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
