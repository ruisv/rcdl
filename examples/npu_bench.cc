// NPU throughput bench: one model, N contexts (rknn_dup_context) pinned to
// cores, each thread running back-to-back inferences for a fixed duration.
//
//   ./npu_bench model.rknn [seconds=5] [cores=0,1,2]
//
// cores: comma list of per-thread core masks (auto|0|1|2|01|012). One thread per
// entry. "0,1,2" = three single-core contexts (max throughput on RK3588 for small
// models); "012" = one context spread across all cores (lowest latency for big
// models); "auto,auto,auto" lets the runtime schedule.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rcdl/rcdl.h"

namespace {
rcdl::NpuCore parseCore(const std::string& c) {
  if (c == "0") return rcdl::NpuCore::Core0;
  if (c == "1") return rcdl::NpuCore::Core1;
  if (c == "2") return rcdl::NpuCore::Core2;
  if (c == "01") return rcdl::NpuCore::Core01;
  if (c == "012") return rcdl::NpuCore::Core012;
  return rcdl::NpuCore::Auto;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s model.rknn [seconds] [cores e.g. 0,1,2]\n", argv[0]);
    return 1;
  }
  const double seconds = argc > 2 ? std::atof(argv[2]) : 5.0;
  std::vector<rcdl::NpuCore> cores;
  {
    std::stringstream ss(argc > 3 ? argv[3] : "0,1,2");
    std::string item;
    while (std::getline(ss, item, ',')) cores.push_back(parseCore(item));
  }
  try {
    rcdl::Engine::Options opts;
    opts.core = cores[0];
    rcdl::Engine base(argv[1], opts);
    std::vector<std::unique_ptr<rcdl::Engine>> engines;
    for (std::size_t k = 1; k < cores.size(); ++k) engines.push_back(base.dup(cores[k]));

    std::vector<rcdl::Engine*> all;
    all.push_back(&base);
    for (auto& e : engines) all.push_back(e.get());
    for (auto* e : all) {
      for (int i = 0; i < e->numInputs(); ++i) {
        std::vector<std::uint8_t> zeros(e->inputPackedBytes(i), 0);
        e->setInput(i, zeros.data(), zeros.size());
      }
      e->infer();
    }

    std::atomic<bool> stop{false};
    std::vector<long> counts(all.size(), 0);
    std::vector<std::thread> threads;
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t k = 0; k < all.size(); ++k) {
      threads.emplace_back([&, k] {
        while (!stop.load(std::memory_order_relaxed)) {
          all[k]->infer();
          ++counts[k];
        }
      });
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    stop.store(true);
    for (auto& t : threads) t.join();
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    long total = 0;
    for (std::size_t k = 0; k < all.size(); ++k) {
      std::printf("ctx %zu (core mask %d): %.1f fps\n", k, static_cast<int>(cores[k]),
                  counts[k] / el);
      total += counts[k];
    }
    std::printf("total: %.1f fps over %.1fs with %zu context(s)\n", total / el, el, all.size());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
