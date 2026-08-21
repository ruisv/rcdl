#pragma once

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "rcdl/backend/engine.h"

namespace rcdl {

/// A set of contexts over ONE model, so several inferences run at once.
///
/// RK3588 has three NPU cores. A small model does not get faster by spreading a
/// single inference across them — the combined core mask only pays off for a
/// model big enough to keep three cores busy. What DOES scale is running three
/// independent contexts, one pinned per core: measured on this project's
/// ResNet-18 int8, one core gives ~248 fps and three pinned contexts give ~707.
///
/// `rknn_dup_context` is what makes that cheap: the duplicates SHARE the model's
/// weights (no second copy in memory) but get their own I/O tensors, so each
/// worker has its own input dma-buf to letterbox into and its own outputs to
/// decode from. That is exactly the per-worker state an async pipeline needs.
///
/// Leases are the ownership rule: an Engine handed out must not be touched by
/// anyone else until the lease is released, because a context has a single
/// thread of use.
class EnginePool {
 public:
  /// Load `path` once and duplicate it to `size` contexts. With `pin_cores`,
  /// context i is pinned to NPU core i (wrapping around when size exceeds the
  /// core count); without it every context runs with NpuCore::Auto and the
  /// driver schedules them.
  EnginePool(const std::string& path, int size = 3, bool pin_cores = true,
             EngineOptions base = EngineOptions());

  /// Duplicate an existing Engine into `size` NEW contexts. `primary` itself is
  /// never leased — it stays free for its owner's own single thread of use, and
  /// a fully-built Engine cannot be re-pinned to another core anyway — so this
  /// leaves `size + 1` live RKNN contexts in total. Budget NPU I/O memory
  /// accordingly. `primary` must outlive the pool.
  EnginePool(Engine& primary, int size = 3, bool pin_cores = true);

  EnginePool(const EnginePool&) = delete;
  EnginePool& operator=(const EnginePool&) = delete;

  int size() const noexcept { return static_cast<int>(engines_.size()); }
  /// Direct access for setup (querying shapes, building decoders per worker).
  /// NOT a substitute for lease() on the data path.
  Engine& at(int i);
  const Engine& at(int i) const;

  /// Exclusive use of one context. Returns it to the pool on destruction, so a
  /// worker that throws mid-inference cannot leak a context out of circulation.
  class Lease {
   public:
    Lease() = default;
    ~Lease();
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    Lease(Lease&& other) noexcept;
    Lease& operator=(Lease&& other) noexcept;

    bool valid() const noexcept { return pool_ != nullptr; }
    /// The leased context. Throws rcdl::Error on an empty lease.
    Engine& engine() const;
    /// Its index in the pool — also the worker index the pipelines key on.
    int index() const noexcept { return index_; }
    void release() noexcept;

   private:
    friend class EnginePool;
    Lease(EnginePool* pool, int index) : pool_(pool), index_(index) {}
    EnginePool* pool_ = nullptr;
    int index_ = -1;
  };

  /// Take a free context, blocking until one is available.
  Lease acquire();
  /// Take a free context if one is available right now; otherwise an invalid
  /// lease. Never blocks.
  Lease tryAcquire();
  /// Contexts currently leased out.
  int inUse() const;

 private:
  friend class Lease;
  void give_back(int index) noexcept;
  void build(int size, bool pin_cores);

  /// Every context the pool hands out, owned. When the pool loaded the model
  /// itself, index 0 IS that Engine (a core mask can only be chosen at
  /// construction, so the loaded context is pinned from the start and there is
  /// no reason to hold a separate un-leasable parent). When the pool was given
  /// an existing Engine, all of these are duplicates of it and the caller's
  /// Engine stays free for its own single thread of use.
  std::vector<std::unique_ptr<Engine>> engines_;
  std::vector<char> free_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  int in_use_ = 0;
};

}  // namespace rcdl
