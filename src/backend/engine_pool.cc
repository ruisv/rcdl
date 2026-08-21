#include "rcdl/backend/engine_pool.h"

#include <string>
#include <utility>

#include "rcdl/core/status.h"

namespace rcdl {

namespace {

/// Core mask for context `i`. RK3588 has three NPU cores, so contexts wrap
/// around when the pool is larger than the core count — two contexts sharing a
/// core still overlap usefully (one decodes its outputs while the other runs),
/// they just do not multiply throughput any further.
NpuCore coreForIndex(int i) {
  switch (i % 3) {
    case 0: return NpuCore::Core0;
    case 1: return NpuCore::Core1;
    default: return NpuCore::Core2;
  }
}

/// Range check for a context index. The message is built only on the failure
/// path — at() sits on the data path, where an allocation per call would be a
/// real cost for a check that never fires.
void checkIndex(int i, int n) {
  if (i >= 0 && i < n) return;
  RCDL_REQUIRE(false, ("EnginePool: context index " + std::to_string(i) + " is out of range [0, " +
                       std::to_string(n) + ")")
                          .c_str());
}

}  // namespace

// ---------------------------------------------------------------------------
// EnginePool
// ---------------------------------------------------------------------------

EnginePool::EnginePool(const std::string& path, int size, bool pin_cores, EngineOptions base) {
  // The loaded model IS context 0 — no separate primary is kept alive beside the
  // pool. A core mask can only be chosen when a context is created (Engine
  // applies it in its constructor and in dup()), so the pin is folded into the
  // load; `base.core` is honoured as given when pinning is off.
  if (pin_cores) base.core = coreForIndex(0);
  engines_.push_back(std::make_unique<Engine>(path, base));
  build(size, pin_cores);
}

EnginePool::EnginePool(Engine& primary, int size, bool pin_cores) {
  // Borrowed primary: the pool leases duplicates only, never `primary` itself.
  // Two reasons, both structural rather than stylistic:
  //   - a core mask is a construction-time choice (Engine applies it in its
  //     constructor and in dup()), so an Engine handed to us fully built cannot
  //     be re-pinned — index 0 must be a duplicate for pin_cores to mean anything;
  //   - the caller keeps its own context free for its own single thread of use
  //     (querying shapes, building a decoder, running a synchronous baseline)
  //     while the pool's workers run, which a shared context would not allow.
  // The duplicate is cheap: it shares the model's weights, only I/O is new.
  engines_.push_back(primary.dup(pin_cores ? coreForIndex(0) : NpuCore::Auto));
  build(size, pin_cores);
}

void EnginePool::build(int size, bool pin_cores) {
  RCDL_REQUIRE(size >= 1,
               ("EnginePool: size must be at least 1, got " + std::to_string(size)).c_str());
  // Both constructors seed context 0 before calling this; duplicating from it
  // gives every other context the same shared weights. (A duplicate is itself a
  // valid parent, so it does not matter that context 0 may be one.)
  Engine& src = *engines_.front();
  for (int i = static_cast<int>(engines_.size()); i < size; ++i) {
    engines_.push_back(src.dup(pin_cores ? coreForIndex(i) : NpuCore::Auto));
  }
  free_.assign(engines_.size(), 1);
  in_use_ = 0;
}

Engine& EnginePool::at(int i) {
  checkIndex(i, size());
  return *engines_[static_cast<std::size_t>(i)];
}

const Engine& EnginePool::at(int i) const {
  checkIndex(i, size());
  return *engines_[static_cast<std::size_t>(i)];
}

EnginePool::Lease EnginePool::acquire() {
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [this] { return in_use_ < static_cast<int>(engines_.size()); });
  for (std::size_t i = 0; i < free_.size(); ++i) {
    if (free_[i]) {
      free_[i] = 0;
      ++in_use_;
      return Lease(this, static_cast<int>(i));
    }
  }
  // in_use_ and free_ are updated together under mu_, so the predicate above
  // guarantees the scan finds one. Reaching here would mean they disagree.
  RCDL_REQUIRE(false, "EnginePool: free-list and in-use count disagree");
  return Lease();
}

EnginePool::Lease EnginePool::tryAcquire() {
  std::lock_guard<std::mutex> lk(mu_);
  for (std::size_t i = 0; i < free_.size(); ++i) {
    if (free_[i]) {
      free_[i] = 0;
      ++in_use_;
      return Lease(this, static_cast<int>(i));
    }
  }
  return Lease();
}

int EnginePool::inUse() const {
  std::lock_guard<std::mutex> lk(mu_);
  return in_use_;
}

void EnginePool::give_back(int index) noexcept {
  {
    std::lock_guard<std::mutex> lk(mu_);
    if (index < 0 || index >= static_cast<int>(free_.size())) return;  // never throw from a dtor
    if (free_[static_cast<std::size_t>(index)]) return;                // already returned
    free_[static_cast<std::size_t>(index)] = 1;
    --in_use_;
  }
  // Notify outside the lock: a waiter woken here would immediately block on mu_
  // if we still held it.
  cv_.notify_one();
}

// ---------------------------------------------------------------------------
// EnginePool::Lease
// ---------------------------------------------------------------------------

EnginePool::Lease::~Lease() { release(); }

EnginePool::Lease::Lease(Lease&& other) noexcept : pool_(other.pool_), index_(other.index_) {
  other.pool_ = nullptr;
  other.index_ = -1;
}

EnginePool::Lease& EnginePool::Lease::operator=(Lease&& other) noexcept {
  if (this != &other) {
    release();  // whatever we held goes back to the pool before we take another
    pool_ = other.pool_;
    index_ = other.index_;
    other.pool_ = nullptr;
    other.index_ = -1;
  }
  return *this;
}

Engine& EnginePool::Lease::engine() const {
  RCDL_REQUIRE(pool_ != nullptr, "EnginePool::Lease::engine: the lease is empty");
  return pool_->at(index_);
}

void EnginePool::Lease::release() noexcept {
  if (pool_ != nullptr) {
    // Returning the context is what makes a lease exception-safe: a worker that
    // throws mid-inference unwinds through ~Lease and the context stays in
    // circulation instead of leaking out of the pool.
    pool_->give_back(index_);
    pool_ = nullptr;
    index_ = -1;
  }
}

}  // namespace rcdl
