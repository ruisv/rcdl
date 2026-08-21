#include "rcdl/pipeline/async_detection_pipeline.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rcdl/backend/engine.h"
#include "rcdl/backend/engine_pool.h"
#include "rcdl/core/status.h"
#include "rcdl/preproc/letterbox.h"

namespace rcdl {

namespace {

inline double msBetween(std::chrono::steady_clock::time_point a,
                        std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

/// Same validation the synchronous pipeline does, on the same shared helpers:
/// one image input, uint8 NHWC, then head/canvas resolution. Every context in
/// the pool is a duplicate of one model, so checking it once is enough.
PipelineConfig prepare(Engine& engine, PipelineConfig cfg) {
  RCDL_REQUIRE(engine.numInputs() == 1,
               ("AsyncDetectionPipeline: single-input models only, this one has " +
                std::to_string(engine.numInputs()) + " inputs")
                   .c_str());
  requireImageInputModel(engine);
  return resolveDetectionConfig(engine, std::move(cfg));
}

int clampWorkers(const AsyncConfig& async) {
  return async.workers > 0 ? async.workers : 1;
}

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
//
// LOCKING, in one place so it can be checked by reading:
//
//   mu       guards every piece of pipeline state: each Slot's state/seq/lb/lease,
//            the reorder map, the sequence counters and the finished/stopping
//            flags. It is NEVER held across an RGA letterbox, an infer() or a
//            postprocess() — those run on state the holder owns exclusively (a
//            slot in kHeld belongs to its caller, a slot in kQueued to its
//            worker), so no lock is needed to protect them.
//   slot->cv worker i sleeps here for work on ITS slot; per-slot rather than
//            shared so committing a frame wakes exactly one thread.
//   cv_ready next() sleeps here for the result it is owed.
//   cv_room  submitters sleep here while the reorder buffer is full (workers
//            never do — see workerLoop()).
//   prof_mu  guards the StageProfile alone — preproc is accumulated by caller
//            threads, infer/postproc by workers, so it has several writers and a
//            separate (never contended with the data path) lock.
//
// Lock ORDER is always mu -> the pool's own mutex (a worker returns its lease
// while holding mu). Callers take the pool's mutex first (acquire()) and mu
// afterwards, never both at once, so the two orders cannot form a cycle.

struct AsyncDetectionPipeline::Impl {
  enum SlotState { kFree, kHeld, kQueued };

  /// One finished frame, waiting in the reorder buffer for its turn.
  struct Result {
    std::vector<Detection> dets;
    LetterboxInfo lb;
    bool ok = true;
    std::string error;  ///< set when the worker caught an exception
  };

  /// Everything worker i owns: its NPU context (via the lease the caller hands
  /// over), the letterbox destination inside that context's input tensor, and
  /// its own decoder.
  struct Slot {
    EnginePool::Lease lease;   ///< taken by acquireSlot(), returned by the worker
    Engine* engine = nullptr;  ///< pool context i — this worker's, for its whole life
    ImageView input_view;      ///< engine i's input tensor as a letterbox destination
    /// Per-worker, NOT shared: a HeadDecoder binds one Engine& and reads that
    /// engine's output buffers, so worker i must decode through the decoder
    /// built on engine i or it would read another inference's outputs.
    std::unique_ptr<HeadDecoder> decoder;
    LetterboxInfo lb;
    std::uint64_t seq = 0;
    int state = kFree;
    std::condition_variable cv;
  };

  /// Hand a drained result to the caller, or surface on the caller's thread —
  /// in submission order — whatever the worker caught. The message is built
  /// only on the failure path, so the hot path allocates nothing here.
  static bool deliver(Result& r, std::vector<Detection>& out) {
    if (!r.ok) {
      RCDL_REQUIRE(false,
                   ("AsyncDetectionPipeline: a worker failed on this frame: " + r.error).c_str());
    }
    out = std::move(r.dets);
    return true;
  }

  PipelineConfig cfg;
  std::size_t reorder_depth;

  // Declaration order is destruction order reversed, and it matters here: the
  // slots hold leases and decoders that reference the pool's Engines, so slots
  // must be destroyed BEFORE the pool. (The threads are joined explicitly in the
  // destructor body, before any member is destroyed.)
  std::unique_ptr<EnginePool> pool;
  std::vector<std::unique_ptr<Slot>> slots;
  std::vector<std::thread> threads;

  mutable std::mutex mu;
  std::condition_variable cv_ready;  ///< a result was parked / the pipeline finished
  std::condition_variable cv_room;   ///< a result was drained / finished: room to submit
  std::map<std::uint64_t, Result> pending;  ///< reorder buffer, keyed on submission order
  std::uint64_t next_in = 0;   ///< sequence number the next commit will take
  std::uint64_t next_out = 0;  ///< sequence number next() is waiting for
  bool finished = false;       ///< no more frames will be submitted
  bool stopping = false;       ///< destructor: abandon everything and exit
  LetterboxInfo last_lb;       ///< geometry of the last delivered result

  mutable std::mutex prof_mu;
  StageProfile prof;

  Impl(std::unique_ptr<EnginePool> p, PipelineConfig resolved, const AsyncConfig& async)
      : cfg(std::move(resolved)), pool(std::move(p)) {
    const int n = pool->size();
    // Bounded on purpose: results park here until the caller drains them, and an
    // unbounded buffer would let a caller that never calls next() grow the heap
    // without limit. Out-of-order occupancy alone can never exceed `workers`
    // (that is how many frames are in flight at once), so with the default of
    // workers * 4 the bound is reached only when the caller falls behind — and
    // then it holds up the SUBMITTER (acquireSlot()), never a worker.
    reorder_depth = static_cast<std::size_t>(
        async.reorder_depth > 0 ? async.reorder_depth : n * 4);

    slots.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      auto s = std::make_unique<Slot>();
      Engine& e = pool->at(i);
      s->engine = &e;
      // Resolved once: the fd + mapping + row stride of this context's input
      // tensor. Every letterbox writes straight into it, so the data path
      // allocates nothing and queries nothing.
      s->input_view = engineInputView(e, 0, cfg.model_input);
      RCDL_REQUIRE(s->input_view.width == cfg.input_w && s->input_view.height == cfg.input_h,
                   ("AsyncDetectionPipeline: configured canvas " + std::to_string(cfg.input_w) +
                    "x" + std::to_string(cfg.input_h) + " does not match the model input tensor " +
                    s->input_view.describe())
                       .c_str());
      s->decoder = std::make_unique<HeadDecoder>(e, cfg);
      slots.push_back(std::move(s));
    }
    // Threads start only once every slot is fully built: a worker touches
    // slots[i] the moment it wakes.
    threads.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) threads.emplace_back([this, i] { workerLoop(i); });
  }

  ~Impl() { shutdown(); }

  int workerCount() const { return static_cast<int>(slots.size()); }

  /// Range- and state-check a slot handle from the caller, and return the slot.
  /// Both messages are built only when the check fails: this runs on the submit
  /// path, once per frame per call, and must not allocate to say "fine".
  Slot& checkedSlot(std::unique_lock<std::mutex>& lk, int slot, const char* fn) {
    (void)lk;  // documents that the caller holds mu
    if (slot < 0 || slot >= workerCount()) {
      RCDL_REQUIRE(false, (std::string("AsyncDetectionPipeline::") + fn + ": slot " +
                           std::to_string(slot) + " is out of range [0, " +
                           std::to_string(workerCount()) +
                           ") — acquireSlot() returns -1 once the pipeline is finished")
                              .c_str());
    }
    Slot& s = *slots[static_cast<std::size_t>(slot)];
    if (s.state != kHeld) {
      RCDL_REQUIRE(false, (std::string("AsyncDetectionPipeline::") + fn + ": slot " +
                           std::to_string(slot) +
                           " is not held — every acquireSlot() pairs with exactly one "
                           "commitSlot() or releaseSlot()")
                              .c_str());
    }
    return s;
  }

  void addPreprocTime(double ms) {
    std::lock_guard<std::mutex> lk(prof_mu);
    prof.preproc_ms += ms;
  }

  void addWorkerTime(double infer_ms, double post_ms) {
    std::lock_guard<std::mutex> lk(prof_mu);
    prof.infer_ms += infer_ms;
    prof.postproc_ms += post_ms;
    ++prof.frames;
  }

  /// Worker i: its context, its decoder, one frame at a time.
  void workerLoop(int i) {
    Slot& s = *slots[static_cast<std::size_t>(i)];
    for (;;) {
      std::uint64_t seq = 0;
      LetterboxInfo lb;
      {
        std::unique_lock<std::mutex> lk(mu);
        s.cv.wait(lk, [&] { return s.state == kQueued || stopping; });
        if (stopping) {
          // Give the context back even on the way out, so a caller blocked in
          // acquireSlot() wakes, sees `stopping`, and returns -1 instead of
          // waiting on a context no worker will ever release.
          //
          // Only if the slot is ours, though. A slot in kHeld belongs to a
          // caller that is between acquireSlot() and commitSlot() — i.e. very
          // possibly inside letterboxIntoSlot(), writing into that context's
          // input dma-buf right now. Releasing its lease here would hand the
          // context back to the pool mid-RGA-op and make the caller's own
          // commitSlot()/releaseSlot() throw "slot is not held". Leave it and
          // let its owner unwind.
          if (s.state != kHeld) {
            s.state = kFree;
            s.lease.release();
          }
          return;
        }
        seq = s.seq;
        lb = s.lb;
      }

      // --- no lock held from here to the parking step ---------------------
      // The context is leased to this worker, so infer() and the decode below
      // touch state nobody else may touch. Holding mu across them would
      // serialise the workers and defeat the whole pipeline.
      Result r;
      r.lb = lb;
      try {
        const auto t0 = std::chrono::steady_clock::now();
        s.engine->infer();
        const auto t1 = std::chrono::steady_clock::now();
        r.dets = s.decoder->postprocess(lb);
        const auto t2 = std::chrono::steady_clock::now();
        addWorkerTime(msBetween(t0, t1), msBetween(t1, t2));
      } catch (const std::exception& e) {
        // An exception must not escape a std::thread (that is std::terminate),
        // and it must not be swallowed either: carry it in the result and let
        // next() rethrow it on the caller's thread, in submission order.
        r.ok = false;
        r.error = e.what();
      }

      EnginePool::Lease done;
      {
        std::lock_guard<std::mutex> lk(mu);
        // A worker NEVER waits here. Parking is unconditional so that every
        // thread in this pipeline can only ever be blocked on work it will
        // itself finish: that is what makes finish() + join() terminate, and it
        // is what keeps a single-threaded caller — which cannot drain while it
        // is inside submit() — from deadlocking against its own workers. The
        // buffer is kept bounded at the other end instead, in acquireSlot().
        pending.emplace(seq, std::move(r));
        s.state = kFree;
        done = std::move(s.lease);  // returned below, outside mu
        cv_ready.notify_all();
      }
      // Returning the context wakes a caller blocked in acquireSlot(); do it
      // outside mu so that caller does not immediately block on us.
      done.release();
    }
  }

  /// finish() + join. Idempotent, and safe to call from the destructor.
  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(mu);
      finished = true;
      // `stopping` is the harder flag: `finished` only ends the stream (the
      // in-flight frames still drain), while this releases every waiter
      // unconditionally — a worker waiting for a frame that will never come, a
      // consumer in next(), a submitter waiting for room nobody will free.
      // Without it the join below could wait forever.
      stopping = true;
    }
    for (auto& s : slots) s->cv.notify_all();
    cv_room.notify_all();
    cv_ready.notify_all();
    for (std::thread& t : threads) {
      if (t.joinable()) t.join();
    }
  }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

AsyncDetectionPipeline::AsyncDetectionPipeline(Engine& engine, PipelineConfig cfg,
                                               AsyncConfig async) {
  cfg_ = prepare(engine, std::move(cfg));
  auto pool = std::make_unique<EnginePool>(engine, clampWorkers(async), async.pin_cores);
  impl_ = std::make_unique<Impl>(std::move(pool), cfg_, async);
}

AsyncDetectionPipeline::AsyncDetectionPipeline(const std::string& model_path, PipelineConfig cfg,
                                               AsyncConfig async) {
  // The pool loads the model, so the config is resolved against one of its
  // contexts — they are all duplicates of the same weights and report the same
  // tensor attributes.
  auto pool = std::make_unique<EnginePool>(model_path, clampWorkers(async), async.pin_cores);
  cfg_ = prepare(pool->at(0), std::move(cfg));
  impl_ = std::make_unique<Impl>(std::move(pool), cfg_, async);
}

AsyncDetectionPipeline::~AsyncDetectionPipeline() {
  // ~Impl() -> shutdown(): finish, release every waiter, join. Results still in
  // the reorder buffer are discarded — a caller that wants them drains before
  // destroying the pipeline.
  impl_.reset();
}

// ---------------------------------------------------------------------------
// Submission
// ---------------------------------------------------------------------------

int AsyncDetectionPipeline::acquireSlot() {
  {
    std::unique_lock<std::mutex> lk(impl_->mu);
    // Back-pressure #1: undrained results. Waiting HERE, before taking a
    // context and with no source frame in hand, is what makes the bound safe —
    // the thread that has to drain is the one we hold up, and no context is
    // parked out of circulation while we wait. (The check is deliberately
    // racy-by-a-few when several threads submit: the bound is a memory guard,
    // not an invariant.)
    impl_->cv_room.wait(lk, [this] {
      return impl_->stopping || impl_->finished ||
             impl_->pending.size() < impl_->reorder_depth;
    });
    // Also the cheap pre-check for a finished pipeline, so it does not queue up
    // behind the contexts still in flight just to be told "no" afterwards.
    if (impl_->finished || impl_->stopping) return -1;
  }
  // Back-pressure #2: contexts. Blocks until a worker returns one, and always
  // terminates — a leased context is held by a worker that releases it as soon
  // as its frame is parked, and parking never waits. At shutdown the workers
  // return their leases on the way out for exactly this reason.
  EnginePool::Lease lease = impl_->pool->acquire();
  const int i = lease.index();

  std::lock_guard<std::mutex> lk(impl_->mu);
  if (impl_->finished || impl_->stopping) return -1;  // ~Lease puts the context back
  Impl::Slot& s = *impl_->slots[static_cast<std::size_t>(i)];
  s.state = Impl::kHeld;
  s.lease = std::move(lease);
  return i;
}

LetterboxInfo AsyncDetectionPipeline::letterboxIntoSlot(int slot, const ImageView& src) {
  if (!src.valid()) {
    // describe() allocates, so it is called only on the failure path.
    RCDL_REQUIRE(false, ("AsyncDetectionPipeline::letterboxIntoSlot: invalid source view: " +
                         src.describe())
                            .c_str());
  }
  ImageView dst;
  {
    std::unique_lock<std::mutex> lk(impl_->mu);
    dst = impl_->checkedSlot(lk, slot, "letterboxIntoSlot").input_view;
  }
  // The RGA op runs with NO lock held. The slot is held by this caller and the
  // destination is that context's own input tensor, so nothing else can touch
  // it; taking mu here would serialise every submitting thread behind the
  // hardware.
  const auto t0 = std::chrono::steady_clock::now();
  const LetterboxInfo lb =
      letterbox(dst, src, cfg_.pad_value, cfg_.backend, cfg_.yuv_range, nullptr);
  impl_->addPreprocTime(msBetween(t0, std::chrono::steady_clock::now()));
  return lb;
}

bool AsyncDetectionPipeline::commitSlot(int slot, const LetterboxInfo& lb) {
  EnginePool::Lease dropped;
  {
    std::unique_lock<std::mutex> lk(impl_->mu);
    Impl::Slot& s = impl_->checkedSlot(lk, slot, "commitSlot");
    if (impl_->finished || impl_->stopping) {
      // finish() won the race: drop the frame rather than extend the stream, and
      // give the context back so the drain can complete.
      s.state = Impl::kFree;
      dropped = std::move(s.lease);
      return false;
    }
    // The sequence number is assigned HERE, not in acquireSlot(): a slot that is
    // acquired and then released (or rejected above) must not leave a hole that
    // next() would wait for forever.
    s.lb = lb;
    s.seq = impl_->next_in++;
    s.state = Impl::kQueued;
    s.cv.notify_one();
  }
  return true;
}

void AsyncDetectionPipeline::releaseSlot(int slot) {
  EnginePool::Lease dropped;
  {
    std::unique_lock<std::mutex> lk(impl_->mu);
    Impl::Slot& s = impl_->checkedSlot(lk, slot, "releaseSlot");
    s.state = Impl::kFree;
    dropped = std::move(s.lease);
  }
  // ~Lease returns the context outside mu.
}

bool AsyncDetectionPipeline::submit(const ImageView& src) {
  const int slot = acquireSlot();
  if (slot < 0) return false;
  LetterboxInfo lb;
  try {
    lb = letterboxIntoSlot(slot, src);
  } catch (...) {
    // Keep the acquire/commit pairing intact: a preproc failure must not take a
    // context out of circulation.
    releaseSlot(slot);
    throw;
  }
  return commitSlot(slot, lb);
}

bool AsyncDetectionPipeline::submit(const std::uint8_t* bgr, int width, int height) {
  RCDL_REQUIRE(bgr != nullptr && width > 0 && height > 0,
               "AsyncDetectionPipeline::submit: invalid BGR frame");
  // ImageView is a non-owning descriptor and preproc only READS the source, so
  // this const_cast describes the buffer, it does not write through it.
  return submit(hostView(const_cast<std::uint8_t*>(bgr), width, height, PixelFormat::BGR888));
}

bool AsyncDetectionPipeline::submit(const VideoFrame& frame) {
  RCDL_REQUIRE(frame.valid(), "AsyncDetectionPipeline::submit: invalid VideoFrame");
  // Zero copy: the view carries the decoder's dma-buf fd, which RGA reads
  // directly. The frame is fully consumed when this returns.
  return submit(frame.view());
}

// ---------------------------------------------------------------------------
// Draining
// ---------------------------------------------------------------------------

bool AsyncDetectionPipeline::next(std::vector<Detection>& out) {
  Impl::Result r;
  {
    std::unique_lock<std::mutex> lk(impl_->mu);
    impl_->cv_ready.wait(lk, [this] {
      return impl_->pending.count(impl_->next_out) != 0 || impl_->stopping ||
             (impl_->finished && impl_->next_out == impl_->next_in);
    });
    auto it = impl_->pending.find(impl_->next_out);
    if (it == impl_->pending.end()) {
      // Either finished and fully drained (next_out == next_in: every committed
      // frame has been delivered), or shutting down.
      return false;
    }
    r = std::move(it->second);
    impl_->pending.erase(it);
    ++impl_->next_out;
    impl_->last_lb = r.lb;
  }
  impl_->cv_room.notify_all();  // a submitter may be waiting for the room we just freed
  return Impl::deliver(r, out);
}

bool AsyncDetectionPipeline::tryNext(std::vector<Detection>& out) {
  Impl::Result r;
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    auto it = impl_->pending.find(impl_->next_out);
    if (it == impl_->pending.end()) return false;
    r = std::move(it->second);
    impl_->pending.erase(it);
    ++impl_->next_out;
    impl_->last_lb = r.lb;
  }
  impl_->cv_room.notify_all();
  return Impl::deliver(r, out);
}

const LetterboxInfo& AsyncDetectionPipeline::lastLetterbox() const noexcept {
  // Written under mu by next()/tryNext(); read here without one, on the same
  // single-consumer contract the synchronous pipeline has.
  return impl_->last_lb;
}

void AsyncDetectionPipeline::finish() {
  {
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (impl_->finished) return;  // idempotent, from any thread
    impl_->finished = true;
  }
  // Wake next(): with nothing left in flight its "finished and drained"
  // condition is now true and it must return false rather than sleep on. And
  // wake acquireSlot(): a submitter waiting for reorder room must be told the
  // stream is over instead of waiting for a drain that may never come.
  impl_->cv_ready.notify_all();
  impl_->cv_room.notify_all();
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

int AsyncDetectionPipeline::workers() const noexcept { return impl_->workerCount(); }

int AsyncDetectionPipeline::inFlight() const {
  std::lock_guard<std::mutex> lk(impl_->mu);
  return static_cast<int>(impl_->next_in - impl_->next_out);
}

StageProfile AsyncDetectionPipeline::profile() const {
  std::lock_guard<std::mutex> lk(impl_->prof_mu);
  return impl_->prof;
}

}  // namespace rcdl
