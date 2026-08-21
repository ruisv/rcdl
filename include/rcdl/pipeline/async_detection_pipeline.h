#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rcdl/media/video_frame.h"
#include "rcdl/pipeline/detection_pipeline.h"
#include "rcdl/preproc/image.h"
#include "rcdl/tasks/detection.h"

namespace rcdl {

class Engine;      // backend/engine.h
class EnginePool;  // backend/engine_pool.h

/// Threading and pool settings for AsyncDetectionPipeline.
struct AsyncConfig {
  /// NPU contexts, and therefore worker threads. 3 matches RK3588's core count;
  /// use 1 to get an ordinary pipeline with a background infer stage, or 2 on
  /// RK3576.
  int workers = 3;
  /// Pin worker i to NPU core i. This is what turns N contexts into N times the
  /// throughput on a small model — leaving it to the driver's Auto scheduling
  /// measurably does not.
  bool pin_cores = true;
  /// Results buffered while waiting for an out-of-order predecessor. 0 =>
  /// workers * 4. Reached only if one frame is much slower than its neighbours.
  int reorder_depth = 0;
};

/// Streaming detector that overlaps preprocessing, inference and decoding across
/// several NPU contexts, and returns results in SUBMISSION order.
///
/// SHAPE OF THE PIPELINE, and why it is this shape:
///
///   caller thread : acquire a free context -> RGA letterbox the source frame
///                   straight into THAT context's input dma-buf -> hand it off
///   worker i      : engine[i].infer() -> decode + NMS -> park the result under
///                   its sequence number -> return the context to the pool
///
/// The letterbox runs on the CALLING thread on purpose. It is an RGA op — the
/// CPU only submits it and waits — and doing it in the caller means the source
/// frame is fully consumed by the time submit() returns. A worker-side
/// letterbox would need the source to stay alive and unmodified until some
/// unspecified later moment, which for a decoded VideoFrame means holding a
/// buffer the decoder wants back. So: no copy, no lifetime rule for the caller
/// to get wrong, and the frame can be recycled immediately.
///
/// Throughput is then bounded by max(letterbox, (infer + decode) / workers),
/// against (letterbox + infer + decode) for the synchronous pipeline.
///
/// Typical use:
///
///   AsyncDetectionPipeline p(engine, cfg, {.workers = 3});
///   std::vector<Detection> dets;
///   int i = 0;
///   for (const auto& frame : stream) {
///     p.submit(frame.view());        // blocks when every context is busy
///     if (i++ >= 3) p.next(dets);    // keep it full, drain in order
///   }
///   p.finish();
///   while (p.next(dets)) { /* the last in-flight results */ }
class AsyncDetectionPipeline {
 public:
  /// Duplicate `engine` into a pool of `async.workers` contexts and start them.
  /// The Engine must outlive the pipeline.
  AsyncDetectionPipeline(Engine& engine, PipelineConfig cfg, AsyncConfig async = AsyncConfig());
  /// Load the model once and own the whole pool.
  AsyncDetectionPipeline(const std::string& model_path, PipelineConfig cfg,
                         AsyncConfig async = AsyncConfig());
  ~AsyncDetectionPipeline();

  AsyncDetectionPipeline(const AsyncDetectionPipeline&) = delete;
  AsyncDetectionPipeline& operator=(const AsyncDetectionPipeline&) = delete;

  /// Letterbox `src` into a free context and queue it. Blocks while every
  /// context is in flight (back-pressure). Returns false once finish()ed.
  ///
  /// `src` is fully consumed before this returns — recycle it freely.
  bool submit(const ImageView& src);
  /// Convenience for an interleaved, row-contiguous host BGR image.
  bool submit(const std::uint8_t* bgr, int width, int height);
  /// Zero-copy submit of a decoded frame: RGA reads the VPU's dma-buf directly.
  bool submit(const VideoFrame& frame);

  /// The three steps submit() is made of, for a caller that must not hold its
  /// source frame while waiting for capacity.
  ///
  /// A video stage that owns a recycled frame and blocks inside submit() keeps
  /// that frame out of circulation, which can starve its own producer and
  /// deadlock the ring. Such a caller waits for a slot FIRST (holding nothing),
  /// then takes a frame, letterboxes it, releases the frame, and commits:
  ///
  ///   int slot = acquireSlot();                  // blocks; -1 once finished
  ///   if (slot < 0) return;
  ///   auto lb = letterboxIntoSlot(slot, frame.view());
  ///   frame.reset();                             // give the buffer back now
  ///   commitSlot(slot, lb);
  ///
  /// Every acquireSlot() >= 0 must be paired with exactly one commitSlot() or
  /// releaseSlot(); the other two throw rcdl::Error on anything else.
  int acquireSlot();
  LetterboxInfo letterboxIntoSlot(int slot, const ImageView& src);
  bool commitSlot(int slot, const LetterboxInfo& lb);
  void releaseSlot(int slot);

  /// Pop the next result in submission order, blocking until it is ready.
  /// Returns false once the pipeline is finished AND drained.
  bool next(std::vector<Detection>& out);
  /// Non-blocking next(): true if a result was ready, false immediately if not.
  bool tryNext(std::vector<Detection>& out);
  /// Letterbox geometry of the result the last next()/tryNext() returned.
  const LetterboxInfo& lastLetterbox() const noexcept;

  /// No more frames will be submitted; after the in-flight ones drain, next()
  /// returns false. Idempotent; the destructor calls it.
  void finish();

  const PipelineConfig& config() const noexcept { return cfg_; }
  DetectHead head() const noexcept { return cfg_.head; }
  int workers() const noexcept;
  /// Frames submitted but not yet drained.
  int inFlight() const;

  /// Per-stage SERVICE timing across the pipeline's lifetime. Each stage is
  /// timed on the thread that ran it, so the fields sum to MORE than wall time —
  /// that is the point: the largest per-frame stage is what bounds throughput.
  /// Read after finish() + full drain for a settled value.
  StageProfile profile() const;

 private:
  PipelineConfig cfg_;  ///< resolved at construction (mirrors impl_'s copy)
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rcdl
