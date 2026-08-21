#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rcdl/media/video_codec.h"
#include "rcdl/pipeline/async_detection_pipeline.h"
#include "rcdl/pipeline/detection_pipeline.h"
#include "rcdl/tasks/detection.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Threading and buffering for AsyncVideoDetectionPipeline, on top of the
/// detector's own AsyncConfig.
struct VideoAsyncConfig {
  /// NPU contexts / infer workers, reorder depth, core pinning.
  AsyncConfig async;
  /// Decoded frames allowed to sit between the VPU and RGA stages. Each one is
  /// a decoder buffer held out of circulation, so this is not free: 2 is enough
  /// to keep the letterbox fed while the decoder works on the next picture, and
  /// the decoder's pool is sized to cover it (see VideoDecConfig::extra_buffers).
  int queue_depth = 2;
  /// Poll granularity for the drain thread while the stream is still live. It
  /// bounds only how quickly the thread notices end-of-stream or shutdown, not
  /// throughput.
  int drain_poll_ms = 10;
};

/// Compressed video in, detections out, with every stage on the unit that owns
/// it and no frame ever copied:
///
///   caller  : submit(bytes)              -> MPP parser (VPU)
///   drain   : VideoDecoder::receive()    -> NV12 in a dma-buf
///   preproc : RGA letterbox from that fd -> straight into an NPU input tensor
///   workers : rknn_run x N contexts      -> head decode + NMS  (AsyncDetectionPipeline)
///   caller  : next()                     -> detections, in DECODE order
///
/// The four stages run concurrently on different frames, so steady-state
/// throughput approaches 1 / max(stage) rather than the sum of the stages that
/// `video_det_demo` measures synchronously. This is the class form of that
/// example's path minus the overlay/encode tail: a caller that only pumps bytes
/// — including a Python one — gets the C++ ceiling instead of being throttled by
/// its own orchestration.
///
/// Feed granularity: submit() takes ARBITRARY chunks of a raw elementary stream.
/// MPP's parser splits them into access units (VideoDecConfig::split_parse), so
/// the caller never looks for start codes. A container (.mp4/.mkv) must be
/// demuxed first — `ffmpeg -i in.mp4 -c copy -f h264 -` is the usual source.
///
/// DRAIN WHILE YOU FEED, and note that submit() will not do it for you. Every
/// queue here is bounded — the decoder's input, the frame queue, the NPU
/// contexts, the detector's reorder buffer — and the last one is emptied only
/// by next(). A single-threaded caller that waits indefinitely inside submit()
/// therefore deadlocks the pipeline against itself: the decoder will not take
/// more input until frames move, frames do not move until a context is free, a
/// context is not free until the reorder buffer has room, and only the caller
/// can make that room. So back-pressure comes back as `false` from submit()
/// rather than as a wait, and the loop below is the shape that always works.
///
///   AsyncVideoDetectionPipeline p(engine, cfg, dcfg);
///   std::vector<Detection> dets;
///   while (std::size_t n = fread(buf, 1, sizeof buf, fp)) {
///     while (!p.submit(buf, n) && !p.finished())
///       while (p.tryNext(dets)) { /* ... */ }  // make room, then retry
///     while (p.tryNext(dets)) { /* ... */ }
///   }
///   p.finish();
///   while (p.next(dets)) { /* the reorder tail and the frames in flight */ }
///
/// A caller that drains on a SEPARATE thread has no such constraint and may
/// pass a negative timeout to make submit() block until the bytes land.
class AsyncVideoDetectionPipeline {
 public:
  /// Duplicate `engine` into the detector's context pool and start the decoder
  /// and the pipeline threads. The Engine must outlive the pipeline.
  /// `dec` carries the codec; its `extra_buffers` is raised if the queue depth
  /// needs more frames in flight than it allows.
  AsyncVideoDetectionPipeline(Engine& engine, PipelineConfig cfg, VideoDecConfig dec,
                              VideoAsyncConfig vcfg = VideoAsyncConfig());
  /// Load the model once and own the whole pool.
  AsyncVideoDetectionPipeline(const std::string& model_path, PipelineConfig cfg, VideoDecConfig dec,
                              VideoAsyncConfig vcfg = VideoAsyncConfig());
  ~AsyncVideoDetectionPipeline();

  AsyncVideoDetectionPipeline(const AsyncVideoDetectionPipeline&) = delete;
  AsyncVideoDetectionPipeline& operator=(const AsyncVideoDetectionPipeline&) = delete;

  /// Feed compressed bytes (any chunk size). True when the decoder took them.
  ///
  /// False means one of two things, and `finished()` tells them apart:
  ///   * back-pressure — the decoder's input queue stayed full for `timeout_ms`.
  ///     The bytes were NOT consumed: drain results and call again with the same
  ///     data. See the class comment for why this is a return value and not a
  ///     longer wait.
  ///   * the pipeline is finished (or a thread failed, in which case the failure
  ///     is rethrown here rather than reported as back-pressure).
  ///
  /// `timeout_ms < 0` waits indefinitely, which is safe only when another thread
  /// drains next().
  bool submit(const std::uint8_t* data, std::size_t size, int timeout_ms = 20);
  bool submit(const std::vector<std::uint8_t>& data, int timeout_ms = 20) {
    return submit(data.data(), data.size(), timeout_ms);
  }

  /// Pop the next frame's detections in DECODE order, blocking until ready.
  /// Returns false once the pipeline is finished AND fully drained.
  bool next(std::vector<Detection>& out);
  /// Non-blocking next(): false immediately when nothing is ready.
  bool tryNext(std::vector<Detection>& out);

  /// Presentation timestamp, decoder frame index and letterbox geometry of the
  /// result the last next()/tryNext() returned. The pts is what a downstream
  /// encoder needs, and it is the only way back to the frame: the decoded
  /// buffer is recycled the instant it has been letterboxed.
  std::uint64_t lastPtsUs() const noexcept;
  std::uint64_t lastFrameIndex() const noexcept;
  const LetterboxInfo& lastLetterbox() const noexcept;

  /// True once finish() has been called or a pipeline thread has failed: no
  /// further bytes will be accepted. Use it to tell a submit() that is applying
  /// back-pressure from one that is refusing input for good.
  bool finished() const noexcept;

  /// End of stream: flush the decoder's reorder tail, let the in-flight frames
  /// drain, then next() returns false. Idempotent; the destructor calls it.
  void finish();

  /// Resolution the stream reported; 0 until the first frame has been decoded.
  int width() const noexcept;
  int height() const noexcept;
  std::uint64_t framesDecoded() const noexcept;
  /// True when the decoder writes into RCDL-allocated dma-bufs (the pool this
  /// pipeline sized) rather than MPP's internal one.
  bool usingExternalBuffers() const noexcept;

  const PipelineConfig& config() const noexcept;
  DetectHead head() const noexcept;
  int workers() const noexcept;

  /// Per-stage SERVICE time: `decode_ms` from the drain thread, the rest from
  /// the detector. Each stage is timed on the thread that ran it, so the fields
  /// sum to more than wall time — the largest one is what bounds throughput.
  /// Settled once finish() has returned and the results are drained.
  StageProfile profile() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rcdl
