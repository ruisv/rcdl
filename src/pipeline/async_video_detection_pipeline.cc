#include "rcdl/pipeline/async_video_detection_pipeline.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include "rcdl/backend/engine.h"
#include "rcdl/core/status.h"
#include "rcdl/media/video_frame.h"

namespace rcdl {

namespace {

inline double msBetween(std::chrono::steady_clock::time_point a,
                        std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

/// Granularity of an indefinite (`timeout_ms < 0`) feed: how long one attempt
/// waits before looking again at whether the pipeline is shutting down.
constexpr int kFeedSliceMs = 20;

/// Consecutive empty flush attempts before the drain gives up on a decoder that
/// has stopped producing frames without ever signalling end of stream. Each
/// attempt carries VideoDecoder::flush's own wait, so this is seconds of silence
/// rather than a spin.
constexpr int kMaxEmptyFlushes = 20;

/// Size the decoder's pool against what this pipeline holds at once: the frames
/// parked in the queue, the one the drain thread is receiving into, and the one
/// the letterbox stage is reading. Too few and the decoder stalls on its own
/// buffers — which looks exactly like a slow stream, so it is worth doing here
/// rather than leaving to the caller.
VideoDecConfig sizePool(VideoDecConfig dec, const VideoAsyncConfig& vcfg) {
  const int held = (vcfg.queue_depth > 0 ? vcfg.queue_depth : 1) + 2;
  if (dec.extra_buffers < held) dec.extra_buffers = held;
  return dec;
}

/// A bounded, closable, move-only channel between two pipeline stages: push()
/// blocks while full (back-pressure), pop() blocks while empty, close() wakes
/// both sides and lets pop() drain what is left before it reports the end.
template <class T>
class Channel {
 public:
  explicit Channel(std::size_t cap) : cap_(cap ? cap : 1) {}

  bool push(T v) {
    std::unique_lock<std::mutex> lk(m_);
    not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
    if (closed_) return false;
    q_.push_back(std::move(v));
    not_empty_.notify_one();
    return true;
  }

  bool pop(T& out) {
    std::unique_lock<std::mutex> lk(m_);
    not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty()) return false;
    out = std::move(q_.front());
    q_.pop_front();
    not_full_.notify_one();
    return true;
  }

  void close() {
    std::lock_guard<std::mutex> lk(m_);
    closed_ = true;
    not_full_.notify_all();
    not_empty_.notify_all();
  }

 private:
  std::size_t cap_;
  std::mutex m_;
  std::condition_variable not_full_, not_empty_;
  std::deque<T> q_;
  bool closed_ = false;
};

}  // namespace

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
//
// FOUR STAGES, three of them threads here and the fourth inside the detector:
//
//   caller thread  submit()  -> VideoDecoder::feed        (MPP parser + VPU)
//   drain thread   receive() -> frames_ channel           (display order)
//   lb thread      acquireSlot / letterboxIntoSlot / commitSlot   (RGA)
//   detector       N workers: rknn_run + head decode + NMS        (NPU + CPU)
//
// One feeder and one drainer, which is exactly the threading VideoDecoder
// documents. The letterbox is a separate stage from the drain so RGA overlaps
// the VPU instead of serialising behind it, and it uses the detector's
// acquire/letterbox/commit split rather than submit(): a stage that blocked
// inside submit() while holding a decoded frame would keep that buffer out of
// the decoder's pool, starving the very producer it is waiting for. Waiting for
// the NPU slot FIRST holds nothing.
struct AsyncVideoDetectionPipeline::Impl {
  /// What the detector cannot carry for us. The decoded buffer is recycled the
  /// moment it has been letterboxed, so the pts and index have to travel beside
  /// the frame: pushed in commit order, popped in delivery order, which is the
  /// same order because the detector delivers by submission sequence.
  struct FrameMeta {
    std::uint64_t pts_us = 0;
    std::uint64_t index = 0;
  };

  VideoDecoder dec;
  std::unique_ptr<AsyncDetectionPipeline> detect;
  Channel<VideoFrame> frames;

  std::thread drain_thr, lb_thr;
  /// Set by finish() on the caller's thread and read by submit()/finished().
  /// Atomic because finished() is the one accessor a second thread has a
  /// reason to call — a driver that feeds on one thread and drains on another.
  std::atomic<bool> finished{false};

  mutable std::mutex mu;            ///< guards the meta FIFO, the flags and the error
  std::deque<FrameMeta> meta;       ///< in flight, oldest first
  FrameMeta last_meta;              ///< of the last delivered result
  bool input_done = false;          ///< every byte fed: the drain may flush the tail
  bool stopping = false;            ///< shutting down: every wait must end
  std::string error;                ///< first failure on any pipeline thread

  mutable std::mutex prof_mu;
  double decode_ms = 0;  ///< drain thread only

  int poll_ms;

  Impl(std::unique_ptr<AsyncDetectionPipeline> det, const VideoDecConfig& dec_cfg,
       const VideoAsyncConfig& vcfg)
      : dec(dec_cfg),
        detect(std::move(det)),
        frames(static_cast<std::size_t>(vcfg.queue_depth > 0 ? vcfg.queue_depth : 1)),
        poll_ms(vcfg.drain_poll_ms > 0 ? vcfg.drain_poll_ms : 10) {
    drain_thr = std::thread([this] { drainLoop(); });
    lb_thr = std::thread([this] { letterboxLoop(); });
  }

  ~Impl() { shutdown(); }

  bool stopped() {
    std::lock_guard<std::mutex> lk(mu);
    return stopping;
  }

  /// Record the first failure and tear the pipeline down. A thread that throws
  /// is not allowed to escape (that is std::terminate) and must not be
  /// swallowed either, so the message is carried to the caller's thread by
  /// submit()/next() instead.
  void fail(const std::string& what) {
    {
      std::lock_guard<std::mutex> lk(mu);
      if (error.empty()) error = what;
      stopping = true;
    }
    frames.close();
    detect->finish();
  }

  /// Rethrow, on the CALLER's thread, whatever a pipeline thread failed with.
  void rethrow() {
    std::string what;
    {
      std::lock_guard<std::mutex> lk(mu);
      what = error;
    }
    if (!what.empty()) {
      RCDL_REQUIRE(false,
                   ("AsyncVideoDetectionPipeline: a pipeline thread failed: " + what).c_str());
    }
  }

  /// VPU -> queue. While the stream is live, poll receive(); once every byte
  /// has been fed, flush() feeds the end-of-stream marker (idempotent) and
  /// drains the reorder tail, which is where a B-frame stream's last pictures
  /// come from.
  void drainLoop() {
    try {
      drainStream();
    } catch (const std::exception& e) {
      fail(e.what());
    }
    // Whatever ended this thread — end of stream, shutdown or a failure — no
    // further frame is coming, and the letterbox stage is parked in pop()
    // waiting to be told so.
    frames.close();
  }

  void drainStream() {
    int empty_flushes = 0;
    for (;;) {
      VideoFrame frame;
      // Timed from here, across the empty polls too: the decode stage's service
      // time is how long this thread takes to produce the next frame, and a
      // frame the VPU has not finished yet is exactly what that measures. (Time
      // spent blocked on a full frame queue is NOT counted — that is downstream
      // back-pressure, and charging it to decode would blame the wrong stage.)
      const auto t0 = std::chrono::steady_clock::now();
      for (;;) {
        if (stopped()) return;
        bool done;
        {
          std::lock_guard<std::mutex> lk(mu);
          done = input_done;
        }
        if (done ? dec.flush(frame) : dec.receive(frame, poll_ms)) {
          empty_flushes = 0;
          break;
        }
        // receive() timing out only means the decoder has nothing ready yet.
        //
        // flush() returning false has TWO meanings and only one of them is the
        // end: the decoder has signalled end of stream, or it simply had nothing
        // ready inside its wait. Treating the second as the first truncates the
        // tail of the stream whenever the board is busy enough for the decoder
        // to fall behind that wait — and a truncated video looks exactly like a
        // short one, with nothing to say frames went missing. So the decoder's
        // own end-of-stream flag decides, and the attempt count is only a
        // backstop for a stream whose EOS marker never arrives.
        if (done && (dec.endOfStream() || ++empty_flushes >= kMaxEmptyFlushes)) return;
      }
      {
        std::lock_guard<std::mutex> lk(prof_mu);
        decode_ms += msBetween(t0, std::chrono::steady_clock::now());
      }
      if (!frames.push(std::move(frame))) return;  // shutting down
    }
  }

  /// queue -> RGA -> NPU input tensor. Takes the NPU slot before the frame, so
  /// nothing is held while waiting for capacity, and gives the frame back to the
  /// decoder the instant the letterbox has read it.
  void letterboxLoop() {
    try {
      for (;;) {
        const int slot = detect->acquireSlot();
        if (slot < 0) break;  // finished or shutting down
        VideoFrame frame;
        if (!frames.pop(frame)) {  // decoder drained: hand the slot back
          detect->releaseSlot(slot);
          break;
        }
        LetterboxInfo lb;
        try {
          lb = detect->letterboxIntoSlot(slot, frame.view());
        } catch (...) {
          detect->releaseSlot(slot);  // keep acquire/commit paired
          throw;
        }
        const FrameMeta fm{frame.ptsUs(), frame.index()};
        frame.reset();  // one buffer back in the decoder's pool, now
        {
          // Published BEFORE the commit: a worker can finish and next() can
          // deliver the moment the frame is queued, and a result whose metadata
          // has not landed yet would pop the wrong entry.
          std::lock_guard<std::mutex> lk(mu);
          meta.push_back(fm);
        }
        if (!detect->commitSlot(slot, lb)) {
          std::lock_guard<std::mutex> lk(mu);
          meta.pop_back();  // finish() won the race; the frame was dropped
          break;
        }
      }
    } catch (const std::exception& e) {
      fail(e.what());
      return;
    }
    detect->finish();  // no more frames will reach the detector
  }

  /// Pair a delivered result with the metadata queued for it.
  void popMeta() {
    std::lock_guard<std::mutex> lk(mu);
    if (meta.empty()) return;  // cannot happen; costs nothing to be sure
    last_meta = meta.front();
    meta.pop_front();
  }

  /// finish() + join, idempotent and safe from the destructor.
  void shutdown() {
    {
      std::lock_guard<std::mutex> lk(mu);
      stopping = true;
      input_done = true;
    }
    frames.close();
    // The letterbox thread may be parked inside acquireSlot() on the detector's
    // own back-pressure, which closing our channel cannot reach.
    detect->finish();
    if (drain_thr.joinable()) drain_thr.join();
    if (lb_thr.joinable()) lb_thr.join();
  }
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

AsyncVideoDetectionPipeline::AsyncVideoDetectionPipeline(Engine& engine, PipelineConfig cfg,
                                                         VideoDecConfig dec,
                                                         VideoAsyncConfig vcfg) {
  auto detect = std::make_unique<AsyncDetectionPipeline>(engine, std::move(cfg), vcfg.async);
  impl_ = std::make_unique<Impl>(std::move(detect), sizePool(dec, vcfg), vcfg);
}

AsyncVideoDetectionPipeline::AsyncVideoDetectionPipeline(const std::string& model_path,
                                                         PipelineConfig cfg, VideoDecConfig dec,
                                                         VideoAsyncConfig vcfg) {
  auto detect = std::make_unique<AsyncDetectionPipeline>(model_path, std::move(cfg), vcfg.async);
  impl_ = std::make_unique<Impl>(std::move(detect), sizePool(dec, vcfg), vcfg);
}

AsyncVideoDetectionPipeline::~AsyncVideoDetectionPipeline() = default;

// ---------------------------------------------------------------------------
// Feeding
// ---------------------------------------------------------------------------

bool AsyncVideoDetectionPipeline::submit(const std::uint8_t* data, std::size_t size,
                                        int timeout_ms) {
  impl_->rethrow();
  if (impl_->finished || impl_->stopped()) return false;
  if (data == nullptr || size == 0) return true;
  // feed() returning false is the decoder's input queue being full. One bounded
  // attempt and the answer goes back to the caller, who is very possibly also
  // the only thread that can unblock us by draining a result.
  if (timeout_ms >= 0) return impl_->dec.feed(data, size, 0, timeout_ms);
  // Indefinite: sliced, so a failure on another thread or a shutdown ends the
  // wait instead of leaving this thread parked in MPP.
  while (!impl_->dec.feed(data, size, 0, kFeedSliceMs)) {
    impl_->rethrow();
    if (impl_->finished || impl_->stopped()) return false;
  }
  return true;
}

bool AsyncVideoDetectionPipeline::finished() const noexcept {
  if (impl_->finished) return true;
  std::lock_guard<std::mutex> lk(impl_->mu);
  return impl_->stopping;
}

// ---------------------------------------------------------------------------
// Draining
// ---------------------------------------------------------------------------

bool AsyncVideoDetectionPipeline::next(std::vector<Detection>& out) {
  impl_->rethrow();
  if (!impl_->detect->next(out)) {
    impl_->rethrow();  // a drain/letterbox failure ends the stream; say why
    return false;
  }
  impl_->popMeta();
  return true;
}

bool AsyncVideoDetectionPipeline::tryNext(std::vector<Detection>& out) {
  impl_->rethrow();
  if (!impl_->detect->tryNext(out)) return false;
  impl_->popMeta();
  return true;
}

std::uint64_t AsyncVideoDetectionPipeline::lastPtsUs() const noexcept {
  return impl_->last_meta.pts_us;
}

std::uint64_t AsyncVideoDetectionPipeline::lastFrameIndex() const noexcept {
  return impl_->last_meta.index;
}

const LetterboxInfo& AsyncVideoDetectionPipeline::lastLetterbox() const noexcept {
  return impl_->detect->lastLetterbox();
}

void AsyncVideoDetectionPipeline::finish() {
  if (impl_->finished) return;
  impl_->finished = true;
  {
    // The drain thread reads this and switches from receive() to flush(), which
    // is what feeds the end-of-stream marker and releases the reorder tail. The
    // detector is NOT finished here: the frames still in the decoder have not
    // been submitted to it yet, and the letterbox thread finishes it once the
    // queue runs dry.
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->input_done = true;
  }
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

int AsyncVideoDetectionPipeline::width() const noexcept { return impl_->dec.width(); }
int AsyncVideoDetectionPipeline::height() const noexcept { return impl_->dec.height(); }

std::uint64_t AsyncVideoDetectionPipeline::framesDecoded() const noexcept {
  return impl_->dec.framesDecoded();
}

bool AsyncVideoDetectionPipeline::usingExternalBuffers() const noexcept {
  return impl_->dec.usingExternalBuffers();
}

const PipelineConfig& AsyncVideoDetectionPipeline::config() const noexcept {
  return impl_->detect->config();
}

DetectHead AsyncVideoDetectionPipeline::head() const noexcept { return impl_->detect->head(); }

int AsyncVideoDetectionPipeline::workers() const noexcept { return impl_->detect->workers(); }

StageProfile AsyncVideoDetectionPipeline::profile() const {
  StageProfile p = impl_->detect->profile();  // preproc / infer / postproc / frames
  std::lock_guard<std::mutex> lk(impl_->prof_mu);
  p.decode_ms = impl_->decode_ms;
  return p;
}

}  // namespace rcdl
