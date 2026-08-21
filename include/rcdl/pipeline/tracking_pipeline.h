#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "rcdl/media/video_frame.h"
#include "rcdl/pipeline/detection_pipeline.h"
#include "rcdl/tasks/detection.h"
#include "rcdl/tasks/embedding.h"
#include "rcdl/tracks/byte_tracker.h"
#include "rcdl/tracks/reid.h"

namespace rcdl {

class Engine;  // backend/engine.h — referenced by ref, not owned.

/// Appearance settings for TrackingPipeline's optional ReID stage.
struct TrackingReidConfig {
  /// Only detections at or above this score are embedded. Everything below is
  /// associated on geometry alone. This is a COST knob with teeth: the ReID
  /// model runs once per crop, so on a crowded frame it, not the detector, sets
  /// the frame time.
  float min_score = 0.5f;
  /// Cap on crops embedded per frame, so one pathological frame cannot stall a
  /// stream. Detections are taken in the order the detector emitted them
  /// (score-descending after NMS), so the cap drops the least confident.
  /// <= 0 means no cap.
  int max_crops = 32;
  /// Crop normalization (see reidPreprocess). The crop SIZE comes from the ReID
  /// Engine's own input shape, so nothing here needs to be told it.
  ReidConfig crop;
};

/// Streaming detect-and-track: a `DetectionPipeline` feeding a `ByteTracker`
/// once per frame. Returns the frame's active tracks, each with a stable
/// `track_id`, in ORIGINAL-image pixels.
///
/// This is the thin composition the two halves were built for. `DetectionPipeline`
/// already un-letterboxes its boxes to original-image pixels — the same
/// coordinate convention `Track` and `ByteTracker` use — so `process()` is
/// essentially `tracker.update(det.process(...))`. The detection half reuses its
/// buffers (the letterbox destination is the NPU's input tensor), and the tracker
/// keeps only its Kalman and lost-track state, so a steady-state frame allocates
/// nothing beyond the returned vectors.
///
/// With a second Engine holding a ReID model, each qualifying detection is also
/// embedded and association runs on geometry AND appearance — which is what keeps
/// identities through the occlusions and crossings that motion alone loses. The
/// crop preprocessing is a CPU path on purpose (small crops, NCHW float32 output
/// that RGA cannot produce); the detector's own letterbox stays on RGA.
class TrackingPipeline {
 public:
  /// Geometry-only tracking.
  TrackingPipeline(Engine& engine, PipelineConfig det_cfg = PipelineConfig(),
                   ByteTrackConfig track_cfg = ByteTrackConfig())
      : det_(engine, std::move(det_cfg)), tracker_(track_cfg) {}

  /// Tracking with appearance. `reid` holds a model mapping one crop to one
  /// embedding; its input shape supplies the crop size.
  TrackingPipeline(Engine& engine, Engine& reid, PipelineConfig det_cfg = PipelineConfig(),
                   ByteTrackConfig track_cfg = ByteTrackConfig(),
                   TrackingReidConfig reid_cfg = TrackingReidConfig());

  TrackingPipeline(const TrackingPipeline&) = delete;
  TrackingPipeline& operator=(const TrackingPipeline&) = delete;

  /// Run one frame end-to-end and return its active tracks.
  ///
  /// `src` may be anything the preproc layer accepts — a host BGR image, or a
  /// decoded frame's `view()` for the zero-copy video path. When ReID is on, the
  /// crops are taken from `src`, which therefore must have a CPU mapping;
  /// geometry-only tracking does not need one.
  std::vector<Track> process(const ImageView& src);
  std::vector<Track> process(const std::uint8_t* bgr, int width, int height);
  /// Zero-copy overload for a decoded frame. Note that with ReID enabled the
  /// crops must be read by the CPU, so the frame is mapped for the crop step.
  std::vector<Track> process(VideoFrame& frame);

  /// Detections from the most recent process(), before association — for
  /// overlays and for debugging an id that did not stick.
  const std::vector<Detection>& lastDetections() const noexcept { return last_dets_; }
  const LetterboxInfo& lastLetterbox() const noexcept { return det_.lastLetterbox(); }

  /// Drop every tracklet and restart ids at 1 (e.g. on a stream cut).
  void reset() { tracker_.reset(); }

  const DetectionPipeline& detection() const noexcept { return det_; }
  ByteTracker& tracker() noexcept { return tracker_; }
  bool hasReid() const noexcept { return reid_ != nullptr; }
  /// Crops embedded on the most recent frame (0 without ReID) — the term that
  /// makes frame time scale with crowd size.
  int lastEmbedCount() const noexcept { return last_embed_count_; }
  /// Per-stage timing of the detection half.
  const StageProfile& profile() const noexcept { return det_.profile(); }

 private:
  void embedDetections(const ImageView& src);

  DetectionPipeline det_;
  ByteTracker tracker_;
  Engine* reid_ = nullptr;
  std::unique_ptr<ImageEmbedder> embedder_;
  TrackingReidConfig reid_cfg_;
  int last_embed_count_ = 0;
  std::vector<Detection> last_dets_;
  std::vector<std::vector<float>> embeddings_;  ///< reused per frame
};

}  // namespace rcdl
