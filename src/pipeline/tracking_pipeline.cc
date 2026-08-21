#include "rcdl/pipeline/tracking_pipeline.h"

#include <algorithm>
#include <utility>

#include "rcdl/backend/engine.h"
#include "rcdl/core/status.h"

namespace rcdl {

TrackingPipeline::TrackingPipeline(Engine& engine, Engine& reid, PipelineConfig det_cfg,
                                   ByteTrackConfig track_cfg, TrackingReidConfig reid_cfg)
    : det_(engine, std::move(det_cfg)),
      tracker_(track_cfg),
      reid_(&reid),
      reid_cfg_(std::move(reid_cfg)) {
  // The ReID model's own input shape supplies the crop size, so nothing has to
  // be told it — and a mismatch between the configured crop and what the model
  // was trained on cannot happen.
  EmbedPreproc pre;
  pre.model_input = PixelFormat::RGB888;
  embedder_ = std::make_unique<ImageEmbedder>(reid, EmbedConfig(), pre);
}

void TrackingPipeline::embedDetections(const ImageView& src) {
  embeddings_.assign(last_dets_.size(), std::vector<float>());
  last_embed_count_ = 0;
  if (embedder_ == nullptr) return;
  RCDL_REQUIRE(src.data != nullptr,
               "TrackingPipeline: appearance tracking reads crops with the CPU, so the "
               "source frame needs a CPU mapping (map the VideoFrame, or drop the ReID Engine)");

  // Detections arrive score-descending out of NMS, so taking them in order means
  // the max_crops cap drops the least confident rather than an arbitrary subset.
  for (std::size_t i = 0; i < last_dets_.size(); ++i) {
    if (reid_cfg_.max_crops > 0 && last_embed_count_ >= reid_cfg_.max_crops) break;
    const Detection& d = last_dets_[i];
    if (d.score < reid_cfg_.min_score) continue;
    // A crop that is empty after clipping is not an error here — it is a box on
    // the frame edge. Leave its entry empty and let that detection associate on
    // geometry, which is exactly what an empty entry means to the tracker.
    try {
      embeddings_[i] = embedder_->embed(src, d.x1, d.y1, d.x2, d.y2);
      ++last_embed_count_;
    } catch (const Error&) {
      embeddings_[i].clear();
    }
  }
}

std::vector<Track> TrackingPipeline::process(const ImageView& src) {
  last_dets_ = det_.process(src);
  if (reid_ == nullptr) {
    last_embed_count_ = 0;
    return tracker_.update(last_dets_);
  }
  embedDetections(src);
  return tracker_.update(last_dets_, embeddings_);
}

std::vector<Track> TrackingPipeline::process(const std::uint8_t* bgr, int width, int height) {
  return process(hostView(const_cast<std::uint8_t*>(bgr), width, height, PixelFormat::BGR888));
}

std::vector<Track> TrackingPipeline::process(VideoFrame& frame) {
  if (reid_ == nullptr) {
    // Pure hardware path: RGA reads the VPU's buffer by fd and the CPU never
    // touches the frame, so no mapping and no cache maintenance.
    return process(frame.view());
  }
  // Appearance tracking has to read pixels, so open a CPU window over the frame
  // for the duration — the VPU wrote it through an IOMMU without touching the
  // caches, and reading it without invalidating gives stale data.
  frame.beginCpuRead();
  std::vector<Track> tracks;
  try {
    tracks = process(frame.view());
  } catch (...) {
    frame.endCpuRead();
    throw;
  }
  frame.endCpuRead();
  return tracks;
}

}  // namespace rcdl
