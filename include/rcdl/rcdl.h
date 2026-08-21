#pragma once

// Umbrella header for RCDL.
//
// Layers (see docs/ROADMAP.md for the full map):
//   core/     Status · DmaBuf                    (Linux dma-heap / dma-buf)
//   backend/  Engine · output readers            (librknnrt -> rknn_*)
//   preproc/  Image · RGA letterbox / resize / cvtColor + CPU fallback (librga im2d)
//   media/    VideoDecoder · VideoEncoder · Jpeg (librockchip_mpp)
//   tasks/    det · cls · pose · seg · obb · ocr · depth ...
//   tracks/   ByteTrack · ReID
//   pipeline/ sync / async detection · async video detection · tracking

#include "rcdl/version.h"
#include "rcdl/core/status.h"
#include "rcdl/core/dma_buf.h"
#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
#include "rcdl/preproc/geometry.h"
#include "rcdl/preproc/image.h"
#include "rcdl/preproc/rga.h"
#include "rcdl/preproc/letterbox_cpu.h"
#include "rcdl/preproc/letterbox.h"
#include "rcdl/media/video_frame.h"
#include "rcdl/media/video_codec.h"
#include "rcdl/media/jpeg_codec.h"
#include "rcdl/tasks/detection.h"
#include "rcdl/tasks/classification.h"
#include "rcdl/tasks/embedding.h"
#include "rcdl/tasks/pose.h"
#include "rcdl/tasks/obb.h"
#include "rcdl/tasks/instance_seg.h"
#include "rcdl/tasks/segmentation.h"
#include "rcdl/tasks/depth.h"
#include "rcdl/tasks/ocr.h"
#include "rcdl/tasks/face.h"
#include "rcdl/tracks/byte_tracker.h"
#include "rcdl/tracks/reid.h"
#include "rcdl/backend/engine_pool.h"
#include "rcdl/pipeline/detection_pipeline.h"
#include "rcdl/pipeline/async_detection_pipeline.h"
#include "rcdl/pipeline/async_video_detection_pipeline.h"
#include "rcdl/pipeline/tracking_pipeline.h"
