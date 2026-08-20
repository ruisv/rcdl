#pragma once

// Umbrella header for RCDL.
//
// Layers (see docs/ROADMAP.md for the full map):
//   core/     Status · DmaBuf                    (Linux dma-heap / dma-buf)
//   backend/  Engine · output readers            (librknnrt -> rknn_*)
//   preproc/  RGA letterbox / resize / cvtColor  (librga im2d)   [planned]
//   media/    VideoDecoder · VideoEncoder · Jpeg (librockchip_mpp) [planned]
//   tasks/    det · cls · pose · seg · obb · ocr · depth ...   [planned]
//   tracks/   ByteTrack                                        [planned]
//   pipeline/ sync / async detection · tracking                [planned]

#include "rcdl/version.h"
#include "rcdl/core/status.h"
#include "rcdl/core/dma_buf.h"
#include "rcdl/backend/engine.h"
#include "rcdl/backend/output_reader.h"
