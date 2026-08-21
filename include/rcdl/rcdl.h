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
//   pipeline/ sync / async detection · tracking

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
