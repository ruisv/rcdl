#pragma once

#include <cstdint>

#include "rcdl/preproc/geometry.h"
#include "rcdl/preproc/image.h"

namespace rcdl {

/// Which engine performs a preprocessing op.
enum class PreprocBackend {
  Auto,  ///< RGA when it is available AND accepts the request; CPU otherwise
  Rga,   ///< force RGA — throws rcdl::Error if unavailable or rejected
  Cpu,   ///< force the CPU fallback (letterbox_cpu.h)
};

const char* backendName(PreprocBackend b) noexcept;

/// Backend-agnostic preprocessing entry points. These are what the pipelines
/// and the Python layer call; they try RGA first and fall back to the CPU only
/// when the hardware cannot do the job (see rgaCanHandle()).
///
/// `used`, when non-null, receives the backend that actually ran — the
/// thread-safe way to tell (there is no hidden global state), and what the
/// examples print so a slow frame is traceable to a CPU fallback.

/// Aspect-preserving letterbox of `src` into the pre-allocated `dst`, padding
/// with `pad`. See rgaLetterbox() / letterboxCpu() for the exact semantics —
/// both produce the same geometry contract.
LetterboxInfo letterbox(const ImageView& dst, const ImageView& src, std::uint8_t pad = 114,
                        PreprocBackend backend = PreprocBackend::Auto,
                        YuvRange range = YuvRange::kStudioToFull,
                        PreprocBackend* used = nullptr);

/// Stretch-resize (no padding, aspect not preserved).
LetterboxInfo resize(const ImageView& dst, const ImageView& src,
                     PreprocBackend backend = PreprocBackend::Auto,
                     YuvRange range = YuvRange::kStudioToFull,
                     PreprocBackend* used = nullptr);

/// Colour-space conversion at identical width/height.
void cvtColor(const ImageView& dst, const ImageView& src,
              PreprocBackend backend = PreprocBackend::Auto,
              YuvRange range = YuvRange::kStudioToFull, PreprocBackend* used = nullptr);

}  // namespace rcdl
