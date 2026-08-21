#include "rcdl/preproc/letterbox.h"

#include "rcdl/core/status.h"
#include "rcdl/preproc/letterbox_cpu.h"
#include "rcdl/preproc/rga.h"

namespace rcdl {
namespace {

/// Decide which engine runs this op. The whole decision is a pure function of
/// the arguments — no cached "RGA is broken" flag, no last-used-backend memo —
/// so several pipeline threads can call the entry points below concurrently
/// without sharing anything mutable. (rgaAvailable() does cache its probe, but
/// it is idempotent and reports the same answer to every caller.)
bool chooseRga(PreprocBackend backend, const ImageView& dst, const ImageView& src) noexcept {
  switch (backend) {
    case PreprocBackend::Cpu:
      return false;
    case PreprocBackend::Rga:
      // Forced: run the RGA op even if it is going to refuse, so the caller
      // gets the rcdl::Error explaining why instead of a silent slow path.
      return true;
    case PreprocBackend::Auto:
      break;
  }
  // Ask the hardware's own imcheck. A "no" here is an ordinary, expected event
  // (scale out of range, unaligned YUV stride, no librga in this build), not an
  // error — library code must not print anything about it; the caller learns
  // what ran through `used`.
  return rgaCanHandle(dst, src, nullptr);
}

/// Run `hw` on RGA, falling back to `sw` if the hardware refuses at SUBMIT time.
///
/// `rgaCanHandle()` runs librga's `imcheck`, which is a parameter check, not a
/// promise: the driver still picks a core afterwards, and a request the check
/// accepted can fail there. Measured on RK3588: a ratio outside RGA3's own
/// [1/8, 8] window is routed to the RGA2 core, which has no IOMMU and cannot
/// map pages above 4 GB, so it fails with "job buffer map failed" — and
/// `imcheck` never sees a problem, because im2d's documented range is the wider
/// [1/16, 16] that RGA2 would support if it could reach the memory.
///
/// `Auto` promises the CPU path when RGA cannot do the job, so it has to honour
/// that for a run-time refusal too, not just a pre-flight one. Only `Auto`
/// retries: `Rga` is an explicit request whose failure the caller asked to see.
template <typename Hw, typename Sw>
auto runWithFallback(PreprocBackend backend, PreprocBackend* used, Hw&& hw, Sw&& sw)
    -> decltype(sw()) {
  if (backend != PreprocBackend::Auto) return hw();
  try {
    return hw();
  } catch (const Error&) {
    if (used != nullptr) *used = PreprocBackend::Cpu;
    return sw();
  }
}

}  // namespace

const char* backendName(PreprocBackend b) noexcept {
  switch (b) {
    case PreprocBackend::Auto:
      return "auto";
    case PreprocBackend::Rga:
      return "rga";
    case PreprocBackend::Cpu:
      return "cpu";
  }
  return "unknown";
}

// `used` is written BEFORE the op runs, so it still says which backend was
// attempted when that backend throws.

LetterboxInfo letterbox(const ImageView& dst, const ImageView& src, std::uint8_t pad,
                        PreprocBackend backend, YuvRange range, PreprocBackend* used) {
  const bool rga = chooseRga(backend, dst, src);
  if (used != nullptr) *used = rga ? PreprocBackend::Rga : PreprocBackend::Cpu;
  if (!rga) return letterboxCpu(dst, src, pad, range);
  return runWithFallback(
      backend, used, [&] { return rgaLetterbox(dst, src, pad, range); },
      [&] { return letterboxCpu(dst, src, pad, range); });
}

LetterboxInfo resize(const ImageView& dst, const ImageView& src, PreprocBackend backend,
                     YuvRange range, PreprocBackend* used) {
  const bool rga = chooseRga(backend, dst, src);
  if (used != nullptr) *used = rga ? PreprocBackend::Rga : PreprocBackend::Cpu;
  if (!rga) return resizeCpu(dst, src, range);
  return runWithFallback(
      backend, used, [&] { return rgaResize(dst, src, range); },
      [&] { return resizeCpu(dst, src, range); });
}

void cvtColor(const ImageView& dst, const ImageView& src, PreprocBackend backend, YuvRange range,
              PreprocBackend* used) {
  const bool rga = chooseRga(backend, dst, src);
  if (used != nullptr) *used = rga ? PreprocBackend::Rga : PreprocBackend::Cpu;
  if (!rga) {
    cvtColorCpu(dst, src, range);
    return;
  }
  runWithFallback(
      backend, used,
      [&] {
        rgaCvtColor(dst, src, range);
        return 0;
      },
      [&] {
        cvtColorCpu(dst, src, range);
        return 0;
      });
}

}  // namespace rcdl
