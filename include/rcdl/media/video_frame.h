#pragma once

#include <cstdint>
#include <string>

#include "rcdl/preproc/image.h"

namespace rcdl {

/// One decoded frame, still living in the buffer the VPU wrote it into.
///
/// This is the type that makes "VPU -> RGA -> NPU with no memcpy" real. MPP
/// hands back an `MppFrame` wrapping an `MppBuffer`; that buffer is a dma-buf,
/// so `view()` carries its fd and RGA can letterbox straight out of it into the
/// NPU's input tensor. Nothing is copied on the way out of the decoder — which
/// is exactly why the frame must be RELEASED promptly: every frame held is one
/// buffer the decoder cannot reuse, and holding the whole group deadlocks the
/// decode loop. Destroy it, `reset()` it, or move it along.
///
/// Move-only, like every other buffer owner in RCDL. The destructor returns the
/// underlying MPP frame (and with it the buffer) to the decoder's pool.
///
/// Cache discipline: the VPU wrote these pixels through an IOMMU without
/// touching the CPU caches. Reading them with the CPU therefore needs
/// `syncStart()` / `syncEnd()` around the access; handing the fd to RGA or the
/// NPU needs neither.
class VideoFrame {
 public:
  VideoFrame() = default;
  ~VideoFrame();

  VideoFrame(const VideoFrame&) = delete;
  VideoFrame& operator=(const VideoFrame&) = delete;
  VideoFrame(VideoFrame&& other) noexcept;
  VideoFrame& operator=(VideoFrame&& other) noexcept;

  bool valid() const noexcept { return handle_ != nullptr; }
  /// Descriptor for the preproc / media layers: fd, CPU pointer, strides,
  /// format. The CPU pointer may be null when the buffer was never mapped.
  const ImageView& view() const noexcept { return view_; }
  int fd() const noexcept { return view_.fd; }
  int width() const noexcept { return view_.width; }
  int height() const noexcept { return view_.height; }
  PixelFormat format() const noexcept { return view_.format; }
  /// Presentation timestamp in microseconds, as carried by the bitstream.
  std::uint64_t ptsUs() const noexcept { return pts_us_; }
  /// Display-order index assigned by the decoder, counting from 0.
  std::uint64_t index() const noexcept { return index_; }

  /// Map the buffer for CPU access and open a coherency window. Returns the
  /// mapped pointer. Pair with `endCpuAccess()`.
  const std::uint8_t* beginCpuRead();
  void endCpuRead();

  /// Release the frame back to the decoder's pool now, before the destructor.
  void reset() noexcept;

  std::string describe() const;

  /// Adopt an MppFrame (as `void*`, so this header stays free of MPP's headers).
  /// Used by the decoder; not part of the public data path.
  static VideoFrame adopt(void* mpp_frame, const ImageView& view, std::uint64_t pts_us,
                          std::uint64_t index);

 private:
  void* handle_ = nullptr;  ///< MppFrame, owned
  ImageView view_;
  std::uint64_t pts_us_ = 0;
  std::uint64_t index_ = 0;
  bool cpu_window_ = false;
};

}  // namespace rcdl
