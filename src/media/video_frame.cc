#ifndef MODULE_TAG
#define MODULE_TAG "rcdl_media"  // names this file in MPP's logs; see mpp_common.cc
#endif

#include "rcdl/media/video_frame.h"

#include <sstream>
#include <utility>

#include "rcdl/core/dma_buf.h"
#include "rcdl/core/status.h"

#include "mpp_common.h"

namespace rcdl {

// ---------------------------------------------------------------------------
// Lifetime
//
// `handle_` is an MppFrame. mpp_frame_deinit() drops the frame's reference on
// its MppBuffer, and the buffer goes back to the group it came from — which is
// how the decoder gets the slot back. Nothing here frees memory: the dma-bufs
// belong to the decoder's buffer group (see video_decoder.cc), never to the
// frame. A VideoFrame must therefore be released BEFORE the decoder that
// produced it is destroyed.
// ---------------------------------------------------------------------------

VideoFrame::~VideoFrame() { reset(); }

VideoFrame::VideoFrame(VideoFrame&& other) noexcept
    : handle_(other.handle_),
      view_(other.view_),
      pts_us_(other.pts_us_),
      index_(other.index_),
      cpu_window_(other.cpu_window_) {
  // The moved-from frame must not deinit the MppFrame or close the coherency
  // window we just took over.
  other.handle_ = nullptr;
  other.view_ = ImageView();
  other.pts_us_ = 0;
  other.index_ = 0;
  other.cpu_window_ = false;
}

VideoFrame& VideoFrame::operator=(VideoFrame&& other) noexcept {
  if (this == &other) return *this;
  reset();  // return the frame we currently hold before taking another
  handle_ = other.handle_;
  view_ = other.view_;
  pts_us_ = other.pts_us_;
  index_ = other.index_;
  cpu_window_ = other.cpu_window_;
  other.handle_ = nullptr;
  other.view_ = ImageView();
  other.pts_us_ = 0;
  other.index_ = 0;
  other.cpu_window_ = false;
  return *this;
}

void VideoFrame::reset() noexcept {
  if (cpu_window_) {
    // Close any coherency window first: releasing the buffer to the pool while
    // the CPU still has an open read window would let the VPU write into it
    // under a stale cache view.
    //
    // Swallow a failure here rather than let it out. This function is noexcept
    // and runs from the destructor, and dmaBufSyncEnd throws on a failed ioctl —
    // which is reachable by a frame outliving its decoder (EBADF once MPP has
    // closed the fd), the very mistake this file warns about. Aborting the
    // process is a strictly worse outcome than an unclosed sync window on a
    // buffer that is being discarded anyway.
    try {
      dmaBufSyncEnd(view_.fd, /*read=*/true, /*write=*/false);
    } catch (...) {  // NOLINT(bugprone-empty-catch) — see above
    }
    cpu_window_ = false;
  }
#if RCDL_HAVE_MPP
  if (handle_ != nullptr) {
    MppFrame f = static_cast<MppFrame>(handle_);
    mpp_frame_deinit(&f);  // buffer ref -1 -> slot returns to the decoder's group
  }
#endif
  handle_ = nullptr;
  view_ = ImageView();
  pts_us_ = 0;
  index_ = 0;
}

VideoFrame VideoFrame::adopt(void* mpp_frame, const ImageView& view, std::uint64_t pts_us,
                             std::uint64_t index) {
#if RCDL_HAVE_MPP
  RCDL_REQUIRE(mpp_frame != nullptr, "VideoFrame::adopt given a null MppFrame");
  VideoFrame f;
  f.handle_ = mpp_frame;
  f.view_ = view;
  f.pts_us_ = pts_us;
  f.index_ = index;
  return f;
#else
  (void)mpp_frame;
  (void)view;
  (void)pts_us;
  (void)index;
  throw Error(-1, mpp::kUnavailable);
#endif
}

const std::uint8_t* VideoFrame::beginCpuRead() {
#if RCDL_HAVE_MPP
  RCDL_REQUIRE(handle_ != nullptr, "VideoFrame::beginCpuRead on an empty frame");
  if (view_.data == nullptr) {
    // Map lazily: the hardware path (RGA / NPU / encoder) consumes the fd and
    // never needs a virtual address, so an unconditional mmap in the decode
    // loop would be a page walk per frame for nothing.
    view_.data = mpp::mapFrame(static_cast<MppFrame>(handle_));
    RCDL_REQUIRE(view_.data != nullptr,
                 "VideoFrame::beginCpuRead: MPP could not map the frame buffer");
  }
  if (!cpu_window_) {
    // The VPU wrote these pixels over the bus, through the IOMMU, without going
    // near the CPU's caches. On a cached dma-heap the CPU's view of those lines
    // is whatever was there before, so the read has to be bracketed by
    // DMA_BUF_IOCTL_SYNC to get the caches invalidated. Read-only: we are not
    // writing the frame back, and claiming write here would force a needless
    // clean of the whole buffer on endCpuRead().
    dmaBufSyncStart(view_.fd, /*read=*/true, /*write=*/false);
    cpu_window_ = true;
  }
  return view_.bytePtr();
#else
  throw Error(-1, mpp::kUnavailable);
#endif
}

void VideoFrame::endCpuRead() {
  if (!cpu_window_) return;
  dmaBufSyncEnd(view_.fd, /*read=*/true, /*write=*/false);
  cpu_window_ = false;
}

std::string VideoFrame::describe() const {
  std::ostringstream os;
  if (handle_ == nullptr) return "VideoFrame(empty)";
  os << "VideoFrame#" << index_ << " pts=" << pts_us_ << "us " << view_.describe();
  if (view_.data != nullptr) os << " mapped";
  if (cpu_window_) os << " cpu-window";
  return os.str();
}

}  // namespace rcdl
