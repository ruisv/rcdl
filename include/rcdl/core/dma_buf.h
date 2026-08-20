#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace rcdl {

/// RAII wrapper over a Linux dma-buf — the one buffer type every hardware unit
/// on a Rockchip SoC can consume by file descriptor:
///
///   NPU   rknn_create_mem_from_fd()          (tensor I/O, zero-copy)
///   RGA   importbuffer_fd() / wrapbuffer_handle()  (resize / cvtColor / letterbox)
///   VPU   MPP external buffer groups (MppBufferInfo{fd}) (decode into / encode from)
///
/// So a decoded NV12 frame can be letterboxed by RGA straight into the NPU's
/// input tensor and an annotated frame encoded by the VPU without a memcpy.
///
/// Allocated from a dma-heap (/dev/dma_heap/<name>). Cached heaps are fast for
/// the CPU but need explicit coherency around CPU accesses:
///   - before the CPU reads/writes:  syncStart()   (DMA_BUF_SYNC_START)
///   - after the CPU is done:        syncEnd()     (DMA_BUF_SYNC_END)
/// Uncached heaps skip both at the cost of slow CPU access. The RKNN runtime
/// flushes its own I/O tensors around rknn_run, so a DmaBuf only handed to the
/// NPU needs none of this; one the CPU touches between hardware stages does.
class DmaBuf {
 public:
  enum class Heap {
    System,          ///< /dev/dma_heap/system          — IOMMU-backed units (RK3588 NPU/RGA3/VPU)
    SystemUncached,  ///< /dev/dma_heap/system-uncached
    Cma,             ///< /dev/dma_heap/cma             — physically contiguous (units without an IOMMU)
    CmaUncached,     ///< /dev/dma_heap/cma-uncached
  };
  static const char* heapName(Heap heap) noexcept;

  DmaBuf() = default;
  /// Allocate `size` bytes from `heap`. Throws rcdl::Error on failure (typically
  /// EACCES on /dev/dma_heap/* — see docs for the udev rule).
  static DmaBuf alloc(std::size_t size, Heap heap = Heap::System);
  /// Wrap an existing dma-buf fd (e.g. one exported by MPP or the RKNN runtime).
  /// The fd is dup()'d, so the caller keeps ownership of its own descriptor.
  static DmaBuf fromFd(int fd, std::size_t size);
  ~DmaBuf();

  DmaBuf(const DmaBuf&) = delete;
  DmaBuf& operator=(const DmaBuf&) = delete;
  DmaBuf(DmaBuf&& other) noexcept;
  DmaBuf& operator=(DmaBuf&& other) noexcept;

  bool valid() const noexcept { return fd_ >= 0; }
  int fd() const noexcept { return fd_; }
  std::size_t size() const noexcept { return size_; }
  /// CPU view (mmap'd lazily on first call, MAP_SHARED read/write).
  void* data();
  const void* data() const { return const_cast<DmaBuf*>(this)->data(); }

  /// Begin a CPU access window (read, write or both). Required on cached heaps.
  void syncStart(bool read = true, bool write = true) const;
  /// End the CPU access window started by syncStart().
  void syncEnd(bool read = true, bool write = true) const;

  /// Release the mapping + fd now (also done by the destructor).
  void release() noexcept;

 private:
  void sync(bool start, bool read, bool write) const;

  int fd_ = -1;
  std::size_t size_ = 0;
  void* map_ = nullptr;
};

}  // namespace rcdl
