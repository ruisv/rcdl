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
/// Cache maintenance on a bare dma-buf fd, for code that holds an ImageView (a
/// non-owning descriptor) rather than the DmaBuf itself — the RGA and media
/// layers, which are handed buffers allocated elsewhere (the RKNN runtime, MPP).
/// Both are no-ops for fd < 0, so a host-only buffer needs no special case.
void dmaBufSyncStart(int fd, bool read = true, bool write = true);
void dmaBufSyncEnd(int fd, bool read = true, bool write = true);

class DmaBuf {
 public:
  enum class Heap {
    System,          ///< /dev/dma_heap/system          — IOMMU-backed units (RK3588 NPU/RGA3/VPU)
    SystemUncached,  ///< /dev/dma_heap/system-uncached
    Cma,             ///< /dev/dma_heap/cma             — physically contiguous (units without an IOMMU)
    CmaUncached,     ///< /dev/dma_heap/cma-uncached
    /// /dev/dma_heap/system-dma32 — pages guaranteed BELOW 4 GB physical. This
    /// is what a unit with a 32-bit MMU needs: on RK3588 the RGA2 core (colour
    /// fill, rectangle overlay, YUV planar, GRAY8, scale ratios beyond 8x) can
    /// address nothing above that line, so a buffer it must write comes from
    /// here. Finite — it is the low quarter of a 16 GB board — so it is the
    /// heap for the few buffers that need it, not the default.
    SystemDma32,
    SystemUncachedDma32,  ///< /dev/dma_heap/system-uncached-dma32
  };
  static const char* heapName(Heap heap) noexcept;
  /// Does every page of a buffer from `heap` sit below 4 GB physical? True for
  /// the dma32 heaps only: the cma heap is contiguous but not guaranteed low.
  static bool heapBelow4G(Heap heap) noexcept;

  DmaBuf() = default;
  /// Allocate `size` bytes from `heap`. Throws rcdl::Error on failure (typically
  /// EACCES on /dev/dma_heap/* — see docs for the udev rule).
  static DmaBuf alloc(std::size_t size, Heap heap = Heap::System);
  /// Wrap an existing dma-buf fd (e.g. one exported by MPP or the RKNN runtime).
  /// The fd is dup()'d, so the caller keeps ownership of its own descriptor.
  /// `below_4g` says the caller knows the pages are below 4 GB physical (it
  /// allocated them from a dma32 heap itself); it cannot be discovered from
  /// the fd, so it defaults to "unknown", i.e. false.
  static DmaBuf fromFd(int fd, std::size_t size, bool below_4g = false);
  ~DmaBuf();

  DmaBuf(const DmaBuf&) = delete;
  DmaBuf& operator=(const DmaBuf&) = delete;
  DmaBuf(DmaBuf&& other) noexcept;
  DmaBuf& operator=(DmaBuf&& other) noexcept;

  bool valid() const noexcept { return fd_ >= 0; }
  int fd() const noexcept { return fd_; }
  std::size_t size() const noexcept { return size_; }
  /// The heap this buffer was allocated from; Heap::System for a wrapped fd.
  Heap heap() const noexcept { return heap_; }
  /// Every page is known to be below 4 GB physical — allocated from a dma32
  /// heap, or wrapped with that assurance. False means unknown, not "above".
  bool below4G() const noexcept { return below_4g_; }
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
  Heap heap_ = Heap::System;
  bool below_4g_ = false;
};

}  // namespace rcdl
