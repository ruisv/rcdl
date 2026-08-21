#include "rcdl/core/dma_buf.h"

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <sstream>
#include <utility>

#include "rcdl/core/status.h"

// The dma-heap / dma-buf UAPI. Present in linux-libc-dev on any distro with a
// 5.6+ kernel; the fallbacks below only exist so the file also compiles against
// an older sysroot (the ioctl numbers are ABI and will not change).
#if __has_include(<linux/dma-heap.h>)
#include <linux/dma-heap.h>
#else
struct dma_heap_allocation_data {
  __u64 len;
  __u32 fd;
  __u32 fd_flags;
  __u64 heap_flags;
};
#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)
#endif
#if __has_include(<linux/dma-buf.h>)
#include <linux/dma-buf.h>
#else
struct dma_buf_sync {
  __u64 flags;
};
#define DMA_BUF_SYNC_READ (1 << 0)
#define DMA_BUF_SYNC_WRITE (2 << 0)
#define DMA_BUF_SYNC_RW (DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE)
#define DMA_BUF_SYNC_START (0 << 2)
#define DMA_BUF_SYNC_END (1 << 2)
#define DMA_BUF_BASE 'b'
#define DMA_BUF_IOCTL_SYNC _IOW(DMA_BUF_BASE, 0, struct dma_buf_sync)
#endif

namespace rcdl {

namespace {
[[noreturn]] void throwErrno(const char* what, const std::string& detail) {
  int e = errno;
  std::ostringstream os;
  os << "RCDL: " << what << " failed: " << std::strerror(e) << " (errno " << e << ")";
  if (!detail.empty()) os << " — " << detail;
  throw Error(-e, os.str());
}
}  // namespace

const char* DmaBuf::heapName(Heap heap) noexcept {
  switch (heap) {
    case Heap::System: return "system";
    case Heap::SystemUncached: return "system-uncached";
    case Heap::Cma: return "cma";
    case Heap::CmaUncached: return "cma-uncached";
  }
  return "system";
}

DmaBuf DmaBuf::alloc(std::size_t size, Heap heap) {
  RCDL_REQUIRE(size > 0, "DmaBuf::alloc: size must be > 0");
  std::string path = std::string("/dev/dma_heap/") + heapName(heap);
  int heap_fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
  if (heap_fd < 0) {
    throwErrno("open dma-heap", path + " (is the heap present, and does your user have rw "
                                        "access? see the udev note in the docs)");
  }
  dma_heap_allocation_data req{};
  req.len = size;
  req.fd_flags = O_RDWR | O_CLOEXEC;
  int rc = ::ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &req);
  ::close(heap_fd);
  if (rc < 0) throwErrno("DMA_HEAP_IOCTL_ALLOC", path);
  DmaBuf b;
  b.fd_ = static_cast<int>(req.fd);
  b.size_ = size;
  return b;
}

DmaBuf DmaBuf::fromFd(int fd, std::size_t size) {
  RCDL_REQUIRE(fd >= 0, "DmaBuf::fromFd: invalid fd");
  int d = ::dup(fd);
  if (d < 0) throwErrno("dup dma-buf fd", "");
  DmaBuf b;
  b.fd_ = d;
  b.size_ = size;
  return b;
}

DmaBuf::~DmaBuf() { release(); }

DmaBuf::DmaBuf(DmaBuf&& other) noexcept
    : fd_(other.fd_), size_(other.size_), map_(other.map_) {
  other.fd_ = -1;
  other.size_ = 0;
  other.map_ = nullptr;
}

DmaBuf& DmaBuf::operator=(DmaBuf&& other) noexcept {
  if (this != &other) {
    release();
    fd_ = other.fd_;
    size_ = other.size_;
    map_ = other.map_;
    other.fd_ = -1;
    other.size_ = 0;
    other.map_ = nullptr;
  }
  return *this;
}

void* DmaBuf::data() {
  RCDL_REQUIRE(valid(), "DmaBuf::data on an empty buffer");
  if (!map_) {
    void* p = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (p == MAP_FAILED) throwErrno("mmap dma-buf", "");
    map_ = p;
  }
  return map_;
}

namespace {
void syncFd(int fd, bool start, bool read, bool write) {
  if (fd < 0) return;  // host-only buffer: nothing to keep coherent
  dma_buf_sync s{};
  s.flags = (start ? DMA_BUF_SYNC_START : DMA_BUF_SYNC_END);
  if (read) s.flags |= DMA_BUF_SYNC_READ;
  if (write) s.flags |= DMA_BUF_SYNC_WRITE;
  if (::ioctl(fd, DMA_BUF_IOCTL_SYNC, &s) < 0) throwErrno("DMA_BUF_IOCTL_SYNC", "");
}
}  // namespace

void dmaBufSyncStart(int fd, bool read, bool write) { syncFd(fd, true, read, write); }
void dmaBufSyncEnd(int fd, bool read, bool write) { syncFd(fd, false, read, write); }

void DmaBuf::sync(bool start, bool read, bool write) const {
  RCDL_REQUIRE(valid(), "DmaBuf::sync on an empty buffer");
  syncFd(fd_, start, read, write);
}

void DmaBuf::syncStart(bool read, bool write) const { sync(true, read, write); }
void DmaBuf::syncEnd(bool read, bool write) const { sync(false, read, write); }

void DmaBuf::release() noexcept {
  if (map_) {
    ::munmap(map_, size_);
    map_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  size_ = 0;
}

}  // namespace rcdl
