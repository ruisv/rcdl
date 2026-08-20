// dma-heap probe: allocate from each heap, map, write, sync, and report. This
// is the buffer every hardware unit shares (NPU / RGA / VPU), so it has to work
// for the unprivileged user before any zero-copy pipeline can.
//
//   ./dma_buf_probe [bytes=4194304]

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rcdl/rcdl.h"

int main(int argc, char** argv) {
  const std::size_t bytes = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : (4u << 20);
  const rcdl::DmaBuf::Heap heaps[] = {
      rcdl::DmaBuf::Heap::System, rcdl::DmaBuf::Heap::SystemUncached,
      rcdl::DmaBuf::Heap::Cma, rcdl::DmaBuf::Heap::CmaUncached};
  int failures = 0;
  for (auto h : heaps) {
    try {
      rcdl::DmaBuf b = rcdl::DmaBuf::alloc(bytes, h);
      b.syncStart();
      std::memset(b.data(), 0xA5, b.size());
      b.syncEnd();
      const auto* p = static_cast<const unsigned char*>(b.data());
      std::printf("heap %-16s fd=%d size=%zu first=0x%02x last=0x%02x  OK\n",
                  rcdl::DmaBuf::heapName(h), b.fd(), b.size(), p[0], p[b.size() - 1]);
    } catch (const std::exception& e) {
      std::printf("heap %-16s FAILED: %s\n", rcdl::DmaBuf::heapName(h), e.what());
      ++failures;
    }
  }
  return failures == 4 ? 1 : 0;
}
