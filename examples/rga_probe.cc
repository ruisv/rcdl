// RGA probe: which core can reach which memory on this board, and whether the
// user-space core selection behaves the way the wrappers in preproc/rga.cc rely
// on. Run it once on a new board or a new kernel; the output is what decides
// the fill/overlay strategy (see docs/RGA.md §3).
//
//   ./rga_probe
//
// What it checks, in order:
//   1. colour fill on a `system` buffer          (fails above 4 GB: RGA2 has a 32-bit MMU)
//   2. colour fill on a `system-dma32` buffer    (must pass: pages are below 4 GB)
//   3. a fill pinned to the RGA3 cores           (must FAIL: proves the core mask is honoured)
//   4. is imconfig(IM_CONFIG_SCHEDULER_CORE) per thread?
//   5. fill colour on an NV12 destination        (does the driver convert RGB -> YUV?)
//   6. imrectangleArray with many boxes          (one submit per frame; task-count limit)
//   7. NV12 -> RGB888 letterbox on RGA3 vs RGA2  (do the two cores resample identically?)
//   8. RGB -> YUV BT.709 pinned to RGA3          (the driver advertises it; librga routes it to RGA2)
//   9. GRAY8 and >8x scaling on RGA2 with dma32 buffers

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "rcdl/rcdl.h"

#if RCDL_HAVE_RGA
#include <stdio.h>
#include <string.h>

#include <rga/im2d.h>

namespace {

using rcdl::DmaBuf;

constexpr int kRga3 = IM_SCHEDULER_RGA3_CORE0 | IM_SCHEDULER_RGA3_CORE1;
constexpr int kRga2 = IM_SCHEDULER_RGA2_CORE0;

bool ok(IM_STATUS s) { return s == IM_STATUS_SUCCESS || s == IM_STATUS_NOERROR; }

void say(const char* what, IM_STATUS s, bool expect_ok) {
  std::printf("  %-58s %s  (%s)%s\n", what, ok(s) ? "OK  " : "FAIL", imStrError_t(s),
              ok(s) == expect_ok ? "" : "   <-- unexpected");
}

void core(int mask) { imconfig(IM_CONFIG_SCHEDULER_CORE, static_cast<uint64_t>(mask)); }

struct Buf {
  DmaBuf b;
  rga_buffer_t r{};
  int w = 0, h = 0, fmt = 0;
};

Buf make(DmaBuf::Heap heap, int w, int h, int fmt, std::size_t bytes) {
  Buf x;
  x.b = DmaBuf::alloc(bytes, heap);
  x.w = w;
  x.h = h;
  x.fmt = fmt;
  x.r = wrapbuffer_fd_t(x.b.fd(), w, h, w, h, fmt);
  return x;
}

Buf rgba(DmaBuf::Heap heap, int w, int h) {
  return make(heap, w, h, RK_FORMAT_RGBA_8888, static_cast<std::size_t>(w) * h * 4);
}
Buf rgb(DmaBuf::Heap heap, int w, int h) {
  return make(heap, w, h, RK_FORMAT_RGB_888, static_cast<std::size_t>(w) * h * 3);
}
Buf nv12(DmaBuf::Heap heap, int w, int h) {
  return make(heap, w, h, RK_FORMAT_YCbCr_420_SP, static_cast<std::size_t>(w) * h * 3 / 2);
}
Buf gray(DmaBuf::Heap heap, int w, int h) {
  return make(heap, w, h, RK_FORMAT_YCbCr_400, static_cast<std::size_t>(w) * h);
}

IM_STATUS fill(Buf& d, int x, int y, int w, int h, uint32_t abgr) {
  const im_rect r{x, y, w, h};
  return imfill_t(d.r, r, static_cast<int>(abgr), 1);
}

IM_STATUS blit(Buf& s, Buf& d, int mask, int csc = IM_COLOR_SPACE_DEFAULT,
               im_rect drect = im_rect{}) {
  rga_buffer_t src = s.r, dst = d.r, pat{};
  if (csc != IM_COLOR_SPACE_DEFAULT) {
    imsetColorSpace(&src, static_cast<IM_COLOR_SPACE_MODE>(csc));
    imsetColorSpace(&dst, static_cast<IM_COLOR_SPACE_MODE>(csc));
  }
  const im_rect srect{0, 0, s.w, s.h};
  if (drect.width == 0) drect = im_rect{0, 0, d.w, d.h};
  im_rect prect{};
  im_opt_t opt{};
  opt.core = mask;
  core(mask);
  const IM_STATUS chk = imcheck_t(src, dst, pat, srect, drect, prect, 0);
  if (!ok(chk)) return chk;
  return improcess(src, dst, pat, srect, drect, prect, -1, nullptr, &opt, IM_SYNC);
}

// Deterministic content: a gradient with a hard edge, so a sampling-phase
// difference between the cores shows up as a pixel difference.
void paintNv12(Buf& b) {
  b.b.syncStart(false, true);
  auto* p = static_cast<std::uint8_t*>(b.b.data());
  for (int y = 0; y < b.h; ++y) {
    for (int x = 0; x < b.w; ++x) {
      p[static_cast<std::size_t>(y) * b.w + x] =
          static_cast<std::uint8_t>(((x / 7) % 2 == 0 ? 40 : 200) + (y % 50));
    }
  }
  std::uint8_t* uv = p + static_cast<std::size_t>(b.w) * b.h;
  for (int y = 0; y < b.h / 2; ++y) {
    for (int x = 0; x < b.w; x += 2) {
      uv[static_cast<std::size_t>(y) * b.w + x] = static_cast<std::uint8_t>(64 + (x / 4) % 128);
      uv[static_cast<std::size_t>(y) * b.w + x + 1] = static_cast<std::uint8_t>(192 - (y / 4) % 128);
    }
  }
  b.b.syncEnd(false, true);
}

void diff(Buf& a, Buf& b, const char* what) {
  a.b.syncStart(true, false);
  b.b.syncStart(true, false);
  const auto* pa = static_cast<const std::uint8_t*>(a.b.data());
  const auto* pb = static_cast<const std::uint8_t*>(b.b.data());
  const std::size_t n = a.b.size();
  std::size_t ne = 0;
  int mx = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const int d = std::abs(static_cast<int>(pa[i]) - static_cast<int>(pb[i]));
    if (d) ++ne;
    if (d > mx) mx = d;
  }
  a.b.syncEnd(true, false);
  b.b.syncEnd(true, false);
  std::printf("  %-58s %zu of %zu bytes differ, max |diff| %d\n", what, ne, n, mx);
}

double ms(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int run() {
  std::printf("rcdl %s | %s\n\n", RCDL_VERSION_STRING, rcdl::rgaVersion().c_str());

  std::printf("1-2. colour fill vs heap (default scheduling)\n");
  {
    Buf sys = rgba(DmaBuf::Heap::System, 64, 64);
    say("imfill 64x64 RGBA, system heap", fill(sys, 0, 0, 64, 64, 0xFF0000FFu), false);
  }
  Buf d32 = rgba(DmaBuf::Heap::SystemDma32, 64, 64);
  say("imfill 64x64 RGBA, system-dma32 heap", fill(d32, 0, 0, 64, 64, 0xFF0000FFu), true);

  std::printf("3. the core mask is honoured\n");
  core(kRga3);
  say("imfill on dma32 pinned to RGA3 (no fill on RGA3)", fill(d32, 0, 0, 64, 64, 0xFF00FF00u),
      false);
  core(kRga2);
  say("imfill on dma32 pinned to RGA2", fill(d32, 0, 0, 64, 64, 0xFF00FF00u), true);

  std::printf("4. is the core mask per thread?\n");
  {
    core(kRga3);  // this thread: RGA3 only
    IM_STATUS other = IM_STATUS_FAILED;
    std::thread([&] { other = fill(d32, 0, 0, 64, 64, 0xFFFF0000u); }).join();
    say("fill from a fresh thread while this thread is pinned to RGA3", other, true);
    std::printf("  => imconfig(IM_CONFIG_SCHEDULER_CORE) is %s\n",
                ok(other) ? "THREAD-LOCAL (each thread pins its own)" : "PROCESS-WIDE");
    core(kRga2);
  }

  std::printf("5. fill colour on an NV12 destination (pure red 0xFF0000FF)\n");
  Buf frame = nv12(DmaBuf::Heap::SystemDma32, 1920, 1080);
  {
    paintNv12(frame);
    say("imfill 200x100 at (100,100) on NV12 dma32", fill(frame, 100, 100, 200, 100, 0xFF0000FFu),
        true);
    frame.b.syncStart(true, false);
    const auto* p = static_cast<const std::uint8_t*>(frame.b.data());
    const std::size_t y = 150u * 1920u + 200u;
    const std::size_t uv = 1920u * 1080u + 75u * 1920u + 200u;
    std::printf("  Y=%d Cb=%d Cr=%d   (BT.601 full-range red is Y=76 Cb=91 Cr=255)\n", p[y],
                p[uv], p[uv + 1]);
    frame.b.syncEnd(true, false);
  }

  std::printf("6. imrectangleArray: many boxes in one submit\n");
  for (int n : {1, 8, 12, 13, 20, 40}) {
    std::vector<im_rect> rects;
    for (int i = 0; i < n; ++i) {
      rects.push_back(im_rect{20 + (i * 44) % 1600, 20 + (i * 26) % 900, 120 + (i % 5) * 20,
                              80 + (i % 3) * 20});
    }
    const auto t = std::chrono::steady_clock::now();
    const IM_STATUS s = imrectangleArray(frame.r, rects.data(), n, 0xFF00FF00u, 2, 1, nullptr);
    const double dt = ms(t);
    char what[96];
    std::snprintf(what, sizeof what, "imrectangleArray n=%d thickness=2  (%.2f ms)", n, dt);
    say(what, s, true);
  }
  {
    std::vector<im_rect> rects;
    for (int i = 0; i < 10; ++i) rects.push_back(im_rect{40 * i, 40 * i, 200, 120});
    const auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) imrectangle(frame.r, rects[i], 0xFF0000FFu, 2, 1, nullptr);
    std::printf("  10 x imrectangle (one submit each)                       (%.2f ms)\n", ms(t));
  }

  std::printf("7. NV12 1920x1080 -> RGB888 640x360, RGA3 vs RGA2\n");
  {
    paintNv12(frame);
    Buf out3 = rgb(DmaBuf::Heap::System, 640, 360);
    Buf out2 = rgb(DmaBuf::Heap::SystemDma32, 640, 360);
    Buf out3b = rgb(DmaBuf::Heap::System, 640, 360);
    auto t = std::chrono::steady_clock::now();
    say("RGA3, dst in system heap", blit(frame, out3, kRga3, IM_YUV_TO_RGB_BT601_LIMIT), true);
    std::printf("    %.2f ms\n", ms(t));
    say("RGA3 again", blit(frame, out3b, kRga3, IM_YUV_TO_RGB_BT601_LIMIT), true);
    diff(out3, out3b, "RGA3 run 1 vs run 2");
    t = std::chrono::steady_clock::now();
    say("RGA2, dst in dma32 heap", blit(frame, out2, kRga2, IM_YUV_TO_RGB_BT601_LIMIT), true);
    std::printf("    %.2f ms\n", ms(t));
    diff(out3, out2, "RGA3 vs RGA2");
    Buf outsys = rgb(DmaBuf::Heap::System, 640, 360);
    say("RGA2 with dst in SYSTEM heap (source dma32)",
        blit(frame, outsys, kRga2, IM_YUV_TO_RGB_BT601_LIMIT), false);
  }

  std::printf("8. RGB888 -> NV12 in BT.709, pinned to RGA3\n");
  {
    Buf src = rgb(DmaBuf::Heap::System, 640, 360);
    Buf dst = nv12(DmaBuf::Heap::System, 640, 360);
    say("RGB -> YUV BT.709 limited on RGA3", blit(src, dst, kRga3, IM_RGB_TO_YUV_BT709_LIMIT),
        true);
    say("RGB -> YUV BT.709 limited, default scheduling",
        blit(src, dst, IM_SCHEDULER_DEFAULT, IM_RGB_TO_YUV_BT709_LIMIT), true);
  }

  std::printf("9. RGA2-only formats and ratios, on dma32 buffers\n");
  {
    Buf g = gray(DmaBuf::Heap::SystemDma32, 640, 360);
    Buf c = rgb(DmaBuf::Heap::SystemDma32, 640, 360);
    say("GRAY8 -> RGB888 on RGA2", blit(g, c, kRga2, IM_YUV_TO_RGB_BT601_FULL), true);
    Buf g2 = gray(DmaBuf::Heap::SystemDma32, 320, 180);
    say("GRAY8 -> GRAY8 half size on RGA2", blit(g, g2, kRga2), true);
    Buf gsys = gray(DmaBuf::Heap::System, 640, 360);
    say("GRAY8 (system heap) -> RGB888 on RGA2", blit(gsys, c, kRga2, IM_YUV_TO_RGB_BT601_FULL),
        false);
    say("GRAY8 -> RGB888 pinned to RGA3", blit(g, c, kRga3, IM_YUV_TO_RGB_BT601_FULL), false);
    Buf small = rgb(DmaBuf::Heap::SystemDma32, 64, 64);
    Buf big = rgb(DmaBuf::Heap::SystemDma32, 768, 768);
    say("RGB 64x64 -> 768x768 (12x) on RGA2", blit(small, big, kRga2), true);
    say("RGB 64x64 -> 768x768 (12x) on RGA3", blit(small, big, kRga3), false);
  }
  std::printf("10. per-call fd import vs a handle imported once (NV12 1080p -> RGB888 640x640)\n");
  {
    paintNv12(frame);
    Buf out = rgb(DmaBuf::Heap::System, 640, 640);
    const im_rect drect{0, 140, 640, 360};
    constexpr int kIters = 100;
    // warm up
    for (int i = 0; i < 3; ++i) blit(frame, out, kRga3, IM_YUV_TO_RGB_BT601_LIMIT, drect);
    auto t = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) blit(frame, out, kRga3, IM_YUV_TO_RGB_BT601_LIMIT, drect);
    const double per_fd = ms(t) / kIters;
    // the same, with both buffers imported once and wrapped by handle
    const rga_buffer_handle_t hs = importbuffer_fd(frame.b.fd(), static_cast<int>(frame.b.size()));
    const rga_buffer_handle_t hd = importbuffer_fd(out.b.fd(), static_cast<int>(out.b.size()));
    Buf fh, oh;  // descriptors only; the DmaBufs stay owned by `frame` / `out`
    fh.w = frame.w, fh.h = frame.h, fh.fmt = frame.fmt;
    oh.w = out.w, oh.h = out.h, oh.fmt = out.fmt;
    fh.r = wrapbuffer_handle_t(hs, frame.w, frame.h, frame.w, frame.h, frame.fmt);
    oh.r = wrapbuffer_handle_t(hd, out.w, out.h, out.w, out.h, out.fmt);
    for (int i = 0; i < 3; ++i) blit(fh, oh, kRga3, IM_YUV_TO_RGB_BT601_LIMIT, drect);
    t = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) blit(fh, oh, kRga3, IM_YUV_TO_RGB_BT601_LIMIT, drect);
    const double per_handle = ms(t) / kIters;
    // and without the up-front imcheck, to see what the check itself costs
    t = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
      rga_buffer_t src = fh.r, dst = oh.r, pat{};
      imsetColorSpace(&src, IM_YUV_TO_RGB_BT601_LIMIT);
      imsetColorSpace(&dst, IM_YUV_TO_RGB_BT601_LIMIT);
      const im_rect srect{0, 0, frame.w, frame.h};
      im_rect prect{};
      im_opt_t opt{};
      opt.core = kRga3;
      improcess(src, dst, pat, srect, drect, prect, -1, nullptr, &opt, IM_SYNC);
    }
    const double per_handle_nocheck = ms(t) / kIters;
    std::printf("  wrapbuffer_fd per call     %.3f ms/op\n", per_fd);
    std::printf("  handle imported once       %.3f ms/op\n", per_handle);
    std::printf("  handle, no imcheck         %.3f ms/op\n", per_handle_nocheck);
    // overlay by handle
    std::vector<im_rect> rects;
    for (int i = 0; i < 8; ++i) rects.push_back(im_rect{20 + i * 200, 20 + i * 100, 160, 120});
    core(kRga2);  // the blits above left this thread pinned to RGA3
    t = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) imrectangleArray(frame.r, rects.data(), 8, 0xFF00FF00u, 2, 1, nullptr);
    const double rect_fd = ms(t) / 20;
    t = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) imrectangleArray(fh.r, rects.data(), 8, 0xFF00FF00u, 2, 1, nullptr);
    const double rect_h = ms(t) / 20;
    std::printf("  imrectangleArray 8 boxes: fd %.3f ms, handle %.3f ms\n", rect_fd, rect_h);
    releasebuffer_handle(hs);
    releasebuffer_handle(hd);
  }
  return 0;
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
#else
int main() {
  std::printf("built without librga (RCDL_HAVE_RGA off)\n");
  return 0;
}
#endif
