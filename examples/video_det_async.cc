// The headline pipeline, overlapped: the same VPU -> RGA -> NPU path as
// video_det_demo, but every stage on its own thread inside one class.
//
//   ./video_det_async <model.rknn> <stream.h264> [--frames N] [--conf C]
//                     [--workers W] [--depth D] [--sync]
//
//   caller  submit(bytes) ─> VPU decode ─> RGA letterbox ─> NPU x W ─> next()
//              (this thread)   (thread)      (thread)      (W threads)
//
// video_det_demo runs those stages back to back and reports what a frame costs;
// this program runs them concurrently on different frames and reports what the
// stream costs. The gap between the two is the whole point of the class, so
// `--sync` re-runs the same stream through the synchronous DetectionPipeline
// and prints both numbers side by side — measured here, not asserted.
//
// The caller never touches a frame: no VideoFrame, no letterbox, no NPU context.
// It reads bytes off a file (an ffmpeg pipe or a socket would do as well) and
// drains detections. That is what the class exists for — a thin driver, in any
// language, reaching the C++ ceiling.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

namespace {

double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

void printProfile(const char* title, const rcdl::StageProfile& sp, int frames, double wall_ms) {
  const double per_frame = frames > 0 ? wall_ms / frames : 0.0;
  std::printf("\n=== %s (%d frames) ===\n", title, frames);
  auto row = [&](const char* name, const char* unit, double per) {
    std::printf("  %-9s %-26s %7.2f ms/f\n", name, unit, per);
  };
  row("decode", "VPU  H.26x -> NV12", sp.decodePerFrame());
  row("preproc", "RGA  letterbox -> tensor", sp.preprocPerFrame());
  row("infer", "NPU  rknn_run", sp.inferPerFrame());
  row("postproc", "CPU  head decode + NMS", sp.postprocPerFrame());
  std::printf("  %-9s %-26s %7.2f ms/f  => %.1f fps\n", "= wall", "end to end", per_frame,
              per_frame > 0 ? 1000.0 / per_frame : 0.0);
}

/// The synchronous baseline, over the same bytes: decode a frame, detect on it,
/// release it, repeat. This is what video_det_demo does.
double syncRun(rcdl::Engine& engine, const rcdl::PipelineConfig& cfg,
               const rcdl::VideoDecConfig& dcfg, const std::vector<std::uint8_t>& stream,
               int max_frames, int* frames_out, long* dets_out) {
  rcdl::DetectionPipeline pipe(engine, cfg);
  rcdl::VideoDecoder dec(dcfg);
  rcdl::VideoFrame frame;
  int frames = 0;
  long dets = 0;
  std::size_t fed = 0;
  constexpr std::size_t kChunk = 128 * 1024;
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    if (dec.receive(frame, 0) || (fed >= stream.size() && dec.flush(frame))) {
      dets += static_cast<long>(pipe.process(frame.view()).size());
      frame.reset();
      if (++frames >= max_frames && max_frames > 0) break;
      continue;
    }
    if (fed >= stream.size()) break;  // flush() said the stream is over
    const std::size_t n = std::min(kChunk, stream.size() - fed);
    if (dec.feed(stream.data() + fed, n, 0, 20)) fed += n;
  }
  const double wall = msSince(t0);
  *frames_out = frames;
  *dets_out = dets;
  return wall;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <model.rknn> <stream.h264> [--frames N] [--conf C] "
                 "[--workers W] [--depth D] [--sync]\n",
                 argv[0]);
    return 1;
  }
  try {
    const std::string model_path = argv[1];
    const std::string stream_path = argv[2];
    int max_frames = 0;  // 0 => the whole stream
    float conf = 0.25f;
    int workers = 3, depth = 2;
    bool also_sync = false;
    for (int i = 3; i < argc; ++i) {
      const std::string opt = argv[i];
      const bool has_value = i + 1 < argc;
      if (opt == "--frames") {
        RCDL_REQUIRE(has_value, "--frames needs a count");
        max_frames = std::atoi(argv[++i]);
      } else if (opt == "--conf") {
        RCDL_REQUIRE(has_value, "--conf needs a threshold");
        conf = static_cast<float>(std::atof(argv[++i]));
        RCDL_REQUIRE(conf > 0.0f && conf < 1.0f, "--conf must be in (0,1)");
      } else if (opt == "--workers") {
        RCDL_REQUIRE(has_value, "--workers needs a count");
        workers = std::atoi(argv[++i]);
      } else if (opt == "--depth") {
        RCDL_REQUIRE(has_value, "--depth needs a count");
        depth = std::atoi(argv[++i]);
      } else if (opt == "--sync") {
        also_sync = true;
      } else {
        RCDL_REQUIRE(false, "unknown option (see usage)");
      }
    }

    // Read the whole elementary stream up front: this program measures the
    // pipeline, and file I/O in the feed loop would be measuring the disk.
    std::vector<std::uint8_t> stream;
    {
      std::FILE* fp = std::fopen(stream_path.c_str(), "rb");
      RCDL_REQUIRE(fp != nullptr, "cannot open the input stream");
      std::uint8_t buf[64 * 1024];
      std::size_t n;
      while ((n = std::fread(buf, 1, sizeof buf, fp)) > 0) stream.insert(stream.end(), buf, buf + n);
      std::fclose(fp);
    }
    RCDL_REQUIRE(!stream.empty(), "the input stream is empty");

    rcdl::Engine engine(model_path);
    rcdl::PipelineConfig pcfg;
    pcfg.detect.conf_thresh = pcfg.ltrb.conf_thresh = conf;
    rcdl::VideoDecConfig dcfg;
    RCDL_REQUIRE(rcdl::codecFromExtension(stream_path, &dcfg.codec),
                 "cannot guess the codec from the stream's extension");
    rcdl::VideoAsyncConfig vcfg;
    vcfg.async.workers = workers;
    vcfg.queue_depth = depth;

    std::printf("rcdl %s | librknnrt %s | driver %s\n", RCDL_VERSION_STRING,
                engine.sdkVersion().c_str(), engine.driverVersion().c_str());
    std::printf("model : %s\nstream: %s (%s, %zu KiB)\n", engine.path().c_str(),
                stream_path.c_str(), rcdl::codecName(dcfg.codec), stream.size() / 1024);
    std::printf("async : %d NPU contexts, queue depth %d, conf %.2f\n\n", workers, depth, conf);

    rcdl::AsyncVideoDetectionPipeline pipe(engine, pcfg, dcfg, vcfg);

    // The whole driver: push bytes, pull detections. tryNext() never blocks, so
    // results are drained between feeds and the reorder buffer stays shallow.
    std::vector<rcdl::Detection> dets;
    int frames = 0;
    long total_dets = 0;
    bool stop = false;
    constexpr std::size_t kChunk = 128 * 1024;
    const auto t0 = std::chrono::steady_clock::now();
    auto take = [&]() {
      total_dets += static_cast<long>(dets.size());
      if (frames == 0) {
        std::printf("frame 1 (pts %llu): %zu detection(s)\n",
                    static_cast<unsigned long long>(pipe.lastPtsUs()), dets.size());
        for (const rcdl::Detection& d : dets) {
          std::printf("  %-16s %.3f  %.1f %.1f %.1f %.1f\n", rcdl::cocoClassName(d.class_id),
                      d.score, d.x1, d.y1, d.x2, d.y2);
        }
      }
      if (++frames >= max_frames && max_frames > 0) stop = true;
    };
    for (std::size_t off = 0; off < stream.size() && !stop; off += kChunk) {
      const std::size_t n = std::min(kChunk, stream.size() - off);
      // submit() returning false is back-pressure: the decoder's input queue is
      // full because the pipeline is waiting on US to take results. Drain, then
      // offer the same bytes again. Blocking here instead would deadlock — the
      // one thread that can make room is this one.
      while (!pipe.submit(stream.data() + off, n)) {
        if (pipe.finished()) { stop = true; break; }
        while (!stop && pipe.tryNext(dets)) take();
        if (stop) break;
      }
      while (!stop && pipe.tryNext(dets)) take();
    }
    pipe.finish();
    while (!stop && pipe.next(dets)) take();
    const double wall = msSince(t0);
    RCDL_REQUIRE(frames > 0, "no frames decoded — wrong codec, or not an elementary stream?");

    std::printf("\ndecoded %dx%d, %llu frames through the VPU (%s buffer group)\n", pipe.width(),
                pipe.height(), static_cast<unsigned long long>(pipe.framesDecoded()),
                pipe.usingExternalBuffers() ? "external" : "MPP-internal");
    printProfile("async: stages overlap, so they sum to more than wall time", pipe.profile(),
                 frames, wall);
    std::printf("detections: %ld total, %.1f per frame\n", total_dets,
                static_cast<double>(total_dets) / frames);

    if (also_sync) {
      int sync_frames = 0;
      long sync_dets = 0;
      const double sync_wall =
          syncRun(engine, pipe.config(), dcfg, stream, frames, &sync_frames, &sync_dets);
      std::printf("\nsynchronous baseline: %d frames, %.1f fps (%.2f ms/f), %ld detections\n",
                  sync_frames, sync_frames * 1000.0 / sync_wall, sync_wall / sync_frames,
                  sync_dets);
      std::printf("speed-up: %.2fx  (%.1f fps async vs %.1f fps sync)\n",
                  (sync_wall / sync_frames) / (wall / frames), frames * 1000.0 / wall,
                  sync_frames * 1000.0 / sync_wall);
      if (sync_frames == frames) {
        std::printf("detection totals: %ld async vs %ld sync%s\n", total_dets, sync_dets,
                    total_dets == sync_dets ? "  (identical)" : "  (DIFFERENT — investigate)");
      }
    }
    return 0;
  } catch (const rcdl::Error& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
