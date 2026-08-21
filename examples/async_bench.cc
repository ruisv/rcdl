// M3 headline benchmark: how much throughput the multi-core async pipeline buys
// over the synchronous one, on the same model and the same frame.
//
//   ./async_bench model.rknn image.jpg [--workers N] [--frames N] [--sync|--async]
//
// The synchronous DetectionPipeline runs letterbox -> infer -> decode back to
// back on one NPU context, so its frame time is the SUM of the three stages.
// AsyncDetectionPipeline duplicates the model into N contexts pinned one per
// NPU core: the caller's thread letterboxes frame k+1 straight into a free
// context's input tensor while N workers infer and decode frames k, k-1, ...
// Its frame time is therefore bounded by
//
//     max( preproc , (infer + postproc) / workers )
//
// which is what the "throughput bound" line below prints, next to how close the
// measured number gets to it. Both pipelines are driven with the SAME frame, so
// the difference is scheduling, not work.
//
// Image decode is the one thing RCDL leaves to OpenCV, so this example needs a
// build where OpenCV was found.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"
#include "rcdl/pipeline/async_detection_pipeline.h"

#if RCDL_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#endif

#if RCDL_HAVE_OPENCV
namespace {

struct Args {
  const char* model = nullptr;
  const char* image = nullptr;
  int workers = 3;
  int frames = 300;
  bool run_sync = true;
  bool run_async = true;
};

bool parseArgs(int argc, char** argv, Args& a) {
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const char* v = argv[i];
    if (std::strcmp(v, "--workers") == 0 && i + 1 < argc) {
      a.workers = std::atoi(argv[++i]);
    } else if (std::strcmp(v, "--frames") == 0 && i + 1 < argc) {
      a.frames = std::atoi(argv[++i]);
    } else if (std::strcmp(v, "--sync") == 0) {
      a.run_async = false;
    } else if (std::strcmp(v, "--async") == 0) {
      a.run_sync = false;
    } else if (v[0] == '-') {
      return false;
    } else if (positional == 0) {
      a.model = v;
      ++positional;
    } else if (positional == 1) {
      a.image = v;
      ++positional;
    } else {
      return false;
    }
  }
  return a.model != nullptr && a.image != nullptr && a.workers > 0 && a.frames > 0;
}

double msSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

/// Per-frame view of the stage times accumulated over one measured window.
struct Stages {
  double preproc = 0, infer = 0, postproc = 0;
  double sum() const { return preproc + infer + postproc; }
};

/// The async pipeline's profile cannot be reset, so a measured window is the
/// DIFFERENCE of two snapshots — which is also how you would sample it live.
Stages between(const rcdl::StageProfile& before, const rcdl::StageProfile& after) {
  const double n = static_cast<double>(after.frames - before.frames);
  Stages s;
  if (n > 0) {
    s.preproc = (after.preproc_ms - before.preproc_ms) / n;
    s.infer = (after.infer_ms - before.infer_ms) / n;
    s.postproc = (after.postproc_ms - before.postproc_ms) / n;
  }
  return s;
}

bool sameDetections(const std::vector<rcdl::Detection>& a,
                    const std::vector<rcdl::Detection>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].class_id != b[i].class_id) return false;
    if (std::fabs(a[i].score - b[i].score) > 1e-4f) return false;
    if (std::fabs(a[i].x1 - b[i].x1) > 1e-2f || std::fabs(a[i].y1 - b[i].y1) > 1e-2f ||
        std::fabs(a[i].x2 - b[i].x2) > 1e-2f || std::fabs(a[i].y2 - b[i].y2) > 1e-2f) {
      return false;
    }
  }
  return true;
}

}  // namespace
#endif  // RCDL_HAVE_OPENCV

int main(int argc, char** argv) {
#if !RCDL_HAVE_OPENCV
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "async_bench needs OpenCV to decode the input image, and this build was "
               "configured without it. Install OpenCV (headers + libs) and re-run cmake, or "
               "drive rcdl::AsyncDetectionPipeline from your own decoder via rcdl::ImageView.\n");
  return 1;
#else
  Args args;
  if (!parseArgs(argc, argv, args)) {
    std::fprintf(stderr,
                 "usage: %s model.rknn image.jpg [--workers N] [--frames N] [--sync|--async]\n"
                 "  --workers N  NPU contexts / worker threads (default 3, one per RK3588 core)\n"
                 "  --frames  N  timed frames per pipeline (default 300)\n"
                 "  --sync       run only the synchronous baseline\n"
                 "  --async      run only the async pipeline\n",
                 argv[0]);
    return 1;
  }

  try {
    cv::Mat bgr = cv::imread(args.image, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::fprintf(stderr, "error: cannot read image %s\n", args.image);
      return 1;
    }
    // cv::imread rows may be padded, so describe the Mat by its own stride
    // (in PIXELS) rather than assuming packed rows.
    const rcdl::ImageView src =
        rcdl::hostView(bgr.data, bgr.cols, bgr.rows, rcdl::PixelFormat::BGR888,
                       static_cast<int>(bgr.step / 3), bgr.rows);

    rcdl::Engine engine(args.model);
    const rcdl::PipelineConfig base_cfg;  // RGB888 input, auto head, auto backend

    std::printf("rcdl %s | librknnrt %s | driver %s\n", RCDL_VERSION_STRING,
                engine.sdkVersion().c_str(), engine.driverVersion().c_str());
    std::printf("model : %s\n", engine.path().c_str());
    std::printf("image : %s  %dx%d BGR\n", args.image, bgr.cols, bgr.rows);
    std::printf("frames: %d timed per pipeline\n", args.frames);

    double sync_fps = 0.0, async_fps = 0.0;
    int async_workers = args.workers;
    Stages sync_stages, async_stages;
    std::vector<rcdl::Detection> sync_dets, async_dets;

    // ---------------- synchronous baseline: one context, stages in series -----
    if (args.run_sync) {
      rcdl::DetectionPipeline sync(engine, base_cfg);
      const rcdl::PipelineConfig& cfg = sync.config();
      std::printf("canvas: %dx%d %s   head: %s\n", cfg.input_w, cfg.input_h,
                  rcdl::formatName(cfg.model_input), rcdl::headName(cfg.head));

      for (int i = 0; i < 10; ++i) sync_dets = sync.process(src);  // warm-up
      std::printf("preproc backend: %s\n", rcdl::backendName(sync.lastBackend()));

      sync.resetProfile();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < args.frames; ++i) sync_dets = sync.process(src);
      const double wall = msSince(t0);

      const rcdl::StageProfile& p = sync.profile();
      sync_stages.preproc = p.preprocPerFrame();
      sync_stages.infer = p.inferPerFrame();
      sync_stages.postproc = p.postprocPerFrame();
      sync_fps = 1000.0 * args.frames / wall;
      std::printf("\nsync  : %d frames | wall %.1f ms | %.3f ms/frame | %.1f fps\n", args.frames,
                  wall, wall / args.frames, sync_fps);
      std::printf("        stages/frame: preproc %.3f | infer %.3f | postproc %.3f | sum %.3f ms\n",
                  sync_stages.preproc, sync_stages.infer, sync_stages.postproc,
                  sync_stages.sum());
    }

    // ---------------- async: N pinned contexts, stages overlapped -------------
    if (args.run_async) {
      rcdl::AsyncConfig acfg;
      acfg.workers = args.workers;
      acfg.pin_cores = true;
      rcdl::AsyncDetectionPipeline async(engine, base_cfg, acfg);
      const int workers = async.workers();
      async_workers = workers;
      if (!args.run_sync) {  // the sync branch prints this when it runs
        const rcdl::PipelineConfig& cfg = async.config();
        std::printf("canvas: %dx%d %s   head: %s\n", cfg.input_w, cfg.input_h,
                    rcdl::formatName(cfg.model_input), rcdl::headName(cfg.head));
      }

      // PRIME the pipeline and keep it primed: submit `workers` frames without
      // draining, so every context is busy before the clock starts. Draining
      // down to one in-flight frame first would serialise the stages again and
      // hide exactly the overlap this bench exists to measure.
      for (int i = 0; i < workers; ++i) async.submit(src);
      for (int i = 0; i < 10; ++i) {  // warm-up, still `workers` in flight
        async.submit(src);
        async.next(async_dets);
      }

      const rcdl::StageProfile p0 = async.profile();
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < args.frames; ++i) {
        async.submit(src);  // blocks only when all `workers` contexts are busy
        async.next(async_dets);
      }
      const double wall = msSince(t0);
      const rcdl::StageProfile p1 = async.profile();

      async.finish();
      std::vector<rcdl::Detection> tail;
      while (async.next(tail)) { /* the last `workers` frames still in flight */ }

      async_stages = between(p0, p1);
      async_fps = 1000.0 * args.frames / wall;
      // What the shape of the pipeline allows: the caller's letterbox runs once
      // per frame on one thread, while infer+decode is spread over `workers`.
      const double bound =
          std::fmax(async_stages.preproc, (async_stages.infer + async_stages.postproc) / workers);

      std::printf("\nasync : %d frames | wall %.1f ms | %.3f ms/frame | %.1f fps  (%d workers%s)\n",
                  args.frames, wall, wall / args.frames, async_fps, workers,
                  acfg.pin_cores ? ", pinned one per core" : "");
      std::printf("        service/frame, timed on the thread that ran it:\n");
      std::printf("          preproc  %.3f ms  (caller thread: letterbox into the context tensor)\n",
                  async_stages.preproc);
      std::printf("          infer    %.3f ms  (worker threads)\n", async_stages.infer);
      std::printf("          postproc %.3f ms  (worker threads)\n", async_stages.postproc);
      std::printf("        sum of services %.3f ms/frame overlapped into %.3f ms of wall time\n",
                  async_stages.sum(), wall / args.frames);
      if (bound > 0) {
        std::printf("        throughput bound: max(preproc %.3f, (infer+postproc)/%d = %.3f)"
                    " = %.3f ms => %.1f fps  (reached %.0f%%)\n",
                    async_stages.preproc, workers,
                    (async_stages.infer + async_stages.postproc) / workers, bound, 1000.0 / bound,
                    100.0 * bound / (wall / args.frames));
      }
    }

    // ---------------- the headline ------------------------------------------
    if (args.run_sync && args.run_async) {
      std::printf("\nspeed-up: %.2fx over the synchronous pipeline (ideal ceiling with %d"
                  " workers: %.2fx)\n",
                  async_fps / sync_fps, async_workers,
                  sync_stages.sum() > 0
                      ? sync_stages.sum() /
                            std::fmax(sync_stages.preproc,
                                      (sync_stages.infer + sync_stages.postproc) / async_workers)
                      : 0.0);
      const bool same = sameDetections(sync_dets, async_dets);
      std::printf("detections: sync %zu | async %zu | identical: %s\n", sync_dets.size(),
                  async_dets.size(), same ? "yes" : "NO");
      if (!same) {
        std::fprintf(stderr,
                     "error: the two pipelines disagree on the same frame — run async_check\n");
        return 1;
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
#endif  // RCDL_HAVE_OPENCV
}
