// M3 correctness gate: the async pipeline must return EXACTLY what the
// synchronous one returns, for the same frames, in SUBMISSION order.
//
//   ./async_check model.rknn image.jpg [--frames N] [--workers N]
//
// The async pipeline letterboxes on the caller's thread into one of N NPU
// contexts and decodes on N worker threads, so two things can go wrong that the
// synchronous pipeline cannot get wrong:
//
//   - aliasing: a worker decodes another context's outputs, or a caller
//     letterboxes into a context that is still inferring;
//   - reordering: results come back in completion order rather than submission
//     order (frames on different cores finish at different times).
//
// Both are invisible when every frame is the same picture, so this check builds
// a set of VISIBLY different frames (crops, a mirror, a blank) from the input
// image, records the synchronous detections for each as the reference, then
// cycles them through the async pipeline and compares field by field. A swap of
// two adjacent results is a mismatch, and the report says which frame the
// result actually belonged to.
//
// Image decode is the one thing RCDL leaves to OpenCV, so this example needs a
// build where OpenCV was found.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"
#include "rcdl/pipeline/async_detection_pipeline.h"

#if RCDL_HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

#if RCDL_HAVE_OPENCV
namespace {

struct Args {
  const char* model = nullptr;
  const char* image = nullptr;
  int frames = 64;
  int workers = 3;
};

bool parseArgs(int argc, char** argv, Args& a) {
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const char* v = argv[i];
    if (std::strcmp(v, "--frames") == 0 && i + 1 < argc) {
      a.frames = std::atoi(argv[++i]);
    } else if (std::strcmp(v, "--workers") == 0 && i + 1 < argc) {
      a.workers = std::atoi(argv[++i]);
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
  return a.model != nullptr && a.image != nullptr && a.frames > 0 && a.workers > 0;
}

/// One test frame plus a name for the report.
struct Variant {
  std::string name;
  cv::Mat bgr;  ///< owns its pixels: every variant stays alive for the whole run
  rcdl::ImageView view() const {
    // cv::Mat rows may be padded; describe the stride in PIXELS, as RGA wants.
    return rcdl::hostView(const_cast<unsigned char*>(bgr.data), bgr.cols, bgr.rows,
                          rcdl::PixelFormat::BGR888, static_cast<int>(bgr.step / 3), bgr.rows);
  }
};

/// A handful of frames that a detector sees differently: progressive crops
/// (which change both the content and the letterbox geometry), a mirror, and a
/// blank one that should produce nothing at all.
std::vector<Variant> makeVariants(const cv::Mat& src) {
  std::vector<Variant> v;
  v.push_back({"full", src.clone()});
  for (int k = 1; k <= 4; ++k) {
    const int mx = (src.cols * k) / 12;
    const int my = (src.rows * k) / 16;
    const cv::Rect roi(mx, my, src.cols - 2 * mx, src.rows - 2 * my);
    if (roi.width > 16 && roi.height > 16) {
      v.push_back({"crop" + std::to_string(k), src(roi).clone()});
    }
  }
  cv::Mat flipped;
  cv::flip(src, flipped, 1);
  v.push_back({"mirror", flipped});
  v.push_back({"blank", cv::Mat(src.rows, src.cols, CV_8UC3, cv::Scalar(0, 0, 0))});
  return v;
}

/// Field-by-field comparison. Both pipelines run the same model over the same
/// letterboxed pixels, so the floats should be bit-identical; the epsilons only
/// absorb a differing summation order, never a different detection.
bool sameDetections(const std::vector<rcdl::Detection>& a, const std::vector<rcdl::Detection>& b,
                    double* worst = nullptr) {
  if (a.size() != b.size()) return false;
  bool ok = true;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double ds = std::fabs(static_cast<double>(a[i].score) - b[i].score);
    const double dx = std::fmax(std::fabs(static_cast<double>(a[i].x1) - b[i].x1),
                                std::fabs(static_cast<double>(a[i].x2) - b[i].x2));
    const double dy = std::fmax(std::fabs(static_cast<double>(a[i].y1) - b[i].y1),
                                std::fabs(static_cast<double>(a[i].y2) - b[i].y2));
    if (worst) *worst = std::fmax(*worst, std::fmax(ds, std::fmax(dx, dy)));
    if (a[i].class_id != b[i].class_id || ds > 1e-5 || dx > 1e-3 || dy > 1e-3) ok = false;
  }
  return ok;
}

/// Which reference (if any) a result actually matches — names the frame an
/// out-of-order delivery really came from.
int matchingVariant(const std::vector<std::vector<rcdl::Detection>>& refs,
                    const std::vector<rcdl::Detection>& got) {
  for (std::size_t i = 0; i < refs.size(); ++i) {
    if (sameDetections(refs[i], got)) return static_cast<int>(i);
  }
  return -1;
}

/// Report one bad result and say whether it is a reordering (it matches some
/// OTHER submitted frame) or a wrong result (it matches nothing we submitted).
/// Returns true when the result was merely out of order.
bool reportMismatch(const char* where, int index, const std::vector<Variant>& variants,
                    const std::vector<std::vector<rcdl::Detection>>& refs,
                    const std::vector<rcdl::Detection>& got, int expect) {
  const int actual = matchingVariant(refs, got);
  const std::string blame = actual >= 0 ? "frame '" + variants[static_cast<std::size_t>(actual)].name + "'"
                                        : std::string("no submitted frame");
  std::fprintf(stderr, "  MISMATCH %s, result %d: expected '%s' (%zu dets), got %s (%zu dets)\n",
               where, index, variants[static_cast<std::size_t>(expect)].name.c_str(),
               refs[static_cast<std::size_t>(expect)].size(), blame.c_str(), got.size());
  return actual >= 0;
}

}  // namespace
#endif  // RCDL_HAVE_OPENCV

int main(int argc, char** argv) {
#if !RCDL_HAVE_OPENCV
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "async_check needs OpenCV to decode the input image, and this build was "
               "configured without it. Install OpenCV (headers + libs) and re-run cmake.\n");
  return 1;
#else
  Args args;
  if (!parseArgs(argc, argv, args)) {
    std::fprintf(stderr,
                 "usage: %s model.rknn image.jpg [--frames N] [--workers N]\n"
                 "  --frames  N  frames pushed through the async pipeline (default 64)\n"
                 "  --workers N  NPU contexts / worker threads (default 3)\n",
                 argv[0]);
    return 1;
  }

  try {
    const cv::Mat input = cv::imread(args.image, cv::IMREAD_COLOR);
    if (input.empty()) {
      std::fprintf(stderr, "error: cannot read image %s\n", args.image);
      return 1;
    }
    const std::vector<Variant> variants = makeVariants(input);

    rcdl::Engine engine(args.model);
    const rcdl::PipelineConfig base_cfg;

    // ---- reference: the synchronous pipeline, one context, one thread -------
    rcdl::DetectionPipeline sync(engine, base_cfg);
    const rcdl::PipelineConfig& cfg = sync.config();
    std::printf("model : %s\n", engine.path().c_str());
    std::printf("canvas: %dx%d %s   head: %s\n", cfg.input_w, cfg.input_h,
                rcdl::formatName(cfg.model_input), rcdl::headName(cfg.head));
    std::printf("image : %s  %dx%d BGR  -> %zu test frames\n", args.image, input.cols, input.rows,
                variants.size());

    std::vector<std::vector<rcdl::Detection>> refs;
    for (const Variant& v : variants) {
      const std::vector<rcdl::Detection> a = sync.process(v.view());
      // Run each frame twice: if the synchronous pipeline is not repeatable,
      // the comparison below proves nothing and the epsilons are wrong.
      const std::vector<rcdl::Detection> b = sync.process(v.view());
      if (!sameDetections(a, b)) {
        std::fprintf(stderr,
                     "error: the synchronous pipeline is not repeatable on frame '%s' — the "
                     "async comparison would be meaningless\n",
                     v.name.c_str());
        return 2;
      }
      std::printf("  ref %-8s %4dx%-4d -> %zu detections\n", v.name.c_str(), v.bgr.cols,
                  v.bgr.rows, a.size());
      refs.push_back(a);
    }

    // An ordering bug can only show up if neighbouring frames disagree.
    int distinct = 0;
    for (std::size_t i = 0; i < refs.size(); ++i) {
      bool seen = false;
      for (std::size_t j = 0; j < i; ++j) seen = seen || sameDetections(refs[i], refs[j]);
      if (!seen) ++distinct;
    }
    if (distinct < 2) {
      std::fprintf(stderr,
                   "error: every test frame produced the same detections (%d distinct), so this "
                   "check cannot detect reordering. Use an image the model actually fires on.\n",
                   distinct);
      return 2;
    }
    std::printf("distinct reference results: %d of %zu\n", distinct, refs.size());

    // ---- the pipeline under test -------------------------------------------
    rcdl::AsyncConfig acfg;
    acfg.workers = args.workers;
    rcdl::AsyncDetectionPipeline async(engine, base_cfg, acfg);
    const int nv = static_cast<int>(variants.size());
    const int workers = async.workers();
    std::printf("async : %d workers, %d frames submitted in a %d-frame cycle\n", workers,
                args.frames, nv);

    int produced = 0, mismatches = 0, out_of_order = 0;
    double worst = 0.0;
    std::vector<rcdl::Detection> got;

    // Keep the pipeline full: prime `workers` frames, then submit one and drain
    // one. Results must come back in submission order regardless of which core
    // finishes first.
    for (int i = 0; i < args.frames; ++i) {
      if (!async.submit(variants[i % nv].view())) {
        std::fprintf(stderr, "error: submit() refused frame %d before finish()\n", i);
        return 3;
      }
      if (i < workers) continue;  // priming: nothing to drain yet
      if (!async.next(got)) {
        std::fprintf(stderr, "error: next() returned false with %d frames still in flight\n",
                     workers);
        return 3;
      }
      const int expect = produced % nv;
      if (!sameDetections(refs[expect], got, &worst)) {
        ++mismatches;
        if (reportMismatch("in the steady state", produced, variants, refs, got, expect)) {
          ++out_of_order;
        }
      }
      ++produced;
    }

    // ---- finish / drain protocol -------------------------------------------
    async.finish();
    async.finish();  // idempotent by contract
    while (async.next(got)) {
      const int expect = produced % nv;
      if (!sameDetections(refs[expect], got, &worst)) {
        ++mismatches;
        if (reportMismatch("while draining", produced, variants, refs, got, expect)) {
          ++out_of_order;
        }
      }
      ++produced;
    }
    if (async.next(got)) {
      std::fprintf(stderr, "error: next() produced a result after the pipeline drained\n");
      return 3;
    }
    if (async.submit(variants[0].view())) {
      std::fprintf(stderr, "error: submit() accepted a frame after finish()\n");
      return 3;
    }
    if (async.inFlight() != 0) {
      std::fprintf(stderr, "error: %d frames still counted in flight after the drain\n",
                   async.inFlight());
      return 3;
    }

    std::printf("submitted=%d produced=%d mismatches=%d (of which reordered=%d)"
                "  worst field delta=%.3g\n",
                args.frames, produced, mismatches, out_of_order, worst);
    if (produced != args.frames) {
      std::fprintf(stderr, "FAIL: produced != submitted — frames were lost or duplicated\n");
      return 3;
    }
    if (mismatches != 0) {
      std::fprintf(stderr, "FAIL: %d results disagree with the synchronous reference\n",
                   mismatches);
      return 3;
    }

    const rcdl::StageProfile p = async.profile();
    std::printf("PASS: %d async results identical to the sync pipeline, in submission order.\n",
                produced);
    std::printf("      service/frame: preproc %.3f | infer %.3f | postproc %.3f ms over %llu"
                " frames\n",
                p.preprocPerFrame(), p.inferPerFrame(), p.postprocPerFrame(),
                static_cast<unsigned long long>(p.frames));
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
