// End-to-end object detection on a still image, and the M1 benchmark.
//
//   ./det_demo model.rknn image.jpg [output.jpg]
//
// One DetectionPipeline drives the whole board: RGA letterboxes the decoded
// image straight into the NPU's input tensor (no intermediate canvas, no CPU
// copy), the NPU runs the model, and the CPU decodes + NMSes the head. Boxes
// come back in ORIGINAL-image pixels. After the first frame the pipeline is
// re-run a few times with the profile reset, so the per-stage means printed at
// the end are steady-state numbers rather than first-call costs.
//
// Image I/O (decode + drawing) is the one thing RCDL leaves to OpenCV, so this
// example needs a build where OpenCV was found.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

#if RCDL_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#if RCDL_HAVE_OPENCV
namespace {

/// Stable per-class colour so the same class keeps its colour across frames.
cv::Scalar classColor(int id) {
  const int h = (id * 67) % 180;  // spread the 80 COCO classes over the hue circle
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, 220, 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const cv::Vec3b p = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(p[0], p[1], p[2]);
}

void drawDetections(cv::Mat& img, const std::vector<rcdl::Detection>& dets) {
  const double fs = 0.5;
  const int th = 1;
  for (const rcdl::Detection& d : dets) {
    const cv::Scalar color = classColor(d.class_id);
    const cv::Point tl(static_cast<int>(d.x1), static_cast<int>(d.y1));
    const cv::Point br(static_cast<int>(d.x2), static_cast<int>(d.y2));
    cv::rectangle(img, tl, br, color, 2);

    char label[128];
    std::snprintf(label, sizeof(label), "%s %.2f", rcdl::cocoClassName(d.class_id), d.score);
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, fs, th, &baseline);
    // Keep the caption inside the image when the box touches the top edge.
    const int ly = std::max(tl.y, ts.height + baseline + 2);
    cv::rectangle(img, cv::Point(tl.x, ly - ts.height - baseline - 2),
                  cv::Point(tl.x + ts.width + 4, ly), color, cv::FILLED);
    cv::putText(img, label, cv::Point(tl.x + 2, ly - baseline - 1), cv::FONT_HERSHEY_SIMPLEX, fs,
                cv::Scalar(0, 0, 0), th, cv::LINE_AA);
  }
}

}  // namespace
#endif  // RCDL_HAVE_OPENCV

int main(int argc, char** argv) {
#if !RCDL_HAVE_OPENCV
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "det_demo needs OpenCV for image decode/encode, and this build was configured "
               "without it. Install OpenCV (headers + libs) and re-run cmake, or drive "
               "rcdl::DetectionPipeline from your own decoder via rcdl::ImageView.\n");
  return 1;
#else
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.rknn image.jpg [output.jpg]\n", argv[0]);
    return 1;
  }
  try {
    cv::Mat bgr = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::fprintf(stderr, "error: cannot read image %s\n", argv[2]);
      return 1;
    }

    rcdl::Engine engine(argv[1]);
    // Defaults: RGB888 model input (the rknn_model_zoo export order), auto head,
    // auto preproc backend, 114 letterbox pad, canvas taken from the model.
    rcdl::DetectionPipeline pipeline(engine, rcdl::PipelineConfig());
    const rcdl::PipelineConfig& cfg = pipeline.config();

    std::printf("rcdl %s | librknnrt %s | driver %s\n", RCDL_VERSION_STRING,
                engine.sdkVersion().c_str(), engine.driverVersion().c_str());
    std::printf("model: %s\n", engine.path().c_str());
    std::printf("image: %s  %dx%d BGR\n", argv[2], bgr.cols, bgr.rows);
    std::printf("canvas: %dx%d %s   head: %s\n", cfg.input_w, cfg.input_h,
                rcdl::formatName(cfg.model_input), rcdl::headName(cfg.head));
    if (const rcdl::YoloLtrbDetector* ltrb = pipeline.decoder().ltrb()) {
      std::printf("head layout: %s\n", ltrb->layout().describe().c_str());
    }

    // cv::imread rows may be padded, so describe the Mat by its own stride
    // (in PIXELS) instead of assuming packed rows.
    const rcdl::ImageView src =
        rcdl::hostView(bgr.data, bgr.cols, bgr.rows, rcdl::PixelFormat::BGR888,
                       static_cast<int>(bgr.step / 3), bgr.rows);

    const std::vector<rcdl::Detection> dets = pipeline.process(src);

    const rcdl::LetterboxInfo& lb = pipeline.lastLetterbox();
    std::printf("preproc: %s   letterbox scale=%.4f pad=(%.1f,%.1f)\n",
                rcdl::backendName(pipeline.lastBackend()), lb.scale, lb.padX, lb.padY);
    std::printf("detections: %zu\n", dets.size());
    for (const rcdl::Detection& d : dets) {
      std::printf("  %-16s %.3f  %.1f %.1f %.1f %.1f\n", rcdl::cocoClassName(d.class_id), d.score,
                  d.x1, d.y1, d.x2, d.y2);
    }

    // --- steady-state timing --------------------------------------------------
    // The first call paid for RGA context setup and first-touch page faults;
    // drop it from the profile and re-run to get numbers worth quoting.
    const int iters = 10;
    pipeline.resetProfile();
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iters; ++k) (void)pipeline.process(src);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
        iters;

    const rcdl::StageProfile& p = pipeline.profile();
    std::printf("mean over %d frames: preproc %.3f ms | infer %.3f ms | postproc %.3f ms"
                " | stages %.3f ms | wall %.3f ms (%.1f FPS)\n",
                iters, p.preprocPerFrame(), p.inferPerFrame(), p.postprocPerFrame(),
                p.totalMs() / iters, wall_ms, 1000.0 / wall_ms);
    std::printf("npu time of last infer: %.3f ms (RKNN_QUERY_PERF_RUN)\n",
                engine.lastRunMicros() / 1000.0);

    if (argc > 3) {
      drawDetections(bgr, dets);
      if (!cv::imwrite(argv[3], bgr)) {
        std::fprintf(stderr, "error: cannot write %s\n", argv[3]);
        return 1;
      }
      std::printf("wrote %s\n", argv[3]);
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
