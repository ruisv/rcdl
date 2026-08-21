// Top-k image classification on a still image.
//
//   ./cls_demo model.rknn image.jpg [--topk N] [--labels synset.txt]
//
// One rcdl::Classifier drives the board: RGA crops the ImageNet centre square
// out of the frame and scales it straight into the NPU's input tensor (no
// intermediate canvas, no CPU copy), the NPU runs the model, and the CPU
// softmaxes + sorts the score vector. The crop is the resize-shorter-side-then-
// centre-crop recipe classifiers are evaluated with, NOT a letterbox: padding
// bars would be pixels the model never saw in training.
//
// `--labels` takes the file the model ships with — one class name per line in
// class-index order, `nXXXXXXXX ` wnid prefixes stripped automatically. Without
// it predictions print as bare class ids.
//
// Image I/O is the one thing RCDL leaves to OpenCV, so this example needs a
// build where OpenCV was found.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

#if RCDL_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#endif

int main(int argc, char** argv) {
#if !RCDL_HAVE_OPENCV
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "cls_demo needs OpenCV to decode the image, and this build was configured "
               "without it. Install OpenCV (headers + libs) and re-run cmake, or drive "
               "rcdl::Classifier from your own decoder via rcdl::ImageView.\n");
  return 1;
#else
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.rknn image.jpg [--topk N] [--labels synset.txt]\n",
                 argv[0]);
    return 1;
  }
  int top_k = 5;
  std::string label_file;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--topk" && i + 1 < argc) {
      top_k = std::atoi(argv[++i]);
    } else if (arg == "--labels" && i + 1 < argc) {
      label_file = argv[++i];
    } else {
      std::fprintf(stderr, "error: unknown argument %s\n", argv[i]);
      return 1;
    }
  }

  try {
    cv::Mat bgr = cv::imread(argv[2], cv::IMREAD_COLOR);
    if (bgr.empty()) {
      std::fprintf(stderr, "error: cannot read image %s\n", argv[2]);
      return 1;
    }

    std::vector<std::string> labels;
    if (!label_file.empty()) labels = rcdl::loadClassLabels(label_file);

    rcdl::Engine engine(argv[1]);
    rcdl::ClsConfig cfg;
    cfg.top_k = top_k;
    // Defaults: RGB888 model input (the rknn_model_zoo export order), 0.875
    // crop margin (the ImageNet eval convention), auto preproc backend.
    rcdl::Classifier classifier(engine, cfg);

    std::printf("rcdl %s | librknnrt %s | driver %s\n", RCDL_VERSION_STRING,
                engine.sdkVersion().c_str(), engine.driverVersion().c_str());
    std::printf("model: %s\n", engine.path().c_str());
    std::printf("image: %s  %dx%d BGR\n", argv[2], bgr.cols, bgr.rows);
    std::printf("input: %dx%d   classes: %d   labels: %s\n", classifier.inputWidth(),
                classifier.inputHeight(), classifier.numClasses(),
                labels.empty() ? "(none)" : label_file.c_str());

    // cv::imread rows may be padded, so describe the Mat by its own stride
    // (in PIXELS) instead of assuming packed rows.
    const rcdl::ImageView src =
        rcdl::hostView(bgr.data, bgr.cols, bgr.rows, rcdl::PixelFormat::BGR888,
                       static_cast<int>(bgr.step / 3), bgr.rows);

    const rcdl::CropBox crop = classifier.cropFor(bgr.cols, bgr.rows);
    const std::vector<rcdl::ClsResult> res = classifier.classify(src);

    std::printf("preproc: %s   crop: %dx%d at (%d,%d)\n",
                rcdl::backendName(classifier.lastBackend()), crop.w, crop.h, crop.x, crop.y);
    std::printf("top-%d:\n", static_cast<int>(res.size()));
    for (const rcdl::ClsResult& r : res) {
      if (labels.empty()) {
        std::printf("  %4d  %.4f\n", r.class_id, r.score);
      } else {
        std::printf("  %4d  %.4f  %s\n", r.class_id, r.score,
                    rcdl::classLabel(labels, r.class_id).c_str());
      }
    }

    // --- steady-state timing --------------------------------------------------
    // The first call paid for RGA context setup and first-touch page faults;
    // re-run to get numbers worth quoting.
    const int iters = 10;
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < iters; ++k) (void)classifier.classify(src);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
        iters;
    std::printf("mean over %d frames: %.3f ms (%.1f FPS)   npu time of last infer: %.3f ms\n",
                iters, wall_ms, 1000.0 / wall_ms, engine.lastRunMicros() / 1000.0);
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
