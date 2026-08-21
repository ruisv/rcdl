// Monocular depth estimation on a still image.
//
//   ./depth_demo model.rknn image.jpg [--out out.png] [--inverse] [--blend]
//
// Same front half as every other RCDL task: RGA letterboxes the decoded image
// straight into the NPU's input tensor, one infer(), and the head is decoded on
// the CPU. rcdl::DepthEstimator::postprocess(lb) then projects the map back onto
// the ORIGINAL frame — each source pixel centre is pushed forward through the
// letterbox and sampled, so the padding never contributes.
//
// `--inverse` is for the DISPARITY heads (MiDaS-style): they emit 1/depth, so
// near is large; inverting turns the map back into something monotone in metres
// before it is normalised. Which one a model emits is a property of the export,
// not something that can be sniffed, so it is a flag.
//
// The rendered map is the Turbo colourmap over the map's own observed range —
// blue near, red far after `--inverse`, and the other way round without it.
//
// Image I/O (decode + encode) is the one thing RCDL leaves to OpenCV, so this
// example needs a build where OpenCV was found.

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

#if RCDL_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

int main(int argc, char** argv) {
#if !RCDL_HAVE_OPENCV
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "depth_demo needs OpenCV for image decode/encode, and this build was configured "
               "without it. Install OpenCV (headers + libs) and re-run cmake, or drive "
               "rcdl::DepthEstimator from your own decoder via rcdl::ImageView.\n");
  return 1;
#else
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.rknn image.jpg [--out out.png] [--inverse] [--blend]\n",
                 argv[0]);
    return 1;
  }
  std::string out_path;
  bool inverse = false;
  bool blend = false;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--inverse") {
      inverse = true;
    } else if (arg == "--blend") {
      blend = true;
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

    rcdl::Engine engine(argv[1]);
    rcdl::DepthConfig cfg;
    cfg.inverse = inverse;
    rcdl::DepthEstimator depth(engine, cfg);

    std::printf("rcdl %s | librknnrt %s | driver %s\n", RCDL_VERSION_STRING,
                engine.sdkVersion().c_str(), engine.driverVersion().c_str());
    std::printf("model: %s\n", engine.path().c_str());
    std::printf("image: %s  %dx%d BGR\n", argv[2], bgr.cols, bgr.rows);
    std::printf("output: %s %s   inverse: %s\n", engine.outputName(0).c_str(),
                rcdl::dtypeName(engine.outputType(0)), inverse ? "yes" : "no");

    // cv::imread rows may be padded, so describe the Mat by its own stride
    // (in PIXELS) instead of assuming packed rows.
    const rcdl::ImageView src =
        rcdl::hostView(bgr.data, bgr.cols, bgr.rows, rcdl::PixelFormat::BGR888,
                       static_cast<int>(bgr.step / 3), bgr.rows);
    const rcdl::ImageView input = rcdl::engineInputView(engine, 0, rcdl::PixelFormat::RGB888);

    rcdl::PreprocBackend used = rcdl::PreprocBackend::Auto;
    const auto t0 = std::chrono::steady_clock::now();
    const rcdl::LetterboxInfo lb = rcdl::letterbox(input, src, 114, rcdl::PreprocBackend::Auto,
                                                   rcdl::YuvRange::kStudioToFull, &used);
    const auto t1 = std::chrono::steady_clock::now();
    engine.infer();
    const auto t2 = std::chrono::steady_clock::now();
    const rcdl::DepthMap map = depth.postprocess(lb);
    const auto t3 = std::chrono::steady_clock::now();

    const auto ms = [](std::chrono::steady_clock::time_point a,
                       std::chrono::steady_clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("preproc: %s   letterbox scale=%.4f pad=(%.1f,%.1f)\n", rcdl::backendName(used),
                lb.scale, lb.padX, lb.padY);
    std::printf("depth map: %dx%d   raw range [%.4f, %.4f]   centre %.4f\n", map.width, map.height,
                map.vmin, map.vmax, map.at(map.width / 2, map.height / 2));
    std::printf("preproc %.3f ms | infer %.3f ms | postproc %.3f ms | npu %.3f ms\n", ms(t0, t1),
                ms(t1, t2), ms(t2, t3), engine.lastRunMicros() / 1000.0);

    if (!out_path.empty()) {
      // depthColorize() already emits BGR in OpenCV's channel order, so the
      // buffer can be wrapped as a cv::Mat without a conversion.
      const std::vector<std::uint8_t> colored = rcdl::depthColorize(map);
      if (colored.size() != static_cast<std::size_t>(map.width) * map.height * 3) {
        std::fprintf(stderr, "error: colorized map does not cover the depth map\n");
        return 1;
      }
      cv::Mat vis(map.height, map.width, CV_8UC3, const_cast<std::uint8_t*>(colored.data()));
      cv::Mat out;
      if (blend) {
        cv::addWeighted(bgr, 0.5, vis, 0.5, 0.0, out);
      } else {
        out = vis;
      }
      if (!cv::imwrite(out_path, out)) {
        std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
        return 1;
      }
      std::printf("wrote %s\n", out_path.c_str());
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
