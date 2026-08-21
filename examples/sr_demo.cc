// x4 super-resolution of a whole image, by tiling.
//
//   ./sr_demo model.rknn image.jpg [--out up.png] [--overlap N] [--max-side N]
//             [--check]
//
// The model upscales one fixed 128x128 tile; rcdl::SuperResolver cuts the image
// into overlapping tiles, runs them, and cross-fades the results back together.
// Cost is linear in the tile count, which this prints — a 640x480 source is 20
// tiles, and each tile is a whole inference.
//
// `--check` shrinks the source by the model's own factor first and then upscales
// it, so the original IS the right answer and the demo can score itself. Read
// the numbers it prints with the perception/distortion trade-off in mind: a
// perceptually-trained upscaler comes out BELOW a bicubic resize on PSNR while
// looking obviously sharper, which is why edge energy is printed beside it.
//
// Image I/O (decode + encode) is the one thing RCDL leaves to OpenCV, so this
// example needs a build where OpenCV was found.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

#if RCDL_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

double psnr(const cv::Mat& a, const cv::Mat& b) {
  cv::Mat d;
  cv::absdiff(a, b, d);
  d.convertTo(d, CV_64F);
  const double mse = cv::mean(d.mul(d))[0] + cv::mean(d.mul(d))[1] + cv::mean(d.mul(d))[2];
  const double m = mse / 3.0;
  return m <= 0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / m);
}

/// Mean gradient magnitude — the stand-in for "did detail come back", because
/// PSNR cannot tell an inventive upscaler from a broken one.
double edgeEnergy(const cv::Mat& img) {
  cv::Mat g, gx, gy, mag;
  cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
  cv::Sobel(g, gx, CV_32F, 1, 0, 3);
  cv::Sobel(g, gy, CV_32F, 0, 1, 3);
  cv::magnitude(gx, gy, mag);
  return cv::mean(mag)[0];
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if !RCDL_HAVE_OPENCV
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "sr_demo needs OpenCV for image decode/encode, and this build was configured "
               "without it. Install OpenCV (headers + libs) and re-run cmake, or drive "
               "rcdl::SuperResolver from your own decoder.\n");
  return 1;
#else
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s model.rknn image.jpg [--out up.png] [--overlap N] "
                 "[--max-side N] [--check]\n",
                 argv[0]);
    return 1;
  }
  std::string out_path;
  rcdl::SuperResConfig cfg;
  int max_side = 512;  // an upscale is linear in pixels; keep the demo bounded
  bool check = false;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--overlap" && i + 1 < argc) {
      cfg.overlap = std::atoi(argv[++i]);
    } else if (arg == "--max-side" && i + 1 < argc) {
      max_side = std::atoi(argv[++i]);
    } else if (arg == "--check") {
      check = true;
    }
  }

  cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (img.empty()) {
    std::fprintf(stderr, "cannot read %s\n", argv[2]);
    return 1;
  }
  if (std::max(img.cols, img.rows) > max_side) {
    const double s = static_cast<double>(max_side) / std::max(img.cols, img.rows);
    cv::resize(img, img, cv::Size(), s, s, cv::INTER_AREA);
  }

  rcdl::Engine engine(argv[1]);
  rcdl::SuperResolver sr(engine, cfg);
  std::printf("model: x%d, %dx%d tiles, input %s\n", sr.scale(), sr.tile(), sr.tileHeight(),
              rcdl::dtypeName(engine.inputType(0)));

  cv::Mat truth;
  if (check) {
    // Shrink by exactly the model's factor, so the original is the answer.
    truth = img.clone();
    cv::resize(img, img, cv::Size(img.cols / sr.scale(), img.rows / sr.scale()), 0, 0,
               cv::INTER_AREA);
    truth = truth(cv::Rect(0, 0, img.cols * sr.scale(), img.rows * sr.scale())).clone();
  }

  const auto t0 = std::chrono::steady_clock::now();
  rcdl::SrImage up = sr.upscale(img.data, img.cols, img.rows, static_cast<int>(img.step));
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("%dx%d -> %dx%d in %.0f ms (%d tiles, %.0f ms/tile)\n", img.cols, img.rows,
              up.width, up.height, ms, sr.lastTileCount(),
              ms / std::max(sr.lastTileCount(), 1));

  cv::Mat result(up.height, up.width, CV_8UC3, up.data.data());
  if (check) {
    cv::Mat bicubic;
    cv::resize(img, bicubic, cv::Size(up.width, up.height), 0, 0, cv::INTER_CUBIC);
    std::printf("vs the original: model %.2f dB, bicubic %.2f dB\n", psnr(result, truth),
                psnr(bicubic, truth));
    std::printf("edge energy   : model %.1f, bicubic %.1f, original %.1f\n",
                edgeEnergy(result), edgeEnergy(bicubic), edgeEnergy(truth));
  }

  if (!out_path.empty()) {
    if (!cv::imwrite(out_path, result)) {
      std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
      return 1;
    }
    std::printf("wrote %s\n", out_path.c_str());
  }
  return 0;
#endif
}
