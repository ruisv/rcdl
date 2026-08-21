// Dense optical flow between two frames.
//
//   ./flow_demo model.rknn a.jpg [b.jpg] [--out flow.png] [--shift N]
//
// With one image the demo MAKES the second one by shifting a window across the
// same photograph, so the correct field is a known constant everywhere and the
// demo can score itself. That is the useful property of this task: ground truth
// costs nothing to manufacture.
//
// The visualisation is the Middlebury colour wheel — hue is direction,
// saturation is speed — normalised by the field's own 99th percentile.
//
// This model needs a runtime kernel RCDL registers for it (GridSample; see
// backend/custom_ops.h) and each of its nine calls crosses the CPU boundary, so
// expect well over a second a frame. It is correct, not fast.
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
#endif

int main(int argc, char** argv) {
#if !RCDL_HAVE_OPENCV
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "flow_demo needs OpenCV for image decode/encode, and this build was configured "
               "without it. Install OpenCV (headers + libs) and re-run cmake, or drive "
               "rcdl::OpticalFlowEstimator from your own decoder.\n");
  return 1;
#else
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.rknn a.jpg [b.jpg] [--out flow.png] [--shift N]\n",
                 argv[0]);
    return 1;
  }
  std::string b_path, out_path;
  int shift = 8;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--shift" && i + 1 < argc) {
      shift = std::atoi(argv[++i]);
    } else if (arg[0] != '-' && b_path.empty()) {
      b_path = arg;
    }
  }

  cv::Mat src = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (src.empty()) {
    std::fprintf(stderr, "cannot read %s\n", argv[2]);
    return 1;
  }

  rcdl::Engine engine(argv[1]);
  rcdl::OpticalFlowEstimator flow(engine);
  const int W = flow.inputWidth(), H = flow.inputHeight();

  cv::Mat a, b;
  bool synthetic = b_path.empty();
  double gt_u = 0.0, gt_v = 0.0;
  if (!synthetic) {
    cv::Mat other = cv::imread(b_path, cv::IMREAD_COLOR);
    if (other.empty()) {
      std::fprintf(stderr, "cannot read %s\n", b_path.c_str());
      return 1;
    }
    cv::resize(src, a, cv::Size(W, H));
    cv::resize(other, b, cv::Size(W, H));
  } else {
    // Two windows of one photograph, offset by `shift`. A window that moves
    // right sees the world move LEFT, so the true flow is negative.
    if (src.cols < W + shift + 1 || src.rows < H + 1) {
      const double s = std::max(static_cast<double>(W + shift + 2) / src.cols,
                                static_cast<double>(H + 2) / src.rows);
      cv::resize(src, src, cv::Size(), s, s);
    }
    const int x0 = (src.cols - W - shift) / 2, y0 = (src.rows - H) / 2;
    a = src(cv::Rect(x0, y0, W, H)).clone();
    b = src(cv::Rect(x0 + shift, y0, W, H)).clone();
    gt_u = -shift;
    std::printf("second frame: the same photograph, window moved %d px right\n", shift);
  }

  const auto t0 = std::chrono::steady_clock::now();
  rcdl::FlowField f = flow.estimate(a.data, b.data, a.cols, a.rows, static_cast<int>(a.step));
  const auto t1 = std::chrono::steady_clock::now();
  std::printf("%dx%d flow in %.0f ms\n", f.width, f.height,
              std::chrono::duration<double, std::milli>(t1 - t0).count());

  double mu = 0, mv = 0, mag = 0;
  const int margin = 48;
  int n = 0;
  double epe = 0;
  for (int y = margin; y < f.height - margin; ++y) {
    for (int x = margin; x < f.width - margin; ++x) {
      const float u = f.u(x, y), v = f.v(x, y);
      mu += u;
      mv += v;
      mag += std::hypot(u, v);
      epe += std::hypot(u - gt_u, v - gt_v);
      ++n;
    }
  }
  std::printf("mean vector (%.3f, %.3f), mean speed %.3f px\n", mu / n, mv / n, mag / n);
  if (synthetic) {
    // The border band is excluded: content entered or left the frame there and
    // has no correct answer at all.
    std::printf("vs the known shift (%.1f, %.1f): endpoint error %.4f px\n", gt_u, gt_v,
                epe / n);
  }

  if (!out_path.empty()) {
    std::vector<std::uint8_t> viz = rcdl::flowColorize(f);
    cv::Mat colored(f.height, f.width, CV_8UC3, viz.data());
    cv::Mat canvas(f.height, f.width * 2, CV_8UC3);
    a.copyTo(canvas(cv::Rect(0, 0, a.cols, a.rows)));
    colored.copyTo(canvas(cv::Rect(a.cols, 0, f.width, f.height)));
    if (!cv::imwrite(out_path, canvas)) {
      std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
      return 1;
    }
    std::printf("wrote %s (frame | flow)\n", out_path.c_str());
  }
  return 0;
#endif
}
