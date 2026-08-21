// Sparse local features and matching between two frames.
//
//   ./xfeat_demo model.rknn a.jpg [b.jpg] [--out matches.png] [--top-k N]
//                [--thresh T] [--min-cossim C]
//
// With one image the demo MAKES the second one — a known rotation and scale of
// the first — so it can score itself: the warp says exactly where every match
// should land, and the demo reports how many of them agree. That is the only
// self-checking demo in this repo, and it is possible here because geometry
// supplies its own ground truth.
//
// The input this model wants is NOT image bytes but a normalized grey map, so
// the Engine is constructed with `float_inputs = {0}` and the preprocessing
// (grey by channel mean, resize, InstanceNorm) happens on the CPU in
// rcdl::xfeatPreprocess. Everything after the three maps — softmax, NMS, top-k,
// descriptor sampling — is CPU too, for the reason in tasks/features.h.
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
               "xfeat_demo needs OpenCV for image decode/encode, and this build was configured "
               "without it. Install OpenCV (headers + libs) and re-run cmake, or drive "
               "rcdl::FeatureExtractor from your own decoder.\n");
  return 1;
#else
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s model.rknn a.jpg [b.jpg] [--out matches.png] [--top-k N] "
                 "[--thresh T] [--min-cossim C]\n",
                 argv[0]);
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string a_path = argv[2];
  std::string b_path, out_path;
  rcdl::XfeatConfig cfg;
  float min_cossim = 0.82f;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--top-k" && i + 1 < argc) {
      cfg.top_k = std::atoi(argv[++i]);
    } else if (arg == "--thresh" && i + 1 < argc) {
      cfg.detection_thresh = static_cast<float>(std::atof(argv[++i]));
    } else if (arg == "--min-cossim" && i + 1 < argc) {
      min_cossim = static_cast<float>(std::atof(argv[++i]));
    } else if (arg[0] != '-' && b_path.empty()) {
      b_path = arg;
    }
  }

  cv::Mat a = cv::imread(a_path, cv::IMREAD_COLOR);
  if (a.empty()) {
    std::fprintf(stderr, "cannot read %s\n", a_path.c_str());
    return 1;
  }

  // The XFeat input is a computed map, not pixels; see EngineOptions.
  rcdl::EngineOptions opts;
  opts.float_inputs = {0};
  rcdl::Engine engine(model_path, opts);
  rcdl::FeatureExtractor extractor(engine, cfg);

  // Score at the model's own size and aspect. A portrait photo squeezed into a
  // landscape input comes back with its y errors scaled by the same factor, so
  // the numbers below would measure the resize rather than the model.
  cv::resize(a, a, cv::Size(extractor.inputWidth(), extractor.inputHeight()));

  cv::Mat b;
  cv::Mat warp;  // 2x3, a -> b, only when we made b ourselves
  if (!b_path.empty()) {
    b = cv::imread(b_path, cv::IMREAD_COLOR);
    if (b.empty()) {
      std::fprintf(stderr, "cannot read %s\n", b_path.c_str());
      return 1;
    }
    cv::resize(b, b, a.size());
  } else {
    // The 2x3 is written out rather than asked for: cv::getRotationMatrix2D
    // moved from imgproc to the geometry module in OpenCV 5, and six numbers
    // are cheaper than a version check — and say what the convention is.
    const double theta = 12.0 * CV_PI / 180.0, scale = 0.85;
    const double alpha = scale * std::cos(theta), beta = scale * std::sin(theta);
    const double cx = a.cols / 2.0, cy = a.rows / 2.0;
    warp = (cv::Mat_<double>(2, 3) << alpha, beta,
            (1 - alpha) * cx - beta * cy + 25.0, -beta, alpha,
            beta * cx + (1 - alpha) * cy - 15.0);
    cv::warpAffine(a, b, warp, a.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    std::printf("second frame: synthesized (rotate 12 deg, scale 0.85, shift +25/-15)\n");
  }

  using clk = std::chrono::steady_clock;
  const auto t0 = clk::now();
  rcdl::FeatureSet fa = extractor.extract(a.data, a.cols, a.rows, static_cast<int>(a.step));
  const auto t1 = clk::now();
  rcdl::FeatureSet fb = extractor.extract(b.data, b.cols, b.rows, static_cast<int>(b.step));
  const auto t2 = clk::now();
  std::vector<rcdl::FeatureMatch> matches = rcdl::matchFeatures(fa, fb, min_cossim);
  const auto t3 = clk::now();

  auto ms = [](clk::time_point x, clk::time_point y) {
    return std::chrono::duration<double, std::milli>(y - x).count();
  };
  std::printf("features: %zu + %zu   matches: %zu\n", fa.size(), fb.size(), matches.size());
  std::printf("timing: extract %.1f / %.1f ms, match %.1f ms\n", ms(t0, t1), ms(t1, t2),
              ms(t2, t3));

  if (!warp.empty()) {
    // Ground truth: push every matched point of A through the warp and see
    // where it lands. Points whose true correspondent falls outside the frame
    // are dropped — they legitimately had nothing to match.
    int inside = 0, good = 0;
    std::vector<double> errs;
    for (const rcdl::FeatureMatch& m : matches) {
      const rcdl::Feature& pa = fa.keypoints[m.a];
      const rcdl::Feature& pb = fb.keypoints[m.b];
      const double px = warp.at<double>(0, 0) * pa.x + warp.at<double>(0, 1) * pa.y +
                        warp.at<double>(0, 2);
      const double py = warp.at<double>(1, 0) * pa.x + warp.at<double>(1, 1) * pa.y +
                        warp.at<double>(1, 2);
      if (px < 16 || py < 16 || px >= a.cols - 16 || py >= a.rows - 16) continue;
      ++inside;
      const double e = std::hypot(px - pb.x, py - pb.y);
      errs.push_back(e);
      if (e < 3.0) ++good;
    }
    std::sort(errs.begin(), errs.end());
    const double median = errs.empty() ? 0.0 : errs[errs.size() / 2];
    std::printf("known warp: %d/%d matches within 3 px (%.1f%%), median error %.2f px\n", good,
                inside, inside ? 100.0 * good / inside : 0.0, median);
  }

  if (!out_path.empty()) {
    cv::Mat canvas(a.rows, a.cols * 2, CV_8UC3);
    a.copyTo(canvas(cv::Rect(0, 0, a.cols, a.rows)));
    b.copyTo(canvas(cv::Rect(a.cols, 0, b.cols, b.rows)));
    // Drawing every one of a few thousand lines paints the image solid; a
    // regular sample keeps it readable and is not a claim about quality.
    const int step = static_cast<int>(matches.size() / 120) + 1;
    for (std::size_t i = 0; i < matches.size(); i += step) {
      const rcdl::Feature& pa = fa.keypoints[matches[i].a];
      const rcdl::Feature& pb = fb.keypoints[matches[i].b];
      const cv::Scalar color(60 + (i * 37) % 195, 60 + (i * 91) % 195, 60 + (i * 53) % 195);
      cv::line(canvas, cv::Point2f(pa.x, pa.y), cv::Point2f(pb.x + a.cols, pb.y), color, 1,
               cv::LINE_AA);
      cv::circle(canvas, cv::Point2f(pa.x, pa.y), 2, color, -1, cv::LINE_AA);
      cv::circle(canvas, cv::Point2f(pb.x + a.cols, pb.y), 2, color, -1, cv::LINE_AA);
    }
    if (cv::imwrite(out_path, canvas)) {
      std::printf("wrote %s (every %d-th match drawn)\n", out_path.c_str(), step);
    } else {
      std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
      return 1;
    }
  }
  return 0;
#endif
}
