// Promptable segmentation: point at something, get its mask.
//
//   ./sam_demo encoder.rknn decoder.rknn image.jpg [--box x1,y1,x2,y2]
//              [--point x,y] [--out mask.png] [--all]
//
// The encoder runs once (~300 ms) and every prompt after that is a decoder pass
// (~140 ms) against the same embedding, which is why this demo encodes once and
// then prompts several times. With no prompt given it uses the frame's centre.
//
// `--all` prints all four masks the decoder returns instead of the best one:
// SAM answers an ambiguous click with several nestings (the shirt, the person,
// the crowd), each with its own predicted quality.
//
// Image I/O (decode + encode) is the one thing RCDL leaves to OpenCV, so this
// example needs a build where OpenCV was found.

#include <chrono>
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
               "sam_demo needs OpenCV for image decode/encode, and this build was configured "
               "without it. Install OpenCV (headers + libs) and re-run cmake, or drive "
               "rcdl::PromptableSegmenter from your own decoder.\n");
  return 1;
#else
  if (argc < 4) {
    std::fprintf(stderr,
                 "usage: %s encoder.rknn decoder.rknn image.jpg [--box x1,y1,x2,y2] "
                 "[--point x,y] [--out mask.png] [--all]\n",
                 argv[0]);
    return 1;
  }
  std::string out_path;
  bool all = false;
  float box[4] = {0, 0, 0, 0};
  float pt[2] = {0, 0};
  bool have_box = false, have_point = false;
  for (int i = 4; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--all") {
      all = true;
    } else if (arg == "--box" && i + 1 < argc) {
      have_box = std::sscanf(argv[++i], "%f,%f,%f,%f", &box[0], &box[1], &box[2], &box[3]) == 4;
    } else if (arg == "--point" && i + 1 < argc) {
      have_point = std::sscanf(argv[++i], "%f,%f", &pt[0], &pt[1]) == 2;
    }
  }

  cv::Mat img = cv::imread(argv[3], cv::IMREAD_COLOR);
  if (img.empty()) {
    std::fprintf(stderr, "cannot read %s\n", argv[3]);
    return 1;
  }
  if (!have_box && !have_point) {
    pt[0] = img.cols / 2.0f;
    pt[1] = img.rows / 2.0f;
    have_point = true;
    std::printf("no prompt given; clicking the centre of the frame\n");
  }

  rcdl::Engine encoder(argv[1]);
  rcdl::Engine decoder(argv[2]);
  rcdl::PromptableSegmenter sam(encoder, decoder);

  rcdl::ImageView view{};
  view.data = img.data;
  view.width = img.cols;
  view.height = img.rows;
  view.wstride = static_cast<int>(img.step / 3);
  view.format = rcdl::PixelFormat::BGR888;

  using clk = std::chrono::steady_clock;
  auto ms = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  const auto t0 = clk::now();
  sam.setImage(view);
  const auto t1 = clk::now();
  std::printf("encoded %dx%d into a %dx%d canvas in %.0f ms (%s)\n", img.cols, img.rows,
              sam.inputWidth(), sam.inputHeight(), ms(t0, t1),
              rcdl::backendName(sam.lastBackend()));

  const auto t2 = clk::now();
  rcdl::PromptMask mask = have_box ? sam.box(box[0], box[1], box[2], box[3])
                                   : sam.point(pt[0], pt[1]);
  const auto t3 = clk::now();
  std::printf("prompt %s -> score %.3f, %.2f%% of the frame, bbox [%d %d %d %d] in %.0f ms\n",
              have_box ? "box" : "point", mask.score, 100.0 * mask.area(), mask.x0, mask.y0,
              mask.x1, mask.y1, ms(t2, t3));

  if (all) {
    // The other nestings cost nothing extra: they came out of the same pass.
    const std::vector<rcdl::PromptMask> every = sam.masks();
    for (std::size_t i = 0; i < every.size(); ++i) {
      std::printf("  mask %zu: score %.3f, %.2f%% of the frame\n", i, every[i].score,
                  100.0 * every[i].area());
    }
  }

  if (!out_path.empty()) {
    cv::Mat overlay = img.clone();
    for (int y = 0; y < mask.height; ++y) {
      for (int x = 0; x < mask.width; ++x) {
        if (!mask.data[static_cast<std::size_t>(y) * mask.width + x]) continue;
        cv::Vec3b& p = overlay.at<cv::Vec3b>(y, x);
        p[1] = static_cast<uchar>(std::min(255, p[1] + 90));  // tint green
      }
    }
    if (have_box) {
      cv::rectangle(overlay, cv::Point2f(box[0], box[1]), cv::Point2f(box[2], box[3]),
                    cv::Scalar(0, 0, 255), 2);
    } else {
      cv::circle(overlay, cv::Point2f(pt[0], pt[1]), 6, cv::Scalar(0, 0, 255), -1);
    }
    if (!cv::imwrite(out_path, overlay)) {
      std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
      return 1;
    }
    std::printf("wrote %s\n", out_path.c_str());
  }
  return 0;
#endif
}
