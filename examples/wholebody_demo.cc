// Whole-body pose: 133 keypoints per person, top-down.
//
//   ./wholebody_demo pose.rknn image.jpg [--det yolov8n.rknn] [--box x1,y1,x2,y2]
//                    [--out pose.png]
//
// This head is handed ONE person's box and answers with the COCO-WholeBody 133:
// 17 body joints, 6 feet, 68 face landmarks and 21 points per hand. It runs once
// per person (~25 ms), so a detector goes first — pass one with `--det`, or a
// single box with `--box`, or neither and the whole frame is used as the box.
//
// The five regions are drawn in different colours, which is also how you can see
// at a glance that the layout is being sliced correctly: the face cluster
// belongs on the head and each hand cluster at its own wrist.
//
// Image I/O (decode + encode) is the one thing RCDL leaves to OpenCV, so this
// example needs a build where OpenCV was found.

#include <array>
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
               "wholebody_demo needs OpenCV for image decode/encode, and this build was "
               "configured without it. Install OpenCV (headers + libs) and re-run cmake, or "
               "drive rcdl::WholeBodyEstimator from your own decoder.\n");
  return 1;
#else
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s pose.rknn image.jpg [--det yolov8n.rknn] "
                 "[--box x1,y1,x2,y2] [--out pose.png]\n",
                 argv[0]);
    return 1;
  }
  std::string det_path, out_path;
  float box[4] = {0, 0, 0, 0};
  bool have_box = false;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--det" && i + 1 < argc) {
      det_path = argv[++i];
    } else if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--box" && i + 1 < argc) {
      have_box = std::sscanf(argv[++i], "%f,%f,%f,%f", &box[0], &box[1], &box[2], &box[3]) == 4;
    }
  }

  cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (img.empty()) {
    std::fprintf(stderr, "cannot read %s\n", argv[2]);
    return 1;
  }
  rcdl::ImageView view{};
  view.data = img.data;
  view.width = img.cols;
  view.height = img.rows;
  view.wstride = static_cast<int>(img.step / 3);
  view.format = rcdl::PixelFormat::BGR888;

  std::vector<std::array<float, 4>> boxes;
  if (have_box) {
    boxes.push_back({box[0], box[1], box[2], box[3]});
  } else if (!det_path.empty()) {
    rcdl::Engine det_engine(det_path);
    rcdl::DetectionPipeline det(det_engine, rcdl::PipelineConfig());
    for (const rcdl::Detection& d : det.process(view)) {
      if (rcdl::cocoClassName(d.class_id) == std::string("person")) {
        boxes.push_back({d.x1, d.y1, d.x2, d.y2});
      }
    }
    std::printf("detector found %zu people\n", boxes.size());
  } else {
    boxes.push_back({0.0f, 0.0f, static_cast<float>(img.cols), static_cast<float>(img.rows)});
    std::printf("no detector and no box: using the whole frame\n");
  }

  rcdl::Engine engine(argv[1]);
  rcdl::WholeBodyEstimator pose(engine);
  std::printf("model %dx%d, %d keypoints\n", pose.inputWidth(), pose.inputHeight(),
              pose.numKeypoints());

  const cv::Scalar colors[5] = {{0, 255, 0}, {255, 200, 0}, {0, 200, 255},
                                {255, 0, 200}, {200, 0, 255}};
  for (const std::array<float, 4>& b : boxes) {
    const std::vector<rcdl::Keypoint> kp = pose.estimate(view, b[0], b[1], b[2], b[3]);
    int counts[5] = {0, 0, 0, 0, 0};
    for (std::size_t i = 0; i < kp.size(); ++i) {
      if (kp[i].score < pose.config().kpt_thresh) continue;
      const int part = static_cast<int>(rcdl::bodyPart(static_cast<int>(i)));
      ++counts[part];
      cv::circle(img, cv::Point2f(kp[i].x, kp[i].y), part == 2 ? 1 : 2, colors[part], -1,
                 cv::LINE_AA);
    }
    std::printf("box [%.0f %.0f %.0f %.0f]: body %d/17, feet %d/6, face %d/68, hands %d+%d/21\n",
                b[0], b[1], b[2], b[3], counts[0], counts[1], counts[2], counts[3], counts[4]);
    cv::rectangle(img, cv::Point2f(b[0], b[1]), cv::Point2f(b[2], b[3]), cv::Scalar(60, 60, 60),
                  1);
  }

  if (!out_path.empty()) {
    if (!cv::imwrite(out_path, img)) {
      std::fprintf(stderr, "failed to write %s\n", out_path.c_str());
      return 1;
    }
    std::printf("wrote %s\n", out_path.c_str());
  }
  return 0;
#endif
}
