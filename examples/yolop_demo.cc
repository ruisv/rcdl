// Panoptic driving on a still image: vehicles, drivable area and lane lines.
//
//   ./yolop_demo yolop_cut_640_i8_rk3588.rknn frame.png [--out out.png]
//                [--conf 0.35]
//
// This is the multi-head shape the other examples do not show. One inference
// feeds THREE decoders off the same Engine:
//
//   outputs 0,1,2  three raw ANCHOR-BASED head convolutions -> AnchorDetector
//   output  3      drivable area, a 2-class logit volume    -> Segmenter
//   output  4      lane lines, the same                     -> Segmenter
//
// Two things are worth knowing before reading the code.
//
// FIRST, the model has to be the one cut before the decode. The published ONNX
// export ends the detection branch with the anchor arithmetic itself, assembled
// out of ScatterND writes. That compiles without a single error into a model
// whose objectness and class columns are never written, so a detector reading it
// finds nothing at any threshold — the graph is cut at the three head
// convolutions instead and the arithmetic runs here, on three small tensors.
//
// SECOND, the priors are part of the model. A box's size IS its prior times a
// bounded multiplier, so AnchorDetectConfig's default prior set belongs to this
// network; another anchor-based model needs its own, and the wrong set gives
// plausible-looking boxes of the wrong size rather than an error.
//
// Image I/O (decode + drawing) is the one thing RCDL leaves to OpenCV, so this
// example needs a build where OpenCV was found.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

#if RCDL_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#if RCDL_HAVE_OPENCV
namespace {

/// Tint every pixel a 2-class mask marks, leaving the rest alone.
std::size_t paintMask(cv::Mat& img, const rcdl::SegMask& mask, const cv::Scalar& color,
                      double alpha) {
  std::size_t on = 0;
  const int h = std::min(img.rows, mask.height);
  const int w = std::min(img.cols, mask.width);
  for (int y = 0; y < h; ++y) {
    cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
    for (int x = 0; x < w; ++x) {
      if (mask.labels[static_cast<std::size_t>(y) * mask.width + x] == 0) continue;
      ++on;
      cv::Vec3b& p = row[x];
      for (int c = 0; c < 3; ++c) {
        p[c] = static_cast<std::uint8_t>(p[c] * (1.0 - alpha) + color[c] * alpha);
      }
    }
  }
  return on;
}

/// Where a mask's marked pixels sit vertically, as a fraction of the height.
/// Both of these masks are road surface seen from a car, so a sane result is
/// well below the middle — a cheap sanity check that beats "how many pixels".
double centroidY(const rcdl::SegMask& mask) {
  double sum = 0.0;
  std::size_t n = 0;
  for (int y = 0; y < mask.height; ++y) {
    for (int x = 0; x < mask.width; ++x) {
      if (mask.labels[static_cast<std::size_t>(y) * mask.width + x] == 0) continue;
      sum += y;
      ++n;
    }
  }
  return n == 0 ? -1.0 : sum / static_cast<double>(n) / std::max(1, mask.height);
}

}  // namespace
#endif

int main(int argc, char** argv) {
#if !RCDL_HAVE_OPENCV
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "yolop_demo needs OpenCV for image decode/encode, and this build was configured "
               "without it. Install OpenCV (headers + libs) and re-run cmake, or drive "
               "rcdl::AnchorDetector / rcdl::Segmenter from your own decoder via "
               "rcdl::ImageView.\n");
  return 1;
#else
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.rknn image.png [--out out.png] [--conf 0.35]\n", argv[0]);
    return 1;
  }
  std::string out_path;
  float conf = 0.35f;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
    } else if (arg == "--conf" && i + 1 < argc) {
      conf = static_cast<float>(std::atof(argv[++i]));
    } else {
      std::fprintf(stderr, "error: unknown argument %s\n", argv[i]);
      return 1;
    }
  }

  cv::Mat bgr = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (bgr.empty()) {
    std::fprintf(stderr, "error: cannot read %s\n", argv[2]);
    return 1;
  }

  try {
    rcdl::Engine engine(argv[1]);
    if (engine.numOutputs() != 5) {
      std::fprintf(stderr,
                   "error: expected 5 outputs (3 raw detection heads + drivable + lane), got %d. "
                   "A model with one big decoded detection tensor is the export that bakes the "
                   "anchor decode into the graph — that one comes back with its objectness and "
                   "class columns empty.\n",
                   engine.numOutputs());
      return 1;
    }

    rcdl::AnchorDetectConfig cfg;  // priors default to this network's
    cfg.num_classes = 1;           // vehicles
    cfg.conf_thresh = conf;
    rcdl::AnchorDetector detector(engine, cfg, /*output_base=*/0);

    rcdl::SegConfig scfg;
    scfg.num_classes = 2;
    rcdl::Segmenter drivable(engine, scfg, /*output_index=*/3);
    rcdl::Segmenter lanes(engine, scfg, /*output_index=*/4);

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

    // One inference, three decoders — none of them touches the NPU again.
    const std::vector<rcdl::Detection> dets = detector.postprocess(lb);
    const rcdl::SegMask road = drivable.postprocess(lb);
    const rcdl::SegMask lane = lanes.postprocess(lb);
    const auto t3 = std::chrono::steady_clock::now();

    const auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("preproc: %s %.1f ms   infer %.1f ms (npu %.1f ms)   3 decoders %.1f ms\n",
                rcdl::backendName(used), ms(t0, t1), ms(t1, t2),
                engine.lastRunMicros() / 1000.0, ms(t2, t3));

    std::printf("vehicles: %zu\n", dets.size());
    for (const rcdl::Detection& d : dets) {
      std::printf("  %.3f  %.1f %.1f %.1f %.1f\n", d.score, d.x1, d.y1, d.x2, d.y2);
    }

    cv::Mat overlay = bgr.clone();
    const std::size_t road_px = paintMask(overlay, road, cv::Scalar(0, 200, 0), 0.45);
    const std::size_t lane_px = paintMask(overlay, lane, cv::Scalar(0, 0, 255), 0.70);
    const double frame = static_cast<double>(bgr.rows) * bgr.cols;
    std::printf("drivable: %.2f%% of the frame, centroid at %.2f of the height\n",
                100.0 * static_cast<double>(road_px) / frame, centroidY(road));
    std::printf("lane:     %.2f%% of the frame, centroid at %.2f of the height\n",
                100.0 * static_cast<double>(lane_px) / frame, centroidY(lane));

    if (!out_path.empty()) {
      for (const rcdl::Detection& d : dets) {
        cv::rectangle(overlay, cv::Point(static_cast<int>(d.x1), static_cast<int>(d.y1)),
                      cv::Point(static_cast<int>(d.x2), static_cast<int>(d.y2)),
                      cv::Scalar(255, 255, 0), 2);
      }
      if (!cv::imwrite(out_path, overlay)) {
        std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
        return 1;
      }
      std::printf("wrote %s\n", out_path.c_str());
    }
  } catch (const rcdl::Error& e) {
    std::fprintf(stderr, "rcdl error %d: %s\n", e.code(), e.what());
    return 1;
  } catch (const std::exception& e) {
    // cv::imwrite throws cv::Exception for an output extension it cannot
    // encode, and that is not an rcdl::Error.
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
#endif
}
