// Segmentation on a still image — instance OR semantic, decided by the model.
//
//   ./seg_demo model.rknn image.jpg [--out out.png] [--labels names.txt]
//
// The two segmentation families need different post-processing but the same
// front half, so this example picks between them from the Engine's output
// signature and says which it picked:
//
//   * a YOLO-style INSTANCE head has one output group per scale (cls + box +
//     mask coefficients, sometimes a score-sum branch) plus a prototype tensor —
//     many outputs. resolveInstanceSegHead() reads the whole layout off the
//     model, so grids, class and coefficient counts, DFL reg_max and both
//     channel orders come from the tensors rather than from a config.
//   * a SEMANTIC head is a single [1,C,H,W] (or NHWC) logit volume that is
//     argmaxed per pixel into a label map.
//
// The front half is identical either way and is the point of the stack: RGA
// letterboxes the decoded image straight into the NPU's input tensor, then one
// infer(). Masks and label maps come back in ORIGINAL-image pixels.
//
// Class NAMES are not guessed: pass `--labels` (one name per line, in class-id
// order) or the overlay is annotated with bare class ids. A VOC-trained model
// and a Cityscapes-trained one both produce small integers, and printing the
// wrong table's names is worse than printing none.
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
  const int h = (id * 67) % 180;  // spread the classes over the hue circle
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, 220, 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const cv::Vec3b p = bgr.at<cv::Vec3b>(0, 0);
  return cv::Scalar(p[0], p[1], p[2]);
}

/// Class name for the overlay. A `--labels` table wins; failing that, an
/// 80-class head is the COCO taxonomy every YOLO instance-seg export uses (the
/// same assumption det_demo makes), and anything else prints bare ids — a
/// semantic model may be VOC-21, Cityscapes-19 or something bespoke, and
/// printing the wrong table's names is worse than printing none.
std::string nameOf(const std::vector<std::string>& labels, int class_id, int num_classes) {
  if (!labels.empty()) return rcdl::classLabel(labels, class_id);
  if (num_classes == 80) return rcdl::cocoClassName(class_id);
  return "class " + std::to_string(class_id);
}

/// Preprocess into the NPU's own input tensor and run one inference.
/// Returns the geometry RGA actually used, which is what the decoders invert.
rcdl::LetterboxInfo runOnce(rcdl::Engine& engine, const rcdl::ImageView& input,
                            const rcdl::ImageView& src, rcdl::PreprocBackend* used) {
  const rcdl::LetterboxInfo lb =
      rcdl::letterbox(input, src, 114, rcdl::PreprocBackend::Auto, rcdl::YuvRange::kStudioToFull,
                      used);
  engine.infer();
  return lb;
}

/// Paint one instance's binary mask over the image, 50% alpha, plus its box.
void drawInstance(cv::Mat& img, const rcdl::InstanceMask& im,
                  const std::vector<std::string>& labels, int num_classes) {
  const cv::Scalar color = classColor(im.class_id);
  const int x0 = std::max(0, im.mask_x0);
  const int y0 = std::max(0, im.mask_y0);
  const int x1 = std::min(img.cols, im.mask_x0 + im.mask_w);
  const int y1 = std::min(img.rows, im.mask_y0 + im.mask_h);
  for (int y = y0; y < y1; ++y) {
    cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
    for (int x = x0; x < x1; ++x) {
      if (!im.at(x, y)) continue;
      cv::Vec3b& p = row[x];
      for (int c = 0; c < 3; ++c) {
        p[c] = static_cast<std::uint8_t>((p[c] + color[c]) * 0.5);
      }
    }
  }
  cv::rectangle(img, cv::Point(static_cast<int>(im.x1), static_cast<int>(im.y1)),
                cv::Point(static_cast<int>(im.x2), static_cast<int>(im.y2)), color, 2);

  char text[160];
  std::snprintf(text, sizeof(text), "%s %.2f",
                nameOf(labels, im.class_id, num_classes).c_str(), im.score);
  int baseline = 0;
  const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
  const int ly = std::max(static_cast<int>(im.y1), ts.height + baseline + 2);
  cv::rectangle(img, cv::Point(static_cast<int>(im.x1), ly - ts.height - baseline - 2),
                cv::Point(static_cast<int>(im.x1) + ts.width + 4, ly), color, cv::FILLED);
  cv::putText(img, text, cv::Point(static_cast<int>(im.x1) + 2, ly - baseline - 1),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

int runInstanceSeg(rcdl::Engine& engine, cv::Mat& bgr, const rcdl::ImageView& src,
                   const std::string& out_path, const std::vector<std::string>& labels) {
  rcdl::InstanceSegmenter seg(engine, rcdl::InstanceSegConfig());
  std::printf("head: instance segmentation (%d outputs)\n%s\n", engine.numOutputs(),
              seg.layout().describe().c_str());

  const rcdl::ImageView input = rcdl::engineInputView(engine, 0, rcdl::PixelFormat::RGB888);
  rcdl::PreprocBackend used = rcdl::PreprocBackend::Auto;
  const rcdl::LetterboxInfo lb = runOnce(engine, input, src, &used);
  const std::vector<rcdl::InstanceMask> masks = seg.postprocess(lb);
  const int nc = seg.layout().num_classes;

  std::printf("preproc: %s   letterbox scale=%.4f pad=(%.1f,%.1f)\n", rcdl::backendName(used),
              lb.scale, lb.padX, lb.padY);
  std::printf("instances: %zu\n", masks.size());
  for (const rcdl::InstanceMask& im : masks) {
    std::size_t on = 0;
    for (std::uint8_t v : im.mask) on += v;
    const double box_area =
        std::max(1.0, static_cast<double>(im.x2 - im.x1) * static_cast<double>(im.y2 - im.y1));
    std::printf("  %-16s %.3f  %.1f %.1f %.1f %.1f   mask %zu px (%.0f%% of box)\n",
                nameOf(labels, im.class_id, nc).c_str(), im.score, im.x1, im.y1, im.x2, im.y2, on,
                100.0 * static_cast<double>(on) / box_area);
  }

  if (!out_path.empty()) {
    for (const rcdl::InstanceMask& im : masks) drawInstance(bgr, im, labels, nc);
    if (!cv::imwrite(out_path, bgr)) {
      std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
      return 1;
    }
    std::printf("wrote %s\n", out_path.c_str());
  }
  return 0;
}

int runSemanticSeg(rcdl::Engine& engine, cv::Mat& bgr, const rcdl::ImageView& src,
                   const std::string& out_path, const std::vector<std::string>& labels) {
  rcdl::Segmenter seg(engine, rcdl::SegConfig());
  const std::vector<int> shape = engine.outputShape(0);
  std::string dims;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    dims += (i ? "x" : "") + std::to_string(shape[i]);
  }
  std::printf("head: semantic segmentation (1 output %s %s)\n", dims.c_str(),
              rcdl::dtypeName(engine.outputType(0)));

  const rcdl::ImageView input = rcdl::engineInputView(engine, 0, rcdl::PixelFormat::RGB888);
  rcdl::PreprocBackend used = rcdl::PreprocBackend::Auto;
  const rcdl::LetterboxInfo lb = runOnce(engine, input, src, &used);
  // postprocess(lb) argmaxes at the model's own resolution and then projects the
  // label map onto the frame, so it lines up with the image pixel for pixel.
  const rcdl::SegMask mask = seg.postprocess(lb);

  std::printf("preproc: %s   letterbox scale=%.4f pad=(%.1f,%.1f)\n", rcdl::backendName(used),
              lb.scale, lb.padX, lb.padY);
  std::printf("label map: %dx%d over %d classes\n", mask.width, mask.height, mask.num_classes);

  std::vector<std::size_t> hist(static_cast<std::size_t>(std::max(1, mask.num_classes)), 0);
  for (std::int32_t v : mask.labels) {
    if (v >= 0 && static_cast<std::size_t>(v) < hist.size()) ++hist[static_cast<std::size_t>(v)];
  }
  const double total = std::max<double>(1.0, static_cast<double>(mask.labels.size()));
  std::vector<int> order;
  for (std::size_t i = 0; i < hist.size(); ++i) {
    if (hist[i]) order.push_back(static_cast<int>(i));
  }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return hist[static_cast<std::size_t>(a)] > hist[static_cast<std::size_t>(b)];
  });
  std::printf("classes present: %zu\n", order.size());
  for (int id : order) {
    // 0: never guess a taxonomy for a semantic head — VOC-21, Cityscapes-19
    // and bespoke label sets all look the same from here.
    std::printf("  %-16s %5.1f%%\n", nameOf(labels, id, 0).c_str(),
                100.0 * static_cast<double>(hist[static_cast<std::size_t>(id)]) / total);
  }

  if (!out_path.empty()) {
    // segColorize() already emits BGR in OpenCV's channel order, so the buffer
    // can be wrapped as a cv::Mat without a conversion.
    const std::vector<std::uint8_t> colored = rcdl::segColorize(mask);
    if (colored.size() != static_cast<std::size_t>(mask.width) * mask.height * 3) {
      std::fprintf(stderr, "error: colorized map does not cover the label map\n");
      return 1;
    }
    cv::Mat overlay(mask.height, mask.width, CV_8UC3, const_cast<std::uint8_t*>(colored.data()));
    cv::Mat blended;
    cv::addWeighted(bgr, 0.5, overlay, 0.5, 0.0, blended);
    if (!cv::imwrite(out_path, blended)) {
      std::fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
      return 1;
    }
    std::printf("wrote %s\n", out_path.c_str());
  }
  return 0;
}

}  // namespace
#endif  // RCDL_HAVE_OPENCV

int main(int argc, char** argv) {
#if !RCDL_HAVE_OPENCV
  (void)argc;
  (void)argv;
  std::fprintf(stderr,
               "seg_demo needs OpenCV for image decode/encode, and this build was configured "
               "without it. Install OpenCV (headers + libs) and re-run cmake, or drive "
               "rcdl::InstanceSegmenter / rcdl::Segmenter from your own decoder via "
               "rcdl::ImageView.\n");
  return 1;
#else
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s model.rknn image.jpg [--out out.png] [--labels names.txt]\n",
                 argv[0]);
    return 1;
  }
  std::string out_path, label_file;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out_path = argv[++i];
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
    std::printf("rcdl %s | librknnrt %s | driver %s\n", RCDL_VERSION_STRING,
                engine.sdkVersion().c_str(), engine.driverVersion().c_str());
    std::printf("model: %s\n", engine.path().c_str());
    std::printf("image: %s  %dx%d BGR\n", argv[2], bgr.cols, bgr.rows);

    // cv::imread rows may be padded, so describe the Mat by its own stride
    // (in PIXELS) instead of assuming packed rows.
    const rcdl::ImageView src =
        rcdl::hostView(bgr.data, bgr.cols, bgr.rows, rcdl::PixelFormat::BGR888,
                       static_cast<int>(bgr.step / 3), bgr.rows);

    // A single output cannot be an instance head (that needs at least one
    // detection branch plus a prototype), and more than one cannot be a plain
    // logit volume — that is the whole decision.
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = engine.numOutputs() == 1
                       ? runSemanticSeg(engine, bgr, src, out_path, labels)
                       : runInstanceSeg(engine, bgr, src, out_path, labels);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("total: %.3f ms (preproc + infer + postproc + overlay)   npu time of last "
                "infer: %.3f ms\n",
                wall_ms, engine.lastRunMicros() / 1000.0);
    return rc;
  } catch (const rcdl::Error& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
#endif  // RCDL_HAVE_OPENCV
}
