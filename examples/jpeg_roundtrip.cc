// Hardware JPEG on the VPU: decode a file, re-encode it, and letterbox the
// decoded frame into a model-input canvas with RGA.
//
//   ./jpeg_roundtrip <image.jpg> [--quality Q] [--out out.jpg]
//
// Two things are demonstrated, and the second is the one that pays:
//
//   1. the round trip — decode -> encode -> decode, reporting file sizes, the
//      luma-only PSNR of the second decode against the first, and per-op
//      latency. That is the fidelity/latency budget of hardware JPEG;
//
//   2. the batch-inference chain — JPEG -> RGA letterbox -> an rcdl::Image in
//      RGB888. The decoded picture never leaves its dma-buf: MPP writes NV12
//      into it, RGA reads that fd and writes the letterboxed RGB into another
//      one, and the same fd would go straight to an NPU input tensor
//      (DetectionPipeline does exactly this, into the tensor itself). For a
//      directory of images this replaces libjpeg + cv::resize on the CPU.
//
// PSNR here is LUMA ONLY — the Y plane, ignoring chroma (see video_roundtrip.cc
// for why that is the useful shorthand and what it hides).
//
// The VPU decodes baseline 4:2:0 JPEG. A progressive JPEG produces no frame,
// which this program reports rather than pretending it decoded something.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

#if RCDL_HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace {

constexpr int kCanvas = 640;  // model input canvas for the letterbox demo
constexpr int kIters = 5;     // repeats for the steady-state timings

std::vector<std::uint8_t> readFile(const std::string& path) {
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  RCDL_REQUIRE(fp != nullptr, "cannot open the input image");
  std::fseek(fp, 0, SEEK_END);
  const long n = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::vector<std::uint8_t> buf(n > 0 ? static_cast<std::size_t>(n) : 0);
  const bool ok = buf.empty() || std::fread(buf.data(), 1, buf.size(), fp) == buf.size();
  std::fclose(fp);
  RCDL_REQUIRE(ok && !buf.empty(), "could not read the input image");
  return buf;
}

/// Copy the top-left `cw` x `ch` of a frame's Y plane into a packed host buffer,
/// honouring the VPU's row stride (0,0 => the whole plane). beginCpuRead() opens
/// the coherency window the hardware bypassed.
std::vector<std::uint8_t> copyLuma(rcdl::VideoFrame& f, int cw = 0, int ch = 0) {
  const std::uint8_t* p = f.beginCpuRead();
  RCDL_REQUIRE(p != nullptr, "frame buffer could not be mapped for CPU read");
  const rcdl::ImageView& v = f.view();
  if (cw <= 0 || cw > v.width) cw = v.width;
  if (ch <= 0 || ch > v.height) ch = v.height;
  std::vector<std::uint8_t> y(static_cast<std::size_t>(cw) * ch);
  const std::size_t ws = static_cast<std::size_t>(v.effWStride());
  for (int r = 0; r < ch; ++r) {
    std::memcpy(y.data() + static_cast<std::size_t>(r) * cw, p + static_cast<std::size_t>(r) * ws,
                static_cast<std::size_t>(cw));
  }
  f.endCpuRead();
  return y;
}

double psnr(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0;
  double se = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    se += d * d;
  }
  const double mse = se / static_cast<double>(a.size());
  return mse <= 0.0 ? 99.0 : 10.0 * std::log10(255.0 * 255.0 / mse);
}

double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <image.jpg> [--quality Q] [--out out.jpg]\n", argv[0]);
    return 1;
  }
  try {
    const std::string path = argv[1];
    std::string out_path;
    int quality = 80;
    for (int i = 2; i < argc; ++i) {
      const std::string opt = argv[i];
      const bool has_value = i + 1 < argc;
      if (opt == "--quality") {
        RCDL_REQUIRE(has_value, "--quality needs a value");
        quality = std::atoi(argv[++i]);
        RCDL_REQUIRE(quality >= 1 && quality <= 99, "--quality must be in 1..99");
      } else if (opt == "--out") {
        RCDL_REQUIRE(has_value, "--out needs a path");
        out_path = argv[++i];
      } else {
        RCDL_REQUIRE(false, "unknown option (see usage)");
      }
    }

    const std::vector<std::uint8_t> src = readFile(path);
    std::printf("rcdl %s | %s (%zu bytes)\n", RCDL_VERSION_STRING, path.c_str(), src.size());

    // --- decode on the VPU ---------------------------------------------------
    rcdl::JpegDecoder dec;  // NV12 out: the VPU's native format, and RGA's input
    rcdl::VideoFrame frame;
    RCDL_REQUIRE(dec.decode(src, frame),
                 "the VPU produced no frame — progressive or non-4:2:0 JPEG?");
    const int w = frame.width(), h = frame.height();
    double dec_ms = 0;
    for (int k = 0; k < kIters; ++k) {
      rcdl::VideoFrame tmp;
      const auto t = std::chrono::steady_clock::now();
      RCDL_REQUIRE(dec.decode(src, tmp), "decode failed on a repeat run");
      dec_ms += msSince(t);
    }
    std::printf("decode : %dx%d %s  stride %dx%d  fd %d  |  %.2f ms  (%.2f MP/s)\n", w, h,
                rcdl::formatName(frame.format()), frame.view().effWStride(),
                frame.view().effHStride(), frame.fd(), dec_ms / kIters,
                dec_ms > 0 ? static_cast<double>(w) * h * kIters / (dec_ms * 1000.0) : 0.0);

    // --- re-encode on the VPU, straight out of the decoder's dma-buf ---------
    rcdl::JpegEncoder enc(w, h, frame.format(), quality);
    std::vector<std::uint8_t> jpeg = enc.encode(frame.view());
    double enc_ms = 0;
    for (int k = 0; k < kIters; ++k) {
      const auto t = std::chrono::steady_clock::now();
      (void)enc.encode(frame.view());
      enc_ms += msSince(t);
    }
    RCDL_REQUIRE(jpeg.size() > 4, "the encoder produced no JPEG");
    const bool soi = jpeg[0] == 0xFF && jpeg[1] == 0xD8;
    const bool eoi = jpeg[jpeg.size() - 2] == 0xFF && jpeg[jpeg.size() - 1] == 0xD9;
    std::printf("encode : q=%d -> %zu bytes (%.2f bpp, %.2fx the source)  |  %.2f ms%s\n", quality,
                jpeg.size(), 8.0 * jpeg.size() / (static_cast<double>(w) * h),
                static_cast<double>(jpeg.size()) / static_cast<double>(src.size()),
                enc_ms / kIters, (soi && eoi) ? "" : "   WARNING: missing SOI/EOI marker");

    // --- decode the re-encode and compare luma -------------------------------
    rcdl::VideoFrame again;
    RCDL_REQUIRE(dec.decode(jpeg, again), "the re-encoded JPEG did not decode");
    // The hardware may align a decoded size up, so compare the region both
    // frames actually have rather than insisting the sizes match exactly.
    const int cw = std::min(w, again.width()), ch = std::min(h, again.height());
    std::printf("round-trip luma-only PSNR over %dx%d: %.2f dB\n", cw, ch,
                psnr(copyLuma(frame, cw, ch), copyLuma(again, cw, ch)));
    again.reset();

    if (!out_path.empty()) {
      std::FILE* fp = std::fopen(out_path.c_str(), "wb");
      RCDL_REQUIRE(fp != nullptr, "cannot open the --out file");
      std::fwrite(jpeg.data(), 1, jpeg.size(), fp);
      std::fclose(fp);
      std::printf("wrote %s\n", out_path.c_str());
    }

    // --- the useful path: JPEG -> RGA letterbox -> RGB888 image --------------
    // Range note: a JFIF JPEG carries FULL-RANGE YUV, unlike a video decoder's
    // studio-swing NV12 — so kAsIs is correct here and kStudioToFull would
    // stretch the levels a second time and clip the highlights.
    rcdl::Image canvas = rcdl::Image::alloc(kCanvas, kCanvas, rcdl::PixelFormat::RGB888);
    rcdl::PreprocBackend used = rcdl::PreprocBackend::Auto;
    rcdl::LetterboxInfo lb;
    double lb_ms = 0;
    for (int k = 0; k < kIters; ++k) {
      const auto t = std::chrono::steady_clock::now();
      // deviceView(): the fd only. Mapping the canvas for the CPU would be
      // wasted work on a path where only hardware touches it.
      lb = rcdl::letterbox(canvas.deviceView(), frame.view(), 114, rcdl::PreprocBackend::Auto,
                           rcdl::YuvRange::kAsIs, &used);
      lb_ms += msSince(t);
    }
    std::printf("letterbox: %dx%d NV12 -> %dx%d RGB888 on %s  scale=%.4f pad=(%.1f,%.1f)"
                "  |  %.2f ms\n",
                w, h, kCanvas, kCanvas, rcdl::backendName(used), lb.scale, lb.padX, lb.padY,
                lb_ms / kIters);

    // Sanity-check the canvas with the CPU (a hardware-only pipeline would not
    // do this): a plausible mean says RGA wrote a picture, not zeros.
    {
      const rcdl::ImageView cv_view = canvas.view();
      canvas.syncStart(true, false);
      const std::uint8_t* p = cv_view.bytePtr();
      double sum = 0;
      const std::size_t row = static_cast<std::size_t>(cv_view.effWStride()) * 3;
      for (int r = 0; r < cv_view.height; ++r) {
        for (int c = 0; c < cv_view.width * 3; ++c) sum += p[static_cast<std::size_t>(r) * row + c];
      }
      canvas.syncEnd(true, false);
      std::printf("           canvas mean %.1f (pad is 114; a flat 0 or 114 would mean RGA wrote "
                  "nothing)\n",
                  sum / (static_cast<double>(cv_view.width) * cv_view.height * 3));
    }

#if RCDL_HAVE_OPENCV
    // Optional cross-check: the same file through libjpeg. Chroma upsampling and
    // the YUV matrix differ between implementations, so this is a "same picture"
    // test, not a bit-exactness one.
    {
      const cv::Mat cpu = cv::imdecode(cv::Mat(1, static_cast<int>(src.size()), CV_8UC1,
                                               const_cast<std::uint8_t*>(src.data())),
                                       cv::IMREAD_GRAYSCALE);
      if (!cpu.empty() && cpu.cols == w && cpu.rows == h) {
        const std::vector<std::uint8_t> hw = copyLuma(frame);
        double sad = 0;
        for (int r = 0; r < h; ++r) {
          const std::uint8_t* row = cpu.ptr<std::uint8_t>(r);
          for (int c = 0; c < w; ++c) {
            sad += std::abs(static_cast<int>(hw[static_cast<std::size_t>(r) * w + c]) -
                            static_cast<int>(row[c]));
          }
        }
        std::printf("vs libjpeg (OpenCV): mean |luma diff| %.2f\n",
                    sad / (static_cast<double>(w) * h));
      }
    }
#endif

    frame.reset();
    return 0;
  } catch (const rcdl::Error& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
