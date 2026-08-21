// M2 quality check: VPU decode -> VPU re-encode -> VPU decode, with the luma
// PSNR between the first and the second decode.
//
//   ./video_roundtrip <stream.h264> [--frames N] [--codec h264|h265|vp9|av1]
//                     [--out out.h264] [--enc h264|h265] [--bitrate KBPS]
//
// The encode stage takes the DECODED FRAME'S dma-buf fd directly: the picture
// the VPU wrote is the picture the VPU encodes, with no memcpy and no host
// buffer in between (whether that actually happened is printed — a frame
// without an fd would silently fall back to a copy). That hand-off is the M2
// claim; the PSNR is what says the hand-off moved the right pixels.
//
// PSNR is computed on the LUMA (Y) PLANE ONLY — the standard shorthand for
// video quality, and the plane both codecs spend their bits on. Chroma is
// ignored here, so a chroma-only defect (swapped U/V, wrong plane offset) would
// NOT show up in this number; video_decode --save is the way to eyeball that.
//
// The reference frames are the first decode's Y planes, copied into host
// memory, so `--frames` bounds the memory this program uses (1080p luma is
// ~2 MB per frame). The two decoders do not live at the same time: the first
// decoder and the encoder are destroyed before the second decoder is built, so
// three 4K codec contexts never contend for VPU memory at once.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

namespace {

/// Pull decoded frames out of a file or an in-memory elementary stream, feeding
/// the decoder only when it has nothing ready. See video_decode.cc for why the
/// cadence is "drain first, feed second, resume the chunk afterwards".
class StreamPump {
 public:
  /// Feed from a file, one chunk at a time (the whole stream never has to be
  /// resident).
  StreamPump(rcdl::VideoDecoder& dec, const std::string& path, std::size_t chunk = 256 * 1024)
      : dec_(dec), buf_(chunk) {
    fp_ = std::fopen(path.c_str(), "rb");
    RCDL_REQUIRE(fp_ != nullptr, "cannot open the input stream");
  }
  /// Feed from bytes already in memory (the freshly encoded stream).
  StreamPump(rcdl::VideoDecoder& dec, const std::uint8_t* data, std::size_t size)
      : dec_(dec), mem_(data), mem_left_(size) {}

  ~StreamPump() {
    if (fp_) std::fclose(fp_);
  }

  StreamPump(const StreamPump&) = delete;
  StreamPump& operator=(const StreamPump&) = delete;

  bool next(rcdl::VideoFrame& out) {
    for (;;) {
      if (dec_.receive(out, 0)) return true;
      if (fed_ < len_) {
        if (dec_.feed(cur_ + fed_, len_ - fed_, 0, 20)) {
          fed_ = len_;
          stall_ = 0;
        } else if (dec_.receive(out, 5)) {
          stall_ = 0;
          return true;  // back-pressure: drain before feeding more
        } else {
          // Neither accepting input nor producing a frame is not back-pressure,
          // it is a wedged decoder — fail instead of spinning forever.
          RCDL_REQUIRE(++stall_ < 500, "the decoder stalled: no input accepted and no frame out");
        }
        continue;
      }
      if (!eof_) {
        refill();
        continue;
      }
      if (dec_.flush(out)) return true;  // reorder tail
      return false;
    }
  }

  std::size_t bytesRead() const noexcept { return bytes_; }

 private:
  void refill() {
    if (fp_) {
      len_ = std::fread(buf_.data(), 1, buf_.size(), fp_);
      cur_ = buf_.data();
    } else {
      len_ = mem_left_ < kMemChunk ? mem_left_ : kMemChunk;
      cur_ = mem_;
      mem_ += len_;
      mem_left_ -= len_;
    }
    fed_ = 0;
    bytes_ += len_;
    if (len_ == 0) {
      eof_ = true;
      dec_.feedEndOfStream();
    }
  }

  static constexpr std::size_t kMemChunk = 256 * 1024;

  rcdl::VideoDecoder& dec_;
  std::FILE* fp_ = nullptr;
  std::vector<std::uint8_t> buf_;
  const std::uint8_t* mem_ = nullptr;
  std::size_t mem_left_ = 0;
  const std::uint8_t* cur_ = nullptr;
  std::size_t len_ = 0;
  std::size_t fed_ = 0;
  std::size_t bytes_ = 0;
  int stall_ = 0;
  bool eof_ = false;
};

rcdl::VideoCodec codecFromName(const std::string& n) {
  if (n == "h264" || n == "avc") return rcdl::VideoCodec::H264;
  if (n == "h265" || n == "hevc") return rcdl::VideoCodec::H265;
  if (n == "vp8") return rcdl::VideoCodec::VP8;
  if (n == "vp9") return rcdl::VideoCodec::VP9;
  if (n == "av1") return rcdl::VideoCodec::AV1;
  if (n == "mjpeg") return rcdl::VideoCodec::MJPEG;
  RCDL_REQUIRE(false, "unknown codec name (h264|h265|vp8|vp9|av1|mjpeg)");
  return rcdl::VideoCodec::H264;  // unreachable — RCDL_REQUIRE threw
}

/// Copy a frame's Y plane out of the dma-buf into a packed host buffer.
///
/// This is the one CPU touch in the program, and it exists only to measure
/// quality: beginCpuRead() opens the coherency window the VPU's IOMMU writes
/// bypassed, so the bytes read here are the bytes the hardware wrote.
std::vector<std::uint8_t> copyLuma(rcdl::VideoFrame& f) {
  const std::uint8_t* p = f.beginCpuRead();
  RCDL_REQUIRE(p != nullptr, "frame buffer could not be mapped for CPU read");
  const rcdl::ImageView& v = f.view();
  std::vector<std::uint8_t> y(static_cast<std::size_t>(v.width) * v.height);
  const std::size_t ws = static_cast<std::size_t>(v.effWStride());
  for (int r = 0; r < v.height; ++r) {
    std::memcpy(y.data() + static_cast<std::size_t>(r) * v.width,
                p + static_cast<std::size_t>(r) * ws, static_cast<std::size_t>(v.width));
  }
  f.endCpuRead();
  return y;
}

/// Peak signal-to-noise ratio of two equally sized 8-bit planes, in dB.
/// Identical planes have no error, hence infinite PSNR — reported as a large
/// sentinel so the caller can print it without a special case.
double psnr(const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size() || a.empty()) return 0.0;
  double se = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    se += d * d;
  }
  const double mse = se / static_cast<double>(a.size());
  if (mse <= 0.0) return 99.0;
  return 10.0 * std::log10(255.0 * 255.0 / mse);
}

double fps(int frames, double ms) { return ms > 0.0 ? frames * 1000.0 / ms : 0.0; }

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <stream.h264> [--frames N] [--codec h264|h265|vp9|av1] "
                 "[--out out.h264] [--enc h264|h265] [--bitrate KBPS]\n",
                 argv[0]);
    return 1;
  }
  try {
    const std::string path = argv[1];
    std::string codec_name, enc_name = "h264", out_path;
    int max_frames = 60;  // bounds the host memory the reference planes take
    int bitrate_kbps = 4000;
    for (int i = 2; i < argc; ++i) {
      const std::string opt = argv[i];
      const bool has_value = i + 1 < argc;
      if (opt == "--frames") {
        RCDL_REQUIRE(has_value, "--frames needs a count");
        max_frames = std::atoi(argv[++i]);
      } else if (opt == "--codec") {
        RCDL_REQUIRE(has_value, "--codec needs a name");
        codec_name = argv[++i];
      } else if (opt == "--enc") {
        RCDL_REQUIRE(has_value, "--enc needs a name");
        enc_name = argv[++i];
      } else if (opt == "--out") {
        RCDL_REQUIRE(has_value, "--out needs a path");
        out_path = argv[++i];
      } else if (opt == "--bitrate") {
        RCDL_REQUIRE(has_value, "--bitrate needs a value in kbps");
        bitrate_kbps = std::atoi(argv[++i]);
      } else {
        RCDL_REQUIRE(false, "unknown option (see usage)");
      }
    }
    RCDL_REQUIRE(max_frames > 0, "--frames must be > 0 (the reference planes are held in memory)");

    rcdl::VideoDecConfig dcfg;
    if (!codec_name.empty()) {
      dcfg.codec = codecFromName(codec_name);
    } else {
      RCDL_REQUIRE(rcdl::codecFromExtension(path, &dcfg.codec),
                   "cannot guess the codec from the file extension — pass --codec");
    }
    const rcdl::VideoCodec enc_codec = codecFromName(enc_name);
    RCDL_REQUIRE(enc_codec == rcdl::VideoCodec::H264 || enc_codec == rcdl::VideoCodec::H265,
                 "--enc must be h264 or h265 (the VPU encodes those, plus MJPEG via JpegEncoder)");

    std::printf("rcdl %s | stream: %s | decode %s -> encode %s @ %d kbps\n", RCDL_VERSION_STRING,
                path.c_str(), rcdl::codecName(dcfg.codec), rcdl::codecName(enc_codec),
                bitrate_kbps);

    std::vector<std::vector<std::uint8_t>> reference;  // first decode's Y planes
    std::vector<std::uint8_t> stream;                  // the re-encoded bitstream
    std::size_t packets = 0, header_bytes = 0;
    int width = 0, height = 0;
    int decoded1 = 0, encoded = 0, fd_inputs = 0;
    double dec1_ms = 0, enc_ms = 0, read_ms = 0;
    bool external_group = false;

    // --- pass 1: decode, and encode straight out of the decoder's dma-buf ----
    {
      rcdl::VideoDecoder dec(dcfg);
      StreamPump pump(dec, path);
      std::unique_ptr<rcdl::VideoEncoder> enc;
      rcdl::VideoFrame frame;

      auto drain = [&](int timeout_ms) {
        std::vector<std::uint8_t> pkt;
        while (enc->receive(pkt, timeout_ms)) {
          stream.insert(stream.end(), pkt.begin(), pkt.end());
          ++packets;
          timeout_ms = 0;
        }
      };

      for (;;) {
        auto t = std::chrono::steady_clock::now();
        if (!pump.next(frame)) break;
        dec1_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t)
                       .count();
        ++decoded1;

        if (!enc) {
          width = frame.width();
          height = frame.height();
          external_group = dec.usingExternalBuffers();
          rcdl::VideoEncConfig ecfg;
          ecfg.codec = enc_codec;
          ecfg.width = width;
          ecfg.height = height;
          ecfg.format = frame.format();
          ecfg.bitrate_kbps = bitrate_kbps;
          enc = std::make_unique<rcdl::VideoEncoder>(ecfg);
          header_bytes = enc->extraData().size();
          std::printf("decoded %dx%d %s (VPU stride %dx%d, buffer group: %s)\n", width, height,
                      rcdl::formatName(frame.format()), dec.widthStride(), dec.heightStride(),
                      external_group ? "external" : "MPP-internal");
        }

        // Reference luma for the PSNR. Not part of the pipeline — timed apart so
        // it does not distort the decode/encode numbers.
        t = std::chrono::steady_clock::now();
        reference.push_back(copyLuma(frame));
        read_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();

        // The zero-copy hand-off: the encoder is given the decoded frame's view,
        // fd and all. A view without an fd would still encode — by copying.
        if (frame.fd() >= 0) ++fd_inputs;
        t = std::chrono::steady_clock::now();
        bool fed = false;
        for (int retry = 0; retry < 200 && !fed; ++retry) {
          fed = enc->feed(frame.view(), frame.ptsUs());
          if (!fed) drain(5);  // input queue full: drain packets and retry
        }
        RCDL_REQUIRE(fed, "the encoder never accepted a frame");
        drain(0);
        enc_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        ++encoded;

        // Releasing our reference here is safe only because the encoder keeps its
        // own on the buffer until it is done with it (MppBuffer is refcounted) —
        // recycling a frame mid-encode would tear the picture.
        frame.reset();  // back to the decoder's pool before we ask for another
        if (decoded1 >= max_frames) break;
      }
      RCDL_REQUIRE(enc != nullptr, "no frames decoded — wrong codec, or not an elementary stream?");

      const auto t = std::chrono::steady_clock::now();
      std::vector<std::uint8_t> pkt;
      while (enc->flush(pkt)) {  // the encoder still holds packets for fed frames
        stream.insert(stream.end(), pkt.begin(), pkt.end());
        ++packets;
      }
      enc_ms +=
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
    }  // decoder 1 and the encoder are destroyed here

    std::printf("encoder input carried a dma-buf fd: %d / %d frame(s)%s\n", fd_inputs, encoded,
                fd_inputs == encoded ? "  (zero-copy VPU -> VPU)" : "  (some frames were COPIED)");
    std::printf("encoded %d frame(s) -> %zu packet(s), %zu bytes (%.0f kbps at 30 fps, "
                "%zu-byte codec header)\n",
                encoded, packets, stream.size(),
                encoded > 0 ? stream.size() * 8.0 * 30.0 / encoded / 1000.0 : 0.0, header_bytes);
    if (!out_path.empty()) {
      std::FILE* fp = std::fopen(out_path.c_str(), "wb");
      RCDL_REQUIRE(fp != nullptr, "cannot open the --out file");
      std::fwrite(stream.data(), 1, stream.size(), fp);
      std::fclose(fp);
      std::printf("wrote %s\n", out_path.c_str());
    }

    // --- pass 2: decode what we just encoded --------------------------------
    int decoded2 = 0;
    double dec2_ms = 0, psnr_sum = 0, psnr_worst = 1e9;
    int compared = 0;
    {
      rcdl::VideoDecConfig cfg2;
      cfg2.codec = enc_codec;
      rcdl::VideoDecoder dec(cfg2);
      StreamPump pump(dec, stream.data(), stream.size());
      rcdl::VideoFrame frame;
      for (;;) {
        const auto t = std::chrono::steady_clock::now();
        if (!pump.next(frame)) break;
        dec2_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        if (decoded2 < static_cast<int>(reference.size()) && frame.width() == width &&
            frame.height() == height) {
          const double p = psnr(reference[static_cast<std::size_t>(decoded2)], copyLuma(frame));
          psnr_sum += p;
          if (p < psnr_worst) psnr_worst = p;
          ++compared;
        }
        ++decoded2;
        frame.reset();
      }
    }

    std::printf("\n=== round-trip ===\n");
    std::printf("  decode #1 : %5d frame(s)  %8.1f ms  %6.1f fps\n", decoded1, dec1_ms,
                fps(decoded1, dec1_ms));
    std::printf("  encode    : %5d frame(s)  %8.1f ms  %6.1f fps\n", encoded, enc_ms,
                fps(encoded, enc_ms));
    std::printf("  decode #2 : %5d frame(s)  %8.1f ms  %6.1f fps\n", decoded2, dec2_ms,
                fps(decoded2, dec2_ms));
    std::printf("  (host luma readback for the PSNR, not part of the pipeline: %.1f ms)\n",
                read_ms);
    if (compared > 0) {
      std::printf("  luma-only PSNR over %d frame(s): mean %.2f dB, worst %.2f dB\n", compared,
                  psnr_sum / compared, psnr_worst);
    } else {
      std::fprintf(stderr, "error: the second decode produced nothing comparable\n");
      return 1;
    }
    if (decoded2 < decoded1) {
      std::printf("  note: the second decode returned %d of %d frames — a re-encoded stream can\n"
                  "        lose the tail if the encoder buffered frames past our flush.\n",
                  decoded2, decoded1);
    }
    return 0;
  } catch (const rcdl::Error& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
