// VPU hardware video decode of a raw elementary stream, and the M2 zero-copy
// claim in one number: how many decoded frames came back carrying a dma-buf fd.
//
//   ./video_decode <stream.h264|stream.h265> [--frames N] [--codec h264|h265|vp9|av1]
//                  [--save out.nv12]
//
// The file is fed to the decoder in fixed-size CHUNKS, not access units: MPP's
// parser is left in split mode (VideoDecConfig::split_parse), so it finds the
// picture boundaries itself and the caller never has to walk NAL start codes.
// That is the whole point of the flag — an Annex-B splitter in the application
// is a bug farm (and useless for VP9/AV1, whose bitstreams have no start codes).
//
// What is reported and why it matters:
//   * display size vs the VPU's hor/ver stride — the decoder writes rows padded
//     up to its own alignment, so reading at `width` instead of `wstride` shears
//     the picture. Both are printed to make the difference visible;
//   * whether the frames came from an RCDL-allocated (external) buffer group;
//   * how many frames carried a dma-buf fd — a frame with an fd is one RGA can
//     letterbox straight into an NPU input tensor with no copy.
//
// The input must be a raw elementary stream. A container (.mkv/.mp4) has to be
// demuxed first — RCDL has no demuxer; `ffmpeg -i in.mkv -c:v copy out.h265`
// (or a GStreamer parse pipeline) produces what this program wants.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

namespace {

/// Pull decoded frames out of a file, feeding the decoder only when it has
/// nothing ready.
///
/// Feed and drain are decoupled in the API because a hardware decoder cannot
/// promise a frame per access unit (reorder), and because the decoder's input
/// queue fills: feeding faster than we drain stalls the VPU. This little pump
/// keeps the correct cadence — always take a ready frame first, feed only when
/// there is none, and resume a partially-fed chunk after the drain — behind a
/// blocking next() the examples can loop on.
class StreamPump {
 public:
  StreamPump(rcdl::VideoDecoder& dec, const std::string& path, std::size_t chunk = 256 * 1024)
      : dec_(dec), buf_(chunk) {
    fp_ = std::fopen(path.c_str(), "rb");
    RCDL_REQUIRE(fp_ != nullptr, "cannot open the input stream");
  }
  ~StreamPump() {
    if (fp_) std::fclose(fp_);
  }

  StreamPump(const StreamPump&) = delete;
  StreamPump& operator=(const StreamPump&) = delete;

  /// Next frame in display order, or false at end of stream.
  bool next(rcdl::VideoFrame& out) {
    for (;;) {
      if (dec_.receive(out, 0)) return true;  // something is ready: take it
      if (fed_ < len_) {                      // resume the chunk we were feeding
        if (dec_.feed(buf_.data() + fed_, len_ - fed_, 0, 20)) {
          fed_ = len_;
          stall_ = 0;
        } else if (dec_.receive(out, 5)) {
          stall_ = 0;
          return true;  // back-pressure: the decoder wants us to drain first
        } else {
          // Neither accepting input nor producing a frame is not back-pressure,
          // it is a wedged decoder — fail instead of spinning forever.
          RCDL_REQUIRE(++stall_ < 500, "the decoder stalled: no input accepted and no frame out");
        }
        continue;
      }
      if (!eof_) {
        len_ = std::fread(buf_.data(), 1, buf_.size(), fp_);
        fed_ = 0;
        bytes_ += len_;
        if (len_ == 0) {
          eof_ = true;
          dec_.feedEndOfStream();
        }
        continue;
      }
      if (dec_.flush(out)) return true;  // reorder tail
      return false;
    }
  }

  std::size_t bytesRead() const noexcept { return bytes_; }

 private:
  rcdl::VideoDecoder& dec_;
  std::FILE* fp_ = nullptr;
  std::vector<std::uint8_t> buf_;
  std::size_t len_ = 0;    ///< valid bytes in buf_
  std::size_t fed_ = 0;    ///< of which already accepted by the decoder
  std::size_t bytes_ = 0;  ///< total read from the file
  int stall_ = 0;          ///< consecutive spins with no progress
  bool eof_ = false;
};

rcdl::VideoCodec codecFromName(const std::string& n) {
  if (n == "h264" || n == "avc") return rcdl::VideoCodec::H264;
  if (n == "h265" || n == "hevc") return rcdl::VideoCodec::H265;
  if (n == "vp8") return rcdl::VideoCodec::VP8;
  if (n == "vp9") return rcdl::VideoCodec::VP9;
  if (n == "av1") return rcdl::VideoCodec::AV1;
  if (n == "mjpeg") return rcdl::VideoCodec::MJPEG;
  RCDL_REQUIRE(false, "unknown --codec (h264|h265|vp8|vp9|av1|mjpeg)");
  return rcdl::VideoCodec::H264;  // unreachable — RCDL_REQUIRE threw
}

/// Write the display-size NV12 planes of a frame, dropping the VPU's row
/// padding, so the file is exactly W*H*3/2 bytes (what a raw viewer expects:
/// `ffplay -f rawvideo -pix_fmt nv12 -video_size WxH out.nv12`).
void saveNv12(rcdl::VideoFrame& f, const std::string& path) {
  const std::uint8_t* p = f.beginCpuRead();
  RCDL_REQUIRE(p != nullptr, "frame buffer could not be mapped for CPU read");
  const rcdl::ImageView& v = f.view();
  std::FILE* fp = std::fopen(path.c_str(), "wb");
  if (!fp) {
    f.endCpuRead();
    RCDL_REQUIRE(false, "cannot open the --save output file");
  }
  const std::size_t ws = static_cast<std::size_t>(v.effWStride());
  for (int r = 0; r < v.height; ++r) {
    std::fwrite(p + static_cast<std::size_t>(r) * ws, 1, static_cast<std::size_t>(v.width), fp);
  }
  const std::uint8_t* uv = p + v.uvOffset();
  for (int r = 0; r < v.height / 2; ++r) {
    std::fwrite(uv + static_cast<std::size_t>(r) * ws, 1, static_cast<std::size_t>(v.width), fp);
  }
  std::fclose(fp);
  f.endCpuRead();
  std::printf("saved the first frame to %s (%d x %d NV12, %zu bytes)\n", path.c_str(), v.width,
              v.height, static_cast<std::size_t>(v.width) * v.height * 3 / 2);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <stream.h264|stream.h265> [--frames N] "
                 "[--codec h264|h265|vp8|vp9|av1] [--save out.nv12]\n",
                 argv[0]);
    return 1;
  }
  try {
    std::string path = argv[1];
    std::string codec_name, save_path;
    int max_frames = 0;  // 0 => the whole stream
    for (int i = 2; i < argc; ++i) {
      const std::string opt = argv[i];
      const bool has_value = i + 1 < argc;
      if (opt == "--frames") {
        RCDL_REQUIRE(has_value, "--frames needs a count");
        max_frames = std::atoi(argv[++i]);
      } else if (opt == "--codec") {
        RCDL_REQUIRE(has_value, "--codec needs a name");
        codec_name = argv[++i];
      } else if (opt == "--save") {
        RCDL_REQUIRE(has_value, "--save needs an output path");
        save_path = argv[++i];
      } else {
        RCDL_REQUIRE(false, "unknown option (see usage)");
      }
    }

    rcdl::VideoDecConfig cfg;
    if (!codec_name.empty()) {
      cfg.codec = codecFromName(codec_name);
    } else {
      RCDL_REQUIRE(rcdl::codecFromExtension(path, &cfg.codec),
                   "cannot guess the codec from the file extension — pass --codec");
    }

    rcdl::VideoDecoder dec(cfg);
    StreamPump pump(dec, path);

    std::printf("rcdl %s | stream: %s | codec: %s\n", RCDL_VERSION_STRING, path.c_str(),
                rcdl::codecName(cfg.codec));

    int frames = 0, with_fd = 0;
    bool reported = false;
    rcdl::VideoFrame frame;
    const auto t0 = std::chrono::steady_clock::now();
    while (pump.next(frame)) {
      if (frame.fd() >= 0) ++with_fd;
      ++frames;

      if (!reported) {
        reported = true;
        // Only now does the decoder know the geometry: the resolution and the
        // buffer size arrive with the stream's first info-change, not with the
        // constructor.
        std::printf("resolution: %dx%d %s\n", dec.width(), dec.height(),
                    rcdl::formatName(frame.format()));
        std::printf("VPU stride: %d x %d (%+d x %+d vs display) — rows are padded up to the\n"
                    "            decoder's alignment; read them at the stride, not the width\n",
                    dec.widthStride(), dec.heightStride(), dec.widthStride() - dec.width(),
                    dec.heightStride() - dec.height());
        std::printf("buffer group: %s\n",
                    dec.usingExternalBuffers() ? "external (RCDL-allocated dma-bufs)"
                                               : "MPP-internal (external group was refused)");
        std::printf("first frame: %s\n", frame.describe().c_str());
        if (!save_path.empty()) saveNv12(frame, save_path);
      }

      // Release the frame before asking for the next one. Every frame held is a
      // buffer the decoder cannot write into; holding the whole group deadlocks
      // the decode loop.
      frame.reset();
      if (max_frames && frames >= max_frames) break;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    std::printf("\ndecoded %d frame(s) from %zu bytes in %.1f ms — %.1f fps\n", frames,
                pump.bytesRead(), ms, frames > 0 ? frames * 1000.0 / ms : 0.0);
    std::printf("frames carrying a dma-buf fd: %d / %d%s\n", with_fd, frames,
                (frames > 0 && with_fd == frames)
                    ? "  (every frame is RGA/NPU-importable without a copy)"
                    : "");
    if (frames == 0) {
      std::fprintf(stderr,
                   "error: no frames decoded — wrong codec, or not an elementary stream?\n");
      return 1;
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
