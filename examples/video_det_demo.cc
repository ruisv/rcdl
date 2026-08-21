// The headline pipeline: a video file in, an annotated video file out, with
// every stage on the hardware that is meant to run it.
//
//   ./video_det_demo <model.rknn> <stream.h264> [--frames N] [--out out.h264] [--conf C]
//
//   VPU decode ──> NV12 dma-buf ──> RGA letterbox ──> NPU input tensor ──> NPU
//        ↑                              (no copy)         (no copy)          │
//        └── VPU encode <── RGA box overlay <── boxes in source pixels <──────┘
//
// The frame the VPU decoded is the frame RGA letterboxes, the frame RGA draws
// on, and the frame the VPU encodes: one dma-buf, one allocation per pool slot,
// zero memcpy on the frame path. The CPU only decodes the detection head and
// runs NMS. `DetectionPipeline::process(frame.view())` is where the hand-off
// happens — it hands the frame's fd to RGA and RGA's destination is the NPU
// input tensor Engine already bound with rknn_set_io_mem.
//
// The per-stage breakdown at the end is the number this program exists to
// produce: which unit bounds the pipeline, and what an end-to-end frame costs.
// Note that these stages run SEQUENTIALLY here — a synchronous pipeline is the
// honest baseline. Overlapping them (decode ‖ infer ‖ encode on their own
// threads) is a later milestone, and its speed-up is measured against this.
//
// Input must be a raw elementary stream (see video_decode.cc). Without --out
// the encode stage is skipped and the run is decode + detect only.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "rcdl/rcdl.h"

namespace {

/// Pull decoded frames out of a file, feeding the decoder only when it has
/// nothing ready. See video_decode.cc for why the cadence has to be this way.
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

  bool next(rcdl::VideoFrame& out) {
    for (;;) {
      if (dec_.receive(out, 0)) return true;
      if (fed_ < len_) {
        if (dec_.feed(buf_.data() + fed_, len_ - fed_, 0, 20)) {
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
        len_ = std::fread(buf_.data(), 1, buf_.size(), fp_);
        fed_ = 0;
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

 private:
  rcdl::VideoDecoder& dec_;
  std::FILE* fp_ = nullptr;
  std::vector<std::uint8_t> buf_;
  std::size_t len_ = 0;
  std::size_t fed_ = 0;
  int stall_ = 0;
  bool eof_ = false;
};

/// Eight distinguishable overlay colours, in im2d's 0xAABBGGRR packing. The
/// class id picks one, so a class keeps its colour from frame to frame.
constexpr std::uint32_t kColors[8] = {
    0xFF00FF00,  // green
    0xFF0000FF,  // red
    0xFFFF0000,  // blue
    0xFF00FFFF,  // yellow
    0xFFFF00FF,  // magenta
    0xFFFFFF00,  // cyan
    0xFF0080FF,  // orange
    0xFFFFFFFF,  // white
};

/// Draw one detection box onto an NV12 frame with RGA.
///
/// Coordinates are snapped to EVEN pixels: on 4:2:0 the chroma plane has half
/// the resolution, so an odd rectangle edge has no chroma sample of its own and
/// RGA rejects (or mis-renders) the fill. Clipping to the frame is left to
/// rgaDrawRect, which is documented to clip.
void drawBox(const rcdl::ImageView& dst, const rcdl::Detection& d, int thickness) {
  auto even = [](int v) { return v & ~1; };
  const int x1 = even(static_cast<int>(d.x1) < 0 ? 0 : static_cast<int>(d.x1));
  const int y1 = even(static_cast<int>(d.y1) < 0 ? 0 : static_cast<int>(d.y1));
  const int x2 = even(static_cast<int>(d.x2) > dst.width ? dst.width : static_cast<int>(d.x2));
  const int y2 = even(static_cast<int>(d.y2) > dst.height ? dst.height : static_cast<int>(d.y2));
  if (x2 - x1 < 4 || y2 - y1 < 4) return;  // too small to outline
  rcdl::rgaDrawRect(dst, x1, y1, x2 - x1, y2 - y1,
                    kColors[static_cast<unsigned>(d.class_id) % 8], thickness);
}

double msSince(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <model.rknn> <stream.h264> [--frames N] [--out out.h264] [--conf C]\n",
                 argv[0]);
    return 1;
  }
  try {
    const std::string model_path = argv[1];
    const std::string stream_path = argv[2];
    std::string out_path;
    int max_frames = 0;  // 0 => the whole stream
    float conf = 0.25f;
    for (int i = 3; i < argc; ++i) {
      const std::string opt = argv[i];
      const bool has_value = i + 1 < argc;
      if (opt == "--frames") {
        RCDL_REQUIRE(has_value, "--frames needs a count");
        max_frames = std::atoi(argv[++i]);
      } else if (opt == "--out") {
        RCDL_REQUIRE(has_value, "--out needs a path");
        out_path = argv[++i];
      } else if (opt == "--conf") {
        RCDL_REQUIRE(has_value, "--conf needs a threshold");
        conf = static_cast<float>(std::atof(argv[++i]));
        RCDL_REQUIRE(conf > 0.0f && conf < 1.0f, "--conf must be in (0,1)");
      } else {
        RCDL_REQUIRE(false, "unknown option (see usage)");
      }
    }

    // --- NPU -----------------------------------------------------------------
    rcdl::Engine engine(model_path);
    rcdl::PipelineConfig pcfg;
    pcfg.detect.conf_thresh = pcfg.ltrb.conf_thresh = conf;
    // The source is decoder NV12, i.e. studio-swing levels; the default
    // kStudioToFull is what expands them to the range the model was calibrated
    // on. Leaving it kAsIs costs accuracy silently.
    pcfg.yuv_range = rcdl::YuvRange::kStudioToFull;
    rcdl::DetectionPipeline pipe(engine, pcfg);
    const rcdl::PipelineConfig& rcfg = pipe.config();

    // --- VPU decode ----------------------------------------------------------
    rcdl::VideoDecConfig dcfg;
    RCDL_REQUIRE(rcdl::codecFromExtension(stream_path, &dcfg.codec),
                 "cannot guess the codec from the stream's extension");
    rcdl::VideoDecoder dec(dcfg);
    StreamPump pump(dec, stream_path);

    std::printf("rcdl %s | librknnrt %s | driver %s\n", RCDL_VERSION_STRING,
                engine.sdkVersion().c_str(), engine.driverVersion().c_str());
    std::printf("model : %s\n", engine.path().c_str());
    std::printf("canvas: %dx%d %s   head: %s   conf: %.2f\n", rcfg.input_w, rcfg.input_h,
                rcdl::formatName(rcfg.model_input), rcdl::headName(rcfg.head), conf);
    std::printf("stream: %s (%s)\n", stream_path.c_str(), rcdl::codecName(dcfg.codec));

    const bool can_draw = rcdl::rgaAvailable();
    if (!can_draw) {
      std::printf("note: RGA is unavailable, so the overlay stage is skipped "
                  "(boxes are still printed)\n");
    }

    // --- the loop ------------------------------------------------------------
    std::unique_ptr<rcdl::VideoEncoder> enc;
    std::FILE* out_fp = nullptr;
    std::vector<std::uint8_t> pkt;
    int frames = 0, fd_frames = 0;
    long total_dets = 0;
    std::size_t out_bytes = 0;
    double decode_ms = 0, draw_ms = 0, encode_ms = 0, wall_ms = 0;
    rcdl::VideoFrame frame;

    auto drainEncoder = [&](int timeout_ms) {
      while (enc->receive(pkt, timeout_ms)) {
        if (out_fp) std::fwrite(pkt.data(), 1, pkt.size(), out_fp);
        out_bytes += pkt.size();
        timeout_ms = 0;
      }
    };

    for (;;) {
      const auto frame_t0 = std::chrono::steady_clock::now();
      auto t = frame_t0;
      if (!pump.next(frame)) break;
      decode_ms += msSince(t);
      if (frame.fd() >= 0) ++fd_frames;

      if (frames == 0) {
        std::printf("decoded %dx%d %s (VPU stride %dx%d, buffer group: %s)\n\n", frame.width(),
                    frame.height(), rcdl::formatName(frame.format()), dec.widthStride(),
                    dec.heightStride(), dec.usingExternalBuffers() ? "external" : "MPP-internal");
        if (!out_path.empty()) {
          rcdl::VideoEncConfig ecfg;
          if (!rcdl::codecFromExtension(out_path, &ecfg.codec)) ecfg.codec = rcdl::VideoCodec::H264;
          ecfg.width = frame.width();
          ecfg.height = frame.height();
          ecfg.format = frame.format();
          enc = std::make_unique<rcdl::VideoEncoder>(ecfg);
          out_fp = std::fopen(out_path.c_str(), "wb");
          RCDL_REQUIRE(out_fp != nullptr, "cannot open the --out file");
          std::printf("encoding the annotated frames to %s (%s%s)\n", out_path.c_str(),
                      rcdl::codecName(ecfg.codec),
                      frame.fd() >= 0 ? ", straight from the decoder's dma-buf" : ", by COPY");
        }
      }

      // preproc + infer + postproc; the pipeline times its own three stages.
      const std::vector<rcdl::Detection> dets = pipe.process(frame.view());
      total_dets += static_cast<long>(dets.size());

      if (can_draw) {
        t = std::chrono::steady_clock::now();
        for (const rcdl::Detection& d : dets) drawBox(frame.view(), d, 2);
        draw_ms += msSince(t);
      }

      if (enc) {
        t = std::chrono::steady_clock::now();
        bool fed = false;
        for (int retry = 0; retry < 200 && !fed; ++retry) {
          fed = enc->feed(frame.view(), frame.ptsUs());
          if (!fed) drainEncoder(5);  // input queue full: drain and retry
        }
        RCDL_REQUIRE(fed, "the encoder never accepted a frame");
        drainEncoder(0);
        encode_ms += msSince(t);
      }

      ++frames;
      wall_ms += msSince(frame_t0);

      if (frames == 1) {
        std::printf("frame 1: %zu detection(s)  [preproc on %s, letterbox scale %.4f "
                    "pad (%.1f,%.1f)]\n",
                    dets.size(), rcdl::backendName(pipe.lastBackend()), pipe.lastLetterbox().scale,
                    pipe.lastLetterbox().padX, pipe.lastLetterbox().padY);
        for (const rcdl::Detection& d : dets) {
          std::printf("  %-16s %.3f  %.1f %.1f %.1f %.1f\n", rcdl::cocoClassName(d.class_id),
                      d.score, d.x1, d.y1, d.x2, d.y2);
        }
      } else if (frames <= 3 || frames % 30 == 0) {
        std::printf("frame %d: %zu detection(s)\n", frames, dets.size());
      }

      // Our reference only; the encoder keeps its own on the buffer until it is
      // done (MppBuffer is refcounted), so recycling it here cannot tear a frame.
      frame.reset();
      if (max_frames && frames >= max_frames) break;
    }

    if (enc) {  // the encoder still holds packets for frames already fed
      const auto t = std::chrono::steady_clock::now();
      while (enc->flush(pkt)) {
        if (out_fp) std::fwrite(pkt.data(), 1, pkt.size(), out_fp);
        out_bytes += pkt.size();
      }
      encode_ms += msSince(t);
    }
    if (out_fp) std::fclose(out_fp);

    if (frames == 0) {
      std::fprintf(stderr,
                   "error: no frames decoded — wrong codec, or not an elementary stream?\n");
      return 1;
    }

    // --- the number this program exists for ---------------------------------
    const rcdl::StageProfile& sp = pipe.profile();
    const double per_frame = wall_ms / frames;
    auto row = [&](const char* name, const char* unit, double total) {
      const double f = total / frames;
      std::printf("  %-9s %-26s %7.2f ms/f  %5.1f%%\n", name, unit, f,
                  per_frame > 0 ? 100.0 * f / per_frame : 0.0);
    };
    std::printf("\n=== per-frame breakdown (%d frames) ===\n", frames);
    row("decode", "VPU  H.26x -> NV12", decode_ms);
    row("preproc", "RGA  letterbox -> tensor", sp.preproc_ms);
    row("infer", "NPU  rknn_run", sp.infer_ms);
    row("postproc", "CPU  head decode + NMS", sp.postproc_ms);
    row("draw", can_draw ? "RGA  box overlay" : "(skipped, no RGA)", draw_ms);
    row("encode", enc ? "VPU  NV12 -> H.26x" : "(skipped, no --out)", encode_ms);
    std::printf("  %-9s %-26s %7.2f ms/f  100.0%%\n", "= total", "end to end", per_frame);
    std::printf("\nend-to-end: %.1f fps  |  detect-only (no decode/encode): %.1f fps"
                "  |  NPU infer alone: %.1f fps\n",
                1000.0 / per_frame,
                sp.totalMs() > 0 ? frames * 1000.0 / sp.totalMs() : 0.0,
                sp.infer_ms > 0 ? frames * 1000.0 / sp.infer_ms : 0.0);
    std::printf("frames carrying a dma-buf fd: %d / %d%s\n", fd_frames, frames,
                fd_frames == frames ? "  (decode -> RGA -> NPU -> encode without a copy)" : "");
    std::printf("detections: %ld total, %.1f per frame\n", total_dets,
                static_cast<double>(total_dets) / frames);
    if (enc) {
      std::printf("wrote %s (%zu bytes, %.0f kbps at 30 fps)\n", out_path.c_str(), out_bytes,
                  out_bytes * 8.0 * 30.0 / frames / 1000.0);
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
