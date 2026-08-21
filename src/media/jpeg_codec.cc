// MPP's mpp_buffer_* / mpp_log_* convenience macros expand MODULE_TAG into the
// allocator's bookkeeping tag, and MPP's headers deliberately do not define a
// fallback — every translation unit that uses them has to name itself first.
#define MODULE_TAG "rcdl_jpeg"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "mpp_common.h"
#include "rcdl/core/status.h"
#include "rcdl/media/jpeg_codec.h"
#include "rcdl/media/video_codec.h"

// There is no JPEG engine of its own in MPP's API: a JPEG is
// MPP_VIDEO_CodingMJPEG driven through the same rk_mpi context, cfg and
// packet/frame plumbing as H.264. So the encoder here is literally an MJPEG
// VideoEncoder with the streaming API folded away, and the decoder is an MJPEG
// MppCtx fed one self-contained access unit at a time.

namespace rcdl {

#if RCDL_HAVE_MPP

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kEncodeFeedMs = 200;    ///< how long to wait for a free input slot
constexpr int kEncodeTotalMs = 2000;  ///< a 4K frame encodes in single-digit ms
constexpr int kEncodePollMs = 10;
constexpr int kDecodeTotalMs = 2000;
constexpr int kDecodePollMs = 20;  ///< MPP blocks this long inside get_frame

struct PacketGuard {
  MppPacket p = nullptr;
  ~PacketGuard() {
    if (p) mpp_packet_deinit(&p);
  }
};

struct BufferGuard {
  MppBuffer b = nullptr;
  ~BufferGuard() {
    if (b) mpp_buffer_put(b);
  }
};

struct FrameGuard {
  MppFrame f = nullptr;
  ~FrameGuard() {
    if (f) mpp_frame_deinit(&f);
  }
  MppFrame release() noexcept {
    MppFrame r = f;
    f = nullptr;
    return r;
  }
};

/// What a JFIF frame header (SOFn) says about the picture.
struct JpegInfo {
  int width = 0;
  int height = 0;
  int components = 0;
  int marker = 0;  ///< the SOFn byte: 0xC0 baseline, 0xC2 progressive, ...
};

/// Walk the marker segments as far as the frame header. Cheap — it stops before
/// any entropy-coded data — and the decoder genuinely needs it; see decodeOne().
bool scanJpegHeader(const std::uint8_t* d, std::size_t n, JpegInfo* out) noexcept {
  if (n < 4 || d[0] != 0xFF || d[1] != 0xD8) return false;  // no SOI
  std::size_t i = 2;
  while (i + 3 < n) {
    if (d[i] != 0xFF) {  // padding between segments
      ++i;
      continue;
    }
    const std::uint8_t m = d[i + 1];
    if (m == 0xFF) {  // fill byte
      ++i;
      continue;
    }
    if (m == 0x01 || (m >= 0xD0 && m <= 0xD9)) {  // standalone, no length field
      i += 2;
      continue;
    }
    // SOFn is 0xC0..0xCF except DHT (0xC4), JPG (0xC8) and DAC (0xCC).
    if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC) {
      if (i + 9 >= n) return false;
      out->marker = m;
      out->height = (d[i + 5] << 8) | d[i + 6];
      out->width = (d[i + 7] << 8) | d[i + 8];
      out->components = d[i + 9];
      return out->width > 0 && out->height > 0;
    }
    if (m == 0xDA) return false;  // scan data before any frame header
    const std::size_t len = (static_cast<std::size_t>(d[i + 2]) << 8) | d[i + 3];
    if (len < 2) return false;
    i += 2 + len;
  }
  return false;
}

/// The VPU's JPEG core codes baseline (SOF0) and extended-sequential Huffman
/// (SOF1). Progressive (SOF2), lossless (SOF3) and every arithmetic-coded
/// variant are software-only.
bool isVpuDecodable(int sof_marker) noexcept {
  return sof_marker == 0xC0 || sof_marker == 0xC1;
}

std::string markerName(int m) {
  switch (m) {
    case 0xC2:
      return "progressive";
    case 0xC3:
      return "lossless";
    case 0xC9:
    case 0xCA:
    case 0xCB:
      return "arithmetic-coded";
    default:
      break;
  }
  char buf[8];
  std::snprintf(buf, sizeof(buf), "0xFF%02X", static_cast<unsigned>(m) & 0xFFu);
  return std::string("SOF marker ") + buf;
}

}  // namespace

// ---------------------------------------------------------------------------
// JpegEncoder
// ---------------------------------------------------------------------------

namespace {

/// A still image has no rate to control: FIXQP is the mode that leaves the
/// requested quantization factor alone (VideoEncoder additionally pins
/// jpeg:qf_max/qf_min to it in this mode, so the quality holds whatever the
/// runtime's rate controller believes). The bitrate and frame rate below are
/// only there to satisfy the shared config's validation.
VideoEncConfig toEncConfig(const JpegEncConfig& cfg) {
  VideoEncConfig v;
  v.codec = VideoCodec::MJPEG;
  v.width = cfg.width;
  v.height = cfg.height;
  v.format = cfg.format;
  v.fps = 30;
  v.bitrate_kbps = 10000;
  v.gop = 1;
  v.rc = RcMode::FixQp;
  v.qp = cfg.quality;
  return v;
}

}  // namespace

struct JpegEncoder::Impl {
  VideoEncoder enc;
  int width;
  int height;
  /// encode() is a feed AND a drain, so it is one transaction: two threads
  /// sharing the encoder would otherwise collect each other's images. The
  /// header promises nothing about concurrency here, and serialising is
  /// cheaper than surprising the caller.
  std::mutex mu;

  explicit Impl(const JpegEncConfig& cfg)
      : enc(toEncConfig(cfg)), width(cfg.width), height(cfg.height) {}
};

JpegEncoder::JpegEncoder(const JpegEncConfig& cfg) {
  RCDL_REQUIRE(cfg.quality >= 1 && cfg.quality <= 99,
               "JpegEncoder: quality must be within [1, 99]");
  impl_ = std::make_unique<Impl>(cfg);
}

JpegEncoder::~JpegEncoder() = default;

std::vector<std::uint8_t> JpegEncoder::encode(const ImageView& src) {
  std::lock_guard<std::mutex> lock(impl_->mu);

  // A dma-buf source is read in place by the VPU; a host-only view is copied
  // into an encoder-owned buffer. Both cases are VideoEncoder::feed's job.
  RCDL_REQUIRE(impl_->enc.feed(src, /*pts_us=*/0, kEncodeFeedMs),
               "JpegEncoder.encode: the VPU did not accept the frame in time");

  // One frame in, one complete JFIF out — but the packet still has to be
  // drained, and it does not appear in the same instant the frame is queued.
  std::vector<std::uint8_t> out;
  const auto deadline = Clock::now() + std::chrono::milliseconds(kEncodeTotalMs);
  do {
    if (impl_->enc.receive(out, kEncodePollMs) && !out.empty()) return out;
  } while (Clock::now() < deadline);

  throw Error(-1, "RCDL: JpegEncoder.encode produced no packet within " +
                      std::to_string(kEncodeTotalMs) + " ms");
}

void JpegEncoder::encodeToFile(const ImageView& src, const std::string& path) {
  const std::vector<std::uint8_t> bytes = encode(src);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  RCDL_REQUIRE(f.good(), "JpegEncoder.encodeToFile: cannot open the output file");
  f.write(reinterpret_cast<const char*>(bytes.data()),
          static_cast<std::streamsize>(bytes.size()));
  RCDL_REQUIRE(f.good(), "JpegEncoder.encodeToFile: write failed");
}

int JpegEncoder::width() const noexcept { return impl_->width; }
int JpegEncoder::height() const noexcept { return impl_->height; }

// ---------------------------------------------------------------------------
// JpegDecoder
// ---------------------------------------------------------------------------

struct JpegDecoder::Impl {
  MppCtx ctx = nullptr;
  MppApi* mpi = nullptr;
  MppBufferGroup grp = nullptr;      ///< decoded pictures
  MppBufferGroup pkt_grp = nullptr;  ///< compressed input
  MppFrameFormat fmt = MPP_FMT_YUV420SP;
  bool stream_ended = false;  ///< an EOS packet has been through this context
  std::uint64_t index = 0;
  std::string last_error;
  std::mutex mu;  ///< decode() is a feed and a drain: one image at a time

  explicit Impl(PixelFormat format) {
    fmt = mpp::frameFormat(format);
    mpp::check(mpp_create(&ctx, &mpi), "mpp_create");
    try {
      // Same rk_mpi decoder as H.264, only the coding type differs.
      mpp::check(mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingMJPEG), "mpp_init(jpeg decoder)");

      // One packet is one whole file, so MPP's frame splitter must stay off.
      // This goes through MppDecCfg (read-modify-write) rather than the legacy
      // MPP_DEC_SET_PARSER_SPLIT_MODE control, which is what MPP's own decoder
      // sample does and the only path exercised on current runtimes.
      MppDecCfg cfg = nullptr;
      mpp::check(mpp_dec_cfg_init(&cfg), "mpp_dec_cfg_init");
      try {
        mpp::check(mpi->control(ctx, MPP_DEC_GET_CFG, cfg), "MPP_DEC_GET_CFG");
        mpp::check(mpp_dec_cfg_set_u32(cfg, "base:split_parse", 0), "base:split_parse");
        mpp::check(mpi->control(ctx, MPP_DEC_SET_CFG, cfg), "MPP_DEC_SET_CFG");
      } catch (...) {
        mpp_dec_cfg_deinit(cfg);
        throw;
      }
      mpp_dec_cfg_deinit(cfg);

      MppFrameFormat want = fmt;
      mpp::check(mpi->control(ctx, MPP_DEC_SET_OUTPUT_FORMAT, &want),
                 "MPP_DEC_SET_OUTPUT_FORMAT");

      // Block briefly inside decode_get_frame instead of spinning; the decode
      // loop below bounds the total wait itself.
      RK_S64 timeout = kDecodePollMs;
      mpp::check(mpi->control(ctx, MPP_SET_OUTPUT_TIMEOUT, &timeout),
                 "MPP_SET_OUTPUT_TIMEOUT");

      // Pool the output pictures are decoded into. MPP's INTERNAL group is
      // used deliberately: an EXTERNAL group is the video decoder's business,
      // where the pool has to be sized against reorder depth and consumer queue
      // depth. A still image has neither. The buffers are dma-bufs either way,
      // so the zero-copy path to RGA and the NPU is unaffected.
      mpp::check(mpp_buffer_group_get(&grp, MPP_BUFFER_TYPE_DMA_HEAP, MPP_BUFFER_INTERNAL,
                                      MODULE_TAG, __FUNCTION__),
                 "mpp_buffer_group_get(jpeg output)");
      // Separate pool for the compressed bytes: the two have wildly different
      // sizes and a shared pool would hand a picture-sized buffer to a packet.
      mpp::check(mpp_buffer_group_get(&pkt_grp, MPP_BUFFER_TYPE_DMA_HEAP, MPP_BUFFER_INTERNAL,
                                      MODULE_TAG, __FUNCTION__),
                 "mpp_buffer_group_get(jpeg input)");
    } catch (...) {
      if (pkt_grp) mpp_buffer_group_put(pkt_grp);
      if (grp) mpp_buffer_group_put(grp);
      mpp_destroy(ctx);  // a throwing constructor never runs ~Impl
      throw;
    }
  }

  ~Impl() {
    // The context first: that stops MPP's worker threads before the pool the
    // pictures live in goes away. Buffers a caller still holds through a
    // VideoFrame keep the group alive until they are released.
    if (ctx) mpp_destroy(ctx);
    if (pkt_grp) mpp_buffer_group_put(pkt_grp);
    if (grp) mpp_buffer_group_put(grp);
  }

  bool fail(std::string why) {
    last_error = std::move(why);
    return false;
  }

  bool decodeOne(const std::uint8_t* data, std::size_t size, VideoFrame& out) {
    std::lock_guard<std::mutex> lock(mu);
    last_error.clear();

    // WHY the file is parsed here before the hardware sees it:
    //
    // MPP's MJPEG decoder does not allocate the output picture. Unlike H.264 —
    // where the decoder announces an info-change, sizes its own pool and hands
    // frames back — the JPEG path takes the destination frame IN, attached to
    // the packet's metadata as KEY_OUTPUT_FRAME, and writes the picture into
    // that buffer. (MPP's own decoder sample switches to this flow for
    // MPP_VIDEO_CodingMJPEG and takes the dimensions from its command line.)
    // Feeding a JPEG the ordinary way returns MPP_NOK from decode_get_frame
    // forever: no buffer in, no picture out, and no info-change either.
    //
    // So the buffer has to be sized before anything has decoded the file, which
    // means reading the frame header ourselves. It pays for itself twice: the
    // same scan says whether the VPU can code this file at all, which turns a
    // mystery timeout into an accurate "the hardware does not do progressive".
    JpegInfo info;
    if (!scanJpegHeader(data, size, &info)) {
      return fail("not a JPEG, or truncated before its frame header (no SOFn marker)");
    }
    if (!isVpuDecodable(info.marker)) {
      return fail(markerName(info.marker) +
                  " JPEG: the VPU's JPEG core decodes baseline and extended-sequential "
                  "Huffman only");
    }

    // The previous image was fed with EOS set, which leaves the context in the
    // end-of-stream state. Reset before the next one. Doing it here rather than
    // after the decode keeps the frame just handed out clear of the reset;
    // MppBuffer is reference counted, so a frame the caller still holds stays
    // valid across it either way.
    if (stream_ended) {
      mpp::check(mpi->reset(ctx), "mpp reset(jpeg decoder)");
      stream_ended = false;
    }

    // The destination picture. Only the buffer is filled in — MPP writes the
    // real width/height/stride/format into the frame as it parses.
    //
    // Size: the strides are the picture aligned up to 16 (what the JPEG core
    // writes with), and the byte budget is 4 per pixel. That is deliberately
    // generous — this buffer is committed before the chroma sampling is known,
    // and 4:2:0, 4:2:2 and 4:4:4 need 1.5, 2 and 3 bytes per pixel. MPP's own
    // sample commits exactly the same 4x for exactly this reason.
    const int hor_stride = alignUp(info.width, 16);
    const int ver_stride = alignUp(info.height, 16);
    const std::size_t need =
        static_cast<std::size_t>(hor_stride) * static_cast<std::size_t>(ver_stride) * 4u;

    FrameGuard frame;
    mpp::check(mpp_frame_init(&frame.f), "mpp_frame_init");
    {
      MppBuffer buf = nullptr;
      mpp::check(mpp_buffer_get(grp, &buf, need), "mpp_buffer_get(jpeg output)");
      RCDL_REQUIRE(buf != nullptr, "JpegDecoder: MPP returned a null output buffer");
      mpp_frame_set_buffer(frame.f, buf);  // takes its own reference
      mpp_buffer_put(buf);                 // ... so drop ours: the frame owns it now
    }

    // The compressed bytes have to live in an MppBuffer, not in the caller's
    // heap: this decode path hands the packet's buffer straight to the JPEG
    // core's bitstream DMA, and a packet with no buffer is rejected outright
    // ("Get no buffer from input packet") without ever reaching the parser.
    // So one copy of the compressed file is unavoidable here — it is the
    // *encoded* size, and the picture itself is still never copied.
    BufferGuard pkt_buf;
    // Padded to a page: the bitstream DMA may fetch past the last byte of the
    // file, and the pool reuses one buffer for images of similar size.
    constexpr std::size_t kPktAlign = 4096;
    const std::size_t pkt_bytes = ((size + kPktAlign - 1) / kPktAlign) * kPktAlign;
    mpp::check(mpp_buffer_get(pkt_grp, &pkt_buf.b, pkt_bytes), "mpp_buffer_get(jpeg input)");
    RCDL_REQUIRE(pkt_buf.b != nullptr, "JpegDecoder: MPP returned a null input buffer");
    auto* pkt_ptr = static_cast<std::uint8_t*>(mpp_buffer_get_ptr(pkt_buf.b));
    RCDL_REQUIRE(pkt_ptr != nullptr, "JpegDecoder: input buffer has no CPU mapping");
    std::memcpy(pkt_ptr, data, size);
    mpp_buffer_sync_end(pkt_buf.b);  // CPU wrote it, the VPU will DMA-read it

    PacketGuard pkt;
    mpp::check(mpp_packet_init_with_buffer(&pkt.p, pkt_buf.b), "mpp_packet_init_with_buffer");
    mpp_packet_set_pos(pkt.p, pkt_ptr);
    mpp_packet_set_length(pkt.p, size);  // the buffer is padded; the file is not
    // One packet is one whole file: tell the decoder there is nothing more
    // coming, so it emits the picture instead of waiting for the next image.
    mpp_packet_set_eos(pkt.p);
    stream_ended = true;

    MppMeta meta = mpp_packet_get_meta(pkt.p);
    RCDL_REQUIRE(meta != nullptr, "JpegDecoder: MPP packet carries no metadata");
    mpp::check(mpp_meta_set_frame(meta, KEY_OUTPUT_FRAME, frame.f),
               "mpp_meta_set_frame(KEY_OUTPUT_FRAME)");

    bool fed = false;
    const auto deadline = Clock::now() + std::chrono::milliseconds(kDecodeTotalMs);
    for (;;) {
      if (!fed && mpi->decode_put_packet(ctx, pkt.p) == MPP_OK) fed = true;

      MppFrame got = nullptr;
      const MPP_RET ret = mpi->decode_get_frame(ctx, &got);
      // A non-OK return WITH a frame attached is a real failure, and the frame
      // still has to go back — dropping it here leaks one MppFrame (and its
      // buffer reference) per occurrence until the deadline. A non-OK return
      // with no frame just means nothing is ready yet; the deadline ends that.
      if (ret != MPP_OK && got != nullptr) {
        if (got != frame.f) mpp_frame_deinit(&got);
        got = nullptr;
      }
      if (ret == MPP_OK && got != nullptr) {
        // Not expected on this path — the buffer was handed in, so there is
        // nothing for the decoder to ask for — but acknowledge it rather than
        // deadlocking if a runtime does raise one.
        if (mpp_frame_get_info_change(got)) {
          if (got != frame.f) mpp_frame_deinit(&got);
          mpp::check(mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr),
                     "MPP_DEC_SET_INFO_CHANGE_READY");
          continue;
        }
        // The decoder hands back the very frame it was given; anything else
        // would leave ours to clean up.
        if (got != frame.f) mpp_frame_deinit(&frame.f);
        frame.f = got;

        const ImageView view = mpp::viewOfFrame(got);
        if (mpp_frame_get_errinfo(got) || mpp_frame_get_discard(got)) {
          return fail("the VPU reported a decode error for this image (corrupt or truncated)");
        }
        if (view.fd < 0 || view.format == PixelFormat::Unknown) {
          return fail("the decoded picture is in a format RCDL has no name for");
        }
        out = VideoFrame::adopt(frame.release(), view,
                                static_cast<std::uint64_t>(mpp_frame_get_pts(got)), index++);
        return true;
      }

      if (Clock::now() >= deadline) {
        return fail(fed ? "the VPU produced no picture within the decode timeout"
                        : "the VPU never accepted the image (its input queue stayed full)");
      }
    }
  }
};

JpegDecoder::JpegDecoder(PixelFormat format) : impl_(std::make_unique<Impl>(format)) {}

JpegDecoder::~JpegDecoder() = default;

bool JpegDecoder::decode(const std::uint8_t* data, std::size_t size, VideoFrame& out) {
  RCDL_REQUIRE(data != nullptr && size > 0, "JpegDecoder.decode: empty input");
  return impl_->decodeOne(data, size, out);
}

const std::string& JpegDecoder::lastError() const noexcept { return impl_->last_error; }

#else  // !RCDL_HAVE_MPP

// No MPP in this build: every entry point fails with the same named cause
// instead of surfacing as a null dereference. The constructors throw, so no
// instance exists and the rest is unreachable — but it still has to link.

struct JpegEncoder::Impl {};
struct JpegDecoder::Impl {};

JpegEncoder::JpegEncoder(const JpegEncConfig&) { throw Error(-1, mpp::kUnavailable); }
JpegEncoder::~JpegEncoder() = default;

std::vector<std::uint8_t> JpegEncoder::encode(const ImageView&) {
  throw Error(-1, mpp::kUnavailable);
}
void JpegEncoder::encodeToFile(const ImageView&, const std::string&) {
  throw Error(-1, mpp::kUnavailable);
}
int JpegEncoder::width() const noexcept { return 0; }
int JpegEncoder::height() const noexcept { return 0; }

JpegDecoder::JpegDecoder(PixelFormat) { throw Error(-1, mpp::kUnavailable); }
JpegDecoder::~JpegDecoder() = default;

bool JpegDecoder::decode(const std::uint8_t*, std::size_t, VideoFrame&) {
  throw Error(-1, mpp::kUnavailable);
}
const std::string& JpegDecoder::lastError() const noexcept {
  static const std::string kNone;
  return kNone;
}

#endif  // RCDL_HAVE_MPP

// Plain file I/O, identical either way: the build without MPP fails inside
// decode() with the "no hardware codec" message rather than a file error.
bool JpegDecoder::decodeFile(const std::string& path, VideoFrame& out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  RCDL_REQUIRE(f.good(), "JpegDecoder.decodeFile: cannot open the file");
  const std::streamoff size = f.tellg();
  RCDL_REQUIRE(size > 0, "JpegDecoder.decodeFile: the file is empty");
  f.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  f.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  RCDL_REQUIRE(f.gcount() == static_cast<std::streamsize>(size),
               "JpegDecoder.decodeFile: read failed");
  return decode(bytes.data(), bytes.size(), out);
}

}  // namespace rcdl
