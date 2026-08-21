// MPP's mpp_buffer_* / mpp_log_* convenience macros expand MODULE_TAG into the
// allocator's bookkeeping tag, and MPP's headers deliberately do not define a
// fallback — every translation unit that uses them has to name itself first.
#define MODULE_TAG "rcdl_venc"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "mpp_common.h"
#include "rcdl/core/status.h"
#include "rcdl/media/video_codec.h"

namespace rcdl {

#if RCDL_HAVE_MPP

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kEncodeWaitMs = 40;    ///< encode(): wait for one ready packet
constexpr int kFlushWaitMs = 200;    ///< flush(): drain the rate controller's tail
constexpr int kRetrySleepUs = 500;   ///< back-off between feed() retries
constexpr int kHeaderBytes = 4096;   ///< SPS/PPS (+VPS) never comes near this
constexpr int kMinDim = 16;
constexpr int kMaxDim = 8192;

/// MPP clamps a control timeout to this many milliseconds (MPP_TIMEOUT_MAX).
constexpr long long kMppTimeoutMaxMs = 8000;

// ---------------------------------------------------------------------------
// Stride rule
// ---------------------------------------------------------------------------
//
// The VEPU fetches the input frame with a DMA whose row pitch is programmed in
// 16-byte units, and it codes in 16x16 macroblocks (H.264) / 16-row CTU rows
// (H.265), so both the row pitch and the plane height are aligned up:
//
//   hor_stride (BYTES) = alignUp(width, 16) * bytes-per-pixel  (1 for planar YUV)
//   ver_stride (ROWS)  = alignUp(height, 16)
//
// H.265 tolerates an 8-byte pitch and MJPEG an 8-row plane height, so 16
// satisfies all three encoders; it is also what MPP's own encoder sample
// computes. NOTE that MPP counts hor_stride in BYTES while ImageView::wstride
// counts PIXELS — mixing the two is the classic way to get a sheared picture out
// of a packed-RGB source.
//
// 4:2:0 input additionally needs even width and height: the chroma plane is half
// resolution in both axes and there is no half sample to code.

int horStrideBytes(int width, PixelFormat f) noexcept {
  const int bpp = bytesPerPixel(f);
  return alignUp(width, 16) * (bpp > 0 ? bpp : 1);
}

int verStrideRows(int height) noexcept { return alignUp(height, 16); }

bool needsEvenDims(PixelFormat f) noexcept { return isPlanarYuv(f); }

// ---------------------------------------------------------------------------
// Small RAII guards. Every early return below has to hand MPP's objects back,
// and the error paths outnumber the happy one.
// ---------------------------------------------------------------------------

struct FrameGuard {
  MppFrame f = nullptr;
  ~FrameGuard() {
    if (f) mpp_frame_deinit(&f);
  }
};

struct BufferGuard {
  MppBuffer b = nullptr;
  ~BufferGuard() {
    if (b) mpp_buffer_put(b);
  }
};

struct PacketGuard {
  MppPacket p = nullptr;
  ~PacketGuard() {
    if (p) mpp_packet_deinit(&p);
  }
};

MppEncRcMode toMppRcMode(RcMode m) noexcept {
  switch (m) {
    case RcMode::Vbr:
      return MPP_ENC_RC_MODE_VBR;
    case RcMode::FixQp:
      return MPP_ENC_RC_MODE_FIXQP;
    case RcMode::AvBr:
      return MPP_ENC_RC_MODE_AVBR;
    case RcMode::Cbr:
      break;
  }
  return MPP_ENC_RC_MODE_CBR;
}

/// Set one MppEncCfg key, or throw naming the key. MPP validates both the key
/// and the value here, so a typo fails at construction rather than silently
/// encoding with a default.
void setCfg(MppEncCfg cfg, const char* key, int val) {
  const MPP_RET ret = mpp_enc_cfg_set_s32(cfg, key, val);
  if (ret != MPP_OK) {
    throw Error(static_cast<int>(ret),
                std::string("RCDL: MppEncCfg rejected \"") + key + "\" = " +
                    std::to_string(val) + " (" + mpp::retName(ret) + ")");
  }
}

/// Set a key that only some MPP versions know, ignoring "no such key". Used for
/// the tuning knobs that are advisory (frame-rate denominators, H.265 profile
/// and level): losing one costs a default, not correctness.
bool setCfgOptional(MppEncCfg cfg, const char* key, int val) noexcept {
  return mpp_enc_cfg_set_s32(cfg, key, val) == MPP_OK;
}

/// Copy `rows` rows of `bytes` from a source with `src_stride` into a
/// destination with `dst_stride`. The two strides are almost never equal: the
/// caller's buffer is packed or RGA-aligned, the encoder's is 16-byte aligned.
void copyPlane(std::uint8_t* dst, std::size_t dst_stride, const std::uint8_t* src,
               std::size_t src_stride, int rows, std::size_t bytes) noexcept {
  for (int y = 0; y < rows; ++y) {
    std::memcpy(dst + static_cast<std::size_t>(y) * dst_stride,
                src + static_cast<std::size_t>(y) * src_stride, bytes);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// VideoEncoder::Impl
// ---------------------------------------------------------------------------

/// rk_mpi-backed decoupled encoder. Frames are queued on one side and packets
/// drained on the other, because a rate controller may hold a frame and emit
/// nothing for it — "one frame in, one packet out" is not a contract any
/// hardware encoder keeps.
struct VideoEncoder::Impl {
  VideoEncConfig cfg;
  MppCtx ctx = nullptr;
  MppApi* mpi = nullptr;
  MppEncCfg enc_cfg = nullptr;
  MppFrameFormat fmt = MPP_FMT_BUTT;

  /// Buffer type that worked for importing a caller's dma-buf fd. Probed once:
  /// the first import walks the candidate list, the rest go straight to the hit.
  MppBufferType import_type = MPP_BUFFER_TYPE_DMA_HEAP;
  bool import_type_known = false;

  /// Staging pool for host-only sources (fd < 0). Created on first use, so a
  /// pipeline that only ever feeds dma-bufs never allocates it.
  MppBufferGroup host_group = nullptr;

  /// Layout currently programmed into `prep:`. Starts at the rule above and is
  /// re-applied if a fed dma-buf is laid out differently (see feedFrame). Atomic
  /// because the feeder can change it while the drainer is inside flush(),
  /// which stamps the same geometry onto the end-of-stream marker.
  std::atomic<int> hor_stride{0};
  std::atomic<int> ver_stride{0};

  std::vector<std::uint8_t> extra;  ///< SPS/PPS (H.264) or VPS/SPS/PPS (H.265)
  bool header_pending = false;      ///< prepend `extra` to the first keyframe

  std::atomic<std::uint64_t> frames{0};
  std::atomic<bool> last_keyframe{false};
  bool eos_received = false;  ///< drainer-only

  /// Guards `eos_fed` only. Never held across an MPP call: feedEndOfStream()
  /// claims the flag under the lock and queues the marker outside it.
  std::mutex eos_mu;
  bool eos_fed = false;

  /// Cached MPP_SET_*_TIMEOUT values, so the common case does not re-issue a
  /// control on every frame. The feeder owns the input side and the drainer the
  /// output side, which is exactly the threading the header documents.
  long long in_timeout_ms = -2;   // -2: nothing programmed yet
  long long out_timeout_ms = -2;

  explicit Impl(const VideoEncConfig& c) : cfg(c) {
    const MppCodingType coding = mpp::codingType(cfg.codec);
    fmt = mpp::frameFormat(cfg.format);
    hor_stride = horStrideBytes(cfg.width, cfg.format);
    ver_stride = verStrideRows(cfg.height);

    RCDL_REQUIRE(mpp_check_support_format(MPP_CTX_ENC, coding) == MPP_OK,
                 "VideoEncoder: this MPP build cannot encode this codec");

    mpp::check(mpp_create(&ctx, &mpi), "mpp_create");
    // From here on the context exists, and a throwing constructor never runs
    // ~Impl — so every failure below must tear it down by hand or it leaks one
    // VPU context per attempt.
    try {
      mpp::check(mpp_init(ctx, MPP_CTX_ENC, coding), "mpp_init(encoder)");
      mpp::check(mpp_enc_cfg_init(&enc_cfg), "mpp_enc_cfg_init");
      configure();
      captureExtraData(coding);
    } catch (...) {
      if (enc_cfg) mpp_enc_cfg_deinit(enc_cfg);
      mpp_destroy(ctx);
      throw;
    }
    header_pending = !extra.empty();
  }

  ~Impl() {
    // Order matters: destroying the context stops MPP's worker threads, so
    // nothing can still be reading the staging pool when it is released.
    if (ctx) mpp_destroy(ctx);
    if (enc_cfg) mpp_enc_cfg_deinit(enc_cfg);
    if (host_group) mpp_buffer_group_put(host_group);
  }

  // -------------------------------------------------------------------------
  // Configuration
  // -------------------------------------------------------------------------

  void configure() {
    setCfg(enc_cfg, "prep:width", cfg.width);
    setCfg(enc_cfg, "prep:height", cfg.height);
    setCfg(enc_cfg, "prep:hor_stride", hor_stride);
    setCfg(enc_cfg, "prep:ver_stride", ver_stride);
    setCfg(enc_cfg, "prep:format", static_cast<int>(fmt));

    const int fps = cfg.fps > 0 ? cfg.fps : 30;
    const int gop = cfg.gop > 0 ? cfg.gop : fps * 2;
    const int bps = cfg.bitrate_kbps * 1000;

    setCfg(enc_cfg, "rc:mode", static_cast<int>(toMppRcMode(cfg.rc)));
    // Fixed input and output frame rate. The denominator key is spelled
    // differently across MPP releases and defaults to 1 when absent, so it is
    // set best-effort; the numerators are what the rate controller needs.
    setCfgOptional(enc_cfg, "rc:fps_in_flex", 0);
    setCfg(enc_cfg, "rc:fps_in_num", fps);
    if (!setCfgOptional(enc_cfg, "rc:fps_in_denom", 1)) {
      setCfgOptional(enc_cfg, "rc:fps_in_denorm", 1);
    }
    setCfgOptional(enc_cfg, "rc:fps_out_flex", 0);
    setCfg(enc_cfg, "rc:fps_out_num", fps);
    if (!setCfgOptional(enc_cfg, "rc:fps_out_denom", 1)) {
      setCfgOptional(enc_cfg, "rc:fps_out_denorm", 1);
    }
    setCfg(enc_cfg, "rc:gop", gop);

    // Bitrate bounds. MPP's own sample uses these ratios and the rate
    // controller is tuned for them: CBR gets a narrow +/- 1/16 corridor around
    // the target, VBR/AVBR keep the ceiling but drop the floor to 1/16 so a
    // static scene can spend almost nothing. FIXQP ignores the bounds entirely.
    setCfg(enc_cfg, "rc:bps_target", bps);
    switch (cfg.rc) {
      case RcMode::Cbr:
        setCfg(enc_cfg, "rc:bps_max", bps * 17 / 16);
        setCfg(enc_cfg, "rc:bps_min", bps * 15 / 16);
        break;
      case RcMode::Vbr:
      case RcMode::AvBr:
        setCfg(enc_cfg, "rc:bps_max", bps * 17 / 16);
        setCfg(enc_cfg, "rc:bps_min", bps * 1 / 16);
        break;
      case RcMode::FixQp:
        break;
    }

    setCfg(enc_cfg, "codec:type", static_cast<int>(mpp::codingType(cfg.codec)));
    switch (cfg.codec) {
      case VideoCodec::H264:
        configureH264();
        break;
      case VideoCodec::H265:
        configureH265();
        break;
      case VideoCodec::MJPEG:
        configureMjpeg();
        break;
      default:
        // Unreachable: the public constructor rejects everything else.
        throw Error(-1, "RCDL: VideoEncoder: unsupported codec");
    }

    mpp::check(mpi->control(ctx, MPP_ENC_SET_CFG, enc_cfg), "MPP_ENC_SET_CFG");
  }

  void configureH264() {
    const int profile = cfg.profile > 0 ? cfg.profile : 100;
    setCfg(enc_cfg, "h264:profile", profile);
    // level_idc only advertises what a decoder must cope with; picking one that
    // is too low makes strict decoders refuse the stream. 4.0 covers 1080p30,
    // 5.1 covers 4K.
    setCfg(enc_cfg, "h264:level", cfg.width * cfg.height > 1920 * 1088 ? 51 : 40);
    // CABAC and 8x8 transform exist only above Baseline — enabling either on a
    // profile that forbids it produces a stream no decoder will take.
    setCfg(enc_cfg, "h264:cabac_en", profile >= 77 ? 1 : 0);
    if (profile >= 77) setCfgOptional(enc_cfg, "h264:cabac_idc", 0);
    setCfgOptional(enc_cfg, "h264:trans8x8", profile >= 100 ? 1 : 0);
    configureQp();
  }

  void configureH265() {
    // general_profile_idc: 1 == Main (8-bit 4:2:0, which is all the VEPU emits).
    // general_level_idc is the level times 30, so 120 == 4.0 and 153 == 5.1.
    setCfgOptional(enc_cfg, "h265:profile", 1);
    setCfgOptional(enc_cfg, "h265:level", cfg.width * cfg.height > 1920 * 1088 ? 153 : 120);
    setCfgOptional(enc_cfg, "h265:tier", 0);
    configureQp();
  }

  void configureMjpeg() {
    // MJPEG has no QP: quality is a quantization-table scale factor, 1..99.
    const int q = cfg.qp > 0 ? std::min(cfg.qp, 99) : 80;
    setCfg(enc_cfg, "jpeg:q_factor", q);
    // Under CBR/VBR the rate controller drives q_factor between these bounds and
    // the q_factor above is only a starting point; under FIXQP pinning both ends
    // to q is what actually holds the requested quality.
    if (cfg.rc == RcMode::FixQp) {
      setCfgOptional(enc_cfg, "jpeg:qf_max", q);
      setCfgOptional(enc_cfg, "jpeg:qf_min", q);
    } else {
      setCfgOptional(enc_cfg, "jpeg:qf_max", 99);
      setCfgOptional(enc_cfg, "jpeg:qf_min", 1);
    }
  }

  /// QP block for H.264 / H.265. These keys live under `rc:` (not the codec
  /// block) because the rate controller owns them.
  void configureQp() {
    if (cfg.rc == RcMode::FixQp) {
      const int qp = cfg.qp > 0 ? cfg.qp : 26;
      setCfg(enc_cfg, "rc:qp_init", qp);
      setCfg(enc_cfg, "rc:qp_max", qp);
      setCfg(enc_cfg, "rc:qp_min", qp);
      setCfg(enc_cfg, "rc:qp_max_i", qp);
      setCfg(enc_cfg, "rc:qp_min_i", qp);
      setCfgOptional(enc_cfg, "rc:qp_ip", 0);
    } else {
      setCfg(enc_cfg, "rc:qp_init", -1);  // let the rate controller choose
      setCfg(enc_cfg, "rc:qp_max", 51);
      setCfg(enc_cfg, "rc:qp_min", 10);
      setCfg(enc_cfg, "rc:qp_max_i", 51);
      setCfg(enc_cfg, "rc:qp_min_i", 10);
      setCfgOptional(enc_cfg, "rc:qp_ip", 2);
    }
  }

  /// Re-program the input layout. A dma-buf handed to feed() carries its own
  /// strides — a decoded 1080p frame is 1920x1088, an rcdl::Image is 1920x1080 —
  /// and the chroma plane's offset is hor_stride * ver_stride, so encoding a
  /// buffer with someone else's strides shears the chroma. Rather than reject
  /// one of the two producers, re-apply `prep:` when the layout changes. Only
  /// the strides move; width/height do not, so the SPS/PPS captured at
  /// construction stays valid.
  void applyStrides(int hor, int ver) {
    setCfg(enc_cfg, "prep:hor_stride", hor);
    setCfg(enc_cfg, "prep:ver_stride", ver);
    mpp::check(mpi->control(ctx, MPP_ENC_SET_CFG, enc_cfg), "MPP_ENC_SET_CFG(stride)");
    hor_stride = hor;
    ver_stride = ver;
  }

  /// Capture the out-of-band codec header once, after configuration.
  ///
  /// MPP_ENC_GET_HDR_SYNC fills a packet the caller owns and marks the header as
  /// already delivered, so the encoder will not repeat it in front of the first
  /// frame — which is exactly why receive() has to prepend it there itself for a
  /// raw concatenation of packets to be playable. MPP_ENC_GET_EXTRA_INFO is the
  /// deprecated spelling that hands back an MPP-owned packet (never deinit it);
  /// it is the fallback for older runtimes.
  void captureExtraData(MppCodingType coding) {
    if (coding != MPP_VIDEO_CodingAVC && coding != MPP_VIDEO_CodingHEVC) {
      return;  // MJPEG carries its tables in every frame; there is no header
    }

    std::vector<std::uint8_t> scratch(kHeaderBytes);
    PacketGuard pkt;
    if (mpp_packet_init(&pkt.p, scratch.data(), scratch.size()) == MPP_OK && pkt.p) {
      mpp_packet_set_length(pkt.p, 0);  // MPP reports the header size through it
      if (mpi->control(ctx, MPP_ENC_GET_HDR_SYNC, pkt.p) == MPP_OK) {
        const auto* pos = static_cast<const std::uint8_t*>(mpp_packet_get_pos(pkt.p));
        const std::size_t len = mpp_packet_get_length(pkt.p);
        if (pos && len > 0) {
          extra.assign(pos, pos + len);
          return;
        }
      }
    }

    MppPacket legacy = nullptr;
    if (mpi->control(ctx, MPP_ENC_GET_EXTRA_INFO, &legacy) == MPP_OK && legacy) {
      const auto* pos = static_cast<const std::uint8_t*>(mpp_packet_get_pos(legacy));
      const std::size_t len = mpp_packet_get_length(legacy);
      if (pos && len > 0) extra.assign(pos, pos + len);
      // No deinit: that packet belongs to the encoder.
    }
  }

  // -------------------------------------------------------------------------
  // Timeouts
  // -------------------------------------------------------------------------

  void setTimeout(MpiCmd cmd, long long& cached, int ms) {
    long long want = ms <= 0 ? MPP_TIMEOUT_NON_BLOCK : std::min<long long>(ms, kMppTimeoutMaxMs);
    if (want == cached) return;
    RK_S64 param = static_cast<RK_S64>(want);
    mpp::check(mpi->control(ctx, cmd, &param), "MPP_SET_*_TIMEOUT");
    cached = want;
  }

  // -------------------------------------------------------------------------
  // Input
  // -------------------------------------------------------------------------

  /// Wrap the caller's dma-buf so the VPU reads the pixels where they already
  /// are. The import dup()s the fd, so the returned MppBuffer is independent of
  /// what the caller does with its own descriptor afterwards; the MppFrame takes
  /// a reference of its own, which is why the caller's reference can be dropped
  /// as soon as the frame is queued.
  MppBuffer importDmaBuf(int fd, std::size_t size) {
    MppBufferInfo info;
    std::memset(&info, 0, sizeof(info));
    info.fd = fd;
    info.size = size;
    info.index = -1;

    // dma-heap is what MPP itself allocates from on a 5.10+ kernel; EXT_DMA is
    // the generic "someone else's dma_buf fd" importer; DRM and ION are the
    // older paths. Whichever takes the fd first is remembered.
    static const MppBufferType kCandidates[] = {
        MPP_BUFFER_TYPE_DMA_HEAP,
        MPP_BUFFER_TYPE_EXT_DMA,
        MPP_BUFFER_TYPE_DRM,
        MPP_BUFFER_TYPE_ION,
    };

    if (import_type_known) {
      MppBuffer buf = nullptr;
      info.type = import_type;
      if (mpp_buffer_import(&buf, &info) == MPP_OK && buf) return buf;
      import_type_known = false;  // the pool changed under us; re-probe
    }
    for (MppBufferType type : kCandidates) {
      MppBuffer buf = nullptr;
      info.type = type;
      info.fd = fd;
      if (mpp_buffer_import(&buf, &info) == MPP_OK && buf) {
        import_type = type;
        import_type_known = true;
        return buf;
      }
    }
    throw Error(-1, "RCDL: VideoEncoder.feed could not import the source dma-buf fd into MPP");
  }

  /// Staging buffer for a host-only source, taken from an internal pool so a
  /// frame the encoder still holds is never overwritten by the next feed().
  MppBuffer stagingBuffer(std::size_t size) {
    if (!host_group) {
      static const MppBufferType kCandidates[] = {
          MPP_BUFFER_TYPE_DMA_HEAP,
          MPP_BUFFER_TYPE_DRM,
          MPP_BUFFER_TYPE_ION,
      };
      for (MppBufferType type : kCandidates) {
        // mpp_buffer_group_get_internal() is a variadic macro; call the function
        // it wraps so the empty __VA_ARGS__ is not an issue.
        if (mpp_buffer_group_get(&host_group, type, MPP_BUFFER_INTERNAL, MODULE_TAG,
                                 __FUNCTION__) == MPP_OK &&
            host_group) {
          break;
        }
        host_group = nullptr;
      }
      RCDL_REQUIRE(host_group != nullptr,
                   "VideoEncoder.feed: no MPP buffer heap available for a host-only source");
    }
    MppBuffer buf = nullptr;
    mpp::check(mpp_buffer_get(host_group, &buf, size), "mpp_buffer_get(encoder staging)");
    RCDL_REQUIRE(buf != nullptr, "VideoEncoder.feed: MPP returned a null staging buffer");
    return buf;
  }

  /// Copy a host-only image into an encoder-owned buffer, honouring both the
  /// source's stride and the encoder's — they are almost never the same number.
  void copyIntoStaging(const ImageView& src, MppBuffer buf) {
    auto* dst = static_cast<std::uint8_t*>(mpp_buffer_get_ptr(buf));
    RCDL_REQUIRE(dst != nullptr, "VideoEncoder.feed: staging buffer has no CPU mapping");

    const auto* s = static_cast<const std::uint8_t*>(src.data);
    const std::size_t src_row = src.rowBytes();
    const std::size_t dst_row = static_cast<std::size_t>(hor_stride);
    const int bpp = bytesPerPixel(src.format);

    if (bpp > 0) {
      copyPlane(dst, dst_row, s, src_row, src.height,
                static_cast<std::size_t>(src.width) * bpp);
    } else if (src.format == PixelFormat::NV12 || src.format == PixelFormat::NV21) {
      copyPlane(dst, dst_row, s, src_row, src.height, static_cast<std::size_t>(src.width));
      // One interleaved chroma plane at half height, same row pitch as luma.
      copyPlane(dst + dst_row * static_cast<std::size_t>(ver_stride), dst_row,
                s + src.uvOffset(), src_row, src.height / 2,
                static_cast<std::size_t>(src.width));
    } else {  // YUV420P: two quarter planes at half pitch
      copyPlane(dst, dst_row, s, src_row, src.height, static_cast<std::size_t>(src.width));
      const std::size_t dst_luma = dst_row * static_cast<std::size_t>(ver_stride);
      const std::size_t src_luma = src.uvOffset();
      const std::size_t dst_c_row = dst_row / 2;
      const std::size_t src_c_row = src_row / 2;
      const std::size_t dst_c_plane = dst_c_row * static_cast<std::size_t>(ver_stride / 2);
      const std::size_t src_c_plane = src_c_row * static_cast<std::size_t>(src.effHStride() / 2);
      const std::size_t bytes = static_cast<std::size_t>(src.width) / 2;
      copyPlane(dst + dst_luma, dst_c_row, s + src_luma, src_c_row, src.height / 2, bytes);
      copyPlane(dst + dst_luma + dst_c_plane, dst_c_row, s + src_luma + src_c_plane, src_c_row,
                src.height / 2, bytes);
    }

    // The CPU just wrote these bytes into a cached buffer and the VEPU will read
    // them through the IOMMU without snooping — flush before handing it over.
    mpp_buffer_sync_end(buf);
  }

  bool feedFrame(const ImageView& src, std::uint64_t pts_us, int timeout_ms) {
    RCDL_REQUIRE(src.valid(), "VideoEncoder.feed: source image is empty");
    RCDL_REQUIRE(src.format == cfg.format, "VideoEncoder.feed: source format != encoder format");
    RCDL_REQUIRE(src.width == cfg.width && src.height == cfg.height,
                 "VideoEncoder.feed: source size != encoder size");

    FrameGuard frame;
    mpp::check(mpp_frame_init(&frame.f), "mpp_frame_init");
    BufferGuard buf;

    if (src.fd >= 0) {
      // Zero copy: the VPU reads the caller's buffer in place. Its layout wins,
      // so the encoder is re-programmed to match rather than the other way
      // round. No cache maintenance here — whoever produced the pixels (RGA, the
      // VPU, the CPU under syncEnd()) owns that.
      const int src_hor = static_cast<int>(src.rowBytes());
      const int src_ver = src.effHStride();
      RCDL_REQUIRE((src_hor % 16) == 0,
                   "VideoEncoder.feed: dma-buf row stride must be a multiple of 16 bytes "
                   "(the VEPU programs its input pitch in 16-byte units)");
      RCDL_REQUIRE(src_ver >= src.height && (src_ver % 2) == 0,
                   "VideoEncoder.feed: dma-buf plane height must be even and >= height");
      const std::size_t need = mpp::frameBufferBytes(fmt, src_hor, src_ver);
      RCDL_REQUIRE(src.bytes() >= need,
                   "VideoEncoder.feed: dma-buf is smaller than its own strides describe");
      if (src_hor != hor_stride || src_ver != ver_stride) applyStrides(src_hor, src_ver);
      buf.b = importDmaBuf(src.fd, src.bytes());
    } else {
      // Host-only view: there is no fd to hand the VPU, so the rows are copied
      // into a buffer the encoder owns. This is the cost of not having a
      // dma-buf; every RCDL producer (Image, RGA destination, VideoFrame) has one.
      const std::size_t need = mpp::frameBufferBytes(fmt, hor_stride, ver_stride);
      buf.b = stagingBuffer(need);
      copyIntoStaging(src, buf.b);
    }

    mpp_frame_set_width(frame.f, static_cast<RK_U32>(cfg.width));
    mpp_frame_set_height(frame.f, static_cast<RK_U32>(cfg.height));
    mpp_frame_set_hor_stride(frame.f, static_cast<RK_U32>(hor_stride));
    mpp_frame_set_ver_stride(frame.f, static_cast<RK_U32>(ver_stride));
    mpp_frame_set_fmt(frame.f, fmt);
    mpp_frame_set_pts(frame.f, static_cast<RK_S64>(pts_us));
    mpp_frame_set_eos(frame.f, 0);
    mpp_frame_set_buffer(frame.f, buf.b);  // takes its own reference

    setTimeout(MPP_SET_INPUT_TIMEOUT, in_timeout_ms, timeout_ms);

    const auto deadline = Clock::now() + std::chrono::milliseconds(std::max(timeout_ms, 0));
    for (;;) {
      const MPP_RET ret = mpi->encode_put_frame(ctx, frame.f);
      if (ret == MPP_OK) {
        frames.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      // Not an error: the encoder's input queue is full because packets are not
      // being drained (MPP reports that as BUFFER_FULL, or as a timeout once an
      // input timeout is programmed). Tell the caller to drain and come back.
      if (ret != MPP_ERR_BUFFER_FULL && ret != MPP_ERR_TIMEOUT) {
        mpp::check(ret, "encode_put_frame");
      }
      if (Clock::now() >= deadline) return false;
      std::this_thread::sleep_for(std::chrono::microseconds(kRetrySleepUs));
    }
  }

  void feedEos() {
    {
      std::lock_guard<std::mutex> lock(eos_mu);
      if (eos_fed) return;
      eos_fed = true;
    }
    // A frame with no buffer and the EOS flag set is MPP's end-of-stream marker;
    // the geometry stays so the encoder recognises it as belonging to this
    // stream. Best effort: a full input queue at end of stream is not worth
    // throwing over, and the drain loop below terminates on the timeout anyway.
    FrameGuard frame;
    if (mpp_frame_init(&frame.f) != MPP_OK || !frame.f) return;
    mpp_frame_set_width(frame.f, static_cast<RK_U32>(cfg.width));
    mpp_frame_set_height(frame.f, static_cast<RK_U32>(cfg.height));
    mpp_frame_set_hor_stride(frame.f, static_cast<RK_U32>(hor_stride));
    mpp_frame_set_ver_stride(frame.f, static_cast<RK_U32>(ver_stride));
    mpp_frame_set_fmt(frame.f, fmt);
    mpp_frame_set_buffer(frame.f, nullptr);
    mpp_frame_set_eos(frame.f, 1);
    mpi->encode_put_frame(ctx, frame.f);
  }

  // -------------------------------------------------------------------------
  // Output
  // -------------------------------------------------------------------------

  bool recv(std::vector<std::uint8_t>& out, int timeout_ms) {
    out.clear();
    if (eos_received) return false;

    setTimeout(MPP_SET_OUTPUT_TIMEOUT, out_timeout_ms, timeout_ms);

    PacketGuard pkt;
    const MPP_RET ret = mpi->encode_get_packet(ctx, &pkt.p);
    // "Nothing ready yet" comes back as a timeout, as MPP_NOK, or as success
    // with a null packet depending on the runtime version. None of the three is
    // a failure: a configuration error would already have thrown at construction.
    if (ret == MPP_ERR_TIMEOUT || ret == MPP_NOK) return false;
    mpp::check(ret, "encode_get_packet");
    if (!pkt.p) return false;

    if (mpp_packet_get_eos(pkt.p)) eos_received = true;

    const auto* data = static_cast<const std::uint8_t*>(mpp_packet_get_pos(pkt.p));
    const std::size_t len = mpp_packet_get_length(pkt.p);
    if (!data || len == 0) return false;  // bare end-of-stream marker

    // MPP reports intra through the packet's metadata, not a packet flag.
    RK_S32 is_intra = 0;
    if (MppMeta meta = mpp_packet_get_meta(pkt.p)) {
      mpp_meta_get_s32(meta, KEY_OUTPUT_INTRA, &is_intra);
    }
    last_keyframe.store(is_intra != 0, std::memory_order_relaxed);

    // Prepend the codec header to the first keyframe. Fetching it at
    // construction told MPP the header had been delivered out of band, so it is
    // not in the stream — without this, a file written by concatenating packets
    // has no SPS/PPS and nothing will play it. The comparison guards against a
    // runtime that emitted the header anyway.
    out.reserve(len + (header_pending ? extra.size() : 0));
    if (is_intra && header_pending) {
      header_pending = false;
      if (len < extra.size() || std::memcmp(data, extra.data(), extra.size()) != 0) {
        out.insert(out.end(), extra.begin(), extra.end());
      }
    }
    out.insert(out.end(), data, data + len);
    return true;
  }
};

// ---------------------------------------------------------------------------
// VideoEncoder
// ---------------------------------------------------------------------------

VideoEncoder::VideoEncoder(const VideoEncConfig& cfg) {
  RCDL_REQUIRE(cfg.codec == VideoCodec::H264 || cfg.codec == VideoCodec::H265 ||
                   cfg.codec == VideoCodec::MJPEG,
               "VideoEncoder: the VPU encodes H.264, H.265 and MJPEG only");
  RCDL_REQUIRE(cfg.width >= kMinDim && cfg.width <= kMaxDim && cfg.height >= kMinDim &&
                   cfg.height <= kMaxDim,
               "VideoEncoder: width and height must be within [16, 8192]");
  // 4:2:0 has one chroma sample per 2x2 luma block, so an odd edge has no
  // chroma to code and the hardware refuses the frame.
  RCDL_REQUIRE(!needsEvenDims(cfg.format) || ((cfg.width % 2) == 0 && (cfg.height % 2) == 0),
               "VideoEncoder: 4:2:0 input (NV12/NV21/YUV420P) needs even width and height");
  RCDL_REQUIRE(cfg.fps > 0 && cfg.fps <= 240, "VideoEncoder: fps must be within [1, 240]");
  RCDL_REQUIRE(cfg.bitrate_kbps > 0 && cfg.bitrate_kbps <= 200000,
               "VideoEncoder: bitrate_kbps must be within [1, 200000]");
  RCDL_REQUIRE(cfg.gop >= 0, "VideoEncoder: gop must be >= 0");
  RCDL_REQUIRE(cfg.qp >= 0 && cfg.qp <= 99, "VideoEncoder: qp must be within [0, 99]");
  impl_ = std::make_unique<Impl>(cfg);
}

VideoEncoder::~VideoEncoder() = default;

bool VideoEncoder::feed(const ImageView& frame, std::uint64_t pts_us, int timeout_ms) {
  return impl_->feedFrame(frame, pts_us, timeout_ms);
}

bool VideoEncoder::receive(std::vector<std::uint8_t>& out, int timeout_ms) {
  return impl_->recv(out, timeout_ms);
}

std::vector<std::uint8_t> VideoEncoder::encode(const ImageView& frame, std::uint64_t pts_us) {
  std::vector<std::uint8_t> out;
  if (!impl_->feedFrame(frame, pts_us, kEncodeWaitMs)) return out;  // back-pressure
  impl_->recv(out, kEncodeWaitMs);
  return out;  // empty => the rate controller kept the frame; drain again later
}

bool VideoEncoder::flush(std::vector<std::uint8_t>& out) {
  impl_->feedEos();
  return impl_->recv(out, kFlushWaitMs);
}

void VideoEncoder::feedEndOfStream() { impl_->feedEos(); }

const std::vector<std::uint8_t>& VideoEncoder::extraData() const noexcept { return impl_->extra; }

bool VideoEncoder::lastPacketWasKeyframe() const noexcept {
  return impl_->last_keyframe.load(std::memory_order_relaxed);
}

VideoCodec VideoEncoder::codec() const noexcept { return impl_->cfg.codec; }
int VideoEncoder::width() const noexcept { return impl_->cfg.width; }
int VideoEncoder::height() const noexcept { return impl_->cfg.height; }

std::uint64_t VideoEncoder::framesEncoded() const noexcept {
  return impl_->frames.load(std::memory_order_relaxed);
}

#else  // !RCDL_HAVE_MPP

// No MPP in this build: every entry point fails with the same named cause
// instead of surfacing as a null dereference. The constructor throws, so no
// instance exists and the rest is unreachable — but it still has to link.

struct VideoEncoder::Impl {};

VideoEncoder::VideoEncoder(const VideoEncConfig&) { throw Error(-1, mpp::kUnavailable); }
VideoEncoder::~VideoEncoder() = default;

bool VideoEncoder::feed(const ImageView&, std::uint64_t, int) {
  throw Error(-1, mpp::kUnavailable);
}
bool VideoEncoder::receive(std::vector<std::uint8_t>&, int) {
  throw Error(-1, mpp::kUnavailable);
}
std::vector<std::uint8_t> VideoEncoder::encode(const ImageView&, std::uint64_t) {
  throw Error(-1, mpp::kUnavailable);
}
bool VideoEncoder::flush(std::vector<std::uint8_t>&) { throw Error(-1, mpp::kUnavailable); }
void VideoEncoder::feedEndOfStream() { throw Error(-1, mpp::kUnavailable); }

// noexcept accessors cannot throw; they are unreachable without a constructed
// encoder, so they report nothing rather than terminating the process.
const std::vector<std::uint8_t>& VideoEncoder::extraData() const noexcept {
  static const std::vector<std::uint8_t> kNone;
  return kNone;
}
bool VideoEncoder::lastPacketWasKeyframe() const noexcept { return false; }
VideoCodec VideoEncoder::codec() const noexcept { return VideoCodec::H264; }
int VideoEncoder::width() const noexcept { return 0; }
int VideoEncoder::height() const noexcept { return 0; }
std::uint64_t VideoEncoder::framesEncoded() const noexcept { return 0; }

#endif  // RCDL_HAVE_MPP

}  // namespace rcdl
