#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rcdl/media/video_frame.h"
#include "rcdl/preproc/image.h"

namespace rcdl {

/// Compression formats the RK3588 VPU handles through MPP's `rk_mpi` API.
/// Decode covers all of these; encode covers H264, H265 and MJPEG.
enum class VideoCodec {
  H264,
  H265,
  VP8,
  VP9,
  AV1,
  MJPEG,
};
const char* codecName(VideoCodec c) noexcept;

/// Was this build linked against MPP? False means every class below throws.
///
/// Compile-time only: it says the library is there, not that the VPU answered.
/// A real probe needs a context, so the decoder/encoder constructors are where a
/// missing or busy /dev/mpp_service surfaces, as an rcdl::Error naming it.
inline bool mppAvailable() noexcept {
#if RCDL_HAVE_MPP
  return true;
#else
  return false;
#endif
}
/// Guess the codec from a file extension (".h264"/".264", ".h265"/".hevc", ...).
/// Returns false when the extension says nothing.
bool codecFromExtension(const std::string& path, VideoCodec* out) noexcept;

// ===========================================================================
// Decoder
// ===========================================================================

struct VideoDecConfig {
  VideoCodec codec = VideoCodec::H264;
  /// Output pixel format. NV12 is the VPU's native 8-bit output and the only
  /// one RGA consumes without an extra pass; NV21/YUV420P are converted by MPP.
  PixelFormat format = PixelFormat::NV12;
  /// Let MPP's parser split the fed bytes into access units, so `feed()` accepts
  /// arbitrary chunks of an elementary stream. With this off the caller must
  /// feed exactly one access unit per call.
  bool split_parse = true;
  /// Decode frames into dma-bufs RCDL allocated (an MPP *external* buffer
  /// group), rather than MPP's internal pool.
  ///
  /// Both give a dma-buf fd, so zero-copy to RGA works either way. External
  /// buys control: the heap is ours, the count is ours, and the buffers can
  /// outlive the decoder. It is the default because the pipelines size the pool
  /// against their own queue depth. Falls back to the internal pool, with the
  /// same external API, if the group cannot be committed.
  bool external_buffers = true;
  /// Frames in the pool. 0 => derive from what the stream reports it needs
  /// (reference frames + reorder depth) plus headroom for the frames in flight.
  int buffer_count = 0;
  /// Extra frames beyond the stream's own requirement, i.e. how many decoded
  /// frames a consumer may hold at once. Too few stalls the decoder.
  int extra_buffers = 4;
  /// Deinterlacing / frame-rate doubling is off; RCDL targets progressive.
  bool immediate_out = true;  ///< emit frames as soon as they are ready
};

/// Hardware H.264/H.265/VP9/AV1 decoder on the VPU, through MPP's `rk_mpi`.
///
/// DECOUPLED feed/drain, for the same reason every hardware decoder needs it:
/// with B-frames, the display-order frame for an access unit is generally not
/// ready when that unit is fed. Feed bytes on one side, drain frames in display
/// order on the other, and flush the reorder tail at end of stream.
///
/// INFO CHANGE is the part that is easy to get wrong and is handled here: the
/// first frames tell the decoder the real resolution and the buffer size the
/// stream needs (including the horizontal/vertical stride alignment the VPU
/// writes with, which is NOT the display size). MPP signals this by returning a
/// frame with the info-change flag set; the decoder must then size and commit
/// its buffer group and acknowledge with `MPP_DEC_SET_INFO_CHANGE_READY` before
/// any picture arrives. Until that happens `width()`/`height()` report 0.
///
/// THREADING: one thread may feed while another drains — that is the intended
/// use and what the async video pipeline does. Two threads must not both drain.
class VideoDecoder {
 public:
  explicit VideoDecoder(const VideoDecConfig& cfg = VideoDecConfig());
  ~VideoDecoder();

  VideoDecoder(const VideoDecoder&) = delete;
  VideoDecoder& operator=(const VideoDecoder&) = delete;

  /// Queue compressed bytes. With `split_parse` these may be any chunk of the
  /// elementary stream; without it, exactly one access unit. Blocks up to
  /// `timeout_ms` when the decoder's input queue is full and returns false if it
  /// stays full — that is back-pressure, not an error: drain frames and retry.
  bool feed(const std::uint8_t* data, std::size_t size, std::uint64_t pts_us = 0,
            int timeout_ms = 20);
  bool feed(const std::vector<std::uint8_t>& data, std::uint64_t pts_us = 0,
            int timeout_ms = 20) {
    return feed(data.data(), data.size(), pts_us, timeout_ms);
  }

  /// Drain one decoded frame in display order. `timeout_ms == 0` is
  /// non-blocking. Returns false on timeout or at end of stream.
  bool receive(VideoFrame& out, int timeout_ms = 0);

  /// Feed one access unit and wait briefly for a frame. Convenience for
  /// low-latency streams with no reorder; the tail still needs flush().
  bool decode(const std::uint8_t* data, std::size_t size, VideoFrame& out);

  /// Signal end of stream once (idempotent), then drain what is left — call
  /// until it returns false.
  bool flush(VideoFrame& out);
  /// Queue the end-of-stream marker without draining, for a split feed/drain
  /// thread pair. Idempotent.
  void feedEndOfStream();

  /// Discard queued input and buffered frames without ending the stream — for
  /// seeking. Frames already handed out stay valid.
  void reset();

  /// Resolution reported by the stream. 0 until the first info-change.
  int width() const noexcept;
  int height() const noexcept;
  /// Row stride (pixels) and plane height the VPU actually writes with — both
  /// are aligned up from the display size and are what `VideoFrame::view()`
  /// carries. 0 until the first info-change.
  int widthStride() const noexcept;
  int heightStride() const noexcept;
  VideoCodec codec() const noexcept;
  std::uint64_t framesDecoded() const noexcept;
  bool endOfStream() const noexcept;
  /// True when frames come from an RCDL-allocated buffer group (see
  /// VideoDecConfig::external_buffers) — false if it fell back to MPP's pool.
  bool usingExternalBuffers() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ===========================================================================
// Encoder
// ===========================================================================

/// Rate-control mode. CBR is the safe default for a fixed-bandwidth sink; VBR
/// spends bits where they matter; FIXQP pins quality and lets the bitrate move.
enum class RcMode { Cbr, Vbr, FixQp, AvBr };

struct VideoEncConfig {
  VideoCodec codec = VideoCodec::H264;
  int width = 0;
  int height = 0;
  /// Input format. NV12 is what the VPU decoder emits and what RGA converts
  /// into cheapest; the packed RGB formats are accepted and converted by MPP.
  PixelFormat format = PixelFormat::NV12;
  int fps = 30;
  int bitrate_kbps = 4000;
  /// I-frame interval in frames. 0 => fps * 2.
  int gop = 0;
  RcMode rc = RcMode::Cbr;
  /// FIXQP / MJPEG quality: 1 (best) .. 51 for H.264/H.265 QP, or 1..99 for the
  /// MJPEG quantization factor. 0 => a sane default per codec.
  int qp = 0;
  /// H.264 profile: 66 baseline, 77 main, 100 high. Ignored for other codecs.
  int profile = 100;
};

/// Hardware H.264 / H.265 / MJPEG encoder on the VPU, through MPP's `rk_mpi`.
///
/// Same decoupled model as the decoder, and for the same reason: a rate
/// controller may buffer a frame and emit nothing, so "one frame in, one packet
/// out" is not a contract any encoder keeps.
///
/// Input frames are taken by `ImageView`. When the view carries a dma-buf fd
/// (a decoded VideoFrame, an RGA destination, an `rcdl::Image`) the encoder
/// wraps that buffer and reads it in place — no copy. A host-only view is
/// copied into an encoder-owned buffer, which is the cost of not having a
/// dma-buf; the examples take the zero-copy path.
///
/// The elementary stream is the concatenation of the packets in order.
/// `extraData()` returns the SPS/PPS (H.264) or VPS/SPS/PPS (H.265) header for
/// containers that want it out of band; it is also prepended to the first
/// keyframe packet, so writing packets back to back yields a playable file.
///
/// THREADING: as for the decoder — one feeder, one drainer.
class VideoEncoder {
 public:
  explicit VideoEncoder(const VideoEncConfig& cfg);
  ~VideoEncoder();

  VideoEncoder(const VideoEncoder&) = delete;
  VideoEncoder& operator=(const VideoEncoder&) = delete;

  /// Queue one frame. Returns false when the encoder's input queue is full —
  /// back-pressure, not an error: drain packets and retry.
  bool feed(const ImageView& frame, std::uint64_t pts_us = 0, int timeout_ms = 20);
  /// Drain one compressed packet. `timeout_ms == 0` is non-blocking.
  bool receive(std::vector<std::uint8_t>& out, int timeout_ms = 0);
  /// Feed one frame and wait briefly for a packet. May return empty; the tail
  /// still needs flush().
  std::vector<std::uint8_t> encode(const ImageView& frame, std::uint64_t pts_us = 0);
  /// End of stream, then drain — call until it returns false.
  bool flush(std::vector<std::uint8_t>& out);
  void feedEndOfStream();

  /// Codec header bytes (SPS/PPS or VPS/SPS/PPS). Empty for MJPEG.
  const std::vector<std::uint8_t>& extraData() const noexcept;
  /// True when the most recently drained packet was a keyframe.
  bool lastPacketWasKeyframe() const noexcept;

  VideoCodec codec() const noexcept;
  int width() const noexcept;
  int height() const noexcept;
  std::uint64_t framesEncoded() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rcdl
