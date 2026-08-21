#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rcdl/media/video_frame.h"
#include "rcdl/preproc/image.h"

namespace rcdl {

/// Hardware JPEG on the RK3588 VPU. There is no separate JPEG unit in MPP's
/// API: JPEG is `MPP_VIDEO_CodingMJPEG` through the same `rk_mpi` interface as
/// H.264, so these classes are thin, single-image wrappers over that path
/// rather than a different engine. Decode reaches 8K, encode 4K.
///
/// Why a dedicated wrapper at all, when VideoEncoder already accepts MJPEG:
/// a still image is one frame in, one complete file out, with no rate control,
/// no GOP and no reorder — so the streaming feed/drain API is the wrong shape
/// for it and the caller would have to write the flush loop every time.

/// Quantization quality, 1 (smallest) .. 99 (best). MPP's default is 80.
struct JpegEncConfig {
  int width = 0;
  int height = 0;
  /// Source format. NV12 is what the VPU decoder and RGA produce; the packed
  /// RGB formats are converted by MPP on the way in.
  PixelFormat format = PixelFormat::NV12;
  int quality = 80;
};

/// One context per (size, format, quality), reused for every image.
class JpegEncoder {
 public:
  explicit JpegEncoder(const JpegEncConfig& cfg);
  JpegEncoder(int width, int height, PixelFormat format = PixelFormat::NV12, int quality = 80)
      : JpegEncoder(JpegEncConfig{width, height, format, quality}) {}
  ~JpegEncoder();

  JpegEncoder(const JpegEncoder&) = delete;
  JpegEncoder& operator=(const JpegEncoder&) = delete;

  /// Encode one image into a complete JPEG file (JFIF, with headers).
  ///
  /// A `src` carrying a dma-buf fd is read in place by the VPU; a host-only view
  /// is copied into an encoder-owned buffer first. The returned bytes are always
  /// the caller's own copy, taken before the codec buffer is returned.
  std::vector<std::uint8_t> encode(const ImageView& src);

  /// Encode and write it to `path`. Throws rcdl::Error if the file cannot be
  /// written.
  void encodeToFile(const ImageView& src, const std::string& path);

  int width() const noexcept;
  int height() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Hardware JPEG decoder. Output lands in a dma-buf like any decoded video
/// frame, so a decoded JPEG can be letterboxed by RGA into an NPU input tensor
/// with no CPU touch — the batch-inference path for a directory of images.
class JpegDecoder {
 public:
  /// `format` is the output pixel format (NV12 is native; others cost MPP a
  /// conversion pass).
  explicit JpegDecoder(PixelFormat format = PixelFormat::NV12);
  ~JpegDecoder();

  JpegDecoder(const JpegDecoder&) = delete;
  JpegDecoder& operator=(const JpegDecoder&) = delete;

  /// Decode one complete JPEG file. Returns false if the hardware produced no
  /// frame (a truncated or unsupported file — notably progressive JPEG, which
  /// the VPU does not do).
  bool decode(const std::uint8_t* data, std::size_t size, VideoFrame& out);
  bool decode(const std::vector<std::uint8_t>& data, VideoFrame& out) {
    return decode(data.data(), data.size(), out);
  }
  /// Read and decode a file. Throws rcdl::Error if it cannot be read.
  bool decodeFile(const std::string& path, VideoFrame& out);

  /// Why the last decode() returned false, in the caller's words: "progressive
  /// JPEG: the VPU's JPEG core decodes baseline ... only", "not a JPEG, or
  /// truncated before its frame header", "the VPU produced no picture within
  /// the decode timeout". Empty after a successful decode. The bool is the
  /// contract; this is so a failure can be reported accurately instead of
  /// guessed at.
  const std::string& lastError() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rcdl
