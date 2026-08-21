#ifndef MODULE_TAG
#define MODULE_TAG "rcdl_media"  // names this file in MPP's logs; see mpp_common.cc
#endif

#include "rcdl/media/video_codec.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rcdl/core/dma_buf.h"
#include "rcdl/core/status.h"

#include "mpp_common.h"

namespace rcdl {

// ===========================================================================
// Codec naming — free of MPP, so it also works in a build without the VPU.
// ===========================================================================

const char* codecName(VideoCodec c) noexcept {
  switch (c) {
    case VideoCodec::H264: return "H264";
    case VideoCodec::H265: return "H265";
    case VideoCodec::VP8: return "VP8";
    case VideoCodec::VP9: return "VP9";
    case VideoCodec::AV1: return "AV1";
    case VideoCodec::MJPEG: return "MJPEG";
  }
  return "unknown";
}

bool codecFromExtension(const std::string& path, VideoCodec* out) noexcept {
  if (out == nullptr) return false;
  // Only look at the last path component, so a dot in a directory name
  // ("~/clips.raw/stream") cannot be mistaken for an extension.
  const std::size_t slash = path.find_last_of('/');
  const std::size_t start = (slash == std::string::npos) ? 0 : slash + 1;
  const std::size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || dot < start || dot + 1 >= path.size()) return false;

  std::string ext = path.substr(dot + 1);
  for (char& ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

  // Elementary-stream extensions only. A container (.mp4, .mkv, .ts) says
  // nothing about which codec is inside it, so those deliberately return false
  // rather than guessing H.264 and failing at the first access unit.
  if (ext == "h264" || ext == "264" || ext == "avc") { *out = VideoCodec::H264; return true; }
  if (ext == "h265" || ext == "265" || ext == "hevc") { *out = VideoCodec::H265; return true; }
  if (ext == "vp8") { *out = VideoCodec::VP8; return true; }
  if (ext == "vp9") { *out = VideoCodec::VP9; return true; }
  if (ext == "av1" || ext == "obu") { *out = VideoCodec::AV1; return true; }
  if (ext == "mjpeg" || ext == "mjpg") { *out = VideoCodec::MJPEG; return true; }
  return false;
}

#if RCDL_HAVE_MPP

namespace {

using Clock = std::chrono::steady_clock;

// MppPollType caps a timeout at 8000 ms; anything larger is rejected as an
// invalid poll value, so clamp rather than let the control fail.
constexpr int kMaxPollMs = 8000;

// Only used when MPP's own input/output timeouts turn out to be no-ops (older
// runtimes ignore them in the simple decode_put_packet / decode_get_frame API).
// Without a pause here a back-pressured feed() or an empty drain would spin a
// core at 100% for the whole timeout window.
constexpr int kPollSleepUs = 200;

constexpr int kDecodeWaitMs = 40;   // decode(): wait for one ready frame
constexpr int kFlushWaitMs = 300;   // flush(): drain the reorder tail
constexpr int kEosFeedWaitMs = 500; // feedEndOfStream(): the marker must land

// Fallback when the stream does not report how many slots the DPB needs.
// Enough for an H.264 level-4 DPB plus reorder.
constexpr int kDefaultSlots = 8;
constexpr int kMinSlots = 4;
constexpr int kMaxSlots = 64;

// Buffer types tried, in order, for the external frame group. All four end up
// importing a plain dma-buf fd; which one the runtime accepts depends on how
// librockchip_mpp was built and on what the kernel exposes:
//   DMA_HEAP  — /dev/dma_heap/*, the native path on a 5.10+ kernel and what
//               rcdl::DmaBuf allocates from, so it is the first choice.
//   ION       — the legacy Android-style allocator, still present on some images
//   DRM       — imports through the DRM PRIME ioctls
//   EXT_DMA   — the generic "external dma_buf fd" importer; documented as such
//               in mpp_buffer.h and the last thing worth trying before giving
//               up on the external group entirely.
constexpr MppBufferType kBufferTypes[] = {
    MPP_BUFFER_TYPE_DMA_HEAP,
    MPP_BUFFER_TYPE_ION,
    MPP_BUFFER_TYPE_DRM,
    MPP_BUFFER_TYPE_EXT_DMA,
};

// One generation of frame buffers: the group MPP draws slots from, plus the
// dma-bufs behind it when we allocated them ourselves.
struct FramePool {
  MppBufferGroup group = nullptr;
  std::vector<DmaBuf> slots;  ///< empty for an MPP-internal group
  std::size_t slot_bytes = 0;
  int count = 0;
  bool external = false;
  MppBufferType type = MPP_BUFFER_TYPE_DMA_HEAP;
};

}  // namespace

// ===========================================================================
// VideoDecoder::Impl
// ===========================================================================
//
// BUFFER OWNERSHIP (the rule the rest of this file is written against):
//
//   The Impl owns every frame buffer. `pool_->slots` holds the DmaBufs, and
//   `pool_->group` is the MppBufferGroup MPP hands slots out of. A decoded
//   VideoFrame holds only an MppFrame, i.e. a REFERENCE into that group; its
//   destructor returns the reference, never memory.
//
//   Consequences, in order of how likely they are to bite:
//     1. A VideoFrame must be released before the VideoDecoder that produced
//        it. Outliving the decoder means mpp_frame_deinit() on a group that no
//        longer exists.
//     2. ~Impl calls mpp_destroy() FIRST — that stops the decode threads and
//        drops every reference MPP itself still holds — and only then puts the
//        groups and frees the dma-bufs.
//     3. A mid-stream resolution change cannot free the old buffers, because
//        the caller may still be holding frames from them. The old pool is
//        moved to `retired_` and released in ~Impl instead.
//
// THREADING: `mu_` guards only the small shared state (resolution, eos flags,
// frame counter, which pool is current). It is never held across a call into
// MPP — decode_put_packet and decode_get_frame both block, and MPP has its own
// locking, so holding ours across them would serialise the feeder against the
// drainer for no reason.
struct VideoDecoder::Impl {
  VideoDecConfig cfg;
  MppCtx ctx = nullptr;
  MppApi* mpi = nullptr;

  std::unique_ptr<FramePool> pool_;
  std::vector<std::unique_ptr<FramePool>> retired_;

  mutable std::mutex mu_;
  int width = 0;
  int height = 0;
  int hor_stride = 0;
  int ver_stride = 0;
  bool info_ready = false;
  bool external = false;
  bool eos_fed = false;
  bool eos_seen = false;
  std::uint64_t frames = 0;

  // Cached MPP timeout settings. The feeder only touches the input pair and the
  // drainer only the output pair, so these need no lock. `*_ctl_ok` goes false
  // once a runtime rejects the command, after which the deadline loops below
  // carry the timeout on their own.
  int in_timeout_ms = -999;  // -999 = "never set"
  int out_timeout_ms = -999;
  bool in_ctl_ok = true;
  bool out_ctl_ok = true;

  explicit Impl(const VideoDecConfig& c) : cfg(c) {
    const MppCodingType coding = mpp::codingType(cfg.codec);
    mpp::check(mpp_create(&ctx, &mpi), "mpp_create");
    // From here on the context exists; a throw would leak it (a constructor
    // that throws never runs ~Impl), so everything else goes through a catch.
    try {
      // MPP_DEC_SET_PARSER_SPLIT_MODE is annotated "Need to setup before init"
      // in rk_mpi_cmd.h: the parser is instantiated by mpp_init(), and whether
      // it owns a frame spliter is decided there. With split on, feed() may be
      // handed arbitrary chunks of an elementary stream; with it off every
      // feed() must be exactly one access unit.
      RK_U32 split = cfg.split_parse ? 1u : 0u;
      mpp::check(mpi->control(ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &split),
                 "MPP_DEC_SET_PARSER_SPLIT_MODE");

      mpp::check(mpp_init(ctx, MPP_CTX_DEC, coding), "mpp_init");

      // Newer MPP prefers the MppDecCfg object over the legacy per-command
      // controls and treats it as authoritative, so restate split_parse there.
      // Best-effort on purpose: a runtime that does not know the key name is
      // exactly the runtime whose legacy control above already did the job, and
      // failing the whole decoder over a duplicated setting would be absurd.
      applyDecCfg();

      if (cfg.immediate_out) {
        // Emit each picture as soon as the hardware is done with it instead of
        // holding it back for the display-order queue to fill. Latency, not
        // throughput — and it does not reorder anything: MPP still outputs in
        // display order, it just stops adding slack.
        RK_U32 immediate = 1;
        mpp::check(mpi->control(ctx, MPP_DEC_SET_IMMEDIATE_OUT, &immediate),
                   "MPP_DEC_SET_IMMEDIATE_OUT");
      }

      if (cfg.format != PixelFormat::NV12) {
        // NV12 is what the VPU writes natively; anything else costs a pass
        // through MPP's format converter. Only ask when it is actually needed.
        MppFrameFormat fmt = mpp::frameFormat(cfg.format);
        mpp::check(mpi->control(ctx, MPP_DEC_SET_OUTPUT_FORMAT, &fmt),
                   "MPP_DEC_SET_OUTPUT_FORMAT");
      }

      // Start non-blocking on both ports; feed()/receive() raise the timeout to
      // whatever the caller asked for.
      setInputTimeout(0);
      setOutputTimeout(0);
    } catch (...) {
      mpp_destroy(ctx);
      ctx = nullptr;
      mpi = nullptr;
      throw;
    }
  }

  ~Impl() {
    // Step 2 of the ownership rule: tear the context down before the memory it
    // was decoding into.
    if (ctx != nullptr) {
      mpp_destroy(ctx);
      ctx = nullptr;
      mpi = nullptr;
    }
    releasePool(pool_.get());
    for (auto& p : retired_) releasePool(p.get());
  }

  static void releasePool(FramePool* p) noexcept {
    if (p == nullptr) return;
    if (p->group != nullptr) {
      mpp_buffer_group_put(p->group);
      p->group = nullptr;
    }
    p->slots.clear();  // closes our dma-buf fds; MPP dup'd its own on commit
  }

  // -------------------------------------------------------------------------
  // Configuration helpers
  // -------------------------------------------------------------------------

  void applyDecCfg() noexcept {
    MppDecCfg dec_cfg = nullptr;
    if (mpp_dec_cfg_init(&dec_cfg) != MPP_OK || dec_cfg == nullptr) return;
    // Read the decoder's current config first: MPP_DEC_SET_CFG applies every
    // field of the object, so writing one built from scratch would stamp zeroed
    // defaults over settings the runtime chose for this codec.
    if (mpi->control(ctx, MPP_DEC_GET_CFG, dec_cfg) == MPP_OK) {
      mpp_dec_cfg_set_u32(dec_cfg, "base:split_parse", cfg.split_parse ? 1u : 0u);
      if (cfg.immediate_out) mpp_dec_cfg_set_u32(dec_cfg, "base:fast_out", 1u);
      mpi->control(ctx, MPP_DEC_SET_CFG, dec_cfg);
    }
    mpp_dec_cfg_deinit(dec_cfg);
  }

  // MPP_SET_INPUT_TIMEOUT / MPP_SET_OUTPUT_TIMEOUT take an RK_S64 holding an
  // MppPollType: 0 = non-blocking, negative = block forever, positive = ms
  // (capped at 8000). Blocking inside MPP is far better than polling from here
  // — it parks the thread on the runtime's own condition variable instead of
  // burning a core — so this is the primary waiting mechanism and the sleep
  // loops in feed()/recv() are only the fallback.
  void setInputTimeout(int ms) noexcept {
    if (!in_ctl_ok || ms == in_timeout_ms) return;
    RK_S64 v = std::min(ms, kMaxPollMs);
    if (mpi->control(ctx, MPP_SET_INPUT_TIMEOUT, &v) != MPP_OK) {
      in_ctl_ok = false;
      return;
    }
    in_timeout_ms = ms;
  }

  void setOutputTimeout(int ms) noexcept {
    if (!out_ctl_ok || ms == out_timeout_ms) return;
    RK_S64 v = std::min(ms, kMaxPollMs);
    if (mpi->control(ctx, MPP_SET_OUTPUT_TIMEOUT, &v) != MPP_OK) {
      out_ctl_ok = false;
      return;
    }
    out_timeout_ms = ms;
  }

  // -------------------------------------------------------------------------
  // Feed side
  // -------------------------------------------------------------------------

  // Queue one chunk of bitstream. Returns false only for back-pressure.
  bool putPacket(const std::uint8_t* data, std::size_t size, std::uint64_t pts_us,
                 bool eos, int timeout_ms) {
    // A dummy byte for the end-of-stream marker: the packet carries no payload
    // (length 0) but mpp_packet_init wants a non-null base address, and the
    // packet is deinit'd before this frame exits, so a local is fine.
    std::uint8_t eos_byte = 0;
    void* base = eos ? static_cast<void*>(&eos_byte)
                     : const_cast<void*>(static_cast<const void*>(data));
    const std::size_t len = eos ? 0 : size;

    MppPacket pkt = nullptr;
    mpp::check(mpp_packet_init(&pkt, base, len), "mpp_packet_init");
    // decode_put_packet COPIES the bytes into MPP's own stream buffer before it
    // returns, so the caller's memory only has to stay alive for this call —
    // which is what lets feed() take a plain const pointer.
    mpp_packet_set_pos(pkt, base);
    mpp_packet_set_length(pkt, len);
    mpp_packet_set_pts(pkt, static_cast<RK_S64>(pts_us));
    if (eos) mpp_packet_set_eos(pkt);

    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    setInputTimeout(timeout_ms);

    bool queued = false;
    MPP_RET ret = MPP_OK;
    for (;;) {
      ret = mpi->decode_put_packet(ctx, pkt);
      if (ret == MPP_OK) {
        queued = true;
        break;
      }
      // MPP_ERR_BUFFER_FULL is the decoder saying "my input queue is full" —
      // back-pressure, not a failure. The caller's contract is to drain frames
      // and try again, so it must not surface as an exception.
      if (ret != MPP_ERR_BUFFER_FULL) break;
      if (Clock::now() >= deadline) break;
      // Only reached when the runtime ignored MPP_SET_INPUT_TIMEOUT and
      // returned immediately; otherwise the call above already consumed the
      // whole budget and the deadline check exits the loop.
      std::this_thread::sleep_for(std::chrono::microseconds(kPollSleepUs));
    }

    mpp_packet_deinit(&pkt);
    if (!queued && ret != MPP_ERR_BUFFER_FULL) mpp::check(ret, "decode_put_packet");
    return queued;
  }

  void feedEos() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (eos_fed) return;  // idempotent: a second EOS packet confuses the parser
      eos_fed = true;
    }
    // Losing the marker means flush() never terminates, so give it a real
    // budget rather than the caller's usual 20 ms.
    if (!putPacket(nullptr, 0, 0, /*eos=*/true, kEosFeedWaitMs)) {
      std::lock_guard<std::mutex> lock(mu_);
      eos_fed = false;  // it never landed — let the next call try again
    }
  }

  // -------------------------------------------------------------------------
  // Info change: sizing and committing the frame buffers
  // -------------------------------------------------------------------------
  //
  // MPP cannot know the picture geometry until the first sequence header has
  // been parsed, so it decodes nothing until the caller has agreed on a buffer
  // pool. It signals that point by returning a frame with the info-change flag
  // set — a description, not a picture: it carries width/height, the aligned
  // hor_stride/ver_stride the VPU will actually write with, the output format
  // and the required buffer size, and it has no MppBuffer attached. The decode
  // thread is parked until MPP_DEC_SET_INFO_CHANGE_READY, so this must not be
  // slow and must not be skipped.
  //
  // The sequence, and why each step is where it is:
  //   1. read the geometry off the info-change frame;
  //   2. size one slot — mpp_frame_get_buf_size() is what MPP itself wants,
  //      widened to our own computation if that is larger;
  //   3. count the slots: cfg.buffer_count if pinned, else what the stream says
  //      it needs (MPP_DEC_GET_VPUMEM_USED_COUNT: reference frames + reorder)
  //      plus cfg.extra_buffers of headroom for frames in flight downstream;
  //   4. allocate them as rcdl::DmaBufs and commit each into an EXTERNAL group,
  //      which is what stops MPP allocating its own and makes the heap ours;
  //   5. hand the group over with MPP_DEC_SET_EXT_BUF_GROUP;
  //   6. acknowledge with MPP_DEC_SET_INFO_CHANGE_READY, which unblocks decode;
  //   7. release the info-change frame.
  //
  // Any failure in 4-5 falls back to an MPP-internal group (still a dma-buf, so
  // zero-copy downstream still works — we just do not choose the heap or the
  // count) and, failing even that, to no group at all, which leaves MPP on its
  // default pool. The external path is an optimisation; a board where it does
  // not work must still decode.
  void handleInfoChange(MppFrame frame) {
    const int w = static_cast<int>(mpp_frame_get_width(frame));
    const int h = static_cast<int>(mpp_frame_get_height(frame));
    const int hs = static_cast<int>(mpp_frame_get_hor_stride(frame));
    const int vs = static_cast<int>(mpp_frame_get_ver_stride(frame));
    const MppFrameFormat fmt = mpp_frame_get_fmt(frame);

    // MPP fills buf_size on the info-change frame precisely so the caller can
    // size an external group; our own arithmetic is the floor under it, for the
    // formats or runtimes where it comes back as 0.
    std::size_t bytes = mpp_frame_get_buf_size(frame);
    const std::size_t computed = mpp::frameBufferBytes(fmt, hs, vs);
    if (computed > bytes) bytes = computed;
    RCDL_REQUIRE(bytes > 0, "VideoDecoder: info change reported a zero frame buffer size");

    int count = cfg.buffer_count;
    if (count <= 0) {
      RK_S32 used = 0;
      // Best-effort: this is the slot count the parser reserved for references
      // and reordering. Not every codec/runtime answers, hence the default.
      if (mpi->control(ctx, MPP_DEC_GET_VPUMEM_USED_COUNT, &used) != MPP_OK || used <= 0) {
        used = kDefaultSlots;
      }
      count = used + std::max(0, cfg.extra_buffers);
    }
    count = std::min(std::max(count, kMinSlots), kMaxSlots);

    // A repeat info change with the SAME geometry — which is what a seek or a
    // reset() produces — must reuse the pool it already has. Rebuilding it every
    // time would allocate a fresh set of frame buffers per seek and, because the
    // old set cannot be freed (below), grow without bound.
    bool same_geometry = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      same_geometry = info_ready && width == w && height == h && hor_stride == hs &&
                      ver_stride == vs;
    }
    // Reuse the existing pool when the geometry has not actually changed —
    // MPP re-raises an info change after every reset(), i.e. after every seek.
    // The `slot_bytes`/`count` test only means anything for a pool WE sized, so
    // an internal pool (either by config or because the external path fell back
    // on this board) qualifies on geometry alone. Leaving it out was retiring
    // one whole MppBufferGroup per seek into `retired_`, which is not released
    // until ~Impl — unbounded growth in exactly the long-running case.
    const bool pool_fits = pool_ && (!pool_->external ||
                                     (pool_->slot_bytes >= bytes && pool_->count >= count));
    if (same_geometry && pool_fits) {
      mpp::check(mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr),
                 "MPP_DEC_SET_INFO_CHANGE_READY");
      publishGeometry(w, h, hs, vs, true);
      return;
    }
    // The geometry really changed. The old buffers cannot be freed here — the
    // caller may still be holding frames backed by them — so the whole pool is
    // retired and released in ~Impl, after mpp_destroy().
    if (pool_) retired_.push_back(std::move(pool_));

    bool ext = false;
    if (cfg.external_buffers) {
      // Start from the type that worked last time — on a re-negotiation there is
      // no point rediscovering that this board rejects ION.
      const MppBufferType first =
          retired_.empty() || !retired_.back()->external ? kBufferTypes[0]
                                                         : retired_.back()->type;
      ext = buildExternalPool(first, count, bytes);
    }
    if (!ext) buildInternalPool();

    mpp::check(mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr),
               "MPP_DEC_SET_INFO_CHANGE_READY");
    publishGeometry(w, h, hs, vs, ext);
  }

  void publishGeometry(int w, int h, int hs, int vs, bool ext) noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    width = w;
    height = h;
    hor_stride = hs;
    ver_stride = vs;
    info_ready = true;
    external = ext;
  }

  // Try every buffer type in turn; each attempt is all-or-nothing (group +
  // every commit + handover), because a half-committed group is worse than no
  // group at all.
  bool buildExternalPool(MppBufferType first, int count, std::size_t bytes) {
    if (tryExternalPool(first, count, bytes)) return true;
    for (MppBufferType t : kBufferTypes) {
      if (t == first) continue;
      if (tryExternalPool(t, count, bytes)) return true;
    }
    return false;
  }

  bool tryExternalPool(MppBufferType type, int count, std::size_t bytes) {
    auto pool = std::unique_ptr<FramePool>(new FramePool());
    pool->slot_bytes = bytes;
    pool->count = count;
    pool->external = true;
    pool->type = type;

    if (mpp_buffer_group_get_external(&pool->group, type) != MPP_OK ||
        pool->group == nullptr) {
      pool->group = nullptr;
      return false;
    }
    try {
      pool->slots.reserve(static_cast<std::size_t>(count));
      for (int i = 0; i < count; ++i) {
        // The system dma-heap is the one every unit on this SoC can import:
        // NPU, RGA and VPU all sit behind an IOMMU, so physical contiguity is
        // not required and the cached heap keeps CPU inspection affordable.
        DmaBuf b = DmaBuf::alloc(bytes, DmaBuf::Heap::System);
        MppBufferInfo info;
        info.type = type;
        info.size = bytes;
        info.ptr = nullptr;  // fd-only import; MPP mmaps on demand
        info.hnd = nullptr;
        info.fd = b.fd();
        info.index = i;
        // commit == "this slot exists and is unused". MPP dup()s the fd, so our
        // DmaBuf keeps its own descriptor and both sides can close independently.
        if (mpp_buffer_commit(pool->group, &info) != MPP_OK) return releaseAndFail(pool.get());
        pool->slots.push_back(std::move(b));
      }
      // Handing the group over is what switches the decoder off its internal
      // allocator: from here MPP draws every output frame from our slots.
      if (mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, pool->group) != MPP_OK) {
        return releaseAndFail(pool.get());
      }
    } catch (const std::exception&) {
      // DmaBuf::alloc throws (no dma-heap, EACCES, out of memory). Nothing has
      // been handed to the decoder yet, so unwinding here is clean.
      return releaseAndFail(pool.get());
    }
    pool_ = std::move(pool);
    return true;
  }

  static bool releaseAndFail(FramePool* p) noexcept {
    releasePool(p);
    return false;
  }

  // Fallback: let MPP allocate, but still through a group we hold, so the
  // frames are dma-bufs and the zero-copy path downstream is unchanged.
  void buildInternalPool() {
    auto pool = std::unique_ptr<FramePool>(new FramePool());
    pool->external = false;
    for (MppBufferType t : kBufferTypes) {
      if (mpp_buffer_group_get_internal(&pool->group, t) == MPP_OK && pool->group != nullptr) {
        pool->type = t;
        // No limit_config: MPP grows this pool as the stream needs it, and a
        // ceiling picked here is a deadlock waiting for a stream that needs one
        // more reference frame than we guessed.
        if (mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, pool->group) == MPP_OK) {
          pool_ = std::move(pool);
          return;
        }
        releasePool(pool.get());
      }
      pool->group = nullptr;
    }
    // Last resort: acknowledge the info change with no group at all. MPP then
    // uses its own default pool — least control, still correct.
    pool_.reset();
  }

  // -------------------------------------------------------------------------
  // Drain side
  // -------------------------------------------------------------------------

  bool recv(VideoFrame& out, int timeout_ms) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (eos_seen) return false;  // the stream is over; nothing more will come
    }
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    // Set once for the whole call, not per iteration: an info change or a
    // dropped frame can push a single receive() past its nominal budget, which
    // is a far better trade than a control() ioctl per loop turn.
    setOutputTimeout(timeout_ms);

    for (;;) {
      MppFrame frame = nullptr;
      const MPP_RET ret = mpi->decode_get_frame(ctx, &frame);
      // A non-OK return WITH a frame is a real failure. A non-OK return with no
      // frame is not: this runtime answers a poll that has nothing ready with
      // MPP_NOK as readily as with MPP_ERR_TIMEOUT, and which one you get
      // depends on where the decoder happens to be — so treating MPP_NOK as
      // fatal works at 1080p and then kills a 4K stream, where the decoder is
      // more often mid-picture when we ask. The frame pointer is the signal;
      // the deadline below is what ends the wait.
      if (ret != MPP_OK && frame != nullptr) {
        mpp_frame_deinit(&frame);
        mpp::check(ret, "decode_get_frame");
      }

      if (frame == nullptr) {
        if (Clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(kPollSleepUs));
        continue;
      }

      if (mpp_frame_get_info_change(frame) != 0) {
        // The info-change frame is a description, never a picture; it goes back
        // to MPP whether the renegotiation succeeded or threw.
        try {
          handleInfoChange(frame);
        } catch (...) {
          mpp_frame_deinit(&frame);
          throw;
        }
        mpp_frame_deinit(&frame);
        if (Clock::now() >= deadline) return false;
        continue;  // the first real picture follows once the decoder restarts
      }

      const bool eos = mpp_frame_get_eos(frame) != 0;
      if (eos) {
        std::lock_guard<std::mutex> lock(mu_);
        eos_seen = true;
      }

      // errinfo is the hardware's own verdict on this picture (missing
      // reference, bitstream error, unsupported feature); discard is the
      // parser asking for it to be thrown away. Either way the pixels are not
      // the picture the stream describes, so hand back the slot rather than
      // pass a corrupt frame down a detection pipeline.
      const bool bad = mpp_frame_get_errinfo(frame) != 0 || mpp_frame_get_discard(frame) != 0;
      MppBuffer buf = mpp_frame_get_buffer(frame);
      if (bad || buf == nullptr) {
        // A buffer-less frame is normal: it is how MPP delivers the end-of-
        // stream marker once the reorder queue has drained.
        mpp_frame_deinit(&frame);
        if (eos || Clock::now() >= deadline) return false;
        continue;
      }

      ImageView view = mpp::viewOfFrame(frame);
      const std::uint64_t pts = static_cast<std::uint64_t>(mpp_frame_get_pts(frame));
      std::uint64_t index = 0;
      {
        std::lock_guard<std::mutex> lock(mu_);
        index = frames++;
      }
      // Ownership of the MppFrame moves into the VideoFrame here — no pixel
      // copy anywhere on this path; the caller gets the fd the VPU wrote into.
      out = VideoFrame::adopt(frame, view, pts, index);
      return true;
    }
  }
};

// ===========================================================================
// VideoDecoder — public surface
// ===========================================================================

VideoDecoder::VideoDecoder(const VideoDecConfig& cfg)
    : impl_(std::unique_ptr<Impl>(new Impl(cfg))) {}

VideoDecoder::~VideoDecoder() = default;

bool VideoDecoder::feed(const std::uint8_t* data, std::size_t size, std::uint64_t pts_us,
                        int timeout_ms) {
  RCDL_REQUIRE(data != nullptr && size > 0, "VideoDecoder::feed given empty input");
  return impl_->putPacket(data, size, pts_us, /*eos=*/false, std::max(0, timeout_ms));
}

bool VideoDecoder::receive(VideoFrame& out, int timeout_ms) {
  return impl_->recv(out, std::max(0, timeout_ms));
}

bool VideoDecoder::decode(const std::uint8_t* data, std::size_t size, VideoFrame& out) {
  if (!feed(data, size)) return false;
  return impl_->recv(out, kDecodeWaitMs);
}

bool VideoDecoder::flush(VideoFrame& out) {
  feedEndOfStream();
  return impl_->recv(out, kFlushWaitMs);
}

void VideoDecoder::feedEndOfStream() { impl_->feedEos(); }

void VideoDecoder::reset() {
  // mpi->reset() blocks until the decoder's threads have dropped everything
  // queued, so it is called without the lock; the flags are only touched
  // afterwards, when there is nothing left in flight to disagree with them.
  mpp::check(impl_->mpi->reset(impl_->ctx), "MppApi::reset");
  std::lock_guard<std::mutex> lock(impl_->mu_);
  impl_->eos_fed = false;
  impl_->eos_seen = false;
}

int VideoDecoder::width() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mu_);
  return impl_->width;
}

int VideoDecoder::height() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mu_);
  return impl_->height;
}

int VideoDecoder::widthStride() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mu_);
  return impl_->hor_stride;
}

int VideoDecoder::heightStride() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mu_);
  return impl_->ver_stride;
}

VideoCodec VideoDecoder::codec() const noexcept { return impl_->cfg.codec; }

std::uint64_t VideoDecoder::framesDecoded() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mu_);
  return impl_->frames;
}

bool VideoDecoder::endOfStream() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mu_);
  return impl_->eos_seen;
}

bool VideoDecoder::usingExternalBuffers() const noexcept {
  std::lock_guard<std::mutex> lock(impl_->mu_);
  return impl_->external;
}

#else  // !RCDL_HAVE_MPP

// No MPP in this build: the class still exists and still links, but every entry
// point says so instead of dereferencing a null Impl.
struct VideoDecoder::Impl {};

VideoDecoder::VideoDecoder(const VideoDecConfig&) { throw Error(-1, mpp::kUnavailable); }
VideoDecoder::~VideoDecoder() = default;

bool VideoDecoder::feed(const std::uint8_t*, std::size_t, std::uint64_t, int) {
  throw Error(-1, mpp::kUnavailable);
}
bool VideoDecoder::receive(VideoFrame&, int) { throw Error(-1, mpp::kUnavailable); }
bool VideoDecoder::decode(const std::uint8_t*, std::size_t, VideoFrame&) {
  throw Error(-1, mpp::kUnavailable);
}
bool VideoDecoder::flush(VideoFrame&) { throw Error(-1, mpp::kUnavailable); }
void VideoDecoder::feedEndOfStream() { throw Error(-1, mpp::kUnavailable); }
void VideoDecoder::reset() { throw Error(-1, mpp::kUnavailable); }
int VideoDecoder::width() const noexcept { return 0; }
int VideoDecoder::height() const noexcept { return 0; }
int VideoDecoder::widthStride() const noexcept { return 0; }
int VideoDecoder::heightStride() const noexcept { return 0; }
VideoCodec VideoDecoder::codec() const noexcept { return VideoCodec::H264; }
std::uint64_t VideoDecoder::framesDecoded() const noexcept { return 0; }
bool VideoDecoder::endOfStream() const noexcept { return false; }
bool VideoDecoder::usingExternalBuffers() const noexcept { return false; }

#endif  // RCDL_HAVE_MPP

}  // namespace rcdl
