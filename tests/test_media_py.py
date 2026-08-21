"""Board tests for the media (VPU codec) layer through the Python bindings.

These need the compiled module AND the hardware: MPP talks to the VPU driver, so
they only run on the board. Everything is guarded — the file skips cleanly when
the module is missing, when the codec bindings are not compiled in, and when the
sample streams are not staged.

    RCDL_STREAMS=/path/to/streams \\
        PYTHONPATH=build:python pytest -s tests/test_media_py.py

`RCDL_STREAMS` names a DIRECTORY holding raw elementary streams (the C++
examples take the same files as an argv path):

    h264_1080p.h264   h265_1080p.h265   h265_4k.h265

Raw elementary streams only — a container (.mkv/.mp4) would need a demuxer,
which RCDL does not have.

The binding surface these tests assume
--------------------------------------
    rcdl.VideoDecoder(codec="h264", format="nv12", external_buffers=True,
                      buffer_count=0, extra_buffers=4)
        .feed(data: bytes, pts_us=0, timeout_ms=20) -> bool   # False = back-pressure
        .receive(timeout_ms=0) -> VideoFrame | None
        .flush() -> VideoFrame | None
        .feed_end_of_stream() -> None
        .width .height .width_stride .height_stride            -> int
        .frames_decoded -> int   .using_external_buffers -> bool
        .codec -> str            .end_of_stream -> bool

    rcdl.VideoFrame                       (returned by the decoders, not constructed)
        .width .height .width_stride .height_stride .fd .index .pts_us -> int
        .format -> str ("nv12")           .valid -> bool
        .to_numpy() -> uint8 array, display size, padding removed:
                       (h*3//2, w) for NV12, (h, w, c) for packed formats
        .letterbox(dst_w, dst_h, dst_fmt="rgb888", pad=114, backend="auto",
                   studio_range=True) -> (image, lb_tuple, backend_name)
                       same return shape as rcdl.letterbox(), but the source is
                       the frame's dma-buf fd (the zero-copy path)
        .release() -> None                # back to the decoder pool, now

    rcdl.VideoEncoder(codec="h264", width=..., height=..., format="nv12", fps=30,
                      bitrate_kbps=4000, gop=0, rc="cbr", qp=0, profile=100)
        .feed(image, w, h, fmt="nv12", pts_us=0, timeout_ms=20) -> bool
        .feed_frame(frame: VideoFrame, pts_us=0, timeout_ms=20) -> bool  # zero copy
        .receive(timeout_ms=0) -> bytes | None
        .flush() -> bytes | None
        .feed_end_of_stream() -> None
        .extra_data -> bytes   .width .height .frames_encoded -> int  .codec -> str

    rcdl.JpegEncoder(width, height, format="nv12", quality=80)
        .encode(image, fmt="nv12") -> bytes
        .encode_frame(frame: VideoFrame) -> bytes                       # zero copy
    rcdl.JpegDecoder(format="nv12")
        .decode(data: bytes) -> VideoFrame | None

Format and codec names are the lower-case strings the rest of the Python layer
already uses ("nv12", "rgb888", "h264", "rga"/"cpu"/"auto").
"""

import os

import numpy as np
import pytest

rcdl = pytest.importorskip("rcdl", reason="build the module on the board first")

STREAMS = os.environ.get("RCDL_STREAMS", "")

# (file name, codec, expected resolution or None when it should not be pinned)
SAMPLE_STREAMS = [
    ("h264_1080p.h264", "h264", (1920, 1080)),
    ("h265_1080p.h265", "h265", (1920, 1080)),
    ("h265_4k.h265", "h265", (3840, 2160)),
]

_CHUNK = 256 * 1024


# --------------------------------------------------------------------------- #
# guards                                                                       #
# --------------------------------------------------------------------------- #
def require_bindings(*names):
    missing = [n for n in names if not hasattr(rcdl, n)]
    if missing:
        pytest.skip("compiled module predates the media bindings: " + ", ".join(missing))


def require_frame_method(name):
    """VideoFrame methods cannot be probed without a frame, so probe the class."""
    require_bindings("VideoFrame")
    if not hasattr(rcdl.VideoFrame, name):
        pytest.skip(f"VideoFrame.{name}() not exposed by this build")


def require_stream(name):
    if not STREAMS:
        pytest.skip("set RCDL_STREAMS to the directory holding the sample elementary streams")
    path = os.path.join(STREAMS, name)
    if not os.path.isfile(path):
        pytest.skip(f"stream not staged: {path}")
    return path


def read_stream(name):
    with open(require_stream(name), "rb") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# decode pump                                                                  #
# --------------------------------------------------------------------------- #
def decode(payload, codec, max_frames=0, collect=None, **cfg):
    """Feed `payload` to a VideoDecoder and return (decoder, [collect(frame), ...]).

    The cadence is the one every hardware decoder needs and the C++ examples use:
    take a ready frame FIRST, feed only when there is none, and when the input
    queue reports back-pressure drain a frame before retrying. `collect` is
    called with each frame while it is still alive; the frame is released right
    after, because every frame held is a buffer the decoder cannot reuse.
    """
    dec = rcdl.VideoDecoder(codec=codec, **cfg)
    out = []
    off, total, eos = 0, len(payload), False
    while not max_frames or len(out) < max_frames:
        frame = dec.receive(0)
        if frame is None:
            if off < total:
                end = min(off + _CHUNK, total)
                if dec.feed(payload[off:end]):
                    off = end
                else:
                    frame = dec.receive(5)  # back-pressure: drain, then retry
            elif not eos:
                dec.feed_end_of_stream()
                eos = True
            else:
                frame = dec.flush()  # reorder tail
                if frame is None:
                    break
        if frame is None:
            continue
        out.append(frame_info(frame) if collect is None else collect(frame))
        frame.release()
    return dec, out


def frame_info(frame):
    return {
        "width": frame.width,
        "height": frame.height,
        "width_stride": frame.width_stride,
        "height_stride": frame.height_stride,
        "fd": frame.fd,
        "format": frame.format,
        "index": frame.index,
    }


def luma_of(frame):
    """The frame's Y plane as an (h, w) uint8 array (padding already dropped)."""
    nv12 = frame.to_numpy()
    return np.ascontiguousarray(nv12[: frame.height])


def psnr(a, b):
    """Luma-only PSNR in dB. Identical planes report a large sentinel."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 0 else 10.0 * np.log10(255.0 * 255.0 / mse)


# --------------------------------------------------------------------------- #
# decode                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,codec,size", SAMPLE_STREAMS,
                         ids=[s[0] for s in SAMPLE_STREAMS])
def test_decodes_a_real_elementary_stream(name, codec, size):
    """Real frames come back, at the right size, each in its own dma-buf."""
    require_bindings("VideoDecoder")
    payload = read_stream(name)
    dec, frames = decode(payload, codec, max_frames=30)

    assert frames, f"the VPU decoded no frames from {name}"
    print(f"\n{name}: {len(frames)} frames, {dec.width}x{dec.height}, "
          f"stride {dec.width_stride}x{dec.height_stride}, "
          f"external buffers: {dec.using_external_buffers}")

    w, h = size
    assert (dec.width, dec.height) == (w, h), f"expected {w}x{h}, got {dec.width}x{dec.height}"
    # The VPU writes rows padded up to its own alignment — never below the
    # display size, and reading at `width` instead of the stride shears the image.
    assert dec.width_stride >= dec.width and dec.height_stride >= dec.height
    assert dec.frames_decoded >= len(frames)

    for f in frames:
        assert (f["width"], f["height"]) == (w, h)
        assert f["format"] == "nv12"
        # The zero-copy claim: an fd is what RGA and the NPU import.
        assert f["fd"] >= 0, "decoded frame carries no dma-buf fd"


def test_decoded_pixels_are_a_picture():
    """A flat or constant luma plane means the buffer was never written."""
    require_bindings("VideoDecoder")
    require_frame_method("to_numpy")
    payload = read_stream(SAMPLE_STREAMS[0][0])
    _, lumas = decode(payload, SAMPLE_STREAMS[0][1], max_frames=3, collect=luma_of)
    assert lumas
    for i, y in enumerate(lumas):
        print(f"frame {i}: luma mean {y.mean():.1f} std {y.std():.1f}")
        assert y.std() > 1.0, "decoded luma is flat — not real pixels"
        assert 5 < y.mean() < 250


def test_frames_come_back_in_display_order():
    require_bindings("VideoDecoder")
    payload = read_stream(SAMPLE_STREAMS[0][0])
    _, frames = decode(payload, SAMPLE_STREAMS[0][1], max_frames=20)
    idx = [f["index"] for f in frames]
    assert idx == sorted(idx), f"frames arrived out of order: {idx}"
    assert idx == list(range(len(idx))), f"display index is not dense: {idx}"


def test_internal_buffer_group_also_yields_fds():
    """external_buffers=False falls back to MPP's own pool. Both give a dma-buf
    fd — external only buys control over the heap and the pool size."""
    require_bindings("VideoDecoder")
    payload = read_stream(SAMPLE_STREAMS[0][0])
    dec, frames = decode(payload, SAMPLE_STREAMS[0][1], max_frames=5, external_buffers=False)
    assert frames
    assert not dec.using_external_buffers
    assert all(f["fd"] >= 0 for f in frames)


# --------------------------------------------------------------------------- #
# decode -> encode -> decode                                                   #
# --------------------------------------------------------------------------- #
def roundtrip(name, codec, enc_codec="h264", frames=12, bitrate_kbps=8000):
    """Decode, re-encode from the decoded dma-buf, decode again.

    Returns (reference lumas, encoded stream, decoded-again lumas).
    """
    payload = read_stream(name)
    state = {"enc": None}
    packets = []

    def drain(timeout_ms=0):
        while True:
            pkt = state["enc"].receive(timeout_ms)
            if pkt is None:
                return
            packets.append(pkt)
            timeout_ms = 0

    def collect(frame):
        if state["enc"] is None:
            state["enc"] = rcdl.VideoEncoder(codec=enc_codec, width=frame.width,
                                             height=frame.height, format=frame.format,
                                             bitrate_kbps=bitrate_kbps)
        # The zero-copy hand-off: the encoder reads the decoder's buffer in place.
        for _ in range(200):
            if state["enc"].feed_frame(frame, frame.pts_us):
                break
            drain(5)  # input queue full: drain packets and retry
        else:
            pytest.fail("the encoder never accepted a frame")
        drain()
        return luma_of(frame)

    _, reference = decode(payload, codec, max_frames=frames, collect=collect)
    assert reference, "nothing decoded"
    while True:
        pkt = state["enc"].flush()
        if pkt is None:
            break
        packets.append(pkt)

    stream = b"".join(packets)
    _, again = decode(stream, enc_codec, max_frames=len(reference), collect=luma_of)
    return reference, stream, again


@pytest.mark.parametrize("enc_codec", ["h264", "h265"])
def test_decode_encode_decode_luma_psnr(enc_codec):
    """The M2 quality check: what survives a VPU round trip.

    PSNR is measured on the LUMA PLANE ONLY. At a generous bitrate on 1080p a
    single re-encode should stay well above 30 dB; anything near 10 dB means the
    frames were shifted, mis-strided or paired up wrong, not merely compressed.
    """
    require_bindings("VideoDecoder", "VideoEncoder")
    require_frame_method("to_numpy")
    name, codec, _ = SAMPLE_STREAMS[0]
    reference, stream, again = roundtrip(name, codec, enc_codec=enc_codec)

    assert stream, "the encoder produced no bytes"
    assert again, "the re-encoded stream did not decode"
    n = min(len(reference), len(again))
    values = [psnr(reference[i], again[i]) for i in range(n)]
    print(f"\n{codec} -> {enc_codec} round trip over {n} frames: "
          f"mean {np.mean(values):.2f} dB, worst {min(values):.2f} dB, "
          f"{len(stream)} bytes encoded")
    assert min(values) > 30.0, f"worst luma PSNR {min(values):.2f} dB"


def test_encoded_stream_starts_with_a_codec_header():
    """Concatenated packets must be a playable elementary stream: a start code,
    then the parameter set (SPS for H.264, VPS for H.265)."""
    require_bindings("VideoDecoder", "VideoEncoder")
    name, codec, _ = SAMPLE_STREAMS[0]
    _, stream, _ = roundtrip(name, codec, frames=4)
    assert stream[:4] == b"\x00\x00\x00\x01", stream[:8].hex()
    assert stream[4] == 0x67, hex(stream[4])  # H.264 SPS


# --------------------------------------------------------------------------- #
# JPEG                                                                         #
# --------------------------------------------------------------------------- #
def synthetic_nv12(w=640, h=480):
    """A deterministic NV12 picture with structure at several scales (a noisy
    source would make every codec look equally bad and prove nothing)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y = 128 + 100 * np.sin(xx / 40.0) * np.cos(yy / 55.0)
    u = 128 + 60 * np.sin(xx / 90.0)
    v = 128 + 60 * np.cos(yy / 70.0)
    out = np.empty((h * 3 // 2, w), dtype=np.uint8)
    out[:h] = np.clip(y, 0, 255).astype(np.uint8)
    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = np.clip(u[::2, ::2], 0, 255).astype(np.uint8)
    uv[:, 1::2] = np.clip(v[::2, ::2], 0, 255).astype(np.uint8)
    out[h:] = uv
    return out


def test_jpeg_encode_decode_roundtrip():
    require_bindings("JpegEncoder", "JpegDecoder")
    require_frame_method("to_numpy")
    w, h = 640, 480  # both 16-aligned, so the VPU pads nothing
    src = synthetic_nv12(w, h)

    enc = rcdl.JpegEncoder(w, h, format="nv12", quality=90)
    jpeg = enc.encode(src)
    assert isinstance(jpeg, (bytes, bytearray)) and len(jpeg) > 1000
    assert jpeg[:2] == b"\xff\xd8", "no JPEG SOI marker"
    assert jpeg[-2:] == b"\xff\xd9", "no JPEG EOI marker"

    frame = rcdl.JpegDecoder().decode(bytes(jpeg))
    assert frame is not None, "the VPU decoded nothing"
    try:
        # The hardware may align the decoded size up; it must not shrink it.
        assert frame.width >= w and frame.width - w < 16
        assert frame.height >= h and frame.height - h < 16
        assert frame.fd >= 0, "decoded JPEG carries no dma-buf fd"
        got = luma_of(frame)[:h, :w]
        db = psnr(src[:h], got)
        print(f"\nJPEG q=90 {w}x{h}: {len(jpeg)} bytes "
              f"({8.0 * len(jpeg) / (w * h):.2f} bpp), luma PSNR {db:.2f} dB")
        assert db > 35.0, f"luma PSNR {db:.2f} dB after a q=90 round trip"
    finally:
        frame.release()


def test_jpeg_quality_trades_size_for_fidelity():
    """A sanity check on the quality knob: lower q must produce a smaller file.
    A binding that silently drops `quality` passes every other test here."""
    require_bindings("JpegEncoder")
    w, h = 640, 480
    src = synthetic_nv12(w, h)
    sizes = {q: len(rcdl.JpegEncoder(w, h, format="nv12", quality=q).encode(src))
             for q in (30, 60, 90)}
    print(f"\nJPEG bytes by quality: {sizes}")
    assert sizes[30] < sizes[60] < sizes[90]


def test_jpeg_decode_matches_libjpeg():
    """Hardware decode against the software one, on a real photo. Chroma
    upsampling and the YUV matrix differ between implementations, so this asks
    for the same picture, not the same bytes."""
    require_bindings("JpegDecoder")
    require_frame_method("to_numpy")
    cv2 = pytest.importorskip("cv2", reason="OpenCV needed for the software decode")
    import board_models as bm

    path = bm.require_image("bus.jpg")
    with open(path, "rb") as f:
        raw = f.read()

    cpu = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    frame = rcdl.JpegDecoder().decode(raw)
    assert frame is not None, "the VPU decoded nothing (progressive JPEG?)"
    try:
        h, w = cpu.shape[:2]
        assert frame.width >= w and frame.height >= h
        hw = luma_of(frame)[:h, :w]
        diff = np.abs(hw.astype(np.int16) - cpu.astype(np.int16))
        print(f"\nVPU vs libjpeg on {os.path.basename(path)} ({w}x{h}): "
              f"mean {diff.mean():.2f} max {diff.max()}")
        assert diff.mean() < 8.0
    finally:
        frame.release()


# --------------------------------------------------------------------------- #
# the pipeline hand-off: a decoded frame straight into the preproc layer        #
# --------------------------------------------------------------------------- #
def _letterbox_paths(frame, size=640):
    """The same frame letterboxed three ways: RGA from the dma-buf fd, RGA from
    a host copy of the same pixels, and the CPU reference."""
    nv12 = frame.to_numpy()
    hw_fd, lb_fd, used_fd = frame.letterbox(size, size, dst_fmt="rgb888", backend="rga")
    hw_host, lb_host, used_host = rcdl.letterbox(nv12, size, size, src_fmt="nv12",
                                                 dst_fmt="rgb888", backend="rga")
    cpu, lb_cpu, used_cpu = rcdl.letterbox(nv12, size, size, src_fmt="nv12",
                                           dst_fmt="rgb888", backend="cpu")
    assert (used_fd, used_host, used_cpu) == ("rga", "rga", "cpu")
    return (hw_fd, lb_fd), (hw_host, lb_host), (cpu, lb_cpu)


def _first_frame_paths():
    if not rcdl.rga_available():
        pytest.skip("RGA not available on this board/build")
    name, codec, _ = SAMPLE_STREAMS[0]
    _, out = decode(read_stream(name), codec, max_frames=1, collect=_letterbox_paths)
    assert out, "nothing decoded"
    return out[0]


def test_frame_letterbox_from_fd_matches_the_host_copy():
    """The zero-copy path must see exactly the pixels the CPU would read out of
    the same buffer. Same hardware, same filter, so any real difference here is
    a stride or an offset bug — not resampling."""
    require_bindings("VideoDecoder")
    require_frame_method("letterbox")
    (fd_img, fd_lb), (host_img, host_lb), _ = _first_frame_paths()
    assert fd_img.shape == host_img.shape == (640, 640, 3)
    assert fd_lb == host_lb, "the fd and host paths disagree on the letterbox geometry"
    d = np.abs(fd_img.astype(np.int16) - host_img.astype(np.int16))
    print(f"\nRGA from fd vs RGA from a host copy: max {d.max()} mean {d.mean():.3f}")
    assert d.mean() < 1.0, "the dma-buf path and the host path saw different pixels"


def test_frame_letterbox_agrees_with_the_cpu_fallback():
    """RGA and the CPU fallback must produce the same picture from a decoded
    NV12 frame. Not the same bytes: RGA pre-filters when it shrinks while the
    CPU path point-samples bilinearly (pinned as a documented property in
    test_letterbox.py), so the bound is on the mean, not the max."""
    require_bindings("VideoDecoder")
    require_frame_method("letterbox")
    (fd_img, fd_lb), _, (cpu_img, cpu_lb) = _first_frame_paths()
    assert fd_img.shape == cpu_img.shape
    # Geometry must match exactly — that is what post-processing inverts.
    assert fd_lb[0] == pytest.approx(cpu_lb[0], rel=1e-6)
    assert abs(fd_lb[1] - cpu_lb[1]) <= 1 and abs(fd_lb[2] - cpu_lb[2]) <= 1
    d = np.abs(fd_img.astype(np.int16) - cpu_img.astype(np.int16))
    print(f"RGA vs CPU on a decoded frame: max {d.max()} mean {d.mean():.2f} "
          f"| means {fd_img.mean():.1f} vs {cpu_img.mean():.1f}")
    assert d.mean() < 12.0, "RGA and the CPU path have diverged beyond filter choice"
    assert abs(float(fd_img.mean()) - float(cpu_img.mean())) < 3.0
