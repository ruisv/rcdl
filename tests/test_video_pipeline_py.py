"""Board tests for AsyncVideoDetectionPipeline — compressed video in, detections out.

These need the board: MPP decodes the stream, RGA letterboxes it and the NPU
runs the model. They skip cleanly without the module, without the codec
bindings, or without the detection model.

    PYTHONPATH=build:python pytest -s tests/test_video_pipeline_py.py

WHY THE STREAM IS SYNTHESISED HERE instead of read from disk. A detection test
needs a clip with things in it, and the sample elementary streams staged for the
codec tests contain no COCO object at all — every frame of them decodes to zero
detections, which cannot tell a working pipeline from a broken one. So the test
encodes its own stream, on the same VPU, from the image whose contents the rest
of the suite already pins: `bus.jpg`, one bus and four people, panned a few
pixels per frame. The right answer is then knowable frame by frame.

The binding surface these tests assume
--------------------------------------
    engine.video_detector(codec="h264", **detector_kwargs) -> AsyncVideoDetectionPipeline
        .submit(data: bytes, timeout_ms=20) -> bool   # False = back-pressure OR finished
        .next() -> [Detection] | None                 # blocks; decode order
        .try_next() -> [Detection] | None             # never blocks
        .finish() -> None
        .finished -> bool          .pts_us -> int      .frame_index -> int
        .width .height .frames_decoded -> int          .workers -> int
        .letterbox -> (scale, pad_x, pad_y)            .head -> str
        .profile -> (decode_ms, preproc_ms, infer_ms, postproc_ms, frames)
"""

import numpy as np
import pytest

import board_models as bm

FRAMES = 16          # frames encoded into the test stream
PAN_STEP = 8         # pixels the scene moves per frame
CHUNK = 64 * 1024    # feed granularity; any size is legal (MPP splits the stream)


@pytest.fixture(scope="module")
def rcdl():
    return pytest.importorskip("rcdl", reason="rcdl module not importable (build on the board)")


def need(mod, *names):
    for n in names:
        if not hasattr(mod, n):
            pytest.skip(f"{n} is not in this build")
    if not mod.mpp_available():
        pytest.skip("this build has no MPP (VPU codecs)")


def bgr_to_nv12(img):
    """BGR HxWx3 -> NV12 (H*3/2)xW, the encoder's native input.

    Via OpenCV's I420 conversion and then an interleave, rather than a hand-
    rolled colour matrix: the point of the test is the pipeline, so the pixels
    should come from a conversion someone else already got right.
    """
    import cv2
    h, w = img.shape[:2]
    i420 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    y = i420[:h]
    u = i420[h:h + h // 4].reshape(h // 2, w // 2)
    v = i420[h + h // 4:].reshape(h // 2, w // 2)
    nv12 = np.empty((h * 3 // 2, w), np.uint8)
    nv12[:h] = y
    uv = nv12[h:].reshape(h // 2, w)
    uv[:, 0::2] = u   # NV12 is U,V interleaved (NV21 is the other order)
    uv[:, 1::2] = v
    return nv12


def panning_frames(img, frames=FRAMES, step=PAN_STEP):
    """`img` on a 16-aligned canvas, moving `step` px left per frame.

    The canvas is padded rather than resized so the pixels the detector sees are
    the pixels the rest of the suite pins, and both dimensions are multiples of
    16 because that is what the encoder's stride wants.
    """
    import cv2
    h, w = img.shape[:2]
    cw, ch = (w + 15) // 16 * 16, (h + 15) // 16 * 16
    canvas = cv2.copyMakeBorder(img, 0, ch - h, 0, cw - w, cv2.BORDER_REPLICATE)
    for i in range(frames):
        m = np.float32([[1, 0, -i * step], [0, 1, 0]])
        yield cv2.warpAffine(canvas, m, (cw, ch), borderMode=cv2.BORDER_REPLICATE)


@pytest.fixture(scope="module")
def stream(rcdl):
    """A real H.264 elementary stream of the panning bus scene, encoded on the VPU."""
    need(rcdl, "VideoEncoder")
    img = bm.load_bgr("bus.jpg")
    frames = list(panning_frames(img))
    h, w = frames[0].shape[:2]
    # A high bitrate on purpose: this stream is a test fixture, and compression
    # artefacts would show up as detection differences that have nothing to do
    # with the pipeline under test.
    enc = rcdl.VideoEncoder(width=w, height=h, codec="h264", format="nv12", fps=30,
                            bitrate_kbps=20000, gop=FRAMES)
    packets = []
    for i, f in enumerate(frames):
        nv12 = bgr_to_nv12(f)
        assert enc.feed(nv12, w, h, "nv12", i * 33333, 200), "the encoder refused a frame"
        while (p := enc.receive(0)) is not None:
            packets.append(p)
    while (p := enc.flush()) is not None:
        packets.append(p)
    data = b"".join(packets)
    if not data:
        pytest.skip("the VPU encoder produced no packets")
    return data, w, h


def sync_reference(rcdl, engine, data, max_frames=FRAMES):
    """The same stream through the synchronous path: decode a frame, detect, release.

    This is the oracle. It shares no code with the async pipeline beyond the
    decoder and the detector themselves, so agreeing with it means the threading,
    the frame hand-off and the reordering are all right.
    """
    dec = rcdl.VideoDecoder(codec="h264")
    pipe = engine.detector()
    out, fed = [], 0
    while len(out) < max_frames:
        frame = dec.receive(5)
        if frame is None and fed >= len(data):
            frame = dec.flush()
        if frame is None:
            if fed >= len(data):
                break
            n = min(CHUNK, len(data) - fed)
            if dec.feed(data[fed:fed + n], 0, 20):
                fed += n
            continue
        out.append(pipe.process_frame(frame))
        frame.release()
    return out


def async_run(pipe, data, max_frames=FRAMES):
    """The documented driver loop: push bytes, pull results, never block on both."""
    out = []
    fed = 0
    while fed < len(data) and len(out) < max_frames:
        n = min(CHUNK, len(data) - fed)
        if pipe.submit(data[fed:fed + n]):
            fed += n
        elif pipe.finished:
            break
        # Draining is what makes room for the next chunk — see the class docs.
        while len(out) < max_frames and (d := pipe.try_next()) is not None:
            out.append(d)
    pipe.finish()
    while len(out) < max_frames and (d := pipe.next()) is not None:
        out.append(d)
    return out


def same_detection(a, b, eps=1e-3):
    return (a.class_id == b.class_id and abs(a.score - b.score) <= eps
            and abs(a.x1 - b.x1) <= eps and abs(a.y1 - b.y1) <= eps
            and abs(a.x2 - b.x2) <= eps and abs(a.y2 - b.y2) <= eps)


def test_async_video_matches_the_synchronous_path_frame_by_frame(rcdl, stream):
    """Every frame, every box: the overlapped pipeline must decode to the same thing.

    Three stages run concurrently on different frames and three NPU contexts
    take turns, so the ways this can go wrong are all silent ones: a frame
    letterboxed into the context another frame is inferring on, results returned
    out of order, a recycled buffer read after it went back to the decoder. Each
    of those still yields plausible boxes. Only a frame-by-frame comparison
    against the synchronous path catches them.
    """
    need(rcdl, "AsyncVideoDetectionPipeline", "VideoDecoder")
    model = bm.require_model("yolov8n_rk3588.rknn")
    data, w, h = stream

    engine = rcdl.Engine(model)
    expected = sync_reference(rcdl, engine, data)
    assert len(expected) >= FRAMES - 2, f"the reference only decoded {len(expected)} frames"

    pipe = engine.video_detector(codec="h264")
    got = async_run(pipe, data, max_frames=len(expected))

    assert len(got) == len(expected), (
        f"async returned {len(got)} frames, the synchronous path {len(expected)}")
    assert pipe.width == w and pipe.height == h
    for i, (a, e) in enumerate(zip(got, expected)):
        assert len(a) == len(e), (
            f"frame {i}: {len(a)} detections async vs {len(e)} sync — "
            f"async {[(d.class_id, round(d.score, 3)) for d in a]}, "
            f"sync {[(d.class_id, round(d.score, 3)) for d in e]}")
        for j, (da, de) in enumerate(zip(a, e)):
            assert same_detection(da, de), (
                f"frame {i} detection {j}: async {da} vs sync {de}")


def test_async_video_finds_the_bus_scene_in_every_frame(rcdl, stream):
    """The scene is known: one bus and four people, in every frame of the pan.

    Agreeing with the synchronous path proves the plumbing, not that the
    plumbing carries a picture. This is the assertion that the frames arriving
    at the NPU are the frames that were encoded — a pipeline feeding it stale or
    garbage buffers would agree with nothing.
    """
    need(rcdl, "AsyncVideoDetectionPipeline")
    model = bm.require_model("yolov8n_rk3588.rknn")
    data, _, _ = stream

    engine = rcdl.Engine(model)
    pipe = engine.video_detector(codec="h264", conf_thresh=0.35)
    frames = async_run(pipe, data)
    assert len(frames) >= FRAMES - 2, f"only {len(frames)} frames came out"

    # Measured over the 16-frame pan at conf 0.35: the bus is found in every
    # frame at 0.82-0.91, four people in the first three frames and three from
    # there on — the fourth is at the right edge and the pan carries it out. So
    # "one bus and at least two people, every frame" is the floor that is
    # actually true, and the full four-person scene is pinned on frame 0, where
    # it matches what the rest of the suite asserts about `bus.jpg`.
    for i, dets in enumerate(frames):
        classes = [d.class_id for d in dets]
        assert classes.count(5) == 1, f"frame {i}: expected exactly one bus, got {classes}"
        assert classes.count(0) >= 2, f"frame {i}: expected >= 2 people, got {classes}"
        bus = max(d.score for d in dets if d.class_id == 5)
        assert bus > 0.7, f"frame {i}: the bus scored only {bus:.2f}"
    first = [d.class_id for d in frames[0]]
    assert first.count(0) == 4 and first.count(5) == 1, (
        f"frame 0 is the unpanned scene and should be 1 bus + 4 people, got {first}")


def test_results_carry_the_decoder_timestamps_in_order(rcdl, stream):
    """pts and frame index travel beside the detections, in decode order.

    The decoded buffer is recycled the moment it has been letterboxed, so these
    two numbers are the only thing tying a result back to its frame — a caller
    muxing boxes into an output stream has nothing else to key on. They are
    carried in a FIFO alongside the detector's own reorder buffer, which is
    exactly the kind of pairing that breaks silently under load.
    """
    need(rcdl, "AsyncVideoDetectionPipeline")
    model = bm.require_model("yolov8n_rk3588.rknn")
    data, _, _ = stream

    engine = rcdl.Engine(model)
    pipe = engine.video_detector(codec="h264")
    seen = []
    fed = 0
    while fed < len(data):
        n = min(CHUNK, len(data) - fed)
        if pipe.submit(data[fed:fed + n]):
            fed += n
        elif pipe.finished:
            break
        while (d := pipe.try_next()) is not None:
            seen.append((pipe.frame_index, pipe.pts_us))
    pipe.finish()
    while pipe.next() is not None:
        seen.append((pipe.frame_index, pipe.pts_us))

    assert len(seen) >= FRAMES - 2
    indices = [i for i, _ in seen]
    assert indices == list(range(len(indices))), f"frame indices out of order: {indices}"
    pts = [p for _, p in seen]
    assert pts == sorted(pts), f"timestamps out of order: {pts}"
    # And note what the pts is NOT: a raw Annex-B elementary stream carries no
    # container timestamps, so however the encoder was stamped, every frame comes
    # back with pts 0 here. The decode index is the key that actually identifies
    # a frame on this input; the pts only becomes meaningful when the bytes come
    # from a demuxer that sets it.
    assert all(p == pts[0] for p in pts), (
        f"a raw elementary stream has no timestamps, but these vary: {pts}")


def test_back_pressure_is_a_return_value_not_a_deadlock(rcdl, stream):
    """A driver that only feeds must be REFUSED, not blocked.

    This is the regression that shaped the API. Every queue in the pipeline is
    bounded and the last one is drained only by the caller, so if submit() waited
    indefinitely for the decoder to accept bytes, a single-threaded driver would
    stall the whole chain against itself: no input taken until frames move, no
    frames until a context frees, no context until results are drained, and the
    only thread that can drain is the one parked in submit(). Feeding without
    ever draining must therefore terminate — with submit() saying no.
    """
    need(rcdl, "AsyncVideoDetectionPipeline")
    model = bm.require_model("yolov8n_rk3588.rknn")
    data, _, _ = stream

    engine = rcdl.Engine(model)
    pipe = engine.video_detector(codec="h264")
    refused = False
    fed = 0
    # Bounded on purpose: with the bug this loop never gets here at all (it hangs
    # inside submit()), and without it the refusal shows up within a few chunks.
    for _ in range(4 * (len(data) // CHUNK + 2)):
        if fed >= len(data):
            break
        n = min(CHUNK, len(data) - fed)
        if pipe.submit(data[fed:fed + n], 50):
            fed += n
        else:
            refused = True
            break
    assert refused or fed >= len(data), "submit() neither took the stream nor refused it"
    if refused:
        # And the refusal must be recoverable: drain, and it takes bytes again.
        assert pipe.next() is not None, "nothing to drain, so that was not back-pressure"
        while pipe.try_next() is not None:
            pass
        assert pipe.submit(data[fed:fed + CHUNK], 200), "still refusing after a drain"


def test_the_same_stream_detects_identically_on_every_run(rcdl, stream):
    """Same bytes, same pipeline, twice: not one box may move.

    This is the test that found the letterbox border bug, and it is pinned at
    this layer because that is where it was invisible. Nothing errored, no ioctl
    failed and every box looked reasonable — but the CPU-painted pad band was a
    read-modify-write of cache lines the RGA blit had just written, so the same
    clip gave different boxes on different runs: 7 to 16 of these 16 frames,
    ~1 px and ~0.005 of score. The border is painted by RGA now (docs/RGA.md
    §3.1) and the runs agree exactly.

    A padded source is what exercises it: this stream's 816x1088 frames letterbox
    into 640x640 with an 80 px band on each side. An aspect-matched source has no
    border and never showed the problem.
    """
    need(rcdl, "VideoDecoder")
    model = bm.require_model("yolov8n_rk3588.rknn")
    data, _, _ = stream

    engine = rcdl.Engine(model)
    first = sync_reference(rcdl, engine, data)
    second = sync_reference(rcdl, engine, data)
    assert len(first) == len(second) >= FRAMES - 2
    assert first[0], "the reference found nothing at all — check the model and the stream"
    for i, (a, b) in enumerate(zip(first, second)):
        assert len(a) == len(b), f"frame {i}: {len(a)} detections then {len(b)}"
        for j, (da, db) in enumerate(zip(a, b)):
            assert same_detection(da, db, eps=0.0), (
                f"frame {i} detection {j} moved between two identical runs: {da} vs {db}")


def test_profile_reports_every_stage(rcdl, stream):
    """The per-stage service times are what say which unit bounds the stream."""
    need(rcdl, "AsyncVideoDetectionPipeline")
    model = bm.require_model("yolov8n_rk3588.rknn")
    data, _, _ = stream

    engine = rcdl.Engine(model)
    pipe = engine.video_detector(codec="h264")
    frames = async_run(pipe, data)
    decode_ms, preproc_ms, infer_ms, postproc_ms, counted = pipe.profile
    assert counted == len(frames)
    assert decode_ms > 0 and preproc_ms > 0 and infer_ms > 0 and postproc_ms > 0
    # Nothing here is a performance claim, only that the numbers are of the
    # right kind: a per-frame stage cost above a second means the profile is
    # accumulating something other than one frame's work.
    for name, v in (("decode", decode_ms), ("preproc", preproc_ms), ("infer", infer_ms),
                    ("postproc", postproc_ms)):
        assert v < 1000.0, f"{name} reported {v:.1f} ms/frame"
    assert pipe.workers == 3
    assert pipe.head == "yolo-ltrb"
