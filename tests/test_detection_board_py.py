"""End-to-end detection on the board: RGA letterbox -> NPU -> decode.

Needs a YOLO `.rknn` staged in `models/` (scripts/fetch_models.sh) and OpenCV to
decode the sample image; skips cleanly otherwise.

    PYTHONPATH=build:python pytest -s tests/test_detection_board_py.py

The pinned expectation on `bus.jpg` (the Ultralytics COCO sample) is the same
one every YOLO reference implementation reports: one bus and four people. That
is what makes this a real end-to-end check rather than a smoke test — a wrong
channel order, a wrong letterbox inverse, a mis-detected head layout or a
mis-ordered DFL reduction all break it in an obvious way.
"""

import numpy as np
import pytest

import board_models as bm

DET_MODELS = ("yolov8n_rk3588.rknn", "yolo11n_rk3588.rknn", "yolov8n.rknn", "yolo11n.rknn")


@pytest.fixture(scope="module")
def rcdl_mod():
    m = pytest.importorskip("rcdl", reason="build the module on the board first")
    if not hasattr(m, "DetectionPipeline"):
        pytest.skip("compiled module predates the detection bindings")
    return m


@pytest.fixture(scope="module")
def det_model():
    return bm.require_model(*DET_MODELS)


@pytest.fixture(scope="module")
def pipeline(rcdl_mod, det_model):
    return rcdl_mod.Engine(det_model).detector(model_input="rgb888")


@pytest.fixture(scope="module")
def bus():
    return bm.load_bgr("bus.jpg")


def test_head_layout_is_resolved(rcdl_mod, pipeline):
    print(f"\nhead: {pipeline.head}\n{pipeline.head_layout}")
    assert pipeline.head in ("yolo-ltrb", "single-tensor")


def test_bus_jpg_finds_one_bus_and_four_people(rcdl_mod, pipeline, bus):
    dets = rcdl_mod.detect(pipeline, bus, "bgr888")
    h, w = bus.shape[:2]
    print(f"\n{w}x{h} -> {len(dets)} detections (preproc backend: {pipeline.backend})")
    counts = {}
    for d in sorted(dets, key=lambda d: -d.score):
        name = rcdl_mod.coco_class_name(d.class_id)
        counts[name] = counts.get(name, 0) + 1
        print(f"  {name:<12} {d.score:.3f}  [{d.x1:7.1f} {d.y1:7.1f} {d.x2:7.1f} {d.y2:7.1f}]")
    assert counts.get("bus", 0) == 1, f"expected exactly one bus, got {counts}"
    assert counts.get("person", 0) == 4, f"expected four people, got {counts}"


def test_boxes_are_inside_the_source_image(rcdl_mod, pipeline, bus):
    dets = rcdl_mod.detect(pipeline, bus, "bgr888")
    h, w = bus.shape[:2]
    assert dets, "no detections to check"
    for d in dets:
        assert 0.0 <= d.x1 < d.x2 <= w, f"x out of range: {d}"
        assert 0.0 <= d.y1 < d.y2 <= h, f"y out of range: {d}"
        assert 0.0 <= d.score <= 1.0


def test_bus_box_covers_most_of_the_frame(rcdl_mod, pipeline, bus):
    """The bus in this sample spans nearly the whole width — a letterbox inverse
    that forgot the padding, or applied it on the wrong axis, fails here."""
    dets = rcdl_mod.detect(pipeline, bus, "bgr888")
    h, w = bus.shape[:2]
    buses = [d for d in dets if rcdl_mod.coco_class_name(d.class_id) == "bus"]
    assert buses, "no bus detected"
    b = max(buses, key=lambda d: d.score)
    # Measured on this sample: the bus spans ~0.94 of the width and ~0.50 of the
    # height (it sits in the middle band of a portrait frame, with the people in
    # front of it). The bounds below are those numbers with room to move, not a
    # guess — a letterbox inverse that dropped the padding, or applied it on the
    # wrong axis, misses them by far more than the margin.
    assert (b.x2 - b.x1) / w > 0.85, f"bus width {(b.x2 - b.x1) / w:.2f} of the frame"
    assert 0.40 < (b.y2 - b.y1) / h < 0.62, f"bus height {(b.y2 - b.y1) / h:.2f} of the frame"


def test_results_are_deterministic(rcdl_mod, pipeline, bus):
    a = rcdl_mod.detect(pipeline, bus, "bgr888")
    b = rcdl_mod.detect(pipeline, bus, "bgr888")
    assert len(a) == len(b)
    for u, v in zip(a, b):
        assert u.class_id == v.class_id
        assert u.score == pytest.approx(v.score, abs=1e-6)
        assert u.x1 == pytest.approx(v.x1, abs=1e-4)


def test_rga_and_cpu_backends_agree_on_boxes(rcdl_mod, det_model, bus):
    """The hardware and software preproc paths must feed the model close enough
    pixels that it finds the same objects — that is the accuracy claim behind
    using RGA at all.

    What is asserted, and why it is not "every box matches": measured on this
    sample the two backends return the same five objects, and the CONFIDENT ones
    land within ~1.5 px of each other. The two MARGINAL ones do not — a person
    cut off at the frame edge scores 0.28 on RGA against 0.47 on the CPU, and an
    occluded one differs by 75 px on its top edge. That is the documented filter
    difference doing what it does (RGA pre-filters when it downscales, bilinear
    point-samples; see docs/RGA.md), amplified by a detection sitting near the
    confidence threshold where a small pixel change moves the box a lot.

    So this pins the claim that holds — same objects, and stable agreement where
    the detector is actually confident — instead of a tighter one that would be
    false.
    """
    if not rcdl_mod.rga_available():
        pytest.skip("RGA not available")
    # RGA needs a 16-pixel-aligned row stride for a 3-byte packed source, and
    # bus.jpg is 810 wide. Crop to 800 rather than letting the forced-RGA
    # pipeline raise: the point here is to compare the two resamplers on
    # identical pixels, not to re-check the alignment rule (test_letterbox.py
    # pins that).
    src = np.ascontiguousarray(bus[:, :800])
    e = rcdl_mod.Engine(det_model)
    hw = e.detector(model_input="rgb888", backend="rga")
    sw = e.detector(model_input="rgb888", backend="cpu")

    def by_place(d):
        return (d.class_id, round(d.x1), round(d.y1))

    a = sorted(rcdl_mod.detect(hw, src, "bgr888"), key=by_place)
    b = sorted(rcdl_mod.detect(sw, src, "bgr888"), key=by_place)

    def names(ds):
        return sorted(rcdl_mod.coco_class_name(d.class_id) for d in ds)

    print(f"\nRGA {len(a)} dets vs CPU {len(b)} dets")
    for u in a:
        print(f"  rga {rcdl_mod.coco_class_name(u.class_id):<8} {u.score:.3f} "
              f"[{u.x1:6.1f} {u.y1:6.1f} {u.x2:6.1f} {u.y2:6.1f}]")
    assert names(a) == names(b), f"different objects: {names(a)} vs {names(b)}"

    # Pair each RGA box with its nearest CPU box of the same class (by centre),
    # then report how far apart they are. Every object is matched — the question
    # is only how tightly.
    deltas = []
    for u in a:
        same = [v for v in b if v.class_id == u.class_id]
        assert same, f"CPU found no {rcdl_mod.coco_class_name(u.class_id)} at all"
        cu = ((u.x1 + u.x2) / 2, (u.y1 + u.y2) / 2)
        v = min(same, key=lambda z: abs((z.x1 + z.x2) / 2 - cu[0]) +
                                    abs((z.y1 + z.y2) / 2 - cu[1]))
        d = max(abs(u.x1 - v.x1), abs(u.y1 - v.y1), abs(u.x2 - v.x2), abs(u.y2 - v.y2))
        deltas.append(d)
        print(f"      vs cpu {v.score:.3f} [{v.x1:6.1f} {v.y1:6.1f} {v.x2:6.1f} {v.y2:6.1f}]"
              f"  max edge delta {d:.1f} px")

    # MOST boxes must land within a few pixels. Not all of them: measured on this
    # sample, three of the five agree to ~1.5 px, while the occluded person on the
    # right differs by ~70 px on its TOP edge and the one cut off at the frame
    # edge scores 0.28 against 0.47. Those two are exactly where a small pixel
    # change has leverage — a partly-visible object whose extent the model is
    # already unsure about — and the filter difference (RGA pre-filters on
    # downscale, bilinear point-samples; docs/RGA.md) supplies it.
    #
    # Asserting "every box within 8 px" would be asserting something false. What
    # is true, and worth protecting against regression, is that the same objects
    # are found and that the well-conditioned majority agrees tightly.
    tight = sum(1 for d in deltas if d <= 8.0)
    assert tight >= len(deltas) - 2, (
        f"only {tight} of {len(deltas)} boxes agree within 8 px: {[round(d, 1) for d in deltas]}")
    assert max(deltas) < 120.0, f"a box moved {max(deltas):.0f} px between backends"


def test_nv12_source_matches_bgr_source(rcdl_mod, pipeline, bus):
    """The video path (VPU emits NV12) must agree with the still-image path."""
    from test_letterbox import make_nv12
    h, w = bus.shape[:2]
    nv12 = make_nv12(bus[: h // 2 * 2, : w // 2 * 2])
    a = rcdl_mod.detect(pipeline, bus, "bgr888")
    b = pipeline.process(np.ascontiguousarray(nv12).reshape(-1), w // 2 * 2, h // 2 * 2, "nv12")
    names_a = sorted(rcdl_mod.coco_class_name(d.class_id) for d in a)
    names_b = sorted(rcdl_mod.coco_class_name(d.class_id) for d in b)
    print(f"\nBGR: {names_a}\nNV12: {names_b}")
    assert names_a == names_b


def test_profile_is_populated(rcdl_mod, pipeline, bus):
    pipeline.reset_profile()
    for _ in range(10):
        rcdl_mod.detect(pipeline, bus, "bgr888")
    pre, inf, post, frames = pipeline.profile
    print(f"\npreproc {pre:.2f} ms | infer {inf:.2f} ms | postproc {post:.2f} ms "
          f"| {frames} frames | {1000.0 / (pre + inf + post):.1f} fps")
    assert frames == 10
    assert pre > 0 and inf > 0 and post > 0


def test_padded_source_is_stable_across_many_frames(rcdl_mod, pipeline, bus):
    """The letterbox destination IS the NPU's input tensor, reused every frame,
    so anything left unwritten in the pad bands feeds the model the previous
    frame's pixels.

    That was a real bug: writing the border BEFORE the RGA blit left 64-byte
    runs of stale content in the band on most frames, because librga's
    destination import discards CPU writes made beforehand. It is silent — the
    boxes stay plausible — so the guard is repetition: a source that needs
    padding, run many times, must give byte-identical detections every time.
    """
    h, w = bus.shape[:2]
    # Crop to a wide aspect so the canvas gets top and bottom bands.
    src = np.ascontiguousarray(bus[h // 4: h // 4 + 320, :])
    first = rcdl_mod.detect(pipeline, src, "bgr888")
    assert first, "no detections on the padded source"
    for i in range(30):
        again = rcdl_mod.detect(pipeline, src, "bgr888")
        assert len(again) == len(first), f"iteration {i}: {len(again)} vs {len(first)}"
        for u, v in zip(first, again):
            assert u.class_id == v.class_id
            assert u.score == pytest.approx(v.score, abs=1e-6), f"iteration {i}"
            assert u.x1 == pytest.approx(v.x1, abs=1e-4), f"iteration {i}"
