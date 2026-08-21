"""RetinaFace post-processing tests.

These are PURE-NUMPY tests of the prior-box generator + variance decode — the
oracle that the C++ ``rcdl::generatePriors`` / ``rcdl::decodeFaces`` mirror. They
need only numpy and run anywhere: no board, no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_face.py

RetinaFace is an anchor-based (SSD-family) head, so the decoder is only half of
the contract: the other half is regenerating the exact prior set the model was
exported with. Two things are therefore pinned hard here rather than only
against the implementation itself:

  * the PRIOR COUNT and a handful of individually hand-computed priors, because
    the count is a fingerprint of the (steps, min_sizes, input size) triple —
    4200 at 320x320, 16800 at 640x640 for the reference configuration;
  * the VARIANCE DECODE, whose fixed point is the giveaway: a zero delta must
    reproduce the prior exactly, on the centre AND on the size. A swapped
    variance pair, or a variance applied to the wrong axis, still passes a
    "boxes look about right" eyeball check but fails this.

Decode contract mirrored here (src/tasks/face.cc decodeFaces), all in NORMALIZED
coordinates before being scaled by the model input size:

    cx = p.cx + d0 * var_center * p.w     w = p.w * exp(d2 * var_size)
    cy = p.cy + d1 * var_center * p.h     h = p.h * exp(d3 * var_size)
    lx = p.cx + e0 * var_center * p.w     ly = p.cy + e1 * var_center * p.h

then (x1,y1,x2,y2) = centre +- half-size, times (input_w, input_h), then
un-letterboxed to original-image pixels and clamped. Landmarks take the same
un-letterbox + clamp as the box.
"""

import math

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Letterbox geometry — mirrors include/rcdl/preproc/geometry.h                 #
# --------------------------------------------------------------------------- #
class Letterbox:
    def __init__(self, scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h):
        self.scale, self.pad_x, self.pad_y = scale, pad_x, pad_y
        self.src_w, self.src_h = src_w, src_h
        self.dst_w, self.dst_h = dst_w, dst_h

    def inv_x(self, x):
        return (x - self.pad_x) / self.scale

    def inv_y(self, y):
        return (y - self.pad_y) / self.scale

    def clamp_x(self, x):
        return min(max(x, 0.0), float(self.src_w))

    def clamp_y(self, y):
        return min(max(y, 0.0), float(self.src_h))

    def as_tuple(self):
        return (self.scale, self.pad_x, self.pad_y, self.src_w, self.src_h,
                self.dst_w, self.dst_h)


def compute_letterbox(src_w, src_h, dst_w, dst_h, center_pad=True):
    scale = min(dst_w / src_w, dst_h / src_h)
    pad_x = (dst_w - src_w * scale) * 0.5 if center_pad else 0.0
    pad_y = (dst_h - src_h * scale) * 0.5 if center_pad else 0.0
    return Letterbox(scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h)


# --------------------------------------------------------------------------- #
# Reference configuration + prior generator                                    #
# --------------------------------------------------------------------------- #
STEPS = (8, 16, 32)
MIN_SIZES = ((16, 32), (64, 128), (256, 512))
VAR_CENTER = 0.1
VAR_SIZE = 0.2
NUM_LANDMARKS = 5


def ref_priors(input_w, input_h, steps=STEPS, min_sizes=MIN_SIZES, clip=False):
    """Prior boxes as (N, 4) of normalized (cx, cy, w, h), in output-row order.

    The nesting is the part that matters and is deliberately written out long-
    hand: scale, then grid row, then grid column, then min_size — min_size
    varying FASTEST. Any other nesting yields the same N with a permuted layout,
    which decodes into plausible-looking but wrong boxes.
    """
    out = []
    for k, step in enumerate(steps):
        grid_h = math.ceil(input_h / step)
        grid_w = math.ceil(input_w / step)
        for i in range(grid_h):
            for j in range(grid_w):
                cx = (j + 0.5) * step / input_w
                cy = (i + 0.5) * step / input_h
                for ms in min_sizes[k]:
                    p = [cx, cy, ms / input_w, ms / input_h]
                    if clip:
                        p = [min(max(v, 0.0), 1.0) for v in p]
                    out.append(p)
    return np.array(out, dtype=np.float64).reshape(-1, 4)


# --------------------------------------------------------------------------- #
# Reference decode oracle                                                      #
# --------------------------------------------------------------------------- #
def ref_iou(a, b):
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    uni = area_a + area_b - inter
    return inter / uni if uni > 0 else 0.0


def ref_nms(dets, iou_t, max_dets):
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
    keep, dead = [], set()
    for oi, i in enumerate(order):
        if i in dead:
            continue
        keep.append(i)
        if max_dets > 0 and len(keep) >= max_dets:
            break
        for j in order[oi + 1:]:
            if j in dead:
                continue
            if ref_iou(dets[i], dets[j]) > iou_t:
                dead.add(j)
    return keep


def ref_decode_faces(loc, conf, landm, priors, lb, input_w, input_h,
                     conf_thresh=0.5, iou_thresh=0.4, max_faces=100,
                     var_center=VAR_CENTER, var_size=VAR_SIZE,
                     face_class=1, apply_softmax=False):
    """loc (N,4), conf (N,2), landm (N,10), priors (N,4) -> list of faces."""
    loc = np.asarray(loc, dtype=np.float64).reshape(-1, 4)
    conf = np.asarray(conf, dtype=np.float64).reshape(-1, 2)
    landm = np.asarray(landm, dtype=np.float64).reshape(-1, 10)
    assert len(loc) == len(conf) == len(landm) == len(priors)

    if apply_softmax:
        m = conf.max(axis=1, keepdims=True)
        e = np.exp(conf - m)
        scores = (e / e.sum(axis=1, keepdims=True))[:, face_class]
    else:
        scores = conf[:, face_class]

    pcx, pcy, pw, ph = priors[:, 0], priors[:, 1], priors[:, 2], priors[:, 3]
    cx = pcx + loc[:, 0] * var_center * pw
    cy = pcy + loc[:, 1] * var_center * ph
    w = pw * np.exp(loc[:, 2] * var_size)
    h = ph * np.exp(loc[:, 3] * var_size)

    x1 = (cx - w / 2) * input_w
    y1 = (cy - h / 2) * input_h
    x2 = (cx + w / 2) * input_w
    y2 = (cy + h / 2) * input_h

    lx = np.stack([(pcx + landm[:, 2 * t] * var_center * pw) * input_w
                   for t in range(NUM_LANDMARKS)], axis=1)
    ly = np.stack([(pcy + landm[:, 2 * t + 1] * var_center * ph) * input_h
                   for t in range(NUM_LANDMARKS)], axis=1)

    cands = []
    for i in np.flatnonzero(scores >= conf_thresh):
        cands.append({
            "x1": lb.clamp_x(lb.inv_x(x1[i])), "y1": lb.clamp_y(lb.inv_y(y1[i])),
            "x2": lb.clamp_x(lb.inv_x(x2[i])), "y2": lb.clamp_y(lb.inv_y(y2[i])),
            "score": float(scores[i]),
            "landmarks": [(lb.clamp_x(lb.inv_x(lx[i, t])), lb.clamp_y(lb.inv_y(ly[i, t])))
                          for t in range(NUM_LANDMARKS)],
        })
    return [cands[i] for i in ref_nms(cands, iou_thresh, max_faces)]


# --------------------------------------------------------------------------- #
# Optional C++ cross-check                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cxx():
    """The compiled module, or None — every test still asserts on the oracle."""
    try:
        import rcdl
    except Exception:
        return None
    return rcdl


# --------------------------------------------------------------------------- #
# Prior boxes                                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size,want", [(320, 4200), (640, 16800)])
def test_prior_count_is_the_configuration_fingerprint(size, want):
    # 320: (40*40 + 20*20 + 10*10) * 2 = 4200;  640: (80*80 + 40*40 + 20*20) * 2.
    priors = ref_priors(size, size)
    assert len(priors) == want
    per_scale = [math.ceil(size / s) ** 2 * len(m) for s, m in zip(STEPS, MIN_SIZES)]
    assert sum(per_scale) == want


def test_prior_count_handles_a_non_multiple_input():
    # ceil, not floor: a 300px side at stride 8 still gets 38 rows of cells.
    priors = ref_priors(300, 300)
    assert len(priors) == (38 * 38 + 19 * 19 + 10 * 10) * 2


def test_known_priors_at_320():
    """Hand-computed priors at the three scale boundaries."""
    p = ref_priors(320, 320)
    # First cell of stride 8, both min_sizes — min_size varies fastest.
    assert p[0] == pytest.approx([0.5 * 8 / 320, 0.5 * 8 / 320, 16 / 320, 16 / 320])
    assert p[1] == pytest.approx([0.5 * 8 / 320, 0.5 * 8 / 320, 32 / 320, 32 / 320])
    # Second cell along the row: cx advances by one stride, cy does not.
    assert p[2] == pytest.approx([1.5 * 8 / 320, 0.5 * 8 / 320, 16 / 320, 16 / 320])
    # Start of the second row of stride 8 (grid is 40 wide, 2 priors per cell).
    assert p[40 * 2] == pytest.approx([0.5 * 8 / 320, 1.5 * 8 / 320, 16 / 320, 16 / 320])
    # First prior of stride 16 (3200 priors of stride 8 precede it).
    assert p[3200] == pytest.approx([0.5 * 16 / 320, 0.5 * 16 / 320, 64 / 320, 64 / 320])
    # First prior of stride 32 (3200 + 800 precede it).
    assert p[4000] == pytest.approx([0.5 * 32 / 320, 0.5 * 32 / 320, 256 / 320, 256 / 320])
    # Last prior: bottom-right cell of stride 32, largest min_size. It is WIDER
    # than the canvas (512 > 320) — the reference generator does not clip, and
    # clipping it would shrink every large border face.
    assert p[4199] == pytest.approx([9.5 * 32 / 320, 9.5 * 32 / 320, 512 / 320, 512 / 320])
    assert p[4199][2] > 1.0


def test_clip_only_affects_oversized_priors():
    plain = ref_priors(320, 320, clip=False)
    clipped = ref_priors(320, 320, clip=True)
    assert np.all(clipped <= 1.0)
    # Only the stride-32 priors exceed the canvas; the small ones are untouched.
    assert np.array_equal(plain[:3200], clipped[:3200])
    assert not np.array_equal(plain[4000:], clipped[4000:])


def test_priors_are_ordered_scale_row_col_minsize():
    """The generator's nesting, read back off the array."""
    p = ref_priors(320, 320)
    grid = 40
    for i in (0, 1, 17, 39):
        for j in (0, 5, 39):
            base = (i * grid + j) * 2
            assert p[base][0] == pytest.approx((j + 0.5) * 8 / 320)
            assert p[base][1] == pytest.approx((i + 0.5) * 8 / 320)
            assert p[base][2] == pytest.approx(16 / 320)
            assert p[base + 1][2] == pytest.approx(32 / 320)


def test_priors_match_cxx(cxx):
    if cxx is None or not hasattr(cxx, "generate_priors"):
        pytest.skip("compiled rcdl module without generate_priors bindings")
    for size in (320, 640):
        got = np.asarray(cxx.generate_priors(size, size), dtype=np.float64).reshape(-1, 4)
        want = ref_priors(size, size)
        assert got.shape == want.shape
        assert np.allclose(got, want, atol=1e-6)


# --------------------------------------------------------------------------- #
# Variance decode                                                              #
# --------------------------------------------------------------------------- #
def _identity_letterbox(size):
    """scale 1, no padding: model pixels ARE source pixels."""
    return Letterbox(1.0, 0.0, 0.0, size, size, size, size)


def test_zero_delta_reproduces_the_prior_exactly():
    """The decode's fixed point — the decisive check on the variance wiring.

    With every delta zero the centre offset term vanishes and exp(0) == 1, so the
    decoded box must be the prior itself, scaled to model pixels. This is what
    catches a variance applied to the wrong term: a swapped (0.1, 0.2) pair is
    invisible on the centre (0 * anything == 0) but would still be exercised on
    the size if exp() were fed the wrong factor.
    """
    size = 320
    priors = ref_priors(size, size)
    n = len(priors)
    loc = np.zeros((n, 4))
    landm = np.zeros((n, 10))
    conf = np.zeros((n, 2))
    conf[:, 1] = 1.0  # everything is a face, so nothing is thresholded away

    # A stride-16 prior near the middle of the canvas: big enough that an error
    # would be obvious, but wholly INSIDE the image so the clamp cannot mask it.
    # Scale 1 starts at 3200; its grid is 20x20 with 2 priors per cell.
    idx = 3200 + (10 * 20 + 10) * 2
    faces = ref_decode_faces(loc, conf, landm, priors, _identity_letterbox(size),
                             size, size, iou_thresh=1.1, max_faces=n)
    p = priors[idx]
    want = {
        "x1": (p[0] - p[2] / 2) * size, "y1": (p[1] - p[3] / 2) * size,
        "x2": (p[0] + p[2] / 2) * size, "y2": (p[1] + p[3] / 2) * size,
    }
    # iou_thresh > 1 suppresses nothing, so candidate order is score order and
    # ties keep the input order: face `idx` is still at index `idx`.
    got = faces[idx]
    for f in ("x1", "y1", "x2", "y2"):
        assert got[f] == pytest.approx(want[f], abs=1e-6), f
    # ... and all five landmarks collapse onto the prior centre.
    for lx, ly in got["landmarks"]:
        assert lx == pytest.approx(p[0] * size, abs=1e-6)
        assert ly == pytest.approx(p[1] * size, abs=1e-6)


def test_size_delta_uses_var_size_and_centre_delta_uses_var_center():
    """One prior, one non-zero delta at a time — the two variances pinned apart."""
    size = 320
    priors = np.array([[0.5, 0.5, 0.25, 0.125]])
    lb = _identity_letterbox(size)
    conf = np.array([[0.0, 1.0]])
    landm = np.zeros((1, 10))

    # Centre only: dx = 2 shifts cx by 2 * 0.1 * prior_w, and NOT by prior_h.
    loc = np.array([[2.0, -3.0, 0.0, 0.0]])
    f = ref_decode_faces(loc, conf, landm, priors, lb, size, size)[0]
    cx = (f["x1"] + f["x2"]) / 2
    cy = (f["y1"] + f["y2"]) / 2
    assert cx == pytest.approx((0.5 + 2.0 * VAR_CENTER * 0.25) * size)
    assert cy == pytest.approx((0.5 - 3.0 * VAR_CENTER * 0.125) * size)
    # Size untouched.
    assert f["x2"] - f["x1"] == pytest.approx(0.25 * size)
    assert f["y2"] - f["y1"] == pytest.approx(0.125 * size)

    # Size only: log-space, scaled by var_size.
    loc = np.array([[0.0, 0.0, 1.0, -1.0]])
    f = ref_decode_faces(loc, conf, landm, priors, lb, size, size)[0]
    assert f["x2"] - f["x1"] == pytest.approx(0.25 * math.exp(1.0 * VAR_SIZE) * size)
    assert f["y2"] - f["y1"] == pytest.approx(0.125 * math.exp(-1.0 * VAR_SIZE) * size)
    # A swapped variance pair would give exp(0.1) here instead of exp(0.2).
    assert f["x2"] - f["x1"] != pytest.approx(0.25 * math.exp(1.0 * VAR_CENTER) * size)


def test_landmarks_use_var_center_on_both_axes():
    """Landmarks are centre-like offsets: var_size must never appear in them."""
    size = 320
    priors = np.array([[0.5, 0.5, 0.25, 0.125]])
    lb = _identity_letterbox(size)
    conf = np.array([[0.0, 1.0]])
    loc = np.zeros((1, 4))
    deltas = np.array([1.0, -1.0, 2.0, -2.0, 0.5, 0.5, -1.5, 1.0, 3.0, -0.25])
    f = ref_decode_faces(loc, conf, deltas.reshape(1, 10), priors, lb, size, size)[0]
    for t in range(NUM_LANDMARKS):
        want_x = (0.5 + deltas[2 * t] * VAR_CENTER * 0.25) * size
        want_y = (0.5 + deltas[2 * t + 1] * VAR_CENTER * 0.125) * size
        assert f["landmarks"][t][0] == pytest.approx(want_x), t
        assert f["landmarks"][t][1] == pytest.approx(want_y), t


def test_landmark_order_is_eyes_nose_mouth():
    """The canonical order the whole face stack assumes, asserted geometrically.

    A synthetic upright face: the two eyes above the nose, the two mouth corners
    below it, left of the subject first. Any permutation of the 10 channels
    breaks at least one of these relations.
    """
    size = 320
    priors = np.array([[0.5, 0.5, 0.25, 0.25]])
    lb = _identity_letterbox(size)
    #        left eye     right eye    nose       left mouth   right mouth
    deltas = [-4.0, -3.0, 4.0, -3.0, 0.0, 0.0, -3.0, 4.0, 3.0, 4.0]
    f = ref_decode_faces(np.zeros((1, 4)), np.array([[0.0, 1.0]]),
                         np.array([deltas]), priors, lb, size, size)[0]
    le, re, nose, lm, rm = f["landmarks"]
    assert le[0] < nose[0] < re[0]          # eyes straddle the nose horizontally
    assert le[1] == pytest.approx(re[1])    # eyes level
    assert le[1] < nose[1] < lm[1]          # eyes above nose above mouth
    assert lm[0] < nose[0] < rm[0]
    assert lm[1] == pytest.approx(rm[1])


# --------------------------------------------------------------------------- #
# Un-letterboxing + clamping                                                   #
# --------------------------------------------------------------------------- #
def test_letterbox_geometry_roundtrip():
    lb = compute_letterbox(1280, 720, 320, 320)
    assert lb.scale == pytest.approx(0.25)
    assert lb.pad_x == pytest.approx(0.0)
    assert lb.pad_y == pytest.approx(70.0)
    for x, y in ((0, 0), (640, 360), (1279, 719)):
        assert lb.inv_x(x * lb.scale + lb.pad_x) == pytest.approx(x)
        assert lb.inv_y(y * lb.scale + lb.pad_y) == pytest.approx(y)


def test_box_and_landmarks_map_back_to_source_pixels():
    """A box placed at a known spot in the ORIGINAL image survives the round trip.

    The prior is chosen so that a zero delta puts the box exactly where we want
    it in model coordinates; decoding must then hand back the source-pixel box
    the letterbox was built from.
    """
    src_w, src_h, size = 1280, 720, 320
    lb = compute_letterbox(src_w, src_h, size, size)
    # Target: a 200x200 box centred at (640, 360) in the source image.
    want = (540.0, 260.0, 740.0, 460.0)
    cx_m = lb.scale * 640 + lb.pad_x
    cy_m = lb.scale * 360 + lb.pad_y
    w_m = 200 * lb.scale
    h_m = 200 * lb.scale
    priors = np.array([[cx_m / size, cy_m / size, w_m / size, h_m / size]])

    f = ref_decode_faces(np.zeros((1, 4)), np.array([[0.0, 0.9]]), np.zeros((1, 10)),
                         priors, lb, size, size)[0]
    for got, exp, name in zip((f["x1"], f["y1"], f["x2"], f["y2"]), want,
                              ("x1", "y1", "x2", "y2")):
        assert got == pytest.approx(exp, abs=1e-3), name
    # Landmarks live in the same coordinate system, so a zero delta puts them all
    # on the source-pixel box centre.
    for lx, ly in f["landmarks"]:
        assert lx == pytest.approx(640.0, abs=1e-3)
        assert ly == pytest.approx(360.0, abs=1e-3)


def test_padding_region_clamps_box_and_landmarks():
    """A prior sitting in the letterbox bars decodes to the image border, not
    to negative pixels — for the landmarks as much as for the box."""
    src_w, src_h, size = 1280, 720, 320
    lb = compute_letterbox(src_w, src_h, size, size)
    # Centre this prior 20 model-pixels ABOVE the image content, i.e. inside the
    # top bar, and make it large enough to hang off the left edge too.
    cy_m = lb.pad_y - 20.0
    priors = np.array([[0.02, cy_m / size, 0.30, 0.30]])
    f = ref_decode_faces(np.zeros((1, 4)), np.array([[0.0, 0.9]]), np.zeros((1, 10)),
                         priors, lb, size, size)[0]
    assert f["x1"] == 0.0
    assert f["y1"] == 0.0
    assert 0.0 <= f["x2"] <= src_w
    assert 0.0 <= f["y2"] <= src_h
    for lx, ly in f["landmarks"]:
        assert 0.0 <= lx <= src_w
        assert 0.0 <= ly <= src_h
    assert f["landmarks"][0][1] == 0.0  # the un-clamped y would be negative


# --------------------------------------------------------------------------- #
# Scores + NMS                                                                 #
# --------------------------------------------------------------------------- #
def test_scores_are_taken_from_the_face_column_unactivated():
    """The deployed export softmaxes in the graph, so column 1 IS the probability.

    Re-activating it would squash every score toward 0.5 and quietly wreck the
    threshold, so the default path must pass the value straight through.
    """
    size = 320
    priors = np.array([[0.5, 0.5, 0.2, 0.2]])
    lb = _identity_letterbox(size)
    conf = np.array([[0.13, 0.87]])
    f = ref_decode_faces(np.zeros((1, 4)), conf, np.zeros((1, 10)), priors, lb, size, size)
    assert len(f) == 1
    assert f[0]["score"] == pytest.approx(0.87)
    # Background column below threshold => nothing survives.
    assert ref_decode_faces(np.zeros((1, 4)), np.array([[0.87, 0.13]]), np.zeros((1, 10)),
                            priors, lb, size, size) == []


def test_softmax_path_normalizes_logits():
    size = 320
    priors = np.array([[0.5, 0.5, 0.2, 0.2]])
    lb = _identity_letterbox(size)
    logits = np.array([[-1.0, 2.0]])
    f = ref_decode_faces(np.zeros((1, 4)), logits, np.zeros((1, 10)), priors, lb,
                         size, size, apply_softmax=True)[0]
    assert f["score"] == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))


def test_nms_keeps_the_best_of_two_overlapping_priors():
    size = 320
    lb = _identity_letterbox(size)
    priors = np.array([[0.50, 0.50, 0.20, 0.20],
                       [0.51, 0.51, 0.20, 0.20],
                       [0.10, 0.10, 0.05, 0.05]])
    conf = np.array([[0.1, 0.7], [0.1, 0.9], [0.1, 0.8]])
    faces = ref_decode_faces(np.zeros((3, 4)), conf, np.zeros((3, 10)), priors, lb, size, size)
    assert [pytest.approx(f["score"]) for f in faces] == [0.9, 0.8]
    # The kept face is the higher-scoring of the pair, so it carries ITS box.
    assert faces[0]["x1"] == pytest.approx((0.51 - 0.10) * size)


def test_max_faces_truncates():
    size = 320
    lb = _identity_letterbox(size)
    n = 10
    priors = np.stack([np.array([0.05 + 0.09 * i, 0.5, 0.02, 0.02]) for i in range(n)])
    conf = np.stack([np.array([0.0, 0.5 + 0.04 * i]) for i in range(n)])
    faces = ref_decode_faces(np.zeros((n, 4)), conf, np.zeros((n, 10)), priors, lb,
                             size, size, max_faces=3)
    assert len(faces) == 3


# --------------------------------------------------------------------------- #
# Layout invariance + C++ cross-check                                          #
# --------------------------------------------------------------------------- #
def test_channels_first_and_last_agree_in_the_oracle():
    """[1,N,C] and [1,C,N] are the same numbers; transposing must not change the
    result. The C++ decoder reads each tensor's real shape to decide, so this is
    the property it has to hold."""
    size, n = 320, 64
    rng = np.random.default_rng(7)
    priors = ref_priors(size, size)[:n]
    loc = rng.uniform(-2, 2, size=(n, 4))
    conf = rng.uniform(0, 1, size=(n, 2))
    landm = rng.uniform(-2, 2, size=(n, 10))
    lb = compute_letterbox(1280, 720, size, size)
    a = ref_decode_faces(loc, conf, landm, priors, lb, size, size)
    b = ref_decode_faces(loc.T.T, conf.T.T, landm.T.T, priors, lb, size, size)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x["score"] == pytest.approx(y["score"])


def test_decode_faces_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "decode_faces"):
        pytest.skip("compiled rcdl module without decode_faces bindings")
    size = 320
    priors = ref_priors(size, size)
    n = len(priors)
    rng = np.random.default_rng(11)
    loc = rng.uniform(-2.0, 2.0, size=(n, 4)).astype(np.float32)
    landm = rng.uniform(-3.0, 3.0, size=(n, 10)).astype(np.float32)
    conf = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
    lb = compute_letterbox(1280, 720, size, size)

    ref = ref_decode_faces(loc, conf, landm, priors, lb, size, size)
    got = cxx.decode_faces(np.ascontiguousarray(loc), np.ascontiguousarray(conf),
                           np.ascontiguousarray(landm), lb.as_tuple())
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert abs(a.score - b["score"]) < 1e-4
        for f in ("x1", "y1", "x2", "y2"):
            assert abs(getattr(a, f) - b[f]) < 1e-2, f
        for t in range(NUM_LANDMARKS):
            assert abs(a.landmarks[t][0] - b["landmarks"][t][0]) < 1e-2
            assert abs(a.landmarks[t][1] - b["landmarks"][t][1]) < 1e-2


# --------------------------------------------------------------------------- #
# Board test — real model, real image                                          #
# --------------------------------------------------------------------------- #
def test_face_detector_on_the_board(cxx, model_path):
    """End-to-end on a real .rknn. Skips unless --model points at a face model."""
    if cxx is None or not hasattr(cxx, "FaceDetector"):
        pytest.skip("compiled rcdl module without FaceDetector bindings")
    eng = cxx.Engine(model_path)
    try:
        det = cxx.FaceDetector(eng)
    except Exception as exc:  # not a RetinaFace model — the head resolver says so
        pytest.skip(f"not a face model: {exc}")
    # The prior count is derived from the model, so it must agree with the oracle.
    layout = det.layout
    assert len(det.priors) == layout.num_priors
    assert len(ref_priors(layout.input_w, layout.input_h)) == layout.num_priors


# --------------------------------------------------------------------------- #
# Alignment — the 5-point similarity transform recognition depends on          #
# --------------------------------------------------------------------------- #
ARCFACE_112 = np.array([[38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float64)


def ref_similarity(src, dst):
    """The oracle for `rcdl::similarityTransform`, as a (2,3) affine.

    A 2-D similarity is one complex multiplication, so the least-squares fit is
    a single quotient over the centred points. Written this way rather than with
    an SVD because the SVD form can return a REFLECTION unless the determinant
    is checked, and a mirrored face is a perfectly good fit to five points.
    """
    src = np.asarray(src, dtype=np.float64).reshape(5, 2)
    dst = np.asarray(dst, dtype=np.float64).reshape(5, 2)
    sc, dc = src.mean(0), dst.mean(0)
    z = (src - sc)[:, 0] + 1j * (src - sc)[:, 1]
    w = (dst - dc)[:, 0] + 1j * (dst - dc)[:, 1]
    den = float((z * z.conj()).sum().real)
    c = (w * z.conj()).sum() / den if den > 1e-12 else complex(1.0, 0.0)
    a, b = c.real, c.imag
    m = np.array([[a, -b, 0.0], [b, a, 0.0]])
    m[:, 2] = dc - m[:, :2] @ sc
    return m


def apply_affine(m, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return pts @ np.asarray(m)[:, :2].T + np.asarray(m)[:, 2]


def test_alignment_is_exact_when_the_face_is_a_similarity_of_the_template():
    """A rotated, scaled, shifted copy of the template must map back onto it exactly.

    This is the only case with a knowable answer, and it is the one that catches
    a transposed matrix, a swapped sign on the rotation, or degrees-for-radians:
    all of those still produce a plausible-looking crop.
    """
    for angle, scale, tx, ty in [(0.0, 1.0, 0, 0), (0.35, 2.5, -40, 17),
                                 (-1.2, 0.4, 300, 200), (np.pi, 1.7, 5, -9)]:
        c, s = np.cos(angle) * scale, np.sin(angle) * scale
        r = np.array([[c, -s], [s, c]])
        src = ARCFACE_112 @ r.T + np.array([tx, ty])
        m = ref_similarity(src, ARCFACE_112)
        back = apply_affine(m, src)
        assert np.allclose(back, ARCFACE_112, atol=1e-6), f"angle={angle} scale={scale}"


def test_alignment_never_mirrors_a_face():
    """A mirrored set of landmarks must NOT be un-mirrored by the transform.

    A reflection fits five points just as well as a rotation, and an alignment
    that silently mirrors would hand the identity model a face that is not the
    one in the picture. The similarity form cannot express a reflection: its
    matrix is [[a,-b],[b,a]], whose determinant a²+b² is never negative.
    """
    mirrored = ARCFACE_112.copy()
    mirrored[:, 0] = 112.0 - mirrored[:, 0]
    m = ref_similarity(mirrored, ARCFACE_112)
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    assert det > 0, f"the fit reflected: determinant {det}"
    # And it does not fit well, which is the honest consequence: a mirrored face
    # cannot be aligned by a similarity, so the residual is large.
    residual = np.abs(apply_affine(m, mirrored) - ARCFACE_112).max()
    assert residual > 5.0, f"a mirrored face aligned suspiciously well (residual {residual})"


def test_alignment_of_coincident_points_is_a_translation_not_a_crash():
    pts = np.tile([50.0, 50.0], (5, 1))
    m = ref_similarity(pts, ARCFACE_112)
    assert np.all(np.isfinite(m))
    assert np.allclose(apply_affine(m, pts), ARCFACE_112.mean(0), atol=1e-6)


def test_template_scales_each_axis_on_its_own(cxx):
    """A non-square output must scale x by w/112 and y by h/112, not by one factor.

    Scaling both axes by the same number on a non-square crop tilts every face,
    which is exactly the kind of thing that costs accuracy without erroring.
    """
    if cxx is None or not hasattr(cxx, "arcface_template"):
        pytest.skip("compiled module without arcface_template")
    got = np.asarray(cxx.arcface_template(224, 112))
    expect = ARCFACE_112 * np.array([2.0, 1.0])
    assert np.allclose(got, expect, atol=1e-4)
    assert np.allclose(np.asarray(cxx.arcface_template()), ARCFACE_112, atol=1e-4)


def test_alignment_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "similarity_transform"):
        pytest.skip("compiled module without similarity_transform")
    rng = np.random.default_rng(11)
    for _ in range(64):
        src = (ARCFACE_112 + rng.normal(scale=6.0, size=(5, 2))) * rng.uniform(0.3, 4.0)
        src += rng.uniform(-200, 200, size=2)
        got = np.asarray(cxx.similarity_transform(src.astype(np.float32).reshape(-1),
                                                  ARCFACE_112.astype(np.float32).reshape(-1)))
        assert np.allclose(got, ref_similarity(src, ARCFACE_112), atol=1e-3)
        got2 = np.asarray(cxx.face_align_transform(src.astype(np.float32).reshape(-1), 112, 112))
        assert np.allclose(got2, got, atol=1e-6)
