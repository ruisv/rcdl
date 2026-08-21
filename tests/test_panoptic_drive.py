"""Anchor-based (YOLOv5-style) detection decode — pure-numpy oracle.

These need only numpy: no board, no ``.rknn``. The module-level ``ref_*``
functions are the documented reference, and when the compiled module is
importable ``rcdl.decode_yolov5_anchor`` is checked against them. The real-model
path is in tests/test_tasks_board_py.py.

    PYTHONPATH=build:python pytest -s tests/test_panoptic_drive.py

WHY THIS IS A SEPARATE DECODE. Everything else in this library decodes
anchor-FREE heads (LTRB, with or without DFL), where a cell predicts distances to
the four box edges. An anchor-BASED head instead predicts, per prior box, an
offset from the cell and a multiplier on that prior's size — arithmetic that
cannot be expressed in the LTRB decoder's terms at all.
"""

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


def _identity_lb():
    """Model pixels == original pixels, so decoded boxes can be compared
    directly against the formula's model-space output."""
    return compute_letterbox(64, 64, 64, 64)


# --------------------------------------------------------------------------- #
# Reference decode                                                            #
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ref_decode_anchor(raw, stride, anchors, num_classes):
    """The published anchor decode, written out plainly.

    ``raw``: [na*(5+nc), H, W] unactivated, channels-first. Returns
    (boxes_cxcywh, scores, class_ids) in MODEL-INPUT pixels, unthresholded and
    unsuppressed."""
    c, h, w = raw.shape
    na, no = len(anchors), 5 + num_classes
    assert c == na * no, f"{c} channels != {na} anchors x {no}"
    t = _sigmoid(raw.reshape(na, no, h, w).transpose(0, 2, 3, 1))   # (na,H,W,no)
    gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    grid = np.stack((gx, gy), -1).astype(np.float32)                # (H,W,2)
    a = np.array([[p[0], p[1]] for p in anchors], np.float32).reshape(na, 1, 1, 2)
    xy = (t[..., 0:2] * 2 - 0.5 + grid) * stride
    wh = (t[..., 2:4] * 2) ** 2 * a
    obj, cls = t[..., 4], t[..., 5:]
    return (np.concatenate([xy, wh], -1).reshape(-1, 4),
            (obj * cls.max(-1)).reshape(-1),
            cls.argmax(-1).reshape(-1))


def ref_corners(boxes, lb):
    """cxcywh in model pixels -> corners in ORIGINAL pixels, clamped — the
    decoder maps back through the letterbox, so the reference must too."""
    out = np.empty_like(boxes)
    out[:, 0] = [lb.clamp_x(lb.inv_x(v)) for v in boxes[:, 0] - boxes[:, 2] / 2]
    out[:, 1] = [lb.clamp_y(lb.inv_y(v)) for v in boxes[:, 1] - boxes[:, 3] / 2]
    out[:, 2] = [lb.clamp_x(lb.inv_x(v)) for v in boxes[:, 0] + boxes[:, 2] / 2]
    out[:, 3] = [lb.clamp_y(lb.inv_y(v)) for v in boxes[:, 1] + boxes[:, 3] / 2]
    return out


@pytest.fixture(scope="module")
def cxx():
    """The compiled module, or None — every test still asserts on the oracle."""
    try:
        import rcdl
    except Exception:
        return None
    return rcdl if hasattr(rcdl, "decode_yolov5_anchor") else None


def _anchors(cxx_mod, pairs):
    return [cxx_mod.Anchor(w, h) for w, h in pairs]


def _decode(cxx_mod, raws, grids, strides, anchor_pairs, lb, num_classes=1,
            conf=0.0, iou=1.0):
    return cxx_mod.decode_yolov5_anchor(
        [np.ascontiguousarray(r, np.float32) for r in raws],
        grids, list(strides), [_anchors(cxx_mod, p) for p in anchor_pairs],
        lb.as_tuple(), num_classes=num_classes, conf_thresh=conf,
        iou_thresh=iou, max_dets=10000)


# --------------------------------------------------------------------------- #
# The formula                                                                 #
# --------------------------------------------------------------------------- #
def test_single_cell_is_known_by_construction():
    """One anchor, one cell, all raw logits zero => sigmoid is 0.5 everywhere,
    so every term collapses to a number that can be worked out by hand:
    centre = (0.5*2 - 0.5 + 0) * stride, size = (0.5*2)^2 * anchor."""
    stride, aw, ah = 8, 10.0, 20.0
    raw = np.zeros((6, 1, 1), np.float32)                 # na=1, nc=1
    boxes, scores, ids = ref_decode_anchor(raw, stride, [(aw, ah)], 1)
    assert boxes.shape == (1, 4)
    assert boxes[0, 0] == pytest.approx(4.0)              # (1.0 - 0.5 + 0) * 8
    assert boxes[0, 1] == pytest.approx(4.0)
    assert boxes[0, 2] == pytest.approx(aw)               # (1.0)^2 * anchor_w
    assert boxes[0, 3] == pytest.approx(ah)
    assert scores[0] == pytest.approx(0.25)               # obj 0.5 * cls 0.5
    assert ids[0] == 0


def test_the_offset_form_reaches_outside_its_own_cell():
    """(y*2 - 0.5) is not (y): the reference form lets a cell's centre land
    anywhere in [-0.5, +1.5] cells, which is why the original exponential
    formulation was replaced. A decoder that dropped the *2-0.5 would confine
    every centre to its own cell and still look plausible."""
    stride = 8
    raw = np.zeros((6, 1, 1), np.float32)
    raw[0, 0, 0] = -20.0                                  # sigmoid -> 0
    raw[1, 0, 0] = +20.0                                  # sigmoid -> 1
    boxes, _, _ = ref_decode_anchor(raw, stride, [(10.0, 20.0)], 1)
    assert boxes[0, 0] == pytest.approx(-0.5 * stride)     # left of cell 0
    assert boxes[0, 1] == pytest.approx(1.5 * stride)      # a cell and a half down


def test_the_size_form_is_bounded_at_four_times_the_prior():
    """(y*2)^2 saturates at 4x the prior. A box bigger than that is not
    representable at this scale at all, which is what the prior set is for."""
    raw = np.full((6, 1, 1), 20.0, np.float32)             # sigmoid -> 1
    boxes, _, _ = ref_decode_anchor(raw, 8, [(10.0, 20.0)], 1)
    assert boxes[0, 2] == pytest.approx(40.0)
    assert boxes[0, 3] == pytest.approx(80.0)


def test_objectness_gates_the_class_score():
    """Score is objectness * class probability, so a dead objectness kills the
    candidate however confident the class is."""
    raw = np.zeros((6, 1, 1), np.float32)
    raw[4, 0, 0] = -10.0                                   # objectness ~ 4.5e-5
    raw[5, 0, 0] = +10.0                                   # class ~ 1.0
    _, scores, _ = ref_decode_anchor(raw, 8, [(10.0, 20.0)], 1)
    assert scores[0] < 0.001
    raw[4, 0, 0] = +10.0
    _, scores, _ = ref_decode_anchor(raw, 8, [(10.0, 20.0)], 1)
    assert scores[0] > 0.99


def test_class_argmax_across_classes():
    nc = 3
    raw = np.zeros((5 + nc, 1, 1), np.float32)
    raw[4, 0, 0] = 10.0
    raw[5 + 2, 0, 0] = 10.0
    _, _, ids = ref_decode_anchor(raw, 8, [(10.0, 20.0)], nc)
    assert ids[0] == 2


def test_anchor_channel_order_is_anchor_major():
    """The head convolution emits na*(5+nc) channels laid out anchor-major: all
    of anchor 0's attributes, then anchor 1's. Reading it attribute-major
    decodes every box from the wrong prior — plausible boxes, wrong sizes."""
    anchors = [(10.0, 10.0), (100.0, 100.0)]
    raw = np.zeros((2 * 6, 1, 1), np.float32)
    raw[4, 0, 0] = 10.0                                    # anchor 0 objectness
    raw[6 + 4, 0, 0] = -10.0                               # anchor 1 objectness
    boxes, scores, _ = ref_decode_anchor(raw, 8, anchors, 1)
    assert scores[0] > scores[1]
    assert boxes[0, 2] == pytest.approx(10.0)              # the small prior
    assert boxes[1, 2] == pytest.approx(100.0)


# --------------------------------------------------------------------------- #
# C++ cross-check                                                             #
# --------------------------------------------------------------------------- #
def test_matches_cxx_on_random_logits(cxx):
    """Random logits, no threshold and no suppression: every candidate must come
    back, and match the numpy formula box for box."""
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_yolov5_anchor")
    rng = np.random.default_rng(7)
    stride, h, w = 16, 5, 4
    pairs = [(3.0, 9.0), (5.0, 11.0), (4.0, 20.0)]
    raw = rng.normal(0, 1.5, (len(pairs) * 6, h, w)).astype(np.float32)
    lb = _identity_lb()

    dets = _decode(cxx, [raw], [(h, w)], [stride], [pairs], lb)
    boxes, scores, _ = ref_decode_anchor(raw, stride, pairs, 1)
    assert len(dets) == boxes.shape[0] == len(pairs) * h * w

    corners = ref_corners(boxes, lb)
    got = sorted(round(float(d.score), 5) for d in dets)
    want = sorted(round(float(s), 5) for s in scores)
    np.testing.assert_allclose(got, want, atol=1e-4)

    by_score = {round(float(d.score), 5): (d.x1, d.y1, d.x2, d.y2) for d in dets}
    for c, s in zip(corners, scores):
        g = by_score[round(float(s), 5)]
        np.testing.assert_allclose(g, c, atol=2e-3)


def test_cxx_pools_every_scale(cxx):
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_yolov5_anchor")
    raws = [np.zeros((6, 8, 8), np.float32),
            np.zeros((6, 4, 4), np.float32),
            np.zeros((6, 2, 2), np.float32)]
    dets = _decode(cxx, raws, [(8, 8), (4, 4), (2, 2)], [8, 16, 32],
                   [[(3, 9)], [(7, 18)], [(19, 50)]], _identity_lb())
    assert len(dets) == 8 * 8 + 4 * 4 + 2 * 2


def test_cxx_applies_the_letterbox_inverse(cxx):
    """Boxes come back in ORIGINAL-image pixels. Letterboxing a 128x64 source
    into a 64x64 canvas halves the scale, so a model-space box maps back to
    twice its coordinates, offset by the padding."""
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_yolov5_anchor")
    raw = np.zeros((6, 1, 1), np.float32)
    lb_id = _identity_lb()
    lb = compute_letterbox(128, 64, 64, 64)               # scale 0.5, pad_y 16
    d_model = _decode(cxx, [raw], [(1, 1)], [8], [[(10.0, 20.0)]], lb_id)[0]
    d_orig = _decode(cxx, [raw], [(1, 1)], [8], [[(10.0, 20.0)]], lb)[0]
    assert d_orig.x1 == pytest.approx(lb.clamp_x(lb.inv_x(d_model.x1)), abs=1e-3)
    assert d_orig.y1 == pytest.approx(lb.clamp_y(lb.inv_y(d_model.y1)), abs=1e-3)


def test_cxx_threshold_and_nms(cxx):
    """Two heavily overlapping boxes of the same class: NMS keeps one. Of a
    different class: it keeps both."""
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_yolov5_anchor")
    lb = _identity_lb()
    nc = 2
    raw = np.zeros((2 * (5 + nc), 2, 2), np.float32)
    # Two anchors on the same cell, both confident, same class, near-identical
    # sizes -> one survives.
    for a in (0, 1):
        base = a * (5 + nc)
        raw[base + 4, 0, 0] = 10.0
        raw[base + 5, 0, 0] = 10.0
    both = _decode(cxx, [raw], [(2, 2)], [8], [[(40.0, 40.0), (41.0, 41.0)]], lb,
                   num_classes=nc, conf=0.5, iou=0.5)
    assert len(both) == 1

    raw[(5 + nc) + 5, 0, 0] = 0.0        # anchor 1 -> class 1 instead
    raw[(5 + nc) + 6, 0, 0] = 10.0
    split = _decode(cxx, [raw], [(2, 2)], [8], [[(40.0, 40.0), (41.0, 41.0)]], lb,
                    num_classes=nc, conf=0.5, iou=0.5)
    assert len(split) == 2
    assert {d.class_id for d in split} == {0, 1}


def test_cxx_rejects_inconsistent_calls(cxx):
    """The decoder indexes by grid/anchor/class counts, none of which numpy ties
    to the array sizes — an inconsistent call must raise, not read out of
    bounds."""
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_yolov5_anchor")
    lb = _identity_lb()
    tiny = np.zeros(16, np.float32)
    with pytest.raises(Exception):                        # buffer too small
        _decode(cxx, [tiny], [(80, 80)], [8], [[(3, 9)]], lb)
    with pytest.raises(Exception):                        # list lengths differ
        _decode(cxx, [np.zeros((6, 2, 2), np.float32)], [(2, 2)], [8, 16],
                [[(3, 9)]], lb)
    with pytest.raises(Exception):                        # float64 read as float32
        cxx.decode_yolov5_anchor([np.zeros((6, 2, 2), np.float64)], [(2, 2)], [8],
                                 [_anchors(cxx, [(3, 9)])], lb.as_tuple())


def test_cxx_rejects_a_config_that_cannot_describe_a_head(cxx):
    """One prior set per scale, and no empty set. These are caught up front
    rather than surfacing later as a confusing channel-count complaint."""
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_yolov5_anchor")
    lb = _identity_lb()
    raw = np.zeros((6, 2, 2), np.float32)
    with pytest.raises(Exception):                        # two strides, one prior set
        _decode(cxx, [raw, raw], [(2, 2), (2, 2)], [8, 16], [[(3, 9)]], lb)
    with pytest.raises(Exception):                        # a scale with no priors
        _decode(cxx, [raw], [(2, 2)], [8], [[]], lb)
    with pytest.raises(Exception):                        # ragged prior counts
        _decode(cxx, [raw, np.zeros((12, 2, 2), np.float32)], [(2, 2), (2, 2)],
                [8, 16], [[(3, 9)], [(7, 18), (6, 39)]], lb)


def test_cxx_rejects_ragged_anchor_counts(cxx):
    """Every scale must declare the same number of priors: the channel count
    per scale is na*(5+nc), so a ragged set means one scale is being read with
    the wrong stride between attributes."""
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_yolov5_anchor")
    raws = [np.zeros((6, 2, 2), np.float32), np.zeros((12, 2, 2), np.float32)]
    with pytest.raises(Exception):
        _decode(cxx, raws, [(2, 2), (2, 2)], [8, 16],
                [[(3, 9)], [(7, 18), (6, 39)]], _identity_lb())
