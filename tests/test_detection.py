"""Detection post-processing tests.

These are PURE-NUMPY tests of the decode + NMS reference — the oracle that the
C++ ``rcdl::decode`` / ``rcdl::decodeYoloLtrb`` / ``rcdl::nms`` mirror. They need
only numpy and run anywhere: no board, no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_detection.py

The module-level ``ref_*`` functions double as the documented "numpy path":
given a raw model-output tensor and a letterbox, they produce the same
Detections the C++ decoders do. When the compiled module is importable its
``decode`` / ``decode_yolo_ltrb`` / ``nms`` are exercised against the same
oracle, but the core assertions never depend on it.
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


def compute_letterbox(src_w, src_h, dst_w, dst_h, center_pad=True):
    scale = min(dst_w / src_w, dst_h / src_h)
    pad_x = (dst_w - src_w * scale) * 0.5 if center_pad else 0.0
    pad_y = (dst_h - src_h * scale) * 0.5 if center_pad else 0.0
    return Letterbox(scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h)


# --------------------------------------------------------------------------- #
# Reference decode + NMS oracle                                               #
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ref_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    uni = area_a + area_b - inter
    return inter / uni if uni > 0.0 else 0.0


def ref_nms(dets, iou_thresh, max_dets):
    """Per-class greedy NMS over dicts {x1,y1,x2,y2,score,class_id}.

    Mirrors rcdl::nms exactly, INCLUDING the early break once max_dets boxes are
    kept (so a box that a later-kept box would have suppressed can survive when
    the cap is hit first). Returns kept indices into ``dets``.
    """
    order = sorted(range(len(dets)), key=lambda i: dets[i]["score"], reverse=True)
    suppressed = [False] * len(dets)
    keep = []
    for oi, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        if max_dets > 0 and len(keep) >= max_dets:
            break
        for j in order[oi + 1:]:
            if suppressed[j] or dets[j]["class_id"] != dets[i]["class_id"]:
                continue
            a = (dets[i]["x1"], dets[i]["y1"], dets[i]["x2"], dets[i]["y2"])
            b = (dets[j]["x1"], dets[j]["y1"], dets[j]["x2"], dets[j]["y2"])
            if ref_iou(a, b) > iou_thresh:
                suppressed[j] = True
    return keep


def ref_decode(data, shape, lb, num_classes=80, conf=0.25, iou_t=0.45,
               max_dets=300, channels_first=True, apply_sigmoid=False,
               has_obj=False):
    """Fused single-tensor head — mirrors rcdl::decode."""
    attrs = (5 if has_obj else 4) + num_classes
    total = int(np.prod(shape))
    n = total // attrs
    flat = np.asarray(data, dtype=np.float32).reshape(-1)
    a = flat.reshape(attrs, n) if channels_first else flat.reshape(n, attrs).T

    cls_start = 5 if has_obj else 4
    cls = a[cls_start:cls_start + num_classes]
    best_k = cls.argmax(axis=0)
    best_raw = cls.max(axis=0)
    score = _sigmoid(best_raw) if apply_sigmoid else best_raw
    if has_obj:
        obj = _sigmoid(a[4]) if apply_sigmoid else a[4]
        score = obj * score

    cx, cy, w, h = a[0], a[1], a[2], a[3]
    dets = []
    for j in np.nonzero(score >= conf)[0]:
        mx1, my1 = cx[j] - w[j] * 0.5, cy[j] - h[j] * 0.5
        mx2, my2 = cx[j] + w[j] * 0.5, cy[j] + h[j] * 0.5
        dets.append({"x1": lb.clamp_x(lb.inv_x(mx1)), "y1": lb.clamp_y(lb.inv_y(my1)),
                     "x2": lb.clamp_x(lb.inv_x(mx2)), "y2": lb.clamp_y(lb.inv_y(my2)),
                     "score": float(score[j]), "class_id": int(best_k[j])})
    return [dets[i] for i in ref_nms(dets, iou_t, max_dets)]


def ref_dfl(box, reg_max):
    """Σ b·softmax(b) over each side's reg_max bins. ``box`` is [4*reg_max, ...]."""
    x = box.reshape(4, reg_max, -1).astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=1, keepdims=True)
    return (p * np.arange(reg_max).reshape(1, reg_max, 1)).sum(axis=1)  # [4, cells]


def ref_decode_ltrb(cls_list, box_list, grids, strides, lb, num_classes=80,
                    conf=0.25, iou_t=0.45, max_dets=300, reg_max=0,
                    channels_first=True, apply_sigmoid=False):
    """Anchor-free LTRB multi-scale head — mirrors rcdl::decodeYoloLtrb."""
    dets = []
    for cls, box, (h, w), stride in zip(cls_list, box_list, grids, strides):
        cells = h * w
        box_ch = 4 * reg_max if reg_max > 0 else 4
        if channels_first:
            c = np.asarray(cls, dtype=np.float32).reshape(num_classes, cells)
            b = np.asarray(box, dtype=np.float32).reshape(box_ch, cells)
        else:
            c = np.asarray(cls, dtype=np.float32).reshape(cells, num_classes).T
            b = np.asarray(box, dtype=np.float32).reshape(cells, box_ch).T

        best_k = c.argmax(axis=0)
        best_raw = c.max(axis=0)
        score = _sigmoid(best_raw) if apply_sigmoid else best_raw
        d = ref_dfl(b, reg_max) if reg_max > 0 else b.astype(np.float64)

        gy, gx = np.divmod(np.arange(cells), w)
        cx, cy = gx + 0.5, gy + 0.5
        mx1, my1 = (cx - d[0]) * stride, (cy - d[1]) * stride
        mx2, my2 = (cx + d[2]) * stride, (cy + d[3]) * stride
        for j in np.nonzero(score >= conf)[0]:
            dets.append({"x1": lb.clamp_x(lb.inv_x(mx1[j])), "y1": lb.clamp_y(lb.inv_y(my1[j])),
                         "x2": lb.clamp_x(lb.inv_x(mx2[j])), "y2": lb.clamp_y(lb.inv_y(my2[j])),
                         "score": float(score[j]), "class_id": int(best_k[j])})
    return [dets[i] for i in ref_nms(dets, iou_t, max_dets)]


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
    return rcdl if hasattr(rcdl, "decode") else None


def _same(cxx_dets, ref_dets, tol=1e-3):
    assert len(cxx_dets) == len(ref_dets), f"{len(cxx_dets)} vs {len(ref_dets)} detections"
    for a, b in zip(cxx_dets, ref_dets):
        assert a.class_id == b["class_id"]
        assert abs(a.score - b["score"]) < tol
        for f in ("x1", "y1", "x2", "y2"):
            assert abs(getattr(a, f) - b[f]) < tol, f"{f}: {getattr(a, f)} vs {b[f]}"


# --------------------------------------------------------------------------- #
# Geometry                                                                     #
# --------------------------------------------------------------------------- #
def test_letterbox_geometry_roundtrip():
    lb = compute_letterbox(1280, 720, 640, 640)
    assert lb.scale == pytest.approx(0.5)
    assert lb.pad_x == pytest.approx(0.0)
    assert lb.pad_y == pytest.approx(140.0)
    # forward then inverse is the identity inside the image
    for x, y in ((0, 0), (640, 360), (1279, 719)):
        assert lb.inv_x(x * lb.scale + lb.pad_x) == pytest.approx(x)
        assert lb.inv_y(y * lb.scale + lb.pad_y) == pytest.approx(y)


def test_letterbox_clamps_padding_region():
    lb = compute_letterbox(1280, 720, 640, 640)
    # a box in the grey band above the image maps to a negative y, clamped to 0
    assert lb.clamp_y(lb.inv_y(10.0)) == 0.0
    assert lb.clamp_y(lb.inv_y(639.0)) == 720.0


# --------------------------------------------------------------------------- #
# IoU / NMS                                                                    #
# --------------------------------------------------------------------------- #
def test_iou_basics():
    assert ref_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert ref_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    # half-overlap: inter 50, union 150
    assert ref_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_nms_is_per_class():
    dets = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10, "score": 0.9, "class_id": 0},
        {"x1": 1, "y1": 1, "x2": 11, "y2": 11, "score": 0.8, "class_id": 0},  # suppressed
        {"x1": 1, "y1": 1, "x2": 11, "y2": 11, "score": 0.7, "class_id": 1},  # kept: other class
    ]
    keep = ref_nms(dets, 0.45, 300)
    assert keep == [0, 2]


def test_nms_matches_cxx(cxx):
    if cxx is None:
        pytest.skip("compiled rcdl module without decode bindings")
    rng = np.random.default_rng(7)
    dets = []
    for _ in range(200):
        x, y = rng.uniform(0, 500, 2)
        w, h = rng.uniform(10, 120, 2)
        dets.append({"x1": x, "y1": y, "x2": x + w, "y2": y + h,
                     "score": float(rng.uniform(0.3, 1.0)), "class_id": int(rng.integers(0, 4))})
    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"], d["score"], d["class_id"]]
                      for d in dets], dtype=np.float32)
    keep_ref = ref_nms(dets, 0.45, 300)
    keep_cxx = cxx.nms(boxes, 0.45, 300)
    assert list(keep_cxx) == keep_ref


# --------------------------------------------------------------------------- #
# Fused single-tensor head                                                     #
# --------------------------------------------------------------------------- #
def _one_box_tensor(n=100, nc=4, cx=320.0, cy=320.0, w=64.0, h=32.0, k=2,
                    logit=3.0, channels_first=True):
    attrs = 4 + nc
    a = np.full((attrs, n), -10.0, dtype=np.float32)
    a[0], a[1], a[2], a[3] = 1.0, 1.0, 1.0, 1.0
    a[0, 0], a[1, 0], a[2, 0], a[3, 0] = cx, cy, w, h
    a[4 + k, 0] = logit
    return a if channels_first else a.T


def test_decode_single_box_maps_back_to_source_pixels():
    lb = compute_letterbox(1280, 720, 640, 640)
    t = _one_box_tensor()
    got = ref_decode(t, t.shape, lb, num_classes=4, conf=0.25, apply_sigmoid=True)
    assert len(got) == 1
    d = got[0]
    assert d["class_id"] == 2
    assert d["score"] == pytest.approx(1 / (1 + np.exp(-3.0)), abs=1e-6)
    # box centre (320,320) in the canvas -> ((320-0)/0.5, (320-140)/0.5) = (640, 360)
    assert (d["x1"] + d["x2"]) / 2 == pytest.approx(640.0, abs=1e-3)
    assert (d["y1"] + d["y2"]) / 2 == pytest.approx(360.0, abs=1e-3)
    assert d["x2"] - d["x1"] == pytest.approx(64.0 / 0.5, abs=1e-3)


@pytest.mark.parametrize("channels_first", [True, False])
def test_decode_layout_invariance(channels_first):
    lb = compute_letterbox(640, 480, 640, 640)
    t = _one_box_tensor(channels_first=channels_first)
    got = ref_decode(t, t.shape, lb, num_classes=4, apply_sigmoid=True,
                     channels_first=channels_first)
    assert len(got) == 1 and got[0]["class_id"] == 2


def test_decode_matches_cxx(cxx):
    if cxx is None:
        pytest.skip("compiled rcdl module without decode bindings")
    lb = compute_letterbox(1280, 720, 640, 640)
    rng = np.random.default_rng(3)
    nc, n = 6, 512
    t = rng.standard_normal((4 + nc, n)).astype(np.float32)
    t[:4] = rng.uniform(20, 600, size=(4, n)).astype(np.float32)
    ref = ref_decode(t, t.shape, lb, num_classes=nc, apply_sigmoid=True)
    got = cxx.decode(np.ascontiguousarray(t),
                     (lb.scale, lb.pad_x, lb.pad_y, lb.src_w, lb.src_h, lb.dst_w, lb.dst_h),
                     num_classes=nc, apply_sigmoid=True, channels_first=True)
    _same(got, ref)


# --------------------------------------------------------------------------- #
# LTRB multi-scale head (plain and DFL, both channel orders)                    #
# --------------------------------------------------------------------------- #
def _ltrb_scene(reg_max=0, channels_first=True, nc=4, seed=11):
    """One synthetic 3-scale head with a handful of above-threshold cells."""
    rng = np.random.default_rng(seed)
    grids = [(80, 80), (40, 40), (20, 20)]
    strides = [8, 16, 32]
    cls_list, box_list = [], []
    box_ch = 4 * reg_max if reg_max > 0 else 4
    for (h, w) in grids:
        cells = h * w
        c = np.full((nc, cells), -8.0, dtype=np.float32)
        # light up a few cells with a clear winner class
        for j in rng.choice(cells, size=5, replace=False):
            c[int(rng.integers(0, nc)), j] = float(rng.uniform(1.0, 4.0))
        if reg_max > 0:
            b = rng.standard_normal((box_ch, cells)).astype(np.float32)
        else:
            b = rng.uniform(0.5, 4.0, size=(box_ch, cells)).astype(np.float32)
        cls_list.append(c if channels_first else np.ascontiguousarray(c.T))
        box_list.append(b if channels_first else np.ascontiguousarray(b.T))
    return cls_list, box_list, grids, strides


@pytest.mark.parametrize("reg_max", [0, 16])
@pytest.mark.parametrize("channels_first", [True, False])
def test_ltrb_decode_runs_and_is_layout_invariant(reg_max, channels_first):
    lb = compute_letterbox(1280, 720, 640, 640)
    cf = _ltrb_scene(reg_max=reg_max, channels_first=True)
    nf = _ltrb_scene(reg_max=reg_max, channels_first=False)
    a = ref_decode_ltrb(*cf[:2], cf[2], cf[3], lb, num_classes=4, reg_max=reg_max,
                        channels_first=True, apply_sigmoid=True)
    b = ref_decode_ltrb(*nf[:2], nf[2], nf[3], lb, num_classes=4, reg_max=reg_max,
                        channels_first=False, apply_sigmoid=True)
    assert len(a) > 0
    # the same scene expressed in either channel order must decode identically
    assert len(a) == len(b)
    for u, v in zip(a, b):
        assert u["class_id"] == v["class_id"]
        assert u["score"] == pytest.approx(v["score"], abs=1e-6)
        assert u["x1"] == pytest.approx(v["x1"], abs=1e-4)


def test_dfl_reduction_matches_expectation():
    """A one-hot DFL distribution reduces to that bin's index."""
    reg_max = 16
    b = np.full((4 * reg_max, 1), -20.0, dtype=np.float32)
    for side, bin_idx in enumerate((0, 3, 7, 15)):
        b[side * reg_max + bin_idx, 0] = 20.0
    d = ref_dfl(b, reg_max)
    np.testing.assert_allclose(d[:, 0], [0, 3, 7, 15], atol=1e-4)


def test_ltrb_box_geometry_is_cell_centred():
    """A single cell with LTRB=(1,1,1,1) is a 2*stride box centred on the cell."""
    nc, h, w, stride = 2, 4, 4, 16
    cells = h * w
    cls = np.full((nc, cells), -20.0, dtype=np.float32)
    # An interior cell: a box at the edge would be clamped to the source extent
    # by the letterbox inverse, which is correct but would hide the geometry.
    cell = 1 * w + 1  # gy=1, gx=1
    cls[1, cell] = 20.0
    box = np.ones((4, cells), dtype=np.float32)
    lb = compute_letterbox(64, 64, 64, 64)  # identity: scale 1, no padding
    got = ref_decode_ltrb([cls], [box], [(h, w)], [stride], lb, num_classes=nc,
                          apply_sigmoid=True)
    assert len(got) == 1
    d = got[0]
    assert d["class_id"] == 1
    cx, cy = (1 + 0.5) * stride, (1 + 0.5) * stride
    assert d["x1"] == pytest.approx(cx - stride)
    assert d["y1"] == pytest.approx(cy - stride)
    assert d["x2"] == pytest.approx(cx + stride)
    assert d["y2"] == pytest.approx(cy + stride)


def test_ltrb_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "decode_yolo_ltrb"):
        pytest.skip("compiled rcdl module without decode_yolo_ltrb bindings")
    lb = compute_letterbox(1280, 720, 640, 640)
    cls_list, box_list, grids, strides = _ltrb_scene(reg_max=16, channels_first=True)
    ref = ref_decode_ltrb(cls_list, box_list, grids, strides, lb, num_classes=4,
                          reg_max=16, channels_first=True, apply_sigmoid=True)
    got = cxx.decode_yolo_ltrb(
        [np.ascontiguousarray(c) for c in cls_list],
        [np.ascontiguousarray(b) for b in box_list],
        grids, strides,
        (lb.scale, lb.pad_x, lb.pad_y, lb.src_w, lb.src_h, lb.dst_w, lb.dst_h),
        num_classes=4, reg_max=16, channels_first=True, apply_sigmoid=True)
    _same(got, ref)


# --------------------------------------------------------------------------- #
# Binding safety: the compiled decoders reinterpret raw buffers, so the         #
# bindings must reject anything whose dtype or size does not match.            #
# --------------------------------------------------------------------------- #
def test_bindings_reject_non_float32(cxx):
    if cxx is None:
        pytest.skip("compiled rcdl module without decode bindings")
    lb = (0.5, 0.0, 140.0, 1280, 720, 640, 640)
    # float64 is what numpy gives a plain Python list — reading it as float32
    # would silently produce nonsense rather than fail.
    with pytest.raises(Exception):
        cxx.nms(np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float64))
    with pytest.raises(Exception):
        cxx.decode(np.zeros((84, 100), dtype=np.float64), lb, num_classes=80)
    with pytest.raises(Exception):
        cxx.decode(np.zeros((84, 100), dtype=np.float16), lb, num_classes=80)


def test_decode_yolo_ltrb_rejects_undersized_buffers(cxx):
    """The decoder indexes by grid/class/reg_max, none of which numpy ties to the
    array sizes — an inconsistent call must raise, not read out of bounds."""
    if cxx is None or not hasattr(cxx, "decode_yolo_ltrb"):
        pytest.skip("compiled rcdl module without decode_yolo_ltrb bindings")
    lb = (0.5, 0.0, 140.0, 1280, 720, 640, 640)
    tiny = np.zeros(16, dtype=np.float32)
    with pytest.raises(Exception):
        cxx.decode_yolo_ltrb([tiny], [tiny], [(80, 80)], [8], lb, num_classes=80)
    with pytest.raises(Exception):  # mismatched list lengths
        cxx.decode_yolo_ltrb([tiny, tiny], [tiny], [(2, 2)], [8], lb, num_classes=1)


def test_nms_rejects_wrong_shape(cxx):
    if cxx is None:
        pytest.skip("compiled rcdl module without decode bindings")
    with pytest.raises(Exception):
        cxx.nms(np.zeros((10, 5), dtype=np.float32))
