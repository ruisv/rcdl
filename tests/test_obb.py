"""Oriented-bounding-box post-processing tests.

These are PURE-NUMPY tests of the OBB decode + rotated-IoU reference — the
oracle that the C++ ``rcdl::decodeObb`` / ``rcdl::rotatedIoU`` /
``rcdl::rotatedNms`` mirror. They need only numpy and run anywhere: no board,
no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_obb.py

Rotated IoU is the one piece of this head that axis-aligned detection has no
answer for, so it is pinned against hand-computed values rather than only
against itself: identical boxes, a square rotated by a quarter turn, disjoint
boxes, and two overlaps whose areas are worked out on paper below.

Decode contract mirrored here (src/tasks/obb.cc decodeObb):
  l,t,r,b = |LTRB|;  a = (angle_value - angle_bias) * pi
  xf = (r-l)/2, yf = (b-t)/2                  (centre offset in the BOX frame)
  cx = (gx+0.5 + xf*cos a - yf*sin a) * S     (rotated into the image frame)
  cy = (gy+0.5 + xf*sin a + yf*cos a) * S
  w  = (l+r)*S,  h = (t+b)*S;  if regularize and w<h: swap(w,h), a += pi/2
  score = sigmoid(max cls logit); then rotated per-class NMS, then un-letterbox.
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

    def as_tuple(self):
        return (self.scale, self.pad_x, self.pad_y, self.src_w, self.src_h,
                self.dst_w, self.dst_h)


def compute_letterbox(src_w, src_h, dst_w, dst_h, center_pad=True):
    scale = min(dst_w / src_w, dst_h / src_h)
    pad_x = (dst_w - src_w * scale) * 0.5 if center_pad else 0.0
    pad_y = (dst_h - src_h * scale) * 0.5 if center_pad else 0.0
    return Letterbox(scale, pad_x, pad_y, src_w, src_h, dst_w, dst_h)


# --------------------------------------------------------------------------- #
# Rotated-rectangle geometry oracle                                            #
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ref_corners(box):
    """(cx,cy,w,h,angle) -> 4 corners — mirrors rcdl::rotatedBoxCorners."""
    cx, cy, w, h, a = box
    c, s = math.cos(a), math.sin(a)
    dx, dy = w / 2.0, h / 2.0
    local = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]
    return [(cx + lx * c - ly * s, cy + lx * s + ly * c) for lx, ly in local]


def ref_polygon_area(poly):
    """Shoelace area, absolute so the winding does not matter."""
    if len(poly) < 3:
        return 0.0
    acc = 0.0
    for i in range(len(poly)):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % len(poly)]
        acc += ax * by - bx * ay
    return abs(acc) * 0.5


def _cross(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _edge_len_sq(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2


# A point within this many edge-lengths of a clip line counts as ON it, hence
# inside. Without the slack, coincident edges — identical boxes, or two boxes the
# head predicted at the same angle, which is the everyday case — resolve to a
# random side per vertex and the clipper emits a self-crossing polygon.
ON_LINE_TOL = 1e-9


def _line_intersect(p1, p2, a, b):
    a1, b1 = b[1] - a[1], a[0] - b[0]
    c1 = a1 * a[0] + b1 * a[1]
    a2, b2 = p2[1] - p1[1], p1[0] - p2[0]
    c2 = a2 * p1[0] + b2 * p1[1]
    det = a1 * b2 - a2 * b1
    if abs(det) <= 1e-12 * (abs(a1 * b2) + abs(a2 * b1) + 1.0):
        # (Near-)parallel: no meaningful crossing, so keep whichever endpoint is
        # already closest to the line. Returning the far one would push a vertex
        # outside the clip region and inflate the intersection area.
        return p1 if abs(_cross(a, b, p1)) <= abs(_cross(a, b, p2)) else p2
    return ((b2 * c1 - b1 * c2) / det, (a1 * c2 - a2 * c1) / det)


def ref_clip_polygon(subject, clip):
    """Sutherland-Hodgman convex clip — mirrors the C++ helper, including taking
    the 'inside' sign from the clip quad's own winding and the on-line slack."""
    inside_sign = 1.0 if _cross(clip[0], clip[1], clip[2]) >= 0.0 else -1.0
    output = list(subject)
    for e in range(4):
        if not output:
            break
        a, b = clip[e], clip[(e + 1) % 4]
        tol = ON_LINE_TOL * _edge_len_sq(a, b)
        inp, output = output, []
        n = len(inp)
        for i in range(n):
            cur, prv = inp[i], inp[(i + n - 1) % n]
            cur_in = _cross(a, b, cur) * inside_sign >= -tol
            prv_in = _cross(a, b, prv) * inside_sign >= -tol
            if cur_in:
                if not prv_in:
                    output.append(_line_intersect(prv, cur, a, b))
                output.append(cur)
            elif prv_in:
                output.append(_line_intersect(prv, cur, a, b))
    return output


def ref_rotated_iou(a, b):
    """Rotated-rect IoU — mirrors rcdl::rotatedIoU."""
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter = ref_polygon_area(ref_clip_polygon(ref_corners(a), ref_corners(b)))
    if inter <= 0.0:
        return 0.0
    uni = area_a + area_b - inter
    return inter / uni if uni > 0.0 else 0.0


def ref_rotated_nms(dets, iou_thresh, max_dets):
    """Per-class greedy rotated NMS — mirrors rcdl::rotatedNms."""
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
            if ref_rotated_iou(dets[i]["rrect"], dets[j]["rrect"]) > iou_thresh:
                suppressed[j] = True
    return keep


def ref_dfl(box, reg_max):
    """Σ b·softmax(b) over each side's reg_max bins. ``box`` is [4*reg_max, cells]."""
    x = box.reshape(4, reg_max, -1).astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=1, keepdims=True)
    return (p * np.arange(reg_max).reshape(1, reg_max, 1)).sum(axis=1)  # [4, cells]


def ref_decode_obb(cls_list, box_list, angle_list, grids, strides, lb, *,
                   num_classes=15, conf=0.25, iou_t=0.4, max_dets=300, reg_max=0,
                   channels_first=True, apply_sigmoid=True,
                   apply_angle_sigmoid=False, angle_bias=0.25, regularize=True):
    """Anchor-free LTRB OBB head — mirrors rcdl::decodeObb."""
    dets = []
    for cls, box, ang, (h, w), stride in zip(cls_list, box_list, angle_list,
                                             grids, strides):
        cells = h * w
        box_ch = 4 * reg_max if reg_max > 0 else 4
        if channels_first:
            c = np.asarray(cls, dtype=np.float32).reshape(num_classes, cells)
            b = np.asarray(box, dtype=np.float32).reshape(box_ch, cells)
        else:
            c = np.asarray(cls, dtype=np.float32).reshape(cells, num_classes).T
            b = np.asarray(box, dtype=np.float32).reshape(cells, box_ch).T
        # One channel is contiguous per cell in EVERY layout, so the angle needs
        # no channel order — just this scale's window into the tensor.
        a_raw = np.asarray(ang, dtype=np.float32).reshape(-1)[:cells]

        best_k = c.argmax(axis=0)
        best_raw = c.max(axis=0)
        score = _sigmoid(best_raw) if apply_sigmoid else best_raw
        d = np.abs(ref_dfl(b, reg_max) if reg_max > 0 else b.astype(np.float64))

        gy, gx = np.divmod(np.arange(cells), w)
        for j in np.nonzero(score >= conf)[0]:
            l, t, r, bot = d[0][j], d[1][j], d[2][j], d[3][j]
            act = _sigmoid(a_raw[j]) if apply_angle_sigmoid else a_raw[j]
            a = (act - angle_bias) * math.pi
            xf, yf = (r - l) / 2.0, (bot - t) / 2.0
            ca, sa = math.cos(a), math.sin(a)
            cx = (gx[j] + 0.5 + xf * ca - yf * sa) * stride
            cy = (gy[j] + 0.5 + xf * sa + yf * ca) * stride
            bw, bh = (l + r) * stride, (t + bot) * stride
            if regularize and bw < bh:
                bw, bh = bh, bw
                a += math.pi / 2.0
            dets.append({"rrect": (cx, cy, bw, bh, a), "score": float(score[j]),
                         "class_id": int(best_k[j])})

    out = []
    inv = 1.0 / lb.scale if lb.scale > 0.0 else 1.0
    for i in ref_rotated_nms(dets, iou_t, max_dets):
        cx, cy, bw, bh, a = dets[i]["rrect"]
        # Un-letterbox: the centre maps back affinely, the sides divide by the
        # uniform scale, and the angle is invariant under that map.
        out.append({"rrect": (lb.inv_x(cx), lb.inv_y(cy), bw * inv, bh * inv, a),
                    "score": dets[i]["score"], "class_id": dets[i]["class_id"]})
    return out


# --------------------------------------------------------------------------- #
# DOTA-15 — the same table src/tasks/obb.cc ships                              #
# --------------------------------------------------------------------------- #
DOTA_CLASSES = [
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge",
    "large vehicle", "small vehicle", "helicopter", "roundabout",
    "soccer ball field", "swimming pool",
]


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
# Rotated IoU — pinned against hand-computed areas                             #
# --------------------------------------------------------------------------- #
def test_rotated_iou_identical_boxes_is_one():
    """Also the sharpest test of the clipper's degenerate handling: every edge of
    the subject lies exactly ON a clip line, so a naive strict inside test drops
    vertices and the shoelace area of the resulting self-crossing polygon can come
    out several times the box area."""
    box = (10.0, 20.0, 8.0, 4.0, 0.7)
    assert ref_rotated_iou(box, box) == pytest.approx(1.0, abs=1e-5)


def test_rotated_iou_square_turned_by_ninety_degrees_is_one():
    """A square is invariant under a quarter turn, so its IoU with itself
    rotated must still be exactly 1 — the check that catches a decoder that
    confuses w with h, or an IoU that compares axis-aligned extents."""
    a = (0.0, 0.0, 6.0, 6.0, 0.0)
    b = (0.0, 0.0, 6.0, 6.0, math.pi / 2.0)
    assert ref_rotated_iou(a, b) == pytest.approx(1.0, abs=1e-5)
    # ... and the same box at any angle keeps its area
    assert ref_polygon_area(ref_corners((0.0, 0.0, 6.0, 6.0, 0.37))) == pytest.approx(36.0)


def test_rotated_iou_of_disjoint_boxes_is_zero():
    a = (0.0, 0.0, 4.0, 4.0, 0.3)
    b = (100.0, 100.0, 4.0, 4.0, -0.9)
    assert ref_rotated_iou(a, b) == 0.0
    # touching along one edge is still zero area
    assert ref_rotated_iou((0.0, 0.0, 4.0, 4.0, 0.0),
                           (4.0, 0.0, 4.0, 4.0, 0.0)) == pytest.approx(0.0, abs=1e-6)


def test_rotated_iou_axis_aligned_half_overlap():
    """Two 10x10 boxes offset by 5 on x: intersection 5*10 = 50, union
    100 + 100 - 50 = 150, so IoU = 1/3."""
    a = (0.0, 0.0, 10.0, 10.0, 0.0)
    b = (5.0, 0.0, 10.0, 10.0, 0.0)
    assert ref_rotated_iou(a, b) == pytest.approx(50.0 / 150.0, abs=1e-5)


def test_rotated_iou_square_against_the_same_square_turned_45_degrees():
    """Both squares have side 2, so both have an inradius of 1 and their
    intersection is the regular octagon cut by all eight edges: area
    8*tan(pi/8) = 8*(sqrt(2)-1), union 8 - 8*(sqrt(2)-1) = 16 - 8*sqrt(2), and
    the ratio collapses to exactly 1/sqrt(2)."""
    a = (0.0, 0.0, 2.0, 2.0, 0.0)
    b = (0.0, 0.0, 2.0, 2.0, math.pi / 4.0)
    inter = ref_polygon_area(ref_clip_polygon(ref_corners(a), ref_corners(b)))
    assert inter == pytest.approx(8.0 * (math.sqrt(2.0) - 1.0), abs=1e-5)
    assert ref_rotated_iou(a, b) == pytest.approx(1.0 / math.sqrt(2.0), abs=1e-5)


def test_rotated_iou_two_parallel_boxes_slid_along_their_own_axis():
    """A 10x4 box and a copy slid 5 along its OWN long axis at 30 degrees:
    overlap 5*4 = 20, union 40 + 40 - 20 = 60, IoU = 1/3 — the same answer the
    axis-aligned case gives, which an extent-based IoU would get wrong. The two
    boxes share their long edges, so this is the coincident-edge case again — the
    one every cell of a real detection produces against its neighbours."""
    a_rad = math.pi / 6.0
    a = (0.0, 0.0, 10.0, 4.0, a_rad)
    b = (5.0 * math.cos(a_rad), 5.0 * math.sin(a_rad), 10.0, 4.0, a_rad)
    assert ref_rotated_iou(a, b) == pytest.approx(20.0 / 60.0, abs=1e-5)


def test_rotated_iou_of_a_degenerate_box_is_zero():
    assert ref_rotated_iou((0.0, 0.0, 0.0, 5.0, 0.0), (0.0, 0.0, 5.0, 5.0, 0.0)) == 0.0


# --------------------------------------------------------------------------- #
# Regression: rotated IoU on COINCIDENT EDGES                                  #
#                                                                              #
# Two boxes that share an edge are the everyday case — a box against itself     #
# during NMS, or two cells of one head predicting the same object at the same   #
# angle — and they are the only case where the polygon clip is delicate: every  #
# vertex of the shared edge has a signed distance of zero, so its side of the   #
# clip line is decided purely by rounding. Measured on the compiled decoder,    #
# clipping in float32:                                                          #
#                                                                              #
#     rotatedIoU(box, box) for box = (10,20,8,4,0.7 rad)  ->  -39.6            #
#     two 10x4 boxes slid 5 along their shared 30-degree axis  ->  0.247       #
#                                                                              #
# (the exact garbage is platform-dependent — the same build on another host     #
# returned 11.8 for the first one — because it is rounding noise being fed      #
# through a shoelace sum of a self-crossing polygon.) The same pairs in general #
# position are fine in float32, which is why a randomised cross-check over      #
# boxes drawn from a uniform distribution never trips it: the failure needs     #
# collinearity, and random boxes are never collinear.                           #
#                                                                              #
# The fix is to run the clip in DOUBLE. An on-line tolerance and a             #
# near-parallel fallback that keeps the nearer endpoint are also in place, but  #
# they are hardening, not the fix: with float32 arithmetic and both of those    #
# added, the first case still returned 1.88.                                    #
#                                                                              #
# The oracle below is float64 and so cannot itself reproduce a float32          #
# regression. What it can do is state the contract these pairs must satisfy;    #
# test_rotated_iou_matches_cxx runs the SAME table through the compiled         #
# implementation, which is where a precision regression would actually show.    #
# --------------------------------------------------------------------------- #
DEGENERATE_PAIRS = [
    # (name, box a, box b, expected IoU) — every pair shares at least one edge.
    ("identical, rotated", (10.0, 20.0, 8.0, 4.0, 0.7),
     (10.0, 20.0, 8.0, 4.0, 0.7), 1.0),
    ("identical, axis aligned", (0.0, 0.0, 10.0, 10.0, 0.0),
     (0.0, 0.0, 10.0, 10.0, 0.0), 1.0),
    ("axis-aligned half overlap", (0.0, 0.0, 10.0, 10.0, 0.0),
     (5.0, 0.0, 10.0, 10.0, 0.0), 50.0 / 150.0),
    ("slid along a 30-degree axis", (0.0, 0.0, 10.0, 4.0, math.pi / 6.0),
     (5.0 * math.cos(math.pi / 6.0), 5.0 * math.sin(math.pi / 6.0),
      10.0, 4.0, math.pi / 6.0), 20.0 / 60.0),
    ("edge to edge, no overlap", (0.0, 0.0, 4.0, 4.0, 0.0),
     (4.0, 0.0, 4.0, 4.0, 0.0), 0.0),
    ("one contained in the other", (0.0, 0.0, 10.0, 10.0, 0.3),
     (0.0, 0.0, 5.0, 10.0, 0.3), 0.5),
]


@pytest.mark.parametrize("name,a,b,want", DEGENERATE_PAIRS,
                         ids=[pair[0] for pair in DEGENERATE_PAIRS])
def test_rotated_iou_on_coincident_edges(name, a, b, want):
    assert ref_rotated_iou(a, b) == pytest.approx(want, abs=1e-5)
    assert ref_rotated_iou(b, a) == pytest.approx(want, abs=1e-5), "IoU must be symmetric"


def test_rotated_iou_stays_within_zero_and_one():
    """The invariant the float32 clip broke loudest: the area of a self-crossing
    polygon is unrelated to the area of either box, so the ratio leaves [0,1]
    entirely (and can even go negative once the shoelace sum changes sign).
    Sweeps general position AND, for every box, a copy of itself plus a copy
    sharing its angle — the collinear pairs are the ones that matter."""
    rng = np.random.default_rng(31)
    boxes = [(float(rng.uniform(-20, 20)), float(rng.uniform(-20, 20)),
              float(rng.uniform(1, 30)), float(rng.uniform(1, 30)),
              float(rng.uniform(-math.pi, math.pi))) for _ in range(60)]
    pairs = [(a, a) for a in boxes]
    pairs += [(a, (a[0] + a[2] / 2.0, a[1], a[2], a[3], a[4])) for a in boxes]
    pairs += list(zip(boxes, boxes[1:] + boxes[:1]))
    # An identical pair computes inter/(2A - inter) with inter equal to A only up
    # to the last ulp, so the ratio can land a few 1e-16 above 1. The failure this
    # guards against is off by tens, not by an ulp, so allow the ulp.
    eps = 1e-9
    for a, b in pairs:
        v = ref_rotated_iou(a, b)
        assert -eps <= v <= 1.0 + eps, f"IoU {v} outside [0,1] for {a} vs {b}"
        assert v == pytest.approx(ref_rotated_iou(b, a), abs=1e-9), "IoU must be symmetric"


@pytest.mark.parametrize("angle", [0.0, 0.3, 0.7, 1.0, math.pi / 4.0, math.pi / 2.0,
                                   2.0, -0.7, -math.pi / 3.0, 3.0])
def test_rotated_iou_of_a_box_with_itself_is_one_at_every_angle(angle):
    """0.7 rad is the angle the float32 clip returned -39.6 for."""
    box = (10.0, 20.0, 8.0, 4.0, angle)
    assert ref_rotated_iou(box, box) == pytest.approx(1.0, abs=1e-6)


def test_rotated_nms_suppresses_an_exact_duplicate():
    """What the bug meant downstream: rotatedIoU is only ever consumed as
    `> iou_thresh`, and an IoU of -39.6 is NOT greater than 0.4, so a box that
    duplicated another exactly would have survived NMS."""
    box = (10.0, 20.0, 8.0, 4.0, 0.7)
    dets = [{"rrect": box, "score": 0.9, "class_id": 2},
            {"rrect": box, "score": 0.8, "class_id": 2}]
    assert ref_rotated_nms(dets, 0.4, 300) == [0]


@pytest.mark.parametrize("name,a,b,want", DEGENERATE_PAIRS,
                         ids=[pair[0] for pair in DEGENERATE_PAIRS])
def test_rotated_iou_on_coincident_edges_matches_cxx(name, a, b, want):
    """The one test that can catch a precision regression in the compiled clip.
    Skips until the bindings exist; the expectations are hand-computed, so it
    checks the implementation rather than agreement between two copies of it."""
    cxx = pytest.importorskip("rcdl")
    if not hasattr(cxx, "rotated_iou"):
        pytest.skip("compiled rcdl module without rotated_iou bindings")
    assert cxx.rotated_iou(a, b) == pytest.approx(want, abs=1e-4)
    assert cxx.rotated_iou(b, a) == pytest.approx(want, abs=1e-4)


def test_rotated_iou_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "rotated_iou"):
        pytest.skip("compiled rcdl module without rotated_iou bindings")
    rng = np.random.default_rng(19)
    for _ in range(200):
        a = tuple(float(v) for v in (rng.uniform(-20, 20), rng.uniform(-20, 20),
                                     rng.uniform(1, 30), rng.uniform(1, 30),
                                     rng.uniform(-math.pi, math.pi)))
        b = tuple(float(v) for v in (rng.uniform(-20, 20), rng.uniform(-20, 20),
                                     rng.uniform(1, 30), rng.uniform(1, 30),
                                     rng.uniform(-math.pi, math.pi)))
        assert cxx.rotated_iou(a, b) == pytest.approx(ref_rotated_iou(a, b), abs=1e-4)


# --------------------------------------------------------------------------- #
# Rotated NMS                                                                  #
# --------------------------------------------------------------------------- #
def test_rotated_nms_is_per_class():
    dets = [
        {"rrect": (0.0, 0.0, 10.0, 10.0, 0.0), "score": 0.9, "class_id": 0},
        {"rrect": (1.0, 1.0, 10.0, 10.0, 0.0), "score": 0.8, "class_id": 0},
        {"rrect": (1.0, 1.0, 10.0, 10.0, 0.0), "score": 0.7, "class_id": 1},
    ]
    assert ref_rotated_nms(dets, 0.4, 300) == [0, 2]


def test_rotated_nms_keeps_a_perpendicular_neighbour():
    """Two crossing 10x2 boxes overlap on only 4 of 20+20-4 = 36 units, i.e.
    IoU 1/9 — below the threshold, so both survive. Their axis-aligned extents
    would overlap completely and an axis-aligned NMS would drop one."""
    a = (0.0, 0.0, 10.0, 2.0, 0.0)
    b = (0.0, 0.0, 10.0, 2.0, math.pi / 2.0)
    assert ref_rotated_iou(a, b) == pytest.approx(4.0 / 36.0, abs=1e-5)
    dets = [{"rrect": a, "score": 0.9, "class_id": 3},
            {"rrect": b, "score": 0.8, "class_id": 3}]
    assert ref_rotated_nms(dets, 0.4, 300) == [0, 1]


def test_rotated_nms_honours_max_dets():
    dets = [{"rrect": (i * 100.0, 0.0, 5.0, 5.0, 0.0), "score": 1.0 - 0.01 * i,
             "class_id": 0} for i in range(5)]
    assert ref_rotated_nms(dets, 0.4, 2) == [0, 1]


# --------------------------------------------------------------------------- #
# Decode                                                                       #
# --------------------------------------------------------------------------- #
def _one_cell_scene(h=4, w=4, nc=3, gy=1, gx=1, cls_id=1, logit=10.0,
                    ltrb=(0.5, 0.25, 1.5, 0.75), angle_value=0.25,
                    channels_first=True):
    """A single grid with exactly one above-threshold cell."""
    cells = h * w
    cell = gy * w + gx
    cls = np.full((nc, cells), -20.0, dtype=np.float32)
    cls[cls_id, cell] = logit
    box = np.zeros((4, cells), dtype=np.float32)
    for side in range(4):
        box[side, cell] = ltrb[side]
    ang = np.full(cells, 0.25, dtype=np.float32)
    ang[cell] = angle_value
    if not channels_first:
        cls = np.ascontiguousarray(cls.T)
        box = np.ascontiguousarray(box.T)
    return cls, box, ang, cell


def test_decode_axis_aligned_cell_is_hand_computable():
    """angle_value 0.25 maps to 0 rad, so the rotation drops out:
    xf = (1.5-0.5)/2 = 0.5, yf = (0.75-0.25)/2 = 0.25,
    cx = (1.5+0.5)*16 = 32, cy = (1.5+0.25)*16 = 28,
    w = (0.5+1.5)*16 = 32, h = (0.25+0.75)*16 = 16."""
    h, w, stride, nc = 4, 4, 16, 3
    cls, box, ang, _ = _one_cell_scene(h, w, nc)
    lb = compute_letterbox(64, 64, 64, 64)  # identity
    got = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride], lb, num_classes=nc)
    assert len(got) == 1
    assert got[0]["class_id"] == 1
    assert got[0]["score"] == pytest.approx(_sigmoid(10.0), abs=1e-6)
    assert got[0]["rrect"] == pytest.approx((32.0, 28.0, 32.0, 16.0, 0.0), abs=1e-4)


def test_decode_rotates_the_centre_offset_into_the_image_frame():
    """The same cell at angle_value 0.75 (a = +pi/2): the box-frame offset
    (xf,yf) = (0.5,0.25) rotates to (-0.25, 0.5), so
    cx = (1.5-0.25)*16 = 20 and cy = (1.5+0.5)*16 = 32 — the sides are
    unchanged because they do not depend on the angle."""
    h, w, stride, nc = 4, 4, 16, 3
    cls, box, ang, _ = _one_cell_scene(h, w, nc, angle_value=0.75)
    lb = compute_letterbox(64, 64, 64, 64)
    got = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride], lb, num_classes=nc)
    assert got[0]["rrect"] == pytest.approx((20.0, 32.0, 32.0, 16.0, math.pi / 2.0),
                                            abs=1e-4)


@pytest.mark.parametrize("value,radians", [
    (0.0, -math.pi / 4.0),    # the low end of the exported range
    (0.25, 0.0),
    (0.5, math.pi / 4.0),
    (0.75, math.pi / 2.0),
    (1.0, 3.0 * math.pi / 4.0),  # the high end
])
def test_angle_parameterisation_is_the_exported_affine_map(value, radians):
    """The deployed export applies the angle sigmoid INSIDE the graph, so the
    tensor already holds a value in [0,1] and post-processing only maps it
    affinely: (v - 0.25) * pi, i.e. [-pi/4, 3*pi/4]."""
    h, w, stride, nc = 2, 2, 8, 2
    # A square box so `regularize` cannot fire and shift the angle by pi/2.
    cls, box, ang, _ = _one_cell_scene(h, w, nc, gy=0, gx=0, cls_id=0,
                                       ltrb=(0.5, 0.5, 0.5, 0.5),
                                       angle_value=value)
    lb = compute_letterbox(64, 64, 64, 64)
    got = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride], lb, num_classes=nc)
    assert got[0]["rrect"][4] == pytest.approx(radians, abs=1e-6)


def test_angle_sigmoid_can_be_moved_back_to_the_cpu():
    """A raw head that emits the logit instead: sigmoid(0) = 0.5 must give the
    same angle as a value of 0.5 straight from a graph that already did it."""
    h, w, stride, nc = 2, 2, 8, 2
    cls, box, ang, _ = _one_cell_scene(h, w, nc, gy=0, gx=0, cls_id=0,
                                       ltrb=(0.5, 0.5, 0.5, 0.5), angle_value=0.0)
    lb = compute_letterbox(64, 64, 64, 64)
    got = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride], lb,
                         num_classes=nc, apply_angle_sigmoid=True)
    assert got[0]["rrect"][4] == pytest.approx(math.pi / 4.0, abs=1e-6)


def test_regularize_canonicalises_to_a_wide_box():
    """w < h means the same physical box has two encodings; regularise swaps the
    sides and adds pi/2 so rotated NMS sees one shape, not two."""
    h, w, stride, nc = 1, 1, 8, 1
    cls = np.full((nc, 1), 10.0, dtype=np.float32)
    box = np.array([[0.25], [1.0], [0.25], [1.0]], dtype=np.float32)  # w=0.5S < h=2S
    ang = np.array([0.25], dtype=np.float32)  # a = 0
    lb = compute_letterbox(64, 64, 64, 64)
    got = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride], lb, num_classes=nc)
    assert got[0]["rrect"][:4] == pytest.approx((0.5 * stride, 0.5 * stride,
                                                 2.0 * stride, 0.5 * stride), abs=1e-4)
    assert got[0]["rrect"][4] == pytest.approx(math.pi / 2.0, abs=1e-6)

    off = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride], lb,
                         num_classes=nc, regularize=False)
    assert off[0]["rrect"][2:5] == pytest.approx((0.5 * stride, 2.0 * stride, 0.0),
                                                 abs=1e-4)
    # Both encodings describe the SAME rectangle, which is the whole point.
    assert ref_rotated_iou(got[0]["rrect"], off[0]["rrect"]) == pytest.approx(1.0, abs=1e-5)


def test_decode_un_letterboxes_centre_and_sides_but_not_the_angle():
    """1280x720 into 640x640 is scale 0.5 with 140px of top/bottom padding."""
    h, w, stride, nc = 4, 4, 160, 3  # 640 / 4 = 160
    cls, box, ang, _ = _one_cell_scene(h, w, nc, angle_value=0.5)
    lb = compute_letterbox(1280, 720, 640, 640)
    assert (lb.scale, lb.pad_x, lb.pad_y) == pytest.approx((0.5, 0.0, 140.0))
    canvas = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride],
                            compute_letterbox(640, 640, 640, 640), num_classes=nc)
    got = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride], lb, num_classes=nc)
    cx, cy, bw, bh, a = canvas[0]["rrect"]
    assert got[0]["rrect"] == pytest.approx(
        (lb.inv_x(cx), lb.inv_y(cy), bw / lb.scale, bh / lb.scale, a), abs=1e-3)


def test_decode_below_threshold_is_empty():
    h, w, nc = 2, 2, 3
    cls, box, ang, _ = _one_cell_scene(h, w, nc, logit=-10.0)
    lb = compute_letterbox(64, 64, 64, 64)
    assert ref_decode_obb([cls], [box], [ang], [(h, w)], [8], lb, num_classes=nc) == []


def test_decode_dfl_box_reduces_like_the_detection_head():
    reg_max, h, w, stride, nc = 16, 2, 2, 8, 2
    cells = h * w
    cell = 1 * w + 1
    cls = np.full((nc, cells), -20.0, dtype=np.float32)
    cls[0, cell] = 20.0
    box = np.full((4 * reg_max, cells), -20.0, dtype=np.float32)
    for side, bin_idx in enumerate((2, 1, 2, 1)):  # l=r=2, t=b=1 -> w=4S, h=2S
        box[side * reg_max + bin_idx, cell] = 20.0
    ang = np.full(cells, 0.25, dtype=np.float32)
    lb = compute_letterbox(1000, 1000, 1000, 1000)
    got = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride], lb,
                         num_classes=nc, reg_max=reg_max)
    assert len(got) == 1
    # symmetric distances -> the centre stays on the cell centre
    assert got[0]["rrect"] == pytest.approx((1.5 * stride, 1.5 * stride,
                                             4.0 * stride, 2.0 * stride, 0.0), abs=1e-3)


def test_channel_order_is_only_a_layout():
    """The same scene written [C,H,W] or [H,W,C] must decode identically — the
    flag comes from each output's rknn fmt, never from an assumption. The angle
    branch has a single channel, so it is byte-identical either way."""
    h, w, stride, nc = 4, 4, 16, 3
    cf = _one_cell_scene(h, w, nc, angle_value=0.6, channels_first=True)
    nf = _one_cell_scene(h, w, nc, angle_value=0.6, channels_first=False)
    lb = compute_letterbox(64, 64, 64, 64)
    a = ref_decode_obb([cf[0]], [cf[1]], [cf[2]], [(h, w)], [stride], lb, num_classes=nc)
    b = ref_decode_obb([nf[0]], [nf[1]], [nf[2]], [(h, w)], [stride], lb,
                       num_classes=nc, channels_first=False)
    assert len(a) == len(b) == 1
    assert a[0]["rrect"] == pytest.approx(b[0]["rrect"], abs=1e-6)
    assert a[0]["class_id"] == b[0]["class_id"]


def test_shared_angle_tensor_gives_each_scale_its_own_anchor_window():
    """The deployed export emits ONE [1,1,A] angle tensor for the whole model,
    anchors concatenated finest-scale-first; each scale reads its own window."""
    grids = [(2, 2), (1, 1)]
    strides = [8, 16]
    nc = 1
    shared = np.array([0.25, 0.25, 0.25, 0.25, 0.75], dtype=np.float32)
    offsets = [0, 4]
    cls_list, box_list, ang_list = [], [], []
    for (gh, gw), off in zip(grids, offsets):
        cells = gh * gw
        cls_list.append(np.full((nc, cells), 20.0, dtype=np.float32))
        # square boxes: regularize cannot fire, so the angle survives verbatim
        box_list.append(np.full((4, cells), 0.5, dtype=np.float32))
        ang_list.append(shared[off:off + cells])
    lb = compute_letterbox(100, 100, 100, 100)
    got = ref_decode_obb(cls_list, box_list, ang_list, grids, strides, lb,
                         num_classes=nc, iou_t=1.1)  # keep them all
    assert len(got) == 5
    angles = sorted(d["rrect"][4] for d in got)
    assert angles == pytest.approx([0.0, 0.0, 0.0, 0.0, math.pi / 2.0], abs=1e-6)


def test_decode_obb_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "decode_obb"):
        pytest.skip("compiled rcdl module without decode_obb bindings")
    h, w, stride, nc = 8, 8, 8, 4
    rng = np.random.default_rng(23)
    cells = h * w
    cls = rng.uniform(-6.0, 4.0, size=(nc, cells)).astype(np.float32)
    box = rng.uniform(0.2, 3.0, size=(4, cells)).astype(np.float32)
    ang = rng.uniform(0.0, 1.0, size=cells).astype(np.float32)
    lb = compute_letterbox(1280, 720, 64, 64)
    ref = ref_decode_obb([cls], [box], [ang], [(h, w)], [stride], lb, num_classes=nc)
    got = cxx.decode_obb([np.ascontiguousarray(cls)], [np.ascontiguousarray(box)],
                         [np.ascontiguousarray(ang)], [(h, w)], [stride],
                         lb.as_tuple(), num_classes=nc, channels_first=True)
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert a.class_id == b["class_id"]
        assert abs(a.score - b["score"]) < 1e-4
        for i, f in enumerate(("cx", "cy", "w", "h", "angle")):
            assert abs(getattr(a.rrect, f) - b["rrect"][i]) < 1e-3, f


# --------------------------------------------------------------------------- #
# DOTA-15 metadata                                                             #
# --------------------------------------------------------------------------- #
def test_dota_class_list_is_the_export_order():
    assert len(DOTA_CLASSES) == 15
    assert len(set(DOTA_CLASSES)) == 15
    assert DOTA_CLASSES[0] == "plane" and DOTA_CLASSES[-1] == "swimming pool"


def test_cxx_exposes_the_same_dota_table(cxx):
    if cxx is None or not hasattr(cxx, "dota_class_names"):
        pytest.skip("compiled rcdl module without OBB metadata bindings")
    assert list(cxx.dota_class_names()) == DOTA_CLASSES
