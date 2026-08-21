"""Pose post-processing tests.

These are PURE-NUMPY tests of the pose decode reference — the oracle that the
C++ ``rcdl::decodePose`` mirrors. They need only numpy and run anywhere: no
board, no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_pose.py

The module-level ``ref_*`` functions double as the documented "numpy path":
given the raw per-scale tensors and a letterbox, they produce the same
PoseDetections the C++ decoder does. When the compiled module exposes the
matching entry points they are exercised against the same oracle, but the core
assertions never depend on it.

Decode contract mirrored here (src/tasks/pose.cc decodePose):
  score = sigmoid(max cls logit) when apply_sigmoid, else the raw value
  box   = LTRB about the cell centre, exactly as the detection head:
          x1 = (gx+0.5 - l)*S ... y2 = (gy+0.5 + b)*S, then un-letterboxed
  kpt   = the tensor's own pixels ("model_pixels", what the deployed export
          emits), or (2*raw + grid)*S ("cell_relative", a raw head), then
          un-letterboxed; visibility is sigmoided only for a raw head
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


# --------------------------------------------------------------------------- #
# Reference decode oracle                                                      #
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
    """Per-class greedy NMS over the pose boxes — mirrors rcdl::nms, including
    the early break once max_dets boxes are kept."""
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


def ref_dfl(box, reg_max):
    """Σ b·softmax(b) over each side's reg_max bins. ``box`` is [4*reg_max, cells]."""
    x = box.reshape(4, reg_max, -1).astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=1, keepdims=True)
    return (p * np.arange(reg_max).reshape(1, reg_max, 1)).sum(axis=1)  # [4, cells]


class KeypointPlane:
    """One scale's window into a keypoint tensor — mirrors rcdl::KeypointPlane.

    ``data`` is the flat tensor, ``offset`` this scale's first element, and the
    two strides say where a cell's channels live. The three real layouts are

        shared    [1,K*3,A] : offset = anchors before this scale, cell 1, chan A
        per-scale [K*3,H,W] : offset 0, cell 1,   chan H*W
        per-scale [H,W,K*3] : offset 0, cell K*3, chan 1
    """

    def __init__(self, data, offset, cell_step, chan_step):
        self.data = np.asarray(data, dtype=np.float32).reshape(-1)
        self.offset, self.cell_step, self.chan_step = offset, cell_step, chan_step

    def at(self, cell, channel):
        return float(self.data[self.offset + cell * self.cell_step
                               + channel * self.chan_step])


def per_scale_plane(arr, h, w, num_keypoints, channels_first):
    """Plane for a per-scale [K*3,H,W] / [H,W,K*3] keypoint tensor."""
    kch = num_keypoints * 3
    if channels_first:
        return KeypointPlane(arr, 0, 1, h * w)
    return KeypointPlane(arr, 0, kch, 1)


def shared_planes(arr, grids):
    """Planes for one shared [1,K*3,A] tensor covering every scale's anchors."""
    total = sum(h * w for h, w in grids)
    planes, offset = [], 0
    for h, w in grids:
        planes.append(KeypointPlane(arr, offset, 1, total))
        offset += h * w
    return planes


def ref_decode_pose(cls_list, box_list, kpt_planes, grids, strides, lb, *,
                    num_classes=1, num_keypoints=17, conf=0.25, iou_t=0.45,
                    max_dets=300, reg_max=0, channels_first=True,
                    apply_sigmoid=True, kpt_decode="model_pixels",
                    kpt_apply_sigmoid=False):
    """Anchor-free LTRB pose head — mirrors rcdl::decodePose."""
    dets = []
    for cls, box, plane, (h, w), stride in zip(cls_list, box_list, kpt_planes,
                                               grids, strides):
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
            joints = []
            for k in range(num_keypoints):
                raw_x = plane.at(int(j), 3 * k + 0)
                raw_y = plane.at(int(j), 3 * k + 1)
                raw_s = plane.at(int(j), 3 * k + 2)
                if kpt_decode == "cell_relative":
                    mkx = (2.0 * raw_x + gx[j]) * stride
                    mky = (2.0 * raw_y + gy[j]) * stride
                else:
                    mkx, mky = raw_x, raw_y
                joints.append({
                    "x": lb.clamp_x(lb.inv_x(mkx)),
                    "y": lb.clamp_y(lb.inv_y(mky)),
                    "score": float(_sigmoid(raw_s)) if kpt_apply_sigmoid else raw_s,
                })
            dets.append({
                "x1": lb.clamp_x(lb.inv_x(mx1[j])), "y1": lb.clamp_y(lb.inv_y(my1[j])),
                "x2": lb.clamp_x(lb.inv_x(mx2[j])), "y2": lb.clamp_y(lb.inv_y(my2[j])),
                "score": float(score[j]), "class_id": int(best_k[j]),
                "keypoints": joints,
            })
    return [dets[i] for i in ref_nms(dets, iou_t, max_dets)]


# --------------------------------------------------------------------------- #
# COCO-17 layout — the same table src/tasks/pose.cc ships                      #
# --------------------------------------------------------------------------- #
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1),
    (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
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
# Scene builders                                                               #
# --------------------------------------------------------------------------- #
def _one_cell_scene(h=4, w=4, nk=2, nc=1, gy=1, gx=1, logit=10.0,
                    ltrb=(1.0, 1.0, 1.0, 1.0), channels_first=True):
    """A single grid with exactly one above-threshold cell."""
    cells = h * w
    cell = gy * w + gx
    cls = np.full((nc, cells), -20.0, dtype=np.float32)
    cls[nc - 1, cell] = logit
    box = np.zeros((4, cells), dtype=np.float32)
    for side in range(4):
        box[side, cell] = ltrb[side]
    kpt = np.zeros((nk * 3, cells), dtype=np.float32)
    if not channels_first:
        cls = np.ascontiguousarray(cls.T)
        box = np.ascontiguousarray(box.T)
        kpt = np.ascontiguousarray(kpt.T)
    return cls, box, kpt, cell


# --------------------------------------------------------------------------- #
# Box geometry — the pose box IS the detection box                             #
# --------------------------------------------------------------------------- #
def test_pose_box_geometry_is_cell_centred():
    """A cell with LTRB=(1,1,1,1) is a 2*stride box centred on that cell."""
    h, w, stride, nk = 4, 4, 16, 2
    cls, box, kpt, _ = _one_cell_scene(h, w, nk)
    lb = compute_letterbox(64, 64, 64, 64)  # identity: scale 1, no padding
    got = ref_decode_pose([cls], [box], [per_scale_plane(kpt, h, w, nk, True)],
                          [(h, w)], [stride], lb, num_keypoints=nk)
    assert len(got) == 1
    d = got[0]
    assert d["class_id"] == 0
    assert d["score"] == pytest.approx(_sigmoid(10.0), abs=1e-6)
    cx, cy = (1 + 0.5) * stride, (1 + 0.5) * stride
    assert (d["x1"], d["y1"], d["x2"], d["y2"]) == pytest.approx(
        (cx - stride, cy - stride, cx + stride, cy + stride), abs=1e-4)


def test_pose_dfl_box_reduces_like_the_detection_head():
    """A one-hot DFL distribution reduces to that bin's index, per side."""
    reg_max, h, w, stride, nk = 16, 8, 8, 8, 1
    cells = h * w
    # An INTERIOR cell, far enough from the top-left that the largest DFL bin
    # (3 cells up on the `top` side) still lands at a positive coordinate. The
    # letterbox inverse clamps to [0, src], so a cell near the edge would clamp
    # y1 to 0 and hide the reduction this test exists to check — no size of
    # source rescues a negative coordinate.
    gy, gx = 5, 5
    cell = gy * w + gx
    cls = np.full((1, cells), -20.0, dtype=np.float32)
    cls[0, cell] = 20.0
    box = np.full((4 * reg_max, cells), -20.0, dtype=np.float32)
    for side, bin_idx in enumerate((0, 3, 7, 15)):
        box[side * reg_max + bin_idx, cell] = 20.0
    kpt = np.zeros((nk * 3, cells), dtype=np.float32)
    # Large enough that bin 15 on the `bottom` side stays inside the source too.
    lb = compute_letterbox(1000, 1000, 1000, 1000)
    got = ref_decode_pose([cls], [box], [per_scale_plane(kpt, h, w, nk, True)],
                          [(h, w)], [stride], lb, num_keypoints=nk, reg_max=reg_max)
    assert len(got) == 1
    cx, cy = gx + 0.5, gy + 0.5
    assert got[0]["x1"] == pytest.approx((cx - 0) * stride, abs=1e-3)
    assert got[0]["y1"] == pytest.approx((cy - 3) * stride, abs=1e-3)
    assert got[0]["x2"] == pytest.approx((cx + 7) * stride, abs=1e-3)
    assert got[0]["y2"] == pytest.approx((cy + 15) * stride, abs=1e-3)


def test_pose_below_threshold_is_empty():
    h, w, nk = 2, 2, 1
    cls, box, kpt, _ = _one_cell_scene(h, w, nk, logit=-10.0)  # sigmoid(-10) << 0.25
    lb = compute_letterbox(64, 64, 64, 64)
    got = ref_decode_pose([cls], [box], [per_scale_plane(kpt, h, w, nk, True)],
                          [(h, w)], [8], lb, num_keypoints=nk)
    assert got == []


def test_pose_nms_keeps_the_stronger_of_two_overlapping_cells():
    h, w, stride, nk = 1, 2, 8, 1
    cells = h * w
    cls = np.array([[2.0, 3.0]], dtype=np.float32)  # both pass; cell 1 is stronger
    box = np.zeros((4, cells), dtype=np.float32)
    box[:, 0] = [2.0, 1.0, 2.0, 1.0]
    box[:, 1] = [3.0, 1.0, 1.0, 1.0]
    kpt = np.zeros((nk * 3, cells), dtype=np.float32)
    lb = compute_letterbox(200, 200, 200, 200)
    got = ref_decode_pose([cls], [box], [per_scale_plane(kpt, h, w, nk, True)],
                          [(h, w)], [stride], lb, num_keypoints=nk, iou_t=0.5)
    assert len(got) == 1
    assert got[0]["score"] == pytest.approx(_sigmoid(3.0), abs=1e-6)


# --------------------------------------------------------------------------- #
# Keypoints                                                                    #
# --------------------------------------------------------------------------- #
def test_keypoints_land_on_the_named_pixel_for_a_decoded_branch():
    """kpt_decode="model_pixels": the branch already holds model-input pixels,
    so an identity letterbox must hand them back untouched."""
    h, w, stride, nk = 4, 4, 16, 3
    cls, box, kpt, cell = _one_cell_scene(h, w, nk)
    want = [(12.5, 40.25), (0.0, 0.0), (63.0, 1.0)]
    for k, (x, y) in enumerate(want):
        kpt[3 * k + 0, cell] = x
        kpt[3 * k + 1, cell] = y
        kpt[3 * k + 2, cell] = 0.75  # already a probability in this export
    lb = compute_letterbox(64, 64, 64, 64)
    got = ref_decode_pose([cls], [box], [per_scale_plane(kpt, h, w, nk, True)],
                          [(h, w)], [stride], lb, num_keypoints=nk)
    assert len(got) == 1
    joints = got[0]["keypoints"]
    assert len(joints) == nk
    for j, (x, y) in zip(joints, want):
        assert (j["x"], j["y"]) == pytest.approx((x, y), abs=1e-4)
        assert j["score"] == pytest.approx(0.75, abs=1e-6)


def test_keypoints_of_a_raw_branch_are_cell_relative():
    """kpt_decode="cell_relative": kx = (2*raw + gx) * stride, and visibility is
    a logit that needs the sigmoid. A raw of 0 sits half a cell before the cell
    centre, i.e. exactly on the cell's top-left grid corner."""
    h, w, stride, nk, gy, gx = 4, 4, 16, 2, 2, 3
    cls, box, kpt, cell = _one_cell_scene(h, w, nk, gy=gy, gx=gx)
    kpt[0, cell], kpt[1, cell], kpt[2, cell] = 0.0, 0.0, 2.0
    kpt[3, cell], kpt[4, cell], kpt[5, cell] = 0.25, -0.5, -2.0
    lb = compute_letterbox(64, 64, 64, 64)
    got = ref_decode_pose([cls], [box], [per_scale_plane(kpt, h, w, nk, True)],
                          [(h, w)], [stride], lb, num_keypoints=nk,
                          kpt_decode="cell_relative", kpt_apply_sigmoid=True)
    assert len(got) == 1
    j0, j1 = got[0]["keypoints"]
    assert (j0["x"], j0["y"]) == pytest.approx((gx * stride, gy * stride), abs=1e-4)
    assert j0["score"] == pytest.approx(_sigmoid(2.0), abs=1e-6)
    assert (j1["x"], j1["y"]) == pytest.approx(
        ((2 * 0.25 + gx) * stride, lb.clamp_y((2 * -0.5 + gy) * stride)), abs=1e-4)
    assert j1["score"] == pytest.approx(_sigmoid(-2.0), abs=1e-6)


def test_keypoints_un_letterbox_with_the_box():
    """The whole point: a joint and its box come back in the SAME original-image
    frame. 1280x720 into 640x640 is scale 0.5 with 140px of top/bottom padding."""
    h, w, stride, nk = 4, 4, 160, 1  # 640 / 4 = 160
    cls, box, kpt, cell = _one_cell_scene(h, w, nk, gy=1, gx=1)
    kpt[0, cell], kpt[1, cell], kpt[2, cell] = 320.0, 320.0, 0.9  # canvas centre
    lb = compute_letterbox(1280, 720, 640, 640)
    assert (lb.scale, lb.pad_x, lb.pad_y) == pytest.approx((0.5, 0.0, 140.0))
    got = ref_decode_pose([cls], [box], [per_scale_plane(kpt, h, w, nk, True)],
                          [(h, w)], [stride], lb, num_keypoints=nk)
    assert len(got) == 1
    j = got[0]["keypoints"][0]
    # canvas (320,320) -> ((320-0)/0.5, (320-140)/0.5) = (640, 360): the image centre
    assert (j["x"], j["y"]) == pytest.approx((640.0, 360.0), abs=1e-3)
    # the box corners went through the very same map
    cx, cy = 1.5 * stride, 1.5 * stride
    assert got[0]["x1"] == pytest.approx(lb.clamp_x(lb.inv_x(cx - stride)), abs=1e-3)
    assert got[0]["y1"] == pytest.approx(lb.clamp_y(lb.inv_y(cy - stride)), abs=1e-3)


def test_keypoints_outside_the_frame_are_clamped_to_the_source_extent():
    h, w, stride, nk = 2, 2, 8, 1
    cls, box, kpt, cell = _one_cell_scene(h, w, nk, gy=0, gx=0)
    kpt[0, cell], kpt[1, cell] = -50.0, 5000.0
    lb = compute_letterbox(16, 16, 16, 16)
    got = ref_decode_pose([cls], [box], [per_scale_plane(kpt, h, w, nk, True)],
                          [(h, w)], [stride], lb, num_keypoints=nk)
    j = got[0]["keypoints"][0]
    assert (j["x"], j["y"]) == pytest.approx((0.0, 16.0))


def test_shared_keypoint_tensor_gives_each_scale_its_own_anchor_window():
    """The deployed export emits ONE [1,K*3,A] keypoint tensor for the whole
    model, anchors concatenated finest-scale-first. Each scale must read its own
    window, and channels are a whole anchor plane apart."""
    grids = [(2, 2), (1, 1)]
    strides = [8, 16]
    nk, total = 1, sum(h * w for h, w in grids)  # 5 anchors
    cls_list, box_list = [], []
    for (h, w) in grids:
        cells = h * w
        c = np.full((1, cells), 20.0, dtype=np.float32)  # every cell fires
        cls_list.append(c)
        box_list.append(np.zeros((4, cells), dtype=np.float32))
    # [K*3, A]: joint 0's x plane, then its y plane, then its visibility plane.
    kpt = np.zeros((nk * 3, total), dtype=np.float32)
    kpt[0] = [10.0, 11.0, 12.0, 13.0, 20.0]  # x per anchor
    kpt[1] = [30.0, 31.0, 32.0, 33.0, 40.0]  # y per anchor
    kpt[2] = 0.5
    lb = compute_letterbox(100, 100, 100, 100)
    got = ref_decode_pose(cls_list, box_list, shared_planes(kpt, grids), grids,
                          strides, lb, num_keypoints=nk, iou_t=1.1)  # keep them all
    assert len(got) == total
    # All boxes are degenerate (LTRB 0) so NMS keeps insertion order after the
    # score sort; look the joints up by position instead of assuming an order.
    seen = sorted((d["keypoints"][0]["x"], d["keypoints"][0]["y"]) for d in got)
    flat = [v for xy in seen for v in xy]
    assert flat == pytest.approx([10.0, 30.0, 11.0, 31.0, 12.0, 32.0,
                                  13.0, 33.0, 20.0, 40.0])


def test_channel_order_is_only_a_layout():
    """The same scene written [C,H,W] or [H,W,C] must decode identically — the
    flag comes from each output's rknn fmt, never from an assumption."""
    h, w, stride, nk = 4, 4, 16, 2
    cf = _one_cell_scene(h, w, nk, ltrb=(0.5, 0.5, 1.5, 1.0), channels_first=True)
    nf = _one_cell_scene(h, w, nk, ltrb=(0.5, 0.5, 1.5, 1.0), channels_first=False)
    lb = compute_letterbox(64, 64, 64, 64)
    a = ref_decode_pose([cf[0]], [cf[1]], [per_scale_plane(cf[2], h, w, nk, True)],
                        [(h, w)], [stride], lb, num_keypoints=nk)
    b = ref_decode_pose([nf[0]], [nf[1]], [per_scale_plane(nf[2], h, w, nk, False)],
                        [(h, w)], [stride], lb, num_keypoints=nk,
                        channels_first=False)
    assert len(a) == len(b) == 1
    for f in ("x1", "y1", "x2", "y2", "score"):
        assert a[0][f] == pytest.approx(b[0][f], abs=1e-6)
    assert a[0]["class_id"] == b[0]["class_id"]
    for ja, jb in zip(a[0]["keypoints"], b[0]["keypoints"]):
        assert (ja["x"], ja["y"]) == pytest.approx((jb["x"], jb["y"]), abs=1e-6)


# --------------------------------------------------------------------------- #
# COCO-17 metadata                                                             #
# --------------------------------------------------------------------------- #
def test_coco_skeleton_is_self_consistent():
    assert len(COCO_KEYPOINTS) == 17
    assert len(set(COCO_KEYPOINTS)) == 17
    for a, b in COCO_SKELETON:
        assert 0 <= a < 17 and 0 <= b < 17 and a != b
    # every joint except the ears' outer links participates in at least one bone
    linked = {i for edge in COCO_SKELETON for i in edge}
    assert linked == set(range(17))


def test_cxx_exposes_the_same_coco_tables(cxx):
    if cxx is None or not hasattr(cxx, "coco_keypoint_names"):
        pytest.skip("compiled rcdl module without pose metadata bindings")
    assert list(cxx.coco_keypoint_names()) == COCO_KEYPOINTS
    assert [tuple(e) for e in cxx.coco_skeleton()] == COCO_SKELETON


# --------------------------------------------------------------------------- #
# Opportunistic C++ cross-check                                                #
# --------------------------------------------------------------------------- #
def test_decode_pose_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "decode_pose"):
        pytest.skip("compiled rcdl module without decode_pose bindings")
    h, w, stride, nk = 8, 8, 8, 3
    rng = np.random.default_rng(5)
    cells = h * w
    cls = rng.uniform(-6.0, 4.0, size=(1, cells)).astype(np.float32)
    box = rng.uniform(0.2, 3.0, size=(4, cells)).astype(np.float32)
    kpt = rng.uniform(0.0, 64.0, size=(nk * 3, cells)).astype(np.float32)
    lb = compute_letterbox(1280, 720, 64, 64)
    ref = ref_decode_pose([cls], [box], [per_scale_plane(kpt, h, w, nk, True)],
                          [(h, w)], [stride], lb, num_keypoints=nk)
    got = cxx.decode_pose([np.ascontiguousarray(cls)], [np.ascontiguousarray(box)],
                          [np.ascontiguousarray(kpt)], [(h, w)], [stride],
                          lb.as_tuple(), num_classes=1, num_keypoints=nk,
                          channels_first=True, apply_sigmoid=True)
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert abs(a.box.score - b["score"]) < 1e-4
        assert abs(a.box.x1 - b["x1"]) < 1e-3 and abs(a.box.y2 - b["y2"]) < 1e-3
        for ja, jb in zip(a.keypoints, b["keypoints"]):
            assert abs(ja.x - jb["x"]) < 1e-3 and abs(ja.y - jb["y"]) < 1e-3
