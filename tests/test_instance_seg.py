"""Instance-segmentation post-processing tests.

PURE-NUMPY tests of the decode + mask-assembly reference that ``rcdl::
decodeInstanceSeg`` mirrors. They need only numpy and run anywhere: no board,
no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_instance_seg.py

The interesting half is the MASK GEOMETRY. A YOLO seg mask is born at the
prototype grid (160x160 for a 640 input) and has to arrive in original-image
pixels, which takes four steps that are easy to get subtly wrong:

    proto·coef -> sigmoid -> upsample to the model canvas -> zero outside the
    instance box -> cut the letterbox padding off -> resize to the source frame

The scenarios below are built so the answer is exact rather than approximate:
``test_mask_lands_on_known_source_pixels`` uses a saturated prototype and a
2x letterbox with padding, and the resulting mask's bounding box has to equal
the un-letterboxed BOX to the pixel. ``test_prototype_rectangle_maps_to_known
_source_pixels`` does the mirror image — box wide open, a rectangle in the
prototype — and pins where that rectangle's edges land.
"""

import numpy as np
import pytest

# One oracle for the shared machinery: the letterbox geometry and the per-class
# NMS are the SAME code paths the detector uses (rcdl::nms is literally shared),
# so a second copy here could drift away from the thing under test.
from test_detection import compute_letterbox, ref_nms


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _lround(v):
    """C's lround: half away from zero. np.round is banker's rounding, which
    disagrees on exact .5 — and letterbox pads land on .5 all the time."""
    return int(np.floor(v + 0.5)) if v >= 0 else -int(np.floor(-v + 0.5))


# --------------------------------------------------------------------------- #
# Mask geometry oracle — mirrors the MaskAssembler in src/tasks/instance_seg.cc #
# --------------------------------------------------------------------------- #
def axis_lerp(dst_n, src_n):
    """Pixel-CENTRE bilinear map, dst index -> (i0, i1, weight), clamped.

        f = (d + 0.5) * src_n/dst_n - 0.5

    This is cv2.resize(INTER_LINEAR) / torch interpolate(align_corners=False).
    """
    d = np.arange(dst_n, dtype=np.float64)
    f = np.clip((d + 0.5) * src_n / dst_n - 0.5, 0.0, src_n - 1)
    i0 = np.floor(f).astype(np.int64)
    i1 = np.minimum(i0 + 1, src_n - 1)
    return i0, i1, f - i0


def depad_window(lb):
    """The canvas window the scaled source image occupies: (px, py, nw, nh).

    The preprocessor writes integer rectangles (the RGA blit rounds and reflects
    the rounding back into the LetterboxInfo), so the mask is cut on integers
    too. Everything outside this window is letterbox padding and must be gone
    BEFORE the mask is resized to the source frame.
    """
    nw = min(max(_lround(lb.src_w * lb.scale), 1), lb.dst_w)
    nh = min(max(_lround(lb.src_h * lb.scale), 1), lb.dst_h)
    px = min(max(_lround(lb.pad_x), 0), lb.dst_w - nw)
    py = min(max(_lround(lb.pad_y), 0), lb.dst_h - nh)
    return px, py, nw, nh


def ref_process_mask(coef, proto_hwc, model_box, lb, mask_thresh=0.5, full_frame=True):
    """One instance's binary mask in ORIGINAL-image pixels.

    ``proto_hwc`` is (proto_h, proto_w, num_coef); ``model_box`` is the
    (x1,y1,x2,y2) box in MODEL-INPUT pixels — the crop happens on the canvas,
    before the un-letterbox. Returns (mask, x0, y0): the mask covers
    [x0, x0+w) x [y0, y0+h) of the source frame.
    """
    ph, pw, _ = proto_hwc.shape
    pm = _sigmoid(proto_hwc @ np.asarray(coef, dtype=np.float64))  # (ph, pw)

    # Upsample to the canvas, but ONLY inside the box: crop_mask zeroes the rest,
    # so an integer canvas pixel k survives iff k >= m1 and k < m2, that is
    # k in [ceil(m1), ceil(m2)).
    mx1, my1, mx2, my2 = model_box
    cx0 = int(np.clip(np.ceil(mx1), 0, lb.dst_w))
    cx1 = int(np.clip(np.ceil(mx2), 0, lb.dst_w))
    cy0 = int(np.clip(np.ceil(my1), 0, lb.dst_h))
    cy1 = int(np.clip(np.ceil(my2), 0, lb.dst_h))

    canvas = np.zeros((lb.dst_h, lb.dst_w), dtype=np.float64)
    if cx1 > cx0 and cy1 > cy0:
        iy0, iy1, wy = axis_lerp(lb.dst_h, ph)
        ix0, ix1, wx = axis_lerp(lb.dst_w, pw)
        ys = np.arange(cy0, cy1)
        xs = np.arange(cx0, cx1)
        top = (pm[np.ix_(iy0[ys], ix0[xs])] * (1.0 - wx[xs]) +
               pm[np.ix_(iy0[ys], ix1[xs])] * wx[xs])
        bot = (pm[np.ix_(iy1[ys], ix0[xs])] * (1.0 - wx[xs]) +
               pm[np.ix_(iy1[ys], ix1[xs])] * wx[xs])
        canvas[cy0:cy1, cx0:cx1] = top * (1.0 - wy[ys])[:, None] + bot * wy[ys][:, None]

    # De-pad + resize to the source frame, in one sampling pass.
    px, py, nw, nh = depad_window(lb)
    oy0, oy1, owy = axis_lerp(lb.src_h, nh)
    ox0, ox1, owx = axis_lerp(lb.src_w, nw)
    r0, r1 = py + oy0, py + oy1
    c0, c1 = px + ox0, px + ox1
    top = canvas[np.ix_(r0, c0)] * (1.0 - owx) + canvas[np.ix_(r0, c1)] * owx
    bot = canvas[np.ix_(r1, c0)] * (1.0 - owx) + canvas[np.ix_(r1, c1)] * owx
    full = top * (1.0 - owy)[:, None] + bot * owy[:, None]
    mask = (full > mask_thresh).astype(np.uint8)

    if full_frame:
        return mask, 0, 0
    # Box-cropped window: the instance's clipped integer bounding box, which is
    # all that can ever be non-zero.
    sx1, sy1, sx2, sy2 = (lb.clamp_x(lb.inv_x(mx1)), lb.clamp_y(lb.inv_y(my1)),
                          lb.clamp_x(lb.inv_x(mx2)), lb.clamp_y(lb.inv_y(my2)))
    wx0 = int(np.clip(np.floor(sx1), 0, lb.src_w))
    wy0 = int(np.clip(np.floor(sy1), 0, lb.src_h))
    wx1 = max(wx0, int(np.clip(np.ceil(sx2), 0, lb.src_w)))
    wy1 = max(wy0, int(np.clip(np.ceil(sy2), 0, lb.src_h)))
    return mask[wy0:wy1, wx0:wx1], wx0, wy0


def ref_dfl(box, reg_max):
    """Σ b·softmax(b) over each side's reg_max bins; ``box`` is [4*reg_max, N]."""
    x = box.reshape(4, reg_max, -1).astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=1, keepdims=True)
    return (p * np.arange(reg_max).reshape(1, reg_max, 1)).sum(axis=1)


def ref_decode_instance_seg(cls_list, box_list, mc_list, grids, strides, proto_hwc, lb,
                            num_classes, conf=0.25, iou_t=0.45, max_dets=100, reg_max=0,
                            channels_first=True, apply_sigmoid=False, mask_thresh=0.5,
                            compute_masks=True, full_frame=True):
    """Mirrors rcdl::decodeInstanceSeg.

    Buffers are [C,H,W] when ``channels_first``, else [H,W,C]. ``proto_hwc`` is
    always given HWC here; the C++ side takes the layout from the tensor's fmt.
    """
    ncoef = proto_hwc.shape[2]
    box_ch = 4 * reg_max if reg_max > 0 else 4
    dets, coefs = [], []
    for cls, box, mc, (h, w), stride in zip(cls_list, box_list, mc_list, grids, strides):
        cells = h * w
        if channels_first:
            c = np.asarray(cls, np.float64).reshape(num_classes, cells)
            b = np.asarray(box, np.float64).reshape(box_ch, cells)
            k = np.asarray(mc, np.float64).reshape(ncoef, cells)
        else:
            c = np.asarray(cls, np.float64).reshape(cells, num_classes).T
            b = np.asarray(box, np.float64).reshape(cells, box_ch).T
            k = np.asarray(mc, np.float64).reshape(cells, ncoef).T

        best_k = c.argmax(axis=0)
        score = c.max(axis=0)
        if apply_sigmoid:
            score = _sigmoid(score)
        d = ref_dfl(b, reg_max) if reg_max > 0 else b

        gy, gx = np.divmod(np.arange(cells), w)
        cx, cy = gx + 0.5, gy + 0.5
        mx1, my1 = (cx - d[0]) * stride, (cy - d[1]) * stride
        mx2, my2 = (cx + d[2]) * stride, (cy + d[3]) * stride
        for j in np.nonzero(score >= conf)[0]:
            # NMS runs on MODEL-INPUT boxes: the letterbox map is a uniform scale
            # plus a translation, so IoU (and therefore the surviving set) is the
            # same either side of it, and the mask crop needs canvas coordinates.
            dets.append({"x1": float(mx1[j]), "y1": float(my1[j]),
                         "x2": float(mx2[j]), "y2": float(my2[j]),
                         "score": float(score[j]), "class_id": int(best_k[j])})
            coefs.append(k[:, j].copy())

    out = []
    for i in ref_nms(dets, iou_t, max_dets):
        d = dets[i]
        r = {"x1": lb.clamp_x(lb.inv_x(d["x1"])), "y1": lb.clamp_y(lb.inv_y(d["y1"])),
             "x2": lb.clamp_x(lb.inv_x(d["x2"])), "y2": lb.clamp_y(lb.inv_y(d["y2"])),
             "score": d["score"], "class_id": d["class_id"]}
        if compute_masks:
            m, x0, y0 = ref_process_mask(coefs[i], proto_hwc,
                                         (d["x1"], d["y1"], d["x2"], d["y2"]), lb,
                                         mask_thresh, full_frame)
            r["mask"], r["mask_x0"], r["mask_y0"] = m, x0, y0
        else:
            r["mask"], r["mask_x0"], r["mask_y0"] = np.zeros((0, 0), np.uint8), 0, 0
        out.append(r)
    return out


def bbox_of(mask):
    """(x1, y1, x2, y2) half-open bounding box of the non-zero pixels."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# --------------------------------------------------------------------------- #
# Scene builders                                                               #
# --------------------------------------------------------------------------- #
def one_cell_scene(grid_h, grid_w, stride, num_classes, num_coef, cell_xy, ltrb, cls_id,
                   cls_value, coef, channels_first=True):
    """A single above-threshold cell on one scale; everything else is background."""
    cells = grid_h * grid_w
    c = np.full((num_classes, cells), -20.0, np.float32)
    b = np.zeros((4, cells), np.float32)
    k = np.zeros((num_coef, cells), np.float32)
    gx, gy = cell_xy
    j = gy * grid_w + gx
    c[cls_id, j] = cls_value
    b[:, j] = ltrb
    k[:, j] = coef
    if not channels_first:
        c, b, k = (np.ascontiguousarray(t.T) for t in (c, b, k))
    return ([c], [b], [k], [(grid_h, grid_w)], [stride])


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
    return rcdl if hasattr(rcdl, "decode_instance_seg") else None


# --------------------------------------------------------------------------- #
# THE geometry test: a known mask must land on known source pixels             #
# --------------------------------------------------------------------------- #
def test_mask_lands_on_known_source_pixels():
    """A saturated prototype makes the BOX the only thing shaping the mask, so
    the mask's bounding box must come out equal to the un-letterboxed box.

    128x64 source into a 64x64 canvas: scale 0.5, no x padding, 16 px of y
    padding top and bottom. Anything that forgets to strip that padding — or
    strips it after the resize instead of before — puts the mask 32 source pixels
    off, which this asserts to the pixel.
    """
    lb = compute_letterbox(128, 64, 64, 64)
    assert (lb.scale, lb.pad_x, lb.pad_y) == (0.5, 0.0, 16.0)

    proto = np.full((16, 16, 1), 20.0, np.float32)  # sigmoid(20) ~ 1 everywhere
    # Cell (2,3) at stride 8 -> centre (2.5, 3.5); LTRB (1.5, 1.0, 0.5, 1.0)
    # gives the canvas box (8, 20, 24, 36).
    cls, box, mc, grids, strides = one_cell_scene(
        8, 8, 8, 2, 1, cell_xy=(2, 3), ltrb=(1.5, 1.0, 0.5, 1.0), cls_id=1,
        cls_value=5.0, coef=(1.0,))

    got = ref_decode_instance_seg(cls, box, mc, grids, strides, proto, lb,
                                  num_classes=2, apply_sigmoid=True)
    assert len(got) == 1
    r = got[0]
    assert r["class_id"] == 1
    assert r["score"] == pytest.approx(_sigmoid(5.0), abs=1e-6)

    # Canvas box (8,20,24,36) un-letterboxed: x/0.5 and (y-16)/0.5.
    assert (r["x1"], r["y1"], r["x2"], r["y2"]) == pytest.approx((16.0, 8.0, 48.0, 40.0))
    assert r["mask"].shape == (64, 128)  # full source frame
    assert bbox_of(r["mask"]) == (16, 8, 48, 40)
    assert int(r["mask"].sum()) == 32 * 32  # solid, no holes


def test_mask_cropped_to_box_window_matches_full_frame():
    """The box-cropped mask is the full-frame one restricted to its own box."""
    lb = compute_letterbox(128, 64, 64, 64)
    proto = np.full((16, 16, 1), 20.0, np.float32)
    args = one_cell_scene(8, 8, 8, 2, 1, (2, 3), (1.5, 1.0, 0.5, 1.0), 1, 5.0, (1.0,))

    full = ref_decode_instance_seg(*args[:3], args[3], args[4], proto, lb, num_classes=2,
                                   apply_sigmoid=True, full_frame=True)[0]
    crop = ref_decode_instance_seg(*args[:3], args[3], args[4], proto, lb, num_classes=2,
                                   apply_sigmoid=True, full_frame=False)[0]
    assert (crop["mask_x0"], crop["mask_y0"]) == (16, 8)
    assert crop["mask"].shape == (32, 32)
    assert crop["mask"].all()
    x0, y0 = crop["mask_x0"], crop["mask_y0"]
    h, w = crop["mask"].shape
    np.testing.assert_array_equal(crop["mask"], full["mask"][y0:y0 + h, x0:x0 + w])
    # ... and nothing outside that window survived the crop-to-box.
    assert int(full["mask"].sum()) == int(crop["mask"].sum())


def test_prototype_rectangle_maps_to_known_source_pixels():
    """Mirror image of the test above: box wide open, a rectangle in the
    PROTOTYPE. Pins the prototype-grid -> source-pixel mapping.

    Identity letterbox (64x64 -> 64x64) so the second resize is a no-op and only
    the 16x16 -> 64x64 upsample is under test. The prototype is +20 on rows 4..11
    and cols 2..9; the upsample's 0.5 crossing sits half a prototype cell outside
    those, i.e. at canvas y = 15.5 / 47.5 and x = 7.5 / 39.5.
    """
    lb = compute_letterbox(64, 64, 64, 64)
    assert (lb.scale, lb.pad_x, lb.pad_y) == (1.0, 0.0, 0.0)

    proto = np.full((16, 16, 1), -20.0, np.float32)
    proto[4:12, 2:10, 0] = 20.0

    # 1x1 grid at stride 64, LTRB all 0.5 -> the box is the whole canvas.
    cls, box, mc, grids, strides = one_cell_scene(
        1, 1, 64, 1, 1, (0, 0), (0.5, 0.5, 0.5, 0.5), 0, 5.0, (1.0,))
    got = ref_decode_instance_seg(cls, box, mc, grids, strides, proto, lb,
                                  num_classes=1, apply_sigmoid=True)
    assert len(got) == 1
    assert bbox_of(got[0]["mask"]) == (8, 16, 40, 48)


def test_mask_is_cropped_to_the_box():
    """The prototype is on everywhere, so anything outside the box that survives
    means the crop_mask step is missing."""
    lb = compute_letterbox(64, 64, 64, 64)
    proto = np.full((16, 16, 1), 20.0, np.float32)
    cls, box, mc, grids, strides = one_cell_scene(
        8, 8, 8, 1, 1, (2, 2), (2.5, 2.5, 1.5, 1.5), 0, 5.0, (1.0,))
    # centre (2.5,2.5) -> canvas box (0, 0, 32, 32)
    got = ref_decode_instance_seg(cls, box, mc, grids, strides, proto, lb, num_classes=1,
                                  apply_sigmoid=True)
    m = got[0]["mask"]
    assert bbox_of(m) == (0, 0, 32, 32)
    assert not m[32:, :].any() and not m[:, 32:].any()


# --------------------------------------------------------------------------- #
# Coefficients, layouts, thresholds                                            #
# --------------------------------------------------------------------------- #
def test_mask_is_a_linear_combination_of_prototypes():
    """mask = sigmoid(Σ coef·proto): a negative coefficient must turn a plane
    OFF, which is what catches an accidental activation on the coefficients."""
    lb = compute_letterbox(64, 64, 64, 64)
    proto = np.zeros((16, 16, 2), np.float32)
    proto[:, :8, 0] = 20.0   # plane 0: left half
    proto[:, 8:, 1] = 20.0   # plane 1: right half

    def run(coef):
        cls, box, mc, grids, strides = one_cell_scene(
            1, 1, 64, 1, 2, (0, 0), (0.5, 0.5, 0.5, 0.5), 0, 5.0, coef)
        return ref_decode_instance_seg(cls, box, mc, grids, strides, proto, lb,
                                       num_classes=1, apply_sigmoid=True)[0]["mask"]

    left = run((1.0, 0.0))
    assert left[:, :30].all() and not left[:, 34:].any()
    right = run((0.0, 1.0))
    assert right[:, 34:].all() and not right[:, :30].any()
    # A negative coefficient drives sigmoid below the threshold everywhere.
    assert not run((-1.0, -1.0)).any()


@pytest.mark.parametrize("channels_first", [True, False])
def test_decode_is_channel_order_invariant(channels_first):
    """The same head expressed [C,H,W] or [H,W,C] must decode identically —
    the RKNN exports differ exactly here."""
    lb = compute_letterbox(128, 64, 64, 64)
    proto = np.full((16, 16, 1), 20.0, np.float32)
    args = one_cell_scene(8, 8, 8, 3, 1, (2, 3), (1.5, 1.0, 0.5, 1.0), 2, 5.0, (1.0,),
                          channels_first=channels_first)
    got = ref_decode_instance_seg(*args[:3], args[3], args[4], proto, lb, num_classes=3,
                                  channels_first=channels_first, apply_sigmoid=True)
    assert len(got) == 1 and got[0]["class_id"] == 2
    assert bbox_of(got[0]["mask"]) == (16, 8, 48, 40)


def test_below_threshold_yields_nothing():
    lb = compute_letterbox(64, 64, 64, 64)
    proto = np.full((16, 16, 1), 20.0, np.float32)
    cls, box, mc, grids, strides = one_cell_scene(
        1, 1, 64, 1, 1, (0, 0), (0.5, 0.5, 0.5, 0.5), 0, -5.0, (1.0,))
    got = ref_decode_instance_seg(cls, box, mc, grids, strides, proto, lb, num_classes=1,
                                  conf=0.25, apply_sigmoid=True)
    assert got == []


def test_compute_masks_false_keeps_boxes_only():
    lb = compute_letterbox(128, 64, 64, 64)
    proto = np.full((16, 16, 1), 20.0, np.float32)
    cls, box, mc, grids, strides = one_cell_scene(
        8, 8, 8, 2, 1, (2, 3), (1.5, 1.0, 0.5, 1.0), 1, 5.0, (1.0,))
    got = ref_decode_instance_seg(cls, box, mc, grids, strides, proto, lb, num_classes=2,
                                  apply_sigmoid=True, compute_masks=False)
    assert len(got) == 1
    assert got[0]["mask"].size == 0
    assert (got[0]["x1"], got[0]["y1"], got[0]["x2"], got[0]["y2"]) == \
        pytest.approx((16.0, 8.0, 48.0, 40.0))


def test_nms_suppresses_the_overlapping_instance():
    """Two near-identical cells of the same class collapse to one instance, and
    the survivor is the higher-scoring one."""
    lb = compute_letterbox(64, 64, 64, 64)
    proto = np.full((16, 16, 1), 20.0, np.float32)
    cells = 8 * 8
    c = np.full((1, cells), -20.0, np.float32)
    b = np.zeros((4, cells), np.float32)
    k = np.zeros((1, cells), np.float32)
    for cell, (ltrb, val) in {
        2 * 8 + 2: ((2.5, 2.5, 1.5, 1.5), 5.0),   # canvas box (0,0,32,32)
        2 * 8 + 3: ((3.5, 2.5, 0.5, 1.5), 3.0),   # the same box, lower score
    }.items():
        c[0, cell] = val
        b[:, cell] = ltrb
        k[:, cell] = 1.0
    got = ref_decode_instance_seg([c], [b], [k], [(8, 8)], [8], proto, lb, num_classes=1,
                                 apply_sigmoid=True)
    assert len(got) == 1
    assert got[0]["score"] == pytest.approx(_sigmoid(5.0), abs=1e-6)


def test_dfl_box_head_decodes_the_same_boxes():
    """A one-hot DFL distribution must reduce to that bin's index, so a DFL head
    and a plain-LTRB head describing the same box produce the same mask."""
    lb = compute_letterbox(128, 64, 64, 64)
    proto = np.full((16, 16, 1), 20.0, np.float32)
    reg_max = 16
    cells = 8 * 8
    c = np.full((1, cells), -20.0, np.float32)
    k = np.zeros((1, cells), np.float32)
    b = np.full((4 * reg_max, cells), -20.0, np.float32)
    cell = 3 * 8 + 2
    c[0, cell] = 5.0
    k[0, cell] = 1.0
    # LTRB (1.5, 1.0, 0.5, 1.0) is not representable one-hot, so use integers:
    # (2, 1, 1, 1) -> canvas box (4, 20, 28, 36).
    for side, bin_idx in enumerate((2, 1, 1, 1)):
        b[side * reg_max + bin_idx, cell] = 20.0
    got = ref_decode_instance_seg([c], [b], [k], [(8, 8)], [8], proto, lb, num_classes=1,
                                  reg_max=reg_max, apply_sigmoid=True)
    assert len(got) == 1
    assert (got[0]["x1"], got[0]["y1"], got[0]["x2"], got[0]["y2"]) == \
        pytest.approx((8.0, 8.0, 56.0, 40.0), abs=1e-3)


# --------------------------------------------------------------------------- #
# Opportunistic C++ cross-check                                                #
# --------------------------------------------------------------------------- #
def test_decode_instance_seg_matches_cxx(cxx):
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_instance_seg bindings")
    lb = compute_letterbox(128, 64, 64, 64)
    proto = np.full((16, 16, 1), 20.0, np.float32)
    cls, box, mc, grids, strides = one_cell_scene(
        8, 8, 8, 2, 1, (2, 3), (1.5, 1.0, 0.5, 1.0), 1, 5.0, (1.0,))
    ref = ref_decode_instance_seg(cls, box, mc, grids, strides, proto, lb, num_classes=2,
                                  apply_sigmoid=True)
    got = cxx.decode_instance_seg(
        [np.ascontiguousarray(t) for t in cls],
        [np.ascontiguousarray(t) for t in box],
        [np.ascontiguousarray(t) for t in mc],
        grids, strides, np.ascontiguousarray(proto),
        (lb.scale, lb.pad_x, lb.pad_y, lb.src_w, lb.src_h, lb.dst_w, lb.dst_h),
        num_classes=2, apply_sigmoid=True, channels_first=True,
        proto_channels_first=False)
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert a.class_id == b["class_id"]
        assert abs(a.score - b["score"]) < 1e-3
        for f in ("x1", "y1", "x2", "y2"):
            assert abs(getattr(a, f) - b[f]) < 1e-3
        assert (a.mask_w, a.mask_h) == (b["mask"].shape[1], b["mask"].shape[0])
        np.testing.assert_array_equal(np.asarray(a.mask).reshape(b["mask"].shape), b["mask"])
