"""OCR post-processing tests (PP-OCR detection + CTC recognition, numpy path).

These are PURE-NUMPY tests of the reference the C++ ``rcdl::decodeTextBoxes`` /
``rcdl::ctcGreedyDecode`` / ``rcdl::minAreaQuad`` / ``rcdl::unclipQuad`` mirror.
They need only numpy and run anywhere: no board, no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_ocr.py

The module-level ``ref_*`` functions double as the documented "numpy path":
given a DB probability map (or a recognition head's per-step scores) they produce
the same TextBoxes / text the C++ decoders do. Where the compiled module exposes
the decoders they are exercised against the same oracle, but no core assertion
depends on the compiled module being there.

What is pinned here on purpose, because each has a way of being silently wrong:
  * the CTC collapse rule — repeats merge, a BLANK separates them, so
    ``[a, a, blank, a]`` is "aa" and not "a";
  * probability-map thresholding — bin_thresh decides what a region IS,
    box_thresh decides whether it survives, and they are not the same test;
  * the unclip expansion, checked on a rectangle where the answer is exact;
  * un-letterboxing — every coordinate that leaves the decoder is in
    ORIGINAL-image pixels, through a down-sampled map and a padded canvas.
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


IDENTITY_LB = Letterbox(1.0, 0.0, 0.0, 0, 0, 0, 0)  # dst_w == 0 => no mapping


# --------------------------------------------------------------------------- #
# Quadrilateral geometry oracle                                                #
# --------------------------------------------------------------------------- #
def ref_convex_hull(points):
    """Andrew's monotone chain; collinear points dropped, closing point removed."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def ref_order_quad(corners):
    """TL, TR, BR, BL — sort by x, then the upper of each pair is the top one."""
    s = sorted(corners, key=lambda p: p[0])  # python's sort is stable, as is ours
    i0, i3 = (0, 1) if s[1][1] > s[0][1] else (1, 0)
    i1, i2 = (2, 3) if s[3][1] > s[2][1] else (3, 2)
    return [s[i0], s[i1], s[i2], s[i3]]


def ref_min_area_quad(points):
    """Minimum-area enclosing rectangle -> (4 corners TL/TR/BR/BL, shorter side).

    Rotating calipers over the convex hull: the minimum-area rectangle always has
    a side flush with a hull edge, so trying every hull edge is exact.
    """
    hull = ref_convex_hull(points)
    if not hull:
        return [(0.0, 0.0)] * 4, 0.0
    best = None
    m = len(hull)
    for i in range(m):
        a, b = hull[i], hull[(i + 1) % m]
        dx, dy = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(dx, dy)
        if ln <= 0.0:
            continue
        u = (dx / ln, dy / ln)
        v = (-u[1], u[0])
        pu = [p[0] * u[0] + p[1] * u[1] for p in hull]
        pv = [p[0] * v[0] + p[1] * v[1] for p in hull]
        w, h = max(pu) - min(pu), max(pv) - min(pv)
        if best is None or w * h < best[0]:
            best = (w * h, u, v, min(pu), min(pv), w, h)
    if best is None:  # every point identical
        return ref_order_quad([hull[0]] * 4), 0.0
    _, u, v, umin, vmin, w, h = best

    def corner(a_, b_):
        return (a_ * u[0] + b_ * v[0], a_ * u[1] + b_ * v[1])

    c = [corner(umin, vmin), corner(umin + w, vmin),
         corner(umin + w, vmin + h), corner(umin, vmin + h)]
    return ref_order_quad(c), min(w, h)


def ref_unclip_quad(quad, ratio):
    """Miter offset of a convex quad by d = area * ratio / perimeter."""
    area2 = 0.0
    perim = 0.0
    for k in range(4):
        j = (k + 1) % 4
        area2 += quad[k][0] * quad[j][1] - quad[j][0] * quad[k][1]
        perim += math.hypot(quad[j][0] - quad[k][0], quad[j][1] - quad[k][1])
    area = abs(area2) * 0.5
    d = area * ratio / perim if perim > 0 else 0.0
    wind = 1.0 if area2 >= 0 else -1.0

    def normal(f, t):
        ex, ey = quad[t][0] - quad[f][0], quad[t][1] - quad[f][1]
        ln = math.hypot(ex, ey)
        if ln <= 0.0:
            return (0.0, 0.0)
        return (wind * ey / ln, -wind * ex / ln)

    out = []
    for k in range(4):
        n0 = normal((k + 3) % 4, k)
        n1 = normal(k, (k + 1) % 4)
        den = 1.0 + n0[0] * n1[0] + n0[1] * n1[1]
        if den > 1e-6:
            mx, my = d * (n0[0] + n1[0]) / den, d * (n0[1] + n1[1]) / den
        else:
            mx, my = d * n1[0], d * n1[1]
        out.append((quad[k][0] + mx, quad[k][1] + my))
    return out


def ref_box_score_fast(prob, quad):
    """Mean of the probability map inside the quad (scanline even-odd fill)."""
    h, w = prob.shape
    xs_all = [p[0] for p in quad]
    ys_all = [p[1] for p in quad]
    x0 = int(min(max(math.floor(min(xs_all)), 0), w - 1))
    x1 = int(min(max(math.ceil(max(xs_all)), 0), w - 1))
    y0 = int(min(max(math.floor(min(ys_all)), 0), h - 1))
    y1 = int(min(max(math.ceil(max(ys_all)), 0), h - 1))
    total, count = 0.0, 0
    for y in range(y0, y1 + 1):
        xs = []
        for k in range(4):
            j = (k + 1) % 4
            ay, by = quad[k][1], quad[j][1]
            if (ay <= y) == (by <= y):
                continue
            ax, bx = quad[k][0], quad[j][0]
            xs.append(ax + (y - ay) * (bx - ax) / (by - ay))
        xs.sort()
        for s in range(0, len(xs) - 1, 2):
            xa = max(x0, int(math.ceil(xs[s])))
            xb = min(x1, int(math.floor(xs[s + 1])))
            if xb >= xa:
                total += float(prob[y, xa:xb + 1].sum())
                count += xb - xa + 1
    if count > 0:
        return total / count
    cx = int(min(max((min(xs_all) + max(xs_all)) * 0.5, 0), w - 1))
    cy = int(min(max((min(ys_all) + max(ys_all)) * 0.5, 0), h - 1))
    return float(prob[cy, cx])


# --------------------------------------------------------------------------- #
# Detection oracle — mirrors rcdl::decodeTextBoxes                             #
# --------------------------------------------------------------------------- #
DET_DEFAULTS = dict(bin_thresh=0.3, box_thresh=0.6, unclip_ratio=1.5, min_size=3,
                    min_box_side=3, max_candidates=1000, connectivity=8,
                    apply_sigmoid=False)


def _regions(fg, connectivity):
    """Flood the foreground mask, yielding each region's BOUNDARY pixels."""
    h, w = fg.shape
    seen = np.zeros_like(fg, dtype=bool)
    nb4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    nb8 = nb4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    nb = nb4 if connectivity == 4 else nb8
    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx] or not fg[sy, sx]:
                continue
            stack, border = [(sx, sy)], []
            seen[sy, sx] = True
            while stack:
                cx, cy = stack.pop()
                is_border = False
                for i, (dx, dy) in enumerate(nb):
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < w and 0 <= ny < h):
                        if i < 4:
                            is_border = True
                        continue
                    if not fg[ny, nx]:
                        if i < 4:
                            is_border = True
                        continue
                    if not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
                if is_border:
                    border.append((float(cx), float(cy)))
            yield border


def ref_decode_text_boxes(prob, lb=IDENTITY_LB, **over):
    """DB probability map -> text quads in original-image pixels."""
    cfg = dict(DET_DEFAULTS)
    cfg.update(over)
    prob = np.asarray(prob, dtype=np.float32)
    if cfg["apply_sigmoid"]:
        prob = (1.0 / (1.0 + np.exp(-prob))).astype(np.float32)
    h, w = prob.shape

    has_lb = lb.dst_w > 0 and lb.dst_h > 0
    sx = lb.dst_w / w if has_lb else 1.0
    sy = lb.dst_h / h if has_lb else 1.0

    def to_image(px, py):
        if not has_lb:
            return px, py
        return lb.clamp_x(lb.inv_x(px * sx)), lb.clamp_y(lb.inv_y(py * sy))

    boxes = []
    fg = prob > cfg["bin_thresh"]
    for n, border in enumerate(_regions(fg, cfg["connectivity"])):
        if n >= cfg["max_candidates"]:
            break
        if not border:
            continue
        quad, sside = ref_min_area_quad(border)
        if sside < cfg["min_size"]:
            continue
        score = ref_box_score_fast(prob, quad)
        if score < cfg["box_thresh"]:
            continue
        refit, sside2 = ref_min_area_quad(ref_unclip_quad(quad, cfg["unclip_ratio"]))
        if sside2 < cfg["min_size"] + 2:
            continue
        mapped = [to_image(px, py) for px, py in refit]
        bw = int(math.hypot(mapped[1][0] - mapped[0][0], mapped[1][1] - mapped[0][1]))
        bh = int(math.hypot(mapped[3][0] - mapped[0][0], mapped[3][1] - mapped[0][1]))
        if bw <= cfg["min_box_side"] or bh <= cfg["min_box_side"]:
            continue
        boxes.append({
            "pts": mapped,
            "x1": min(p[0] for p in mapped), "y1": min(p[1] for p in mapped),
            "x2": max(p[0] for p in mapped), "y2": max(p[1] for p in mapped),
            "score": float(score),
        })
    return boxes


def ref_sort_text_boxes(boxes, row_tol=10.0):
    """Reading order: top to bottom, then left to right within a visual row."""
    out = sorted(boxes, key=lambda b: (b["pts"][0][1], b["pts"][0][0]))
    for i in range(len(out) - 1):
        for j in range(i, -1, -1):
            lo, hi = out[j], out[j + 1]
            if abs(hi["pts"][0][1] - lo["pts"][0][1]) < row_tol and \
                    hi["pts"][0][0] < lo["pts"][0][0]:
                out[j], out[j + 1] = out[j + 1], out[j]
            else:
                break
    return out


# --------------------------------------------------------------------------- #
# Recognition oracle — mirrors rcdl::ctcGreedyDecode                           #
# --------------------------------------------------------------------------- #
def ref_ctc_greedy(logits, dictionary, blank=0, apply_softmax=False):
    """CTC best-path decode of a row-major [T, C] score matrix -> (text, score).

    Emit the step's argmax when it is neither the blank NOR a repeat of the
    previous step's argmax. The blank is not a character: it is the separator
    that keeps a genuinely doubled character from collapsing.
    """
    a = np.asarray(logits, dtype=np.float32)
    idx = a.argmax(axis=1)
    val = a.max(axis=1)
    text, confs, prev = "", [], -1
    for t, k in enumerate(idx):
        k = int(k)
        if k != blank and k != prev and k < len(dictionary):
            text += dictionary[k]
            if apply_softmax:
                e = np.exp(a[t] - val[t])
                confs.append(float(1.0 / e.sum()))
            else:
                confs.append(float(val[t]))
        prev = k
    return text, (float(np.mean(confs)) if confs else 0.0)


# --------------------------------------------------------------------------- #
# Output-layout resolution — mirrors the two rules in src/tasks/ocr.cc         #
# --------------------------------------------------------------------------- #
def ref_resolve_map_hwc(dims, nhwc):
    """(H, W, C) of a DB map tensor from its dims + format.

    The last three dims are (H,W,C) under NHWC and (C,H,W) otherwise — the
    runtime reports UNDEFINED for a plain NCHW logical layout, so anything that
    is not explicitly NHWC is NCHW. With C == 1 the two hold the SAME bytes in
    the same order; only the dims order differs, which is the whole reason the
    format has to be consulted instead of guessed.
    """
    if len(dims) >= 3:
        d0, d1, d2 = dims[-3], dims[-2], dims[-1]
        return (d0, d1, d2) if nhwc else (d1, d2, d0)
    if len(dims) == 2:
        return (dims[0], dims[1], 1)
    return None


def ref_resolve_rec_axes(dims, dict_size, time_major=True):
    """(T, C, time_major) of a recognition tensor from its dims + the dictionary.

    A 3-D tensor's format is UNDEFINED, so the dictionary is the discriminator:
    C is the class count by definition, so the axis whose length equals the
    dictionary size is the class axis. Only an UNAMBIGUOUS match decides.
    """
    d0, d1 = dims[-2], dims[-1]
    if dict_size > 0 and d1 == dict_size and d0 != dict_size:
        time_major = True
    elif dict_size > 0 and d0 == dict_size and d1 != dict_size:
        time_major = False
    return (d0, d1, True) if time_major else (d1, d0, False)


# --------------------------------------------------------------------------- #
# Optional compiled-module cross-checks                                        #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cxx():
    """The compiled module, or None — every test still asserts on the oracle."""
    try:
        import rcdl
    except Exception:
        return None
    return rcdl


def _cxx_boxes(cxx, prob, lb, **over):
    cfg = cxx.OcrDetConfig()
    for k, v in over.items():
        setattr(cfg, k, v)
    return cxx.decode_text_boxes(np.ascontiguousarray(prob, dtype=np.float32), cfg,
                                 lb.as_tuple())


# --------------------------------------------------------------------------- #
# CTC greedy decode                                                            #
# --------------------------------------------------------------------------- #
DICT = ["<blank>", "a", "b", "c", "d"]  # index 0 is the blank, as PaddleOCR's is


def _peaks(seq, num_classes=5, height=5.0):
    """[T, C] scores whose per-step argmax is exactly `seq`."""
    a = np.zeros((len(seq), num_classes), np.float32)
    for t, k in enumerate(seq):
        a[t, k] = height
    return a


def test_ctc_blank_separates_a_doubled_character():
    """THE rule: [a, a, blank, a] is "aa", not "a" and not "aaa".

    The first two steps are one character (a repeat), the blank ends it, and the
    fourth step starts a NEW one. Decoding blanks-then-dedupe gives "a"; not
    deduping at all gives "aaa". Both are wrong.
    """
    text, _ = ref_ctc_greedy(_peaks([1, 1, 0, 1]), DICT)
    assert text == "aa"


def test_ctc_collapses_repeats_and_drops_blanks():
    # a, a(repeat), blank, b, b(repeat), c  ->  "abc"
    text, score = ref_ctc_greedy(_peaks([1, 1, 0, 2, 2, 3]), DICT)
    assert text == "abc"
    assert score == pytest.approx(5.0)  # mean peak over the 3 emitting steps


def test_ctc_all_blank_is_empty_with_zero_score():
    text, score = ref_ctc_greedy(_peaks([0, 0, 0, 0]), DICT)
    assert text == ""
    assert score == 0.0


def test_ctc_adjacent_distinct_classes_both_emit():
    """No blank needed between DIFFERENT classes — only between equal ones."""
    text, _ = ref_ctc_greedy(_peaks([1, 2, 1, 2]), DICT)
    assert text == "abab"


def test_ctc_looks_up_the_dictionary_at_the_class_index_itself():
    """dict[idx], never dict[idx-1]: the blank occupies class 0 IN the table.

    An off-by-one here decodes every image to plausible-looking garbage, which is
    why the dictionaries in data/ carry the blank line themselves.
    """
    text, _ = ref_ctc_greedy(_peaks([4, 3, 2, 1]), DICT)
    assert text == "dcba"


def test_ctc_softmax_score_is_a_probability():
    logits = _peaks([1, 0, 2], height=3.0)
    _, raw = ref_ctc_greedy(logits, DICT)
    _, prob = ref_ctc_greedy(logits, DICT, apply_softmax=True)
    assert raw == pytest.approx(3.0)          # the raw logit, unbounded
    assert 0.0 < prob <= 1.0                   # softmaxed: a real confidence
    # exp(3) / (exp(3) + 4*exp(0)) for a 5-class step with one peak of 3.
    assert prob == pytest.approx(math.exp(3.0) / (math.exp(3.0) + 4.0), abs=1e-6)


def test_ctc_score_averages_only_the_emitting_steps():
    """A long blank run must not dilute the confidence of the characters."""
    a = _peaks([1, 0, 0, 0, 2], height=1.0)
    a[0, 1] = 0.8
    a[4, 2] = 0.6
    _, score = ref_ctc_greedy(a, DICT)
    assert score == pytest.approx((0.8 + 0.6) / 2)


def test_ctc_out_of_range_class_is_skipped_not_read():
    """A dictionary shorter than the head's vocabulary must not index past it."""
    short = ["<blank>", "a"]
    text, _ = ref_ctc_greedy(_peaks([1, 3, 1]), short)
    assert text == "aa"


def test_ctc_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "ctc_greedy_decode"):
        pytest.skip("compiled rcdl module without ctc_greedy_decode bindings")
    rng = np.random.default_rng(5)
    logits = rng.standard_normal((40, len(DICT))).astype(np.float32)
    text, score = ref_ctc_greedy(logits, DICT)
    got = cxx.ctc_greedy_decode(logits, DICT)
    assert got.text == text
    assert got.score == pytest.approx(score, abs=1e-5)


# --------------------------------------------------------------------------- #
# Unclip (box expansion)                                                       #
# --------------------------------------------------------------------------- #
def test_unclip_grows_a_rectangle_by_exactly_2d_on_each_axis():
    """A 10x4 rectangle at ratio 1.5: area 40, perimeter 28, d = 60/28.

    Each side moves out by d, so the box gains 2d in width AND in height. This is
    also exactly what a polygon clipper's offset gives for a rectangle, whatever
    its join style, which is why no clipper is needed to match the reference.
    """
    rect = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)]
    d = 40.0 * 1.5 / 28.0
    got = ref_unclip_quad(rect, 1.5)
    expect = [(-d, -d), (10.0 + d, -d), (10.0 + d, 4.0 + d), (-d, 4.0 + d)]
    for g, e in zip(got, expect):
        assert g[0] == pytest.approx(e[0], abs=1e-5)
        assert g[1] == pytest.approx(e[1], abs=1e-5)


def test_unclip_of_a_square_matches_the_closed_form():
    """side s at ratio r: d = s²r / 4s = s*r/4, so the side becomes s*(1 + r/2)."""
    s, r = 10.0, 2.0
    sq = [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)]
    got = ref_unclip_quad(sq, r)
    side = math.hypot(got[1][0] - got[0][0], got[1][1] - got[0][1])
    assert side == pytest.approx(s * (1.0 + r / 2.0), abs=1e-4)


def test_unclip_expands_regardless_of_winding():
    """Reversing the corner order must not shrink the box into itself."""
    rect = [(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0)]
    cw = ref_unclip_quad(rect, 1.5)
    ccw = ref_unclip_quad(list(reversed(rect)), 1.5)
    assert min(p[0] for p in cw) == pytest.approx(min(p[0] for p in ccw), abs=1e-5)
    assert max(p[0] for p in cw) == pytest.approx(max(p[0] for p in ccw), abs=1e-5)
    assert min(p[0] for p in cw) < 0.0  # grew outward, not inward


def test_unclip_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "unclip_quad"):
        pytest.skip("compiled rcdl module without unclip_quad bindings")
    rect = [(3.0, 5.0), (23.0, 5.0), (23.0, 12.0), (3.0, 12.0)]
    ref = ref_unclip_quad(rect, 1.6)
    got = np.asarray(cxx.unclip_quad(np.asarray(rect, np.float32), 1.6)).reshape(4, 2)
    np.testing.assert_allclose(got, np.asarray(ref, np.float32), atol=1e-4)


# --------------------------------------------------------------------------- #
# Minimum-area rectangle                                                       #
# --------------------------------------------------------------------------- #
def test_min_area_quad_recovers_an_axis_aligned_block():
    pts = [(x, y) for x in range(10, 25) for y in range(5, 15)]
    quad, sside = ref_min_area_quad(pts)
    assert sside == pytest.approx(9.0)  # 5..14 inclusive is 9 units across
    xs = sorted(p[0] for p in quad)
    ys = sorted(p[1] for p in quad)
    assert xs[0] == pytest.approx(10.0) and xs[-1] == pytest.approx(24.0)
    assert ys[0] == pytest.approx(5.0) and ys[-1] == pytest.approx(14.0)


def test_min_area_quad_beats_the_bounding_box_on_a_rotated_rect():
    """The point of a ROTATED fit: a 45-degree line's extent is twice its area."""
    w, h, ang = 40.0, 8.0, math.radians(30.0)
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for u in np.linspace(-w / 2, w / 2, 60):
        for v in (-h / 2, h / 2):
            pts.append((100 + u * ca - v * sa, 60 + u * sa + v * ca))
    quad, sside = ref_min_area_quad(pts)
    assert sside == pytest.approx(h, abs=1e-3)
    side_a = math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])
    side_b = math.hypot(quad[3][0] - quad[0][0], quad[3][1] - quad[0][1])
    assert sorted((side_a, side_b))[1] == pytest.approx(w, abs=1e-3)
    # the axis-aligned extent of the same points is much bigger — that's the win
    aabb = (max(p[0] for p in pts) - min(p[0] for p in pts)) * \
           (max(p[1] for p in pts) - min(p[1] for p in pts))
    assert aabb > 1.5 * (w * h)


def test_min_area_quad_orders_corners_tl_tr_br_bl():
    pts = [(x, y) for x in range(0, 20) for y in range(0, 6)]
    quad, _ = ref_min_area_quad(pts)
    tl, tr, br, bl = quad
    assert tl[0] <= tr[0] and bl[0] <= br[0]  # left pair before right pair
    assert tl[1] <= bl[1] and tr[1] <= br[1]  # top pair above bottom pair


def test_min_area_quad_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "min_area_quad"):
        pytest.skip("compiled rcdl module without min_area_quad bindings")
    rng = np.random.default_rng(9)
    pts = rng.uniform(0.0, 50.0, size=(200, 2)).astype(np.float32)
    ref, sside = ref_min_area_quad([tuple(p) for p in pts])
    got, got_sside = cxx.min_area_quad(pts)
    np.testing.assert_allclose(np.asarray(got).reshape(4, 2),
                               np.asarray(ref, np.float32), atol=1e-3)
    assert got_sside == pytest.approx(sside, abs=1e-3)


# --------------------------------------------------------------------------- #
# Probability-map thresholding                                                 #
# --------------------------------------------------------------------------- #
def _two_blocks(h=40, w=60, hot=0.9, bg=0.1):
    prob = np.full((h, w), bg, np.float32)
    prob[5:15, 10:25] = hot    # block 1
    prob[20:33, 35:55] = hot   # block 2
    return prob


def test_two_separated_blocks_become_two_boxes():
    boxes = ref_decode_text_boxes(_two_blocks())
    assert len(boxes) == 2

    def covers(cx, cy):
        return any(b["x1"] <= cx <= b["x2"] and b["y1"] <= cy <= b["y2"] for b in boxes)

    assert covers(17, 10)   # block 1's centre
    assert covers(45, 26)   # block 2's centre
    for b in boxes:
        assert b["x2"] > b["x1"] and b["y2"] > b["y1"]
        assert b["score"] == pytest.approx(0.9, abs=1e-3)


def test_everything_below_bin_thresh_yields_nothing():
    prob = np.full((32, 32), 0.1, np.float32)  # nothing above bin_thresh 0.3
    assert ref_decode_text_boxes(prob) == []


def test_bin_thresh_decides_what_a_region_is():
    """A block at 0.4 is background at bin_thresh 0.5 and a region at 0.3.

    box_thresh has to be lowered too, since a region's score is the mean of the
    same values — the point here is that the two thresholds answer different
    questions and are both live.
    """
    prob = np.full((30, 40), 0.05, np.float32)
    prob[5:20, 5:30] = 0.4
    assert len(ref_decode_text_boxes(prob, box_thresh=0.3, bin_thresh=0.3)) == 1
    assert ref_decode_text_boxes(prob, box_thresh=0.3, bin_thresh=0.5) == []


def test_box_thresh_drops_a_region_it_did_not_create():
    """0.45 is above bin_thresh (so it IS a region) but below box_thresh 0.6."""
    prob = np.full((30, 40), 0.05, np.float32)
    prob[5:20, 5:30] = 0.45
    assert ref_decode_text_boxes(prob, box_thresh=0.6) == []
    assert len(ref_decode_text_boxes(prob, box_thresh=0.4)) == 1


def test_touching_blocks_are_one_region_diagonal_ones_depend_on_connectivity():
    """8-connectivity joins a diagonal touch; 4-connectivity keeps them apart."""
    prob = np.full((40, 40), 0.05, np.float32)
    prob[4:14, 4:14] = 0.9
    prob[14:24, 14:24] = 0.9  # shares only the corner (13,13)-(14,14)
    # box_thresh is lowered because a single staircase-shaped region is scored
    # over its fitted rectangle, half of which is the background between them.
    assert len(ref_decode_text_boxes(prob, connectivity=8, box_thresh=0.3)) == 1
    assert len(ref_decode_text_boxes(prob, connectivity=4, box_thresh=0.3)) == 2


def test_min_size_drops_a_thin_region():
    """A bar 2 pixels across is below min_size 3 and never reaches scoring."""
    prob = np.full((30, 40), 0.05, np.float32)
    prob[10:13, 5:35] = 0.9  # rows 10..12, i.e. 2 units across
    assert ref_decode_text_boxes(prob) == []
    assert len(ref_decode_text_boxes(prob, min_size=1)) == 1


def test_apply_sigmoid_reads_a_logit_map():
    """The same scene, exported with and without the sigmoid in the graph."""
    prob = np.full((30, 40), 0.02, np.float32)
    prob[5:20, 5:30] = 0.95
    logits = np.log(prob / (1.0 - prob)).astype(np.float32)  # inverse sigmoid
    a = ref_decode_text_boxes(prob)
    b = ref_decode_text_boxes(logits, apply_sigmoid=True)
    assert len(a) == len(b) == 1
    assert b[0]["score"] == pytest.approx(a[0]["score"], abs=1e-4)
    assert b[0]["x1"] == pytest.approx(a[0]["x1"], abs=1e-4)


def test_decode_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "decode_text_boxes"):
        pytest.skip("compiled rcdl module without decode_text_boxes bindings")
    prob = _two_blocks()
    lb = compute_letterbox(60, 40, 60, 40)
    ref = ref_decode_text_boxes(prob, lb)
    got = _cxx_boxes(cxx, prob, lb)
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert a.score == pytest.approx(b["score"], abs=1e-4)
        np.testing.assert_allclose(np.asarray(a.pts).reshape(4, 2),
                                   np.asarray(b["pts"], np.float32), atol=1e-3)


# --------------------------------------------------------------------------- #
# Un-letterboxing: every emitted coordinate is an ORIGINAL-image pixel          #
# --------------------------------------------------------------------------- #
def test_quad_is_unletterboxed_through_a_downsampled_map():
    """160x160 map -> 640x640 canvas -> 1280x720 source, checked in closed form.

    Three transforms stacked, each of which is easy to drop: the map is a quarter
    of the canvas, the canvas is a half-scale fit of the source, and the source is
    letterboxed with 140 rows of padding top and bottom.
    """
    lb = compute_letterbox(1280, 720, 640, 640)
    assert lb.scale == pytest.approx(0.5)
    assert lb.pad_x == pytest.approx(0.0) and lb.pad_y == pytest.approx(140.0)

    prob = np.full((160, 160), 0.05, np.float32)
    prob[60:80, 20:60] = 0.9  # rows 60..79, cols 20..59 in MAP pixels
    boxes = ref_decode_text_boxes(prob, lb)
    assert len(boxes) == 1

    # The fitted rectangle spans the block's pixel centres: 39 x 19 map pixels.
    w, h = 59.0 - 20.0, 79.0 - 60.0
    d = (w * h) * 1.5 / (2.0 * (w + h))
    left, right = 20.0 - d, 59.0 + d
    top, bottom = 60.0 - d, 79.0 + d
    # map -> canvas is a plain 4x (640/160); canvas -> source is the letterbox
    # inverse, which subtracts the padding BEFORE dividing by the scale.
    to_x = lambda px: (px * 4.0 - lb.pad_x) / lb.scale
    to_y = lambda py: (py * 4.0 - lb.pad_y) / lb.scale
    tl, tr, br, bl = boxes[0]["pts"]
    assert tl[0] == pytest.approx(to_x(left), abs=1e-3)
    assert tl[1] == pytest.approx(to_y(top), abs=1e-3)
    assert br[0] == pytest.approx(to_x(right), abs=1e-3)
    assert br[1] == pytest.approx(to_y(bottom), abs=1e-3)
    assert tr[0] == pytest.approx(br[0], abs=1e-3) and tr[1] == pytest.approx(tl[1], abs=1e-3)
    assert bl[0] == pytest.approx(tl[0], abs=1e-3) and bl[1] == pytest.approx(br[1], abs=1e-3)
    # and it lands inside the source frame, which is the whole point
    assert 0.0 <= tl[0] < br[0] <= 1280.0
    assert 0.0 <= tl[1] < br[1] <= 720.0


def test_coordinates_are_clamped_to_the_source_extent():
    """A region against the canvas edge unclips PAST it; boxes must not."""
    lb = compute_letterbox(1280, 720, 640, 640)
    prob = np.full((160, 160), 0.05, np.float32)
    prob[35:50, 0:40] = 0.9  # flush against the left edge of the map
    boxes = ref_decode_text_boxes(prob, lb)
    assert len(boxes) == 1
    for x, y in boxes[0]["pts"]:
        assert 0.0 <= x <= 1280.0
        assert 0.0 <= y <= 720.0
    assert boxes[0]["x1"] == pytest.approx(0.0)


def test_identity_letterbox_returns_map_pixels():
    """A default-constructed letterbox means "no mapping" — used by the tests
    above and by anyone decoding a map they preprocessed themselves."""
    prob = np.full((40, 60), 0.1, np.float32)
    prob[10:20, 10:30] = 0.9
    boxes = ref_decode_text_boxes(prob)  # IDENTITY_LB
    assert len(boxes) == 1
    assert 0.0 < boxes[0]["x1"] < 10.0        # unclipped outward from x=10
    assert 30.0 < boxes[0]["x2"] < 60.0


# --------------------------------------------------------------------------- #
# Reading order                                                                #
# --------------------------------------------------------------------------- #
def _box_at(x, y):
    return {"pts": [(x, y), (x + 20, y), (x + 20, y + 8), (x, y + 8)],
            "x1": x, "y1": y, "x2": x + 20, "y2": y + 8, "score": 1.0}


def test_sort_is_top_to_bottom_then_left_to_right():
    boxes = [_box_at(100, 50), _box_at(10, 52), _box_at(60, 10)]
    got = ref_sort_text_boxes(boxes)
    assert [b["pts"][0] for b in got] == [(60, 10), (10, 52), (100, 50)]


def test_sort_keeps_rows_together_despite_a_ragged_top_edge():
    """Boxes 2px apart vertically are the SAME row; 30px apart are not."""
    boxes = [_box_at(200, 20), _box_at(10, 22), _box_at(10, 60)]
    got = ref_sort_text_boxes(boxes, row_tol=10.0)
    assert [b["pts"][0] for b in got] == [(10, 22), (200, 20), (10, 60)]


def test_sort_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "sort_text_boxes"):
        pytest.skip("compiled rcdl module without sort_text_boxes bindings")
    prob = _two_blocks()
    lb = compute_letterbox(60, 40, 60, 40)
    got = cxx.sort_text_boxes(_cxx_boxes(cxx, prob, lb))
    ref = ref_sort_text_boxes(ref_decode_text_boxes(prob, lb))
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert a.pts[0] == pytest.approx(b["pts"][0][0], abs=1e-3)


# --------------------------------------------------------------------------- #
# Output-layout resolution                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dims,nhwc,expect", [
    ([1, 1, 480, 480], False, (480, 480, 1)),   # NCHW  [N,C,H,W]
    ([1, 480, 480, 1], True, (480, 480, 1)),    # NHWC  [N,H,W,C]
    ([1, 1, 120, 200], False, (120, 200, 1)),   # non-square: H and W must not swap
    ([1, 120, 200, 1], True, (120, 200, 1)),
    ([480, 480], False, (480, 480, 1)),         # bare [H,W]
])
def test_db_map_hw_comes_from_the_format_not_a_guess(dims, nhwc, expect):
    assert ref_resolve_map_hwc(dims, nhwc) == expect


def test_db_map_layouts_hold_the_same_bytes():
    """Why the format is the ONLY thing that separates them: at C == 1, an NHWC
    and an NCHW map are the identical buffer — misreading the dims order swaps
    H and W, which silently transposes every box on a non-square map."""
    assert ref_resolve_map_hwc([1, 1, 120, 200], False)[:2] == (120, 200)
    assert ref_resolve_map_hwc([1, 1, 120, 200], True)[:2] == (1, 120)


@pytest.mark.parametrize("dims,expect", [
    ([1, 40, 6625], (40, 6625, True)),    # [1,T,C] — the class axis matches the dict
    ([1, 6625, 40], (40, 6625, False)),   # [1,C,T] — same, transposed
    ([40, 6625], (40, 6625, True)),       # no batch dim
])
def test_rec_axes_come_from_the_dictionary_size(dims, expect):
    assert ref_resolve_rec_axes(dims, 6625) == expect


def test_rec_axes_fall_back_to_the_config_when_the_dict_matches_neither():
    """A mismatched dictionary must not silently transpose the tensor: the
    configured default ([1,T,C], which every PP-OCR export is) still wins."""
    assert ref_resolve_rec_axes([1, 40, 6625], 99, time_major=True) == (40, 6625, True)
    assert ref_resolve_rec_axes([1, 40, 6625], 99, time_major=False) == (6625, 40, False)


# --------------------------------------------------------------------------- #
# Dictionary loading                                                           #
# --------------------------------------------------------------------------- #
def test_shipped_dictionary_has_the_blank_first_and_a_trailing_space(tmp_path):
    """The data/ dictionaries are ready-to-index tables: line i is class i.

    Skips when the file is not present so the suite stays green on a checkout
    without it.
    """
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "ppocr_keys_v1_6625.txt")
    if not os.path.isfile(path):
        pytest.skip("data/ppocr_keys_v1_6625.txt not present")
    with open(path, "rb") as f:
        lines = f.read().decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    assert lines[0] == "blank"          # class 0 is the CTC blank
    assert lines[-1] == " "             # the space token PaddleOCR appends
    assert len(lines) == 6625           # == the model's class count


def test_load_char_dict_matches_cxx(cxx, tmp_path):
    if cxx is None or not hasattr(cxx, "load_char_dict"):
        pytest.skip("compiled rcdl module without load_char_dict bindings")
    p = tmp_path / "keys.txt"
    p.write_text("a\nb\n", encoding="utf-8")
    assert list(cxx.load_char_dict(str(p))) == ["a", "b"]
    # paddle_special applies the reference's convention: blank first, space last
    assert list(cxx.load_char_dict(str(p), True)) == ["blank", "a", "b", " "]
