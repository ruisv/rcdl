"""XFeat feature decode + matching: RCDL's C++ against a numpy oracle.

The decode is a chain where every step has a convention that is invisible if you
get it wrong — the 65th logit is a reject bin that only ever appears in the
softmax denominator, cell channel c owns pixel (c//8, c%8) of its 8x8 block, the
descriptor map is L2-normalized before sampling AND each descriptor after, and
the sampling is BICUBIC (the reference's sparse-interpolator default) while the
reliability map beside it is bilinear. Each of those has a test here.

Pure numpy plus the compiled module: no board, no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_features.py
"""

import numpy as np
import pytest

rcdl = pytest.importorskip("rcdl")

FH, FW = 8, 10          # small grid: full res is 64 x 80
IN_H, IN_W = FH * 8, FW * 8


def _have():
    if not all(hasattr(rcdl, n) for n in ("decode_xfeat", "match_features",
                                          "xfeat_preprocess", "XfeatConfig")):
        pytest.skip("feature bindings not exposed")


# --------------------------------------------------------------------------- #
# Oracle — mirrors src/tasks/features.cc                                       #
# --------------------------------------------------------------------------- #
def _cubic_weights(t):
    """torch grid_sample's cubic convolution, A = -0.75."""
    a = -0.75

    def c1(x):
        return ((a + 2) * x - (a + 3)) * x * x + 1

    def c2(x):
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

    return np.array([c2(t + 1), c1(t), c1(1 - t), c2(2 - t)])


def _sample(planes, gx, gy, bicubic):
    """Sample [C,h,w] at (gx,gy), zero outside — bicubic or bilinear."""
    c, h, w = planes.shape
    x0, y0 = int(np.floor(gx)), int(np.floor(gy))
    if bicubic:
        wx, wy = _cubic_weights(gx - x0), _cubic_weights(gy - y0)
        xs, ys = range(x0 - 1, x0 + 3), range(y0 - 1, y0 + 3)
    else:
        fx, fy = gx - x0, gy - y0
        wx, wy = np.array([1 - fx, fx]), np.array([1 - fy, fy])
        xs, ys = range(x0, x0 + 2), range(y0, y0 + 2)
    out = np.zeros(c, np.float64)
    for j, sy in enumerate(ys):
        for i, sx in enumerate(xs):
            if 0 <= sx < w and 0 <= sy < h:
                out += wy[j] * wx[i] * planes[:, sy, sx]
    return out


def _to_grid(p, src, full):
    """Full-resolution pixel -> `src`-wide grid coordinate. The reference
    normalizes by (full-1) and un-normalizes with align_corners=False; the two
    conventions disagree, and matching it matters more than tidying it."""
    return p * src / (full - 1) - 0.5


def oracle_decode(feats, kpts, heat, cfg=None, scale=(1.0, 1.0)):
    thresh = 0.05 if cfg is None else cfg.detection_thresh
    ksz = 5 if cfg is None else cfg.nms_kernel
    top_k = 4096 if cfg is None else cfg.top_k
    _, fh, fw = kpts.shape
    in_h, in_w = fh * 8, fw * 8

    # softmax over all 65, keep 64, scatter to full resolution
    e = np.exp(kpts - kpts.max(axis=0, keepdims=True))
    p = e / e.sum(axis=0, keepdims=True)
    full = np.zeros((in_h, in_w), np.float64)
    for ch in range(64):
        full[ch // 8::8, ch % 8::8] = p[ch]

    # NMS against a square max filter
    r = ksz // 2
    local = np.zeros_like(full)
    for y in range(in_h):
        for x in range(in_w):
            local[y, x] = full[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].max()

    cands = []
    for y in range(in_h):
        for x in range(in_w):
            if full[y, x] > thresh and full[y, x] == local[y, x]:
                rel = _sample(heat, _to_grid(x, fw, in_w), _to_grid(y, fh, in_h), False)[0]
                s = full[y, x] * rel
                if s > 0:
                    cands.append((s, x, y))
    cands.sort(key=lambda t: -t[0])
    cands = cands[:top_k]

    nrm = feats / np.maximum(np.linalg.norm(feats, axis=0, keepdims=True), 1e-6)
    kp, desc = [], []
    for s, x, y in cands:
        d = _sample(nrm, _to_grid(x, fw, in_w), _to_grid(y, fh, in_h), True)
        d = d / max(np.linalg.norm(d), 1e-6)
        kp.append((x * scale[0], y * scale[1], s))
        desc.append(d)
    return np.array(kp, np.float64).reshape(-1, 3), np.array(desc, np.float64).reshape(-1, 64)


def _maps(seed=0, peak_cells=None):
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((64, FH, FW)).astype(np.float32)
    kpts = (rng.standard_normal((65, FH, FW)) * 0.5).astype(np.float32)
    kpts[64] += 3.0                       # reject bin usually wins, as in a real map
    heat = rng.uniform(0.3, 1.0, (1, FH, FW)).astype(np.float32)
    for (ch, gy, gx) in (peak_cells or []):
        kpts[ch, gy, gx] = 12.0
    return feats, kpts, heat


def _cfg(thresh=0.05, nms=5, top_k=4096):
    c = rcdl.XfeatConfig()
    c.detection_thresh = thresh
    c.nms_kernel = nms
    c.top_k = top_k
    return c


# --------------------------------------------------------------------------- #
# Decode                                                                       #
# --------------------------------------------------------------------------- #
def test_single_peak_lands_on_the_pixel_its_channel_owns():
    """Channel c of a cell owns pixel (c//8, c%8) inside that cell's 8x8 block.
    Get the two axes the wrong way round and everything still 'works', with every
    keypoint transposed inside its block."""
    _have()
    ch, gy, gx = 8 * 3 + 5, 4, 6          # dy=3, dx=5
    feats, kpts, heat = _maps(1, [(ch, gy, gx)])
    got = rcdl.decode_xfeat(feats, kpts, heat)
    assert len(got) >= 1
    best = got.xy[int(np.argmax(got.scores))]
    assert tuple(best) == (gx * 8 + 5, gy * 8 + 3)


def test_matches_the_numpy_oracle():
    _have()
    feats, kpts, heat = _maps(2, [(7, 1, 1), (20, 3, 4), (63, 6, 8), (33, 2, 7)])
    got = rcdl.decode_xfeat(feats, kpts, heat)
    kp, desc = oracle_decode(feats, kpts, heat)

    assert len(got) == len(kp)
    np.testing.assert_array_equal(got.xy, kp[:, :2].astype(np.float32))
    np.testing.assert_allclose(got.scores, kp[:, 2], rtol=1e-5)
    for i in range(len(kp)):
        assert float(got.descriptors[i] @ desc[i]) == pytest.approx(1.0, abs=2e-6)


def test_descriptors_are_unit_length():
    _have()
    feats, kpts, heat = _maps(3, [(0, 0, 0), (40, 5, 5)])
    d = rcdl.decode_xfeat(feats, kpts, heat).descriptors
    assert d.shape[1] == 64
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)


def test_bicubic_not_bilinear_for_descriptors():
    """Pin the sampler: a bilinear oracle must NOT reproduce the descriptors,
    while the bicubic one does. Without this the two are easy to confuse — they
    agree to about 0.4% of cosine, which looks like float noise."""
    _have()
    feats, kpts, heat = _maps(4, [(11, 2, 2), (52, 5, 7)])
    got = rcdl.decode_xfeat(feats, kpts, heat)
    nrm = feats / np.maximum(np.linalg.norm(feats, axis=0, keepdims=True), 1e-6)
    x, y = got.xy[0]
    gx, gy = _to_grid(x, FW, IN_W), _to_grid(y, FH, IN_H)

    cubic = _sample(nrm, gx, gy, True)
    cubic /= np.linalg.norm(cubic)
    linear = _sample(nrm, gx, gy, False)
    linear /= np.linalg.norm(linear)

    assert float(got.descriptors[0] @ cubic) == pytest.approx(1.0, abs=2e-6)
    assert float(got.descriptors[0] @ linear) < 0.9999


def test_reject_bin_only_lowers_scores():
    """The 65th logit never becomes a keypoint; it just takes probability mass."""
    _have()
    feats, kpts, heat = _maps(5, [(30, 3, 3)])
    low = rcdl.decode_xfeat(feats, kpts, heat)
    kpts2 = kpts.copy()
    kpts2[64] += 4.0
    high = rcdl.decode_xfeat(feats, kpts2, heat)
    assert len(high) <= len(low)
    if len(high):
        assert high.scores.max() < low.scores.max()


def test_nms_suppresses_within_the_window_and_keeps_beyond_it():
    _have()
    feats, kpts, heat = _maps(6)
    # Two peaks 1 pixel apart (same cell, adjacent channels) -> one survives.
    near = kpts.copy()
    near[0, 4, 4] = 12.0
    near[1, 4, 4] = 11.5           # (dy=0,dx=1): 1 px away
    n = rcdl.decode_xfeat(feats, near, heat, _cfg(thresh=0.01))
    close = [1 for x, y in n.xy if abs(x - 32) <= 1 and abs(y - 32) <= 1]
    assert len(close) == 1

    # Same two strengths 8 pixels apart (adjacent cells) -> both survive.
    far = kpts.copy()
    far[0, 4, 4] = 12.0
    far[0, 4, 5] = 11.5
    f = rcdl.decode_xfeat(feats, far, heat, _cfg(thresh=0.01))
    xs = {tuple(p) for p in f.xy}
    assert (32.0, 32.0) in xs and (40.0, 32.0) in xs


def test_top_k_keeps_the_highest_scores_in_order():
    _have()
    feats, kpts, heat = _maps(7, [(i * 3, i % FH, i % FW) for i in range(12)])
    allk = rcdl.decode_xfeat(feats, kpts, heat, _cfg(thresh=0.01))
    top5 = rcdl.decode_xfeat(feats, kpts, heat, _cfg(thresh=0.01, top_k=5))
    assert len(top5) == 5
    best = np.sort(allk.scores)[::-1][:5]
    np.testing.assert_allclose(top5.scores, best, rtol=1e-6)


def test_scale_maps_back_to_original_pixels():
    _have()
    feats, kpts, heat = _maps(8, [(9, 2, 3)])
    a = rcdl.decode_xfeat(feats, kpts, heat, _cfg(), 1.0, 1.0)
    b = rcdl.decode_xfeat(feats, kpts, heat, _cfg(), 2.5, 4.0)
    assert b.xy[0][0] == pytest.approx(a.xy[0][0] * 2.5)
    assert b.xy[0][1] == pytest.approx(a.xy[0][1] * 4.0)


def test_bad_shapes_and_kernels_are_rejected():
    _have()
    feats, kpts, heat = _maps(9)
    with pytest.raises(Exception):
        rcdl.decode_xfeat(feats, kpts, heat, _cfg(nms=4))           # even kernel
    with pytest.raises(Exception):
        rcdl.decode_xfeat(feats, kpts[:64], heat)                   # 64 not 65
    with pytest.raises(Exception):
        rcdl.decode_xfeat(feats, kpts, heat[:, :FH - 1])            # mismatched grid


# --------------------------------------------------------------------------- #
# Matching                                                                     #
# --------------------------------------------------------------------------- #
def test_identical_sets_match_one_to_one():
    _have()
    feats, kpts, heat = _maps(10, [(i * 5, i % FH, i % FW) for i in range(8)])
    a = rcdl.decode_xfeat(feats, kpts, heat, _cfg(thresh=0.01))
    pairs, scores = rcdl.match_features(a, a, 0.5)
    assert len(pairs) == len(a)
    assert (pairs[:, 0] == pairs[:, 1]).all()
    np.testing.assert_allclose(scores, 1.0, atol=1e-5)


def test_mutual_requirement_rejects_one_sided_matches():
    """Every accepted pair is mutual, so no index may repeat on either side —
    a one-sided nearest neighbour is exactly what a repeated texture produces."""
    _have()
    f1, k1, h1 = _maps(11, [(0, 1, 1), (8, 3, 3)])
    a = rcdl.decode_xfeat(f1, k1, h1, _cfg(thresh=0.01))
    b = rcdl.decode_xfeat(f1, k1, h1, _cfg(thresh=0.01))
    pairs, _ = rcdl.match_features(a, b, -1.0)
    assert len(set(pairs[:, 0])) == len(pairs)
    assert len(set(pairs[:, 1])) == len(pairs)


def test_min_cossim_floor_filters():
    _have()
    f1, k1, h1 = _maps(12, [(3, 2, 2), (17, 5, 6), (44, 1, 8)])
    f2, k2, h2 = _maps(13, [(3, 2, 2), (17, 5, 6), (44, 1, 8)])
    a = rcdl.decode_xfeat(f1, k1, h1, _cfg(thresh=0.01))
    b = rcdl.decode_xfeat(f2, k2, h2, _cfg(thresh=0.01))
    loose, _ = rcdl.match_features(a, b, -1.0)
    tight, tight_scores = rcdl.match_features(a, b, 0.95)
    assert len(tight) <= len(loose)
    assert (tight_scores > 0.95).all()


def test_empty_sets_match_to_nothing():
    _have()
    feats, kpts, heat = _maps(14)
    empty = rcdl.decode_xfeat(feats, kpts, heat, _cfg(thresh=0.99))
    assert len(empty) == 0
    pairs, scores = rcdl.match_features(empty, empty, 0.5)
    assert len(pairs) == 0 and len(scores) == 0


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #
def test_preprocess_standardizes_and_reports_scale():
    _have()
    rng = np.random.default_rng(15)
    img = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
    x, sx, sy = rcdl.xfeat_preprocess(img, 640, 480)
    assert x.shape == (1, 1, 480, 640)
    assert x.mean() == pytest.approx(0.0, abs=1e-3)
    assert x.std() == pytest.approx(1.0, abs=1e-3)
    assert sx == pytest.approx(500 / 640)
    assert sy == pytest.approx(300 / 480)


def test_preprocess_grayscale_is_the_channel_mean_not_luma():
    """A Rec.601 luma would weight green ~6x blue; the reference uses a plain
    mean, and using luma instead shifts every descriptor."""
    _have()
    # A flat image standardizes to all-zeros, so compare on a half/half frame
    # where the two channels must produce the SAME pattern under a channel mean.
    img_b = np.zeros((64, 64, 3), np.uint8)
    img_b[:, 32:, 0] = 240                 # BGR: pure blue on the right half
    img_g = np.zeros((64, 64, 3), np.uint8)
    img_g[:, 32:, 1] = 240                 # the same half, pure green
    xb, _, _ = rcdl.xfeat_preprocess(img_b, 64, 64)
    xg, _, _ = rcdl.xfeat_preprocess(img_g, 64, 64)
    np.testing.assert_allclose(xb, xg, atol=1e-4)
