"""Monocular-depth post-processing tests.

PURE-NUMPY tests of the reference that ``rcdl::decodeDepth`` / ``depthResize`` /
``depthToSource`` / ``depthToGray8`` mirror. Need only numpy: no board, no
``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_depth.py

A depth head is one dense single-channel map, and the three things that make it
awkward are all pinned below:

  * SHAPE — [1,1,H,W] and [1,H,W,1] are the same H*W plane (one channel makes
    the two channel orders identical), and so are [1,H,W] and [H,W];
  * UNITS — relative-depth models emit inverse depth (disparity) on an arbitrary
    scale, metric models emit real distances. The affine/inverse/clip/normalise
    ORDER is what decides whether a clip threshold means anything;
  * NORMALISATION — a per-frame min-max stretch to [0,1], with the reported
    vmin/vmax being the range observed AFTER the clip and BEFORE the stretch,
    which is what lets a caller undo it.
"""

import numpy as np
import pytest

from test_detection import compute_letterbox  # one letterbox oracle, shared


# --------------------------------------------------------------------------- #
# Reference decode                                                             #
# --------------------------------------------------------------------------- #
def ref_resolve_hw(shape):
    """Mirrors the shape reduction in src/tasks/depth.cc: drop the unit dims and
    take the last two. This is why depth needs no channel-order flag."""
    dims = [d for d in shape if d > 1]
    if len(dims) >= 2:
        return dims[-2], dims[-1]
    return (shape[-2] if len(shape) >= 2 else shape[-1]), shape[-1]


def ref_decode_depth(data, shape, scale=1.0, shift=0.0, inverse=False, inverse_eps=1e-6,
                     clip_lo=0.0, clip_hi=0.0, normalize=True):
    """Mirrors rcdl::decodeDepth. Returns (map HxW float32, vmin, vmax).

    ORDER MATTERS and is: affine -> inverse -> clip -> observe vmin/vmax ->
    normalise. Clipping before the affine would mean the thresholds are in the
    head's arbitrary units; observing the range after normalising would report
    (0, 1) and destroy the information.
    """
    h, w = ref_resolve_hw(list(shape))
    v = np.asarray(data, dtype=np.float64).reshape(-1)[:h * w] * scale + shift
    if inverse:
        v = 1.0 / np.maximum(v, inverse_eps)
    if clip_hi > clip_lo:
        v = np.clip(v, clip_lo, clip_hi)
    vmin, vmax = float(v.min()), float(v.max())
    if normalize:
        rng = vmax - vmin
        # A flat map has no range to stretch: all zeros beats a division by zero,
        # and vmin/vmax still say what the constant was.
        v = (v - vmin) / rng if rng > 0.0 else np.zeros_like(v)
    return v.reshape(h, w).astype(np.float32), vmin, vmax


def ref_axis_lerp(dst_n, src_n):
    """Pixel-centre bilinear map, dst index -> (i0, i1, weight), clamped."""
    d = np.arange(dst_n, dtype=np.float64)
    f = np.clip((d + 0.5) * src_n / dst_n - 0.5, 0.0, src_n - 1)
    i0 = np.floor(f).astype(np.int64)
    return i0, np.minimum(i0 + 1, src_n - 1), f - i0


def _resample(m, sx, sy):
    (ix0, ix1, wx), (iy0, iy1, wy) = sx, sy
    top = m[np.ix_(iy0, ix0)] * (1.0 - wx) + m[np.ix_(iy0, ix1)] * wx
    bot = m[np.ix_(iy1, ix0)] * (1.0 - wx) + m[np.ix_(iy1, ix1)] * wx
    return (top * (1.0 - wy)[:, None] + bot * wy[:, None]).astype(np.float32)


def ref_depth_resize(m, dst_w, dst_h):
    """Mirrors rcdl::depthResize — BILINEAR, because depth is continuous (the
    opposite of a label map, where interpolation would invent a class)."""
    sh, sw = m.shape
    return _resample(np.asarray(m, np.float64),
                     ref_axis_lerp(dst_w, sw), ref_axis_lerp(dst_h, sh))


def ref_depth_to_source(m, lb):
    """Mirrors rcdl::depthToSource: push each SOURCE pixel centre forward through
    the letterbox and sample the map there, so the padding never contributes."""
    mh, mw = m.shape
    fx = np.clip(((np.arange(lb.src_w) + 0.5) * lb.scale + lb.pad_x) * mw / lb.dst_w - 0.5,
                 0.0, mw - 1)
    fy = np.clip(((np.arange(lb.src_h) + 0.5) * lb.scale + lb.pad_y) * mh / lb.dst_h - 0.5,
                 0.0, mh - 1)
    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)
    sx = (ix0, np.minimum(ix0 + 1, mw - 1), fx - ix0)
    sy = (iy0, np.minimum(iy0 + 1, mh - 1), fy - iy0)
    return _resample(np.asarray(m, np.float64), sx, sy)


def ref_depth_to_gray8(m):
    """Mirrors rcdl::depthToGray8: stretch by the map's OWN observed range, so it
    renders whether the data is raw or already normalised."""
    a = np.asarray(m, np.float64)
    lo, hi = a.min(), a.max()
    if not hi > lo:
        return np.zeros(a.shape, np.uint8)
    t = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return np.floor(t * 255.0 + 0.5).astype(np.uint8)  # lround, not banker's


# --------------------------------------------------------------------------- #
# Optional C++ cross-check                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cxx():
    try:
        import rcdl
    except Exception:
        return None
    return rcdl if hasattr(rcdl, "decode_depth") else None


# --------------------------------------------------------------------------- #
# Shape handling                                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", [(1, 1, 3, 4), (1, 3, 4, 1), (1, 3, 4), (3, 4)])
def test_every_single_channel_shape_gives_the_same_map(shape):
    """[1,1,H,W] (NCHW) and [1,H,W,1] (NHWC) are byte-identical when C == 1, so a
    depth decoder needs no channel-order flag — only a shape reduction."""
    raw = np.arange(12, dtype=np.float32)
    got, vmin, vmax = ref_decode_depth(raw, shape, normalize=False)
    assert got.shape == (3, 4)
    np.testing.assert_array_equal(got, raw.reshape(3, 4))
    assert (vmin, vmax) == (0.0, 11.0)


def test_resolve_hw_drops_unit_dims():
    assert ref_resolve_hw([1, 1, 518, 518]) == (518, 518)
    assert ref_resolve_hw([1, 37, 37, 1]) == (37, 37)
    assert ref_resolve_hw([1, 64, 80]) == (64, 80)


# --------------------------------------------------------------------------- #
# Normalisation                                                                #
# --------------------------------------------------------------------------- #
def test_normalize_stretches_to_exactly_zero_and_one():
    """THE normalisation rule: (v - vmin) / (vmax - vmin), per frame, with
    vmin/vmax being the values actually observed in THIS map."""
    raw = np.array([2.0, 4.0, 6.0, 10.0], np.float32)
    got, vmin, vmax = ref_decode_depth(raw, (1, 1, 1, 4))
    assert (vmin, vmax) == (2.0, 10.0)
    np.testing.assert_allclose(got[0], [0.0, 0.25, 0.5, 1.0], atol=1e-6)
    assert got.min() == 0.0 and got.max() == 1.0


def test_normalize_off_keeps_raw_values():
    raw = np.array([2.0, 4.0, 6.0, 10.0], np.float32)
    got, vmin, vmax = ref_decode_depth(raw, (1, 1, 1, 4), normalize=False)
    np.testing.assert_allclose(got[0], raw, atol=1e-6)
    assert (vmin, vmax) == (2.0, 10.0)


def test_flat_map_normalizes_to_zeros_not_nan():
    """vmax == vmin would divide by zero; the range still has to be reported."""
    got, vmin, vmax = ref_decode_depth(np.full(6, 3.25, np.float32), (1, 1, 2, 3))
    assert (vmin, vmax) == (3.25, 3.25)
    assert not np.isnan(got).any()
    assert (got == 0.0).all()


def test_clip_happens_before_the_range_is_observed():
    """The clip is what stops a handful of wild pixels flattening the whole
    stretch — which only works if vmin/vmax are read AFTER it."""
    raw = np.array([-50.0, 1.0, 2.0, 3.0, 900.0], np.float32)
    got, vmin, vmax = ref_decode_depth(raw, (1, 1, 1, 5), clip_lo=0.0, clip_hi=4.0)
    assert (vmin, vmax) == (0.0, 4.0)
    np.testing.assert_allclose(got[0], [0.0, 0.25, 0.5, 0.75, 1.0], atol=1e-6)


def test_clip_is_off_when_hi_is_not_above_lo():
    raw = np.array([-50.0, 900.0], np.float32)
    _, vmin, vmax = ref_decode_depth(raw, (1, 1, 1, 2), clip_lo=0.0, clip_hi=0.0,
                                     normalize=False)
    assert (vmin, vmax) == (-50.0, 900.0)


# --------------------------------------------------------------------------- #
# Units: metric and relative / inverse depth                                   #
# --------------------------------------------------------------------------- #
def test_affine_converts_to_physical_units():
    """A metric head reporting millimetres becomes metres with scale = 0.001 —
    and normalisation is exactly what you must NOT do to it."""
    mm = np.array([1000.0, 2500.0, 8000.0], np.float32)
    got, vmin, vmax = ref_decode_depth(mm, (1, 1, 1, 3), scale=0.001, normalize=False)
    np.testing.assert_allclose(got[0], [1.0, 2.5, 8.0], atol=1e-6)
    assert (vmin, vmax) == (1.0, 8.0)


def test_inverse_turns_disparity_into_depth():
    """Relative-depth heads emit inverse depth: big = near. Inverting makes the
    map a depth (big = far), which also REVERSES the order of the pixels."""
    disp = np.array([0.5, 0.25, 0.125], np.float32)  # near -> far
    got, vmin, vmax = ref_decode_depth(disp, (1, 1, 1, 3), inverse=True, normalize=False)
    np.testing.assert_allclose(got[0], [2.0, 4.0, 8.0], atol=1e-6)
    assert (vmin, vmax) == (2.0, 8.0)
    # Normalised, the nearest pixel is now 0 and the farthest 1 — the opposite of
    # what the same map gives without the inversion.
    norm, _, _ = ref_decode_depth(disp, (1, 1, 1, 3), inverse=True)
    assert norm[0, 0] == 0.0 and norm[0, 2] == 1.0
    plain, _, _ = ref_decode_depth(disp, (1, 1, 1, 3))
    assert plain[0, 0] == 1.0 and plain[0, 2] == 0.0


def test_inverse_eps_guards_zero_disparity():
    """A zero (or negative) disparity is infinitely far away; the epsilon turns
    that into a large finite number instead of an inf that poisons the whole
    min-max stretch."""
    got, _, vmax = ref_decode_depth(np.array([0.0, 0.5], np.float32), (1, 1, 1, 2),
                                    inverse=True, inverse_eps=1e-3, normalize=False)
    assert np.isfinite(got).all()
    assert vmax == pytest.approx(1000.0)


def test_affine_is_applied_before_the_inverse():
    """v = 1 / (raw*scale + shift), not raw*scale + shift applied to 1/raw."""
    got, _, _ = ref_decode_depth(np.array([1.0], np.float32), (1, 1, 1, 1),
                                 scale=2.0, shift=2.0, inverse=True, normalize=False)
    assert got[0, 0] == pytest.approx(0.25)  # 1/(1*2 + 2)


# --------------------------------------------------------------------------- #
# Resampling to source pixels                                                  #
# --------------------------------------------------------------------------- #
def test_depth_resize_reproduces_a_linear_ramp_exactly():
    """Bilinear interpolation is exact on a linear function, so an x-ramp
    resampled onto a finer grid must equal the analytic map — which pins the
    pixel-centre convention, not just 'something smooth happened'."""
    src_w, src_h, dst_w = 8, 4, 16
    ramp = np.tile(np.arange(src_w, dtype=np.float64), (src_h, 1))
    got = ref_depth_resize(ramp, dst_w, src_h)
    want = np.clip((np.arange(dst_w) + 0.5) * src_w / dst_w - 0.5, 0.0, src_w - 1)
    np.testing.assert_allclose(got[0], want, atol=1e-5)


def test_depth_resize_replicates_the_edges():
    """Beyond the outermost sample centres there is nothing to interpolate with,
    so the edge value is held rather than extrapolated."""
    got = ref_depth_resize(np.array([[0.0, 10.0]]), 4, 1)
    assert got[0, 0] == pytest.approx(0.0)
    assert got[0, -1] == pytest.approx(10.0)


def test_depth_to_source_strips_the_letterbox_padding():
    """A canvas-resolution x-ramp projected back must equal the ramp evaluated at
    each source pixel's canvas position — exactly, because bilinear is exact on
    a linear function.

    128x64 source into a 64x64 canvas: scale 0.5, no x padding, 16 px of y
    padding. Forgetting to strip that padding shifts the map by 32 source rows.
    """
    lb = compute_letterbox(128, 64, 64, 64)
    assert (lb.scale, lb.pad_x, lb.pad_y) == (0.5, 0.0, 16.0)
    canvas = np.tile(np.arange(64, dtype=np.float64), (64, 1))  # value == canvas x

    got = ref_depth_to_source(canvas, lb)
    assert got.shape == (64, 128)
    # Clipped at the ends: past the outermost sample centre there is nothing to
    # interpolate with, so the edge value is held.
    want_x = np.clip((np.arange(128) + 0.5) * lb.scale + lb.pad_x - 0.5, 0.0, 63.0)
    np.testing.assert_allclose(got[0], want_x, atol=1e-4)
    # Every row is the same ramp: the y padding contributed nothing.
    np.testing.assert_allclose(got[0], got[-1], atol=1e-4)


def test_depth_to_source_reads_only_inside_the_image_band():
    """Whatever the model predicted in the grey bars must not reach the frame.

    The very first and last source rows still straddle the seam by up to half a
    map pixel — that is inherent to interpolating AT a boundary, and the
    alternative (replicating the edge) would be a different kind of wrong. Every
    other row must see nothing but image.
    """
    lb = compute_letterbox(128, 64, 64, 64)
    canvas = np.zeros((64, 64), np.float64)
    canvas[:16, :] = 1000.0   # top padding band
    canvas[48:, :] = 1000.0   # bottom padding band
    assert ref_depth_to_source(canvas, lb)[1:-1].max() == 0.0


def test_depth_to_source_handles_a_coarse_map():
    """The head's grid is usually much smaller than the canvas (a 518-input
    transformer emits 37x37); the projection must still cover the whole frame."""
    lb = compute_letterbox(200, 100, 64, 64)
    coarse = np.tile(np.arange(16, dtype=np.float64), (16, 1))
    got = ref_depth_to_source(coarse, lb)
    assert got.shape == (100, 200)
    assert np.all(np.diff(got[0]) >= -1e-6)  # still monotone in x


# --------------------------------------------------------------------------- #
# Visualisation                                                                #
# --------------------------------------------------------------------------- #
def test_gray8_spans_the_full_range():
    m = np.array([[1.0, 2.0, 3.0]], np.float32)
    g = ref_depth_to_gray8(m)
    assert (g[0, 0], g[0, -1]) == (0, 255)
    assert g[0, 1] == 128  # lround(0.5*255) = 128, half away from zero


def test_gray8_of_a_flat_map_is_all_zeros():
    assert (ref_depth_to_gray8(np.full((2, 2), 7.0, np.float32)) == 0).all()


def test_gray8_is_the_same_before_and_after_normalisation():
    """Because gray8 stretches by the map's own range, a raw map and its
    normalised twin render identically."""
    raw = np.array([2.0, 4.0, 6.0, 10.0], np.float32)
    a, _, _ = ref_decode_depth(raw, (1, 1, 1, 4), normalize=False)
    b, _, _ = ref_decode_depth(raw, (1, 1, 1, 4), normalize=True)
    np.testing.assert_array_equal(ref_depth_to_gray8(a), ref_depth_to_gray8(b))


# --------------------------------------------------------------------------- #
# Opportunistic C++ cross-check                                                #
# --------------------------------------------------------------------------- #
def test_decode_depth_matches_cxx(cxx):
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_depth bindings")
    rng = np.random.default_rng(31)
    raw = rng.uniform(0.1, 12.0, size=(1, 1, 24, 37)).astype(np.float32)
    ref, vmin, vmax = ref_decode_depth(raw, raw.shape, clip_lo=0.5, clip_hi=9.0)
    got = cxx.decode_depth(np.ascontiguousarray(raw), clip_lo=0.5, clip_hi=9.0,
                           normalize=True)
    assert (got.width, got.height) == (37, 24)
    assert got.vmin == pytest.approx(vmin, abs=1e-5)
    assert got.vmax == pytest.approx(vmax, abs=1e-5)
    np.testing.assert_allclose(np.asarray(got.data).reshape(24, 37), ref, atol=1e-5)
