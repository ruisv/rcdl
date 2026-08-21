"""Semantic-segmentation post-processing tests.

PURE-NUMPY tests of the reference that ``rcdl::decodeSeg`` / ``segResize`` /
``segToSource`` / ``segColorize`` mirror. Need only numpy: no board, no
``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_segmentation.py

Three things are pinned deliberately, because they are the ones that silently
produce a plausible-looking but wrong mask:

  * the ARGMAX TIE-BREAK — ties are routine, not exotic, once a head is
    int8-quantized and whole regions saturate to the same code;
  * the QUANTIZED SHORTCUT — argmaxing the int8 codes instead of the
    dequantized floats is only valid because a per-tensor affine dequant is one
    strictly increasing map, and the tests below both confirm it and show the
    case (negative scale) where it would be wrong;
  * the LETTERBOX PROJECTION — a label map covers the padded canvas, and the
    padding has to be gone before it reaches source pixels.
"""

import numpy as np
import pytest

from test_detection import compute_letterbox  # one letterbox oracle, shared


def _lround(v):
    """C's lround: half away from zero (np.round is banker's rounding)."""
    return int(np.floor(v + 0.5)) if v >= 0 else -int(np.floor(-v + 0.5))


# --------------------------------------------------------------------------- #
# Reference decode                                                             #
# --------------------------------------------------------------------------- #
def ref_decode_seg(data, channels_first=True, num_classes=0, score="none"):
    """Mirrors rcdl::decodeSeg for a logit volume.

    ``data`` is [C,H,W] when ``channels_first``, else [H,W,C]. Returns
    (labels HxW int32, confidence HxW float or None).

    numpy's argmax keeps the FIRST maximum, i.e. the lowest channel index —
    which is exactly what the C++ scan does by only accepting a strictly greater
    value. The two tie-break the same way by construction, not by luck.
    """
    a = np.asarray(data, dtype=np.float64)
    v = a if channels_first else np.moveaxis(a, -1, 0)  # -> [C,H,W]
    if num_classes > 0:
        v = v[:min(num_classes, v.shape[0])]
    labels = v.argmax(axis=0).astype(np.int32)
    if score == "none":
        return labels, None
    best = v.max(axis=0)
    if score == "max":
        return labels, best.astype(np.float32)
    # softmax[winner] = 1 / Σ_c exp(v_c - v_winner): every exponent is <= 0, so
    # it cannot overflow and the denominator is >= 1.
    conf = 1.0 / np.exp(v - best).sum(axis=0)
    return labels, conf.astype(np.float32)


def ref_decode_seg_argmaxed(data):
    """Mirrors rcdl::decodeSeg with SegConfig::argmaxed: round float ids.

    floor(v + 0.5), not np.round: the C++ uses std::lround, which rounds halves
    AWAY from zero, while numpy rounds them to even. Ids are non-negative, so
    the two agree everywhere except exact .5, and this is that half.
    """
    return np.asarray(np.floor(np.asarray(data, dtype=np.float64) + 0.5), dtype=np.int32)


def ref_seg_resize(labels, dst_w, dst_h):
    """Mirrors rcdl::segResize: NEAREST, pixel-centre map.

        src = floor((dst + 0.5) * src_n / dst_n)

    Nearest and never bilinear: interpolating class id 3 and id 7 into 5 invents
    a class the model never predicted.
    """
    sh, sw = labels.shape
    sy = np.clip(((np.arange(dst_h) + 0.5) * sh / dst_h).astype(np.int64), 0, sh - 1)
    sx = np.clip(((np.arange(dst_w) + 0.5) * sw / dst_w).astype(np.int64), 0, sw - 1)
    return labels[np.ix_(sy, sx)]


def ref_seg_to_source(labels, lb):
    """Mirrors rcdl::segToSource.

    ``labels`` covers the whole model-input canvas (lb.dst_w x lb.dst_h) at its
    own resolution. Each SOURCE pixel centre is pushed forward through the
    letterbox and read back out of the map, so the padding never contributes:

        mx = round( (x + 0.5)*scale + pad_x ) * mw/dst_w - 0.5 )
    """
    mh, mw = labels.shape
    fx = ((np.arange(lb.src_w) + 0.5) * lb.scale + lb.pad_x) * mw / lb.dst_w - 0.5
    fy = ((np.arange(lb.src_h) + 0.5) * lb.scale + lb.pad_y) * mh / lb.dst_h - 0.5
    sx = np.clip([_lround(v) for v in fx], 0, mw - 1)
    sy = np.clip([_lround(v) for v in fy], 0, mh - 1)
    return labels[np.ix_(sy, sx)]


def ref_palette(idx):
    """Mirrors rcdl's paletteColor for the VOC range, returning RGB."""
    if idx <= 0:
        return (0, 0, 0)
    c, r, g, b = idx, 0, 0, 0
    for i in range(8):
        r |= ((c >> 0) & 1) << (7 - i)
        g |= ((c >> 1) & 1) << (7 - i)
        b |= ((c >> 2) & 1) << (7 - i)
        c >>= 3
    return (r, g, b)


# --------------------------------------------------------------------------- #
# Optional C++ cross-check                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cxx():
    try:
        import rcdl
    except Exception:
        return None
    return rcdl if hasattr(rcdl, "decode_seg") else None


# --------------------------------------------------------------------------- #
# Argmax and its tie-break                                                     #
# --------------------------------------------------------------------------- #
def test_argmax_picks_the_largest_channel():
    logits = np.zeros((4, 2, 3), np.float32)
    logits[2, 0, 0] = 1.0
    logits[3, 1, 2] = 5.0
    labels, conf = ref_decode_seg(logits)
    assert labels.shape == (2, 3)
    assert labels[0, 0] == 2
    assert labels[1, 2] == 3
    assert labels[0, 1] == 0  # all-zero pixel -> the tie rule below
    assert conf is None


def test_argmax_ties_go_to_the_lowest_channel():
    """THE tie-break rule: a strictly-greater comparison, so the first (lowest)
    channel wins. Pinned explicitly because an int8 head saturates whole regions
    to one code, making exact ties the common case rather than a curiosity."""
    logits = np.zeros((5, 1, 4), np.float32)
    logits[:, 0, 0] = 7.0            # every channel ties
    logits[1:4, 0, 1] = 3.0          # channels 1..3 tie, 0 and 4 lose
    logits[4, 0, 2] = 2.0            # a lone winner in the LAST channel
    logits[0, 0, 3] = 2.0            # a lone winner in the FIRST channel
    labels, _ = ref_decode_seg(logits)
    assert list(labels[0]) == [0, 1, 4, 0]


def test_num_classes_only_narrows_the_argmax():
    """A caller-supplied class count may cut the search short but must never
    widen it past what the tensor holds — that would read out of bounds."""
    logits = np.zeros((6, 1, 1), np.float32)
    logits[5, 0, 0] = 9.0
    assert ref_decode_seg(logits)[0][0, 0] == 5
    assert ref_decode_seg(logits, num_classes=3)[0][0, 0] == 0  # channel 5 excluded
    assert ref_decode_seg(logits, num_classes=99)[0][0, 0] == 5  # clamped, not extended


@pytest.mark.parametrize("channels_first", [True, False])
def test_layout_invariance(channels_first):
    """[1,C,H,W] and [1,H,W,C] hold the same numbers in a different order; only
    the tensor's fmt says which, and both must decode to the same labels."""
    rng = np.random.default_rng(5)
    chw = rng.standard_normal((7, 5, 9)).astype(np.float32)
    hwc = np.ascontiguousarray(np.moveaxis(chw, 0, -1))
    got, _ = ref_decode_seg(chw if channels_first else hwc, channels_first=channels_first)
    np.testing.assert_array_equal(got, chw.argmax(axis=0))


def test_argmaxed_passthrough_rounds_to_ids():
    """Some exports argmax inside the graph and emit float ids."""
    ids = np.array([[0.0, 2.4, 2.6], [17.0, 0.49, 3.5]], np.float32)
    np.testing.assert_array_equal(ref_decode_seg_argmaxed(ids), [[0, 2, 3], [17, 0, 4]])


# --------------------------------------------------------------------------- #
# Confidence                                                                   #
# --------------------------------------------------------------------------- #
def test_softmax_confidence_matches_the_definition():
    logits = np.array([[[0.0]], [[np.log(3.0)]], [[np.log(6.0)]]], np.float32)  # [3,1,1]
    labels, conf = ref_decode_seg(logits, score="softmax")
    assert labels[0, 0] == 2
    assert conf[0, 0] == pytest.approx(6.0 / 10.0, abs=1e-6)


def test_softmax_confidence_of_a_flat_pixel_is_uniform():
    labels, conf = ref_decode_seg(np.zeros((4, 1, 1), np.float32), score="softmax")
    assert labels[0, 0] == 0
    assert conf[0, 0] == pytest.approx(0.25, abs=1e-6)


def test_max_confidence_returns_the_raw_winner():
    """For an export whose softmax is already in the graph, the winning value IS
    the probability and must be reported untouched."""
    probs = np.array([[[0.1]], [[0.7]], [[0.2]]], np.float32)
    labels, conf = ref_decode_seg(probs, score="max")
    assert labels[0, 0] == 1
    assert conf[0, 0] == pytest.approx(0.7, abs=1e-6)


# --------------------------------------------------------------------------- #
# The quantized-argmax shortcut                                                #
# --------------------------------------------------------------------------- #
def test_quantized_argmax_equals_float_argmax():
    """WHY the C++ fast path is allowed to argmax int8 codes directly: an RKNN
    affine-asymmetric output carries ONE (scale, zero_point) for the whole
    tensor, so dequantization is the single map q -> (q - zp)*scale applied
    identically to every channel. With scale > 0 that map is strictly
    increasing, hence order- AND tie-preserving."""
    rng = np.random.default_rng(17)
    q = rng.integers(-128, 128, size=(19, 32, 48)).astype(np.int8)
    for scale, zp in ((0.0234, -13), (1.0, 0), (7.5, 127)):
        deq = (q.astype(np.float64) - zp) * scale
        np.testing.assert_array_equal(q.argmax(axis=0), ref_decode_seg(deq)[0])


def test_quantized_argmax_is_wrong_for_a_negative_scale():
    """The guard earns its keep: a negative scale reverses the order, so the
    shortcut would pick the WRONG class. The C++ path refuses scale <= 0."""
    q = np.array([[[10]], [[-20]]], np.int8)
    deq = (q.astype(np.float64) - 0) * -0.5
    assert q.argmax(axis=0)[0, 0] == 0
    assert ref_decode_seg(deq)[0][0, 0] == 1


# --------------------------------------------------------------------------- #
# Resampling to source pixels                                                  #
# --------------------------------------------------------------------------- #
def test_seg_resize_is_nearest_and_pixel_centred():
    labels = np.array([[0, 1], [2, 3]], np.int32)
    # Exact 2x up: each source pixel becomes a clean 2x2 block.
    np.testing.assert_array_equal(
        ref_seg_resize(labels, 4, 4),
        [[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
    # Exact 2x down of that block map returns the original — the pixel-centre map
    # is symmetric, unlike the truncating floor(dst * src/dst) rule.
    np.testing.assert_array_equal(ref_seg_resize(ref_seg_resize(labels, 4, 4), 2, 2), labels)


def test_seg_resize_never_invents_a_class():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 21, size=(13, 7)).astype(np.int32)
    out = ref_seg_resize(labels, 31, 19)
    assert set(np.unique(out)).issubset(set(np.unique(labels)))


def test_seg_to_source_strips_the_letterbox_padding():
    """A rectangle on the padded canvas must land on known SOURCE pixels.

    128x64 source into a 64x64 canvas: scale 0.5, no x padding, 16 px of y
    padding. The canvas rectangle [20,36) x [8,24) is the projection of the
    source rectangle [8,40) x [16,48) — and that is what has to come back.
    """
    lb = compute_letterbox(128, 64, 64, 64)
    assert (lb.scale, lb.pad_x, lb.pad_y) == (0.5, 0.0, 16.0)
    canvas = np.zeros((64, 64), np.int32)
    canvas[20:36, 8:24] = 1

    src = ref_seg_to_source(canvas, lb)
    assert src.shape == (64, 128)
    ys, xs = np.nonzero(src)
    assert (xs.min(), xs.max() + 1) == (16, 48)
    assert (ys.min(), ys.max() + 1) == (8, 40)
    assert int(src.sum()) == 32 * 32


def test_seg_to_source_ignores_the_padding_band():
    """Whatever the model predicted in the grey bars must not reach the output."""
    lb = compute_letterbox(128, 64, 64, 64)
    canvas = np.zeros((64, 64), np.int32)
    canvas[:16, :] = 9   # top padding band
    canvas[48:, :] = 9   # bottom padding band
    assert not (ref_seg_to_source(canvas, lb) == 9).any()


def test_seg_to_source_handles_a_coarse_output_grid():
    """A head whose grid does not divide the canvas (65x65 logits for a 513
    input is the classic one) still has to project correctly."""
    lb = compute_letterbox(200, 100, 65, 65)
    coarse = np.arange(13 * 13, dtype=np.int32).reshape(13, 13)
    out = ref_seg_to_source(coarse, lb)
    assert out.shape == (100, 200)
    assert set(np.unique(out)).issubset(set(np.unique(coarse)))


# --------------------------------------------------------------------------- #
# Palette                                                                      #
# --------------------------------------------------------------------------- #
def test_voc_palette_matches_the_published_colours():
    """Ids 0..20 must be the classic VOC colours so overlays match reference
    images pixel for pixel."""
    assert ref_palette(0) == (0, 0, 0)          # background
    assert ref_palette(1) == (128, 0, 0)        # aeroplane
    assert ref_palette(2) == (0, 128, 0)        # bicycle
    assert ref_palette(3) == (128, 128, 0)      # bird
    assert ref_palette(15) == (192, 128, 128)   # person


def test_palette_is_deterministic_and_distinct():
    colours = [ref_palette(i) for i in range(1, 21)]
    assert len(set(colours)) == len(colours)


# --------------------------------------------------------------------------- #
# Opportunistic C++ cross-check                                                #
# --------------------------------------------------------------------------- #
def test_decode_seg_matches_cxx(cxx):
    if cxx is None:
        pytest.skip("compiled rcdl module without decode_seg bindings")
    rng = np.random.default_rng(23)
    logits = rng.standard_normal((11, 17, 23)).astype(np.float32)
    ref, _ = ref_decode_seg(logits)
    got = cxx.decode_seg(np.ascontiguousarray(logits), channels_first=True)
    np.testing.assert_array_equal(np.asarray(got.labels).reshape(17, 23), ref)


def test_seg_colorize_matches_cxx(cxx):
    if cxx is None or not hasattr(cxx, "seg_colorize"):
        pytest.skip("compiled rcdl module without seg_colorize bindings")
    labels = np.arange(21, dtype=np.int32).reshape(3, 7)
    bgr = np.asarray(cxx.seg_colorize(labels)).reshape(3, 7, 3)
    for i in range(21):
        r, g, b = ref_palette(i)
        assert tuple(bgr[i // 7, i % 7]) == (b, g, r)  # OpenCV channel order
