"""Flow decode, visualisation and scoring against a numpy oracle.

The decode itself is small — de-planarize and scale — but both halves have a
convention that is invisible when wrong: which axis holds the two components,
and whether the vectors are in model pixels or source pixels. A field read with
the wrong one is still a smooth, plausible field.

Pure numpy plus the compiled module: no board, no ``.rknn``.

    PYTHONPATH=build:python pytest -s tests/test_optical_flow.py
"""

import numpy as np
import pytest

rcdl = pytest.importorskip("rcdl")


def _have():
    if not all(hasattr(rcdl, n) for n in ("decode_flow", "flow_colorize",
                                          "flow_endpoint_error", "flow_preprocess")):
        pytest.skip("optical-flow bindings not exposed")


def _planar(h=6, w=8, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((1, 2, h, w)) * 4).astype(np.float32)


# --------------------------------------------------------------------------- #
# Decode                                                                       #
# --------------------------------------------------------------------------- #
def test_planar_and_interleaved_are_the_same_field():
    """[1,2,H,W] and [1,H,W,2] carry the same numbers in a different order, and
    reading one as the other transposes u and v across the frame."""
    _have()
    t = _planar()
    a = rcdl.decode_flow(t)
    b = rcdl.decode_flow(np.ascontiguousarray(t.transpose(0, 2, 3, 1)), channels_first=False)
    np.testing.assert_allclose(a, b, atol=0)
    np.testing.assert_allclose(a[..., 0], t[0, 0], atol=0)
    np.testing.assert_allclose(a[..., 1], t[0, 1], atol=0)


def test_the_batch_dimension_is_optional():
    _have()
    t = _planar()
    np.testing.assert_allclose(rcdl.decode_flow(t), rcdl.decode_flow(t[0]), atol=0)


def test_scale_converts_model_pixels_to_source_pixels():
    """A model run on a downscaled pair reports displacement in ITS pixels.
    Handing those to a full-resolution warp under-moves everything, silently."""
    _have()
    t = _planar()
    a = rcdl.decode_flow(t)
    b = rcdl.decode_flow(t, scale_x=2.0, scale_y=3.0)
    np.testing.assert_allclose(b[..., 0], a[..., 0] * 2.0, rtol=1e-6)
    np.testing.assert_allclose(b[..., 1], a[..., 1] * 3.0, rtol=1e-6)


def test_wrong_channel_count_is_rejected():
    _have()
    with pytest.raises(Exception):
        rcdl.decode_flow(np.zeros((1, 3, 4, 5), np.float32))
    with pytest.raises(Exception):
        rcdl.decode_flow(np.zeros((1, 4, 5, 3), np.float32), channels_first=False)


# --------------------------------------------------------------------------- #
# Endpoint error                                                               #
# --------------------------------------------------------------------------- #
def test_endpoint_error_is_the_mean_vector_distance():
    _have()
    a = rcdl.decode_flow(_planar(seed=1))
    b = rcdl.decode_flow(_planar(seed=2))
    oracle = float(np.hypot(a[..., 0] - b[..., 0], a[..., 1] - b[..., 1]).mean())
    assert rcdl.flow_endpoint_error(a, b) == pytest.approx(oracle, rel=1e-5)
    assert rcdl.flow_endpoint_error(a, a) == pytest.approx(0.0, abs=1e-7)


def test_endpoint_error_needs_matching_sizes():
    _have()
    a = rcdl.decode_flow(_planar(h=6, w=8))
    b = rcdl.decode_flow(_planar(h=6, w=9))
    with pytest.raises(Exception):
        rcdl.flow_endpoint_error(a, b)


# --------------------------------------------------------------------------- #
# Visualisation                                                                #
# --------------------------------------------------------------------------- #
def test_zero_flow_is_white_and_speed_darkens_it():
    """The colour wheel puts white at the centre — no motion — and saturates
    outward, so a still frame must not come out coloured."""
    _have()
    still = np.zeros((4, 4, 2), np.float32)
    assert (rcdl.flow_colorize(still) == 255).all()

    moving = np.zeros((4, 4, 2), np.float32)
    moving[..., 0] = 5.0
    viz = rcdl.flow_colorize(moving, 5.0)
    assert viz.shape == (4, 4, 3)
    assert viz.min() < 200, "a full-scale vector should be a saturated colour, not white"


def test_opposite_directions_get_different_hues():
    _have()
    right = np.zeros((2, 2, 2), np.float32)
    right[..., 0] = 3.0
    left = -right
    up = np.zeros((2, 2, 2), np.float32)
    up[..., 1] = -3.0
    a, b, c = (rcdl.flow_colorize(f, 3.0)[0, 0] for f in (right, left, up))
    assert not np.array_equal(a, b) and not np.array_equal(a, c)


def test_default_normalisation_uses_the_99th_percentile():
    """One fast pixel must not flatten the rest of the frame to near-grey."""
    _have()
    f = np.zeros((32, 32, 2), np.float32)
    f[..., 0] = 1.0
    f[0, 0, 0] = 1000.0                       # one outlier
    viz = rcdl.flow_colorize(f)               # 0 => percentile normalisation
    assert viz[16, 16].min() < 200, "the bulk of the frame was washed out by one pixel"


# --------------------------------------------------------------------------- #
# Preprocessing                                                                #
# --------------------------------------------------------------------------- #
def test_preprocess_keeps_bgr_and_0_255():
    """The graph divides by 255 itself and was trained on cv2's BGR. Pre-dividing
    or swapping to RGB gives a plausible, wrong field."""
    _have()
    img = np.zeros((40, 60, 3), np.uint8)
    img[:, :, 0] = 10                          # B
    img[:, :, 1] = 120                         # G
    img[:, :, 2] = 240                         # R
    x = rcdl.flow_preprocess(img, 60, 40)
    assert x.shape == (1, 40, 60, 3)           # interleaved, as the tensor wants
    assert x.max() > 200, "the values were normalized; the graph does that itself"
    np.testing.assert_allclose(x[0, 5, 5], [10, 120, 240], atol=1e-3)


def test_preprocess_resizes_to_the_model_size():
    _have()
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
    x = rcdl.flow_preprocess(img, 64, 48)
    assert x.shape == (1, 48, 64, 3)
