"""Prompt encoding and mask projection, without a model.

Both halves are pure geometry and both fail quietly: a prompt encoded with the
wrong labels segments *something* (SAM reads label 2/3 as a box corner and 1 as a
click, so mixing them up asks a different question), and a mask projected with
the letterbox ignored lands on the frame shifted and scaled.

    PYTHONPATH=build:python pytest -s tests/test_promptable_seg.py
"""

import numpy as np
import pytest

rcdl = pytest.importorskip("rcdl")


def _have():
    if not all(hasattr(rcdl, n) for n in ("encode_box_prompt", "encode_point_prompt",
                                          "mask_from_logits")):
        pytest.skip("promptable-segmentation bindings not exposed")


def _lb(src_w=810, src_h=1080, side=1024):
    """The same letterbox the segmenter builds: aspect-preserving and centred."""
    return rcdl.compute_letterbox(src_w, src_h, side, side)


# --------------------------------------------------------------------------- #
# Prompt encoding                                                              #
# --------------------------------------------------------------------------- #
def test_box_becomes_two_labelled_corners_in_model_pixels():
    _have()
    lb = _lb()
    scale, pad_x, pad_y = lb[0], lb[1], lb[2]
    coords, labels = rcdl.encode_box_prompt(10.0, 20.0, 110.0, 220.0, lb)
    assert coords.shape == (1, 2, 2) and labels.shape == (1, 2)
    np.testing.assert_allclose(coords[0, 0], [10 * scale + pad_x, 20 * scale + pad_y], rtol=1e-5)
    np.testing.assert_allclose(coords[0, 1], [110 * scale + pad_x, 220 * scale + pad_y],
                               rtol=1e-5)
    np.testing.assert_array_equal(labels[0], [2, 3])


def test_point_is_padded_with_the_ignored_point():
    """One exported graph serves both prompt kinds because a click is padded to
    two points, the second labelled -1 so the decoder drops it."""
    _have()
    lb = _lb()
    coords, labels = rcdl.encode_point_prompt(100.0, 200.0, True, lb)
    np.testing.assert_allclose(coords[0, 0], [100 * lb[0] + lb[1], 200 * lb[0] + lb[2]],
                               rtol=1e-5)
    np.testing.assert_array_equal(coords[0, 1], [0, 0])
    np.testing.assert_array_equal(labels[0], [1, -1])

    _, neg = rcdl.encode_point_prompt(100.0, 200.0, False, lb)
    np.testing.assert_array_equal(neg[0], [0, -1])


def test_a_prompt_at_the_frame_corner_lands_inside_the_canvas():
    _have()
    lb = _lb()
    coords, _ = rcdl.encode_box_prompt(0.0, 0.0, 810.0, 1080.0, lb)
    assert (coords >= 0).all() and (coords <= 1024).all()


# --------------------------------------------------------------------------- #
# Mask projection                                                              #
# --------------------------------------------------------------------------- #
def _disc(side, cx, cy, r):
    ys, xs = np.mgrid[0:side, 0:side]
    return (r * r - ((xs - cx) ** 2 + (ys - cy) ** 2)).astype(np.float32)


def test_a_logit_blob_lands_where_the_letterbox_says():
    """A disc centred in the model canvas must come back centred in the SOURCE
    frame — that is the whole content of the projection, and it is wrong in an
    obvious-in-hindsight way if the padding is not stripped."""
    _have()
    lb = _lb(810, 1080)
    logits = _disc(256, 128, 128, 40)
    m = rcdl.mask_from_logits(logits, lb, 0.0, 0.9)
    assert (m.width, m.height) == (810, 1080)
    assert not m.empty
    assert m.score == pytest.approx(0.9)
    x0, y0, x1, y1 = m.bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    assert abs(cx - 405) < 8 and abs(cy - 540) < 8, f"blob centre {cx},{cy}"
    # The disc is 40/128 of the canvas half-width; through the letterbox that is
    # the same fraction of the SOURCE's long side, not of its width.
    assert 300 < (x1 - x0) < 360, f"blob width {x1 - x0}"


def test_padding_never_contributes():
    """Everything the model saw in the padding band must be dropped: a logit map
    that is positive ONLY in the padding has to project to an empty mask."""
    _have()
    lb = _lb(810, 1080)                        # pads left and right
    logits = np.full((256, 256), -1.0, np.float32)
    logits[:, :20] = 5.0                       # the left padding band
    logits[:, -20:] = 5.0
    m = rcdl.mask_from_logits(logits, lb, 0.0)
    assert m.empty, f"{100 * m.area:.2f}% of the frame came out of the padding"


def test_threshold_shrinks_the_mask_monotonically():
    _have()
    lb = _lb(400, 400)
    logits = _disc(256, 128, 128, 60)
    areas = [rcdl.mask_from_logits(logits, lb, t).area for t in (-1000.0, 0.0, 2000.0)]
    assert areas[0] > areas[1] > areas[2] >= 0.0


def test_an_all_negative_map_is_an_empty_mask_not_a_crash():
    _have()
    lb = _lb(320, 240)
    m = rcdl.mask_from_logits(np.full((256, 256), -3.0, np.float32), lb, 0.0)
    assert m.empty and m.area == 0.0 and m.bbox == (0, 0, 0, 0)
