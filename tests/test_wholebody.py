"""Whole-body pose: the crop geometry and the SimCC decode, without a model.

Both are pure and both are silently wrong in the same way — the skeleton still
looks like a skeleton, just in the wrong place or at the wrong scale. The two
conventions that decide it are the 1.25 box padding with its aspect fix, and the
SimCC split ratio of two bins per input pixel.

    PYTHONPATH=build:python pytest -s tests/test_wholebody.py
"""

import numpy as np
import pytest

rcdl = pytest.importorskip("rcdl")

IN_W, IN_H = 192, 256


def _have():
    if not all(hasattr(rcdl, n) for n in ("crop_geometry", "decode_simcc", "body_part_range")):
        pytest.skip("whole-body bindings not exposed")


def _simcc(peaks, bins_x=IN_W * 2, bins_y=IN_H * 2, peak=0.9, floor=0.01):
    """SimCC tensors with one sharp peak per joint at the given (bin_x, bin_y)."""
    k = len(peaks)
    x = np.full((k, bins_x), floor, np.float32)
    y = np.full((k, bins_y), floor, np.float32)
    for i, (bx, by) in enumerate(peaks):
        x[i, bx] = peak
        y[i, by] = peak
    return x, y


# --------------------------------------------------------------------------- #
# Crop geometry                                                                #
# --------------------------------------------------------------------------- #
def test_crop_is_padded_and_forced_to_the_model_aspect():
    _have()
    c = rcdl.crop_geometry(100, 100, 200, 400, IN_W, IN_H, 1.25)
    assert c.cx == pytest.approx(150) and c.cy == pytest.approx(250)
    # The tall box is padded to 125x375 and then WIDENED to 3:4, never cropped.
    assert c.h == pytest.approx(375, rel=1e-5)
    assert c.w / c.h == pytest.approx(IN_W / IN_H, rel=1e-5)
    assert c.w >= 125


def test_a_wide_box_grows_in_height_instead():
    _have()
    c = rcdl.crop_geometry(0, 0, 400, 100, IN_W, IN_H, 1.25)
    assert c.w == pytest.approx(500, rel=1e-5)
    assert c.w / c.h == pytest.approx(IN_W / IN_H, rel=1e-5)
    assert c.h > 125, "the short axis must grow; cropping to fit would cut off the feet"


def test_padding_scales_the_rect_not_the_centre():
    _have()
    a = rcdl.crop_geometry(100, 100, 200, 400, IN_W, IN_H, 1.0)
    b = rcdl.crop_geometry(100, 100, 200, 400, IN_W, IN_H, 1.25)
    assert (b.cx, b.cy) == (a.cx, a.cy)
    assert b.h / a.h == pytest.approx(1.25, rel=1e-5)


# --------------------------------------------------------------------------- #
# SimCC decode                                                                 #
# --------------------------------------------------------------------------- #
def test_a_peak_maps_back_through_the_crop():
    _have()
    crop = rcdl.crop_geometry(100, 200, 200, 400, IN_W, IN_H, 1.0)
    # A joint in the exact middle of the model input must land at the crop's
    # centre, whatever the crop is.
    x, y = _simcc([(IN_W, IN_H)])          # bin = pixel * split_ratio(2)
    kp = rcdl.decode_simcc(x, y, crop, IN_W, IN_H)
    assert kp.shape == (1, 3)
    assert kp[0, 0] == pytest.approx(crop.cx, abs=0.5)
    assert kp[0, 1] == pytest.approx(crop.cy, abs=0.5)


def test_the_split_ratio_is_load_bearing():
    """Two bins per input pixel. Decoding as one puts every joint at twice its
    offset inside the crop — still a skeleton, still drawable, in the wrong
    place and at the wrong size."""
    _have()
    crop = rcdl.crop_geometry(0, 0, 400, 400, IN_W, IN_H, 1.0)
    x, y = _simcc([(IN_W, IN_H)])
    right = rcdl.decode_simcc(x, y, crop, IN_W, IN_H, 0.3, 2.0)
    wrong = rcdl.decode_simcc(x, y, crop, IN_W, IN_H, 0.3, 1.0)
    assert right[0, 0] == pytest.approx(crop.cx, abs=0.5)
    off_right = right[0, 0] - crop.x0
    off_wrong = wrong[0, 0] - crop.x0
    assert off_wrong == pytest.approx(2 * off_right, rel=1e-4)


def test_score_is_the_mean_of_the_two_peaks_and_gates_the_position():
    _have()
    crop = rcdl.crop_geometry(0, 0, 200, 300, IN_W, IN_H)
    x, y = _simcc([(10, 20), (30, 40)])
    x[1, 30] = 0.5                            # a weaker peak on one axis only
    y[1, 40] = 0.1
    kp = rcdl.decode_simcc(x, y, crop, IN_W, IN_H, 0.3, 2.0)
    assert kp[0, 2] == pytest.approx(0.9, abs=1e-5)
    assert kp[1, 2] == pytest.approx(0.3, abs=1e-5)
    # Below threshold keeps the score but reports no position, so a caller
    # cannot draw a peak found in noise.
    low = rcdl.decode_simcc(x, y, crop, IN_W, IN_H, 0.5, 2.0)
    assert low[1, 0] == -1.0 and low[1, 1] == -1.0
    assert low[1, 2] == pytest.approx(0.3, abs=1e-5)


def test_every_joint_of_a_133_point_set_is_decoded():
    _have()
    rng = np.random.default_rng(0)
    peaks = [(int(rng.integers(0, IN_W * 2)), int(rng.integers(0, IN_H * 2))) for _ in range(133)]
    x, y = _simcc(peaks)
    crop = rcdl.crop_geometry(50, 60, 250, 460, IN_W, IN_H)
    kp = rcdl.decode_simcc(x, y, crop, IN_W, IN_H)
    assert kp.shape == (133, 3)
    for i, (bx, by) in enumerate(peaks):
        assert kp[i, 0] == pytest.approx(crop.x0 + (bx / 2.0) * crop.w / IN_W, abs=0.01)
        assert kp[i, 1] == pytest.approx(crop.y0 + (by / 2.0) * crop.h / IN_H, abs=0.01)


# --------------------------------------------------------------------------- #
# Keypoint layout                                                              #
# --------------------------------------------------------------------------- #
def test_the_five_regions_tile_the_133_indices():
    _have()
    ranges = [rcdl.body_part_range(p) for p in (rcdl.BodyPart.BODY, rcdl.BodyPart.FOOT,
                                                rcdl.BodyPart.FACE, rcdl.BodyPart.LEFT_HAND,
                                                rcdl.BodyPart.RIGHT_HAND)]
    assert ranges == [(0, 17), (17, 23), (23, 91), (91, 112), (112, 133)]
    covered = []
    for b, e in ranges:
        covered.extend(range(b, e))
    assert covered == list(range(133)), "the regions must tile the index space exactly"
    for i, part in ((0, rcdl.BodyPart.BODY), (17, rcdl.BodyPart.FOOT),
                    (23, rcdl.BodyPart.FACE), (91, rcdl.BodyPart.LEFT_HAND),
                    (132, rcdl.BodyPart.RIGHT_HAND)):
        assert rcdl.body_part(i) == part


def test_the_first_seventeen_are_the_coco_body_joints():
    """The whole-body layout starts with COCO-17 in the same order, which is what
    lets a whole-body result be compared against the plain pose head."""
    _have()
    b, e = rcdl.body_part_range(rcdl.BodyPart.BODY)
    assert (b, e) == (0, 17)
    assert rcdl.coco_keypoint_name(0) == "nose"
    assert rcdl.coco_keypoint_name(9) == "left_wrist"
    assert rcdl.coco_keypoint_name(16) == "right_ankle"
