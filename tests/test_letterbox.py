"""Preprocessing tests: letterbox geometry, RGA vs CPU agreement, colour conversion.

The numpy reference below is the oracle both RCDL backends must match:

  * ``ref_letterbox`` — bilinear resample with the OpenCV pixel-center
    convention, centered in a padded canvas. This is what ``letterboxCpu``
    implements exactly (it should agree to the rounding, not approximately).
  * RGA is fixed-point hardware with its own taps. On BAND-LIMITED content it
    matches the CPU path to ±1 LSB. On content that aliases under the downscale
    it does not, and that is not a bug in either: measured on this board, RGA
    pre-filters when it shrinks (an area-like average) while bilinear point-
    samples, so on a 32-pixel checkerboard downscaled 4x they legitimately
    disagree by up to a full range. The CPU path is the one that reproduces
    cv2.resize(INTER_LINEAR) — i.e. the preprocessing the models were calibrated
    with — so it stays the oracle, and the RGA difference is pinned below as a
    documented property. What actually has to hold end-to-end is that a detector
    fed either produces the same boxes; that is tested in
    test_detection_board_py.py.

Runs anywhere for the geometry checks; the backend comparisons skip without the
compiled module, and the RGA ones skip when the board has no RGA.

    PYTHONPATH=build:python pytest -s tests/test_letterbox.py
"""

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# numpy reference                                                              #
# --------------------------------------------------------------------------- #
def ref_geometry(src_w, src_h, dst_w, dst_h):
    """Integer letterbox rectangle, matching the RGA path's rounding."""
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    return scale, (dst_w - new_w) // 2, (dst_h - new_h) // 2, new_w, new_h


def ref_resize_bilinear(img, out_w, out_h):
    """cv::resize(INTER_LINEAR)-equivalent bilinear resample of an HxWxC array."""
    src = np.asarray(img, dtype=np.float32)
    h, w = src.shape[:2]
    fx, fy = w / out_w, h / out_h
    sx = (np.arange(out_w) + 0.5) * fx - 0.5
    sy = (np.arange(out_h) + 0.5) * fy - 0.5
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    ax = (sx - x0).astype(np.float32)
    ay = (sy - y0).astype(np.float32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    top = src[y0][:, x0] * (1 - ax)[None, :, None] + src[y0][:, x1] * ax[None, :, None]
    bot = src[y1][:, x0] * (1 - ax)[None, :, None] + src[y1][:, x1] * ax[None, :, None]
    out = top * (1 - ay)[:, None, None] + bot * ay[:, None, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def ref_letterbox(img, dst_w, dst_h, pad=114):
    _, px, py, nw, nh = ref_geometry(img.shape[1], img.shape[0], dst_w, dst_h)
    c = img.shape[2] if img.ndim == 3 else 1
    canvas = np.full((dst_h, dst_w, c), pad, dtype=np.uint8)
    canvas[py:py + nh, px:px + nw] = ref_resize_bilinear(
        img.reshape(img.shape[0], img.shape[1], c), nw, nh)
    return canvas if img.ndim == 3 else canvas[:, :, 0]


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rcdl_pre():
    m = pytest.importorskip("rcdl", reason="build the module on the board first")
    if not hasattr(m, "letterbox"):
        pytest.skip("compiled module predates the preproc bindings")
    return m


def _scene(h, w):
    """A deterministic image with structure at every scale (not noise: a noisy
    source makes every resampler disagree and hides real filter differences)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = (128 + 127 * np.sin(xx / 23.0)).astype(np.uint8)
    g = (128 + 127 * np.sin(yy / 17.0)).astype(np.uint8)
    b = ((xx.astype(np.int32) // 32 + yy.astype(np.int32) // 32) % 2 * 255).astype(np.uint8)
    return np.dstack([b, g, r])  # BGR


@pytest.fixture(scope="module")
def scene():
    """1280x720 -> 640x640 is the real case: scale 0.5 on both axes with a
    140-pixel band top and bottom. A 640x480 source would letterbox at scale
    1.0, i.e. a pure copy — which every backend gets right and which therefore
    compares nothing."""
    return _scene(720, 1280)


@pytest.fixture(scope="module")
def smooth_scene():
    """Band-limited content: nothing above the Nyquist rate of a 4x downscale,
    so a pre-filtering resampler and a point-sampling one must agree."""
    h, w = 720, 1280
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 128 + 100 * np.sin(xx / 180.0)
    g = 128 + 100 * np.sin(yy / 150.0)
    b = 128 + 100 * np.sin((xx + yy) / 220.0)
    return np.dstack([b, g, r]).astype(np.uint8)


@pytest.fixture(scope="module")
def gradient_scene():
    """A pure linear ramp — exactly reproducible by any linear resampler."""
    h, w = 720, 1280
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.dstack([(xx + yy) / (w + h) * 255, yy / h * 255, xx / w * 255]).astype(np.uint8)


# --------------------------------------------------------------------------- #
# geometry                                                                     #
# --------------------------------------------------------------------------- #
def test_geometry_matches_reference(rcdl_pre):
    for sw, sh, dw, dh in ((1280, 720, 640, 640), (640, 480, 640, 640),
                           (1920, 1080, 416, 416), (100, 300, 224, 224)):
        scale, px, py, _, _ = ref_geometry(sw, sh, dw, dh)
        lb = rcdl_pre.compute_letterbox(sw, sh, dw, dh)
        assert lb[0] == pytest.approx(scale, rel=1e-6)
        # compute_letterbox reports the EXACT float centering; the ops round the
        # scaled extent first and floor the offset, which can land a whole pixel
        # lower (100x300 -> 224: float 74.667 vs integer 74).
        assert abs(lb[1] - px) <= 1.0 and abs(lb[2] - py) <= 1.0


def test_square_source_has_no_padding(rcdl_pre):
    lb = rcdl_pre.compute_letterbox(500, 500, 640, 640)
    # float32 min(640/500, 640/500) * 500 is 639.99999, so the padding is a hair
    # off zero rather than exactly zero — the ops round it away.
    assert lb[1] == pytest.approx(0.0, abs=1e-4) and lb[2] == pytest.approx(0.0, abs=1e-4)
    assert lb[0] == pytest.approx(1.28)


# --------------------------------------------------------------------------- #
# CPU backend vs the numpy oracle                                              #
# --------------------------------------------------------------------------- #
def test_cpu_letterbox_matches_numpy_reference(rcdl_pre, scene):
    got, lb, used = rcdl_pre.letterbox(scene, 640, 640, src_fmt="bgr888",
                                       dst_fmt="bgr888", backend="cpu")
    assert used == "cpu"
    ref = ref_letterbox(scene, 640, 640)
    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    # Same sampling convention and same rounding, so the only disagreement is a
    # rounding tie landing the other way: bounded at 1 LSB and rare.
    print(f"\nCPU vs numpy: max={diff.max()} mean={diff.mean():.4f} "
          f"differing={100 * (diff > 0).mean():.2f}%")
    assert diff.max() <= 1, f"max |diff| = {diff.max()}"
    assert diff.mean() < 0.30


def test_cpu_letterbox_pad_borders(rcdl_pre, scene):
    got, _, _ = rcdl_pre.letterbox(scene, 640, 640, src_fmt="bgr888", dst_fmt="bgr888",
                                   pad=114, backend="cpu")
    _, _, py, _, nh = ref_geometry(scene.shape[1], scene.shape[0], 640, 640)
    assert (got[:py] == 114).all(), "top border is not the pad colour"
    assert (got[py + nh:] == 114).all(), "bottom border is not the pad colour"


@pytest.mark.parametrize("size", [(333, 111), (100, 300), (1279, 721)])
def test_cpu_letterbox_matches_reference_on_awkward_sizes(rcdl_pre, size, scene):
    """Sizes where src*scale is not integral: the extent is rounded and then
    centred, and both backends plus the oracle must agree on where it lands."""
    w, h = size
    img = np.ascontiguousarray(scene[:h, :w]) if h <= scene.shape[0] and w <= scene.shape[1] \
        else np.ascontiguousarray(np.resize(scene, (h, w, 3)))
    got, lb, _ = rcdl_pre.letterbox(img, 224, 224, src_fmt="bgr888", dst_fmt="bgr888",
                                    backend="cpu")
    _, px, py, nw, nh = ref_geometry(w, h, 224, 224)
    assert (lb[1], lb[2]) == (px, py), f"placement {lb[1], lb[2]} vs {px, py}"
    ref = ref_letterbox(img, 224, 224)
    diff = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    assert diff.max() <= 1, f"max |diff| = {diff.max()} at {w}x{h}"


def test_cpu_letterbox_swaps_channels(rcdl_pre, scene):
    """bgr888 -> rgb888 must reverse the channel order, not just copy."""
    as_bgr, _, _ = rcdl_pre.letterbox(scene, 320, 320, src_fmt="bgr888",
                                      dst_fmt="bgr888", backend="cpu")
    as_rgb, _, _ = rcdl_pre.letterbox(scene, 320, 320, src_fmt="bgr888",
                                      dst_fmt="rgb888", backend="cpu")
    np.testing.assert_array_equal(as_rgb, as_bgr[:, :, ::-1])


# --------------------------------------------------------------------------- #
# RGA backend                                                                  #
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rga(rcdl_pre):
    if not rcdl_pre.rga_available():
        pytest.skip("RGA not available in this build/board")
    return rcdl_pre


@pytest.mark.parametrize("dst", [640, 416, 320, 224])
def test_rga_matches_cpu_on_a_linear_ramp(rga, gradient_scene, dst):
    """The strict form of the milestone budget: on content a linear resampler
    reproduces exactly, RGA and the CPU path agree to ±1 LSB at every downscale
    factor. Any larger difference here is a real defect (wrong rectangle, wrong
    stride, wrong colour matrix), not filter choice."""
    hw, _, u1 = rga.letterbox(gradient_scene, dst, dst, src_fmt="bgr888", dst_fmt="rgb888",
                              backend="rga")
    sw, _, u2 = rga.letterbox(gradient_scene, dst, dst, src_fmt="bgr888", dst_fmt="rgb888",
                              backend="cpu")
    assert u1 == "rga" and u2 == "cpu"
    d = np.abs(hw.astype(np.int16) - sw.astype(np.int16))
    print(f"\nramp {dst}x{dst}: RGA vs CPU max={d.max()} mean={d.mean():.3f}")
    assert d.max() <= 1, f"max |diff| = {d.max()} on a linear ramp"


@pytest.mark.parametrize("dst", [640, 416, 320, 224])
def test_rga_matches_cpu_on_band_limited_content(rga, smooth_scene, dst):
    """Slightly looser: smooth but not linear, so the two rounding schemes can
    part company by a couple of LSB. Measured on this board: max 5, mean 0.67 at
    the most aggressive 0.175x downscale."""
    hw, _, _ = rga.letterbox(smooth_scene, dst, dst, src_fmt="bgr888", dst_fmt="rgb888",
                             backend="rga")
    sw, _, _ = rga.letterbox(smooth_scene, dst, dst, src_fmt="bgr888", dst_fmt="rgb888",
                             backend="cpu")
    d = np.abs(hw.astype(np.int16) - sw.astype(np.int16))
    print(f"\nsmooth {dst}x{dst}: RGA vs CPU max={d.max()} mean={d.mean():.3f} "
          f"p99={np.percentile(d, 99):.0f}")
    assert d.max() <= 8, f"max |diff| = {d.max()}"
    assert d.mean() < 1.0


def test_rga_prefilters_on_downscale_unlike_bilinear(rga, scene):
    """Pin the KNOWN divergence so it stays a documented property rather than a
    surprise: on aliasing content (a 32-pixel checkerboard at 0.5x) the two
    backends differ substantially, because RGA averages the pixels it discards
    and bilinear does not. The bound is loose on purpose — it is here to catch
    'RGA output became garbage', not to constrain the filter."""
    hw, _, _ = rga.letterbox(scene, 640, 640, src_fmt="bgr888", dst_fmt="rgb888",
                             backend="rga")
    sw, _, _ = rga.letterbox(scene, 640, 640, src_fmt="bgr888", dst_fmt="rgb888",
                             backend="cpu")
    d = np.abs(hw.astype(np.int16) - sw.astype(np.int16))
    print(f"\naliasing content: RGA vs CPU max={d.max()} mean={d.mean():.2f} "
          f">1: {(d > 1).mean() * 100:.1f}%")
    assert d.mean() < 8.0, "RGA and the CPU path have diverged far beyond filter choice"
    # and both must still be a picture, not noise: the letterboxed means agree
    assert abs(hw.mean() - sw.mean()) < 3.0


def test_rga_borders_are_pad_colour(rga, scene):
    got, _, _ = rga.letterbox(scene, 640, 640, src_fmt="bgr888", dst_fmt="rgb888",
                              pad=114, backend="rga")
    _, _, py, _, nh = ref_geometry(scene.shape[1], scene.shape[0], 640, 640)
    if py > 1:
        assert np.abs(got[:py - 1].astype(np.int16) - 114).max() <= 1
        assert np.abs(got[py + nh + 1:].astype(np.int16) - 114).max() <= 1


def test_auto_backend_prefers_rga(rga, scene):
    _, _, used = rga.letterbox(scene, 640, 640, src_fmt="bgr888", dst_fmt="rgb888",
                               backend="auto")
    assert used == "rga"


@pytest.mark.parametrize("src_hw,dst", [
    ((2048, 2048), 224),   # 1/9.1: inside im2d's [1/16,16] but outside RGA3's [1/8,8]
    ((4096, 4096), 224),   # 1/18: outside both
    ((8000, 8000), 224),   # 1/35
    ((4, 4), 640),         # 160x up, and below the 68x2 minimum
    ((2, 100), 640),       # 2 rows tall
    ((100, 9000), 640),    # wider than the 8192 maximum
])
def test_auto_never_raises_outside_rga_limits(rga, src_hw, dst):
    """Auto promises the CPU path when RGA cannot do the job — including when
    the refusal only shows up at SUBMIT time.

    The 1/9.1 case is the one that bit: im2d's `imcheck` accepts it because its
    documented range is [1/16, 16], but that upper half belongs to the RGA2 core,
    which has no IOMMU and cannot map pages above 4 GB. The driver routes the op
    there and it dies with "job buffer map failed" — after the check passed. So
    Auto needs both a tighter pre-flight AND a run-time fallback.
    """
    h, w = src_hw
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[h // 4:h // 2, w // 4:w // 2] = 200
    out, _, used = rga.letterbox(img, dst, dst, src_fmt="bgr888", dst_fmt="rgb888",
                                 backend="auto")
    assert out.shape == (dst, dst, 3)
    assert used in ("rga", "cpu")
    print(f"\n{w}x{h} -> {dst}x{dst}: {used}")


@pytest.mark.parametrize("fmt,align", [("bgr888", 16), ("rgb888", 16),
                                       ("rgba8888", 4), ("nv12", 16)])
def test_rga_stride_alignment_rule(rga, fmt, align):
    """RGA3's row-stride rule, measured rather than assumed.

    The documentation only mentions the YUV case, but the hardware also rejects
    a 3-byte packed source whose stride is not 16-pixel aligned ("bgr888 width
    stride should be 16 aligned!") and a 4-byte one that is not 4-aligned. The
    consequence for callers: an image RCDL allocated is always acceptable
    (`strideAlign()` applies this table), while a foreign buffer — a cv::Mat
    over an 810-pixel-wide JPEG, say — may not be, and takes the CPU path.
    """
    ch = {"bgr888": 3, "rgb888": 3, "rgba8888": 4, "nv12": 1}[fmt]
    for w in (640 - align, 640, 640 + align):          # aligned: must be accepted
        img = (np.zeros((480, w, ch), dtype=np.uint8) if ch > 1
               else np.zeros((480 * 3 // 2, w), dtype=np.uint8))
        _, _, used = rga.letterbox(img, 320, 320, src_fmt=fmt, dst_fmt="rgb888",
                                   backend="auto")
        assert used == "rga", f"{fmt} at aligned width {w} fell back to {used}"
    if align > 1:                                       # misaligned: must fall back
        w = 640 + 1 if fmt != "nv12" else 640 + 2       # NV12 also needs even dims
        img = (np.zeros((480, w, ch), dtype=np.uint8) if ch > 1
               else np.zeros((480 * 3 // 2, w), dtype=np.uint8))
        out, _, used = rga.letterbox(img, 320, 320, src_fmt=fmt, dst_fmt="rgb888",
                                     backend="auto")
        assert used == "cpu", f"{fmt} at misaligned width {w} used {used}"
        assert out.shape == (320, 320, 3)


def test_gray8_falls_back_to_cpu(rga):
    """GRAY8 fails at SUBMIT on this board in every direction (the driver routes
    RK_FORMAT_YCbCr_400 to the RGA2 core, which cannot address memory above
    4 GB). Auto must produce a correct image anyway."""
    g = np.tile(np.arange(640, dtype=np.uint8), (480, 1))
    out, _, used = rga.letterbox(g, 320, 320, src_fmt="gray8", dst_fmt="rgb888",
                                 backend="auto")
    assert used == "cpu"
    assert out.shape == (320, 320, 3)


def test_auto_falls_back_when_rga_cannot(rga):
    """A source below RGA3's 68x2 minimum must silently take the CPU path."""
    tiny = np.zeros((4, 4, 3), dtype=np.uint8)
    tiny[1:3, 1:3] = 200
    out, _, used = rga.letterbox(tiny, 640, 640, src_fmt="bgr888", dst_fmt="rgb888",
                                 backend="auto")
    assert out.shape == (640, 640, 3)
    print(f"\n4x4 -> 640x640 used the {used} backend")


# --------------------------------------------------------------------------- #
# NV12 (the video path)                                                        #
# --------------------------------------------------------------------------- #
# (Kr, Kb) of each colour matrix; every coefficient below is derived from these,
# so the oracle is the textbook definition rather than a copy of the C++ table.
KR_KB = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722)}


def make_nv12(bgr, matrix="bt601", studio_range=False):
    """BGR -> NV12 with `matrix` and the given levels — the inverse of
    cvt_color(..., studio_range=..., matrix=...). The default (BT.601, full
    range) is what the older tests here were written against. Chroma averages
    each 2x2 RGB block before converting, as the CPU path does."""
    kr, kb = KR_KB[matrix]
    kg = 1.0 - kr - kb
    h, w = bgr.shape[:2]
    b, g, r = (bgr[:, :, i].astype(np.float64) for i in range(3))
    y = kr * r + kg * g + kb * b
    r2, g2, b2 = (c.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)) for c in (r, g, b))
    y2 = kr * r2 + kg * g2 + kb * b2
    u = (b2 - y2) / (2 * (1 - kb))
    v = (r2 - y2) / (2 * (1 - kr))
    if studio_range:
        y = 16 + y * 219 / 255
        u = u * 224 / 255
        v = v * 224 / 255
    out = np.empty((h * 3 // 2, w), dtype=np.uint8)
    out[:h] = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = np.clip(np.rint(u + 128), 0, 255).astype(np.uint8)
    uv[:, 1::2] = np.clip(np.rint(v + 128), 0, 255).astype(np.uint8)
    out[h:] = uv
    return out


def ref_nv12_to_bgr(nv12, matrix="bt601", studio_range=False):
    """NV12 -> BGR from the textbook equations, nearest-neighbour chroma (what
    cv::cvtColor and the CPU path do)."""
    kr, kb = KR_KB[matrix]
    kg = 1.0 - kr - kb
    rows, w = nv12.shape
    h = rows * 2 // 3
    y = nv12[:h].astype(np.float64)
    uv = nv12[h:].reshape(h // 2, w // 2, 2).astype(np.float64) - 128
    u = uv[:, :, 0].repeat(2, axis=0).repeat(2, axis=1)
    v = uv[:, :, 1].repeat(2, axis=0).repeat(2, axis=1)
    if studio_range:
        y = (y - 16) * 255 / 219
        u = u * 255 / 224
        v = v * 255 / 224
    r = y + 2 * (1 - kr) * v
    b = y + 2 * (1 - kb) * u
    g = (y - kr * r - kb * b) / kg
    return np.clip(np.rint(np.dstack([b, g, r])), 0, 255).astype(np.uint8)


def random_nv12(w=320, h=240, seed=0):
    """Uniform noise over the whole studio-swing cube, out-of-gamut corners
    included, so the clipping is exercised too."""
    rng = np.random.default_rng(seed)
    out = np.empty((h * 3 // 2, w), dtype=np.uint8)
    out[:h] = rng.integers(16, 236, (h, w))
    out[h:] = rng.integers(16, 241, (h // 2, w))
    return out


def test_nv12_roundtrip_is_self_consistent(rcdl_pre, scene):
    """NV12 -> BGR -> NV12 with kAsIs levels must return (nearly) the same NV12."""
    nv12 = make_nv12(scene)
    bgr, _ = rcdl_pre.cvt_color(nv12, "nv12", "bgr888", backend="cpu", studio_range=False)
    again = make_nv12(bgr)
    dy = np.abs(again[:scene.shape[0]].astype(np.int16) - nv12[:scene.shape[0]].astype(np.int16))
    print(f"\nNV12 luma round-trip: max={dy.max()} mean={dy.mean():.3f}")
    assert dy.mean() < 2.0


def test_nv12_letterbox_to_rgb(rcdl_pre, scene):
    """The video path shape: one call turns an NV12 frame into the model canvas."""
    nv12 = make_nv12(scene)
    out, lb, used = rcdl_pre.letterbox(nv12, 640, 640, src_fmt="nv12", dst_fmt="rgb888",
                                       backend="auto", studio_range=False)
    assert out.shape == (640, 640, 3)
    assert lb[3] == scene.shape[1] and lb[4] == scene.shape[0]
    # the letterboxed frame must resemble the same path taken through BGR
    ref, _, _ = rcdl_pre.letterbox(scene, 640, 640, src_fmt="bgr888", dst_fmt="rgb888",
                                   backend="cpu")
    d = np.abs(out.astype(np.int16) - ref.astype(np.int16))
    print(f"NV12 vs BGR letterbox ({used}): mean={d.mean():.2f} p99={np.percentile(d, 99)}")
    assert d.mean() < 12.0  # chroma subsampling loss dominates, not a bug


# --------------------------------------------------------------------------- #
# colour matrix: BT.601 / BT.709, studio / full range                          #
# --------------------------------------------------------------------------- #
COLOR_SPACES = [("bt601", True), ("bt601", False), ("bt709", True), ("bt709", False)]


@pytest.mark.parametrize("matrix,studio", COLOR_SPACES)
def test_cpu_nv12_to_bgr_matches_textbook(rcdl_pre, matrix, studio):
    """Every matrix/range pair against the equations derived from Kr/Kb. The C++
    tables carry 3-4 significant digits, which stays inside one LSB."""
    nv12 = random_nv12()
    got, used = rcdl_pre.cvt_color(nv12, "nv12", "bgr888", backend="cpu",
                                   studio_range=studio, matrix=matrix)
    assert used == "cpu"
    d = np.abs(got.astype(np.int16) - ref_nv12_to_bgr(nv12, matrix, studio))
    print(f"\n{matrix} studio={studio}: CPU vs textbook max={d.max()} mean={d.mean():.4f}")
    assert d.max() <= 1


@pytest.mark.parametrize("matrix,studio", COLOR_SPACES)
def test_cpu_bgr_to_nv12_matches_textbook(rcdl_pre, scene, matrix, studio):
    got, _ = rcdl_pre.cvt_color(scene, "bgr888", "nv12", backend="cpu",
                                studio_range=studio, matrix=matrix)
    d = np.abs(got.astype(np.int16) - make_nv12(scene, matrix, studio))
    h = scene.shape[0]
    print(f"\n{matrix} studio={studio}: CPU vs textbook Y max={d[:h].max()} "
          f"UV max={d[h:].max()}")
    assert d.max() <= 1


def test_matrix_defaults_to_bt601_and_bt709_differs(rcdl_pre):
    """Leaving `matrix` out must keep the old BT.601 bytes exactly, and asking
    for BT.709 must actually reach the conversion."""
    nv12 = random_nv12()
    default, _ = rcdl_pre.cvt_color(nv12, "nv12", "bgr888", backend="cpu")
    bt601, _ = rcdl_pre.cvt_color(nv12, "nv12", "bgr888", backend="cpu", matrix="bt601")
    bt709, _ = rcdl_pre.cvt_color(nv12, "nv12", "bgr888", backend="cpu", matrix="bt709")
    np.testing.assert_array_equal(default, bt601)
    d = np.abs(bt709.astype(np.int16) - bt601.astype(np.int16))
    print(f"\nBT.709 vs BT.601 on the same NV12: max={d.max()} mean={d.mean():.2f}")
    assert d.max() >= 20


@pytest.mark.parametrize("matrix,studio", COLOR_SPACES)
def test_nv12_roundtrip_per_colour_space(rcdl_pre, smooth_scene, matrix, studio):
    """NV12 -> BGR -> NV12 on the CPU with one colour space throughout comes back
    to the same NV12 up to rounding. Band-limited, in-gamut content, so nothing
    clips on the way."""
    nv12 = make_nv12(smooth_scene, matrix, studio)
    kw = dict(backend="cpu", studio_range=studio, matrix=matrix)
    bgr, _ = rcdl_pre.cvt_color(nv12, "nv12", "bgr888", **kw)
    again, _ = rcdl_pre.cvt_color(bgr, "bgr888", "nv12", **kw)
    d = np.abs(again.astype(np.int16) - nv12.astype(np.int16))
    h = smooth_scene.shape[0]
    print(f"\n{matrix} studio={studio}: round-trip Y max={d[:h].max()} "
          f"UV max={d[h:].max()} mean={d.mean():.3f}")
    assert d.max() <= 2


def test_unknown_matrix_is_rejected(rcdl_pre):
    with pytest.raises(ValueError):
        rcdl_pre.cvt_color(random_nv12(), "nv12", "bgr888", backend="cpu", matrix="bt2020")


@pytest.mark.parametrize("matrix,studio", [("bt601", True), ("bt601", False), ("bt709", True)])
def test_rga_nv12_to_rgb_matches_cpu_per_colour_space(rga, smooth_scene, matrix, studio):
    """The three YUV -> RGB modes RGA has. Same size, so nothing is resampled:
    any difference is the matrix (or the chroma upsampling)."""
    nv12 = make_nv12(smooth_scene, matrix, studio)
    kw = dict(studio_range=studio, matrix=matrix)
    hw, u1 = rga.cvt_color(nv12, "nv12", "rgb888", backend="rga", **kw)
    sw, u2 = rga.cvt_color(nv12, "nv12", "rgb888", backend="cpu", **kw)
    assert u1 == "rga" and u2 == "cpu"
    d = np.abs(hw.astype(np.int16) - sw.astype(np.int16))
    print(f"\n{matrix} studio={studio}: RGA vs CPU max={d.max()} mean={d.mean():.3f}")
    assert d.max() <= 2


def test_rga_nv12_bt709_letterbox_matches_cpu(rga, smooth_scene):
    """The HD video hot path — studio-swing BT.709 NV12 letterboxed to RGB888 —
    stays on the hardware and agrees with the CPU path as closely as BGR input
    does in test_rga_matches_cpu_on_band_limited_content."""
    nv12 = make_nv12(smooth_scene, "bt709", studio_range=True)
    kw = dict(src_fmt="nv12", dst_fmt="rgb888", matrix="bt709")
    hw, _, used = rga.letterbox(nv12, 640, 640, backend="auto", **kw)
    sw, _, _ = rga.letterbox(nv12, 640, 640, backend="cpu", **kw)
    assert used == "rga"
    d = np.abs(hw.astype(np.int16) - sw.astype(np.int16))
    print(f"\nBT.709 NV12 letterbox: RGA vs CPU max={d.max()} mean={d.mean():.3f}")
    assert d.max() <= 8
    assert d.mean() < 1.0


@pytest.mark.parametrize("src_fmt,dst_fmt,studio", [
    ("nv12", "rgb888", False),   # librga has no BT.709 full-range mode
    ("rgb888", "nv12", True),    # the driver routes RGB -> YUV BT.709 to RGA2
    ("rgb888", "nv12", False),
])
def test_auto_takes_cpu_where_rga_has_no_bt709_mode(rga, smooth_scene, src_fmt, dst_fmt, studio):
    """rgaCanHandle() must say no up front (no failed ioctl), Auto must produce
    the CPU path's exact bytes, and a forced RGA must raise rather than submit."""
    img = (make_nv12(smooth_scene, "bt709", studio) if src_fmt == "nv12"
           else np.ascontiguousarray(smooth_scene[:, :, ::-1]))
    kw = dict(studio_range=studio, matrix="bt709")
    auto, used = rga.cvt_color(img, src_fmt, dst_fmt, backend="auto", **kw)
    cpu, _ = rga.cvt_color(img, src_fmt, dst_fmt, backend="cpu", **kw)
    assert used == "cpu"
    np.testing.assert_array_equal(auto, cpu)
    with pytest.raises(RuntimeError, match="BT.709"):
        rga.cvt_color(img, src_fmt, dst_fmt, backend="rga", **kw)
