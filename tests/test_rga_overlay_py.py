"""Board tests for the box overlay: the CPU path, the RGA2 hardware path, and
the rule that they produce the same bytes.

Why this exists: on RK3588 colour fill is an RGA2-only feature and RGA2 has a
32-bit MMU, so the hardware can only draw on a frame whose pages are below
4 GB — a decoder pool from the `system-dma32` heap (VideoDecoder(pool_heap=
"system-dma32"), VideoFrame.below_4g). RCDL draws with the CPU by default
(measured faster: one map + one cache sync per frame, ~0.3 ms for 20 boxes on
1080p against ~0.5 ms PER BOX on the single RGA2 core) and keeps the hardware
as an explicit choice. Whichever ran, the frame must come out identical, so
the numpy reference below encodes the shared geometry rules and both backends
are checked against it exactly:

  * clipped to the frame first (a box off the edge is closed at the edge),
  * on 4:2:0 snapped OUTWARD to even pixels, thickness rounded up to even,
  * a box too small to have an inside is drawn solid,
  * the colour lands in BT.601 STUDIO range with the classic 8-bit integer
    coefficients — what the hardware fill writes (red -> Y=82 Cb=90 Cr=240).

Needs the module, a board with RGA and MPP, and a raw H.264 stream:

    RCDL_STREAMS=/path/to/streams PYTHONPATH=build:python pytest -s tests/test_rga_overlay_py.py
"""

import os

import numpy as np
import pytest


def _stream():
    d = os.environ.get("RCDL_STREAMS", "")
    p = os.path.join(d, "h264_1080p.h264")
    if not d or not os.path.isfile(p):
        pytest.skip("set RCDL_STREAMS to a directory holding h264_1080p.h264")
    return p


def _rcdl():
    m = pytest.importorskip("rcdl", reason="build the module on the board first")
    if not hasattr(m, "VideoDecoder") or not hasattr(m.VideoFrame, "draw_rects"):
        pytest.skip("compiled module predates the overlay bindings")
    if not m.rga_available():
        pytest.skip("RGA not available in this build/board")
    return m


def _first_frame(m, data, **cfg):
    dec = m.VideoDecoder(codec="h264", **cfg)
    pos = 0
    for _ in range(2000):
        if pos < len(data) and dec.feed(data[pos:pos + 200000], 0, 5):
            pos += 200000
        f = dec.receive(5)
        if f is not None:
            return dec, f
    raise AssertionError("the decoder produced no frame")


# --------------------------------------------------------------------------- #
# numpy reference                                                              #
# --------------------------------------------------------------------------- #
def ref_yuv(rgb):
    r, g, b = (int(v) for v in rgb[:3])
    y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
    cb = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128
    cr = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128
    return y, cb, cr


def ref_bands(w_img, h_img, box, thickness):
    """Bands of one outline in frame pixels, after the shared normalisation."""
    x1, y1, x2, y2 = box
    x, y = int(np.floor(x1)), int(np.floor(y1))
    w, h = int(np.ceil(x2)) - x, int(np.ceil(y2)) - y
    if thickness <= 0 or w <= 0 or h <= 0:
        return []
    x0, y0 = max(x, 0), max(y, 0)
    xe, ye = min(x + w, w_img), min(y + h, h_img)
    x0 &= ~1
    y0 &= ~1
    xe = min(w_img, (xe + 1) & ~1)
    ye = min(h_img, (ye + 1) & ~1)
    t = (thickness + 1) & ~1
    if xe <= x0 or ye <= y0:
        return []
    bw, bh = xe - x0, ye - y0
    if not (bw > 2 * t and bh > 2 * t):
        return [(x0, y0, bw, bh)]
    return [(x0, y0, bw, t), (x0, ye - t, bw, t), (x0, y0 + t, t, bh - 2 * t),
            (xe - t, y0 + t, t, bh - 2 * t)]


def ref_draw(nv12, boxes, colors, thickness):
    """Paint outlines onto an (H*3/2, W) NV12 array, in order."""
    out = nv12.copy()
    h = out.shape[0] * 2 // 3
    w = out.shape[1]
    for box, col in zip(boxes, colors):
        yv, cb, cr = ref_yuv(col)
        for (x, y, bw, bh) in ref_bands(w, h, box, thickness):
            out[y:y + bh, x:x + bw] = yv
            cy0, cy1 = h + y // 2, h + (y + bh) // 2
            out[cy0:cy1, x:x + bw:2] = cb
            out[cy0:cy1, x + 1:x + bw:2] = cr
    return out


# Odd coordinates, a box off the right edge, one off the left edge, two too
# small for an outline, and overlaps between different colours (so the paint
# ORDER is tested too).
BOXES = [(100.3, 80.7, 400.2, 300.9), (500, 500, 900, 700), (1700, 1000, 2000, 1200),
         (10, 10, 13, 13), (50, 50, 300, 52), (-20, 300, 40, 500), (300, 250, 700, 600),
         (350, 200, 550, 560)]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
          (255, 128, 0), (255, 255, 255)]


def test_dma32_pool_marks_its_frames():
    m = _rcdl()
    data = open(_stream(), "rb").read()
    dec, f = _first_frame(m, data, pool_heap="system")
    assert dec.pool_heap == "system"
    assert f.below_4g is False
    f.release()
    try:
        dec, f = _first_frame(m, data, pool_heap="system-dma32")
    except Exception as e:  # noqa: BLE001 — a board without the heap
        pytest.skip(f"no usable system-dma32 heap: {e}")
    if dec.pool_heap != "system-dma32":
        pytest.skip("the decoder fell back to the system heap (no dma32 heap on this image)")
    assert f.below_4g is True
    f.release()


def test_cpu_overlay_matches_the_reference():
    m = _rcdl()
    data = open(_stream(), "rb").read()
    dec, f = _first_frame(m, data)
    before = f.to_numpy()
    assert f.draw_rects(BOXES, COLORS, thickness=3, backend="cpu") == "cpu"
    after = f.to_numpy()
    f.release()
    ref = ref_draw(before, BOXES, COLORS, 3)
    diff = np.abs(after.astype(int) - ref.astype(int))
    assert diff.max() == 0, f"{int((diff > 0).sum())} bytes differ from the reference"
    # and it only touched the outlines
    touched = (after != before)
    assert touched.any()
    assert not (touched & (ref == before)).any()


def test_hardware_overlay_matches_the_cpu_byte_for_byte():
    m = _rcdl()
    data = open(_stream(), "rb").read()
    try:
        dec, f = _first_frame(m, data, pool_heap="system-dma32")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no usable system-dma32 heap: {e}")
    if not f.below_4g:
        pytest.skip("frames are not below 4 GB on this board; the hardware overlay cannot run")
    before = f.to_numpy()
    try:
        used = f.draw_rects(BOXES, COLORS, thickness=3, backend="rga")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"hardware overlay refused on this board: {e}")
    assert used == "rga"
    hw = f.to_numpy()
    f.release()
    ref = ref_draw(before, BOXES, COLORS, 3)
    diff = np.abs(hw.astype(int) - ref.astype(int))
    assert diff.max() == 0, f"{int((diff > 0).sum())} bytes differ between RGA2 and the reference"


def test_overlay_colour_is_studio_range():
    """A pure red box lands on the hardware's own values, Y=82 Cb=90 Cr=240."""
    m = _rcdl()
    data = open(_stream(), "rb").read()
    dec, f = _first_frame(m, data)
    f.draw_rects([(100, 80, 400, 300)], (255, 0, 0), thickness=4, backend="cpu")
    a = f.to_numpy()
    f.release()
    h = a.shape[0] * 2 // 3
    assert (a[80:84, 100:400] == 82).all()
    assert (a[h + 40:h + 42, 100:400:2] == 90).all()
    assert (a[h + 40:h + 42, 101:400:2] == 240).all()
    assert ref_yuv((0, 255, 0))[0] == 144
