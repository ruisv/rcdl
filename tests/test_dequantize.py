"""Output dequantisation math, pinned without an NPU or a model.

Uses the ``rcdl.dequantize`` hook (same code path as ``Engine.output``), so it
needs the compiled module but not the hardware.
"""

import numpy as np
import pytest


@pytest.fixture
def dq(rcdl_mod):
    return rcdl_mod.dequantize


def test_int8_affine(dq):
    q = np.array([-128, -70, 0, 57, 127], dtype=np.int8)
    zp, scale = -70, 0.118524
    out = dq(q.tobytes(), "i8", qnt_type=2, zp=zp, scale=scale)
    np.testing.assert_allclose(out, (q.astype(np.float32) - zp) * scale, rtol=1e-6)


def test_uint8_affine(dq):
    q = np.array([0, 13, 128, 255], dtype=np.uint8)
    out = dq(q.tobytes(), "u8", qnt_type=2, zp=13, scale=0.0184776)
    np.testing.assert_allclose(out, (q.astype(np.float32) - 13) * 0.0184776, rtol=1e-6)


def test_fp16_roundtrip(dq):
    v = np.array([0.0, -0.0, 1.0, -2.5, 65504.0, 6.1e-5, 5.96e-8, np.inf], dtype=np.float16)
    out = dq(v.tobytes(), "f16")
    np.testing.assert_array_equal(out, v.astype(np.float32))


def test_fp16_nan(dq):
    v = np.array([np.nan], dtype=np.float16)
    assert np.isnan(dq(v.tobytes(), "f16"))[0]


def test_fp32_passthrough(dq):
    v = np.arange(5, dtype=np.float32) * 0.1
    np.testing.assert_array_equal(dq(v.tobytes(), "f32"), v)


def test_dfp_int8(dq):
    q = np.array([-64, 0, 32, 127], dtype=np.int8)
    out = dq(q.tobytes(), "i8", qnt_type=1, fl=5)
    np.testing.assert_allclose(out, q.astype(np.float32) / 32.0)


def test_unknown_dtype(dq):
    with pytest.raises(Exception):
        dq(b"\x00", "q7")
