"""Engine smoke tests on a real .rknn (board only).

    PYTHONPATH=build:python pytest -s tests/test_engine.py --model models/resnet18_rk3588.rknn
"""

import numpy as np
import pytest


def test_load_and_infer(rcdl_mod, model_path):
    rcdl = rcdl_mod
    e = rcdl.Engine(model_path)
    print(f"\nrcdl {rcdl.__version__} | librknnrt {e.sdk_version()} | driver {e.driver_version()}")
    assert e.num_inputs >= 1 and e.num_outputs >= 1

    inputs = []
    for i in range(e.num_inputs):
        shape = e.input_shape(i)
        dt = e.input_dtype(i)
        print(f"  in[{i}] {e.input_name(i)} shape={shape} dtype={dt} fmt={e.input_format(i)} "
              f"bytes={e.input_bytes(i)} packed={e.input_packed_bytes(i)}")
        inputs.append(np.zeros(shape, dtype=dt))

    outs = e.infer(inputs)
    assert len(outs) == e.num_outputs
    for i, o in enumerate(outs):
        print(f"  out[{i}] {e.output_name(i)} shape={o.shape} dtype={o.dtype} "
              f"quant={e.output_quant(i)} npu_us={e.last_run_micros()}")
        assert o.dtype == np.float32
        assert list(o.shape) == list(e.output_shape(i))
        assert np.isfinite(o).all()


def test_outputs_are_deterministic(rcdl_mod, model_path):
    e = rcdl_mod.Engine(model_path)
    rng = np.random.default_rng(0)
    x = [rng.integers(0, 255, size=e.input_shape(i), dtype=np.uint8)
         if e.input_dtype(i) == np.uint8 else rng.standard_normal(e.input_shape(i)).astype(np.float32)
         for i in range(e.num_inputs)]
    a = e.infer(x)
    b = e.infer(x)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)


def test_wrong_input_size_raises(rcdl_mod, model_path):
    e = rcdl_mod.Engine(model_path)
    bad = np.zeros(e.input_packed_bytes(0) + 7, dtype=np.uint8)
    with pytest.raises(Exception):
        e._e.set_input(0, bad)


def test_index_out_of_range_raises(rcdl_mod, model_path):
    e = rcdl_mod.Engine(model_path)
    for bad in (-1, e.num_outputs, 99):
        with pytest.raises(Exception):
            e.output_shape(bad)
        with pytest.raises(Exception):
            e.output(bad)


def test_dup_context_matches(rcdl_mod, model_path):
    rcdl = rcdl_mod
    e = rcdl.Engine(model_path, rcdl.NpuCore.CORE_0)
    d = e.dup(rcdl.NpuCore.CORE_1)
    x = [np.full(e.input_shape(i), 7, dtype=e.input_dtype(i)) for i in range(e.num_inputs)]
    a = e.infer(x)
    b = d.infer(x)
    for u, v in zip(a, b):
        np.testing.assert_allclose(u, v, rtol=0, atol=0)
