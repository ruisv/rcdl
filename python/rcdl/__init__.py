"""RCDL — RKNPU Computational Deep Learning (Python wrapper).

Thin numpy-friendly layer over the compiled ``rcdl_py`` extension. The C++ core
exchanges raw bytes and float32 buffers; this wrapper keeps the Python API
small and numpy-shaped.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import rcdl_py
from rcdl_py import DmaBuf, DmaHeap, NpuCore, dequantize

__version__ = rcdl_py.__version__

__all__ = [
    "Engine",
    "DmaBuf",
    "DmaHeap",
    "NpuCore",
    "dequantize",
    "__version__",
]


class Engine:
    """Load an ``.rknn`` model and run NPU inference with numpy in/out.

    Inputs are given as they come off an image pipeline — ``uint8`` HWC arrays
    for quantized models (the toolkit folds mean/std into the graph), ``float32``
    for float models. Outputs come back as float32 arrays in the model's own
    shape, already dequantized.
    """

    def __init__(self, path: str, core: NpuCore = NpuCore.AUTO, init_flags: int = 0):
        self._e = rcdl_py.Engine(path, core, init_flags)

    @classmethod
    def _wrap(cls, raw: "rcdl_py.Engine") -> "Engine":
        obj = cls.__new__(cls)
        obj._e = raw
        return obj

    def dup(self, core: NpuCore = NpuCore.AUTO) -> "Engine":
        """A second context sharing this model's weights, optionally pinned to
        another NPU core (the way to use all three RK3588 cores concurrently)."""
        return Engine._wrap(self._e.dup(core))

    # --- introspection ---------------------------------------------------
    @property
    def path(self) -> str:
        return self._e.path

    @property
    def core(self) -> NpuCore:
        return self._e.core

    @property
    def num_inputs(self) -> int:
        return self._e.num_inputs

    @property
    def num_outputs(self) -> int:
        return self._e.num_outputs

    def input_shape(self, i: int) -> list[int]:
        return self._e.input_shape(i)

    def output_shape(self, i: int) -> list[int]:
        return self._e.output_shape(i)

    def input_name(self, i: int) -> str:
        return self._e.input_name(i)

    def output_name(self, i: int) -> str:
        return self._e.output_name(i)

    def input_dtype(self, i: int) -> np.dtype:
        return _np_dtype(self._e.input_dtype(i))

    def output_dtype(self, i: int) -> np.dtype:
        return _np_dtype(self._e.output_dtype(i))

    def input_format(self, i: int) -> str:
        """'NHWC' / 'NCHW' — the layout input i is provided in."""
        return self._e.input_format(i)

    def input_bytes(self, i: int) -> int:
        return self._e.input_bytes(i)

    def input_packed_bytes(self, i: int) -> int:
        return self._e.input_packed_bytes(i)

    def input_width_stride(self, i: int) -> int:
        return self._e.input_width_stride(i)

    def output_bytes(self, i: int) -> int:
        return self._e.output_bytes(i)

    def output_quant(self, i: int) -> tuple[int, int, float, int]:
        """(qnt_type, zero_point, scale, fl) of output i."""
        return self._e.output_quant(i)

    # --- data path -------------------------------------------------------
    def set_input(self, i: int, array: np.ndarray) -> None:
        arr = np.ascontiguousarray(array)
        want = self.input_dtype(i)
        if arr.dtype != want:
            arr = arr.astype(want, copy=False)
        self._e.set_input(i, arr)

    def run(self) -> None:
        """Run one inference on the inputs set so far (blocking, GIL released)."""
        self._e.infer()

    def output(self, i: int) -> np.ndarray:
        """Output i as a dequantized float32 array in the model's shape."""
        return self._e.output_float(i)

    def output_raw(self, i: int) -> bytes:
        return self._e.output_raw(i)

    def infer(self, inputs: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
        """Set every input, run, and return all outputs as float32 arrays."""
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        if len(inputs) != self.num_inputs:
            raise ValueError(f"expected {self.num_inputs} inputs, got {len(inputs)}")
        for i, a in enumerate(inputs):
            self.set_input(i, a)
        self.run()
        return [self.output(i) for i in range(self.num_outputs)]

    # --- diagnostics ----------------------------------------------------
    def last_run_micros(self) -> int:
        return self._e.last_run_micros()

    def perf_detail(self) -> str:
        return self._e.perf_detail()

    def sdk_version(self) -> str:
        return self._e.sdk_version()

    def driver_version(self) -> str:
        return self._e.driver_version()


_DTYPES = {
    "f32": np.float32,
    "f16": np.float16,
    "i8": np.int8,
    "u8": np.uint8,
    "i16": np.int16,
    "u16": np.uint16,
    "i32": np.int32,
    "u32": np.uint32,
    "i64": np.int64,
    "bool": np.bool_,
}


def _np_dtype(name: str) -> np.dtype:
    try:
        return np.dtype(_DTYPES[name])
    except KeyError as exc:
        raise ValueError(f"unsupported tensor dtype {name!r}") from exc
