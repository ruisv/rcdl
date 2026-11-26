"""Fixed-shape ONNX -> fp16 .rknn for RK3588, inputs and outputs keeping their ONNX names.

    python conv_fp16.py model.onnx model_fp16_rk3588.rknn

The task classes bind tensors by name or shape, so the graph's own names have to
survive the conversion. Input shapes are read from the ONNX, which must be
static. rknn-toolkit2 2.3.2 (x86 host).
"""
import sys
import time

import onnx
from rknn.api import RKNN


def convert(src, dst):
    m = onnx.load(src)
    inits = {i.name for i in m.graph.initializer}
    ins = [i for i in m.graph.input if i.name not in inits]
    names = [i.name for i in ins]
    sizes = [[d.dim_value for d in i.type.tensor_type.shape.dim] for i in ins]
    if any(0 in s for s in sizes):
        raise SystemExit(f"{src}: every input needs a static shape, got {list(zip(names, sizes))}")
    r = RKNN(verbose=False)
    r.config(target_platform="rk3588", optimization_level=3)
    t = time.time()
    assert r.load_onnx(model=src, inputs=names, input_size_list=sizes) == 0, "load_onnx"
    assert r.build(do_quantization=False) == 0, "build"
    assert r.export_rknn(dst) == 0, "export_rknn"
    print(f">> {dst}  {time.time() - t:.0f}s  inputs {list(zip(names, sizes))}")
    r.release()


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
