"""Replace Max(constant, x) with Relu(x - c) + c (Relu(x) when c == 0).

rknn-toolkit2 2.3.2 lowers a Max whose operand is a constant to the CPU on RK3588.
In the KWS zipformer these are the activations of the convolutional front end
and of the encoder, so they sit on the per-chunk path; the rewrite is exact.
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def rewrite(src, dst):
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    out_nodes, k = [], 0
    for n in g.node:
        consts = [i for i in n.input if i in inits]
        if n.op_type != "Max" or len(n.input) != 2 or len(consts) != 1:
            out_nodes.append(n)
            continue
        c_name = consts[0]
        x = n.input[1] if n.input[0] == c_name else n.input[0]
        c = inits[c_name]
        assert c.size == 1, n.name
        p = f"maxrw_{k}"
        k += 1
        if float(c.reshape(-1)[0]) == 0.0:
            out_nodes.append(helper.make_node("Relu", [x], [n.output[0]], name=f"{p}/relu"))
        else:
            out_nodes.append(helper.make_node("Sub", [x, c_name], [f"{p}/sub"], name=f"{p}/sub"))
            out_nodes.append(helper.make_node("Relu", [f"{p}/sub"], [f"{p}/relu"], name=f"{p}/relu"))
            out_nodes.append(helper.make_node("Add", [f"{p}/relu", c_name], [n.output[0]], name=f"{p}/add"))
    del g.node[:]
    g.node.extend(out_nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"rewrote {k} Max(const, x) -> {dst}")


if __name__ == "__main__":
    rewrite(sys.argv[1], sys.argv[2])
