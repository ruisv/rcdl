"""Shrink the SANM encoder's residual stream by a power of two so it fits fp16.

Paraformer-large's 49 SANM blocks keep adding into one residual stream, and in
the last third of the encoder it passes 500 000, far past fp16's 65 504; the
feed-forward output that is added in reaches 89 000 on its own. On an fp16 NPU
that is inf, then NaN, and every transcript comes back empty.

Every block reads the stream only through a LayerNorm, and LayerNorm does not
care about scale. So scale the stream by s, and every branch added into it by s
too:
  - the stream as it leaves encoders0 gets one Mul(s);
  - each self-attention branch gets a Mul(s) before its residual Add;
  - each feed-forward branch has its w_2 weight and bias scaled by s, so that the
    MatMul itself no longer overflows.
after_norm, the LayerNorm at the end, puts the scale back. s = 1/32 is exact in
binary and leaves the stream under 16 000.
"""
import re
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

S = 1.0 / 32


def rescale(src, dst):
    m = onnx.load(src)
    g = m.graph
    nodes = {n.name: n for n in g.node}
    prod = {o: n for n in g.node for o in n.output}
    inits = {i.name: i for i in g.initializer}
    use = {}
    for n in g.node:
        for i in n.input:
            use[i] = use.get(i, 0) + 1
    g.initializer.append(numpy_helper.from_array(np.array(S, np.float32), "resid_scale"))
    added = []

    def scale_edge(tensor, consumers, tag):
        out = f"{tensor}_x{tag}"
        added.append((prod[tensor].name, helper.make_node("Mul", [tensor, "resid_scale"], [out], name=f"{tag}/resid_scale")))
        for c in consumers:
            for k, i in enumerate(c.input):
                if i == tensor:
                    c.input[k] = out

    # 1. the stream entering the 49-block stack
    first = nodes["/enc/encoders/encoders.0/Add"]
    stream = next(i for i in first.input if prod[i].name.startswith("/enc/encoders0/"))
    scale_edge(stream, [n for n in g.node if stream in n.input], "enc_in")

    blocks = sorted({int(x) for x in re.findall(r"/enc/encoders/encoders\.(\d+)/", " ".join(nodes))})
    for b in blocks:
        p = f"/enc/encoders/encoders.{b}"
        # 2. self-attention branch -> Mul(s) on the edge into the residual Add
        att = nodes[f"{p}/self_attn/Add_2"].output[0]
        scale_edge(att, [nodes[f"{p}/Add"]], f"{p}/self_attn")
        # 3. feed-forward branch -> scale w_2's weight and bias
        mm, bias_add = nodes[f"{p}/feed_forward/w_2/MatMul"], nodes[f"{p}/feed_forward/w_2/Add"]
        for t in [i for i in mm.input if i in inits] + [i for i in bias_add.input if i in inits]:
            assert use[t] == 1, f"{t} is shared"
            arr = numpy_helper.to_array(inits[t]) * S
            inits[t].CopyFrom(numpy_helper.from_array(arr.astype(np.float32), t))

    order = []
    for n in g.node:
        order.append(n)
        order.extend(new for after, new in added if after == n.name)
    del g.node[:]
    g.node.extend(order)
    onnx.save(m, dst)
    print(f"scaled the residual stream of {len(blocks)} blocks by {S} -> {dst}")


if __name__ == "__main__":
    rescale(sys.argv[1], sys.argv[2])
