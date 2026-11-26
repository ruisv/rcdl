"""Replace every Sin -> Pow(2) pair with an NPU-only sin^2.

RK3588 has no NPU Sin; each one falls back to the CPU. sin^2 has period pi, so the
input is folded exactly onto [0, pi/2] with nested Abs, which is continuous and
needs no rounding:
  a = clip(|y|, 0, 4 pi);  d = pi/2 - | | |a - 2 pi| - pi | - pi/2 |
d is the distance from |y| to the nearest multiple of pi, and sin^2(y) = sin^2(d).
A degree-9 odd polynomial gives sin(d) on [0, pi/2] to 4e-6. Inputs past 4 pi are
clamped there; the measured maximum is 5.8.
"""
import math
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

C = [1.0, -1 / 6, 1 / 120, -1 / 5040, 1 / 362880]


def rewrite(src, dst):
    m = onnx.load(src)
    g = m.graph
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    inits = {i.name: i for i in g.initializer}
    consts = {}

    def const(v):
        key = float(v)
        if key not in consts:
            name = f"sin2_c{len(consts)}"
            g.initializer.append(numpy_helper.from_array(np.array(v, np.float32), name))
            consts[key] = name
        return consts[key]

    def exponent(pw):
        if pw.input[1] in inits:
            return float(numpy_helper.to_array(inits[pw.input[1]]))
        for c in g.node:
            if c.op_type == "Constant" and c.output[0] == pw.input[1]:
                return float(numpy_helper.to_array(c.attribute[0].t))
        raise ValueError(pw.name)

    repl, drop, k = {}, set(), 0
    for n in list(g.node):
        if n.op_type != "Sin":
            continue
        users = cons.get(n.output[0], [])
        assert len(users) == 1 and users[0].op_type == "Pow", n.name
        pw = users[0]
        assert exponent(pw) == 2.0, pw.name
        y, out, p = n.input[0], pw.output[0], f"sin2_{k}"
        k += 1
        nodes = []

        def nd(op, ins, name):
            nodes.append(helper.make_node(op, ins, [f"{p}/{name}"], name=f"{p}/{name}"))
            return f"{p}/{name}"

        a = nd("Clip", [nd("Abs", [y], "abs"), const(0.0), const(4 * math.pi)], "clamp")
        f = nd("Abs", [nd("Sub", [a, const(2 * math.pi)], "s1")], "f")
        h = nd("Abs", [nd("Sub", [f, const(math.pi)], "s2")], "g")
        h = nd("Abs", [nd("Sub", [h, const(math.pi / 2)], "s3")], "h")
        d = nd("Sub", [const(math.pi / 2), h], "d")
        d2 = nd("Mul", [d, d], "d2")
        acc = const(C[4])
        for j, c in enumerate(reversed(C[:4])):
            acc = nd("Add", [nd("Mul", [acc, d2], f"m{j}"), const(c)], f"a{j}")
        s = nd("Mul", [acc, d], "sin")
        nodes.append(helper.make_node("Mul", [s, s], [out], name=f"{p}/sq"))
        repl[id(n)] = nodes
        drop.update({id(n), id(pw)})

    order = []
    for n in g.node:
        if id(n) in repl:
            order.extend(repl[id(n)])
        elif id(n) not in drop:
            order.append(n)
    del g.node[:]
    g.node.extend(order)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"replaced {k} Sin/Pow pairs -> {dst}")


if __name__ == "__main__":
    rewrite(sys.argv[1], sys.argv[2])
