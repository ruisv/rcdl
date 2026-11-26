"""List the nodes of an ONNX graph whose fp32 values leave the fp16 range.

An fp16 build of a graph that passes 65 504 anywhere returns inf, then NaN, and
the toolkit does not warn. Run this on the float ONNX with a real input first:

    python find_overflow.py model.onnx inputs.npz

`inputs.npz` holds one array per graph input, under the input's name.
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort

FP16_MAX = 65504.0


def main(src, inputs_npz):
    m = onnx.load(src)
    inits = {i.name for i in m.graph.initializer}
    inferred = onnx.shape_inference.infer_shapes(m)
    elem = {v.name: v.type.tensor_type.elem_type
            for v in list(inferred.graph.value_info) + list(inferred.graph.output)}
    probes = [(o, n.op_type, n.name) for n in m.graph.node for o in n.output
              if o and o not in inits and elem.get(o) == onnx.TensorProto.FLOAT]
    have = {o.name for o in m.graph.output}
    for o, _, _ in probes:
        if o not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    data = np.load(inputs_npz)
    feed = {i.name: data[i.name] for i in sess.get_inputs()}
    rows = []
    for (o, op, name), v in zip(probes, sess.run([p[0] for p in probes], feed)):
        v = np.asarray(v)
        if v.size:
            rows.append((float(np.abs(v).max()), op, name))
    rows.sort(reverse=True)
    print(f"largest |values| (fp16 stops at {FP16_MAX:.0f}):")
    for mx, op, name in rows[:20]:
        print(f"  {mx:12.1f}  {op:<20} {name}")
    over = [r for r in rows if r[0] > FP16_MAX]
    print(f"{len(over)} nodes exceed fp16" + (f": {sorted({r[1] for r in over})}" if over else ""))
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
