"""Rewrite the ONNX graph into the subset an NPU toolchain is comfortable with.

Three changes, all numerically exact or near-exact:
  PRelu -> LeakyRelu   every slope in this network is a single scalar, so the ops are equivalent
  BatchNormalization   folded into the preceding Conv / ConvTranspose weights
  batch dimension      pinned to 1, because QNN and TensorRT both want static shapes

What this cannot do from a laptop is run the Qualcomm converter; see docs/qnn_deployment.md
for the commands that turn the result into a context binary on the target.

    python -m src.models.qnn_prepare --onnx models/spleen_unet.onnx
"""
import argparse
import collections

import numpy as np
import onnx
from onnx import numpy_helper as nh


def _constants(model):
    """Initialisers plus Constant nodes: the exporter uses either, depending on the values."""
    values = {i.name: nh.to_array(i) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    values[node.output[0]] = nh.to_array(attr.t)

    for _ in range(4):
        for node in model.graph.node:
            if node.op_type == "Identity" and node.input[0] in values:
                values.setdefault(node.output[0], values[node.input[0]])
    return values


def op_histogram(model):
    return dict(collections.Counter(n.op_type for n in model.graph.node))


def prelu_to_leakyrelu(model) -> int:
    init = _constants(model)
    changed = 0
    for node in model.graph.node:
        if node.op_type != "PRelu":
            continue
        slope = init.get(node.input[1])
        if slope is None or slope.size != 1:
            continue
        alpha = float(slope.ravel()[0])
        del node.input[1]
        node.op_type = "LeakyRelu"
        node.attribute.append(onnx.helper.make_attribute("alpha", alpha))
        changed += 1
    _drop_unused_initializers(model)
    return changed


def fold_batchnorm(model) -> int:
    init = _constants(model)
    producer = {o: n for n in model.graph.node for o in n.output}
    consumers = collections.Counter(i for n in model.graph.node for i in n.input)

    folded, drop = 0, []
    for bn in model.graph.node:
        if bn.op_type != "BatchNormalization":
            continue
        conv = producer.get(bn.input[0])
        if conv is None or conv.op_type not in ("Conv", "ConvTranspose") or consumers[bn.input[0]] != 1:
            continue

        gamma, beta, mean, var = (init[n] for n in bn.input[1:5])
        eps = next((a.f for a in bn.attribute if a.name == "epsilon"), 1e-5)
        scale = gamma / np.sqrt(var + eps)

        w = init[conv.input[1]]

        axis = 1 if conv.op_type == "ConvTranspose" else 0
        b = init[conv.input[2]] if len(conv.input) > 2 else np.zeros(w.shape[axis], w.dtype)
        shape = [1] * w.ndim
        shape[axis] = -1
        w_new = w * scale.reshape(shape)
        b_new = (b - mean) * scale + beta

        _set_value(model, conv.input[1], w_new.astype(w.dtype))
        if len(conv.input) > 2:
            _set_value(model, conv.input[2], b_new.astype(w.dtype))
        else:
            name = conv.name + "_bias"
            model.graph.initializer.append(nh.from_array(b_new.astype(w.dtype), name))
            conv.input.append(name)

        conv.output[0] = bn.output[0]
        drop.append(bn)
        folded += 1

    for bn in drop:
        model.graph.node.remove(bn)
    _drop_unused_initializers(model)
    return folded


def pin_batch(model, batch=1):
    for tensor in list(model.graph.input) + list(model.graph.output):
        dim = tensor.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_param")
        dim.dim_value = batch


def _set_value(model, name, array):
    for i, old in enumerate(model.graph.initializer):
        if old.name == name:
            model.graph.initializer[i].CopyFrom(nh.from_array(array, name))
            return
    for node in model.graph.node:
        if node.op_type == "Constant" and node.output[0] == name:
            del node.attribute[:]
            node.attribute.append(onnx.helper.make_attribute("value", nh.from_array(array, name)))
            return
        if node.op_type == "Identity" and node.output[0] == name:
            fresh = name + "_folded"
            model.graph.initializer.append(nh.from_array(array, fresh))
            node.op_type = "Identity"
            node.input[0] = fresh
            return
    raise KeyError(name)


def _drop_unused_initializers(model):
    used = {i for n in model.graph.node for i in n.input}
    keep = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)

    orphans = [n for n in model.graph.node
               if n.op_type in ("Constant", "Identity") and n.output[0] not in used]
    for node in orphans:
        model.graph.node.remove(node)


def parity(src, dst, shape, runs=1):
    import onnxruntime as ort

    a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(runs):
        x = rng.random(shape, dtype=np.float32)
        ya = a.run(None, {a.get_inputs()[0].name: x})[0]
        yb = b.run(None, {b.get_inputs()[0].name: x})[0]
        worst = max(worst, float(np.abs(ya - yb).max()))
        agree = float((ya.argmax(1) == yb.argmax(1)).mean())
    return worst, agree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/spleen_unet.onnx")
    ap.add_argument("--out", default="models/spleen_unet_qnn.onnx")
    ap.add_argument("--patch", type=int, default=96)
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    before = op_histogram(model)

    n_act = prelu_to_leakyrelu(model)
    n_bn = fold_batchnorm(model)
    pin_batch(model)
    onnx.checker.check_model(model)
    onnx.save(model, args.out)

    print(f"PRelu -> LeakyRelu: {n_act} | BatchNorm folded: {n_bn}")
    print("before:", before)
    print("after: ", op_histogram(model))

    shape = (1, 1, args.patch, args.patch, args.patch)
    worst, agree = parity(args.onnx, args.out, shape)
    print(f"max |logit diff| vs original {worst:.3e} | argmax agreement {agree:.5f}")

    remaining = set(op_histogram(model))
    print("\nops the Qualcomm converter still has to cover:", sorted(remaining))
    print("Conv/ConvTranspose here are 3D. HTP coverage for 3D kernels depends on the QNN SDK "
          "version, so that is the first thing to check on the target, before quantisation.")


if __name__ == "__main__":
    main()
