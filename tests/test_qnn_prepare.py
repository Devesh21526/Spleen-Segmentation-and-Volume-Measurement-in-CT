import numpy as np
import onnx
import pytest
import torch

from src.models.qnn_prepare import (_constants, fold_batchnorm, op_histogram, pin_batch,
                                    prelu_to_leakyrelu)
from src.models.unet import build_unet


@pytest.fixture(scope="module")
def small_onnx(tmp_path_factory):
    torch.manual_seed(0)
    net = build_unet().eval()
    path = tmp_path_factory.mktemp("onnx") / "net.onnx"
    torch.onnx.export(net, torch.randn(1, 1, 32, 32, 32), str(path),
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17)
    return path


def test_prelu_becomes_leakyrelu_with_the_learned_slope(small_onnx):
    model = onnx.load(small_onnx)
    values = _constants(model)
    slopes = sorted(float(values[n.input[1]].ravel()[0])
                    for n in model.graph.node if n.op_type == "PRelu")

    assert prelu_to_leakyrelu(model) == len(slopes)
    assert "PRelu" not in op_histogram(model)
    alphas = sorted(a.f for n in model.graph.node if n.op_type == "LeakyRelu" for a in n.attribute)
    assert alphas == pytest.approx(slopes, abs=1e-6)


def test_batchnorm_is_folded_away(small_onnx):
    model = onnx.load(small_onnx)
    n_bn = op_histogram(model).get("BatchNormalization", 0)
    assert fold_batchnorm(model) == n_bn
    assert "BatchNormalization" not in op_histogram(model)
    onnx.checker.check_model(model)


def test_rewrites_preserve_the_output(small_onnx, tmp_path):
    ort = pytest.importorskip("onnxruntime")

    model = onnx.load(small_onnx)
    prelu_to_leakyrelu(model)
    fold_batchnorm(model)
    pin_batch(model)
    out = tmp_path / "qnn.onnx"
    onnx.save(model, out)

    x = np.random.default_rng(0).random((1, 1, 32, 32, 32), dtype=np.float32)
    a = ort.InferenceSession(str(small_onnx), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    ya = a.run(None, {"image": x})[0]
    yb = b.run(None, {"image": x})[0]

    assert np.abs(ya - yb).max() < 1e-4
    assert (ya.argmax(1) == yb.argmax(1)).all()


def test_pinned_batch_is_static(small_onnx):
    model = onnx.load(small_onnx)
    pin_batch(model)
    for t in list(model.graph.input) + list(model.graph.output):
        assert t.type.tensor_type.shape.dim[0].dim_value == 1
