"""Export the PyTorch checkpoint to ONNX and check both backends agree.

    python -m src.models.export_onnx --model models/best_model_ft.pth --out models/spleen_unet.onnx
"""
import argparse
import time

import torch

from src.config import DEFAULT_MODEL, ROI_SIZE
from src.models.unet import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_MODEL.with_name("spleen_unet.onnx")))
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = load_model(args.model, "cpu")
    dummy = torch.randn(1, 1, *ROI_SIZE)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Saved {args.out}")

    onnx_pred = load_model(args.out)
    x = torch.rand(2, 1, *ROI_SIZE)
    with torch.no_grad():
        t = time.time(); ref = model(x); t_pt = time.time() - t
    t = time.time(); out = onnx_pred(x); t_ort = time.time() - t
    diff = float((ref - out).abs().max())
    same = float((ref.argmax(1) == out.argmax(1)).float().mean())
    print(f"max |logit diff| {diff:.2e} | argmax agreement {same:.5f}")
    print(f"CPU latency, batch of 2 patches: PyTorch {t_pt:.2f}s | ONNX Runtime {t_ort:.2f}s")
    assert same > 0.999, "ONNX output disagrees with PyTorch"


if __name__ == "__main__":
    main()
