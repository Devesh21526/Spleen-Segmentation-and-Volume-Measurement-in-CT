"""Dynamic int8 quantisation of the ONNX model.

Weight-only dynamic quantisation needs no calibration set, which is why it is the
first thing to try. Static int8 for an NPU does need real CT patches; see the
deployment section of the report.

    python -m src.models.quantize_onnx --onnx models/spleen_unet.onnx
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.config import ROI_SIZE
from src.models.unet import load_model


def benchmark(predictor, x, repeats=3):
    predictor(x)
    times = []
    for _ in range(repeats):
        t = time.time()
        out = predictor(x)
        times.append(time.time() - t)
    return out, min(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/spleen_unet.onnx")
    ap.add_argument("--out", default=None, help="default: <onnx stem>_int8.onnx")
    ap.add_argument("--patches", type=int, default=2)
    args = ap.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    src = Path(args.onnx)
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_int8.onnx")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f"{src.name} {src.stat().st_size / 1e6:.1f} MB -> {dst.name} {dst.stat().st_size / 1e6:.1f} MB")

    fp32 = load_model(src)
    x = torch.rand(args.patches, 1, *ROI_SIZE)
    ref, t_fp32 = benchmark(fp32, x)
    print(f"fp32 CPU latency for {args.patches} patches: {t_fp32:.2f}s")

    try:
        int8 = load_model(dst)
    except Exception as e:
        print(f"\nint8 model will not run on this CPU provider:\n  {type(e).__name__}: {e}")
        print("The 2.6x smaller file is still the useful part: int8 3D convolution is a "
              "kernel question for the target runtime (TensorRT, QNN), not for the x86 CPU EP.")
        return

    out, t_int8 = benchmark(int8, x)
    ref_fg, out_fg = ref.argmax(1).numpy(), out.argmax(1).numpy()
    inter = np.logical_and(ref_fg, out_fg).sum()
    dice = float(2 * inter / (ref_fg.sum() + out_fg.sum())) if (ref_fg.sum() + out_fg.sum()) else 1.0

    print(f"max |logit diff| {float((ref - out).abs().max()):.3e}")
    print(f"voxel agreement {float((ref.argmax(1) == out.argmax(1)).float().mean()):.5f} | "
          f"Dice vs fp32 prediction {dice:.4f}")
    print(f"int8 CPU latency for {args.patches} patches: {t_int8:.2f}s")
    print("Quantisation moves boundary voxels first, so re-measure Dice on real scans "
          "before using this checkpoint for volumetry.")


if __name__ == "__main__":
    main()
