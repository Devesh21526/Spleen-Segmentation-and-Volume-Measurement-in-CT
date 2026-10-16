"""3D U-Net definition and checkpoint loading (PyTorch or ONNX)."""
from pathlib import Path

import numpy as np
import torch
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from src.config import UNET_KWARGS


def build_unet() -> torch.nn.Module:
    return UNet(**UNET_KWARGS, norm=Norm.BATCH)


def load_model(path, device="cpu"):
    """Callable predictor(x[B,1,D,H,W]) -> logits[B,2,D,H,W], from either a .pth or a .onnx file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if path.suffix == ".onnx":
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def predictor(x: torch.Tensor) -> torch.Tensor:
            out = sess.run(None, {name: x.detach().cpu().numpy().astype(np.float32)})[0]
            return torch.from_numpy(out).to(x.device)

        return predictor

    model = build_unet().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
