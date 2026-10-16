"""Flip-based test-time augmentation, used as an uncertainty estimate.

Left-right flips are deliberately not in the default set: mirroring axis 0 puts the
spleen on the patient's right, which no real abdominal CT does, and the model was
only trained with a 10 % flip probability. Those views measure out-of-distribution
behaviour rather than uncertainty. Pass `flip_sets=FLIP_SETS_ALL` to see them.
"""
import itertools

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from src import config as C

FLIP_SETS_ALL = [()] + [c for r in (1, 2, 3) for c in itertools.combinations((0, 1, 2), r)]
FLIP_SETS = [(), (1,), (2,), (1, 2)]


@torch.no_grad()
def tta_probabilities(x: torch.Tensor, predictor, flip_sets=None) -> np.ndarray:
    flip_sets = FLIP_SETS if flip_sets is None else flip_sets
    views = []
    for axes in flip_sets:
        dims = [a + 2 for a in axes]
        xf = torch.flip(x, dims) if dims else x
        logits = sliding_window_inference(xf, C.ROI_SIZE, C.SW_BATCH, predictor, overlap=C.SW_OVERLAP)
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        if dims:
            prob = torch.flip(prob, dims)
        views.append(prob[0, 0].cpu().numpy())
    return np.stack(views)


def summarise(probs: np.ndarray, mask: np.ndarray | None = None):
    from src.inference.predict import postprocess

    mean = probs.mean(axis=0)
    std = probs.std(axis=0)

    volumes = np.array([postprocess(p)[0].sum() * C.VOXEL_ML for p in probs], dtype=float)

    region = (mask > 0) if mask is not None and mask.any() else (mean > 0.5)
    summary = {
        "n_views": int(probs.shape[0]),
        "volume_mean_ml": round(float(volumes.mean()), 1),
        "volume_std_ml": round(float(volumes.std()), 1),
        "volume_min_ml": round(float(volumes.min()), 1),
        "volume_max_ml": round(float(volumes.max()), 1),
        "volume_cv_pct": round(float(100 * volumes.std() / volumes.mean()), 1) if volumes.mean() > 0 else None,
        "mean_voxel_std_in_mask": round(float(std[region].mean()), 4) if region.any() else None,
        "uncertain_voxel_fraction": round(float((std[region] > 0.1).mean()), 4) if region.any() else None,
    }
    return summary, mean, std
