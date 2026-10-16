import numpy as np
import torch

from src.inference.predict import predict
from src.inference.uncertainty import FLIP_SETS, FLIP_SETS_ALL, summarise, tta_probabilities
from src.models.unet import load_model


def test_flip_sets_cover_every_combination():
    assert len(FLIP_SETS_ALL) == 8 and len(FLIP_SETS) == 4
    assert len(set(FLIP_SETS_ALL)) == 8 and (0,) not in FLIP_SETS


def test_tta_views_return_to_the_original_frame():

    def predictor(x):
        return torch.cat([torch.zeros_like(x), x * 10], dim=1)

    torch.manual_seed(1)
    x = torch.rand(1, 1, 32, 32, 32)
    probs = tta_probabilities(x, predictor, flip_sets=FLIP_SETS_ALL)
    assert probs.shape == (8, 32, 32, 32)
    for view in probs[1:]:
        assert np.allclose(view, probs[0], atol=1e-5)


def test_summarise_reports_spread_over_the_mask():
    probs = np.zeros((4, 20, 20, 20), np.float32)
    probs[:, 5:15, 5:15, 5:15] = 0.9
    probs[0, 5:15, 5:15, 5:15] = 0.6
    mask = (probs[0] > 0.5).astype(np.uint8)

    summary, mean, std = summarise(probs, mask)
    assert summary["n_views"] == 4
    assert summary["volume_std_ml"] == 0.0
    assert summary["volume_cv_pct"] == 0.0
    assert summary["mean_voxel_std_in_mask"] > 0
    assert mean.shape == std.shape == (20, 20, 20)


def test_summarise_handles_an_empty_prediction():
    summary, _, _ = summarise(np.zeros((2, 10, 10, 10), np.float32))
    assert summary["volume_mean_ml"] == 0.0
    assert summary["mean_voxel_std_in_mask"] is None


def test_predict_with_uncertainty_adds_volume_statistics(ct_file, dummy_model):
    res = predict(ct_file, load_model(dummy_model), uncertainty=True)
    u = res["uncertainty"]
    assert u["n_views"] == 4
    assert u["volume_min_ml"] <= u["volume_mean_ml"] <= u["volume_max_ml"]
    assert res["_std"].shape == res["_mask"].shape
