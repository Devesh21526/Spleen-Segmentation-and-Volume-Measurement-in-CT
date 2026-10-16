import numpy as np

from src import config as C
from src.inference.predict import postprocess, predict, quality_check
from src.models.unet import load_model


def test_volume_formula():
    assert abs(1000 * C.VOXEL_ML - 4.5) < 1e-9


def test_postprocess_keeps_largest_component():
    prob = np.zeros((40, 40, 40), np.float32)
    prob[2:12, 2:12, 2:12] = 0.9
    prob[25:30, 25:30, 25:30] = 0.9
    mask, info = postprocess(prob)
    assert mask.sum() == 1000
    assert info["n_components"] == 2
    assert abs(info["second_ratio"] - 0.125) < 1e-6


def test_postprocess_empty():
    mask, info = postprocess(np.zeros((10, 10, 10), np.float32))
    assert mask.sum() == 0 and info["n_components"] == 0


def _mask_at(x0, x1, shape=(100, 100, 50)):
    m = np.zeros(shape, np.uint8)
    m[x0:x1, 40:60, 10:40] = 1
    return m


def test_quality_ok_for_plausible_left_sided_spleen():
    m = _mask_at(10, 40)
    vol = m.sum() * C.VOXEL_ML
    flag, reasons = quality_check(m, m.astype(np.float32), vol, {"second_ratio": 0.0})
    assert flag == "ok", reasons


def test_quality_flags_right_side_low_conf_and_second_blob():
    m = _mask_at(60, 90)
    prob = m * 0.6
    flag, reasons = quality_check(m, prob, m.sum() * C.VOXEL_ML, {"second_ratio": 0.8})
    assert flag == "review"
    text = " ".join(reasons)
    assert "right side" in text and "confidence" in text and "second candidate" in text


def test_quality_flags_tiny_volume():
    m = np.zeros((50, 50, 50), np.uint8)
    m[5:8, 5:8, 5:8] = 1
    flag, reasons = quality_check(m, m.astype(np.float32), m.sum() * C.VOXEL_ML, {})
    assert flag == "review" and any("below" in r for r in reasons)


def test_end_to_end_predict_runs(ct_file, dummy_model):
    res = predict(ct_file, load_model(dummy_model))
    assert set(["spleen_volume_ml", "quality_flag", "reasons", "inference_time_s"]) <= res.keys()
    assert res["_mask"].shape == res["_image"].shape
    assert res["quality_flag"] in ("ok", "review")
