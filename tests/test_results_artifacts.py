import json

import pytest

from src.utils.history import STAGE1, STAGE2, parse
from src.utils.splits import KNOWN_TEST, make_splits
from src.utils.splits import test_case_names as case_names


@pytest.fixture
def fake_dataset(tmp_path):
    for i in range(1, 42):
        (tmp_path / "imagesTr").mkdir(exist_ok=True)
        (tmp_path / "labelsTr").mkdir(exist_ok=True)
        (tmp_path / "imagesTr" / f"spleen_{i}.nii.gz").touch()
        (tmp_path / "labelsTr" / f"spleen_{i}.nii.gz").touch()
    return tmp_path


def test_split_sizes_and_disjointness(fake_dataset):
    s = make_splits(fake_dataset)
    assert [len(s["train"]), len(s["val"]), len(s["test"])] == [29, 6, 6]
    names = [f["image"] for part in s.values() for f in part]
    assert len(set(names)) == 41


def test_split_is_deterministic(fake_dataset):
    assert case_names(make_splits(fake_dataset)) == case_names(make_splits(fake_dataset))
    assert case_names(make_splits(fake_dataset, seed=1)) != case_names(make_splits(fake_dataset))


def test_images_and_labels_stay_paired(fake_dataset):
    for part in make_splits(fake_dataset).values():
        for f in part:
            assert f["image"].split("/")[-1] == f["label"].split("/")[-1]


def test_recorded_split_matches_the_reported_test_cases():
    recorded = json.loads(open("results/splits.json").read())
    assert recorded["test"] == KNOWN_TEST
    assert recorded["counts"] == {"train": 29, "val": 6, "test": 6}


def test_history_log_parsing():
    text = "Epoch   5 | loss 1.3443 | val Dice 0.0188 | best 0.0188\nFT  50 | loss 0.2621 | val Dice 0.9005"
    assert parse(text, STAGE1) == [{"epoch": 5, "loss": 1.3443, "val_dice": 0.0188}]
    assert parse(text, STAGE2) == [{"epoch": 50, "loss": 0.2621, "val_dice": 0.9005}]


def test_saved_history_matches_the_reported_best():
    h1 = json.loads(open("results/history.json").read())
    h2 = json.loads(open("results/history_ft.json").read())
    assert h1["best"]["val_dice"] == pytest.approx(0.859, abs=0.001)
    assert h2["best"]["val_dice"] == pytest.approx(0.9005, abs=0.001)
    assert h2["best"]["epoch"] == 50
