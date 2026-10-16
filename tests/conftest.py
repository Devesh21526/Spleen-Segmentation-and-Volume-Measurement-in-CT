import nibabel as nib
import numpy as np
import pytest
import torch

from src.models.unet import build_unet


@pytest.fixture(scope="session")
def dummy_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("m") / "random.pth"
    torch.manual_seed(0)
    torch.save(build_unet().state_dict(), path)
    return path


@pytest.fixture(scope="session")
def ct_file(tmp_path_factory):
    vol = np.full((80, 80, 48), -1000, np.float32)
    vol[10:70, 10:70, 4:44] = 40
    vol[15:30, 30:50, 15:30] = 120
    path = tmp_path_factory.mktemp("ct") / "ct.nii.gz"
    nib.save(nib.Nifti1Image(vol, np.diag([1.5, 1.5, 2.0, 1.0])), path)
    return path


@pytest.fixture
def api_client(dummy_model, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_PATH", str(dummy_model))
    from src.api.main import app

    with TestClient(app) as c:
        yield c
