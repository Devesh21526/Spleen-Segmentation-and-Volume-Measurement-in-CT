import gzip
import io

import nibabel as nib
import numpy as np


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_health_reports_provenance(api_client):
    p = api_client.get("/health").json()["provenance"]
    assert p["model"]["sha256"] and p["config_hash"]
    assert p["versions"]["torch"]


def test_predict_with_uncertainty(api_client, ct_file):
    with open(ct_file, "rb") as f:
        r = api_client.post("/predict?uncertainty=true",
                            files={"file": ("ct.nii.gz", f, "application/gzip")})
    assert r.status_code == 200, r.text
    u = r.json()["uncertainty"]
    assert u["n_views"] == 4 and u["volume_max_ml"] >= u["volume_min_ml"]


def test_predict_valid_nifti(api_client, ct_file):
    with open(ct_file, "rb") as f:
        r = api_client.post("/predict", files={"file": ("ct.nii.gz", f, "application/gzip")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["spleen_volume_ml"], float)
    assert body["quality_flag"] in ("ok", "review")
    assert body["input_shape"] == [80, 80, 48]


def test_rejects_wrong_extension(api_client):
    r = api_client.post("/predict", files={"file": ("scan.png", b"\x89PNG", "image/png")})
    assert r.status_code == 415


def test_rejects_empty_file(api_client):
    r = api_client.post("/predict", files={"file": ("ct.nii.gz", b"", "application/gzip")})
    assert r.status_code == 400


def test_rejects_corrupt_nifti(api_client):
    junk = gzip.compress(b"not a nifti file at all")
    r = api_client.post("/predict", files={"file": ("ct.nii.gz", junk, "application/gzip")})
    assert r.status_code == 422


def test_rejects_2d_image(api_client):
    buf = io.BytesIO()
    img = nib.Nifti1Image(np.zeros((64, 64), np.float32), np.eye(4))
    buf.write(img.to_bytes())
    r = api_client.post("/predict", files={"file": ("slice.nii", buf.getvalue(), "application/octet-stream")})
    assert r.status_code == 422
