"""FastAPI service.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000
    curl -F "file=@spleen_11.nii.gz" http://localhost:8000/predict
"""
import os
import tempfile
from contextlib import asynccontextmanager

import nibabel as nib
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from src import config as C
from src.inference.predict import predict, public
from src.models.unet import load_model
from src.utils.provenance import provenance

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["model_path"] = os.environ.get("MODEL_PATH", str(C.DEFAULT_MODEL))
    STATE["predictor"] = load_model(STATE["model_path"], STATE["device"])
    yield
    STATE.clear()

app = FastAPI(title="Spleen CT segmentation", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(STATE.get("model_path", "")),
            "device": STATE.get("device"),
            "provenance": provenance(STATE.get("model_path", ""), STATE.get("device", "cpu"))}


def _suffix(name: str) -> str:
    name = name.lower()
    for s in C.ALLOWED_SUFFIXES[::-1]:
        if name.endswith(s):
            return s
    return ""


@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    uncertainty: bool = Query(False, description="run 4-view flip TTA; roughly 4x slower"),
):
    suffix = _suffix(file.filename or "")
    if not suffix:
        raise HTTPException(415, f"Expected a NIfTI file ({', '.join(C.ALLOWED_SUFFIXES)}), got '{file.filename}'")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {C.MAX_UPLOAD_MB} MB")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        try:
            img = nib.load(path)
            shape = img.shape
        except Exception as e:
            raise HTTPException(422, f"Could not read NIfTI: {e}")
        if len(shape) != 3:
            raise HTTPException(422, f"Expected a 3D CT volume, got shape {tuple(shape)}")
        if min(shape) < 16:
            raise HTTPException(422, f"Volume too small: {tuple(shape)}")

        raw = predict(path, STATE["predictor"], STATE["device"], uncertainty=uncertainty)
        result = public(raw)
        result["input_shape"] = list(shape)
        result["input_spacing_mm"] = [round(float(s), 3) for s in img.header.get_zooms()[:3]]
        result["model"] = os.path.basename(STATE["model_path"])
        if os.environ.get("RUN_LOG_DIR"):
            from src.utils.provenance import write_run_record
            write_run_record(result, STATE["model_path"], STATE["device"])
        return result
    finally:
        os.unlink(path)
