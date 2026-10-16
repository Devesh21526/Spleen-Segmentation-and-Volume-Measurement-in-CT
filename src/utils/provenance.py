"""Which weights, which constants, which library versions produced a given number."""
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from src import config as C

_CACHE = {}


def file_sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def model_fingerprint(path) -> dict:
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns if path.exists() else 0)
    if key not in _CACHE:
        _CACHE[key] = {
            "name": path.name,
            "sha256": file_sha256(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return _CACHE[key]


def config_hash() -> str:
    payload = json.dumps({
        "hu": [C.HU_MIN, C.HU_MAX],
        "spacing": list(C.SPACING),
        "roi": list(C.ROI_SIZE),
        "overlap": C.SW_OVERLAP,
        "unet": {k: list(v) if isinstance(v, tuple) else v for k, v in C.UNET_KWARGS.items()},
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(C.ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            dirty = subprocess.run(["git", "-C", str(C.ROOT), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=5).stdout.strip()
            return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        pass
    return None


def versions() -> dict:
    import numpy
    import torch

    v = {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__}
    for name in ("monai", "nibabel", "scipy", "onnxruntime"):
        try:
            v[name] = __import__(name).__version__
        except Exception:
            v[name] = None
    return v


def provenance(model_path, device="cpu") -> dict:
    return {
        "model": model_fingerprint(model_path),
        "config_hash": config_hash(),
        "git_commit": git_commit(),
        "device": device,
        "versions": versions(),
        "platform": f"{platform.system()} {platform.machine()}",
    }


def write_run_record(result: dict, model_path, device="cpu", out_dir=None) -> Path:
    out_dir = Path(out_dir or os.environ.get("RUN_LOG_DIR", C.ROOT / "results" / "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": result,
        "provenance": provenance(model_path, device),
    }
    path = out_dir / f"run_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{os.getpid()}.json"
    path.write_text(json.dumps(record, indent=2))
    return path
