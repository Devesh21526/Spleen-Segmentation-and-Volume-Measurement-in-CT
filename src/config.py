"""Constants shared by the training notebook and the inference code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "best_model_multicenter.pth"
BASELINE_MODEL = ROOT / "models" / "best_model_ft.pth"

HU_MIN, HU_MAX = -57, 164
SPACING = (1.5, 1.5, 2.0)
ROI_SIZE = (96, 96, 96)
SW_BATCH = int(os.environ.get("SW_BATCH", 4))
SW_OVERLAP = 0.25

UNET_KWARGS = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)

VOXEL_ML = SPACING[0] * SPACING[1] * SPACING[2] / 1000.0

MIN_VOLUME_ML = 30.0
MAX_VOLUME_ML = 2000.0
MIN_MEAN_CONFIDENCE = 0.85
MAX_SECOND_COMPONENT_RATIO = 0.3

MAX_UPLOAD_MB = 500
ALLOWED_SUFFIXES = (".nii", ".nii.gz")
