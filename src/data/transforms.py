"""Pre-processing pipelines, identical in training and inference."""
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
)

from src.config import HU_MAX, HU_MIN, ROI_SIZE, SPACING


def base_transforms(keys, modes):
    return [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=SPACING, mode=modes),
        ScaleIntensityRanged(keys=["image"], a_min=HU_MIN, a_max=HU_MAX,
                             b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
    ]


def train_transforms():
    keys = ["image", "label"]
    return Compose(base_transforms(keys, ("bilinear", "nearest")) + [
        RandCropByPosNegLabeld(keys=keys, label_key="label", spatial_size=ROI_SIZE,
                               pos=1, neg=1, num_samples=4,
                               image_key="image", image_threshold=0),
        RandFlipd(keys=keys, spatial_axis=[0], prob=0.10),
        RandFlipd(keys=keys, spatial_axis=[1], prob=0.10),
        RandFlipd(keys=keys, spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=keys, prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
        EnsureTyped(keys=keys),
    ])


def eval_transforms():
    keys = ["image", "label"]
    return Compose(base_transforms(keys, ("bilinear", "nearest")) + [EnsureTyped(keys=keys)])


def infer_transforms():
    return Compose(base_transforms(["image"], ("bilinear",)) + [EnsureTyped(keys=["image"])])
