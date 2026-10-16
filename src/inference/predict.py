"""CT volume in, spleen mask and volume out."""
import argparse
import json
import time

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy import ndimage

from src import config as C
from src.data.transforms import infer_transforms
from src.models.unet import load_model

_TRANSFORMS = None


def _transforms():
    global _TRANSFORMS
    if _TRANSFORMS is None:
        _TRANSFORMS = infer_transforms()
    return _TRANSFORMS


def postprocess(prob_spleen: np.ndarray):
    raw = prob_spleen > 0.5
    labelled, n = ndimage.label(raw, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.zeros_like(raw, dtype=np.uint8), dict(n_components=0, second_ratio=0.0)

    sizes = np.bincount(labelled.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    mask = (labelled == order[0] + 1).astype(np.uint8)
    second_ratio = float(sizes[order[1]] / sizes[order[0]]) if n > 1 else 0.0
    return mask, dict(n_components=int(n), second_ratio=second_ratio)


def quality_check(mask: np.ndarray, prob_spleen: np.ndarray, volume_ml: float, info: dict):
    reasons = []
    if volume_ml < C.MIN_VOLUME_ML:
        reasons.append(f"volume {volume_ml:.1f} mL below {C.MIN_VOLUME_ML:.0f} mL")
    if volume_ml > C.MAX_VOLUME_ML:
        reasons.append(f"volume {volume_ml:.1f} mL above {C.MAX_VOLUME_ML:.0f} mL")
    if mask.any():
        conf = float(prob_spleen[mask > 0].mean())
        if conf < C.MIN_MEAN_CONFIDENCE:
            reasons.append(f"mean confidence {conf:.2f} below {C.MIN_MEAN_CONFIDENCE}")

        x_frac = float(np.argwhere(mask)[:, 0].mean() / mask.shape[0])
        if x_frac > 0.5:
            reasons.append(f"mask centroid on patient's right side (x={x_frac:.2f})")
    if info.get("second_ratio", 0.0) > C.MAX_SECOND_COMPONENT_RATIO:
        reasons.append(f"second candidate region is {info['second_ratio']:.0%} of the largest")
    return ("review" if reasons else "ok"), reasons


@torch.no_grad()
def predict(image_path, predictor, device="cpu", uncertainty=False, all_flips=False):
    t0 = time.time()
    x = _transforms()({"image": str(image_path)})["image"].unsqueeze(0).to(device)

    tta = None
    if uncertainty:
        from src.inference.uncertainty import FLIP_SETS_ALL, summarise, tta_probabilities

        views = tta_probabilities(x, predictor, FLIP_SETS_ALL if all_flips else None)
        prob = views[0]
    else:
        logits = sliding_window_inference(x, C.ROI_SIZE, C.SW_BATCH, predictor, overlap=C.SW_OVERLAP)
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

    mask, info = postprocess(prob)
    volume_ml = float(mask.sum() * C.VOXEL_ML)
    flag, reasons = quality_check(mask, prob, volume_ml, info)

    result = {
        "spleen_volume_ml": round(volume_ml, 1),
        "quality_flag": flag,
        "reasons": reasons,
        "n_components_before_lcc": info["n_components"],
        "voxel_spacing_mm": list(C.SPACING),
        "inference_time_s": round(time.time() - t0, 2),
        "_image": x[0, 0].cpu().numpy(),
        "_mask": mask,
        "_prob": prob,
    }

    if uncertainty:
        tta, mean_prob, std_prob = summarise(views, mask)
        result["uncertainty"] = tta
        result["_prob"] = mean_prob
        result["_std"] = std_prob
        if tta["volume_cv_pct"] is not None and tta["volume_cv_pct"] > 10:
            result["reasons"].append(f"volume varies {tta['volume_cv_pct']:.0f}% across test-time flips")
            result["quality_flag"] = "review"
    return result


def public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def _slices(mask, img):
    z = int(mask.sum(axis=(0, 1)).argmax()) if mask.any() else img.shape[2] // 2
    y = int(mask.sum(axis=(0, 2)).argmax()) if mask.any() else img.shape[1] // 2
    return y, z


def _show(ax, im, title):
    ax.imshow(np.fliplr(np.rot90(im)), cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_overlay(result: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, m = result["_image"], result["_mask"]
    y, z = _slices(m, img)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a, im, mk, title in [(ax[0], img[:, :, z], m[:, :, z], f"axial z={z}"),
                             (ax[1], img[:, y, :], m[:, y, :], "coronal")]:
        _show(a, im, title)
        a.imshow(np.fliplr(np.rot90(np.ma.masked_where(mk == 0, mk))), cmap="autumn", alpha=0.5)
        a.text(0.02, 0.5, "R", transform=a.transAxes, color="yellow", fontsize=12, va="center")
        a.text(0.95, 0.5, "L", transform=a.transAxes, color="yellow", fontsize=12, va="center")
    fig.suptitle(f"Spleen {result['spleen_volume_ml']} mL  [{result['quality_flag']}]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_heatmap(result: dict, path):
    """CT, spleen probability and TTA disagreement on the same slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, m, prob = result["_image"], result["_mask"], result["_prob"]
    std = result.get("_std")
    y, z = _slices(m, img)
    panels = 2 if std is None else 3
    fig, ax = plt.subplots(1, panels, figsize=(5 * panels, 5))

    _show(ax[0], img[:, :, z], f"CT, axial z={z}")

    _show(ax[1], img[:, :, z], "spleen probability")
    p = ax[1].imshow(np.fliplr(np.rot90(np.ma.masked_where(prob[:, :, z] < 0.05, prob[:, :, z]))),
                     cmap="inferno", alpha=0.65, vmin=0, vmax=1)
    fig.colorbar(p, ax=ax[1], fraction=0.046)

    if std is not None:
        _show(ax[2], img[:, :, z], "TTA disagreement (std)")
        s = ax[2].imshow(np.fliplr(np.rot90(np.ma.masked_where(std[:, :, z] < 0.01, std[:, :, z]))),
                         cmap="viridis", alpha=0.75, vmin=0, vmax=0.5)
        fig.colorbar(s, ax=ax[2], fraction=0.046)

    fig.suptitle(f"Spleen {result['spleen_volume_ml']} mL  [{result['quality_flag']}]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Spleen segmentation + volume from a CT NIfTI file")
    ap.add_argument("image", help="CT volume, .nii or .nii.gz")
    ap.add_argument("--model", default=str(C.DEFAULT_MODEL), help=".pth or .onnx")
    ap.add_argument("--overlay", help="PNG path for an axial+coronal mask overlay")
    ap.add_argument("--heatmap", help="PNG path for the probability (and uncertainty) maps")
    ap.add_argument("--save-mask", help=".nii.gz path for the mask, in the 1.5x1.5x2 mm space")
    ap.add_argument("--uncertainty", action="store_true",
                    help="4-view flip TTA and the resulting volume spread (4x slower)")
    ap.add_argument("--all-flips", action="store_true",
                    help="use all 8 flips, including the anatomically invalid left-right ones")
    ap.add_argument("--log-run", action="store_true", help="write a JSON run record under results/runs/")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = load_model(args.model, device)
    res = predict(args.image, predictor, device,
                  uncertainty=args.uncertainty or args.all_flips, all_flips=args.all_flips)

    if args.overlay:
        save_overlay(res, args.overlay)
    if args.heatmap:
        save_heatmap(res, args.heatmap)
    if args.save_mask:
        import nibabel as nib
        nib.save(nib.Nifti1Image(res["_mask"], np.diag(list(C.SPACING) + [1.0])), args.save_mask)
    if args.log_run:
        from src.utils.provenance import write_run_record
        path = write_run_record(public(res), args.model, device)
        print(f"run record: {path}")

    print(json.dumps(public(res), indent=2))


if __name__ == "__main__":
    main()
