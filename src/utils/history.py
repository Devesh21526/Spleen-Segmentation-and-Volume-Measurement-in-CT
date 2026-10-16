"""Pull the training history out of the notebook's own log output.

The JSON the notebook wrote went to Drive. The printed log survived in the notebook, and
it carries the same loss and validation Dice every five epochs, which is what the curves
were drawn from.

    python -m src.utils.history
"""
import argparse
import json
import re
from pathlib import Path

STAGE1 = re.compile(r"Epoch\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+val Dice\s+([\d.]+)")
STAGE2 = re.compile(r"FT\s+(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+val Dice\s+([\d.]+)")


def notebook_text(path):
    nb = json.loads(Path(path).read_text())
    chunks = []
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            chunks.append("".join(out.get("text", "")))
            chunks.append("".join(out.get("data", {}).get("text/plain", "")))
    return "\n".join(c for c in chunks if c)


def parse(text, pattern):
    return [{"epoch": int(e), "loss": float(l), "val_dice": float(d)}
            for e, l, d in pattern.findall(text)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="notebooks/train_spleen.ipynb")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    text = notebook_text(args.notebook)
    out = Path(args.out_dir)
    for name, pattern, note in [
        ("history.json", STAGE1, "stage 1, constant lr 1e-4"),
        ("history_ft.json", STAGE2, "stage 2, cosine lr, validation Dice after largest-component"),
    ]:
        points = parse(text, pattern)
        if not points:
            print(f"no {name} entries found in the log")
            continue
        best = max(points, key=lambda p: p["val_dice"])
        (out / name).write_text(json.dumps({
            "source": "parsed from the notebook log, logged every 5 epochs",
            "note": note,
            "best": best,
            "points": points,
        }, indent=2))
        print(f"{name}: {len(points)} points | best val Dice {best['val_dice']:.4f} at epoch {best['epoch']}")


if __name__ == "__main__":
    main()
