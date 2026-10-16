"""Regenerate the patient-level split the notebook used.

The split lives in one line of the notebook, `random.Random(42).shuffle(data)`, so it
can be rebuilt exactly instead of being carried around as a file. KNOWN_TEST is what
that shuffle produced on the real dataset; `make_splits` checks against it, which turns
this from a claim into a test.

    python -m src.utils.splits --data data/Task09_Spleen
"""
import argparse
import glob
import json
import os
import random
from pathlib import Path

SEED = 42
N_TEST = N_VAL = 6
KNOWN_TEST = ["spleen_20.nii.gz", "spleen_14.nii.gz", "spleen_21.nii.gz",
              "spleen_61.nii.gz", "spleen_44.nii.gz", "spleen_41.nii.gz"]


def make_splits(data_dir, seed=SEED):
    data_dir = Path(data_dir)
    images = sorted(glob.glob(str(data_dir / "imagesTr" / "*.nii.gz")))
    labels = sorted(glob.glob(str(data_dir / "labelsTr" / "*.nii.gz")))
    if not images or len(images) != len(labels):
        raise FileNotFoundError(f"{len(images)} images and {len(labels)} labels under {data_dir}")

    data = [{"image": i, "label": l} for i, l in zip(images, labels)]
    random.Random(seed).shuffle(data)
    return {"test": data[:N_TEST], "val": data[N_TEST:N_TEST + N_VAL], "train": data[N_TEST + N_VAL:]}


def test_case_names(splits):
    return [os.path.basename(f["image"]) for f in splits["test"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Task09_Spleen")
    ap.add_argument("--out", default="results/splits.json")
    args = ap.parse_args()

    splits = make_splits(args.data)
    names = test_case_names(splits)
    if names != KNOWN_TEST:
        raise SystemExit(f"split does not match the one used for the reported numbers\n"
                         f"  expected {KNOWN_TEST}\n  got      {names}")

    counts = {k: len(v) for k, v in splits.items()}
    Path(args.out).write_text(json.dumps({"seed": SEED, "counts": counts, **splits}, indent=2))
    print(f"{args.out}: train {counts['train']} | val {counts['val']} | test {counts['test']}")
    print("test cases match the reported run")


if __name__ == "__main__":
    main()
