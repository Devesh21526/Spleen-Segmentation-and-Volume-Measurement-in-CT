# Spleen Segmentation and Volume Measurement in CT

**mPragati assessment** · Devesh · October 2026
Code in `spleen-seg/`, setup in the README. Longer version: `docs/report_full.md`

---

## What it does

Give it a CT scan, get back the spleen outlined, its volume in mL, and a flag saying whether the answer looks sane.

- Doctors care about spleen size in liver disease, infections and some blood cancers. Measuring the real volume by hand means outlining the organ on every slice, so in practice they just measure its length instead [10].
- Meant for a radiologist or researcher who wants the number. Something a person checks, not a diagnosis.
- In: a CT file (`.nii` / `.nii.gz`), any resolution. Out: mask, volume, and `ok` or `review` with reasons.
- Goals set before testing: Dice ≥ 0.85, volume error ≤ 10 %, nothing predicted outside the spleen, under 30 s a scan.
- Simple rules do not work here: the spleen is almost the same brightness as the liver and stomach next to it, so any threshold bleeds into them. A network learns the shape instead [1, 2, 4].
- Assumes contrast CT with the whole spleen in frame. Trained on one hospital's scans. No clinical testing, so no claims about real diagnostic use.

## Data

MSD Task09 [6]: 61 scans from one US hospital, 41 with expert outlines.

- Scans differ a lot: pixel size 0.8–0.98 mm, slice thickness 1.5–8 mm, 31–168 slices.
- Spleen volume 56–502 mL. It fills only 0.08–1.33 % of a scan, so most of the image is background.
- Split 29 train / 6 validation / 6 test, seed 42. One scan is one patient, so nobody appears twice.
- Validation drove every decision. The test set was opened once, at the end.
- The expert outlines look blocky on thick-slice scans after resampling, so some error below is the labels, not the model.

| Step | Setting | Why |
|---|---|---|
| Orientation | RAS | same axes on every scanner |
| Resampling | 1.5 × 1.5 × 2.0 mm | removes the 1.5–8 mm thickness spread |
| Brightness | keep −57 to 164 HU, scale to 0–1 | the soft-tissue range; the full range would wash out organ contrast |
| Crop | cut the air around the body | memory and time |
| Augmentation | flips, 90° rotations, small brightness shifts | only 29 training scans |

Three things keep the background from drowning the spleen: patches are sampled half on the organ, the loss uses Dice, and Dice is scored on the spleen only.

## Model and training

MONAI 3D U-Net, 4.8 M parameters, trained from scratch. It reads the scan in 96³ blocks with 25 % overlap, then keeps the largest connected piece, since there is only one spleen.

- No transformer. nnU-Net Revisited [5] retested those models against well-tuned U-Nets and most of the claimed gains vanished. A CNN is also easier to run on small hardware.
- Loss is Dice plus cross-entropy. Dice ignores the huge background [3]; cross-entropy keeps training moving at the start, when the prediction is still blank.

| Stage | Setting | Best validation Dice |
|---|---|---|
| 1 | 200 epochs, AdamW, lr 1e-4 | 0.859 at epoch 120 |
| 2 | 150 more, cosine lr 1e-4 → 0 | 0.901 at epoch 50 |

Stage 1 jumped around between 0.54 and 0.86, because one bad scan moves an average of six. Stage 2 settled it (Fig. 2). Everything is repeatable: fixed seed, saved split, one settings file, and the API prints the checkpoint hash so any number can be traced back.

## Results

Three measures, since one is never enough [8, 15]: Dice and IoU for overlap, HD95 for how far off the edge is in mm, and volume error, which is the actual output.

| Case | Dice | IoU | HD95 (mm) | Pred (mL) | True (mL) | Error |
|---|---|---|---|---|---|---|
| spleen_20 | 0.938 | 0.883 | 4.74 | 306.3 | 289.0 | +6.0 % |
| spleen_14 | 0.957 | 0.917 | 2.50 | 451.0 | 450.1 | +0.2 % |
| spleen_21 | 0.889 | 0.800 | 12.26 | 149.2 | 133.3 | +11.9 % |
| spleen_61 | 0.939 | 0.886 | 2.50 | 158.7 | 163.0 | −2.7 % |
| spleen_44 | 0.798 | 0.664 | 12.00 | 86.7 | 119.4 | −27.4 % |
| spleen_41 | 0.928 | 0.866 | 3.00 | 179.1 | 189.9 | −5.7 % |
| **Mean ± SD** | **0.908 ± 0.053** | 0.836 ± 0.085 | 6.17 ± 4.28 | | | −2.9 ± 12.4 % |

Both goals met, but six scans is a small sample.

What went wrong (Fig. 3):

- **spleen_44**, the worst at 0.798. The mask is too small everywhere, missing 8,254 voxels and adding 992. Faint contrast, thick slices, and a blocky reference outline. This is the −27 % case.
- **spleen_21**, 0.889. The mask runs out of the spleen into nearby vessels of the same brightness. Keeping the largest piece does not help, because the leak is attached.
- **spleen_14**, 0.957. Only a thin edge error, and it cancels out, so the volume is off by 0.2 %. It is also the biggest spleen, where the same edge error hurts Dice less.

## On another hospital's scans

Tested on 25 scans from TotalSegmentator [9] — different country, different scanners, different contrast timing.

| | Same hospital (n = 6) | Different hospital (n = 25) |
|---|---|---|
| Dice mean ± SD | 0.908 ± 0.053 | 0.548 ± 0.431 |
| Dice median | 0.933 | 0.837 |
| Bad failures (Dice < 0.5) | 0 / 6 | 10 / 25 |

- It is not a gentle drop. 15 scans work as well as at home; 10 fail outright.
- In 5 failures the model found part of the spleen but also a bigger wrong region, and the "keep the largest piece" rule picked the wrong one.
- Brightness does not explain it: failed scans averaged 105 HU in the spleen, working ones 90, and one spleen at 45 HU scored 0.92.
- So the cause is false regions on unfamiliar scans, and the cleanup rule then makes a bad call. The same pattern shows up in other single-hospital models [11].

The fix was more data, not a different model: 80 epochs on the 29 original scans plus 44 from the second hospital, keeping those 25 test scans aside. 72 minutes on a T4.

| Model | Dice (home) | Volume error (home) | Dice (other) | Failures | Volume error (other) |
|---|---|---|---|---|---|
| First version | 0.908 ± 0.053 | 5.8 % | 0.548 ± 0.431 | 10 / 25 | 15.3 % |
| **With second hospital** | **0.924 ± 0.043** | **2.3 %** | **0.895 ± 0.106** | **1 / 25** | **4.6 %** |

44 extra scans took failures from 10 to 1, and nothing got worse at home. Caveats: those 25 scans were looked at while debugging, so they are not untouched; it is still only one extra hospital; one case still fails.

## How much to trust a number

Run the same scan a few times with the image flipped, compare the answers.

- Flipping left to right was a bad idea. It swings the volume by 44 % on `spleen_14`. That flip puts the spleen on the wrong side of the body, which never happens, so the model has never seen it. Using only sensible flips, the volume moves 4.8 %. **That 44 % is a real weakness worth reporting: a mirrored scan breaks this model.**
- Agreeing is not the same as being right. On `spleen_1`, where an older model grabbed the wrong organ, the flipped runs agreed within 3.5 %. The anatomy checks are there for that case, and on the current model they stay quiet on that scan while the older one tripped them.
- `--heatmap` shows the scan, how sure the model is at each voxel, and where the runs disagree. On a good case that disagreement is a one or two voxel line along the edge.

## Pipeline and software

```
CT file → orient, resample, brightness window, crop → 3D U-Net (96³ blocks)
       → threshold → keep largest piece → volume = voxels × 4.5 mm³
       → sanity checks → JSON
```

- Command line: `python -m src.inference.predict ct.nii.gz --overlay out.png`. API: `POST /predict`, add `?uncertainty=true` for the flip test. `GET /health` says which model is loaded.
- Rejects bad input: wrong file type 415, empty 400, over 500 MB 413, unreadable or not 3D 422.
- Says `review` when the volume is outside 30–2000 mL, the mask is on the wrong side of the body, the model is under 85 % sure, a second region is more than 30 % of the kept one, or the flip test moves the volume over 10 %. All five come from anatomy, none from the test data.
- Takes 1.8–6.1 s a scan on a laptop, 1.4–3.9 s on a T4. The flip check costs 4×.
- Each run can log a JSON record with the model hash, settings hash, commit and library versions, so an old number can be reproduced.
- 35 tests, run on random weights so CI needs no checkpoint.

## Running it on small hardware

- ONNX export matches PyTorch to 5e-6 and gives the same volume. It is slower on this laptop CPU, so the point is portability, not speed.
- `qnn_prepare.py` rewrites the exported model into the ops that NPU converters accept — swapping one activation for an equivalent one, folding normalisation into the layer before it, fixing the batch size at 1 — and checks the output still matches. `spleen_14` still comes out at 444.9 mL.
- 8-bit compression shrinks the file from 19.3 MB to 7.3 MB but will not run here: ONNX Runtime's CPU backend has no 8-bit kernel for 3D convolution. That is a gap in this backend, and the sort of thing an NPU exists to do.
- On a real board, check in this order: does the chip support 3D convolution at all, are the input shapes fixed (ours are), calibrate on real CT patches and re-measure Dice, and time the preprocessing, which can cost more than the network. Commands are in `docs/qnn_deployment.md`.
- For real use it would also need more hospitals, a comparison against radiologist measurements, and PACS integration [12, 13].

## Limits

- Two hospitals only, and one case still fails.
- Six test scans, so the headline Dice is uncertain.
- Volume is counted on the resampled grid, and thick-slice references are coarse anyway.
- Keeping the largest piece can throw away the real spleen when a bigger wrong region exists.
- The sanity checks are hand-written rules, tuned against an older failure, and never re-tuned.
- A mirrored scan breaks it.
- The second hospital's outlines were machine-drawn and expert-corrected, not drawn from scratch.
- The NPU path is prepared but never measured on a board.

## What I wrote and what I used

Mine: the framing, the split, the preprocessing choices, the training schedule, the evaluation and failure analysis, the second-hospital fine-tune, the inference pipeline, the sanity checks, the flip test, the logging, the API, the tests, Docker, the ONNX export and the NPU preparation.

Used: MONAI (transforms, U-Net, loss, metrics, sliding-window inference), PyTorch, FastAPI, nibabel, SciPy, ONNX Runtime, Colab. Data from MSD Task09 and TotalSegmentator. An AI assistant helped write code; I can explain all of it.

## Which paper decided what

| Source | What it settled |
|---|---|
| U-Net [1], 3D U-Net [2] | Encoder–decoder with skip connections, in 3D because the organ spans slices. Costs memory, hence 96³ blocks. |
| V-Net [3] | Dice loss when the target is 1 % of the image. |
| nnU-Net [4, 5] | Settings matter more than new architectures — and a CNN over a transformer. |
| MSD [6], TotalSegmentator [9] | The dataset, and the second hospital that fixed generalisation. |
| MONAI [7] | Transforms, network, loss, metrics. |
| Metrics papers [8, 15] | Report overlap, boundary and the clinical number together. |
| Humpire-Mamani [10] | Why volume instead of a length measurement. |
| CLAIM [14, 16] | What this report needs to contain. |

## References

1. Ronneberger O. et al. U-Net. MICCAI 2015.
2. Çiçek Ö. et al. 3D U-Net. MICCAI 2016.
3. Milletari F. et al. V-Net. 3DV 2016.
4. Isensee F. et al. nnU-Net. Nature Methods 2021.
5. Isensee F. et al. nnU-Net Revisited. MICCAI 2024.
6. Antonelli M. et al. The Medical Segmentation Decathlon. Nature Communications 2022.
7. Cardoso M. J. et al. MONAI. arXiv 2022.
8. Maier-Hein L. et al. Metrics Reloaded. Nature Methods 2024.
9. Wasserthal J. et al. TotalSegmentator. Radiology: AI 2023.
10. Humpire-Mamani G. E. et al. Spleen volume at CT using deep learning. Radiology: AI 2020.
11. Ma J. et al. AbdomenCT-1K. IEEE TPAMI 2021.
12. Wang G. et al. Uncertainty estimation with test-time augmentation. Neurocomputing 2019.
13. Kelly C. J. et al. Key challenges for clinical impact with AI. BMC Medicine 2019.
14. Mongan J. et al. CLAIM checklist. Radiology: AI 2020.
15. Taha A. A., Hanbury A. Metrics for 3D medical image segmentation. BMC Medical Imaging 2015.
16. Tejani A. S. et al. CLAIM 2024 update. Radiology: AI 2024.

**Figures** in `results/`: dataset stats and an example scan (Fig. 1), preprocessing before/after, training curves (Fig. 2), the three error cases (Fig. 3), a full pipeline output, the wrong-organ case, and the flip-disagreement maps. Training history and the split are saved there too.
