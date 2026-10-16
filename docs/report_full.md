# Automated Spleen Segmentation and Volumetry in Abdominal CT

**Technical report: mPragati AI / Software Engineer assessment**
Devesh · October 2026 · Code: `spleen-seg/` (README for setup)

---

## 1. Problem Definition

**Clinical context.** Spleen enlargement (splenomegaly) accompanies liver cirrhosis and portal hypertension, infections, and haematological cancers. In routine practice spleen size is usually judged from a single linear measurement (craniocaudal length), because outlining the organ slice by slice takes too long. The spleen's irregular shape makes a 1D measure a weak proxy for volume [10]. Automated 3D segmentation gives a reproducible volume in seconds.

**Intended user.** A radiologist or radiology resident who wants a volume estimate to accompany a report, or a researcher who needs spleen volumes across a cohort. The output is decision support for review, not an autonomous diagnosis.

**Input / output.**
- Input: one abdominal CT volume in NIfTI format (`.nii` / `.nii.gz`), any in-plane resolution and slice thickness.
- Output: a binary spleen mask, the spleen volume in mL, and a quality flag (`ok` / `review`) with reasons.

**Success criteria (set before testing).** On held-out data from the training distribution: mean Dice ≥ 0.85, median absolute volume error ≤ 10 %, and no case where the predicted region lies outside the spleen. Inference under 30 s per scan on a single GPU or CPU.

**Why AI.** The spleen has soft-tissue intensity close to that of the liver, stomach, and pancreatic tail, so thresholding or region growing leaks into neighbours. Its shape and position vary between patients. A 3D convolutional network learns shape and context jointly from labelled examples, and the U-Net family is the established baseline for this task [1, 2, 4].

**Assumptions.** Contrast-enhanced abdominal CT, with the whole spleen inside the field of view. A single spleen (no splenectomy, no accessory spleens).

**Limitations stated up front.** Training data come from a single centre with a single contrast phase. There is no clinical validation and no reader study. Nothing here supports a claim of diagnostic performance.

## 2. Dataset and Data Strategy

**Source.** Medical Segmentation Decathlon, Task09 Spleen [6], licence CC-BY-SA 4.0. It contains 61 portal-venous CT scans from Memorial Sloan Kettering Cancer Center, acquired from patients receiving chemotherapy for liver metastases. 41 scans carry expert labels; the 20 in `imagesTs` have no public labels.

**Exploratory analysis** (Fig. 1). In-plane spacing is below 1 mm (e.g. 0.8–0.98 mm), and slice thickness ranges from 1.5 to 8 mm. Each scan has 31–168 slices. Spleen volume is 56–502 mL (mean 243 mL). The spleen occupies only 0.08–1.33 % of the voxels in a scan, which makes this a strongly class-imbalanced problem.

**Pre-processing** (identical in training, evaluation, and the API; all constants live in `src/config.py`):

| Step | Setting | Reason |
|---|---|---|
| Orientation | RAS | consistent axis meaning across scanners |
| Resampling | 1.5 × 1.5 × 2.0 mm (bilinear image, nearest-neighbour label) | removes the 1.5–8 mm thickness spread; nearest-neighbour keeps the labels binary |
| Intensity | clip HU to [−57, 164], scale to [0, 1] | a soft-tissue window, so organ contrast is not compressed by the air–bone range |
| Crop | remove air around the body | saves memory and time |

**Augmentation** (training only): random flips on the three axes (p = 0.1 each), random 90° rotations (p = 0.1), and an intensity shift of ±0.1 (p = 0.5). The training set has only 29 scans, so augmentation matters.

**Ground truth.** We use the MSD expert labels as provided. Thick-slice scans give blocky labels after resampling (see Section 4), so part of the measured error is annotation resolution rather than model error.

**Split and leakage control.** The 41 labelled scans were shuffled with seed 42 and split into 29 train, 6 validation, and 6 test scans. The split is saved in `splits.json`. Each MSD scan is a different patient, so this is a patient-level split. Model selection used only the validation set. The test set was evaluated once, after the final model had been chosen.

**Class imbalance.** We addressed it in three places:
1. Patch sampling: `RandCropByPosNegLabeld` draws four 96³ patches per scan at a 1:1 ratio of spleen-centred to background-centred, instead of uniform random crops that would mostly miss the spleen.
2. Loss: Dice + cross-entropy (Section 3).
3. Metric: Dice is computed on the spleen class only, because background Dice is always about 0.99.

## 3. Model and Training

**Architecture.** A MONAI 3D U-Net with five resolution levels (16–256 channels), two residual units per level, and batch normalisation. It has 4.8 M parameters and was trained from scratch. Inference uses a sliding window of 96³ with 25 % overlap.

**Why a CNN U-Net and not a transformer.** nnU-Net Revisited [5] shows that a correctly configured CNN U-Net matches or beats transformer and Mamba variants on 3D medical segmentation, and that many reported gains came from weak baselines. A CNN also needs less memory and runs faster on the edge hardware that a medical-device team would target. We chose the stronger-evidence, cheaper option.

**Loss.** DiceCE. The Dice term optimises overlap directly and is insensitive to the 98–99 % background [3]. The cross-entropy term gives smooth, stable gradients early in training, when the Dice term is noisy.

**Training schedule** (Colab, one T4 GPU):

| Stage | Setting | Best validation Dice |
|---|---|---|
| 1 | 200 epochs, AdamW lr 1e-4, wd 1e-5, batch 2 × 4 patches | 0.859 at epoch 120 (raw) |
| 2 | 150 epochs from the stage-1 best, cosine lr 1e-4 → 0 | 0.901 at epoch 50 (with LCC) |

Validation ran every 5 epochs, and we kept the checkpoint with the best validation Dice. Stage-1 validation Dice swung between 0.54 and 0.86 because a single difficult scan moves the mean of 6. The cosine schedule in stage 2 stabilised it to 0.75–0.90 (Fig. 2). Training loss plateaued at about 0.21.

**Post-processing.** We keep the largest 3D connected component (LCC). The spleen is a single organ, so any smaller detached region is treated as a false positive.

**Reproducibility.** We call `set_determinism(42)`, save the split to a file, and keep all hyper-parameters in one config file. Loss and Dice histories are saved as JSON. The repository runs end-to-end in Docker, and the local CPU result reproduces the notebook: spleen_14 gives 451.0 mL in both.

## 4. Evaluation

**Metrics.** Following Metrics Reloaded [8], we report one overlap metric, one boundary metric, and the clinical quantity:
- Dice / IoU for overlap.
- HD95 in mm for boundary error. It catches distant false positives that barely change Dice.
- Volume error in mL and %, because volume is the clinical output.

**Held-out MSD test set (n = 6).**

| Case | Dice | IoU | HD95 (mm) | Pred (mL) | GT (mL) | Error |
|---|---|---|---|---|---|---|
| spleen_20 | 0.938 | 0.883 | 4.74 | 306.3 | 289.0 | +6.0 % |
| spleen_14 | 0.957 | 0.917 | 2.50 | 451.0 | 450.1 | +0.2 % |
| spleen_21 | 0.889 | 0.800 | 12.26 | 149.2 | 133.3 | +11.9 % |
| spleen_61 | 0.939 | 0.886 | 2.50 | 158.7 | 163.0 | −2.7 % |
| spleen_44 | 0.798 | 0.664 | 12.00 | 86.7 | 119.4 | −27.4 % |
| spleen_41 | 0.928 | 0.866 | 3.00 | 179.1 | 189.9 | −5.7 % |
| **Mean ± SD** | **0.908 ± 0.053** | 0.836 ± 0.085 | 6.17 ± 4.28 | | | −2.9 ± 12.4 % |

The Dice and median volume-error criteria are met. With only six cases, the uncertainty on these numbers is wide. LCC changed none of the six internal test cases, because the model produced no detached false positives on in-distribution data.

**Error analysis** (Fig. 3). We found three patterns:
1. **spleen_44, Dice 0.798.** Systematic under-segmentation: 8,254 voxels were missed and only 992 were added. The model captured the core of the spleen but left a rim along the whole boundary and missed the upper and lower tips. This scan has low spleen–liver contrast and thick original slices, and its resampled ground truth is visibly stair-stepped. The −27 % volume error is the largest in the test set.
2. **spleen_21, Dice 0.889.** Leakage at the splenic hilum: 5,260 voxels were added, against 1,723 missed. The prediction extends medially into vessels and soft tissue of similar intensity. LCC cannot remove this region because it is connected to the spleen, and the result is a +11.9 % volume error.
3. **spleen_14, Dice 0.957.** Only thin boundary errors, with the misses and additions roughly balanced, so the volume error is just 0.2 %. This is also the largest spleen (450 mL), and the same absolute boundary error lowers Dice less on a bigger organ.

Errors of equal size cost smaller spleens more Dice. This is consistent with Metrics Reloaded's warning about Dice on small structures.

## 5. Generalisation Beyond the Training Centre

The test set above comes from the same hospital, the same contrast phase, and the same population as training. To see what happens outside that distribution, we used two additional checks.

**Unlabelled MSD scans (qualitative).** On `imagesTs/spleen_1` the model segmented anterior soft tissue (stomach or the left hepatic lobe) and missed the spleen. This was confirmed on four axial slices and a coronal view. The API flags this case `review`, because the second-largest candidate region is 86 % of the size of the kept region.

**External labelled data (quantitative).** We used 25 scans with a complete spleen from the TotalSegmentator small subset [9]. These come from University Hospital Basel, with mixed scanners and contrast phases, and carry AI-assisted labels with expert correction.

| | MSD test (n = 6) | TotalSegmentator (n = 25) |
|---|---|---|
| Dice mean ± SD | 0.908 ± 0.053 | 0.548 ± 0.431 |
| Dice median | 0.933 | 0.837 |
| Failures (Dice < 0.5) | 0 / 6 | 10 / 25 |

The external results are bimodal. On 15 of the 25 scans the model works as well as on the internal test set (Dice 0.83–0.95). On 10 it fails. We investigated the 10 failures:

| Pattern | Cases | Raw Dice → after LCC |
|---|---|---|
| Spleen partly found, but a larger false-positive region exists elsewhere | 5 | 0.15–0.45 → 0 |
| Spleen not found | 3 | ≈ 0 → 0 |
| Not found; very low spleen attenuation (21 HU, likely non-contrast) | 1 | 0.10 → 0 |
| Partial | 2 | 0.37–0.57 → 0.31–0.59 |

The failures did not correlate with mean spleen HU (failed 105 vs. successful 90) or with scan length (584 vs. 634 mm). One spleen at 45 HU was segmented with Dice 0.92. **The main mechanism is out-of-distribution false positives.** When those false positives are larger than the true spleen, largest-component post-processing keeps the wrong region. LCC was neutral on in-distribution data, but under domain shift it can either rescue a case (raw Dice 0.64 → 0.94) or destroy one (0.45 → 0). This matches the multi-centre degradation reported for single-centre abdominal models [11].

**Multi-centre fine-tuning.** The failure analysis pointed to the training distribution, not the architecture, so we changed the data and kept the model. Starting from the MSD model, we fine-tuned on the 29 MSD training scans plus 44 further TotalSegmentator scans (73 in total). Validation used 6 MSD plus 5 TotalSegmentator scans, and the 25 external test scans stayed held out. The run used 80 epochs and a cosine schedule from lr 5e-5, taking 72 minutes on a T4. Short scans (fewer than 96 slices after resampling) were padded before patch sampling. We selected the checkpoint with the best mean validation Dice, at epoch 65: MSD 0.866 and TotalSegmentator 0.894, compared with 0.901 and 0.663 before fine-tuning.

| Model | MSD test Dice | MSD median abs. vol. error | TS test Dice (mean ± SD / median) | TS failures | TS median abs. vol. error |
|---|---|---|---|---|---|
| Baseline (MSD only) | 0.908 ± 0.053 | 5.8 % | 0.548 ± 0.431 / 0.837 | 10 / 25 | 15.3 % |
| **Multi-centre** | **0.924 ± 0.043** | **2.3 %** | **0.895 ± 0.106 / 0.931** | **1 / 25** | **4.6 %** |

Adding 44 scans from a second centre reduced external failures from 10 to 1 and cut the median external volume error from 15.3 % to 4.6 %. MSD test performance did not drop. The fall in MSD validation Dice during fine-tuning (0.901 → 0.866) did not carry over to the MSD test set (0.908 → 0.924), and with six cases per set we read both changes as noise rather than a real trade-off. This is the TotalSegmentator lesson at small scale: data diversity mattered more than architecture [9]. The multi-centre model is the default in the API.

Caveats:
- The 25 external scans were inspected during the failure analysis above. No training or model-selection decision used them, but they are no longer a completely untouched set.
- TotalSegmentator scans still come from only one additional centre.
- One external case still fails.

## 6. Uncertainty and Explainability

Test-time augmentation with flips gives a cheap stability estimate [12]. Two results are worth recording.

**Laterality is not a free augmentation.** Mirroring axis 0 places the spleen on the patient's right, an orientation that does not occur in abdominal CT, and the model saw flips with only 10 % probability during training. Using all eight flip combinations on `spleen_14` gives a volume spread of 44.3 % (76–445 mL); restricting to the four anatomically plausible views gives 4.8 % (390–445 mL, mean 423.1 mL). The default view set therefore excludes left-right flips, and `--all-flips` is kept as a deliberate robustness probe. The 44 % figure is itself a finding: this model is not laterality-robust, and a deployed version should either be trained to be, or have orientation enforced before inference.

**Stability is not correctness.** On `spleen_1`, where the baseline checkpoint picked the wrong organ, the four TTA views agree to within 3.5 %. Agreement across augmentations says the model is consistent, not that it is right. That is what the anatomy-based quality rules are for, and on the multi-centre checkpoint those rules no longer fire on this scan (9 components, no rule triggered, 153.8 mL) where they did on the baseline (39 components, second region 86 % of the largest, 196.5 mL). The thresholds were set against the baseline's failure mode and have not been re-tuned.

Visually, `--heatmap` shows the CT slice, the spleen probability map and the per-voxel TTA disagreement together. On a clean case the disagreement is confined to a one-to-two-voxel boundary band, which is also where the volume error lives.

## 7. Inference Pipeline and Software

```
CT (.nii.gz) → RAS + resample + HU window + crop → 3D U-Net (sliding window 96³)
            → softmax → threshold → largest component → volume = voxels × 4.5 mm³
            → quality checks → JSON
```

- **CLI:** `python -m src.inference.predict ct.nii.gz --overlay out.png`
- **API:** FastAPI `POST /predict`. Input validation rejects anything that is not NIfTI (415), empty files (400), files over 500 MB (413), and unreadable or non-3D volumes (422). `GET /health` reports model status.
- **Quality flag:** returns `review` if the volume falls outside 30–2000 mL, the mask centroid is on the patient's right, the mean spleen probability is below 0.85, or a second candidate region exceeds 30 % of the kept one. These thresholds come from anatomy and were not tuned on the test data. The last rule targets the external failure mode from Section 5, and it caught `spleen_1`.
- **Runtime:** 1.8–5.6 s per scan on a laptop CPU in Docker, and 1.4–3.9 s on a T4.
- **Uncertainty:** `--uncertainty` runs four flipped views and reports the volume spread; a spread above 10 % adds a `review` flag. `--heatmap` writes the CT, the spleen probability map and the voxelwise disagreement side by side. Section 6 has the numbers.
- **Provenance:** `/health` returns the checkpoint sha256, a hash of the pre-processing and network constants, the git commit and the library versions; `--log-run` writes one JSON record per inference under `results/runs/`.
- **Tests:** 29 pytest tests cover the volume formula, LCC, every quality rule, the API happy path, and each validation error. They use random weights with the real architecture so they run in CI without the checkpoint. A GitHub Actions workflow is included.

## 8. Deployment Considerations

**ONNX.** The export matches PyTorch to a maximum logit difference of 5 × 10⁻⁶, with identical argmax and identical volume. On an x86 laptop CPU, ONNX Runtime was slower than PyTorch (0.55 s vs. 0.26 s for two 96³ patches). The value of the export is portability, not speed on this machine.

**Graph preparation for an NPU.** `src/models/qnn_prepare.py` rewrites the exported graph into the subset converters handle well, and checks itself against the original: the 17 `PRelu` nodes become `LeakyRelu` (every slope here is a scalar, so the substitution is exact), the 4 `BatchNormalization` nodes fold into the preceding `ConvTranspose`, and the batch dimension is pinned to 1. Maximum logit difference 3.1 × 10⁻⁶, identical argmax, and the end-to-end volume on `spleen_14` is unchanged at 444.9 mL. The remaining op types are `Conv`, `ConvTranspose`, `Add`, `Concat`, `LeakyRelu`.

**Dynamic int8, attempted.** `quantize_dynamic` takes the model from 19.3 MB to 7.3 MB, but the quantised graph does not load on ONNX Runtime's CPU provider: `NOT_IMPLEMENTED ... ConvInteger(10)`. There is no int8 3D convolution kernel in that provider. This is a provider gap rather than a model property — int8 3D convolution is exactly what an NPU backend supplies — so the size reduction is the part that transfers.

**Edge targets.** ONNX is the entry point for TensorRT on Jetson and for Qualcomm QNN on Snapdragon / QCS-class SoCs. The converter commands, the calibration recipe and the check order are written out in `docs/qnn_deployment.md`; they need the Qualcomm SDK and a board, so they are documented rather than run. From prior hands-on work deploying ONNX models to a QCS6490 HTP (NPU), the practical constraints for this network are:
- 3D convolutions and transposed convolutions have to be checked against the NPU operator set before committing to a target.
- The NPU wants fixed input shapes. The fixed 96³ sliding-window patch fits this naturally, with the window loop staying on the CPU.
- 8-bit weight/activation quantisation needs a calibration set drawn from real CT patches, and Dice has to be re-measured after quantisation, because boundary voxels are the first to flip.
- Pre-processing (resampling a 512 × 512 × N volume) can cost more CPU time than the network itself and should be profiled separately.

**Path to real use.** This would need multi-centre training data, prospective validation against radiologist volumes, uncertainty estimation (for example test-time augmentation [12]), and integration with DICOM/PACS. The usual regulatory and generalisation challenges also apply [13].

## 9. Limitations

- **Two centres only.** The MSD-only baseline failed on 10 of 25 external scans. Multi-centre fine-tuning brought that to 1 of 25, but performance on a third hospital is untested.
- **Six internal test cases** give a wide confidence range on the headline Dice.
- **Volume is measured on the resampled 1.5 × 1.5 × 2 mm grid.** Thick-slice ground truth is itself coarse.
- **Largest-component post-processing** can discard the true spleen when a larger false positive exists.
- **The quality flag is heuristic.** It catches the observed failure mode, but its sensitivity has not been measured.
- **TotalSegmentator labels are AI-assisted.** They are close to, but not the same as, manual expert labels.
- **The quality thresholds were tuned against the baseline's failure mode** and have not been re-checked on the multi-centre checkpoint, where they no longer fire on `spleen_1`.
- **The model is not laterality-robust.** A mirrored volume changes the reported volume by tens of percent.
- **The NPU path is prepared but not measured.** Graph rewriting and parity are verified here; conversion, quantisation and on-device Dice are not.

## 10. Implemented vs. Used

**Implemented by me:** problem framing, data split and leakage control, pre-processing choices, the two-stage training schedule, evaluation code, failure and external-validation analysis, multi-centre fine-tuning, the inference pipeline, quality checks, TTA uncertainty, the provenance and run-record layer, the FastAPI service, tests, Docker, the ONNX export, the int8 and QNN graph preparation, and this report.

**Used:** MONAI (transforms, `UNet`, `DiceCELoss`, metrics, sliding-window inference), PyTorch, FastAPI, nibabel, SciPy, ONNX Runtime, Google Colab. Data come from MSD Task09 and the TotalSegmentator small subset. An AI coding assistant helped write the code, and I can explain every component.

## 11. Where the Method Comes From

Every substantive choice in this project traces to a specific piece of published work. The short version, in the order the decisions were made:

**Architecture.** U-Net [1] is the base: an encoder for context, a decoder for localisation, and skip connections that return the boundary detail pooling throws away. That matters here because the whole output is a boundary — a spleen volume is just the count of voxels inside it. The 3D variant [2] replaces every operation with its volumetric counterpart and adds batch normalisation; CT is natively 3D and the spleen's extent runs across slices, so a 2D model would have to re-learn continuity it could simply be given. The cost is memory, which is why training works on 96³ patches rather than whole volumes.

**Loss.** V-Net [3] introduced Dice loss for exactly this situation: the spleen occupies 0.1–1.3 % of a scan in this dataset, and cross-entropy alone is minimised well by predicting background everywhere. Dice is computed on the overlap, so it does not care how much background there is. Cross-entropy is kept alongside it because Dice gradients are unstable while the prediction is still empty — early in training there is almost no overlap to differentiate. DiceCE is also nnU-Net's fixed default [4].

**Why not a transformer.** nnU-Net [4] showed that preprocessing, patch size, spacing and training schedule decide results more than architectural novelty, and nnU-Net Revisited [5] re-tested the transformer and Mamba variants against properly configured U-Net baselines and found most claimed gains did not survive. A CNN U-Net with residual units is a defensible 2026 choice for a 41-scan dataset, not a dated one. The same work is the reason the configuration here — 1.5 × 1.5 × 2 mm spacing, 96³ patches, DiceCE, foreground-biased crops — follows nnU-Net's rules rather than being hand-tuned.

**Data.** The Medical Segmentation Decathlon [6] supplies Task09 with its single-centre caveat stated up front, which is what makes the external-validation section meaningful rather than decorative. TotalSegmentator [9] provides the second centre: it is an nnU-Net trained on 1,204 scans across scanners and hospitals, and its lesson — diversity and scale beat architecture — is directly what the multi-centre fine-tune in Section 5 tests, taking external Dice from 0.548 to 0.895 with 44 added scans and no architectural change. Its labels are AI-assisted with expert correction, which is why they are treated as a strong reference rather than ground truth.

**Tooling.** MONAI [7] supplies the transforms, `UNet`, `DiceCELoss`, sliding-window inference and the metrics, so the parts that are standard are the community's implementations rather than mine, and the parts that are specific to this problem are visible in `src/`.

**Metrics.** Taha & Hanbury [15] and Metrics Reloaded [8] make the case that one metric cannot describe a segmentation. This report follows that: Dice for overlap, IoU because reviewers expect it even though it carries the same information, HD95 for boundary error because a correct spleen plus one distant false-positive blob still scores Dice ≈ 0.95 while HD95 exposes it, and volume error because the volume is the actual deliverable. HD95 rather than maximum Hausdorff, since a single noisy voxel should not define the number.

**Clinical framing.** Humpire-Mamani et al. [10] is the reason the task is posed as volumetry rather than segmentation for its own sake: splenomegaly is clinically associated with cirrhosis, portal hypertension, infection and haematological malignancy, and routine practice often substitutes a single length measurement because manual volumetry is too slow for an irregular organ. That is the gap an automated volume fills, and it is also why the quality flag reasons in plain anatomical language.

**Reporting.** The CLAIM checklist [14, 16] set the structure of this document: data source and inclusion, ground-truth provenance, patient-level partitions with leakage control, architecture and initialisation, training hyperparameters and model selection, metrics with spread, failure analysis, limitations, and code availability. Uncertainty via test-time augmentation [12] and the deployment caveats [13] close the same loop.

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
12. Wang G. et al. Aleatoric uncertainty estimation with test-time augmentation. Neurocomputing 2019.
13. Kelly C. J. et al. Key challenges for delivering clinical impact with AI. BMC Medicine 2019.
14. Mongan J. et al. CLAIM checklist. Radiology: AI 2020.
15. Taha A. A., Hanbury A. Metrics for evaluating 3D medical image segmentation. BMC Medical Imaging 2015.
16. Tejani A. S. et al. CLAIM 2024 update. Radiology: AI 2024.

**Figures** (all in `results/`, extracted from the notebook run):

1. `eda_stats.png`, `eda_example.png` — dataset statistics, and one scan before and after the HU window.
2. `preprocessing.png` — the same volume before and after RAS + resampling + windowing.
3. `training_curves.png` — DiceCE loss and validation Dice across both stages.
4. `errors_spleen_44.png`, `errors_spleen_21.png` (with the matching `*_area.png` per-slice plots) — the two worst test cases; `errors_spleen_14.png` for contrast.
5. `inference_example.png` — the end-to-end pipeline output on a test scan.
6. `prediction_spleen_1.png` and `local_spleen_1.png` — the wrong-organ case and the `review` flag it raised on the baseline checkpoint.
7. `samples/spleen_14_uncertainty.png` — CT, spleen probability and TTA disagreement side by side.

Training history is in `results/history.json` and `results/history_ft.json` (parsed from the notebook log, every five epochs). The split is in `results/splits.json` and regenerates exactly with `python -m src.utils.splits`.
