# Measure It

Camera-calibrated calculator segmentation and metric measurement assessment, on branch `dev-test01`. The planning documents and fixed split membership were committed before implementation. **Calibration, reviewed joint image/mask correction and CUDA model fine-tuning are complete. The declared 15-epoch run stopped after 8 epochs under its original validation-plateau rule, selecting epoch 2. Frozen seven-image test evaluation is complete; the one-image measurement CLI is implemented; selected weights are publicly downloadable. The supplied ten-scene accuracy experiment is complete: 9 accepted / 1 refused, and MO-1 is not met. The set contains nine new scenes plus the earlier demo.** Review [pull request #1](https://github.com/Khizar7786/camera-calibrated-object-measurement/pull/1) for the implementation and remaining scope.

## Setup

Recorded calibration environment: Python 3.11.0, OpenCV 4.10.0, NumPy 2.3.1 and Pillow 11.2.1 on Windows 11. Install the pinned dependencies in a local environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

The separate [training requirements](requirements-training.txt) pin **PyTorch 2.7.1+cu126** and **Torchvision 0.22.1+cu126** from the CUDA 12.6 wheel index. Upgrading pip before that install resolves the old installer's wheel-metadata problem recorded in [setup notes](models/setup_notes.json). CUDA availability was verified on the NVIDIA GeForce RTX 3050 Ti Laptop GPU with **4 GiB** memory. The calibration/preprocessing dependencies remain in [requirements.txt](requirements.txt).

The commands below call the project environment's Python explicitly; activation is optional. The original recorded calibration run used `C:\Python311\python.exe`, with one OpenCV thread and seed 42. [Hardware metadata](calibration/results/hardware.json): Intel Core i7-12650H, 16,831,713,280 bytes reported physical RAM, Intel UHD Graphics and NVIDIA RTX 3050 Ti Laptop GPU; calibration uses the CPU. No runtime benchmark is claimed.

## Download the selected checkpoint

The public [calculator-maskrcnn-v1 release](https://github.com/Khizar7786/camera-calibrated-object-measurement/releases/tag/calculator-maskrcnn-v1) provides the frozen epoch-2 checkpoint and its SHA-256 file. From the repository root, download to the path used by the saved inference configuration:

```powershell
Invoke-WebRequest -Uri "https://github.com/Khizar7786/camera-calibrated-object-measurement/releases/download/calculator-maskrcnn-v1/checkpoint_best.pt" -OutFile "models/runs/initial_15_epochs/checkpoint_best.pt"
Invoke-WebRequest -Uri "https://github.com/Khizar7786/camera-calibrated-object-measurement/releases/download/calculator-maskrcnn-v1/checkpoint_best.sha256" -OutFile "models/runs/initial_15_epochs/checkpoint_best.sha256"
$expectedCheckpointHash = "f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39"
if ((Get-FileHash -Algorithm SHA256 "models/runs/initial_15_epochs/checkpoint_best.pt").Hash.ToLowerInvariant() -ne $expectedCheckpointHash) { throw "Checkpoint SHA-256 mismatch" }
```

The checkpoint is **176,220,108 bytes**. Anonymous download access and GitHub's matching SHA-256 digest were verified on 2026-11-06. No retraining is needed to run the frozen evaluation or CLI. Binary weights remain outside Git history.

## Camera calibration

Inputs are the 30 original photographs in `calibration/raw/`. [Board settings](calibration/board.json) record **9 x 7 inner corners** and **20 mm square edges**, physically measured by the user. The complete board has 10 x 8 squares; inner corners are not square counts. The camera is a Redmi Note 12. Focus and zoom lock were confirmed by the user; board flatness and rigid backing remain unverified.

From the repository root:

```powershell
.venv\Scripts\python.exe -m calibration.calibrate
```

The default output `calibration/results/` already contains the recorded run. Reproduction must use a fresh directory to prevent stale parameters from being mistaken for a successful new fit:

```powershell
.venv\Scripts\python.exe -m calibration.calibrate --output calibration/recheck
```

Detection may use a smaller preview, but corner positions are restored with the pixel-centre transform and refined using `cornerSubPix` on the original full-resolution grayscale image. All fitting and reported errors use original pixels. Every input has an outcome in `views.csv` and `run.json`. There is no automatic rejection based on reprojection error. Fewer than 20 usable views causes a nonzero exit and no published camera parameters.

The selected geometry is **2296 x 4080 stored pixels, portrait**: 21 views were usable and all 9 landscape 4080 x 2296 views were rejected before fitting. Global RMS was **0.918653470 px**; the <=1.0 px fit target was met. These intrinsics do not apply directly to landscape captures or the 432 x 432 stretched dataset exports. Preprocessing now restores the export geometry and uses the user-confirmed 90-degree clockwise landscape-to-camera rotation explicitly. The shared function still rejects incompatible arrays rather than silently changing intrinsics.

Saved evidence:

- [camera.json](calibration/results/camera.json): matrix, distortion coefficients, resolution, board settings, fit and per-view RMS errors.
- [views.csv](calibration/results/views.csv): all 30 outcomes, rejection reasons and source SHA-256 hashes.
- [console.txt](calibration/results/console.txt) and [run.json](calibration/results/run.json): raw console output, settings, environment and results.
- [raw_calibration.npz](calibration/results/raw_calibration.npz): full-resolution observed corners, board points in mm, raw OpenCV intrinsics/distortion, extrinsics, uncertainties and errors; load with `numpy.load(..., allow_pickle=False)`.
- `calibration/results/detected_corners/`: 21 corner overlays, scaled for display only.
- [first before/after](calibration/results/before_after/IMG_20261104_194240.jpg.comparison.jpg), [middle](calibration/results/before_after/IMG_20261104_194344.jpg.comparison.jpg) and [last](calibration/results/before_after/IMG_20261104_194406.jpg.comparison.jpg): display previews; adjacent `.before.jpg` and `.after.jpg` files retain full resolution. JPEG copies are visual evidence, not the original calibration inputs.

RMS is `sqrt(sum squared 2D corner residuals / corner count)`, computed globally and per view, and checked against OpenCV. It is an **in-sample camera-fit metric, not proof of millimetre accuracy**. Board curvature and limited edge coverage can bias distortion estimates even with low RMS. The subsequent [ruler-referenced placement experiment](measurement/placements_10/README.md) missed the physical-accuracy target. The implementation follows the [OpenCV calibration API](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html).

## Shared full-frame undistortion

Preprocessing and inference must use the same [camera module](calibration/camera.py):

```python
import cv2
from calibration import CameraCalibration, read_image, undistort, Undistorter

camera = CameraCalibration.load("calibration/results/camera.json")
image = read_image("path/to/original_portrait.jpg")
corrected = undistort(image, camera)

# Reuse the exact map for a batch and for corresponding instance masks.
mapper = Undistorter(camera)
corrected_image = mapper.apply(image)
corrected_mask = mapper.apply(raw_mask, interpolation=cv2.INTER_NEAREST)
```

Coordinates are stored pixels with a top-left origin, x right and y down. EXIF rotation is disabled; EXIF orientation other than 1 is rejected. Resolution must match exactly. The output retains the same image size and matrix (`P=K`), with no cropping, resizing, optimal-ROI selection or automatic rotation. Missing source pixels are zero-filled and must not be treated as valid measurements. Masks must already align with the original stored image; using a stretched export mask directly is invalid.

An array's shape cannot reveal a same-size crop/rotation or altered focus/zoom. Those capture conditions remain the caller's responsibility. Dataset preprocessing now calls this module after explicit geometry normalization; normal inference now uses this same module through the [one-image CLI](inference/README.md).

## Joint dataset correction

The original export at `dataset/train`, `dataset/valid`, `dataset/test`, `dataset/README.roboflow.txt` and the raw originals in `dataset/Raw` are read-only inputs. The run records hashes of all **154 source files** and verifies them again after processing. Frozen source annotation/list hashes are checked before processing. The output must be a fresh directory outside those source trees.

```powershell
.venv\Scripts\python.exe -m preprocessing.prepare_dataset
```

The saved output is `dataset/corrected/`; reproduction requires a fresh path, for example `--output dataset/corrected_recheck`. [Geometry settings](preprocessing/geometry.json) record the original full-frame stretch to 432 x 432, no export crop or quarter-turn, and the separately confirmed camera-frame rotation. Every raw/export pair must pass fixed pixel-comparison checks; the program does not guess a crop or rotation to make a mismatch pass.

Each image follows these operations:

1. Decode each original COCO instance segmentation, supporting polygons and compressed/uncompressed RLE.
2. Restore its raster mask to the original stored-pixel frame using exact nearest-neighbour resizing, reversing any explicitly configured export crop/quarter-turn. This export has neither crop nor quarter-turn. Cropped coverage, if configured for another source, remains flagged as unlabelled outside the crop.
3. Apply the same explicit camera-frame normalization to the original image and restored masks: **24 native portrait images; 48 landscape images rotated 90 degrees clockwise**, as confirmed by the user. This does not modify the calibration matrix.
4. Apply the same cached full-frame undistortion map: linear interpolation for images, nearest-neighbour for categorical masks, `P=K`, no crop or final resize.
5. Encode each corrected mask as lossless COCO RLE and binary PNG; recalculate bbox and pixel area from that mask. Original image/annotation/category IDs, `iscrowd`, filenames and split membership are retained. No original segmentation, bbox or area is reused as the corrected geometry.

Result: **72 corrected 2296 x 4080 images and 100 instances, zero failures**, with **50/15/7** images and **75/18/7** instances by train/validation/test. Corrected image JPEGs use quality 95; binary mask PNGs and COCO RLE are lossless. All original files remain byte-identical, including the unassigned extra raw photo. The original 432 x 432 masks have limited boundary precision; upsampling does not recover lost detail.

Review and evidence:

- [Open the review gallery](dataset/corrected/review.html): every corrected image has a full-resolution overlay and a **passed** user-review status.
- [manual_review.csv](dataset/corrected/manual_review.csv): all 72 entries passed user review. Four images have multiple Calculator instances (23, 4, 3 and 2); those flags are recorded as accepted by the user, with all instances retained.
- [review_approval.json](dataset/corrected/review_approval.json): explicit user approval, timestamp, accepted flags and hashes binding the reviewed artifacts. No annotation or image-pixel edits were requested.
- [run.json](dataset/corrected/run.json): source-to-output mapping, declared geometry, comparison scores, original/output hashes, instance mask paths and completion status.
- Corrected COCO files: [train](dataset/corrected/train/_annotations.coco.json), [validation](dataset/corrected/valid/_annotations.coco.json), [test](dataset/corrected/test/_annotations.coco.json). Each regenerated RLE exactly matches its saved mask PNG and its fresh area/bbox.
- [source_snapshot.json](dataset/corrected/source_snapshot.json), copied frozen lists under `dataset/corrected/splits/`, and [valid_pixels.png](dataset/corrected/valid_pixels.png) record source preservation, membership and the undistortion border region.

The user confirmed that all reviewed items passed. Current [run.json](dataset/corrected/run.json) records `manual_review_complete=true` and `ready_for_training=true`; the original processing-time snapshot is preserved separately in [processing_run.json](dataset/corrected/processing_run.json). Approval is user attestation, not a claim of trained-model performance or measured millimetre accuracy. A fresh preprocessing run still starts with review pending and never inherits this approval automatically. Unknown geometry, changed frozen inputs, empty transformed instances or source changes block normal publication; incomplete splits use explicitly named partial COCO files. The original export is never overwritten. Mask handling uses the [official COCO mask API](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py).

## Dataset and project documents

The recorded setup uses one calculator with user-confirmed **70 x 160 mm** ruler dimensions and a flat unmarked paper-square reference confirmed as **50 x 50 mm**. The user corrected the initial approximate 75 mm width before placement evaluation; historical planning estimates are retained. Object photographs are user-provided phone captures, annotated through Roboflow's tool and exported as COCO segmentation. The corrected annotations have passed user review; no Roboflow model is used.

The supplied export has **72 images: 50 train / 15 validation / 7 test**. Exact membership is frozen in [dataset/splits](dataset/splits/README.md) and retained in the corrected dataset. There are 73 raw object photographs; the extra capture remains unassigned. Preserve the test split and all derivatives from training, tuning and checkpoint selection. Model fitting uses the approved training split, with validation used for checkpoint selection. `.gitattributes` preserves exact bytes of the source export and reviewed artifacts so their integrity hashes survive checkout on other platforms.

The three assessment documents are [Plan](docs/Plan.md), [Objectives](docs/Objectives.md) and [Definition of Done](docs/Definition_of_Done.md). They distinguish measured evidence from work still pending.

## Segmentation training

The completed run was started with:

```powershell
.venv\Scripts\python.exe -m models.train --config models/training_config.json --output models/runs/initial_15_epochs
```

This directory already contains run evidence. Reproduction requires a fresh output path, such as `models/runs/reproduction_01`. [Training usage](models/README.md) documents the two-class Mask R-CNN, original **15-epoch** budget, aspect-preserving **512/960** sizing, batch **1**, workers **0**, seed **42**, fixed thresholds and validation-only AP selection. Training constructs only 50 training and 15 validation images; test data is not loaded.

The [actual smoke check](models/runs/initial_15_epochs/smoke.json) passed a forward/backward update on the 23-instance training image and a validation forward. Peak CUDA allocation was **891,221,504 bytes**, with **1,482,686,464 bytes** reserved. Smoke weights were discarded before real training.

The [finished run](models/runs/initial_15_epochs/run.json) records **8 completed epochs and 400 real optimizer updates**. Its original 15-epoch budget ended with `validation_plateau_early_stop`: the minimum of 8 epochs was reached after 6 epochs without AP improvement exceeding 0.001 following epoch 2. No out-of-memory configuration changes were needed. The declared scheduler changed learning rate from **0.0025 to 0.00025 after epoch 5**. Training-loop wall time was **309.344 seconds**, excluding setup, downloads and smoke; this single recorded duration is not a runtime benchmark. Epoch GPU peaks were **1,072,794,112 bytes allocated** and **1,245,708,288 bytes reserved**.

[Epoch 2 was selected](models/runs/initial_15_epochs/selection.json) by validation segmentation AP, retaining the earliest maximum. Its [raw validation outputs](models/runs/initial_15_epochs/validation/epoch_02/metrics.json) cover **15 validation images / 18 ground-truth instances**:

| Validation metric | Selected-epoch result |
| --- | ---: |
| Segmentation mAP@0.50:0.95 | 0.8316831683168316 |
| Segmentation mAP@0.50 | 0.8316831683168316 |
| Precision at score >=0.5 and mask IoU >=0.5 | 1.0 |
| Recall at the same operating point | 0.8333333333333334 |
| Mean ground-truth mask IoU, unmatched GT zero | 0.8070517357958565 |
| True positives / false positives / false negatives | 15 / 0 / 3 |

The selected local checkpoint, `models/runs/initial_15_epochs/checkpoint_best.pt`, is **176,220,108 bytes**. Its SHA-256 is:

```text
f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39
```

[Checkpoint verification](models/runs/initial_15_epochs/verification.json) passed an offline strict CPU reload with both heads configured for two classes, found 400 finite-loss optimizer steps, matched runtime source hashes, and rechecked 227 approved training/validation artifacts and the frozen lists. This integrity check performed no inference or test evaluation. The selected binary is available in the public [model release](https://github.com/Khizar7786/camera-calibrated-object-measurement/releases/tag/calculator-maskrcnn-v1); binary checkpoints remain ignored by Git.

Raw [console output](models/runs/initial_15_epochs/console.txt), [epoch records](models/runs/initial_15_epochs/epochs.jsonl), per-epoch validation outputs and the run's environment/configuration snapshots are preserved. ROIAlign backward reported a CUDA nondeterminism warning in [warnings.txt](models/runs/initial_15_epochs/warnings.txt). Seeds and deterministic settings were requested, but identical results across repeated runs are not guaranteed. These scores describe the validation split used for selection. The frozen test evaluation is recorded below; the supplied placement experiment below reports physical errors and MO-1 not met.

## Frozen held-out mask evaluation

```powershell
.venv\Scripts\python.exe -m models.evaluate_test --output models/evaluations/frozen_test_01
```

The [saved test report](models/evaluations/README.md) evaluates exactly the seven committed test images with the frozen epoch-2 checkpoint. Mean GT mask IoU is **0.967517224**; official COCO segmentation **AP50=1.0, AP50:95=1.0**; operating **precision=1.0, recall=1.0** (7 TP, 0 FP, 0 FN). Operating score and mask probability are 0.5; precision/recall require one-to-one mask IoU >=0.5. Mean IoU divides matched-mask IoUs by every GT instance, with unmatched GT zero. AP retains every returned scored mask at the original 0.05 model floor, without the measurement gate. Geometry stays aspect-preserving 512/960 (512 x 910 here).

The report includes the command, checkpoint/split hashes, configuration, all seven filenames, per-image results, raw scored masks, GT masks and COCO output. Saved-prediction COCO replay and independent mask IoU calculations passed. No settings or checkpoint were changed using test scores. This small split does not establish millimetre accuracy or performance on other calculators. Reproduction requires a fresh output directory.

## One-image measurement CLI

Earlier ad hoc inference runtime photos and diagnostic outputs remain local in ignored folders. The user-supplied placement experiment now saves its ten input copies and raw outputs separately under `measurement/placements_10/`; paths labelled local output still refer to the earlier local runs.

```powershell
.venv\Scripts\python.exe -m inference.measure path/to/new_original.jpg --output inference/runs/my_photo
```

The [CLI configuration and output guide](inference/README.md) uses your ruler-measured **50 x 50 mm unmarked paper square**. It validates the original frame, always applies saved calibration/shared undistortion, loads the frozen calculator checkpoint, detects/refines the paper's outer corners, transforms the actual predicted mask contour to millimetres and fits an oriented rectangle **there**. Width is the shorter side; height the longer. Outputs include a full corrected-frame overlay, mask, geometry and JSON with status, dimensions, confidence, reference, checkpoint/configuration IDs and failure reason. Missing object/reference returns null dimensions. An ambiguous square is refused.

The single conversion is [measure_mask_contour_mm](measurement/planar.py#L11). [Current verification](inference/verification.json) records 88 passing tests and the supplied-photo success; [initial verification](inference/verification_initial.json) records the earlier synthetic geometry and real-network failure checks. The original-photo overlay (local output: `inference/runs/reference_missing_verified/preview.jpg`) shows calculator segmentation with a missing-reference refusal. The first supplied real photo now completes (local output: `inference/runs/my_photo_retry/result.json`) at **80.47 x 170.60 mm** after a wood-grain detector fix; the overlay (local output: `inference/runs/my_photo_retry/preview.jpg`) shows its paper boundary and calculator mask. The subsequent [placement experiment](measurement/placements_10/README.md) records user-confirmed plane alignment and measured errors; independent support-height/flatness/thickness verification remains absent. Segmentation scores and synthetic geometry do not establish mm accuracy.

## Supplied placement accuracy

```powershell
.venv\Scripts\python.exe -m measurement.evaluate_placements --manifest measurement/placements_10/manifest.json --output measurement/placements_10/reproduction_01 --device cuda
```

The saved experiment used `measurement/placements_10/results`; reproduction requires a fresh directory. [Full report](measurement/placements_10/README.md), [Objectives table](docs/Objectives.md#supplied-placement-experiment---2026-11-06), [CSV](measurement/placements_10/results/per_instance.csv), [hand readings](measurement/placements_10/hand_readings.csv) and [overlays](measurement/placements_10/overlay_contact_sheet.jpg) retain every supplied scene. The user confirmed **70 x 160 mm calculator dimensions**, **50 x 50 mm paper**, locked focus/zoom and a shared reference/outline plane before inference. Original/copy hashes and frozen splits/configuration are verified; no model or threshold changes.

**9 accepted / 1 refused (90% coverage).** Accepted-only width MAE is **11.843224 mm**, height MAE **10.962348 mm**; worst width/height errors **17.836929 / 18.858841 mm**. The report includes mean/median/worst absolute and percentage errors, with n=9 per dimension. The reference-not-found refusal keeps null dimensions/missing errors. **MO-1 is not met.** This checks one calculator's repeatability, not product generalization. Nine photos are new; case_01 reuses the earlier demo, so the ten genuinely new scenes requirement remains incomplete. The paired raw-versus-corrected comparison is complete below; the corrected accuracy target remains not met.

## Paired raw-versus-corrected comparison

```powershell
.venv\Scripts\python.exe -m measurement.compare_undistortion --manifest measurement/paired_undistortion_10/manifest.json --output measurement/paired_undistortion_10/reproduction_01 --device cuda
```

[Full comparison](measurement/paired_undistortion_10/README.md), [Objectives tables](docs/Objectives.md#raw-versus-undistorted-measurement---completed-2026-11-06), [paired CSV](measurement/paired_undistortion_10/results/paired.csv), [20-row mode/error CSV](measurement/paired_undistortion_10/results/per_mode.csv), [overlays](measurement/paired_undistortion_10/paired_overlay_contact_sheet.jpg) and [verification](measurement/paired_undistortion_10/verification.json) preserve both fresh runs for every original scene. The saved output is `measurement/paired_undistortion_10/results`; reproduction uses a fresh directory.

Corrected mode accepted **9/10**, raw mode **7/10**, with **7 jointly accepted accuracy scenes**. On that identical paired set, skipping correction changes width MAE **13.317996 -> 15.070784 mm (+1.752788 mm)** and height MAE **12.762610 -> 15.076059 mm (+2.313448 mm)**. Reports show means, medians, worst absolute/percentage errors, per-mode/paired denominators and every refusal. Missing dimensions/errors remain missing. Cases 04 and 05 improve without correction; raw paired median width error is lower despite its higher mean. Case_08 has the largest worsening. This is an end-to-end input comparison; changes in masks, reference corners and refusal coverage also contribute, so the difference is not isolated geometric scale.

Normal inference is unchanged and always undistorts. The [separate raw experiment CLI](measurement/raw_inference.py) requires `--experimental-skip-undistortion`, labels raw frames explicitly, and calculates new reference corners and mask geometry in raw pixels. It never loads corrected corners/masks. Both modes share the frozen configuration/checkpoint and exact contour-to-mm conversion. All ten corrected controls exactly reproduce the prior results/scores/masks. The original set still contains nine new photos plus the earlier demo; neither mode meets the physical-accuracy target.

## Verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On 2026-11-05 the isolated project `.venv` was created with system site-packages disabled. The dependencies include pinned `pycocotools==2.0.10` and the recorded CUDA PyTorch/Torchvision builds. All **88 tests pass inside the actual project environment**. The saved calibration fit retains its original interpreter provenance.

Tests cover calibration, image/mask geometry, source/split integrity, approval checks, joint training augmentation, model/head reconstruction, synthetic validation metrics, empty detections and test-split refusal in the training loader. The CUDA smoke separately exercises the actual GPU forward/backward path. On 2026-11-06 the placement-aggregation regressions brought the suite to **93 passing tests**; [clean test log](measurement/placements_10/verification_tests.txt) and [independent placement verification](measurement/placements_10/verification.json) are saved. The paired experiment adds seven bypass/frame/denominator tests: **100 tests pass**; [preflight log](measurement/paired_undistortion_10/preflight_tests.txt).

## Handover and unfinished evidence

[PR #1](https://github.com/Khizar7786/camera-calibrated-object-measurement/pull/1) proposes `dev-test01` into `main`, starting from the original planning commit. The [completion checklist](docs/Definition_of_Done.md) records the missed MO-1 accuracy target and incomplete ten-new-scene baseline, independent plane/thickness checks and fresh-clone installation/full reproduction. The paired raw-versus-corrected experiment is complete with mixed per-scene effects and explicitly reported coverage. Separate runtime, breaking-condition and trust/refusal-cost studies were deliberately omitted under the agreed time limit. Physical errors and the observed paired correction effect are reported for these scenes only; geometric-scale, segmentation and reference-corner contributions are not isolated.

The author will add the Loom URL directly to the assessment submission email. Its duration/content and submission timing were not verified here; the assessment requires a recording of five minutes or less. This repository is an honest partial submission with implemented stages and missing experimental evidence identified.
