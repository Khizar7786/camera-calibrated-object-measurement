# Measure It - Objectives

**Status:** portrait camera calibration measured and joint dataset correction generated on 2026-11-05. The user approved all 72 corrected images and 100 instances on 2026-11-05; Mask R-CNN fine-tuning completed with validation-only checkpoint selection; the frozen seven-image mask evaluation is complete; millimetre measurement remains unrun. Original objective targets are unchanged.

**Branch:** `dev-test01`. **Units:** millimetres for physical dimensions; pixels for reprojection error.

## 1. Setup and current progress

| Field | Recorded setup |
| --- | --- |
| Object | One rigid calculator, chosen for its clear outer outline and repeatable handling; measure its complete outer body, excluding shadows, hands, individual buttons and the display. |
| Preliminary hand measurements | Width **75 mm**, height **160 mm**, reported using a ruler. Width means the shorter side and height the longer side, regardless of image rotation. Confirm readings for the accuracy experiment. |
| Reference | A **flat square cut out of paper**, reported size **50 x 50 mm**. This is an unmarked square, not an ArUco marker. Measure its actual cut edges and keep the same reference throughout evaluation. |
| Ground-truth instrument | Ruler marked in millimetres. Reading uncertainty must be acknowledged; no sub-millimetre accuracy claim is planned. |
| Camera | **Redmi Note 12**. |
| Verified image geometry | Raw object captures: **49 landscape 4080 x 2296** and **24 portrait 2296 x 4080**; calibration captures: **9 landscape 4080 x 2296** and **21 portrait 2296 x 4080**. All 72 exported object images are **432 x 432**. Export metadata reports auto-orientation and stretch resizing; source-to-export geometry must be resolved before calibration is applied to exported labels. |
| Reported capture metadata | f/1.8; shutter 1/17 s; ISO 594; physical focal length 4.2 mm; no flash. These are reported photo metadata, not a claim that exposure or focus stayed fixed across every capture. |
| Camera consistency | User confirmed focus and zoom locked throughout calibration and calculator captures. Original EXIF records inspected share Xiaomi model 23021RAAEG, focal length 4.22 mm and orientation 1. The user additionally confirmed the landscape originals map into the calibrated portrait frame by **90 degrees clockwise**. Preprocessing records that explicit normalization for 48 images; 24 already match. The shared undistorter accepts only **2296 x 4080** arrays. Physical camera consistency is user-confirmed, not inferred from dimensions alone. |
| Calibration photos | **30 originals** in `calibration/raw/`: **21 usable portrait views**, **9 landscape geometry rejections**, **0 detection failures among the 21 compatible inputs**. All accepted views have 63 full-resolution refined corners. Full outcomes and source hashes are saved in `calibration/results/views.csv`. |
| Calibration board | Complete checkerboard: **10 x 8 squares**, hence **9 x 7 inner corners**, confirmed from the whole board and printed footer. User physically measured square edges as **2 cm x 2 cm = 20 mm**. Board flatness/rigid backing cannot be verified from photographs and remains a limitation. |
| Object annotations | Original phone photos annotated through Roboflow, exported as COCO segmentation. All **100 original instances** were decoded and jointly corrected with their images. Corrected masks, RLE, bboxes and pixel areas passed automated consistency checks, and the user confirmed all 72 images passed manual review. Original export files are byte-identical. |
| Split policy | Preserve existing export membership: **50 train / 15 validation / 7 test = 72 images**, approximately 70/20/10. Exact dataset-relative paths are frozen in `dataset/splits/train.txt`, `valid.txt` and `test.txt`; see the split README for conventions and integrity checks. Commit these lists before code; do not reshuffle. |
| Original/export inventory | All 72 exports map uniquely to raw originals by capture filename. `dataset/Raw/` contains **73** photos; `IMG_20261104_200608.jpg` is unassigned and excluded from the frozen splits. Do not silently introduce it into training or evaluation. |
| Model | Pretrained Torchvision Mask R-CNN ResNet50-FPN, fine-tuned on the calculator class. |
| Reproducibility | Seed 42; exact CUDA environment, full configuration, source/data hashes, command, 400 update logs, eight epoch/validation records and selected-checkpoint SHA-256 are saved in `models/runs/initial_15_epochs/`. |
| Calibration machine | Intel Core i7-12650H; 16,831,713,280 bytes reported physical RAM (about 15.68 GiB); Intel UHD Graphics and NVIDIA GeForce RTX 3050 Ti Laptop GPU; Microsoft Windows 11 Home, build 10.0.26300. Calibration uses the CPU with one OpenCV thread. Exact CIM output: `calibration/results/hardware.json`. Future training hardware must be recorded if different. |
| Calibration software | Original fit: system Python **3.11.0**, OpenCV **4.10.0** (`opencv-python==4.10.0.84`), NumPy **2.3.1**, Pillow **11.2.1**, seed **42**. Versions are pinned in `requirements.txt`; raw fit environment is in `calibration/results/run.json`. An isolated project `.venv` was subsequently created with the same pinned versions; the full project now has **76 passing tests** inside it and dependency checks passed. Ongoing commands use that environment explicitly. |

### Dataset correction and review before training

The original annotations live in **432 x 432 exports of distorted originals**. They remain preserved with those export images and raw captures. `preprocessing/prepare_dataset.py` now decodes each mask, reverses the declared export geometry, applies the same explicit camera-frame rotation as its original image, then uses the **same undistortion map for image and masks** from `calibration/camera.py`. Mask restoration uses exact nearest-neighbour resizing; lens remapping uses nearest-neighbour for masks and linear interpolation for images. Every output segmentation, bbox and area is regenerated from its corrected raster mask. No undistorted image is paired with unchanged source annotations.

The export README reports **auto-orientation and full-frame stretch resizing to 432 x 432**. Automated raw/export comparisons support no crop or additional quarter-turn in that export. The separate raw-to-camera normalization is **0 degrees for 24 portrait originals** and **90 degrees clockwise for 48 landscape originals**, explicitly confirmed by the user. All corrected arrays retain the calibration's **2296 x 4080** full frame and K; mismatches or unverified transforms are refused. The source calibration views and fitted parameters are unchanged.

Command: `.venv\Scripts\python.exe -m preprocessing.prepare_dataset`. Output: `dataset/corrected/`, generated separately from all original files. **72 images and 100 instances processed, zero failures**. Train/validation/test counts remain **50/15/7 images** and **75/18/7 instances**. Image, annotation and category IDs, filenames and `iscrowd` are retained. All **154 source snapshot hashes** and six original freeze checksums match after processing; copied split lists are byte-identical. All 100 corrected COCO RLEs match their saved binary PNGs exactly, with matching regenerated bbox/pixel-area values.

Evidence: [processing run and per-instance mapping](../dataset/corrected/run.json), [source preservation snapshot](../dataset/corrected/source_snapshot.json), [explicit geometry settings](../preprocessing/geometry.json), corrected COCO [train](../dataset/corrected/train/_annotations.coco.json), [validation](../dataset/corrected/valid/_annotations.coco.json) and [test](../dataset/corrected/test/_annotations.coco.json), and [manual-review gallery](../dataset/corrected/review.html). The script uses pinned `pycocotools==2.0.10` for lossless mask RLE and official bbox/area semantics.

**All 72 images passed manual review by user confirmation on 2026-11-05.** Four images retain multiple Calculator instances: training captures `200604` (23) and `200521` (4), validation `200536` (3) and `200523` (2). Their flags are retained as accepted by the user in [manual_review.csv](../dataset/corrected/manual_review.csv); all 100 instances remain, with no label/pixel edits requested. [review_approval.json](../dataset/corrected/review_approval.json) records the user attestation, timestamp and reviewed-artifact hashes. Current run metadata sets `manual_review_complete=true` and `ready_for_training=true`; [processing_run.json](../dataset/corrected/processing_run.json) preserves the original pre-review processing state. Restoring a 432 x 432 raster cannot recover lost boundary precision. User acceptance does not establish trained-model performance or millimetre accuracy.

**Planning alignment, 2026-11-05:** the Plan now records the paper-square reference, joint correction of existing annotations and verified 50/15/7 split. The reference's reported 50 x 50 mm dimensions still need confirmation. Filename lists accompany this initial planning snapshot before implementation code. This inventory check is preparation, not evidence of calibrated images, reviewed masks or training results.

## 2. MO-1 - Measurement accuracy

| Field | Commitment |
| --- | --- |
| What is measured | Error in the calculator's predicted outer width and height against hand measurements. |
| How | Use at least **10 additional photographed placements**, separate from train/validation/test. For each placement, record ruler readings and run the frozen calibrated image-to-mm pipeline. Save every input, overlay and machine-readable output. |
| Target | Corrected width MAE **<=3 mm** and corrected height MAE **<=3 mm**; worst individual dimensional error **<=8 mm**. All 10 placements must return valid measurements for a complete pass. |
| Conditions | Same camera geometry as calibration; calculator and reference fully visible; nearly overhead view; adequate light and sharp boundaries. Mount the paper reference flat on a rigid support without changing its dimensions. Its plane must match the outline being measured; support its height if needed. Calculator thickness remains a limitation. |
| Not included | Thickness measurement, unseen calculator models, arbitrary camera settings, and precision manufacturing inspection. Missing measurements are reported explicitly. |
| Why this target | A few millimetres is a useful external-size estimate for a 75 x 160 mm calculator while remaining meaningfully stricter than a rough visual estimate. The ruler's uncertainty limits the strength of the accuracy claim. |
| Result and evidence | **Not run.** Add the actual command, input IDs, hand readings, raw outputs, errors and met/missed verdict after evaluation. |

For each accepted dimension:

`absolute_error_mm = abs(predicted_mm - ruler_mm)`

`percentage_error = 100 * absolute_error_mm / ruler_mm`

Compute width and height MAE separately. Report mean percentage error, median and worst absolute/percentage errors for both dimensions, plus accepted and failed counts. Do not treat a failed measurement as zero error or hide it by removing the image. If fewer than 10 valid measurements are available, report accepted-only statistics with their denominator and mark the complete target unmet or not reached.

The experiment repeats **one physical calculator across different placements**. It does not establish accuracy across different calculator products. Record actual hand readings for each placement; do not invent variation in the size of a rigid object.

## 3. MO-2 - Calibration fit quality

| Field | Commitment |
| --- | --- |
| What is measured | Global RMS reprojection error in pixels, with usable calibration-image count and per-view errors. |
| How | Detect/refine checkerboard corners with OpenCV; use the measured board geometry to fit intrinsics and distortion coefficients. Save OpenCV's global RMS, per-view errors, included/excluded image IDs, camera matrix, distortion coefficients and calibration image size. |
| Target | Global RMS reprojection error **<=1.0 pixel** using **at least 20 usable views**. |
| Conditions | Sharp visible corners, a flat rigidly supported board, varied poses/positions and consistent camera settings. Record actual inner-corner layout and measured square size. |
| Not included | A guarantee of millimetre accuracy, or validity for a different lens, focus, zoom, resolution or unaccounted orientation. |
| Why this target | A one-pixel RMS is a practical initial quality target for checking the fitted camera model. MO-1 separately verifies physical measurement accuracy. This uses calibration evidence already required by item 1, adding no separate performance experiment. |
| Result and evidence | **MET for the 2296 x 4080 portrait profile:** global RMS **0.918653470 px** from **21 usable views / 30 captures**; 9 landscape views rejected for geometry. Parameters, raw outputs, per-view results and comparisons are linked below. This is a fit-quality verdict, not a measurement-accuracy verdict. |

Keep the global RMS definition consistent with OpenCV's calibration output. Record excluded views and reasons; do not remove images merely to hide a poor result. If fewer than 20 usable views remain, collect additional calibration views or declare the target incomplete.

### Measured calibration evidence - 2026-11-05

Original command run from the repository root: `C:\Python311\python.exe -m calibration.calibrate`. All 30 raw inputs are preserved; the fit uses the 21 compatible portrait views and excludes the 9 landscape inputs solely for mismatched stored geometry. No accepted views were removed based on their errors. Ongoing development and reproduction use `.venv\Scripts\python.exe` as documented in the root README.

OpenCV global RMS: **0.918653470 px**. Independently recomputed from saved full-resolution corner residuals: **0.918652935 px**. Per-view RMS mean: **0.886813970 px**; median: **0.919445761 px**; worst: **1.414178343 px**. Global RMS is the square root of the mean squared 2D residual over all 1,323 corners; it is not the arithmetic mean of per-view RMS values.

Camera matrix K (pixels), for **width 2296 x height 4080**, origin top-left:

```text
  [3354.623257, 0.000000, 1158.996288]
  [0.000000, 3361.925604, 2051.340675]
  [0.000000, 0.000000, 1.000000]
```

Distortion coefficients `[k1, k2, p1, p2, k3]`: `[0.155122733, -0.898786257, 0.000820837, 0.001618363, 2.216512703]`. Full-precision values, resolution, board metadata and all accepted per-view errors are saved in [camera.json](../calibration/results/camera.json).

Undistortion keeps **P=K and the original full frame**. No optimal ROI, crop, automatic rotation or intrinsic scaling is used. The corner detector works on a bounded preview for speed, restores the pixel-centre coordinates to the original frame, and refines all 63 corners with `cornerSubPix` on the original grayscale image. Categorical masks can use the exact same remap with nearest-neighbour interpolation. Non-normal EXIF orientation and mismatched stored resolution are rejected.

Full outcomes below use IDs from `calibration/raw/IMG_20261104_<ID>.jpg.jpeg`; geometry rejections are 4080 x 2296 landscape images and were not submitted to corner detection or fitting.

| View ID | Outcome | Per-view RMS (px) |
| --- | --- | ---: |
| 194233 | Rejected: resolution/orientation | n/a |
| 194236 | Rejected: resolution/orientation | n/a |
| 194240 | Used: 63 refined corners | 1.179019 |
| 194244 | Used: 63 refined corners | 1.062141 |
| 194249 | Used: 63 refined corners | 0.986644 |
| 194254 | Used: 63 refined corners | 1.166659 |
| 194258 | Used: 63 refined corners | 0.550135 |
| 194301 | Rejected: resolution/orientation | n/a |
| 194303 | Rejected: resolution/orientation | n/a |
| 194306 | Rejected: resolution/orientation | n/a |
| 194325 | Used: 63 refined corners | 0.813655 |
| 194328 | Used: 63 refined corners | 0.980455 |
| 194330 | Used: 63 refined corners | 0.920371 |
| 194339 | Used: 63 refined corners | 1.414178 |
| 194342 | Used: 63 refined corners | 1.029803 |
| 194344 | Used: 63 refined corners | 1.114584 |
| 194346 | Used: 63 refined corners | 0.530166 |
| 194348 | Used: 63 refined corners | 0.464852 |
| 194350 | Used: 63 refined corners | 0.550104 |
| 194352 | Used: 63 refined corners | 0.757187 |
| 194354 | Used: 63 refined corners | 0.777067 |
| 194357 | Used: 63 refined corners | 0.852713 |
| 194359 | Used: 63 refined corners | 0.661036 |
| 194401 | Used: 63 refined corners | 0.919446 |
| 194404 | Used: 63 refined corners | 1.035720 |
| 194406 | Used: 63 refined corners | 0.857157 |
| 194409 | Rejected: resolution/orientation | n/a |
| 194411 | Rejected: resolution/orientation | n/a |
| 194414 | Rejected: resolution/orientation | n/a |
| 194606 | Rejected: resolution/orientation | n/a |

Evidence: [all input outcomes/source hashes](../calibration/results/views.csv), [raw console](../calibration/results/console.txt), [run configuration/environment](../calibration/results/run.json), [raw OpenCV arrays and full-resolution observations](../calibration/results/raw_calibration.npz), [hardware metadata](../calibration/results/hardware.json), and [shared undistortion module](../calibration/camera.py). The NPZ includes camera parameters, per-view extrinsics, standard deviations, original board/image points and OpenCV per-view errors, and opens without pickle.

Visually inspected full-frame comparison previews: [first](../calibration/results/before_after/IMG_20261104_194240.jpg.comparison.jpg), [middle](../calibration/results/before_after/IMG_20261104_194344.jpg.comparison.jpg), [last](../calibration/results/before_after/IMG_20261104_194406.jpg.comparison.jpg). The adjacent `.before.jpg` and `.after.jpg` files retain 2296 x 4080 dimensions; the comparison panels and corner overlays are scaled only for display. Black borders are retained, not cropped or treated as measurements.

**Interpretation and limits.** MO-2 is met for this in-sample portrait fit. The board appears mostly central and flatness/rigid backing are not independently verified; paper curvature and extrapolation toward frame edges may bias fitted distortion. Focus/zoom lock is user-confirmed, not established by EXIF. Dataset preprocessing now explicitly resolves export geometry and the user-confirmed landscape-to-portrait rotation before invoking the unchanged strict undistorter; applying these intrinsics directly to an incompatible array remains invalid. **MO-1 millimetre accuracy remains Not run** and requires independent ruler-based measurements; reprojection error alone does not establish it.

## 4. Other required evaluation evidence

### Recorded fine-tuning and validation - 2026-11-05

Command from the repository root:

```powershell
.venv\Scripts\python.exe -m models.train --config models/training_config.json --output models/runs/initial_15_epochs
```

Torchvision Mask R-CNN ResNet50-FPN starts from official COCO pretrained weights; both box and mask prediction heads are replaced for background `0` and calculator `1`. Only the approved frozen **50 train / 15 validation** images are loaded, with **75 / 18 instances**. Training-only horizontal flipping jointly transforms images and masks. The loader preserves aspect ratio, uniformly resizes the current full-frame images to **512 x 910**, uses exact nearest-neighbour masks, and recomputes boxes/areas. Batch size is **1** and Windows loader workers **0**. No test loader, image, annotation or split list is used in training, augmentation, threshold decisions or checkpoint selection.

The initial **15-epoch** budget, SGD learning rate **0.0025**, last **three** trainable backbone stages, mixed precision, seed **42** for Python/NumPy/PyTorch, fixed thresholds and validation plateau rule were recorded before GPU smoke. The smoke exercised the **23-instance** training image plus one validation forward: finite losses/backward/update and validation outputs passed; peak allocated memory was **891,221,504 bytes**, reserved **1,482,686,464 bytes**. Diagnostic weights were discarded and all seeds/model/optimizer state reset before fitting.

Actual training completed **8 epochs / 400 optimizer updates** in **309.344 seconds** of training-loop wall time. The predeclared rule stopped after six epochs without validation AP improvement greater than 0.001, at the minimum eight completed epochs. This is an evidence-based early stop within the 15-epoch initial budget. There was no out-of-memory adaptation or model/resolution/backbone change. The predeclared StepLR changed learning rate from **0.0025 to 0.00025 after epoch 5**, recorded in [configuration_changes.jsonl](../models/runs/initial_15_epochs/configuration_changes.jsonl). This elapsed training time is an observed run cost, not the omitted runtime benchmark.

**Selected checkpoint: epoch 2**, the earliest highest validation COCO mask AP50:95. Later equal scores retain that checkpoint under the declared tie rule. These results use all **15 validation images / 18 labelled instances**, in resized loader coordinates:

| Validation metric | Selected epoch 2 result |
| --- | ---: |
| Segmentation mAP@0.5:0.95 | **0.831683** |
| Segmentation mAP@0.5 | **0.831683** |
| Precision, score >=0.5 and mask IoU >=0.5 | **1.000000** (15 TP, 0 FP) |
| Recall at that operating point | **0.833333** (3 FN) |
| Mean ground-truth mask IoU, unmatched GT zero | **0.807052** across all 18 GT instances |

Mask probability, operating score and matching IoU are fixed at **0.5**; the model returns predictions from score **0.05** with box NMS **0.5**. AP uses the official COCO evaluator over all returned predictions; operating metrics use score-ordered one-to-one matching. Mean GT IoU uses its documented separate positive-overlap matching and includes unmatched GT as zero. These are checkpoint-selection validation results, not held-out test results or millimetre-accuracy evidence. All user-accepted instances are retained.

Selected local checkpoint: `models/runs/initial_15_epochs/checkpoint_best.pt` (**176,220,108 bytes**). SHA-256:

```text
f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39
```

[verification.json](../models/runs/initial_15_epochs/verification.json) confirms the hash, offline CPU reconstruction with strict state loading and both two-class heads, 400 finite update records, unchanged runtime source hashes, and **227** unchanged approved train/validation artifact hashes plus frozen lists. Binary checkpoints are local/ignored; an accessible reviewer download is still pending.

Actual environment: isolated Python **3.11.0**, PyTorch **2.7.1+cu126**, Torchvision **0.22.1+cu126**, CUDA build **12.6**, NVIDIA RTX **3050 Ti Laptop GPU** with **4,294,443,008 bytes** reported GPU memory. Machine/OS/RAM are the calibration machine recorded above. [environment.json](../models/runs/initial_15_epochs/environment.json) and [requirements.freeze.txt](../models/runs/initial_15_epochs/requirements.freeze.txt) preserve exact installed versions; [pip_freeze_raw.txt](../models/runs/initial_15_epochs/pip_freeze_raw.txt) preserves pip's local-wheel provenance. CUDA NMS/backward checks and all **72 tests** passed. PyTorch reports that CUDA ROIAlign backward lacks a deterministic implementation; [warnings.txt](../models/runs/initial_15_epochs/warnings.txt) preserves the actual warning. Seeded settings do not imply bitwise repeatability across runs/hardware.

Evidence: [saved configuration](../models/runs/initial_15_epochs/configuration.json), [command](../models/runs/initial_15_epochs/command.txt), [run status](../models/runs/initial_15_epochs/run.json), [train/validation data manifest](../models/runs/initial_15_epochs/data_manifest.json), [actual model/head summary](../models/runs/initial_15_epochs/model_summary.json), [smoke](../models/runs/initial_15_epochs/smoke.json), [console](../models/runs/initial_15_epochs/console.txt), [step losses](../models/runs/initial_15_epochs/steps.jsonl), [epoch losses/results](../models/runs/initial_15_epochs/epochs.jsonl), [selection and hash](../models/runs/initial_15_epochs/selection.json), selected [COCO output](../models/runs/initial_15_epochs/validation/epoch_02/coco_eval.txt) and [metrics/definitions](../models/runs/initial_15_epochs/validation/epoch_02/metrics.json). [Setup notes](../models/setup_notes.json) retain the pip metadata failure, pip upgrade and SHA-verified resumed CUDA download; no setup success was represented as training success.

### Segmentation model - frozen held-out test, 2026-11-05

Exactly **7 committed test images / 7 GT instances** were evaluated with the original validation-selected **epoch-2** checkpoint. No test-driven training, augmentation, threshold/resolution changes or checkpoint reselection occurred. Full [test report](../models/evaluations/README.md).

```powershell
.venv\Scripts\python.exe -m models.evaluate_test --output models/evaluations/frozen_test_01
```

| Mask metric | Test result |
| --- | ---: |
| Mean ground-truth mask IoU, unmatched GT zero | **0.967517224** |
| Segmentation mAP@0.5 | **1.000000** |
| Segmentation mAP@0.5:0.95 | **1.000000** |
| Precision, score >=0.5 and mask IoU >=0.5 | **1.000000** (7 TP, 0 FP) |
| Recall at the same operating point | **1.000000** (0 FN) |

AP uses official `pycocotools==2.0.10` COCOeval with `iouType="segm"`, standard 101 recall points, mask IoU 0.50:0.05:0.95 and maxDets 100. All returned scored masks are retained at the frozen model floor **0.05** with NMS **0.5**. No measurement/operating gate filters AP. Actual output contains seven predictions, none below score 0.5. Predicted mask probability threshold is **0.5**. Operating precision/recall use score >=0.5 and descending-score per-image/category one-to-one matching to remaining GT at **mask IoU >=0.5**.

Mean GT mask IoU separately matches score >=0.5 predictions one-to-one to remaining same-class GT for any positive mask overlap, sums their IoUs and divides by **all seven non-crowd GT instances**, with unmatched GT zero. All seven GT matched here. Per-image IoU ranges from **0.958935345 to 0.974587348**. Inputs/GT masks use the unchanged aspect-preserving 512/960 transform, becoming **512 x 910** with exact nearest-neighbour masks and no augmentation. These are resized-coordinate mask metrics; no box AP or confidence average is substituted.

Checkpoint SHA-256: `f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39`. Frozen `dataset/splits/test.txt` SHA-256: `542bf7ec62522c86877d99707d6111e98dd516c4ff10f342d054ac77f34a8da1`. [Manifest](../models/evaluations/frozen_test_01/data_manifest.json) binds all seven filenames/image bytes, annotations, approved metadata and split bytes to committed Git objects at `659f022b80ce6e25f41747262bd2290d51228c76`. No frozen source/run artifact was changed.

Evidence: [full configuration](../models/evaluations/frozen_test_01/configuration.json), [exact command](../models/evaluations/frozen_test_01/command.txt), [environment/versions](../models/evaluations/frozen_test_01/environment.json), [metrics](../models/evaluations/frozen_test_01/metrics.json), [raw COCO output](../models/evaluations/frozen_test_01/coco_eval.txt), [scored RLE predictions](../models/evaluations/frozen_test_01/predictions.json), [evaluated GT masks](../models/evaluations/frozen_test_01/ground_truth.json), [COCO parameters](../models/evaluations/frozen_test_01/coco_parameters.json), [raw COCO arrays](../models/evaluations/frozen_test_01/coco_arrays.npz), [per-image results](../models/evaluations/frozen_test_01/per_image.json), [verification](../models/evaluations/frozen_test_01/verification.json) and [artifact hashes](../models/evaluations/frozen_test_01/artifact_sha256.json). Saved-prediction replay reproduces both mask AP scores; independent binary-mask intersection/union reproduces mean IoU. All **76 tests** passed, including low-score AP retention and zero IoU for a missed GT at the operating point.

This is a small seven-image evaluation of one physical calculator, not evidence of generalization to other calculator products or millimetre accuracy. The original MO-1 target remains **Not run**.

### Raw versus undistorted measurement

Use the **same 10 original accuracy scenes**, ruler readings, frozen checkpoint and measurement settings for both modes. In each mode, detect the reference corners and calculate the contour geometry in that mode's own image coordinates. Normal inference always applies undistortion; bypassing it is a labelled comparison experiment.

For every scene, show hand width/height, raw predicted width/height, corrected predicted width/height, absolute and percentage errors, detection confidence and success/failure reason. Then show raw/corrected mean, median and worst errors side by side, with accepted counts and paired-valid count. Save all inputs and outputs under `measurement/` and link them here.

Report the observed difference honestly, including negligible or worse corrected results. This is an end-to-end comparison: undistortion may also change segmentation and reference detection, so the difference cannot automatically be attributed solely to scale geometry.

**Results/table/raw outputs: Not run.**

## 5. Reporting boundaries

The separate item 7 runtime benchmark, item 8 controlled breaking-condition study and item 9 trust-threshold/refusal-cost study are deliberately omitted under the agreed time limit. The two objectives above use measurements already collected for items 1 and 5-6. Make no unmeasured speed, robustness or calibrated-confidence claims.

Basic object-not-found and reference-not-found handling remains part of the required inference command. The decision record belongs inside `Plan.md`; completion status and evidence belong in `Definition_of_Done.md`.

Leave each experimental result as **Not run** until it exists. If a target is missed, preserve the original target, report the actual result and explain the limitation. Before final submission, replace pending equipment/board metadata and result placeholders with facts or explicitly declare what was not reached.
