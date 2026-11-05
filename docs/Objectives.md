# Measure It - Objectives

**Status:** initial document for the planning commit; experiments not run and no results claimed.

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
| Camera consistency | Lens, focus, zoom, resolution and orientation consistency between calibration and object captures: **pending verification**. |
| Calibration photos | **30 JPEG files present** in `Calibration/`. Usable corner-detection count and board suitability: **pending**; file count alone does not establish successful calibration. |
| Calibration board | Record actual inner-corner columns/rows and ruler-measured square spacing before fitting. The assessment requires at least 7 x 9 inner corners, allowing rotation of that arrangement. |
| Object annotations | Own original phone photos, automatically annotated on Roboflow; COCO Segmentation dataset downloaded. Masks still require manual review and geometry correction. |
| Split policy | Preserve existing export membership: **50 train / 15 validation / 7 test = 72 images**, approximately 70/20/10. Exact dataset-relative paths are frozen in `dataset/splits/train.txt`, `valid.txt` and `test.txt`; see the split README for conventions and integrity checks. Commit these lists before code; do not reshuffle. |
| Original/export inventory | All 72 exports map uniquely to raw originals by capture filename. `dataset/Raw/` contains **73** photos; `IMG_20261104_200608.jpg` is unassigned and excluded from the frozen splits. Do not silently introduce it into training or evaluation. |
| Model | Pretrained Torchvision Mask R-CNN ResNet50-FPN, fine-tuned on the calculator class. |
| Reproducibility | Seed 42; exact package versions, training configuration, checkpoint choice, checkpoint SHA-256 and commands must be recorded when implemented. |
| Training machine | CPU, GPU, RAM and operating-system version: **pending recording from the machine used for the experiments**. |

### Preparation required before training

The existing labels belong to **raw, distorted photographs**. Preserve those originals and their COCO annotations. After the planning commit, calibrate the camera, decode the annotations into instance masks and apply the **same undistortion mapping to each image and its masks**. Use nearest-neighbour resampling for categorical masks and review the corrected overlays. Relabel any boundaries that are no longer accurate. Correcting the image alone would misalign its labels.

The export README reports **auto-orientation and stretch resizing to 432 x 432**, confirmed by the exported image dimensions. Re-export at the original geometry or explicitly account for that transformation before applying calibration. Keep split membership unchanged throughout processing, recording the source-to-processed filename mapping. Calibration-board images themselves remain raw for corner detection and fitting. Mixed portrait/landscape captures require a consistent coordinate convention and camera-setting verification.

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
| Result and evidence | **Not run.** Add actual captured/usable counts, RMS, per-view errors, saved parameter path, command and met/missed verdict. |

Keep the global RMS definition consistent with OpenCV's calibration output. Record excluded views and reasons; do not remove images merely to hide a poor result. If fewer than 20 usable views remain, collect additional calibration views or declare the target incomplete.

## 4. Other required evaluation evidence

### Segmentation model

Freeze the checkpoint, preprocessing and model operating cutoffs using validation only. Evaluate the committed test split without retuning. Report:

| Metric | Result |
| --- | --- |
| Mask IoU, with aggregation defined and unmatched ground truth counted as zero | Not run |
| Segmentation mAP@0.5 | Not run |
| Segmentation mAP@0.5:0.95 | Not run |
| Precision at a declared score cutoff and mask IoU >=0.5 | Not run |
| Recall at the same operating point | Not run |

Use a standard COCO-compatible segmentation evaluator for AP. Save raw output, actual test-image count, configuration, checkpoint hash and exact command. Detection confidence must not be presented as guaranteed dimensional accuracy.

### Raw versus undistorted measurement

Use the **same 10 original accuracy scenes**, ruler readings, frozen checkpoint and measurement settings for both modes. In each mode, detect the reference corners and calculate the contour geometry in that mode's own image coordinates. Normal inference always applies undistortion; bypassing it is a labelled comparison experiment.

For every scene, show hand width/height, raw predicted width/height, corrected predicted width/height, absolute and percentage errors, detection confidence and success/failure reason. Then show raw/corrected mean, median and worst errors side by side, with accepted counts and paired-valid count. Save all inputs and outputs under `measurement/` and link them here.

Report the observed difference honestly, including negligible or worse corrected results. This is an end-to-end comparison: undistortion may also change segmentation and reference detection, so the difference cannot automatically be attributed solely to scale geometry.

**Results/table/raw outputs: Not run.**

## 5. Reporting boundaries

The separate item 7 runtime benchmark, item 8 controlled breaking-condition study and item 9 trust-threshold/refusal-cost study are deliberately omitted under the agreed time limit. The two objectives above use measurements already collected for items 1 and 5-6. Make no unmeasured speed, robustness or calibrated-confidence claims.

Basic object-not-found and reference-not-found handling remains part of the required inference command. The decision record belongs inside `Plan.md`; completion status and evidence belong in `Definition_of_Done.md`.

Leave each experimental result as **Not run** until it exists. If a target is missed, preserve the original target, report the actual result and explain the limitation. Before final submission, replace pending equipment/board metadata and result placeholders with facts or explicitly declare what was not reached.
