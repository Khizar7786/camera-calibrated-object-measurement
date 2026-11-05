# Measure It - Objectives

**Status:** portrait camera calibration measured on 2026-11-05; dataset correction, segmentation and millimetre measurement remain unrun. Original objective targets are unchanged.

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
| Camera consistency | User confirmed focus and zoom locked throughout calibration and calculator captures. All original EXIF records inspected share Xiaomi model 23021RAAEG, focal length 4.22 mm and orientation 1. Resolution/orientation is mixed: the fitted profile accepts only **2296 x 4080 stored pixels** and rejects other shapes. A same-size crop/rotation or optical change cannot be ruled out from array shape or EXIF alone. |
| Calibration photos | **30 originals** in `calibration/raw/`: **21 usable portrait views**, **9 landscape geometry rejections**, **0 detection failures among the 21 compatible inputs**. All accepted views have 63 full-resolution refined corners. Full outcomes and source hashes are saved in `calibration/results/views.csv`. |
| Calibration board | Complete checkerboard: **10 x 8 squares**, hence **9 x 7 inner corners**, confirmed from the whole board and printed footer. User physically measured square edges as **2 cm x 2 cm = 20 mm**. Board flatness/rigid backing cannot be verified from photographs and remains a limitation. |
| Object annotations | Own original phone photos, automatically annotated on Roboflow; COCO Segmentation dataset downloaded. Masks still require manual review and geometry correction. |
| Split policy | Preserve existing export membership: **50 train / 15 validation / 7 test = 72 images**, approximately 70/20/10. Exact dataset-relative paths are frozen in `dataset/splits/train.txt`, `valid.txt` and `test.txt`; see the split README for conventions and integrity checks. Commit these lists before code; do not reshuffle. |
| Original/export inventory | All 72 exports map uniquely to raw originals by capture filename. `dataset/Raw/` contains **73** photos; `IMG_20261104_200608.jpg` is unassigned and excluded from the frozen splits. Do not silently introduce it into training or evaluation. |
| Model | Pretrained Torchvision Mask R-CNN ResNet50-FPN, fine-tuned on the calculator class. |
| Reproducibility | Seed 42; exact package versions, training configuration, checkpoint choice, checkpoint SHA-256 and commands must be recorded when implemented. |
| Calibration machine | Intel Core i7-12650H; 16,831,713,280 bytes reported physical RAM (about 15.68 GiB); Intel UHD Graphics and NVIDIA GeForce RTX 3050 Ti Laptop GPU; Microsoft Windows 11 Home, build 10.0.26300. Calibration uses the CPU with one OpenCV thread. Exact CIM output: `calibration/results/hardware.json`. Future training hardware must be recorded if different. |
| Calibration software | Original fit: system Python **3.11.0**, OpenCV **4.10.0** (`opencv-python==4.10.0.84`), NumPy **2.3.1**, Pillow **11.2.1**, seed **42**. Versions are pinned in `requirements.txt`; raw fit environment is in `calibration/results/run.json`. An isolated project `.venv` was subsequently created with the same pinned versions; all **17 tests passed** inside it and dependency checks passed. Ongoing commands use that environment explicitly. |

### Preparation required before training

The existing labels belong to **raw, distorted photographs**. Preserve those originals and their COCO annotations. Calibration is now fitted for the portrait profile; dataset correction has not started. Resolve original/export geometry, decode annotations into aligned instance masks and apply the **same undistortion mapping to each image and its masks** using `calibration/camera.py`. Use nearest-neighbour resampling for categorical masks and review the corrected overlays. Relabel any boundaries that are no longer accurate. Correcting the image alone would misalign its labels. Landscape inputs and 432 x 432 exports are rejected by the current profile; do not silently rotate/resize them or reuse its intrinsics.

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

**Interpretation and limits.** MO-2 is met for this in-sample portrait fit. The board appears mostly central and flatness/rigid backing are not independently verified; paper curvature and extrapolation toward frame edges may bias fitted distortion. Focus/zoom lock is user-confirmed, not established by EXIF. The calibrated profile must not be applied to the landscape captures or 432 x 432 exports without separately resolving and verifying that geometry. **MO-1 millimetre accuracy remains Not run** and requires independent ruler-based measurements; reprojection error alone does not establish it.

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
