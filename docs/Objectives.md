# Measure It - Objectives

**Status:** portrait camera calibration measured and joint dataset correction generated on 2026-11-05. The user approved all 72 corrected images and 100 instances on 2026-11-05; Mask R-CNN fine-tuning completed with validation-only checkpoint selection; the frozen seven-image mask evaluation is complete; the one-image measurement CLI is implemented; one real referenced-photo execution is recorded; the supplied ten-scene accuracy experiment is complete with 9 accepted / 1 refused and MO-1 not met. The supplied set includes nine new scenes and one reused demo. Paired raw/corrected evaluation is also complete: 7 jointly valid scenes, with skipping increasing paired width/height MAE by 1.752788 / 2.313448 mm and mixed per-scene effects. Original objective targets are unchanged.

**Branch:** `dev-test01`. **Units:** millimetres for physical dimensions; pixels for reprojection error.

## 1. Setup and current progress

| Field | Recorded setup |
| --- | --- |
| Object | One rigid calculator, chosen for its clear outer outline and repeatable handling; measure its complete outer body, excluding shadows, hands, individual buttons and the display. |
| Confirmed hand measurements | User corrected the earlier preliminary 75 mm width to **70 mm**, with height **160 mm**, before this experiment made predictions on 2026-11-06. The user confirmed these ruler readings apply to all ten placements; one rigid calculator, no invented variation. Width means shorter side; height longer side. |
| Reference | A **flat unmarked paper square, 50 x 50 mm**, with both outer edges measured using a millimetre ruler and explicitly confirmed by the user on 2026-11-05. Keep this same reference throughout evaluation; its plane must match the measured outline. |
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
| Calibration software | Original fit: system Python **3.11.0**, OpenCV **4.10.0** (`opencv-python==4.10.0.84`), NumPy **2.3.1**, Pillow **11.2.1**, seed **42**. Versions are pinned in `requirements.txt`; raw fit environment is in `calibration/results/run.json`. An isolated project `.venv` was subsequently created with the same pinned versions; the full project now has **88 passing tests** inside it and dependency checks passed. Ongoing commands use that environment explicitly. |

### Dataset correction and review before training

The original annotations live in **432 x 432 exports of distorted originals**. They remain preserved with those export images and raw captures. `preprocessing/prepare_dataset.py` now decodes each mask, reverses the declared export geometry, applies the same explicit camera-frame rotation as its original image, then uses the **same undistortion map for image and masks** from `calibration/camera.py`. Mask restoration uses exact nearest-neighbour resizing; lens remapping uses nearest-neighbour for masks and linear interpolation for images. Every output segmentation, bbox and area is regenerated from its corrected raster mask. No undistorted image is paired with unchanged source annotations.

The export README reports **auto-orientation and full-frame stretch resizing to 432 x 432**. Automated raw/export comparisons support no crop or additional quarter-turn in that export. The separate raw-to-camera normalization is **0 degrees for 24 portrait originals** and **90 degrees clockwise for 48 landscape originals**, explicitly confirmed by the user. All corrected arrays retain the calibration's **2296 x 4080** full frame and K; mismatches or unverified transforms are refused. The source calibration views and fitted parameters are unchanged.

Command: `.venv\Scripts\python.exe -m preprocessing.prepare_dataset`. Output: `dataset/corrected/`, generated separately from all original files. **72 images and 100 instances processed, zero failures**. Train/validation/test counts remain **50/15/7 images** and **75/18/7 instances**. Image, annotation and category IDs, filenames and `iscrowd` are retained. All **154 source snapshot hashes** and six original freeze checksums match after processing; copied split lists are byte-identical. All 100 corrected COCO RLEs match their saved binary PNGs exactly, with matching regenerated bbox/pixel-area values.

Evidence: [processing run and per-instance mapping](../dataset/corrected/run.json), [source preservation snapshot](../dataset/corrected/source_snapshot.json), [explicit geometry settings](../preprocessing/geometry.json), corrected COCO [train](../dataset/corrected/train/_annotations.coco.json), [validation](../dataset/corrected/valid/_annotations.coco.json) and [test](../dataset/corrected/test/_annotations.coco.json), and [manual-review gallery](../dataset/corrected/review.html). The script uses pinned `pycocotools==2.0.10` for lossless mask RLE and official bbox/area semantics.

**All 72 images passed manual review by user confirmation on 2026-11-05.** Four images retain multiple Calculator instances: training captures `200604` (23) and `200521` (4), validation `200536` (3) and `200523` (2). Their flags are retained as accepted by the user in [manual_review.csv](../dataset/corrected/manual_review.csv); all 100 instances remain, with no label/pixel edits requested. [review_approval.json](../dataset/corrected/review_approval.json) records the user attestation, timestamp and reviewed-artifact hashes. Current run metadata sets `manual_review_complete=true` and `ready_for_training=true`; [processing_run.json](../dataset/corrected/processing_run.json) preserves the original pre-review processing state. Restoring a 432 x 432 raster cannot recover lost boundary precision. User acceptance does not establish trained-model performance or millimetre accuracy.

**Planning alignment, 2026-11-05:** the Plan now records the paper-square reference, joint correction of existing annotations and verified 50/15/7 split. The reference's 50 x 50 mm outer edges were subsequently confirmed by the user using a millimetre ruler; see the inference-stage record below. Filename lists accompany this initial planning snapshot before implementation code. This inventory check is preparation, not evidence of calibrated images, reviewed masks or training results.

## 2. MO-1 - Measurement accuracy

| Field | Commitment |
| --- | --- |
| What is measured | Error in the calculator's predicted outer width and height against hand measurements. |
| How | Use at least **10 additional photographed placements**, separate from train/validation/test. For each placement, record ruler readings and run the frozen calibrated image-to-mm pipeline. Save every input, overlay and machine-readable output. |
| Target | Corrected width MAE **<=3 mm** and corrected height MAE **<=3 mm**; worst individual dimensional error **<=8 mm**. All 10 placements must return valid measurements for a complete pass. |
| Conditions | Same camera geometry as calibration; calculator and reference fully visible; nearly overhead view; adequate light and sharp boundaries. Mount the paper reference flat on a rigid support without changing its dimensions. Its plane must match the outline being measured; support its height if needed. Calculator thickness remains a limitation. |
| Not included | Thickness measurement, unseen calculator models, arbitrary camera settings, and precision manufacturing inspection. Missing measurements are reported explicitly. |
| Why this target | The original target used the preliminary approximate 75 x 160 mm calculator estimate; the later ruler correction is 70 x 160 mm and the target is unchanged. A few millimetres remains meaningfully stricter than a rough visual estimate. The ruler's uncertainty limits the strength of the accuracy claim. |
| Result and evidence | **Not met.** Frozen corrected pipeline: 9 accepted / 1 refused; accepted-only width MAE **11.843224 mm**, height MAE **10.962348 mm**; worst width/height error **17.836929 / 18.858841 mm**. The supplied set has nine new scenes plus the earlier demo, so ten genuinely new placements are not available. Full table and evidence below. |

For each accepted dimension:

`absolute_error_mm = abs(predicted_mm - ruler_mm)`

`percentage_error = 100 * absolute_error_mm / ruler_mm`

Compute width and height MAE separately. Report mean percentage error, median and worst absolute/percentage errors for both dimensions, plus accepted and failed counts. Do not treat a failed measurement as zero error or hide it by removing the image. If fewer than 10 valid measurements are available, report accepted-only statistics with their denominator and mark the complete target unmet or not reached.

The experiment repeats **one physical calculator across different placements**. It does not establish accuracy across different calculator products. Record actual hand readings for each placement; do not invent variation in the size of a rigid object.

### Supplied placement experiment - 2026-11-06

The first four stages were working before the experiment: saved calibration, approved jointly corrected labels, a frozen trained/evaluated checkpoint and the one-image measurement CLI. The existing 88 tests passed; the five new coverage/error regressions brought the suite to **93 passing tests** before the batch. See the [experiment report](../measurement/placements_10/README.md), [manifest](../measurement/placements_10/manifest.json), [pre-recorded hand readings](../measurement/placements_10/hand_readings.csv) and [verification](../measurement/placements_10/verification.json).

The user confirmed **70 x 160 mm calculator dimensions**, measured with a millimetre ruler, for every placement, and **50 x 50 mm paper edges**. The 70 mm correction replaced the earlier 75 mm estimate **before any prediction/comparison in this batch**. Locked focus/zoom and matching reference/outline plane were user-confirmed; paper flatness, support height and thickness effects are not independently verified. All ten original files are preserved byte-for-byte under `measurement/placements_10/inputs/`; no content hash overlaps the 217 checked original/export/corrected dataset files. Frozen train/validation/test membership remains 50/15/7.

**Scope/freshness:** one physical calculator across ten supplied photographed placements. Nine captures are new; `calc.jpg.jpeg` (case_01) is the earlier demo used during paper-detector debugging. It is included transparently, not called a new blind scene. These observations do not establish generalization across calculator products or fulfill the ten genuinely new scenes requirement.

```powershell
.venv\Scripts\python.exe -m measurement.evaluate_placements --manifest measurement/placements_10/manifest.json --output measurement/placements_10/results --device cuda
```

The wrapper invokes the unchanged normal `inference.measure` CLI for every image, using mandatory full-frame undistortion, checkpoint SHA-256 `f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39`, inference-config SHA-256 `42429a62c08f51e8b7c856565fa932c5c9781c0464c35cc79fe30f00e5217d9a`, frozen score/mask thresholds 0.5 and aspect-preserving 512/960 geometry. [Exact commands](../measurement/placements_10/results/commands.txt), [raw JSON/CSV](../measurement/placements_10/results/per_instance.csv), [full summary](../measurement/placements_10/results/summary.json) and [all overlay previews](../measurement/placements_10/overlay_contact_sheet.jpg) are saved. No model, calibration, threshold or reference-setting changes were made using these results.

W/H below mean shorter/longer side. Values are rounded for display; the CSV retains full precision. Accepted/refused means the frozen command's status, not proof of accuracy. A refused dimension/error is missing, never zero.

| Image ID / input | Hand W / H (mm) | Predicted W / H (mm) | Absolute W / H error (mm) | W / H error (%) | Confidence | Accepted/refused | Reason / overlay |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| [case_01 (reused demo)](../measurement/placements_10/results/../inputs/calc.jpg.jpeg) | 70 / 160 | 80.466 / 170.600 | 10.466 / 10.600 | 14.952 / 6.625 | 0.995488 | Accepted (success) | None; [overlay](../measurement/placements_10/results/cases/case_01/overlay.jpg) |
| [case_02](../measurement/placements_10/results/../inputs/IMG_20261106_155037.jpg.jpeg) | 70 / 160 | 85.474 / 173.548 | 15.474 / 13.548 | 22.105 / 8.467 | 0.996183 | Accepted (success) | None; [overlay](../measurement/placements_10/results/cases/case_02/overlay.jpg) |
| [case_03](../measurement/placements_10/results/../inputs/IMG_20261106_155047.jpg.jpeg) | 70 / 160 | 73.780 / 162.841 | 3.780 / 2.841 | 5.400 / 1.776 | 0.994135 | Accepted (success) | None; [overlay](../measurement/placements_10/results/cases/case_03/overlay.jpg) |
| [case_04](../measurement/placements_10/results/../inputs/IMG_20261106_155109.jpg.jpeg) | 70 / 160 | 83.244 / 169.978 | 13.244 / 9.978 | 18.920 / 6.236 | 0.996307 | Accepted (success) | None; [overlay](../measurement/placements_10/results/cases/case_04/overlay.jpg) |
| [case_05](../measurement/placements_10/results/../inputs/IMG_20261106_155117.jpg.jpeg) | 70 / 160 | 86.165 / 176.596 | 16.165 / 16.596 | 23.093 / 10.372 | 0.996455 | Accepted (success) | None; [overlay](../measurement/placements_10/results/cases/case_05/overlay.jpg) |
| [case_06](../measurement/placements_10/results/../inputs/IMG_20261106_155128.jpg.jpeg) | 70 / 160 | 87.837 / 178.859 | 17.837 / 18.859 | 25.481 / 11.787 | 0.991796 | Accepted (success) | None; [overlay](../measurement/placements_10/results/cases/case_06/overlay.jpg) |
| [case_07](../measurement/placements_10/results/../inputs/IMG_20261106_155143.jpg.jpeg) | 70 / 160 | missing | missing | missing | 0.993914 | Refused (reference_not_found) | No isolated bright unmarked square boundary passed the configured checks; [overlay](../measurement/placements_10/results/cases/case_07/overlay.jpg) |
| [case_08](../measurement/placements_10/results/../inputs/IMG_20261106_155147.jpg.jpeg) | 70 / 160 | 84.820 / 171.404 | 14.820 / 11.404 | 21.172 / 7.128 | 0.996336 | Accepted (success) | None; [overlay](../measurement/placements_10/results/cases/case_08/overlay.jpg) |
| [case_09](../measurement/placements_10/results/../inputs/IMG_20261106_155827.jpg.jpeg) | 70 / 160 | 75.220 / 168.353 | 5.220 / 8.353 | 7.457 / 5.221 | 0.995801 | Accepted (success) | None; [overlay](../measurement/placements_10/results/cases/case_09/overlay.jpg) |
| [case_10](../measurement/placements_10/results/../inputs/IMG_20261106_155829.jpg.jpeg) | 70 / 160 | 79.583 / 166.482 | 9.583 / 6.482 | 13.690 / 4.051 | 0.994820 | Accepted (success) | None; [overlay](../measurement/placements_10/results/cases/case_10/overlay.jpg) |

| Accepted-only statistic (n=9 per dimension) | Width | Height |
| --- | ---: | ---: |
| MAE (mm) | 11.843224 | 10.962348 |
| Mean percentage error (%) | 16.918891 | 6.851468 |
| Median absolute error (mm) | 13.243668 | 10.600113 |
| Median percentage error (%) | 18.919525 | 6.625071 |
| Worst absolute error (mm) | 17.836929 | 18.858841 |
| Worst percentage error (%) | 25.481328 | 11.786776 |

**Coverage: 9 accepted / 1 refused / 10 supplied (90%).** Every statistic above has denominator **9 accepted width values or 9 accepted height values**. Case_07 (`IMG_20261106_155143.jpg.jpeg`) was refused with `reference_not_found`, because no isolated bright unmarked square boundary passed the configured checks; calculator confidence was 0.993914. Its numeric dimensions remain null and its CSV errors are blank. The nine genuinely new scenes contain **8 accepted / 1 refused**; their supplementary accepted-only MAE is **12.015330 mm width / 11.007627 mm height**, denominator 8 per dimension.

**MO-1 verdict: Not met.** Both MAEs exceed 3 mm, worst dimensional errors exceed 8 mm, one supplied scene lacks a measurement, and only nine scenes are genuinely new. All nine accepted outputs overestimate both confirmed ruler dimensions. Confidence does not establish scale accuracy. Possible mask-boundary, reference-scale/corner and physical-plane/flatness effects are not isolated by this experiment. No settings were changed to improve these figures. The paired raw-versus-undistorted comparison is completed below, with mode-specific coverage and a seven-scene paired estimate.

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

**Interpretation and limits.** MO-2 is met for this in-sample portrait fit. The board appears mostly central and flatness/rigid backing are not independently verified; paper curvature and extrapolation toward frame edges may bias fitted distortion. Focus/zoom lock is user-confirmed, not established by EXIF. Dataset preprocessing now explicitly resolves export geometry and the user-confirmed landscape-to-portrait rotation before invoking the unchanged strict undistorter; applying these intrinsics directly to an incompatible array remains invalid. **MO-1 is not met in the subsequent supplied-placement experiment** above; reprojection error alone does not establish physical accuracy.

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

[verification.json](../models/runs/initial_15_epochs/verification.json) confirms the hash, offline CPU reconstruction with strict state loading and both two-class heads, 400 finite update records, unchanged runtime source hashes, and **227** unchanged approved train/validation artifact hashes plus frozen lists. Binary checkpoints are ignored by Git; the selected checkpoint and checksum are published in the public [calculator-maskrcnn-v1 release](https://github.com/Khizar7786/camera-calibrated-object-measurement/releases/tag/calculator-maskrcnn-v1). Anonymous download access and GitHub's matching checkpoint SHA-256 were verified on 2026-11-06; the [root README](../README.md#download-the-selected-checkpoint) gives download/hash-check commands.

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

This is a small seven-image evaluation of one physical calculator, not evidence of generalization to other calculator products or millimetre accuracy. The subsequent supplied-placement experiment above reports MO-1 **Not met**; test mask AP does not establish physical accuracy.

### One-image measurement implementation - 2026-11-05

```powershell
.venv\Scripts\python.exe -m inference.measure path/to/new_original.jpg --output inference/runs/my_photo
```

[Configuration](../inference/config.json) records the user's ruler-measured **50 x 50 mm unmarked paper square**, the frozen epoch-2 checkpoint/hash and the exact saved training configuration. Normal inference always applies the existing strict full-frame undistorter after explicit camera-frame normalization. The predicted mask is restored into that corrected frame; the paper's outer corners are detected, ordered and refined there. In [measure_mask_contour_mm](../measurement/planar.py#L11), the actual mask contour is projected through the reference homography **before** fitting a metric oriented rectangle. Width is the shorter side and height the longer. No model box dimensions are used.

The [CLI report](../inference/README.md) defines coordinates, fixed detector checks, outputs and scene conditions. JSON records width/height, detection confidence, physical reference/corners, checkpoint/configuration/calibration IDs and failure reason. `object_not_found` and `reference_not_found` return null object dimensions with a saved corrected-frame overlay when geometry is valid. Wrong image geometry is refused; no normal raw/bypass mode exists. A plain square detector cannot prove semantic identity or physical coplanarity; ambiguity is refused. Paper/calculator-plane offset and thickness remain limitations.

[Verification](../inference/verification.json): **87 tests pass**, including known 75 x 160 mm projective contour recovery, rotated/plain-square detection, preview restoration, mask-versus-box geometry and synthetic success. Real frozen-network checks save object-not-found (local output: `inference/runs/object_missing_verified/result.json`) on an explicitly synthetic empty input and reference-not-found (local output: `inference/runs/reference_missing_verified/result.json`) on the unassigned original capture outside all frozen splits. The latter detects the calculator at score **0.996219933** and correctly supplies no dimensions. Full corrected overlays are saved. These are implementation/failure demonstrations, not physical accuracy measurements. The initial implementation had no real referenced photo; the subsequent supplied-photo result is recorded below. At that implementation stage MO-1 had not run; the later supplied-placement experiment above now reports **Not met**.

### First real referenced-photo execution - 2026-11-05

The supplied desktop `calc.jpg.jpeg` initially failed paper detection. Inspection found a visible paper square but no closed Canny paper contour against the wood-grain background. Bright-region proposals were added while retaining all existing reference-acceptance checks, and overlapping proposals are ranked by photometric quality rather than area alone. The same-image rerun (local output: `inference/runs/my_photo_retry/result.json`) succeeds with **80.466370 mm width / 170.600113 mm height**, score **0.995487988**, and mandatory undistortion. Overlay (local output: `inference/runs/my_photo_retry/overlay.jpg`), geometry (local output: `inference/runs/my_photo_retry/geometry.json`), verification (local output: `inference/runs/my_photo_retry/verification.json`) and hashes are retained. The original failed result is preserved. All **88 tests pass**; checkpoint/calibration/segmentation thresholds remain unchanged, and no held-out test photo was used for this fix.

This records execution success on one supplied scene. Paper/calculator-outline coplanarity is not verified, and the placement experiment had not run at that stage. Its subsequent results above report MO-1 **Not met**; one-photo execution success does not establish accuracy.

### Raw versus undistorted measurement - completed 2026-11-06

The same ten original accuracy scenes and pre-recorded **70 x 160 mm calculator / 50 x 50 mm paper** readings were run afresh through both modes. The [comparison declaration](../measurement/paired_undistortion_10/manifest.json) freezes the same checkpoint, effective configuration, 0.5 score/mask thresholds, aspect-preserving 512/960 sizing, seed 42 and detector/conversion settings. Normal inference remains unchanged and always undistorts. A separately labelled experimental raw CLI requires `--experimental-skip-undistortion`. Raw reference corners and raw mask contours are detected afresh in raw pixels; corrected corners/masks are not reused. Both invoke the exact same metric-contour conversion before rectangle fitting.

```powershell
.venv\Scripts\python.exe -m measurement.compare_undistortion --manifest measurement/paired_undistortion_10/manifest.json --output measurement/paired_undistortion_10/results --device cuda
```

[Full report](../measurement/paired_undistortion_10/README.md), [20-row mode CSV](../measurement/paired_undistortion_10/results/per_mode.csv), [10-row paired CSV](../measurement/paired_undistortion_10/results/paired.csv), [summary](../measurement/paired_undistortion_10/results/summary.json), [exact commands](../measurement/paired_undistortion_10/results/commands.txt), [verification](../measurement/paired_undistortion_10/verification.json) and [paired overlays](../measurement/paired_undistortion_10/paired_overlay_contact_sheet.jpg) preserve every input reference, output, error and refusal. The prior accuracy run and original source photos remain byte-identical. All ten corrected results, scores and masks exactly reproduce that prior run. **100 tests pass**.

The set repeats one physical calculator, with **nine new scenes plus the earlier reused demo (case_01)**. The seven paired-valid accuracy scenes include that demo; these are not the separate seven-image segmentation test split. Camera/plane conditions are user-confirmed, not independently measured.

W/H are shorter/longer dimensions. Refusals and unpaired deltas are missing, never zero. Delta means **raw absolute error minus corrected absolute error**; positive means skipping worsened error. Tables round for display; CSV/JSON retain full precision.

| Scene / original | Hand W / H (mm) | Corrected W / H (mm) | Raw W / H (mm) | Confidence C / R | Paired valid | Refusal reason |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| [case_01 (reused demo)](../measurement/placements_10/inputs/calc.jpg.jpeg) | 70 / 160 | [80.466 / 170.600; success](../measurement/paired_undistortion_10/results/corrected/case_01/result.json) | [80.872 / 174.823; success](../measurement/paired_undistortion_10/results/raw/case_01/result.json) | 0.995488 / 0.995532 | Yes | None |
| [case_02](../measurement/placements_10/inputs/IMG_20261106_155037.jpg.jpeg) | 70 / 160 | [85.474 / 173.548; success](../measurement/paired_undistortion_10/results/corrected/case_02/result.json) | [86.657 / 175.673; success](../measurement/paired_undistortion_10/results/raw/case_02/result.json) | 0.996183 / 0.996047 | Yes | None |
| [case_03](../measurement/placements_10/inputs/IMG_20261106_155047.jpg.jpeg) | 70 / 160 | [73.780 / 162.841; success](../measurement/paired_undistortion_10/results/corrected/case_03/result.json) | [missing; reference_not_found](../measurement/paired_undistortion_10/results/raw/case_03/result.json) | 0.994135 / 0.994413 | No | raw: Multiple plausible paper squares; reference is ambiguous |
| [case_04](../measurement/placements_10/inputs/IMG_20261106_155109.jpg.jpeg) | 70 / 160 | [83.244 / 169.978; success](../measurement/paired_undistortion_10/results/corrected/case_04/result.json) | [80.165 / 166.869; success](../measurement/paired_undistortion_10/results/raw/case_04/result.json) | 0.996307 / 0.996198 | Yes | None |
| [case_05](../measurement/placements_10/inputs/IMG_20261106_155117.jpg.jpeg) | 70 / 160 | [86.165 / 176.596; success](../measurement/paired_undistortion_10/results/corrected/case_05/result.json) | [83.229 / 172.005; success](../measurement/paired_undistortion_10/results/raw/case_05/result.json) | 0.996455 / 0.996496 | Yes | None |
| [case_06](../measurement/placements_10/inputs/IMG_20261106_155128.jpg.jpeg) | 70 / 160 | [87.837 / 178.859; success](../measurement/paired_undistortion_10/results/corrected/case_06/result.json) | [88.487 / 179.628; success](../measurement/paired_undistortion_10/results/raw/case_06/result.json) | 0.991796 / 0.992108 | Yes | None |
| [case_07](../measurement/placements_10/inputs/IMG_20261106_155143.jpg.jpeg) | 70 / 160 | [missing; reference_not_found](../measurement/paired_undistortion_10/results/corrected/case_07/result.json) | [missing; reference_not_found](../measurement/paired_undistortion_10/results/raw/case_07/result.json) | 0.993914 / 0.993867 | No | corrected: No isolated bright unmarked square boundary passed the configured checks; raw: No isolated bright unmarked square boundary passed the configured checks |
| [case_08](../measurement/placements_10/inputs/IMG_20261106_155147.jpg.jpeg) | 70 / 160 | [84.820 / 171.404; success](../measurement/paired_undistortion_10/results/corrected/case_08/result.json) | [97.647 / 185.033; success](../measurement/paired_undistortion_10/results/raw/case_08/result.json) | 0.996336 / 0.996286 | Yes | None |
| [case_09](../measurement/placements_10/inputs/IMG_20261106_155827.jpg.jpeg) | 70 / 160 | [75.220 / 168.353; success](../measurement/paired_undistortion_10/results/corrected/case_09/result.json) | [78.439 / 171.502; success](../measurement/paired_undistortion_10/results/raw/case_09/result.json) | 0.995801 / 0.996708 | Yes | None |
| [case_10](../measurement/placements_10/inputs/IMG_20261106_155829.jpg.jpeg) | 70 / 160 | [79.583 / 166.482; success](../measurement/paired_undistortion_10/results/corrected/case_10/result.json) | [missing; reference_not_found](../measurement/paired_undistortion_10/results/raw/case_10/result.json) | 0.994820 / 0.995056 | No | raw: No isolated bright unmarked square boundary passed the configured checks |

| Scene | Corrected absolute W / H error (mm) | Raw absolute W / H error (mm) | Corrected W / H error (%) | Raw W / H error (%) | Raw minus corrected absolute W / H error (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| case_01 | 10.466 / 10.600 | 10.872 / 14.823 | 14.952 / 6.625 | 15.531 / 9.264 | 0.405 / 4.223 |
| case_02 | 15.474 / 13.548 | 16.657 / 15.673 | 22.105 / 8.467 | 23.796 / 9.795 | 1.183 / 2.125 |
| case_03 | 3.780 / 2.841 | missing | 5.400 / 1.776 | missing | missing |
| case_04 | 13.244 / 9.978 | 10.165 / 6.869 | 18.920 / 6.236 | 14.522 / 4.293 | -3.078 / -3.109 |
| case_05 | 16.165 / 16.596 | 13.229 / 12.005 | 23.093 / 10.372 | 18.898 / 7.503 | -2.936 / -4.591 |
| case_06 | 17.837 / 18.859 | 18.487 / 19.628 | 25.481 / 11.787 | 26.410 / 12.268 | 0.650 / 0.770 |
| case_07 | missing | missing | missing | missing | missing |
| case_08 | 14.820 / 11.404 | 27.647 / 25.033 | 21.172 / 7.128 | 39.496 / 15.645 | 12.827 / 13.628 |
| case_09 | 5.220 / 8.353 | 8.439 / 11.502 | 7.457 / 5.221 | 12.055 / 7.189 | 3.219 / 3.149 |
| case_10 | 9.583 / 6.482 | missing | 13.690 / 4.051 | missing | missing |

#### All-valid statistics, each mode's own accepted denominator

Corrected **9 accepted / 1 refused (90%)**; raw **7 accepted / 3 refused (70%)**, from the same ten supplied scenes. Raw refuses case_03 as ambiguous, and cases 07 and 10 because no paper candidate passes; corrected refuses case_07. These statistics use different accepted sets, so their mean difference is not the primary paired effect.

| Statistic | Corrected width | Raw width | Corrected height | Raw height |
| --- | ---: | ---: | ---: | ---: |
| Accepted denominator per dimension | 9 | 7 | 9 | 7 |
| MAE (mm) | 11.843224 | 15.070784 | 10.962348 | 15.076059 |
| Mean percentage error (%) | 16.918891 | 21.529692 | 6.851468 | 9.422537 |
| Median absolute error (mm) | 13.243668 | 13.228676 | 10.600113 | 14.822754 |
| Median percentage error (%) | 18.919525 | 18.898108 | 6.625071 | 9.264221 |
| Worst absolute error (mm) | 17.836929 | 27.647324 | 18.858841 | 25.032730 |
| Worst percentage error (%) | 25.481328 | 39.496177 | 11.786776 | 15.645456 |

#### Paired-valid statistics, identical seven scenes

Paired IDs: **01, 02, 04, 05, 06, 08, 09**. Denominator **7 per dimension per mode**, with paired coverage **7/10 supplied scenes**. Unpaired scenes remain in the tables and refusal counts.

| Statistic | Corrected width | Raw width | Corrected height | Raw height |
| --- | ---: | ---: | ---: | ---: |
| Accepted denominator per dimension | 7 | 7 | 7 | 7 |
| MAE (mm) | 13.317996 | 15.070784 | 12.762610 | 15.076059 |
| Mean percentage error (%) | 19.025709 | 21.529692 | 7.976631 | 9.422537 |
| Median absolute error (mm) | 14.820351 | 13.228676 | 11.404297 | 14.822754 |
| Median percentage error (%) | 21.171929 | 18.898108 | 7.127686 | 9.264221 |
| Worst absolute error (mm) | 17.836929 | 27.647324 | 18.858841 | 25.032730 |
| Worst percentage error (%) | 25.481328 | 39.496177 | 11.786776 | 15.645456 |

| Paired error-change statistic (raw minus corrected) | Width | Height |
| --- | ---: | ---: |
| Mean change (mm) | 1.752788 | 2.313448 |
| Median change (mm) | 0.649925 | 2.124695 |
| Minimum change (mm) | -3.078323 | -4.591156 |
| Maximum change (mm) | 12.826973 | 13.628433 |
| Mean percentage-error change (percentage points) | 2.503983 | 1.445905 |
| Scenes worse when skipping | 5 | 5 |
| Scenes better when skipping | 2 | 2 |

**Measured effect of skipping:** paired width MAE rises **13.317996 -> 15.070784 mm**, a **+1.752788 mm** change; height MAE rises **12.762610 -> 15.076059 mm**, a **+2.313448 mm** change. Mean percentage error increases **2.503983 width / 1.445905 height percentage points**. Optional pooled MAE over **14 paired dimensions** rises **13.040303 -> 15.073421 mm (+2.033118 mm)**; it is not a per-dimension result.

**Mixed effects:** skipping improves cases **04 and 05** in both dimensions and worsens the other five pairs. Raw paired median width absolute error is lower (**13.228676 mm raw / 14.820351 mm corrected**) despite its higher mean. Case_08 has the largest raw-minus-corrected worsening (**+12.826973 mm width / +13.628433 mm height**) and is retained. Both modes miss the accuracy target; normal corrected MO-1 remains **Not met**.

This is an **end-to-end comparison**: segmentation, corner detection/refinement and refusal coverage can change with the input. All raw/corrected mask files differ in their respective pixel frames; that is not an aligned-mask accuracy metric or an isolated segmentation contribution. The experiment does not apportion the error change between geometric scale, mask boundary and reference-corner effects. No data or settings were edited to force an improvement. Ruler/plane/flatness/thickness limitations and the one-calculator/reused-demo scope remain.

Verified numerical reference for an answer in your own words, without reading notes; recording completion remains unverified:

> On the seven accuracy scenes measured by both modes, skipping undistortion increased width MAE from 13.318 to 15.071 mm, a worsening of 1.753 mm, and height MAE from 12.763 to 15.076 mm, a worsening of 2.313 mm. Corrected mode measured 9 of 10 supplied scenes; raw mode measured 7 of 10. The effect was mixed: skipping improved both dimensions in cases 04 and 05, but worsened them in the other five pairs. This is an end-to-end result: segmentation and reference detection also change, so I cannot attribute the entire difference to geometric scale.

## 5. Reporting boundaries

The separate item 7 runtime benchmark, item 8 controlled breaking-condition study and item 9 trust-threshold/refusal-cost study are deliberately omitted under the agreed time limit. The two objectives above use measurements already collected for items 1 and 5-6. Make no unmeasured speed, robustness or calibrated-confidence claims.

Basic object-not-found and reference-not-found handling remains part of the required inference command. The decision record belongs inside `Plan.md`; completion status and evidence belong in `Definition_of_Done.md`.

Leave each experimental result as **Not run** until it exists. If a target is missed, preserve the original target, report the actual result and explain the limitation. Before final submission, replace pending equipment/board metadata and result placeholders with facts or explicitly declare what was not reached.
