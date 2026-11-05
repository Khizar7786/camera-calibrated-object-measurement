# Measure It

Camera-calibrated calculator segmentation and metric measurement assessment, on branch `dev-test01`. The planning documents and fixed split membership were committed before implementation. **Camera calibration and joint image/mask correction are implemented. The user approved all 72 corrected images and 100 instances on 2026-11-05; model training and millimetre measurement are pending.**

## Setup

Recorded calibration environment: Python 3.11.0, OpenCV 4.10.0, NumPy 2.3.1 and Pillow 11.2.1 on Windows 11. Install the pinned dependencies in a local environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The commands below call the project environment's Python explicitly; activation is optional. The original recorded calibration run used `C:\Python311\python.exe`, with one OpenCV thread and seed 42. [Hardware metadata](calibration/results/hardware.json): Intel Core i7-12650H, 16,831,713,280 bytes reported physical RAM, Intel UHD Graphics and NVIDIA RTX 3050 Ti Laptop GPU; calibration uses the CPU. No runtime benchmark is claimed.

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

RMS is `sqrt(sum squared 2D corner residuals / corner count)`, computed globally and per view, and checked against OpenCV. It is an **in-sample camera-fit metric, not proof of millimetre accuracy**. Board curvature and limited edge coverage can bias distortion estimates even with low RMS. Separate ruler-based measurement validation remains required. The implementation follows the [OpenCV calibration API](https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html).

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

An array's shape cannot reveal a same-size crop/rotation or altered focus/zoom. Those capture conditions remain the caller's responsibility. Dataset preprocessing now calls this module after explicit geometry normalization; the inference pipeline is not yet implemented.

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

The recorded setup uses one approximately 75 x 160 mm calculator and a paper-square reference reported as 50 x 50 mm, to be physically confirmed before measurement. Object photographs are user-provided phone captures, annotated through Roboflow's tool and exported as COCO segmentation. The corrected annotations have passed user review; no Roboflow model is used.

The supplied export has **72 images: 50 train / 15 validation / 7 test**. Exact membership is frozen in [dataset/splits](dataset/splits/README.md) and retained in the corrected dataset. There are 73 raw object photographs; the extra capture remains unassigned. Preserve the test split and all derivatives from training, tuning and checkpoint selection. The corrected dataset is user-approved and has not yet been used for model fitting. `.gitattributes` preserves exact bytes of the source export and reviewed artifacts so their integrity hashes survive checkout on other platforms.

The three assessment documents are [Plan](docs/Plan.md), [Objectives](docs/Objectives.md) and [Definition of Done](docs/Definition_of_Done.md). They distinguish measured evidence from work still pending.

## Verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On 2026-11-05 the isolated project `.venv` was created with system site-packages disabled. The dependencies now include pinned `pycocotools==2.0.10` for mask handling. All **42 tests pass inside the environment**. The saved calibration fit retains its original interpreter provenance.

Tests cover known synthetic camera recovery, the 20-view gate, per-corner RMS, unreadable views, stale-output protection, parameter validation, full-frame identity, mask topology, source crop/rotation recovery, exact nearest-neighbour reconstruction, synthetic image/mask alignment, immutable sources/splits and unsafe-geometry refusal.
