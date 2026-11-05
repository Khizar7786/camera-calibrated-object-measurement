# Measure It

Camera-calibrated calculator segmentation and metric measurement assessment, on branch `dev-test01`. The planning documents and fixed split membership were committed before implementation. **Camera calibration is implemented; dataset correction, training and millimetre measurement are pending.**

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

The selected geometry is **2296 x 4080 stored pixels, portrait**: 21 views were usable and all 9 landscape 4080 x 2296 views were rejected before fitting. Global RMS was **0.918653470 px**; the <=1.0 px fit target was met. These intrinsics do not apply directly to landscape captures or the 432 x 432 stretched dataset exports. The next stage must resolve that geometry explicitly; the shared function rejects it.

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

An array's shape cannot reveal a same-size crop/rotation or altered focus/zoom. Those capture conditions remain the caller's responsibility. The preprocessing and inference pipelines themselves are not yet implemented.

## Dataset and project documents

The recorded setup uses one approximately 75 x 160 mm calculator and a paper-square reference reported as 50 x 50 mm, to be physically confirmed before measurement. Object photographs are user-provided phone captures, annotated through Roboflow's tool and exported as COCO segmentation. Annotation review is pending; no Roboflow model is used.

The supplied export has **72 images: 50 train / 15 validation / 7 test**. Exact membership is frozen in [dataset/splits](dataset/splits/README.md). There are 73 raw object photographs; the extra capture remains unassigned. Preserve the test split and all derivatives from training, tuning and checkpoint selection. Do not change membership when resolving export geometry.

The three assessment documents are [Plan](docs/Plan.md), [Objectives](docs/Objectives.md) and [Definition of Done](docs/Definition_of_Done.md). They distinguish measured evidence from work still pending.

## Verification

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On 2026-11-05 the isolated project `.venv` was created, the exact pinned packages were installed, all **17 tests passed inside it**, and `pip check` reported no broken requirements. Imports were verified to come from `.venv`, with system site-packages disabled. This verifies the environment and implementation contracts; the saved calibration fit above retains its original interpreter provenance.

Tests cover known synthetic camera recovery, the 20-view gate, per-corner RMS, unreadable views, stale-output protection, parameter validation, full-frame identity, nearest-neighbour masks, and resolution/EXIF mismatch rejection.
