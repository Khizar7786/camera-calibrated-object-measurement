# Measure It - Plan

**Status:** initial planning snapshot before implementation; no results claimed. **Branch:** `dev-test01`.

**Scope.** Complete items 1-6 and decision record 10; omit experiments 7-9 for time. Choose the rigid calculator for its clear outer outline and repeatable handling; approximate ruler dimensions are 75 x 160 mm. Width is the shorter side. Use the Redmi Note 12, a measured checkerboard (>=7 x 9 inner corners), and the paper-square reference (reported 50 x 50 mm). Verify its edges and support it rigidly in the measurement plane. Keep views nearly overhead; thickness limits planar accuracy.

**Method.** OpenCV calibration/shared undistortion; reviewed masks; pretrained Torchvision Mask R-CNN ResNet50-FPN fine-tuned on my photos. Detect the paper square's four corners, transform mask contours into millimetres and fit an oriented rectangle. Save overlay, dimensions and detection confidence. Handle object/reference not found without invented dimensions.

**Data/Git.** Commit this Plan, initial Objectives/Definition of Done and `dataset/splits/` before code. Preserve the existing 72-image membership: 50 train / 15 validation / 7 test; one extra raw capture stays unassigned. Existing Roboflow COCO labels are on distorted, stretched exports. Resolve export/original geometry, then jointly undistort images and masks; use nearest-neighbour mask resampling and review/repair overlays. Verify camera-setting consistency before calibration. Seed 42; choose checkpoints/cutoffs using validation only. Keep test untouched for tuning. Accuracy uses 10 additional, hand-measured placements.

| Order | Work | Active minutes |
| --- | --- | ---: |
| 1 | Documents, capture inventory, frozen lists, first planning commit | 65 |
| 2 | Calibrate >=20 views; save parameters and reprojection errors | 40 |
| 3 | Resolve export geometry; jointly undistort images/masks and review labels | 75 |
| 4 | Fine-tune; report test mask IoU, mAP@0.5, mAP@0.5:0.95, precision and recall | 45 |
| 5 | One-command raw-photo-to-mm inference and failure handling | 65 |
| 6 | Ten placements: accuracy and paired raw/corrected errors (items 5-6) | 85 |
| 7 | Reproduce; finish three documents, README, <=5-minute Loom and PR | 105 |
| **Total** | **8 active hours; submit within 24 hours of actual start** | **480** |

**Required objectives.** MO-1: width/height MAE each <=3 mm, worst error <=8 mm, across all 10 placements; failures prevent a complete pass. MO-2: global RMS reprojection error <=1.0 pixel across >=20 usable views. Report per-instance errors/percentage errors, medians, worst cases, failures and per-view calibration errors. Training/install waiting adds clock time, not active time. Track existing preparation and actual start honestly.

**Deliberately skipped.** Additional speed/performance benchmarking (7), breaking-condition experiment (8), trust-threshold/refusal-cost study (9), UI, deployment and architecture comparisons. Declare omissions and their impact in Definition of Done. Runtime, robustness and refusal coverage remain unquantified.

**Decision record (initial).** Choose Mask R-CNN for instance masks, detection scores and direct instance-metric evaluation. Reject U-Net/ResNet18, which needs instance extraction/scoring. Expected cost: higher memory demand and potentially slower training/inference. Comparative performance is unmeasured; update this record if the decision changes.

**2026-11-05, before implementation.** Corrected the reference, existing-label workflow and assumed 70-image count after checking the supplied files.

**2026-11-05, calibration stage.** Chose the 2296 x 4080 portrait profile: 21 compatible views, with 9 landscape views explicitly rejected. Board confirmed as 9 x 7 inner corners and user-measured 20 mm squares; focus/zoom lock user-confirmed. Keep the initial targets; resolve landscape/export geometry before dataset correction. No automatic rotation or intrinsic rescaling is permitted by the current shared routine.
