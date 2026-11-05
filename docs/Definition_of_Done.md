# Measure It - Definition of Done

**Status:** portrait calibration measured, joint image/mask correction generated, and all 72 images passed user review. Training, inference and physical accuracy evaluation remain pending; only evidence-backed completed items are checked.

**Branch:** `dev-test01`. **Experimental scope:** assessment items 1-6, with the decision record retained inside `Plan.md`.

Tick an item only when its evidence exists. Replace **Pending** with an actual result, saved artifact path, command output or commit/link. Leave unfinished work unchecked and explain why. A missed objective must be reported honestly; it does not justify changing the original target.

**Recorded setup:** Redmi Note 12; originals include both 4080 x 2296 and 2296 x 4080 orientations; exported dataset images are stretched to 432 x 432. Calibration accepts only the 2296 x 4080 profile. Focus/zoom lock and 90-degree clockwise landscape normalization were user-confirmed. Explicit export recovery, joint image/mask correction and user review are complete; board-flatness limitations remain recorded. Calculator approximately 75 mm wide and 160 mm high; paper-square reference reported as 50 x 50 mm; ground truth measured with a millimetre-marked ruler. Reference dimensions remain pending.

## 1. Planning and Git - before code

- [x] All three initial documents exist. The Plan matches the paper-square reference, joint image/mask correction workflow and verified counts. **Evidence:** [Plan](Plan.md), [Objectives](Objectives.md) and this checklist, aligned on 2026-11-05; no experimental completion claimed.
- [x] Exact filenames are frozen as **50 train / 15 validation / 7 test** without changing exported membership. Lists match COCO image entries and existing files; raw-source identities are disjoint. **Evidence:** [train](../dataset/splits/train.txt), [validation](../dataset/splits/valid.txt), [test](../dataset/splits/test.txt) and [inventory/checks](../dataset/splits/README.md), verified 2026-11-05. The Git-history check is recorded below.
- [x] The first planning commit on `dev-test01` contains the Plan and frozen filename lists before implementation code. **Evidence:** `git rev-list --max-parents=0 HEAD` identifies the root commit; `git show --name-only <root-commit>` lists nine planning/metadata files, including all three documents and split lists, with no implementation code. Committed split bytes pass the recorded SHA-256 checks. Verified 2026-11-05.
- [ ] Subsequent commits show genuine calibration, data, training and measurement progress, with descriptive reasons. **Evidence: Pending.**

## 2. Camera calibration

- [x] At least 20 usable original checkerboard views are included. Actual inner-corner grid and measured square size, captured/usable counts, and exclusions are recorded. **Evidence:** [board settings](../calibration/board.json), [30-view outcome table](../calibration/results/views.csv): 9 x 7 inner corners, user-measured 20 mm squares, 21 usable portrait views, 9 landscape geometry rejections; no error-based removals.
- [x] Supported camera/export geometry is explicitly recorded and verified before applying calibration. **Evidence:** focus/zoom user-confirmed; EXIF camera/focal-length/orientation matches; [geometry settings](../preprocessing/geometry.json) record full-frame export stretch and user-confirmed 90-degree clockwise landscape normalization. [Processing evidence](../dataset/corrected/run.json) verifies all 72 raw/export pairs; every shared-undistorter input is 2296 x 4080. Unconfigured/mismatched geometry is refused.
- [x] OpenCV calibration saves the camera matrix, distortion coefficients, calibration image size, board settings, global RMS and per-view reprojection errors. **Evidence:** [camera parameters](../calibration/results/camera.json), [raw fit](../calibration/results/raw_calibration.npz), [raw console](../calibration/results/console.txt) and [run metadata](../calibration/results/run.json).
- [x] Saved parameters load in one reusable full-frame undistortion module; before/after examples have been inspected. **Evidence:** [shared module](../calibration/camera.py), [comparison](../calibration/results/before_after/IMG_20261104_194240.jpg.comparison.jpg) and [contract tests](../tests/test_camera.py). Output size and K are retained, geometry/EXIF mismatches rejected, nearest-neighbour mapping available for masks. Black borders remain explicit; no mm-accuracy claim is made.
- [ ] Dataset preparation and normal inference call this same shared module on geometry-verified inputs. **Partial evidence:** [dataset preparation](../preprocessing/prepare_dataset.py) now calls it on geometry-verified images and masks. Normal inference is not yet implemented.
- [x] MO-2 is evaluated against RMS <=1.0 pixel with >=20 usable views. **Evidence:** [Objectives](Objectives.md): **0.918653470 px**, **21 usable views**, target **met as an in-sample fit metric only**. Board flatness and independent physical accuracy remain unverified.

## 3. Dataset and segmentation masks

- [x] At least 70 user-provided object photographs, original COCO annotations, actual counts and a source-to-processed filename mapping are available. **Evidence:** [run mapping](../dataset/corrected/run.json): 72 corrected images from 72 originals, 100 instances; [154-file source snapshot](../dataset/corrected/source_snapshot.json) verifies original preservation. The extra raw capture remains unassigned.
- [x] Existing annotations are decoded and jointly corrected with their images using explicit export/camera transforms and the same lens-undistortion map. Categorical masks use nearest-neighbour resampling. Segmentation RLE, bbox and pixel area are regenerated. **Evidence:** [preprocessing code](../preprocessing/prepare_dataset.py), corrected [train](../dataset/corrected/train/_annotations.coco.json), [validation](../dataset/corrected/valid/_annotations.coco.json), [test](../dataset/corrected/test/_annotations.coco.json) and saved per-instance mask PNGs. All 100 RLE/PNG masks and fresh geometry values match.
- [x] Corrected overlays are manually reviewed and accepted. **Evidence:** the user explicitly confirmed all reviewed items passed on 2026-11-05; all 72 entries in the [review gallery](../dataset/corrected/review.html) and [review queue](../dataset/corrected/manual_review.csv) are passed. [Approval record](../dataset/corrected/review_approval.json) binds that attestation to artifact hashes. Four multi-instance flags were accepted by the user; no label edits or automatic deletions were requested.
- [x] Semantic boundary and instance acceptance is recorded from user review; automated dimensions, binary-mask, nonempty-instance and annotation-reference checks pass. **Evidence:** [user approval](../dataset/corrected/review_approval.json), 72 passed review entries and 100 retained instances. This records user acceptance, not an independent model-performance or physical-accuracy claim.
- [x] Frozen membership survives export, conversion and undistortion without mixing test derivatives into other splits. **Evidence:** copied [train](../dataset/corrected/splits/train.txt), [validation](../dataset/corrected/splits/valid.txt), [test](../dataset/corrected/splits/test.txt) lists are byte-identical to the originals; all IDs/membership remain 50/15/7. No training, tuning or checkpoint selection has occurred.

## 4. Model training and evaluation

- [ ] Torchvision Mask R-CNN ResNet50-FPN is actually fine-tuned on my calculator data. No Ultralytics YOLO or Roboflow model is used as the submitted segmentation model. **Evidence: Pending.**
- [ ] Seed 42, exact package versions, model/training configuration, commands and training/validation logs are saved. CPU, GPU, RAM, OS and camera details are recorded in the final documentation. **Evidence: Pending.**
- [ ] Checkpoint selection and model operating cutoffs use validation only and are frozen before test evaluation. The selected checkpoint and its SHA-256 are recorded. **Evidence: Pending.**
- [ ] The committed test split is evaluated for mask IoU, segmentation mAP@0.5, segmentation mAP@0.5:0.95, precision and recall. Sample count, metric definitions, score cutoff, checkpoint/configuration and raw evaluator output are saved and linked in Objectives. **Evidence: Pending.**

## 5. One-command measurement

- [ ] One documented command accepts a new original camera photograph, loads calibration/weights, undistorts it and segments the calculator. **Evidence: Pending.**
- [ ] The actual reference edges are measured and its four corners detected in the measurement image. The paper is flat on rigid support, and its plane matches the measured calculator outline; thickness/plane limitations are documented. **Evidence: Pending.**
- [ ] The mask contour and reference corners use the same image coordinates. Conversion produces width as the shorter metric side and height as the longer; the actual conversion file/function/line can be identified. **Evidence: Pending.**
- [ ] The command saves a mask overlay and machine-readable width/height in mm, detection confidence and status. Confidence is described as a model score, not a guarantee of dimensional accuracy. **Evidence: Pending.**
- [ ] Object-not-found and reference-not-found cases are demonstrated with explicit failure reasons and no invented dimensions. **Evidence: Pending.**

## 6. Accuracy and raw/corrected comparison

- [ ] At least 10 additional calculator placements, separate from train/validation/test, have recorded ruler width/height readings, original photographs, predictions and saved raw outputs. The scope is repeatability on one physical calculator. **Evidence: Pending.**
- [ ] Every placement appears in a per-instance table with ground truth, predicted dimensions, absolute/percentage errors, confidence and success/failure status. Failed measurements are not hidden or counted as zero error. **Evidence: Pending.**
- [ ] Width and height MAE, mean percentage errors, medians and worst cases are reported with accepted/failed counts and denominators. MO-1's <=3 mm MAE per dimension and <=8 mm worst-error targets are marked met/missed/not reached; a complete pass requires all 10 valid measurements. **Evidence: Pending.**
- [ ] The same original scenes, ruler readings, checkpoint and settings are evaluated in raw and undistorted modes. Each mode computes its reference/contour geometry in its own coordinate system. **Evidence: Pending.**
- [ ] Raw/corrected per-instance errors and summary errors are shown side by side, with failure states and paired-valid counts. The actual numerical difference and limitations are explained in Objectives. **Evidence: Pending.**

## 7. Documentation and reproducibility

- [ ] The Plan retains its initial intent and includes dated changes/reasons and a decision record stating the chosen approach, rejected alternative and expected/observed cost. Unmeasured comparisons remain labelled as such. **Evidence: Pending.**
- [ ] Objectives contains actual setup details, calibration/model/measurement results, raw evidence links and honest target verdicts; pending fields are filled or explicitly declared not reached. **Evidence: Pending.**
- [ ] README at the repository root explains setup, hardware, camera, dataset ownership/counts/splits/tool, every implemented command, reference/plane requirements and remaining limitations. **Evidence: Pending.**
- [ ] Model weights are available in Git or through an accessible download, with the actual SHA-256 in README. Required data, labels, fixed lists, configs and calibration parameters are available to the reviewer. **Evidence: Pending.**
- [ ] A clean setup or fresh-clone run follows README and reproduces the implemented stages using declared files/downloads. Any observed numerical tolerance is documented. **Evidence: Pending.**
- [ ] All unfinished work is listed with reasons and impact. The dedicated item 7 runtime benchmark, item 8 breaking-condition study and item 9 trust/refusal-cost study are explicitly declared omitted; basic missing-object/reference handling remains implemented. **Evidence: Pending.**

## 8. Recording

- [ ] One Loom recording is accessible, has clear audio/readable output, and is **5 minutes or less**. Actual link and duration are recorded. **Evidence: Pending.**
- [ ] During that recording, I take a genuinely new photo, run the measurement command on it, show the output and measure the calculator with my ruler on camera. **Evidence: Pending.**
- [ ] I follow that image through the real submitted code: undistortion, segmentation, reference detection and pixel-to-mm conversion. **Evidence: Pending.**
- [ ] K1 is answered aloud without reading notes: the actual conversion file/line and scene conditions required for valid conversion. **Evidence: Pending.**
- [ ] K2 is answered aloud without reading notes: where undistortion happens and my own measured raw/corrected error difference, or an honest statement that the comparison was not reached. **Evidence: Pending.**
- [ ] K3 and K4 are answered aloud without reading notes: a real decision I would reverse and its cost, plus something I did not understand and what I did about it. **Evidence: Pending.**
- [ ] Recorded code/results agree with the submitted repository, checkpoint and configuration; every limitation stated in the recording is consistent with the documents. **Evidence: Pending.**

## 9. Submission

- [ ] The repository is public, work is on `dev-test01`, all three documents are under `docs/`, README is at the root, and progressive commit history is preserved. **Evidence: Pending.**
- [ ] One pull request from `dev-test01` into my own `main` exists; its description identifies working features, evidence and unfinished scope. **Evidence: Pending.**
- [ ] The PR link and Loom link are sent in reply to the original assessment message within 24 hours of the actual start. Actual start/deadline and submission time are recorded; the active-work budget is tracked honestly. **Evidence: Pending.**

## Final completion record - fill at the end

| Field | Initial status |
| --- | --- |
| Implemented stages and evidence | Portrait calibration/shared undistortion; joint correction of 72 images/100 masks; all 72 images passed user review with hash-bound approval. See sections 2-3; model training and later stages pending. |
| MO-1 measurement accuracy verdict | Not run |
| MO-2 calibration-quality verdict | Met: RMS 0.918653470 px, 21 usable views, 2296 x 4080 portrait profile only; no mm-accuracy inference. |
| Additional unfinished work and reasons | Pending |
| Deliberate omissions | Separate experiments 7-9; document their impact at completion |
| Final source commit / PR | Pending |
| Loom URL / duration | Pending |
| Actual start / deadline / submission | Pending |

An honest partial submission leaves unsupported items unchecked. A checked box without evidence does not establish completion.
