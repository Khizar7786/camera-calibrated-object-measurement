# Calculator segmentation training

**Status:** CUDA fine-tuning [finished](runs/initial_15_epochs/run.json) after **8 epochs / 400 real optimizer updates**, using the original plateau rule within the declared 15-epoch budget. **Epoch 2** was selected by validation AP. Frozen seven-image test evaluation is complete; the [one-image measurement CLI](../inference/README.md) is implemented; selected weights are available in the public [model release](https://github.com/Khizar7786/camera-calibrated-object-measurement/releases/tag/calculator-maskrcnn-v1). Physical accuracy validation remains unrun.

Run commands from the repository root using the project `.venv`. Training requires CUDA-enabled PyTorch and a matching Torchvision installation; exact installed versions and GPU details are recorded in each run. The CLI stops with a recorded failure if CUDA is unavailable. Dependency setup history is preserved in [setup_notes.json](setup_notes.json).

The actual environment uses **PyTorch 2.7.1+cu126 / Torchvision 0.22.1+cu126** and an NVIDIA GeForce RTX 3050 Ti Laptop GPU with **4 GiB** memory. Install the pinned base and CUDA dependencies with an up-to-date pip:

```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-training.txt
```

[requirements-training.txt](../requirements-training.txt) declares the CUDA 12.6 wheel index and exact Torch/Torchvision versions. All **72 tests passed in this actual project environment** before the recorded training run.

## Commands

A dedicated diagnostic run performs one optimizer update on the training image with the most approved instances, followed by one validation forward pass:

```powershell
.venv\Scripts\python.exe -m models.train --config models/training_config.json --output models/runs/calculator_smoke_01 --smoke-only
```

The actual declared training run was started with:

```powershell
.venv\Scripts\python.exe -m models.train --config models/training_config.json --output models/runs/initial_15_epochs
```

Use a fresh output directory, such as `models/runs/reproduction_01`, when reproducing the run. The normal command includes the smoke check. Its diagnostic weights are discarded, all seeds are reset, and real training starts from the official pretrained weights. A smoke pass records finite losses, a backward/optimizer step, valid validation outputs and GPU peak memory; it is not an accuracy evaluation. Existing nonempty output directories are rejected so failed attempts and earlier evidence remain available.

## Declared configuration

[training_config.json](training_config.json) is the source of truth. Its initial budget was recorded before smoke or training:

| Setting | Initial value |
| --- | --- |
| Architecture | Torchvision Mask R-CNN ResNet50-FPN with official COCO pretrained weights |
| Classes | Background `0`, calculator `1`; both box and mask prediction heads are replaced for two classes |
| Trainable backbone | Last three ResNet stages; FPN and prediction heads are also trainable |
| Initial epoch budget | **15 epochs**, without automatic extension |
| Batch / workers | **1 / 0**; Windows-safe loading in the main process |
| Seed / CPU threads | **42 / 4** |
| Image geometry | Aspect-preserving short side **512**, long side capped at **960**; current corrected images become **512 x 910** |
| Augmentation | Joint image/mask horizontal flip with probability **0.5**, training only |
| Optimizer | SGD, learning rate **0.0025**, momentum **0.9**, weight decay **0.0005** |
| Schedule | StepLR every **5** epochs, multiplying learning rate by **0.1** |
| Gradient handling | CUDA mixed precision, GradScaler and gradient norm clipping at **5.0** |
| Plateau rule | At least **8** epochs; stop after **6** epochs without validation AP improvement exceeding **0.001** |
| Training-loop wall budget | **45 minutes**, checked after completed epochs; setup, downloads and smoke occur before this timer |

The epoch cap, plateau rule or wall budget may end a run. The wall check can exceed its limit by the time needed to finish the current epoch. Completed epochs and the actual stop reason belong in `run.json`. Any memory-driven change to resolution, backbone training or other settings must preserve the failed record, state the reason, and pass a new smoke check before training. Additional epochs require an explicit decision supported by validation evidence and the remaining time budget.

The code requests seeded, deterministic behavior and disables TF32. The actual run reported ROIAlign backward CUDA nondeterminism in [warnings.txt](runs/initial_15_epochs/warnings.txt); exact repeatability across runs or different hardware/library builds is not guaranteed.

## Approved inputs and geometry

[data.py](data.py) constructs only the approved corrected **50-image training** and **15-image validation** splits. It checks approval hashes, frozen membership and annotation identity. Training, augmentation, validation and checkpoint selection do not construct a test loader or open test lists, images or annotations.

The approved data contains **75 training instances** and **18 validation instances**, including user-accepted multiple-instance annotations. All annotated objects use category `1`; the unused source parent category `0` is not a foreground class. Sources and reviewed corrected artifacts remain unchanged.

Full-resolution corrected images and masks share the calibrated portrait frame. The loader resizes them jointly on the CPU using `scale = min(512 / min(width, height), 960 / max(width, height))`, with rounded output dimensions. Images use area downsampling and masks use `INTER_NEAREST_EXACT`; training does not stretch the aspect ratio. Bounding boxes and areas are recomputed from resized masks. Exclusive upper box bounds retain one-pixel-wide instances. A mask that disappears raises an error and is not silently removed. Validation has no augmentation.

## Validation and checkpoint selection

[validation.py](validation.py) operates only on tensors yielded by the supplied validation loader. Metrics refer to those resized loader coordinates. Thresholds are fixed before training:

| Quantity | Fixed definition |
| --- | --- |
| Binary predicted mask | Probability **>=0.5** |
| Model output score floor / NMS | **0.05 / 0.5**, unchanged across runs |
| Segmentation AP | Official COCO mask AP over **0.50:0.95** and at **0.50**, using all returned predictions and the standard maximum of 100 detections per image |
| Precision / recall | Score **>=0.5**, mask IoU **>=0.5**, score-ordered one-to-one matching per image and category |
| Mean ground-truth mask IoU | Separate score-ordered one-to-one matching for any positive overlap at score **>=0.5**; unmatched ground truth contributes zero |
| Checkpoint selection | Highest validation `segmentation_mAP_0.50_0.95`; the earliest epoch wins an exact tie |

Validation runs after every epoch. Each strict increase in selection AP saves the best checkpoint; the separate plateau rule uses the predeclared 0.001 significance threshold. There is no operating-threshold search. Validation scores select the checkpoint and must be reported as validation evidence. Independent measurement accuracy remains pending.

## Saved evidence and weights

Each fresh run directory records:

- `configuration.json`, `command.txt`, `source_hashes.json`, `environment.json` and `requirements.freeze.txt`: actual settings, source hashes, installed versions, hardware and command.
- `data_manifest.json` and `smoke.json`: training/validation membership evidence and the worst-instance memory/finite-output diagnostic.
- `console.txt`, `steps.jsonl`, `epochs.jsonl` and any `configuration_changes.jsonl`: raw progress, losses, validation metrics and scheduled learning-rate changes.
- `validation/epoch_XX/metrics.json`, `predictions.json` and `coco_eval.txt`: per-epoch metric definitions, compressed-mask predictions and captured official COCO output.
- `selection.json`, `checkpoint_best.sha256` and `run.json`: selected epoch, validation score, checkpoint hash, completion/failure status and actual stop reason.

The local binary files `checkpoint_best.pt` and `last_training_state.pt` are ignored by Git, as is the official-weight cache under `models/pretrained/`. The best checkpoint contains the model state, class/model/training configuration, epoch and validation-selection evidence. The last state additionally preserves optimizer, scheduler and scaler state; the CLI does not currently expose a resume command. The selected checkpoint and checksum are published in the [calculator-maskrcnn-v1 release](https://github.com/Khizar7786/camera-calibrated-object-measurement/releases/tag/calculator-maskrcnn-v1). Follow the [root README download and hash-check commands](../README.md#download-the-selected-checkpoint); the last optimizer/training state remains local.

## Current result record

| Evidence | Status |
| --- | --- |
| Actual CUDA smoke and peak memory | **Passed**: 23-instance training update and first validation image; peak allocated **891,221,504 bytes**, reserved **1,482,686,464 bytes**. [Smoke record](runs/initial_15_epochs/smoke.json). |
| Completed training epochs / stop reason | **8 epochs / 400 updates**, `validation_plateau_early_stop`; original minimum 8 epochs and patience 6, with no AP improvement >0.001 following epoch 2. [Run](runs/initial_15_epochs/run.json). |
| Selected epoch | **2**, the earliest maximum validation AP. [Selection](runs/initial_15_epochs/selection.json). |
| Validation mAP@0.50:0.95 / mAP@0.50 | **0.8316831683168316 / 0.8316831683168316**, over 15 validation images and 18 instances. |
| Validation precision / recall | **1.0 / 0.8333333333333334** at fixed score/mask IoU thresholds 0.5; **15 TP / 0 FP / 3 FN**. |
| Validation mean GT mask IoU | **0.8070517357958565**, including zero for unmatched GT. [Selected-epoch raw metrics](runs/initial_15_epochs/validation/epoch_02/metrics.json). |
| Selected checkpoint | **176,220,108 bytes**, ignored by Git and available in the public [model release](https://github.com/Khizar7786/camera-calibrated-object-measurement/releases/tag/calculator-maskrcnn-v1). Hash below. |
| Independent millimetre measurement accuracy | Pending |

The training-loop elapsed time was **309.344 seconds**, excluding setup/downloads/smoke. Epoch peaks were **1,072,794,112 bytes allocated** and **1,245,708,288 bytes reserved**. No out-of-memory configuration changes were needed. The original StepLR schedule changed learning rate from **0.0025 to 0.00025 after epoch 5**; see the [epoch records](runs/initial_15_epochs/epochs.jsonl). The single run duration is recorded provenance, not a repeated runtime benchmark.

Selected `checkpoint_best.pt` SHA-256:

```text
f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39
```

[Verification](runs/initial_15_epochs/verification.json) passed an offline `weights_only=True` CPU reload and strict state-dict match, with two-class box/mask heads and configuration/source hashes matching the run. It also checked all 400 step losses were finite and rechecked 227 approved training/validation artifacts and unchanged frozen lists. No inference or test evaluation was performed by that integrity check. Download the selected binary from the [model release](https://github.com/Khizar7786/camera-calibrated-object-measurement/releases/tag/calculator-maskrcnn-v1) to `models/runs/initial_15_epochs/checkpoint_best.pt`. Anonymous download access and the matching GitHub SHA-256 were verified on 2026-11-06.

These results are validation evidence used for checkpoint selection. Frozen test metrics are recorded below; independent millimetre accuracy measurements have not been run. Preserve setup notes and failed-attempt evidence when reporting time spent. The recording must stay within five minutes and explain the code and results actually available, including assessment stages that were not reached. The author will supply the Loom link in the submission email; its duration/content have not been reviewed here.

## Frozen held-out test result

The [seven-image evaluation report](evaluations/README.md) records mean GT mask IoU **0.967517224**, mask AP50/AP50:95 **1.0/1.0**, precision/recall **1.0/1.0** (7 TP, 0 FP, 0 FN), with the same epoch-2 checkpoint and frozen settings. All 76 tests pass. Its separate explicit test loader leaves the training class's held-out rejection intact. No test-driven settings change or checkpoint selection occurred. See the report for the command, hashes, configuration, aggregation/matching definitions and raw scored-mask COCO evidence.
