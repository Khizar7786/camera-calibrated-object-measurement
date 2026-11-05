# Frozen seven-image mask evaluation

Completed on 2026-11-05 using exactly the **seven committed corrected test images / seven Calculator instances**, preserving the original frozen filename list. The validation-selected **epoch-2** Mask R-CNN checkpoint, geometry and thresholds were unchanged. Test results were not used for training, augmentation, model/checkpoint selection or threshold tuning.

```powershell
.venv\Scripts\python.exe -m models.evaluate_test --output models/evaluations/frozen_test_01
```

The [saved command](frozen_test_01/command.txt) expands the selection path and CUDA device explicitly. Existing output is preserved; reproduction needs a fresh output such as `--output models/evaluations/reproduction_01`.

| Mask metric | Test result |
| --- | ---: |
| Mean ground-truth mask IoU | **0.967517224** |
| Segmentation mAP@0.5 | **1.000000** |
| Segmentation mAP@0.5:0.95 | **1.000000** |
| Precision, score >=0.5 and mask IoU >=0.5 | **1.000000** |
| Recall at the same operating point | **1.000000** |
| TP / FP / FN | **7 / 0 / 0** |

AP is official `pycocotools==2.0.10` **COCOeval(iouType="segm")**, category calculator only, standard 101 recall points and mask IoU thresholds 0.50:0.05:0.95, maximum 100 detections/image. Every scored mask returned by the frozen model is retained for AP. The model's pre-existing score floor is **0.05** and box NMS **0.5**; no measurement/operating score gate filters the AP predictions. This run returned **7** scored predictions, all above 0.5; there were **0** additional below-operating-cutoff predictions. A synthetic check separately demonstrates that a correct score-0.2 mask contributes to AP while remaining a missed GT at the operating point.

Predicted masks are binarized at probability **>=0.5**. Operating precision/recall use score **>=0.5**, per-image/category descending-score greedy one-to-one matching to the best remaining GT at **mask IoU >=0.5**. Duplicate/unmatched predictions count as false positives; unmatched GT count as false negatives. These are mask metrics, with no box-AP or average-confidence substitution.

Mean GT mask IoU uses a separate per-image/category score-ordered one-to-one best remaining GT match for any positive mask overlap, from predictions with score >=0.5. Sum those IoUs and divide by **all seven** non-crowd GT instances; each unmatched/missed GT contributes **zero**, and no denominator is restricted to successful detections. This run has no missed GT. All masks/metrics use the unchanged aspect-preserving **512/960** loader transform: current images become **512 x 910**, masks use exact nearest-neighbour, augmentation is disabled. Results are in these resized coordinates rather than original-resolution mask coordinates.

| Capture | Frozen COCO image ID | Mask IoU | TP / FP / FN |
| --- | ---: | ---: | --- |
| IMG_20261104_200342 | 5 | 0.967939551 | 1 / 0 / 0 |
| IMG_20261104_200453 | 3 | 0.966988217 | 1 / 0 / 0 |
| IMG_20261104_200656 | 0 | 0.974380433 | 1 / 0 / 0 |
| IMG_20261104_200700 | 4 | 0.974587348 | 1 / 0 / 0 |
| IMG_20261104_200704 | 2 | 0.966233072 | 1 / 0 / 0 |
| IMG_20261104_200710 | 6 | 0.963556605 | 1 / 0 / 0 |
| IMG_20261104_200722 | 1 | 0.958935345 | 1 / 0 / 0 |

Checkpoint: `models/runs/initial_15_epochs/checkpoint_best.pt`, epoch **2**, selected using validation only. SHA-256:

```text
f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39
```

Frozen split: `dataset/splits/test.txt`, SHA-256:

```text
542bf7ec62522c86877d99707d6111e98dd516c4ff10f342d054ac77f34a8da1
```

Committed data/selection provenance: `659f022b80ce6e25f41747262bd2290d51228c76`. [Data manifest](frozen_test_01/data_manifest.json) records Git blob IDs and SHA-256 values for all seven images, original/corrected test annotations, approval metadata and both identical split lists. Corrected annotation SHA-256: `4b3862f4945effa0455e270073786274d8856d7d4cb7402cb0f7e8dd632aa23c`. Checkpoint/list hashes and committed image/annotation bytes were checked again after evaluation.

[Configuration](frozen_test_01/configuration.json) retains the full checkpoint training configuration and actual test settings. [Environment](frozen_test_01/environment.json) records exact installed versions and hardware: Python 3.11.0, PyTorch 2.7.1+cu126, Torchvision 0.22.1+cu126, CUDA build 12.6, RTX 3050 Ti Laptop GPU, mixed precision as used for validation, seed 42, batch 1, workers 0. The training loader continues to reject test; explicit test access is confined to [test_data.py](../test_data.py) and [evaluate_test.py](../evaluate_test.py).

Raw evidence: [metrics and definitions](frozen_test_01/metrics.json), [COCO stdout](frozen_test_01/coco_eval.txt), [scored RLE predictions](frozen_test_01/predictions.json), [evaluated GT masks](frozen_test_01/ground_truth.json), [COCO parameters](frozen_test_01/coco_parameters.json), [COCO precision/recall/score arrays](frozen_test_01/coco_arrays.npz), [per-image matching results](frozen_test_01/per_image.json), [console](frozen_test_01/console.txt), [run status](frozen_test_01/run.json), and [source hashes](frozen_test_01/source_hashes.json). [Saved-prediction replay](frozen_test_01/coco_replay.txt) reproduces both official mask AP scores without a second model inference or any threshold search. Raw COCO -1 small/medium size-bin results mean no GT in those bins; the overall AP results above have all seven GT instances.

[Verification](frozen_test_01/verification.json) confirms independent decoded-mask IoU calculations, unchanged hashes, scored-prediction AP replay, and **76 passing tests**. [Artifact hashes](frozen_test_01/artifact_sha256.json) bind the raw outputs. These seven images repeat one physical calculator under the captured conditions; this result does not establish performance on unseen calculator products or millimetre measurement accuracy. Physical measurement/inference, an accessible checkpoint download and the assessment recording remain pending.
