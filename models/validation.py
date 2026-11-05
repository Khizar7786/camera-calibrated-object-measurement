"""Validation metrics from a supplied loader, with no dataset-file access.

COCO segmentation AP receives every prediction returned by the model. Fixed
operating-point metrics use score >= 0.5 and binary mask probabilities >= 0.5.
Their two score-ordered, one-to-one matchings are separate: precision/recall
requires mask IoU >= 0.5; mean ground-truth IoU credits any positive overlap
with the best remaining same-class ground truth. Unmatched ground truth is zero.
"""

from __future__ import annotations

from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch


class ValidationError(ValueError):
    """Malformed validation tensors, configuration, or repeated input IDs."""


def _fixed_threshold(settings: Mapping, key: str) -> float:
    value = settings.get(key, 0.5)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != 0.5:
        raise ValidationError(f"{key} must remain fixed at 0.5; threshold search is forbidden.")
    return 0.5


def _array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise ValidationError(f"{name} must be a tensor.")
    return value.detach().cpu().numpy()


def _labels(value: Any, count: int, num_classes: int, name: str) -> np.ndarray:
    labels = _array(value, name)
    if labels.shape != (count,) or labels.dtype.kind not in "iu" or np.any(labels < 1) or np.any(labels >= num_classes):
        raise ValidationError(f"{name} must contain {count} integer foreground labels in [1, {num_classes - 1}].")
    return labels.astype(np.int64, copy=False)


def _rle(binary: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    return {"size": [int(v) for v in encoded["size"]], "counts": encoded["counts"].decode("ascii")}


def _native_rle(encoded: dict) -> dict:
    return {"size": encoded["size"], "counts": encoded["counts"].encode("ascii")}


def _operating_point(gt: list[dict], predictions: list[dict], score_cutoff: float, iou_cutoff: float) -> dict:
    selected = sorted((p for p in predictions if p["score"] >= score_cutoff), key=lambda p: -p["score"])
    ordinary = {index for index, target in enumerate(gt) if not target["iscrowd"]}
    available_pr = ordinary.copy()
    available_iou = ordinary.copy()
    tp = fp = ignored_crowd = 0
    iou_sum = 0.0
    if selected and gt:
        overlaps = np.asarray(mask_utils.iou(
            [_native_rle(p["segmentation"]) for p in selected],
            [_native_rle(g["segmentation"]) for g in gt],
            [g["iscrowd"] for g in gt],
        ), dtype=np.float64).reshape(len(selected), len(gt))
    else:
        overlaps = np.zeros((len(selected), len(gt)), dtype=np.float64)
    for pi, prediction in enumerate(selected):
        category = prediction["category_id"]
        remaining = [gi for gi in sorted(available_iou) if gt[gi]["category_id"] == category]
        if remaining:
            best = max(remaining, key=lambda gi: overlaps[pi, gi])
            if overlaps[pi, best] > 0:
                iou_sum += float(overlaps[pi, best])
                available_iou.remove(best)
        remaining = [gi for gi in sorted(available_pr) if gt[gi]["category_id"] == category]
        best = max(remaining, key=lambda gi: overlaps[pi, gi]) if remaining else None
        if best is not None and overlaps[pi, best] >= iou_cutoff:
            tp += 1
            available_pr.remove(best)
        elif any(g["iscrowd"] and g["category_id"] == category and overlaps[pi, gi] >= iou_cutoff for gi, g in enumerate(gt)):
            # pycocotools uses intersection / predicted area for crowd regions.
            ignored_crowd += 1
        else:
            fp += 1
    return {
        "true_positives": tp, "false_positives": fp,
        "false_negatives": len(available_pr), "iou_sum": iou_sum,
        "noncrowd_gt_instances": len(ordinary),
        "operating_point_predictions": len(selected),
        "ignored_crowd_predictions": ignored_crowd,
    }


def evaluate(model, loader, device, config: dict, output_dir: Path | None = None) -> dict:
    """Evaluate only supplied tensors using the full training configuration.

    The model's original training/evaluation mode is restored. No loader dataset
    paths, split lists, annotations, or source images are read here. If requested,
    output_dir receives metrics.json, predictions.json, and coco_eval.txt.
    Empty detections are evaluated as an empty in-memory COCO result object.
    """
    if not isinstance(config, Mapping):
        raise ValidationError("Expected the full training configuration object.")
    validation = config.get("validation", {})
    training = config.get("training", {})
    model_config = config.get("model", {})
    if not all(isinstance(value, Mapping) for value in (validation, training, model_config)):
        raise ValidationError("validation, training and model settings must be objects.")
    mask_threshold = _fixed_threshold(validation, "mask_threshold")
    score_cutoff = _fixed_threshold(validation, "precision_recall_score_threshold")
    iou_cutoff = _fixed_threshold(validation, "precision_recall_mask_iou_threshold")
    num_classes = model_config.get("num_classes", 2)
    if type(num_classes) is not int or num_classes < 2:
        raise ValidationError("model.num_classes must include background and foreground.")
    mixed_precision = training.get("mixed_precision", False)
    if type(mixed_precision) is not bool:
        raise ValidationError("training.mixed_precision must be Boolean.")
    device = torch.device(device)
    use_amp = device.type == "cuda" and mixed_precision
    names = config.get("class_names", [])
    categories = [
        {"id": category, "name": names[category] if isinstance(names, list) and len(names) > category else f"class_{category}"}
        for category in range(1, num_classes)
    ]
    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    coco_predictions: list[dict] = []
    image_ids: set[int] = set()
    counts = {"true_positives": 0, "false_positives": 0, "false_negatives": 0,
              "iou_sum": 0.0, "noncrowd_gt_instances": 0,
              "operating_point_predictions": 0, "ignored_crowd_predictions": 0}
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for images, targets in loader:
                if len(images) != len(targets) or not images:
                    raise ValidationError("Each batch must contain matching nonempty image/target lists.")
                input_images = []
                for image in images:
                    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3 or image.shape[1] <= 0 or image.shape[2] <= 0:
                        raise ValidationError("Validation images must be nonempty 3 x H x W tensors.")
                    input_images.append(image.to(device))
                amp = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
                with amp:
                    outputs = model(input_images)
                if len(outputs) != len(images):
                    raise ValidationError("Model returned a different number of predictions than images.")
                for image, target, output in zip(images, targets, outputs):
                    height, width = int(image.shape[1]), int(image.shape[2])
                    identifier = _array(target.get("image_id"), "target.image_id")
                    if identifier.size != 1 or identifier.dtype.kind not in "iu" or int(identifier.reshape(-1)[0]) < 0:
                        raise ValidationError("target.image_id must contain one nonnegative integer.")
                    image_id = int(identifier.reshape(-1)[0])
                    if image_id in image_ids:
                        raise ValidationError(f"Duplicate validation image_id {image_id}.")
                    image_ids.add(image_id)
                    coco_images.append({"id": image_id, "width": width, "height": height})
                    gt_masks = _array(target.get("masks"), "target.masks")
                    if gt_masks.ndim != 3 or gt_masks.shape[1:] != (height, width) or gt_masks.dtype.kind not in "bui" or not np.isin(gt_masks, [0, 1]).all():
                        raise ValidationError("target.masks must be binary N x H x W masks aligned with the loader image.")
                    gt_labels = _labels(target.get("labels"), len(gt_masks), num_classes, "target.labels")
                    crowd = _array(target["iscrowd"], "target.iscrowd") if "iscrowd" in target else np.zeros(len(gt_masks), dtype=np.int64)
                    if crowd.shape != (len(gt_masks),) or crowd.dtype.kind not in "bui" or not np.isin(crowd, [0, 1]).all():
                        raise ValidationError("target.iscrowd must be a binary vector aligned with masks.")
                    image_gt = []
                    for mask, label, iscrowd in zip(gt_masks, gt_labels, crowd):
                        area = int(np.count_nonzero(mask))
                        if area == 0:
                            raise ValidationError("An empty ground-truth instance cannot establish validation accuracy.")
                        encoded = _rle(mask)
                        annotation = {
                            "id": len(coco_annotations) + 1,
                            "image_id": image_id, "category_id": int(label),
                            "segmentation": encoded, "area": area,
                            "bbox": [float(v) for v in mask_utils.toBbox(_native_rle(encoded))],
                            "iscrowd": int(iscrowd),
                        }
                        coco_annotations.append(annotation)
                        image_gt.append(annotation)
                    scores = _array(output.get("scores"), "prediction.scores")
                    if scores.ndim != 1 or scores.dtype.kind not in "fiu" or not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
                        raise ValidationError("Prediction scores must be a finite vector in [0, 1].")
                    pred_labels = _labels(output.get("labels"), len(scores), num_classes, "prediction.labels")
                    mask_tensor = output.get("masks")
                    if not isinstance(mask_tensor, torch.Tensor):
                        raise ValidationError("prediction.masks must be a tensor.")
                    # Transfer before a possible fp16->fp32 cast to avoid a large
                    # temporary fp32 mask allocation on the 4 GiB GPU.
                    probabilities = mask_tensor.detach().cpu().float().numpy()
                    if probabilities.ndim == 4 and probabilities.shape[1] == 1:
                        probabilities = probabilities[:, 0]
                    if probabilities.shape != (len(scores), height, width) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
                        raise ValidationError("Prediction masks must be N x 1 x H x W probabilities aligned with the loader image.")
                    image_predictions = []
                    for probability, label, score in zip(probabilities, pred_labels, scores):
                        prediction = {
                            "image_id": image_id, "category_id": int(label),
                            "segmentation": _rle(probability >= mask_threshold),
                            "score": float(score),
                        }
                        coco_predictions.append(prediction)
                        image_predictions.append(prediction)
                    probability = None  # Release a NumPy view of the previous batch.
                    observed = _operating_point(image_gt, image_predictions, score_cutoff, iou_cutoff)
                    for key, value in observed.items():
                        counts[key] += value
                # Do not retain a batch of full-size CUDA mask probabilities.
                del outputs, input_images, output, mask_tensor, probabilities
    finally:
        model.train(was_training)
    if not coco_images:
        raise ValidationError("The supplied validation loader yielded no images.")

    capture = io.StringIO()
    with redirect_stdout(capture):
        ground_truth = COCO()
        ground_truth.dataset = {"info": {"description": "Provided validation-loader targets"},
                                "images": coco_images, "categories": categories,
                                "annotations": coco_annotations}
        ground_truth.createIndex()
        if coco_predictions:
            detections = ground_truth.loadRes(copy.deepcopy(coco_predictions))
        else:
            detections = COCO()
            detections.dataset = {"info": {}, "images": copy.deepcopy(coco_images),
                                  "categories": copy.deepcopy(categories), "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, iouType="segm")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = [category["id"] for category in categories]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    tp, fp, fn = counts["true_positives"], counts["false_positives"], counts["false_negatives"]
    noncrowd = counts["noncrowd_gt_instances"]
    result = {
        "segmentation_mAP_0.50_0.95": max(0.0, stats[0]),
        "segmentation_mAP_0.50": max(0.0, stats[1]),
        "precision_at_0.5": tp / (tp + fp) if tp + fp else 0.0,
        "recall_at_0.5": tp / (tp + fn) if tp + fn else 0.0,
        "mean_gt_mask_iou": counts["iou_sum"] / noncrowd if noncrowd else 0.0,
        "images": len(coco_images), "gt_instances": len(coco_annotations),
        "predictions": len(coco_predictions),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "noncrowd_gt_instances": noncrowd,
        "operating_point_predictions": counts["operating_point_predictions"],
        "ignored_crowd_predictions": counts["ignored_crowd_predictions"],
        "ap_has_non_crowd_ground_truth": bool(noncrowd),
        "coco_stats": stats,
        "coco_eval_stdout": capture.getvalue(),
        "thresholds": {"mask_probability": mask_threshold, "score": score_cutoff, "mask_iou": iou_cutoff},
        "metric_definitions": {
            "AP": "Official COCO segmentation AP; all model-returned predictions, 100 max detections/image, loader target pixel coordinates.",
            "precision_recall": "Per-image/category descending-score one-to-one best remaining GT matching at mask IoU >= 0.5; score >= 0.5. Crowd-only matches ignored.",
            "mean_gt_mask_iou": "Separate per-image/category descending-score one-to-one best remaining non-crowd GT matching for any positive IoU; score >= 0.5. Sum matched IoUs divided by every non-crowd GT instance, unmatched GT zero.",
            "unavailable_AP": "COCO -1 for absent ground truth is retained in raw stats; reported AP is 0 with availability flag false.",
        },
        "mixed_precision": use_amp,
        "scope": "Only the provided loader tensors; no dataset source files accessed.",
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "predictions.json").write_text(json.dumps(coco_predictions, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (output_dir / "coco_eval.txt").write_text(capture.getvalue(), encoding="utf-8")
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
