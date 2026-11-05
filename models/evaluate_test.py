"""Evaluate the frozen validation-selected checkpoint on seven committed images."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from torch.utils.data import DataLoader

from .data import collate_fn, seed_everything
from .model import build_model, model_summary
from .test_data import FrozenTestDataset
from .train import environment, save_json, sha256
from .validation import evaluate, _native_rle, _operating_point, _rle


PROJECT = Path(__file__).resolve().parent.parent
FROZEN_CHECKPOINT_SHA256 = "f46337538e462e09c5438a7c7d0aa675538a5ae819573d3ed3b84a96a0043a39"
FROZEN_SPLIT_SHA256 = "542bf7ec62522c86877d99707d6111e98dd516c4ff10f342d054ac77f34a8da1"


def committed_file(path: Path, revision: str) -> dict:
    """Require exact committed bytes, without changing the worktree or index."""
    relative = path.resolve().relative_to(PROJECT).as_posix()
    object_id = subprocess.check_output(
        ["git", "rev-parse", f"{revision}:{relative}"], cwd=PROJECT, text=True).strip()
    actual = subprocess.check_output(
        ["git", "hash-object", "--no-filters", "--", str(path.resolve())], cwd=PROJECT, text=True).strip()
    if actual != object_id:
        raise ValueError(f"File differs from committed test-evaluation input: {relative}")
    return {"path": relative, "sha256": sha256(path), "git_blob": object_id}


class GroundTruthCapture:
    """Capture the exact resized targets sent to the unchanged mask evaluator."""

    def __init__(self, loader, emit):
        self.loader, self.emit = loader, emit
        self.ground_truth = {"info": {"description": "Frozen held-out masks in 512/960 loader coordinates"},
                             "images": [], "annotations": [], "categories": [{"id": 1, "name": "calculator"}]}

    def __iter__(self):
        for images, targets in self.loader:
            for image, target in zip(images, targets):
                identifier = int(target["image_id"].item())
                height, width = image.shape[-2:]
                self.emit(f"TEST image_id={identifier} input={width}x{height} instances={len(target['labels'])}")
                self.ground_truth["images"].append({"id": identifier, "width": width, "height": height})
                for mask, label, crowd, source_id in zip(target["masks"], target["labels"],
                                                         target["iscrowd"], target["annotation_ids"]):
                    encoded = _rle(mask.numpy())
                    self.ground_truth["annotations"].append({
                        "id": len(self.ground_truth["annotations"]) + 1, "image_id": identifier,
                        "category_id": int(label), "iscrowd": int(crowd), "segmentation": encoded,
                        "area": int(mask.sum()), "bbox": mask_utils.toBbox(_native_rle(encoded)).tolist(),
                        "source_annotation_id": int(source_id)})
            yield images, targets


def audit_predictions(ground_truth, predictions, dataset):
    rows = []
    for record in dataset.records:
        identifier = record["id"]
        gt = [g for g in ground_truth["annotations"] if g["image_id"] == identifier]
        predicted = [p for p in predictions if p["image_id"] == identifier]
        point = _operating_point(gt, predicted, 0.5, 0.5)
        rows.append({"image_id": identifier, "file_name": record["file_name"],
                     "ground_truth_instances": len(gt), "all_scored_predictions_for_AP": len(predicted),
                     "AP_predictions_below_operating_score_cutoff": sum(p["score"] < 0.5 for p in predicted),
                     **point,
                     "mean_gt_mask_iou": point["iou_sum"] / point["noncrowd_gt_instances"]
                     if point["noncrowd_gt_instances"] else 0.0})
    return rows


def replay_ap(output, metrics):
    """Audit saved scored masks using official COCOeval; save its raw arrays."""
    capture = io.StringIO()
    with redirect_stdout(capture):
        gt = COCO(str(output / "ground_truth.json"))
        predictions = json.loads((output / "predictions.json").read_text())
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, iouType="segm")
        evaluator.params.imgIds = sorted(gt.getImgIds())
        evaluator.params.catIds = [1]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    if not np.allclose(evaluator.stats[:2], [metrics["segmentation_mAP_0.50_0.95"],
                                           metrics["segmentation_mAP_0.50"]], atol=1e-12, rtol=0):
        raise RuntimeError("Saved COCO mask predictions do not reproduce reported AP.")
    (output / "coco_replay.txt").write_text(capture.getvalue(), encoding="utf-8")
    np.savez_compressed(output / "coco_arrays.npz", precision=evaluator.eval["precision"],
                        recall=evaluator.eval["recall"], scores=evaluator.eval["scores"], stats=evaluator.stats)
    parameters = evaluator.params

    def native(value):
        if isinstance(value, np.ndarray):
            return native(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [native(item) for item in value]
        return value

    save_json(output / "coco_parameters.json", {key: native(value)
              for key, value in vars(parameters).items() if not key.startswith("_")})


def run(selection_path: Path, checkpoint_path: Path | None, output: Path, device_name="cuda") -> int:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh evaluation output; existing evidence is never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "models.evaluate_test", "--selection", str(selection_path),
               "--output", str(output), "--device", device_name]
    if checkpoint_path is not None:
        command.extend(["--checkpoint", str(checkpoint_path)])
    (output / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")

    def emit(message):
        print(message, flush=True)
        with (output / "console.txt").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    record = {"status": "initializing", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": command, "purpose": "held-out test evaluation only; no training or tuning"}
    save_json(output / "run.json", record)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
        selection_input = committed_file(selection_path, revision)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        checkpoint_path = checkpoint_path or selection_path.parent / "checkpoint_best.pt"
        if selection["sha256"] != FROZEN_CHECKPOINT_SHA256 or sha256(checkpoint_path) != FROZEN_CHECKPOINT_SHA256:
            raise ValueError("Checkpoint differs from the frozen validation-selected binary.")
        if selection["selection_split"] != "valid" or selection["epoch"] != 2:
            raise ValueError("Unexpected frozen checkpoint selection record.")
        config_path = selection_path.parent / "configuration.json"
        configuration_input = committed_file(config_path, revision)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source_path = selection_path.parent / "source_hashes.json"
        committed_file(source_path, revision)
        frozen_sources = json.loads(source_path.read_text(encoding="utf-8"))
        for relative, expected in frozen_sources.items():
            if sha256(PROJECT / relative) != expected:
                raise ValueError(f"Frozen training/evaluation source changed: {relative}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["training_config"] != config or checkpoint["epoch"] != selection["epoch"]:
            raise ValueError("Checkpoint metadata differs from committed selection/configuration.")
        torch.set_num_threads(config["training"]["cpu_threads"])
        seed_everything(config["seed"])
        info = environment()
        if str(torch.__version__) != checkpoint["versions"]["torch"] \
                or info["packages"].get("torchvision") != checkpoint["versions"]["torchvision"]:
            raise ValueError("PyTorch/Torchvision versions differ from the frozen checkpoint environment.")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no automatic device or precision substitution.")
        device = torch.device(device_name)
        save_json(output / "environment.json", info)
        save_json(output / "configuration.json", {"checkpoint_training_configuration": config,
                  "evaluation_split": "test", "expected_images": 7, "augmentation": False,
                  "device": device_name, "mixed_precision_active": device.type == "cuda" and config["training"]["mixed_precision"],
                  "geometry": "Same approved full-frame image/mask resize as validation; aspect-preserving 512/960, masks NN_EXACT",
                  "AP": "Official COCO segmentation AP; all returned scored masks at frozen model floor 0.05, no measurement gate",
                  "operating_point": {"score_cutoff": 0.5, "mask_probability": 0.5, "matching_mask_iou": 0.5},
                  "configuration_changes_or_test_tuning": []})
        emit(f"FROZEN checkpoint SHA256={FROZEN_CHECKPOINT_SHA256}; epoch=2; no optimizer or tuning")
        data_root = PROJECT / config["data_root"]
        split_input = committed_file(data_root / "splits/test.txt", revision)
        if split_input["sha256"] != FROZEN_SPLIT_SHA256:
            raise ValueError("Test membership differs from the initially committed split hash.")
        size = config["model"]
        dataset = FrozenTestDataset(data_root, min_size=size["min_size"], max_size=size["max_size"], seed=config["seed"])
        metadata_inputs = [committed_file(path, revision) for path in
                           (dataset.corrected_list_path, dataset.annotation_path, dataset.source_annotation_path,
                            dataset.corrected_root / "run.json", dataset.corrected_root / "review_approval.json",
                            dataset.corrected_root / "processing_run.json")]
        images = [committed_file(path, revision) for path in dataset.image_paths]
        inputs = {"git_commit": revision, "selection": selection_input, "configuration": configuration_input,
                  "checkpoint": {"path": str(checkpoint_path), "sha256": FROZEN_CHECKPOINT_SHA256},
                  "frozen_split": split_input, "metadata": metadata_inputs, "images": images,
                  "image_ids_in_frozen_list_order": dataset.image_ids, "approved_artifact_sha256": dataset.verified_artifacts}
        save_json(output / "data_manifest.json", inputs)
        save_json(output / "source_hashes.json", {**frozen_sources,
                  **{str(path.relative_to(PROJECT)): sha256(path) for path in
                     (Path(__file__).resolve(), PROJECT / "models/test_data.py")}})
        model = build_model(config["model"], pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        del checkpoint
        model.to(device).eval()
        save_json(output / "model_summary.json", model_summary(model))
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
        captured = GroundTruthCapture(loader, emit)
        record.update(status="evaluating", checkpoint_sha256=FROZEN_CHECKPOINT_SHA256,
                      split_sha256=FROZEN_SPLIT_SHA256, git_commit=revision)
        save_json(output / "run.json", record)
        metrics = evaluate(model, captured, device, config, output)
        gt = captured.ground_truth
        if metrics["images"] != 7 or {i["id"] for i in gt["images"]} != set(dataset.image_ids):
            raise RuntimeError("Evaluation did not consume exactly the seven committed test images.")
        save_json(output / "ground_truth.json", gt)
        predictions = json.loads((output / "predictions.json").read_text())
        per_image = audit_predictions(gt, predictions, dataset)
        if sum(r["true_positives"] for r in per_image) != metrics["true_positives"] \
                or not np.isclose(sum(r["iou_sum"] for r in per_image) / metrics["noncrowd_gt_instances"],
                                  metrics["mean_gt_mask_iou"]):
            raise RuntimeError("Per-image mask audit does not match aggregate metrics.")
        save_json(output / "per_image.json", per_image)
        replay_ap(output, metrics)
        metrics.update(split="test", checkpoint_sha256=FROZEN_CHECKPOINT_SHA256, split_sha256=FROZEN_SPLIT_SHA256,
                       score_floor_for_AP=config["model"]["box_score_thresh"], no_measurement_gate_for_AP=True,
                       predictions_below_operating_cutoff_retained_for_AP=sum(p["score"] < 0.5 for p in predictions),
                       scope="Exactly seven committed corrected test images, at unchanged frozen validation geometry/settings")
        save_json(output / "metrics.json", metrics)
        if sha256(checkpoint_path) != FROZEN_CHECKPOINT_SHA256 or sha256(dataset.list_path) != FROZEN_SPLIT_SHA256:
            raise RuntimeError("Frozen checkpoint or split changed during evaluation.")
        for path in dataset.image_paths + [dataset.annotation_path]:
            committed_file(path, revision)
        emit(metrics["coco_eval_stdout"])
        emit(f"TEST COMPLETE images=7 GT={metrics['gt_instances']} mask_IoU={metrics['mean_gt_mask_iou']:.9f} "
             f"mask_AP50={metrics['segmentation_mAP_0.50']:.9f} mask_AP50:95={metrics['segmentation_mAP_0.50_0.95']:.9f} "
             f"precision={metrics['precision_at_0.5']:.9f} recall={metrics['recall_at_0.5']:.9f}")
        record.update(status="completed", completed_at_utc=datetime.now(timezone.utc).isoformat(), images=7,
                      gt_instances=metrics["gt_instances"], checkpoint_and_split_unchanged=True,
                      saved_predictions_reproduce_official_mask_AP=True, configuration_changes_or_tuning=[], metrics=metrics)
        save_json(output / "run.json", record)
        save_json(output / "artifact_sha256.json", {path.relative_to(output).as_posix(): sha256(path)
                  for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_sha256.json"})
        return 0
    except Exception as exc:
        record.update(status="failed", error=str(exc), failed_at_utc=datetime.now(timezone.utc).isoformat())
        save_json(output / "run.json", record)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        emit(f"TEST EVALUATION FAILED: {exc}; raw evidence retained, no result claimed.")
        return 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("models/runs/initial_15_epochs/selection.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/evaluations/frozen_test_01"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    return run(args.selection, args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
