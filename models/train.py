"""Smoke-check and fine-tune the approved two-class Mask R-CNN, train/val only."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import warnings

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent / "pretrained"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CalculatorDataset, collate_fn, seed_everything, seed_worker
from .model import build_model, model_summary
from .validation import evaluate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_json(path: Path, record) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def save_torch(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".pending")
    torch.save(data, temporary)
    temporary.replace(path)


def validate_config(config: dict) -> None:
    if config.get("schema_version") != 1 or config.get("seed") != 42:
        raise ValueError("Training requires schema version 1 and seed 42.")
    if config.get("class_names") != ["background", "calculator"] or config["model"]["num_classes"] != 2:
        raise ValueError("The two classes must be background=0 and calculator=1.")
    settings = config["training"]
    if settings["batch_size"] != 1 or settings["num_workers"] != 0:
        raise ValueError("This declared Windows-safe run requires batch size 1 and num_workers=0.")
    if settings["optimizer"] != "SGD" or settings["epochs"] < 1:
        raise ValueError("Use the declared SGD configuration and a positive epoch budget.")
    for key in ("learning_rate", "maximum_wall_minutes", "gradient_clip_norm", "initial_amp_scale"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if config["model"]["max_size"] < config["model"]["min_size"]:
        raise ValueError("The long-side limit cannot be shorter than the short-side target.")
    if config["validation"]["selection_metric"] != "segmentation_mAP_0.50_0.95":
        raise ValueError("Checkpoint selection must use validation segmentation AP50:95.")


def environment() -> dict:
    packages = {item.metadata["Name"]: item.version for item in metadata.distributions() if item.metadata["Name"]}
    result = {"python_executable": sys.executable, "python_version": sys.version,
              "platform": platform.platform(), "processor": platform.processor(),
              "logical_cpu_count": os.cpu_count(),
              "packages": dict(sorted(packages.items())), "torch": str(torch.__version__),
              "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "torch_home": os.environ["TORCH_HOME"],
              "deterministic_algorithms": "requested with warn_only=True; unsupported nondeterminism is reported, not hidden"}
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        result["gpu"] = {"name": properties.name, "total_memory_bytes": properties.total_memory,
                         "compute_capability": [properties.major, properties.minor]}
    return result


def move_batch(images, targets, device):
    images = [image.to(device, non_blocking=False) for image in images]
    targets = [{key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()} for target in targets]
    return images, targets


def optimizer_for(model, config):
    settings = config["training"]
    return torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad],
                           lr=settings["learning_rate"], momentum=settings["momentum"], weight_decay=settings["weight_decay"])


def train_step(model, optimizer, scaler, images, targets, device, config) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    images, targets = move_batch(images, targets, device)
    use_amp = device.type == "cuda" and config["training"]["mixed_precision"]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        losses = model(images, targets)
        total = sum(losses.values())
    if not torch.isfinite(total).item() or not all(torch.isfinite(loss).item() for loss in losses.values()):
        raise RuntimeError(f"Nonfinite training loss: { {key: float(value.detach()) for key, value in losses.items()} }")
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad],
                                         config["training"]["gradient_clip_norm"], error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    result = {key: float(value.detach().cpu()) for key, value in losses.items()}
    result.update(loss_total=float(total.detach().cpu()), gradient_norm_before_clip=float(norm.detach().cpu()),
                  amp_scale=float(scaler.get_scale()))
    optimizer.zero_grad(set_to_none=True)
    return result


def cuda_memory(device) -> dict:
    if device.type != "cuda":
        return {}
    torch.cuda.synchronize()
    return {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def smoke(config: dict, train_dataset, validation_dataset, device, output: Path, emit) -> dict:
    """One worst-instance train update plus one validation forward, then discard."""
    seed_everything(config["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(config["model"], pretrained=True).to(device)
    save_json(output / "model_summary.json", model_summary(model))
    optimizer = optimizer_for(model, config)
    scaler = torch.amp.GradScaler("cuda", init_scale=config["training"].get("initial_amp_scale", 1024.0),
                                  enabled=device.type == "cuda" and config["training"]["mixed_precision"])
    index = train_dataset.max_instances_index
    # Memory-check the actual worst target without advancing augmentation RNG.
    previous_augmentation = train_dataset.augment
    train_dataset.augment = False
    try:
        image, target = train_dataset[index]
    finally:
        train_dataset.augment = previous_augmentation
    emit(f"SMOKE train image_id={train_dataset.image_ids[index]}, instances={len(target['labels'])}, input={list(image.shape)}")
    losses = train_step(model, optimizer, scaler, [image], [target], device, config)
    validation_image, _ = validation_dataset[0]
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                               enabled=device.type == "cuda" and config["training"]["mixed_precision"]):
        predictions = model([validation_image.to(device)])[0]
    for key in ("boxes", "scores", "masks"):
        if not torch.isfinite(predictions[key]).all().item():
            raise RuntimeError(f"Smoke validation produced nonfinite {key}.")
    if predictions["labels"].numel() and not torch.all(predictions["labels"] == 1).item():
        raise RuntimeError("Smoke validation produced an unexpected foreground class.")
    record = {"status": "passed", "training_image": train_dataset.sample_metadata(index),
              "validation_image": validation_dataset.sample_metadata(0), "train_losses": losses,
              "actual_smoke_augmentation": False,
              "validation_detections": int(predictions["scores"].numel()),
              "note": "This is a finite forward/backward and validation smoke check, not an accuracy result. Smoke weights are discarded before the real run.",
              **cuda_memory(device)}
    record["training_image"]["horizontal_flip_probability"] = 0.0
    save_json(output / "smoke.json", record)
    emit(f"SMOKE PASSED: loss={losses['loss_total']:.6f}; validation detections={record['validation_detections']}; memory={cuda_memory(device)}")
    del model, optimizer, scaler, predictions, image, target, validation_image
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def run(config_path: Path, output: Path, smoke_only: bool = False) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Run output already contains files; choose a fresh directory.")
    output.mkdir(parents=True, exist_ok=True)
    log = output / "console.txt"

    def emit(message):
        print(message, flush=True)
        with log.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    started = datetime.now(timezone.utc).isoformat()
    command = [sys.executable, "-m", "models.train", "--config", str(config_path), "--output", str(output)]
    if smoke_only:
        command.append("--smoke-only")
    summary = {"status": "initializing", "started_at_utc": started, "command": command,
               "epochs_requested": config["training"]["epochs"], "completed_epochs": 0, "seed": config["seed"],
               "test_data_used": False, "selection_split": "valid", "output": str(output),
               "best_epoch": None, "best_validation_metric": None}
    save_json(output / "run.json", summary)
    save_json(output / "configuration.json", config)
    (output / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    source_files = sorted(Path("models").glob("*.py"))
    save_json(output / "source_hashes.json", {str(path): sha256(path) for path in source_files})
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    (output / "pip_freeze_raw.txt").write_text(freeze, encoding="utf-8")
    exact_versions = sorted({f"{item.metadata['Name']}=={item.version}" for item in metadata.distributions()
                             if item.metadata["Name"]}, key=str.lower)
    (output / "requirements.freeze.txt").write_text(
        "--extra-index-url https://download.pytorch.org/whl/cu126\n" + "\n".join(exact_versions) + "\n",
        encoding="utf-8")
    model = None
    original_showwarning = warnings.showwarning

    def record_warning(message, category, filename, lineno, file=None, line=None):
        original_showwarning(message, category, filename, lineno, file, line)
        with (output / "warnings.txt").open("a", encoding="utf-8") as stream:
            stream.write(warnings.formatwarning(message, category, filename, lineno, line))

    warnings.showwarning = record_warning
    try:
        torch.set_num_threads(config["training"]["cpu_threads"])
        seed_everything(config["seed"])
        info = environment()
        save_json(output / "environment.json", info)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Real training is not silently moved to CPU; preserve this evidence and fix the environment.")
        device = torch.device("cuda:0")
        emit(f"GPU: {info['gpu']}; torch={torch.__version__}; CUDA build={torch.version.cuda}")
        size = config["model"]
        train_dataset = CalculatorDataset(Path(config["data_root"]), "train", seed=config["seed"],
                                          min_size=size["min_size"], max_size=size["max_size"], augment=True,
                                          horizontal_flip_probability=config["training"]["horizontal_flip_probability"])
        validation_dataset = CalculatorDataset(Path(config["data_root"]), "valid", seed=config["seed"],
                                               min_size=size["min_size"], max_size=size["max_size"], augment=False)
        if set(train_dataset.image_paths) & set(validation_dataset.image_paths):
            raise RuntimeError("Training and validation paths overlap.")
        manifest = {}
        for split, dataset in (("train", train_dataset), ("valid", validation_dataset)):
            manifest[split] = {"images": len(dataset), "image_ids": dataset.image_ids,
                               "review_approval_sha256": dataset.approval_sha256,
                               "approved_artifact_sha256": dataset.verified_artifacts,
                               "frozen_list": str(dataset.list_path), "frozen_list_sha256": sha256(dataset.list_path),
                               "annotations": str(dataset.annotation_path), "annotations_sha256": sha256(dataset.annotation_path),
                               "image_paths": [str(path) for path in dataset.image_paths]}
        manifest["test_policy"] = "No test list, image or annotation was opened by the training/validation datasets. Only the listed paths are loaded."
        save_json(output / "data_manifest.json", manifest)
        emit(f"Data: {len(train_dataset)} train / {len(validation_dataset)} validation; no test loader exists")
        summary["status"] = "smoke_running"
        save_json(output / "run.json", summary)
        smoke(config, train_dataset, validation_dataset, device, output, emit)
        if smoke_only:
            summary.update(status="smoke_passed_training_not_started", completed_at_utc=datetime.now(timezone.utc).isoformat())
            save_json(output / "run.json", summary)
            return 0
        # Reset all seeds and weights: the diagnostic optimizer step is not epoch 1.
        seed_everything(config["seed"])
        model = build_model(config["model"], pretrained=True).to(device)
        optimizer = optimizer_for(model, config)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"],
                                                    gamma=config["training"]["scheduler_gamma"])
        scaler = torch.amp.GradScaler("cuda", init_scale=config["training"].get("initial_amp_scale", 1024.0),
                                      enabled=config["training"]["mixed_precision"])
        generator = torch.Generator().manual_seed(config["seed"])
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0,
                                  collate_fn=collate_fn, worker_init_fn=seed_worker, generator=generator)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=0,
                                       collate_fn=collate_fn, worker_init_fn=seed_worker)
        best, significant_best, bad_epochs = -float("inf"), -float("inf"), 0
        best_path = output / "checkpoint_best.pt"
        training_started = time.monotonic()
        summary["status"] = "training"
        save_json(output / "run.json", summary)
        emit(f"REAL TRAINING START: declared {config['training']['epochs']} epochs, batch1, aspect-preserving short={size['min_size']}, long<={size['max_size']}")
        stop_reason = "completed_initial_epoch_budget"
        for epoch in range(1, config["training"]["epochs"] + 1):
            epoch_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            sums = {}
            learning_rate = optimizer.param_groups[0]["lr"]
            for step, (images, targets) in enumerate(train_loader, 1):
                result = train_step(model, optimizer, scaler, images, targets, device, config)
                for key, value in result.items():
                    sums[key] = sums.get(key, 0.0) + value
                append_json(output / "steps.jsonl", {"epoch": epoch, "step": step, "learning_rate": learning_rate, **result})
                if step == 1 or step % 10 == 0:
                    emit(f"EPOCH {epoch}/{config['training']['epochs']} STEP {step}/{len(train_loader)} loss={result['loss_total']:.6f} lr={learning_rate:g}")
            mean_losses = {key: value / len(train_loader) for key, value in sums.items()}
            validation = evaluate(model, validation_loader, device, config, output / "validation" / f"epoch_{epoch:02}")
            metric = float(validation[config["validation"]["selection_metric"]])
            if not np.isfinite(metric):
                raise RuntimeError("Validation selection metric is not finite.")
            epoch_result = {"epoch": epoch, "learning_rate": learning_rate, "mean_training_losses": mean_losses,
                            "validation": validation, "elapsed_seconds": time.monotonic() - epoch_started, **cuda_memory(device)}
            append_json(output / "epochs.jsonl", epoch_result)
            emit(f"EPOCH {epoch} VALIDATION AP50:95={metric:.6f} AP50={validation['segmentation_mAP_0.50']:.6f} meanGTIoU={validation['mean_gt_mask_iou']:.6f} train_loss={mean_losses['loss_total']:.6f}")
            if metric > best:
                best = metric
                checkpoint = {"schema_version": 1, "architecture": "maskrcnn_resnet50_fpn", "num_classes": 2,
                              "class_names": config["class_names"], "model_config": config["model"],
                              "training_config": config, "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                              "epoch": epoch, "selection_metric": config["validation"]["selection_metric"],
                              "validation_metrics": validation, "selection_split": "valid", "seed": config["seed"],
                              "command": command, "versions": {"torch": str(torch.__version__), "torchvision": info['packages'].get('torchvision')}}
                save_torch(best_path, checkpoint)
                digest = sha256(best_path)
                (output / "checkpoint_best.sha256").write_text(f"{digest}  checkpoint_best.pt\n", encoding="utf-8")
                selection = {"checkpoint": str(best_path), "sha256": digest, "epoch": epoch,
                             "selection_metric": config["validation"]["selection_metric"], "value": metric,
                             "selection_split": "valid", "validation_images": len(validation_dataset),
                             "tie_breaker": "Earliest epoch retained when validation AP ties", "command": command, "test_data_used": False}
                save_json(output / "selection.json", selection)
                summary.update(best_epoch=epoch, best_validation_metric=metric, checkpoint_sha256=digest,
                               checkpoint=str(best_path))
                emit(f"NEW VALIDATION BEST: epoch={epoch} AP={metric:.6f} SHA256={digest}")
                del checkpoint
            if metric > significant_best + config["training"]["minimum_validation_improvement"]:
                significant_best, bad_epochs = metric, 0
            else:
                bad_epochs += 1
            scheduler.step()
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != learning_rate:
                append_json(output / "configuration_changes.jsonl", {"epoch_completed": epoch, "field": "learning_rate",
                            "before": learning_rate, "after": new_lr, "reason": "Predeclared StepLR schedule; no test evidence"})
            save_torch(output / "last_training_state.pt", {"epoch": epoch, "model_state_dict": model.state_dict(),
                       "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                       "scaler_state_dict": scaler.state_dict(), "configuration": config, "seed": config["seed"]})
            summary.update(completed_epochs=epoch, last_epoch=epoch_result,
                           elapsed_training_seconds=time.monotonic() - training_started)
            save_json(output / "run.json", summary)
            if (epoch >= config["training"]["early_stopping_minimum_epochs"]
                    and bad_epochs >= config["training"]["early_stopping_patience"]):
                stop_reason = "validation_plateau_early_stop"
                break
            if (time.monotonic() - training_started) / 60 >= config["training"]["maximum_wall_minutes"]:
                stop_reason = "declared_wall_time_budget_reached"
                break
        summary.update(status="training_finished", stop_reason=stop_reason,
                       completed_at_utc=datetime.now(timezone.utc).isoformat(), test_data_used=False)
        save_json(output / "run.json", summary)
        emit(f"TRAINING FINISHED: {summary['completed_epochs']} epochs; reason={stop_reason}; selected epoch={summary['best_epoch']} validation AP={summary['best_validation_metric']:.6f}")
        return 0
    except Exception as exc:
        out_of_memory = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        summary.update(status="failed_memory" if out_of_memory else "failed", error=str(exc),
                       failed_at_utc=datetime.now(timezone.utc).isoformat(), traceback=traceback.format_exc())
        save_json(output / "run.json", summary)
        emit(f"RUN FAILED: {exc}")
        emit("Existing raw logs/checkpoints are retained. Do not claim completion; record any changed configuration and rerun smoke before training.")
        return 2
    finally:
        warnings.showwarning = original_showwarning
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("models/training_config.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    arguments = parser.parse_args(argv)
    output = arguments.output or Path("models/runs") / datetime.now(timezone.utc).strftime("maskrcnn_%Y%m%dT%H%M%SZ")
    try:
        return run(arguments.config, output, arguments.smoke_only)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
