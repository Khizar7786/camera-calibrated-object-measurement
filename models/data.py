"""Approved train/validation COCO inputs for the calculator Mask R-CNN.

This module never opens test images, test annotations or test split files.
Only the approved corrected dataset is consumed; original export labels are
read for identity/provenance checks, not paired with corrected images.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from calibration.camera import CalibrationError, read_image
from preprocessing.coco_masks import AnnotationError, decode_segmentation


class TrainingDataError(ValueError):
    """Unapproved, altered or geometrically invalid training data."""


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise TrainingDataError("Seed must be an integer in [0, 2**32).")
    return seed


def seed_everything(seed: int = 42) -> dict[str, Any]:
    """Seed supported RNGs and request deterministic PyTorch operations.

    warn_only keeps unsupported deterministic CUDA operations visible instead
    of claiming bit-for-bit reproducibility across hardware or library builds.
    Call before constructing a model or initializing the CUDA context.
    """
    seed = _seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    cv2.setNumThreads(1)
    return {"seed": seed, "cudnn_deterministic": True, "cudnn_benchmark": False,
            "deterministic_algorithms": True, "deterministic_warn_only": True,
            "tf32": False, "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"]}


def seed_worker(worker_id: int) -> None:
    """Initialize NumPy/Python from the DataLoader's seeded worker RNG."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[list, list]:
    if not batch:
        raise TrainingDataError("Cannot collate an empty batch.")
    images, targets = zip(*batch)
    return list(images), list(targets)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TrainingDataError(f"Expected a JSON object in {path}.")
    return data


def _digest(path: Path) -> str:
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError as exc:
        raise TrainingDataError(f"Cannot hash {path}: {exc}") from exc


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingDataError(f"{field} must be a positive integer.")
    return value


def _identifier(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingDataError(f"{field} must be a nonnegative integer.")
    return value


class CalculatorDataset(Dataset):
    """Immutable approved input membership with joint aspect-preserving resize.

    ``data_root`` is the dataset directory, containing ``splits/``, the original
    train/valid COCO exports and ``corrected/``. Dataset order follows the COCO
    image array; membership must exactly match the corresponding frozen list.
    Approval hashes for this split are checked once during construction, not
    during every epoch. Reconstruct the dataset if files intentionally change.
    """

    def __init__(self, data_root: str | Path, split: str, *, seed: int = 42,
                 min_size: int = 512, max_size: int = 960, augment: bool = False,
                 horizontal_flip_probability: float = 0.5) -> None:
        # Reject test before even resolving a caller-supplied filesystem path.
        if not isinstance(split, str) or split not in ("train", "valid"):
            raise TrainingDataError("Only 'train' and 'valid' are available during training; test is forbidden.")
        self.split = split
        self.seed = _seed(seed)
        self.min_size = _positive_int(min_size, "min_size")
        self.max_size = _positive_int(max_size, "max_size")
        if self.max_size < self.min_size:
            raise TrainingDataError("max_size must be at least min_size.")
        if type(augment) is not bool:
            raise TrainingDataError("augment must be Boolean.")
        if augment and split != "train":
            raise TrainingDataError("Validation augmentation is forbidden.")
        if isinstance(horizontal_flip_probability, bool) or not isinstance(horizontal_flip_probability, (int, float)) \
                or not math.isfinite(horizontal_flip_probability) or not 0 <= horizontal_flip_probability <= 1:
            raise TrainingDataError("horizontal_flip_probability must lie in [0, 1].")
        self.augment = augment
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.data_root = Path(data_root).resolve()
        self.corrected_root = (self.data_root / "corrected").resolve()
        self._hash_cache: dict[Path, str] = {}
        self._generators: dict[int, torch.Generator] = {}
        self.verified_artifacts: dict[str, str] = {}
        self.list_path = self._source_path(f"splits/{split}.txt")
        self.corrected_list_path = self._corrected_path(f"splits/{split}.txt")
        self.annotation_path = self._corrected_path(f"{split}/_annotations.coco.json")
        self.source_annotation_path = self._source_path(f"{split}/_annotations.coco.json")
        self._verify_approval()
        self._load_records()

    def _source_path(self, relative: str) -> Path:
        return self._safe_path(self.data_root, relative)

    def _corrected_path(self, relative: str) -> Path:
        return self._safe_path(self.corrected_root, relative)

    def _safe_path(self, root: Path, relative: str) -> Path:
        parts = PurePosixPath(relative).parts
        if not parts or "\\" in relative or PurePosixPath(relative).is_absolute() or ".." in parts \
                or "test" in parts:
            raise TrainingDataError(f"Unsafe or forbidden training path: {relative}")
        resolved = root.joinpath(*parts).resolve()
        try:
            actual_parts = resolved.relative_to(root).parts
        except ValueError as exc:
            raise TrainingDataError(f"Training path escapes its dataset: {relative}") from exc
        if "test" in actual_parts:
            raise TrainingDataError(f"Training path resolves into test: {relative}")
        return resolved

    def _hash(self, path: Path) -> str:
        if path not in self._hash_cache:
            self._hash_cache[path] = _digest(path)
        return self._hash_cache[path]

    def _verify_approval(self) -> None:
        run = _read_json(self._corrected_path("run.json"))
        manual = run.get("manual_review", {})
        if run.get("manual_review_complete") is not True or run.get("ready_for_training") is not True \
                or run.get("status") != "reviewed_passed" or manual.get("status") != "passed" \
                or manual.get("approval_file") != "review_approval.json":
            raise TrainingDataError("Corrected dataset has not passed explicit manual review.")
        approval_path = self._corrected_path("review_approval.json")
        self.approval_sha256 = self._hash(approval_path)
        if self.approval_sha256 != manual.get("approval_sha256"):
            raise TrainingDataError("Manual-review approval hash is stale or altered.")
        approval = _read_json(approval_path)
        if approval.get("review_status") != "passed" or approval.get("reviewed_by") != "user":
            raise TrainingDataError("A user-approved dataset is required before training.")
        manifest = approval.get("artifact_sha256")
        if not isinstance(manifest, dict):
            raise TrainingDataError("Approval has no artifact hash manifest.")
        required = {f"splits/{self.split}.txt", f"{self.split}/_annotations.coco.json", "processing_run.json"}
        if not required.issubset(manifest):
            raise TrainingDataError("Approval is missing the frozen list, annotations or processing snapshot.")
        if manual.get("processing_snapshot") != "processing_run.json":
            raise TrainingDataError("Unexpected processing snapshot reference.")
        snapshot_path = self._corrected_path("processing_run.json")
        if self._hash(snapshot_path) != manifest["processing_run.json"] \
                or self._hash(snapshot_path) != approval.get("processing_run_sha256"):
            raise TrainingDataError("Original processing snapshot hash does not match approval.")
        processing = _read_json(snapshot_path)
        frozen = processing.get("source_freeze_checksums", {})
        if frozen != run.get("source_freeze_checksums"):
            raise TrainingDataError("Current source freeze differs from the approved processing snapshot.")
        for relative, expected in manifest.items():
            # Metadata may describe test, but no test artifact is opened here.
            if relative.startswith(f"{self.split}/") or relative == f"splits/{self.split}.txt":
                if not isinstance(expected, str) or len(expected) != 64:
                    raise TrainingDataError(f"Invalid approved hash for {relative}.")
                path = self._corrected_path(relative)
                if self._hash(path) != expected:
                    raise TrainingDataError(f"Approved artifact changed: {relative}")
                self.verified_artifacts[relative] = expected
        try:
            original_list = self.list_path.read_bytes()
            corrected_list = self.corrected_list_path.read_bytes()
        except OSError as exc:
            raise TrainingDataError(f"Cannot read frozen split list: {exc}") from exc
        if original_list != corrected_list:
            raise TrainingDataError("Corrected split list differs from frozen original membership.")
        for relative, path in [(f"splits/{self.split}.txt", self.list_path),
                               (f"{self.split}/_annotations.coco.json", self.source_annotation_path)]:
            if self._hash(path) != frozen.get(relative):
                raise TrainingDataError(f"Original frozen source changed: {relative}")
        self.approval = approval
        self.source_freeze_checksums = {key: frozen[key] for key in
                                        (f"splits/{self.split}.txt", f"{self.split}/_annotations.coco.json")}
        self.annotation_sha256 = self._hash(self.annotation_path)
        self.split_sha256 = self._hash(self.list_path)
        self._frozen_bytes = original_list
        size = run.get("image_size")
        if not isinstance(size, list) or len(size) != 2:
            raise TrainingDataError("Corrected dataset has no canonical [width, height].")
        self.canonical_size = tuple(_positive_int(value, "canonical size") for value in size)

    def _load_records(self) -> None:
        coco = _read_json(self.annotation_path)
        original = _read_json(self.source_annotation_path)
        if coco.get("categories") != original.get("categories"):
            raise TrainingDataError("Corrected category identities differ from the original export.")
        categories = coco.get("categories", [])
        calculator_ids = [category.get("id") for category in categories
                          if str(category.get("name", "")).strip().lower() == "calculator"]
        if calculator_ids != [1]:
            raise TrainingDataError("Expected the original Calculator category id 1; parent id 0 is not foreground.")
        meta = coco.get("measure_it_preprocessing", {})
        if meta.get("manual_review_status") != "passed" or meta.get("frozen_split") != self.split:
            raise TrainingDataError("COCO metadata is not approved for this frozen split.")
        original_images = {image["id"]: image for image in original.get("images", [])}
        original_annotations = {ann["id"]: ann for ann in original.get("annotations", [])}
        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        if not isinstance(images, list) or not images or not isinstance(annotations, list):
            raise TrainingDataError("Corrected COCO needs nonempty images and an annotation list.")
        image_ids = [_identifier(image.get("id"), "image id") for image in images]
        annotation_ids = [_identifier(ann.get("id"), "annotation id") for ann in annotations]
        if len(set(image_ids)) != len(image_ids) or len(set(annotation_ids)) != len(annotation_ids):
            raise TrainingDataError("Duplicate COCO image or annotation IDs.")
        if set(image_ids) != set(original_images) or set(annotation_ids) != set(original_annotations):
            raise TrainingDataError("Original image/instance IDs were added or removed.")
        grouped: dict[int, list[dict]] = defaultdict(list)
        for annotation in annotations:
            before = original_annotations[annotation["id"]]
            if annotation.get("category_id") != 1 or annotation.get("category_id") != before.get("category_id") \
                    or annotation.get("image_id") != before.get("image_id"):
                raise TrainingDataError("An instance changed category or image identity.")
            if annotation.get("image_id") not in original_images:
                raise TrainingDataError("Annotation references an unknown image.")
            if annotation.get("iscrowd", 0) not in (0, 1):
                raise TrainingDataError("iscrowd must be 0 or 1.")
            grouped[annotation["image_id"]].append(annotation)
        try:
            names = self._frozen_bytes.decode("utf-8").splitlines()
        except UnicodeError as exc:
            raise TrainingDataError("Frozen split list must be UTF-8.") from exc
        if len(names) != len(set(names)) or not names:
            raise TrainingDataError("Frozen split list is empty or contains duplicate paths.")
        records = []
        observed_names = []
        for image in images:
            image_id = image["id"]
            filename = image.get("file_name")
            if not isinstance(filename, str) or PurePosixPath(filename).name != filename or "\\" in filename:
                raise TrainingDataError("COCO image filenames must be plain filenames in their split.")
            if filename != original_images[image_id].get("file_name"):
                raise TrainingDataError("A corrected image changed its original filename.")
            width, height = (_positive_int(image.get(key), f"image {key}") for key in ("width", "height"))
            if (width, height) != self.canonical_size:
                raise TrainingDataError("COCO image dimensions differ from the approved camera frame.")
            relative = f"{self.split}/{filename}"
            if relative not in self.verified_artifacts:
                raise TrainingDataError(f"Image is not covered by review approval: {relative}")
            if not grouped[image_id]:
                raise TrainingDataError(f"Image {image_id} has an empty target; no instances are silently discarded.")
            records.append({**image, "annotations": grouped[image_id],
                            "path": str(self._corrected_path(relative))})
            observed_names.append(relative)
        if set(observed_names) != set(names) or len(observed_names) != len(names):
            raise TrainingDataError("COCO membership does not exactly match the frozen filename list.")
        approved_count = self.approval.get("scope", {}).get("split_images", {}).get(self.split)
        if approved_count != len(records):
            raise TrainingDataError("Image count differs from the user-approved split scope.")
        self.records = records
        self.image_ids = image_ids
        self.image_paths = [Path(record["path"]) for record in records]
        self.max_instances_index = max(range(len(records)), key=lambda index: len(records[index]["annotations"]))

    def __len__(self) -> int:
        return len(self.records)

    def _resized_size(self, width: int, height: int) -> tuple[int, int]:
        factor = min(self.min_size / min(width, height), self.max_size / max(width, height))
        return max(1, round(width * factor)), max(1, round(height * factor))

    def _flip(self) -> bool:
        if not self.augment:
            return False
        worker = get_worker_info()
        key = -1 if worker is None else worker.id
        if key not in self._generators:
            initial_seed = self.seed if worker is None else (self.seed + worker.seed) % (2**63)
            self._generators[key] = torch.Generator().manual_seed(initial_seed)
        return bool(torch.rand((), generator=self._generators[key]).item() < self.horizontal_flip_probability)

    def sample_metadata(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {"split": self.split, "image_id": record["id"], "file_name": record["file_name"],
                "image_path": record["path"], "annotation_path": str(self.annotation_path),
                "annotation_ids": [ann["id"] for ann in record["annotations"]],
                "instance_count": len(record["annotations"]),
                "original_size": [record["width"], record["height"]],
                "resized_size": list(self._resized_size(record["width"], record["height"])),
                "resize": "uniform scale bounded by min/max, dimensions rounded; no stretch",
                "image_interpolation": "INTER_AREA downsample / INTER_LINEAR upsample",
                "mask_interpolation": "INTER_NEAREST_EXACT",
                "horizontal_flip_probability": self.horizontal_flip_probability if self.augment else 0.0}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records[index]
        width, height = record["width"], record["height"]
        try:
            image = read_image(record["path"])
        except CalibrationError as exc:
            raise TrainingDataError(str(exc)) from exc
        if (image.shape[1], image.shape[0]) != (width, height):
            raise TrainingDataError(f"Actual image size differs from approved COCO metadata: {record['file_name']}")
        size = self._resized_size(width, height)
        interpolation = cv2.INTER_AREA if size[0] < width or size[1] < height else cv2.INTER_LINEAR
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size, interpolation=interpolation)
        flipped = self._flip()
        if flipped:
            image = image[:, ::-1]
        masks, boxes, areas, crowds, annotation_ids = [], [], [], [], []
        for annotation in record["annotations"]:
            try:
                mask = decode_segmentation(annotation.get("segmentation"), width, height)
            except AnnotationError as exc:
                raise TrainingDataError(f"Annotation {annotation['id']} cannot be decoded: {exc}") from exc
            if not np.any(mask):
                raise TrainingDataError(f"Annotation {annotation['id']} has an empty source mask.")
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST_EXACT)
            if flipped:
                mask = mask[:, ::-1]
            rows = np.flatnonzero(np.any(mask, axis=1))
            columns = np.flatnonzero(np.any(mask, axis=0))
            if len(rows) == 0 or len(columns) == 0:
                raise TrainingDataError(f"Annotation {annotation['id']} disappears at training resolution; it was not dropped.")
            # Exclusive upper bounds retain valid one-pixel-wide instances.
            boxes.append([float(columns[0]), float(rows[0]), float(columns[-1] + 1), float(rows[-1] + 1)])
            masks.append(np.ascontiguousarray(mask, dtype=np.uint8))
            areas.append(float(np.count_nonzero(mask)))
            crowds.append(int(annotation.get("iscrowd", 0)))
            annotation_ids.append(annotation["id"])
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).to(torch.float32).div_(255)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(masks), dtype=torch.int64),
            "masks": torch.from_numpy(np.stack(masks)),
            "image_id": torch.tensor(record["id"], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.tensor(crowds, dtype=torch.int64),
            "annotation_ids": torch.tensor(annotation_ids, dtype=torch.int64),
        }
        return image_tensor, target
