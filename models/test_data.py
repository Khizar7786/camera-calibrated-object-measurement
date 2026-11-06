"""Explicit held-out access, separate from the train/validation-only loader."""

from pathlib import Path, PurePosixPath

from .data import CalculatorDataset, TrainingDataError, _positive_int, _seed


class FrozenTestDataset(CalculatorDataset):
    """Reuse approved geometry/targets, without allowing test into training.

    The training class retains its original test rejection. This separately
    invoked dataset allows only test leaves and shared approval metadata,
    disables augmentation, and requires exactly seven frozen filenames.
    """

    def __init__(self, data_root, *, min_size=512, max_size=960, seed=42):
        self.split = "test"
        self.seed = _seed(seed)
        self.min_size = _positive_int(min_size, "min_size")
        self.max_size = _positive_int(max_size, "max_size")
        if self.max_size < self.min_size:
            raise TrainingDataError("max_size must be at least min_size.")
        self.augment = False
        self.horizontal_flip_probability = 0.0
        self.data_root = Path(data_root).resolve()
        self.corrected_root = (self.data_root / "corrected").resolve()
        self._hash_cache = {}
        self._generators = {}
        self.verified_artifacts = {}
        self.list_path = self._source_path("splits/test.txt")
        self.corrected_list_path = self._corrected_path("splits/test.txt")
        self.annotation_path = self._corrected_path("test/_annotations.coco.json")
        self.source_annotation_path = self._source_path("test/_annotations.coco.json")
        self._verify_approval()
        self._load_records()
        names = self._frozen_bytes.decode("utf-8").splitlines()
        if len(names) != 7:
            raise TrainingDataError("Held-out evaluation requires exactly seven frozen test images.")
        by_name = {f"test/{record['file_name']}": record for record in self.records}
        self.records = [by_name[name] for name in names]
        self.image_ids = [record["id"] for record in self.records]
        self.image_paths = [Path(record["path"]) for record in self.records]
        self.max_instances_index = max(range(len(self.records)), key=lambda i: len(self.records[i]["annotations"]))

    def _safe_path(self, root, relative):
        path = PurePosixPath(relative)
        parts = path.parts
        allowed_metadata = {"run.json", "review_approval.json", "processing_run.json", "splits/test.txt"}
        if not parts or "\\" in relative or path.is_absolute() or ".." in parts \
                or not (relative in allowed_metadata or parts[0] == "test"):
            raise TrainingDataError(f"Not an allowed held-out path: {relative}")
        resolved = root.joinpath(*parts).resolve()
        try:
            actual = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise TrainingDataError(f"Held-out path escapes its dataset: {relative}") from exc
        if actual not in allowed_metadata and not actual.startswith("test/"):
            raise TrainingDataError(f"Held-out path resolves outside test: {relative}")
        return resolved

    def _flip(self):
        if self.augment or self.horizontal_flip_probability != 0.0:
            raise TrainingDataError("Held-out augmentation is forbidden.")
        return False
