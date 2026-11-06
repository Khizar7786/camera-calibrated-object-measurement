"""Approved train/validation inputs, geometry, labels and held-out exclusion."""

import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from models.data import CalculatorDataset, TrainingDataError, collate_fn, seed_everything, seed_worker
from preprocessing.coco_masks import encode_mask


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "dataset"
        self.corrected = self.root / "corrected"
        self.width, self.height = 12, 20
        self.categories = [{"id": 0, "name": "Calc-images"}, {"id": 1, "name": "Calculator"}]
        self.mask = np.zeros((self.height, self.width), np.uint8)
        self.mask[4:12, 2:7] = 1
        self.second_mask = np.zeros_like(self.mask)
        self.second_mask[13:18, 7:11] = 1
        for split, image_id in [("train", 10), ("valid", 20)]:
            (self.root / split).mkdir(parents=True)
            (self.corrected / split / "masks").mkdir(parents=True)
            (self.root / "splits").mkdir(exist_ok=True)
            (self.corrected / "splits").mkdir(exist_ok=True)
            names = f"{split}/calculator.png\n".encode()
            (self.root / "splits" / f"{split}.txt").write_bytes(names)
            (self.corrected / "splits" / f"{split}.txt").write_bytes(names)
            image = np.zeros((self.height, self.width, 3), np.uint8)
            image[self.mask != 0] = [0, 0, 255]
            image[self.second_mask != 0] = [0, 255, 0]
            self.assertTrue(cv2.imwrite(str(self.corrected / split / "calculator.png"), image))
            annotations = []
            for number, mask in enumerate([self.mask, self.second_mask], 1):
                annotation_id = image_id * 10 + number
                self.assertTrue(cv2.imwrite(str(self.corrected / split / "masks" / f"{annotation_id}.png"), mask * 255))
                annotations.append({"id": annotation_id, "image_id": image_id, "category_id": 1,
                                    "iscrowd": 0, "segmentation": encode_mask(mask),
                                    "bbox": [0, 0, self.width, self.height], "area": int(mask.sum())})
            coco = {"categories": self.categories,
                    "images": [{"id": image_id, "file_name": "calculator.png", "width": self.width, "height": self.height}],
                    "annotations": annotations}
            self.write_json(self.root / split / "_annotations.coco.json", coco)
            coco["measure_it_preprocessing"] = {"manual_review_status": "passed", "frozen_split": split}
            self.write_json(self.corrected / split / "_annotations.coco.json", coco)
        self.approve_fixture()

    @staticmethod
    def write_json(path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def approve_fixture(self):
        frozen = {}
        for split in ["train", "valid"]:
            for relative in [f"splits/{split}.txt", f"{split}/_annotations.coco.json"]:
                frozen[relative] = self.digest(self.root / relative)
        # Metadata may mention test; no test file or directory exists in fixture.
        frozen["splits/test.txt"] = "f" * 64
        processing = {"source_freeze_checksums": frozen, "manual_review_complete": False,
                      "ready_for_training": False, "status": "processed_pending_manual_review"}
        self.write_json(self.corrected / "processing_run.json", processing)
        manifest = {path.relative_to(self.corrected).as_posix(): self.digest(path)
                    for path in self.corrected.rglob("*")
                    if path.is_file() and path.name not in {"run.json", "review_approval.json"}}
        manifest["test/_annotations.coco.json"] = "e" * 64
        manifest["splits/test.txt"] = "f" * 64
        approval = {"review_status": "passed", "reviewed_by": "user",
                    "processing_run_sha256": self.digest(self.corrected / "processing_run.json"),
                    "artifact_sha256": manifest, "scope": {"split_images": {"train": 1, "valid": 1}}}
        self.write_json(self.corrected / "review_approval.json", approval)
        run = {"manual_review_complete": True, "ready_for_training": True, "status": "reviewed_passed",
               "image_size": [self.width, self.height], "source_freeze_checksums": frozen,
               "manual_review": {"status": "passed", "reviewed_by": "user", "approval_file": "review_approval.json",
                                 "approval_sha256": self.digest(self.corrected / "review_approval.json"),
                                 "processing_snapshot": "processing_run.json"}}
        self.write_json(self.corrected / "run.json", run)

    def mutate_corrected(self, split, function):
        path = self.corrected / split / "_annotations.coco.json"
        data = json.loads(path.read_text())
        function(data)
        self.write_json(path, data)
        self.approve_fixture()

    def dataset(self, split="train", **kwargs):
        return CalculatorDataset(self.root, split, min_size=self.width, max_size=self.height, **kwargs)

    def test_test_split_is_rejected_before_any_filesystem_access(self):
        class InaccessiblePath:
            def __fspath__(self):
                raise AssertionError("Dataset root must not be touched for test")
        with patch("models.data._read_json", side_effect=AssertionError("No file access permitted")):
            with self.assertRaisesRegex(TrainingDataError, "test is forbidden"):
                CalculatorDataset(InaccessiblePath(), "test")

    def test_train_and_validation_never_open_test_artifacts(self):
        original_open = Path.open
        opened = []

        def guarded_open(path, *args, **kwargs):
            self.assertNotIn("test", path.parts)
            self.assertFalse(path.parent.name == "splits" and path.name == "test.txt")
            opened.append(path)
            return original_open(path, *args, **kwargs)

        with patch("pathlib.Path.open", new=guarded_open):
            training = self.dataset()
            validation = self.dataset("valid")
            training[0]
            validation[0]
        self.assertTrue(opened)
        self.assertTrue(all(not key.startswith("test/") for key in training.verified_artifacts))
        self.assertEqual(training.max_instances_index, 0)

    def test_aspect_resize_and_torchvision_target_types(self):
        dataset = CalculatorDataset(self.root, "train", min_size=6, max_size=10)
        image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 10, 6))
        self.assertEqual(image.dtype, torch.float32)
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        self.assertEqual(target["masks"].dtype, torch.uint8)
        self.assertEqual(target["labels"].dtype, torch.int64)
        self.assertEqual(target["boxes"].dtype, torch.float32)
        self.assertEqual(target["iscrowd"].dtype, torch.int64)
        self.assertEqual(target["labels"].tolist(), [1, 1])
        self.assertEqual(target["annotation_ids"].tolist(), [101, 102])
        self.assertEqual(target["image_id"].item(), 10)
        self.assertEqual(dataset.sample_metadata(0)["resized_size"], [6, 10])
        for mask, box, area in zip(target["masks"].numpy(), target["boxes"].tolist(), target["area"].tolist()):
            rows, columns = np.nonzero(mask)
            self.assertEqual(box, [float(columns.min()), float(rows.min()), float(columns.max() + 1), float(rows.max() + 1)])
            self.assertEqual(area, float(mask.sum()))

    def test_rgb_order_and_joint_horizontal_flip_are_exact(self):
        normal = self.dataset()
        flipped = self.dataset(augment=True, horizontal_flip_probability=1.0)
        image, target = normal[0]
        flipped_image, flipped_target = flipped[0]
        self.assertEqual(float(image[0, 6, 4]), 1.0)
        self.assertEqual(float(image[2, 6, 4]), 0.0)
        torch.testing.assert_close(flipped_image, torch.flip(image, dims=[2]))
        torch.testing.assert_close(flipped_target["masks"], torch.flip(target["masks"], dims=[2]))
        expected = target["boxes"].clone()
        expected[:, 0] = self.width - target["boxes"][:, 2]
        expected[:, 2] = self.width - target["boxes"][:, 0]
        torch.testing.assert_close(flipped_target["boxes"], expected)
        torch.testing.assert_close(flipped_target["area"], target["area"])

    def test_one_pixel_wide_instance_gets_positive_exclusive_box(self):
        narrow = np.zeros((self.height, self.width), np.uint8)
        narrow[3:11, 5] = 1
        self.mutate_corrected("train", lambda coco: coco["annotations"][1].update(segmentation=encode_mask(narrow)))
        _, target = self.dataset()[0]
        self.assertEqual(target["boxes"][1].tolist(), [5.0, 3.0, 6.0, 11.0])
        self.assertEqual(target["area"][1].item(), 8.0)

    def test_validation_augmentation_is_forbidden(self):
        with self.assertRaisesRegex(TrainingDataError, "Validation augmentation"):
            self.dataset("valid", augment=True)

    def test_unapproved_or_stale_approval_refuses_construction(self):
        path = self.corrected / "run.json"
        run = json.loads(path.read_text())
        run["manual_review_complete"] = False
        self.write_json(path, run)
        with self.assertRaisesRegex(TrainingDataError, "manual review"):
            self.dataset()
        self.approve_fixture()
        approval_path = self.corrected / "review_approval.json"
        approval_path.write_text(approval_path.read_text() + " ")
        with self.assertRaisesRegex(TrainingDataError, "approval hash"):
            self.dataset()

    def test_changed_image_and_frozen_membership_are_rejected(self):
        image_path = self.corrected / "train" / "calculator.png"
        image_path.write_bytes(image_path.read_bytes() + b"changed")
        with self.assertRaisesRegex(TrainingDataError, "Approved artifact changed"):
            self.dataset()
        self.approve_fixture()
        (self.root / "splits" / "train.txt").write_text("train/another.png\n")
        with self.assertRaisesRegex(TrainingDataError, "frozen original membership"):
            self.dataset()

    def test_parent_category_is_not_used_as_a_foreground_class(self):
        self.mutate_corrected("train", lambda coco: coco["annotations"][0].update(category_id=0))
        with self.assertRaisesRegex(TrainingDataError, "category"):
            self.dataset()

    def test_empty_masks_and_masks_lost_on_resize_raise_instead_of_disappearing(self):
        empty = np.zeros((self.height, self.width), np.uint8)
        self.mutate_corrected("train", lambda coco: coco["annotations"][0].update(segmentation=encode_mask(empty)))
        dataset = self.dataset()
        with self.assertRaisesRegex(TrainingDataError, "empty source mask"):
            dataset[0]
        tiny = empty.copy()
        tiny[0, 0] = 1
        self.mutate_corrected("train", lambda coco: coco["annotations"][0].update(segmentation=encode_mask(tiny)))
        dataset = CalculatorDataset(self.root, "train", min_size=1, max_size=2)
        with self.assertRaisesRegex(TrainingDataError, "disappears at training resolution"):
            dataset[0]

    def test_hashes_are_checked_once_at_construction_not_each_epoch(self):
        dataset = self.dataset()
        with patch("models.data._digest", side_effect=AssertionError("hash repeated during sample loading")):
            dataset[0]
            dataset[0]

    def test_seeds_and_detection_collation(self):
        seed_everything(42)
        first = (random.random(), np.random.random(), torch.rand(1))
        seed_everything(42)
        second = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(first[:2], second[:2])
        torch.testing.assert_close(first[2], second[2])
        with patch("models.data.torch.initial_seed", return_value=123):
            seed_worker(0)
            first_worker = (random.random(), np.random.random())
            seed_worker(0)
            self.assertEqual(first_worker, (random.random(), np.random.random()))
        sample = self.dataset()[0]
        images, targets = collate_fn([sample, sample])
        self.assertIsInstance(images, list)
        self.assertIsInstance(targets, list)
        self.assertEqual(len(images), 2)


if __name__ == "__main__":
    unittest.main()
