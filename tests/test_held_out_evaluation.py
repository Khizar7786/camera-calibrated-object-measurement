"""Synthetic held-out membership, mask/AP separation and evidence contracts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from models.data import CalculatorDataset, TrainingDataError
from models.evaluate_test import GroundTruthCapture, audit_predictions, committed_file, replay_ap
from models.test_data import FrozenTestDataset
from models.validation import evaluate
from preprocessing.coco_masks import encode_mask


class HeldOutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.corrected = self.root / "corrected"

    def fixture(self, count=7):
        for root in (self.root, self.corrected):
            (root / "test").mkdir(parents=True)
            (root / "splits").mkdir()
        mask = np.zeros((20, 12), dtype=np.uint8)
        mask[4:14, 2:8] = 1
        images, annotations = [], []
        for index in range(count):
            name = f"calculator_{index}.png"
            cv2.imwrite(str(self.corrected / "test" / name), np.repeat((mask * 255)[..., None], 3, axis=2))
            images.append({"id": index, "file_name": name, "width": 12, "height": 20})
            annotations.append({"id": index + 10, "image_id": index, "category_id": 1, "iscrowd": 0,
                                "segmentation": encode_mask(mask), "bbox": [2, 4, 6, 10], "area": 60})
        coco = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "Calculator"}]}
        (self.root / "test/_annotations.coco.json").write_text(json.dumps(coco))
        coco["measure_it_preprocessing"] = {"frozen_split": "test", "manual_review_status": "passed"}
        (self.corrected / "test/_annotations.coco.json").write_text(json.dumps(coco))
        names = [f"test/{image['file_name']}" for image in reversed(images)]
        for root in (self.root, self.corrected):
            (root / "splits/test.txt").write_bytes(("\n".join(names) + "\n").encode())
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        frozen = {relative: digest(self.root / relative) for relative in ("splits/test.txt", "test/_annotations.coco.json")}
        processing = {"source_freeze_checksums": frozen}
        (self.corrected / "processing_run.json").write_text(json.dumps(processing))
        artifacts = {p.relative_to(self.corrected).as_posix(): digest(p) for p in self.corrected.rglob("*") if p.is_file()}
        approval = {"review_status": "passed", "reviewed_by": "user", "artifact_sha256": artifacts,
                    "processing_run_sha256": digest(self.corrected / "processing_run.json"),
                    "scope": {"split_images": {"test": count}}}
        (self.corrected / "review_approval.json").write_text(json.dumps(approval))
        run = {"manual_review_complete": True, "ready_for_training": True, "status": "reviewed_passed",
               "image_size": [12, 20], "source_freeze_checksums": frozen,
               "manual_review": {"status": "passed", "approval_file": "review_approval.json",
                                 "approval_sha256": digest(self.corrected / "review_approval.json"),
                                 "processing_snapshot": "processing_run.json"}}
        (self.corrected / "run.json").write_text(json.dumps(run))
        return names

    def test_exact_seven_list_order_geometry_and_no_other_split_access(self):
        names = self.fixture()
        original = Path.open

        def guarded(path, *args, **kwargs):
            self.assertNotIn("train", path.parts)
            self.assertNotIn("valid", path.parts)
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guarded):
            data = FrozenTestDataset(self.root, min_size=6, max_size=10)
            image, target = data[0]
        self.assertEqual([f"test/{r['file_name']}" for r in data.records], names)
        self.assertEqual(data.image_ids, list(reversed(range(7))))
        self.assertEqual(tuple(image.shape), (3, 10, 6))
        self.assertEqual(target["labels"].tolist(), [1])
        self.assertFalse(data.augment)
        data.augment = True
        with self.assertRaisesRegex(TrainingDataError, "augmentation is forbidden"):
            data[0]
        with self.assertRaisesRegex(TrainingDataError, "test is forbidden"):
            CalculatorDataset(self.root, "test")

    def test_six_images_or_changed_approved_pixels_are_rejected(self):
        self.fixture(count=6)
        with self.assertRaisesRegex(TrainingDataError, "exactly seven"):
            FrozenTestDataset(self.root)
        changed = self.corrected / "test/calculator_0.png"
        changed.write_bytes(b"changed")
        with self.assertRaisesRegex(TrainingDataError, "Approved artifact changed"):
            FrozenTestDataset(self.root)

    def test_committed_bytes_must_match_git_object(self):
        path = self.root / "record.txt"
        self.root.mkdir()
        path.write_text("frozen\n")
        with patch("models.evaluate_test.PROJECT", self.root), \
                patch("models.evaluate_test.subprocess.check_output", side_effect=["abc\n", "def\n"]):
            with self.assertRaisesRegex(ValueError, "differs from committed"):
                committed_file(path, "HEAD")

    def test_low_score_mask_stays_in_AP_but_missed_GT_is_zero_at_operating_point(self):
        mask = torch.zeros((1, 10, 6), dtype=torch.uint8)
        mask[:, 2:8, 1:5] = 1
        target = {"image_id": torch.tensor(4), "masks": mask, "labels": torch.tensor([1]),
                  "iscrowd": torch.tensor([0]), "annotation_ids": torch.tensor([44])}

        class Detector(torch.nn.Module):
            def forward(self, images):
                return [{"masks": mask[:, None].float(), "scores": torch.tensor([0.2]), "labels": torch.tensor([1])}]

        config = json.loads(Path("models/training_config.json").read_text())
        output = Path(self.temp.name) / "evaluation"
        captured = GroundTruthCapture([([torch.zeros(3, 10, 6)], [target])], lambda message: None)
        metrics = evaluate(Detector(), captured, "cpu", config, output)
        self.assertAlmostEqual(metrics["segmentation_mAP_0.50_0.95"], 1.0)
        self.assertEqual(metrics["precision_at_0.5"], 0.0)
        self.assertEqual(metrics["recall_at_0.5"], 0.0)
        self.assertEqual(metrics["mean_gt_mask_iou"], 0.0)
        predictions = json.loads((output / "predictions.json").read_text())
        self.assertEqual(len(predictions), 1)
        self.assertLess(predictions[0]["score"], 0.5)
        (output / "ground_truth.json").write_text(json.dumps(captured.ground_truth))
        replay_ap(output, metrics)
        self.assertTrue((output / "coco_arrays.npz").exists())
        rows = audit_predictions(captured.ground_truth, predictions,
                                 SimpleNamespace(records=[{"id": 4, "file_name": "synthetic.png"}]))
        self.assertEqual(rows[0]["false_negatives"], 1)
        self.assertEqual(rows[0]["mean_gt_mask_iou"], 0.0)
        self.assertEqual(rows[0]["AP_predictions_below_operating_score_cutoff"], 1)


if __name__ == "__main__":
    unittest.main()
