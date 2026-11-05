"""Known segmentation outcomes, with in-memory loaders and fake predictions."""

import copy
import unittest
from unittest.mock import patch

import torch

from models.validation import ValidationError, evaluate


class FakeModel(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.position = 0

    def forward(self, images):
        if self.training or not torch.is_inference_mode_enabled():
            raise AssertionError("Validation must use eval and inference_mode.")
        result = self.outputs[self.position:self.position + len(images)]
        self.position += len(images)
        return result


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "model": {"num_classes": 2, "box_score_thresh": 0.05},
            "training": {"mixed_precision": True},
            "validation": {"mask_threshold": 0.5, "precision_recall_score_threshold": 0.5,
                           "precision_recall_mask_iou_threshold": 0.5},
        }
        self.first = torch.zeros((16, 16), dtype=torch.uint8)
        self.first[2:6, 2:6] = 1
        self.second = torch.zeros((16, 16), dtype=torch.uint8)
        self.second[10:14, 10:14] = 1

    def loader(self, masks, image_id=7):
        masks = torch.stack(masks) if masks else torch.empty((0, 16, 16), dtype=torch.uint8)
        target = {"masks": masks, "labels": torch.ones(len(masks), dtype=torch.int64),
                  "image_id": torch.tensor(image_id), "iscrowd": torch.zeros(len(masks), dtype=torch.int64)}
        return [([torch.zeros((3, 16, 16))], [target])]

    def prediction(self, masks, scores=None):
        probabilities = torch.stack(masks).float()[:, None] if masks else torch.empty((0, 1, 16, 16))
        return {"masks": probabilities, "scores": torch.tensor(scores if scores is not None else [0.9] * len(masks)),
                "labels": torch.ones(len(masks), dtype=torch.int64)}

    def test_perfect_mask_has_unit_ap_precision_recall_and_iou_without_file_access(self):
        model = FakeModel([self.prediction([self.first])])
        model.train()
        with patch("builtins.open", side_effect=AssertionError("No source files may be opened")), \
                patch("pathlib.Path.open", side_effect=AssertionError("No source files may be opened")):
            result = evaluate(model, self.loader([self.first]), "cpu", self.config)
        for key in ("segmentation_mAP_0.50_0.95", "segmentation_mAP_0.50", "precision_at_0.5", "recall_at_0.5", "mean_gt_mask_iou"):
            self.assertAlmostEqual(result[key], 1.0, msg=key)
        self.assertTrue(model.training)
        self.assertFalse(result["mixed_precision"])
        self.assertIn("Average Precision", result["coco_eval_stdout"])
        self.assertEqual((result["images"], result["gt_instances"], result["predictions"]), (1, 1, 1))

    def test_empty_predictions_have_zero_metrics_and_all_ground_truth_unmatched(self):
        result = evaluate(FakeModel([self.prediction([])]), self.loader([self.first]), "cpu", self.config)
        for key in ("segmentation_mAP_0.50_0.95", "segmentation_mAP_0.50", "precision_at_0.5", "recall_at_0.5", "mean_gt_mask_iou"):
            self.assertEqual(result[key], 0, key)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["predictions"], 0)

    def test_duplicate_predictions_count_false_positive_without_duplicate_iou_credit(self):
        result = evaluate(FakeModel([self.prediction([self.first, self.first], [0.9, 0.8])]),
                          self.loader([self.first]), "cpu", self.config)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["precision_at_0.5"], 0.5)
        self.assertEqual(result["recall_at_0.5"], 1)
        self.assertEqual(result["mean_gt_mask_iou"], 1)

    def test_empty_binary_prediction_is_preserved_as_a_false_positive(self):
        result = evaluate(FakeModel([self.prediction([torch.zeros_like(self.first)])]),
                          self.loader([self.first]), "cpu", self.config)
        self.assertEqual(result["predictions"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)
        self.assertEqual(result["segmentation_mAP_0.50_0.95"], 0)
        self.assertEqual(result["mean_gt_mask_iou"], 0)

    def test_unmatched_ground_truth_contributes_zero_to_mean_iou(self):
        result = evaluate(FakeModel([self.prediction([self.first])]),
                          self.loader([self.first, self.second]), "cpu", self.config)
        self.assertEqual(result["mean_gt_mask_iou"], 0.5)
        self.assertEqual(result["recall_at_0.5"], 0.5)
        self.assertEqual(result["false_negatives"], 1)

    def test_low_score_predictions_remain_in_ap_but_not_operating_metrics(self):
        result = evaluate(FakeModel([self.prediction([self.first], [0.4])]),
                          self.loader([self.first]), "cpu", self.config)
        self.assertAlmostEqual(result["segmentation_mAP_0.50_0.95"], 1)
        self.assertEqual(result["operating_point_predictions"], 0)
        self.assertEqual(result["mean_gt_mask_iou"], 0)
        self.assertEqual(result["recall_at_0.5"], 0)

    def test_partial_overlap_receives_iou_credit_even_below_tp_threshold(self):
        partial = torch.zeros_like(self.first)
        partial[2:4, 2:5] = 1  # 6 / 16 overlap, entirely inside the GT mask.
        result = evaluate(FakeModel([self.prediction([partial])]), self.loader([self.first]), "cpu", self.config)
        self.assertEqual(result["mean_gt_mask_iou"], 6 / 16)
        self.assertEqual(result["true_positives"], 0)
        self.assertEqual(result["false_positives"], 1)

    def test_two_predictions_cannot_credit_the_same_of_two_ground_truth_instances(self):
        result = evaluate(FakeModel([self.prediction([self.first, self.first], [0.9, 0.8])]),
                          self.loader([self.first, self.second]), "cpu", self.config)
        self.assertEqual(result["mean_gt_mask_iou"], 0.5)
        self.assertEqual(result["true_positives"], 1)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)

    def test_nonfinite_scores_duplicate_ids_and_threshold_search_are_rejected(self):
        prediction = self.prediction([self.first], [float("nan")])
        with self.assertRaises(ValidationError):
            evaluate(FakeModel([prediction]), self.loader([self.first]), "cpu", self.config)
        duplicated_loader = self.loader([self.first]) + self.loader([self.first])
        with self.assertRaisesRegex(ValidationError, "Duplicate"):
            evaluate(FakeModel([self.prediction([self.first])] * 2), duplicated_loader, "cpu", self.config)
        config = copy.deepcopy(self.config)
        config["validation"]["precision_recall_score_threshold"] = 0.7
        with self.assertRaisesRegex(ValidationError, "fixed"):
            evaluate(FakeModel([]), [], "cpu", config)


if __name__ == "__main__":
    unittest.main()
