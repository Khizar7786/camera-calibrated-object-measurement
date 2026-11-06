"""Two-class model and freeze-policy contracts; tests never download weights."""

import json
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

from models.model import build_model, model_summary


class TrainingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Network disabled in model tests")):
            cls.model = build_model({}, pretrained=False)

    def test_both_heads_predict_only_background_and_calculator(self):
        with torch.inference_mode():
            scores, boxes = self.model.roi_heads.box_predictor(torch.zeros((3, 1024)))
            masks = self.model.roi_heads.mask_predictor(torch.zeros((3, 256, 14, 14)))
        self.assertEqual(tuple(scores.shape), (3, 2))
        self.assertEqual(tuple(boxes.shape), (3, 8))
        self.assertEqual(tuple(masks.shape), (3, 2, 28, 28))

    def test_default_freezing_leaves_only_last_three_body_stages_trainable(self):
        for name, parameter in self.model.backbone.body.named_parameters():
            expected = name.startswith(("layer2.", "layer3.", "layer4."))
            self.assertEqual(parameter.requires_grad, expected, name)
        self.assertTrue(all(p.requires_grad for p in self.model.backbone.fpn.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.roi_heads.parameters()))

    def test_offline_backbone_matches_frozen_batch_norm_checkpoint_structure(self):
        body = self.model.backbone.body
        norms = [module for module in body.modules() if isinstance(module, FrozenBatchNorm2d)]
        self.assertGreater(len(norms), 0)
        self.assertTrue(all(module.eps == 0.0 for module in norms))
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in body.modules()))
        self.assertFalse(any(name.endswith("num_batches_tracked") for name in body.state_dict()))

    def test_backbone_freeze_extremes_work_without_pretrained_factory_weights(self):
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Network disabled in model tests")):
            for count in (0, 5):
                with self.subTest(trainable_backbone_layers=count):
                    model = build_model({"trainable_backbone_layers": count}, pretrained=False)
                    self.assertTrue(all(p.requires_grad == (count == 5) for p in model.backbone.body.parameters()))
                    self.assertTrue(all(p.requires_grad for p in model.backbone.fpn.parameters()))

    def test_prohibited_architectures_and_invalid_model_settings_fail_before_factory(self):
        invalid = (
            {"architecture": "yolo11n-seg"},
            {"architecture": "roboflow"},
            {"num_classes": 91},
            {"num_classes": 2.0},
            {"trainable_backbone_layers": 6},
            {"trainable_backbone_layers": True},
            {"min_size": 960, "max_size": 512},
            {"rpn_pre_nms_top_n_test": 10, "rpn_post_nms_top_n_test": 20},
            {"box_score_thresh": 0.9},
            {"box_nms_thresh": 0.8},
        )
        with patch("models.model.maskrcnn_resnet50_fpn") as factory:
            for config in invalid:
                with self.subTest(config=config), self.assertRaises(ValueError):
                    build_model(config, pretrained=False)
            factory.assert_not_called()

    def test_summary_records_real_heads_freezing_resolution_and_proposal_limits(self):
        summary = model_summary(self.model)
        json.dumps(summary, allow_nan=False)
        self.assertEqual(summary["box_classes"], 2)
        self.assertEqual(summary["mask_classes"], 2)
        self.assertEqual(summary["box_regression_outputs"], 8)
        self.assertFalse(summary["backbone_layers"]["layer1"]["trainable"])
        self.assertTrue(summary["backbone_layers"]["layer2"]["trainable"])
        self.assertEqual(summary["input_transform"]["min_size"], [512])
        self.assertEqual(summary["input_transform"]["max_size"], 960)
        self.assertEqual(summary["proposal_limits"], {
            "train_pre_nms": 1000, "train_post_nms": 500,
            "test_pre_nms": 1000, "test_post_nms": 300,
        })
        self.assertEqual(summary["box_score_threshold"], 0.05)
        self.assertEqual(summary["box_nms_threshold"], 0.5)
        self.assertEqual(summary["box_detections_per_image"], 100)
        self.assertGreater(summary["trainable_parameters"], 0)
        self.assertLess(summary["trainable_parameters"], summary["total_parameters"])


if __name__ == "__main__":
    unittest.main()
