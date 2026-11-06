"""Run evidence and smoke-state contracts without downloads or GPU training."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from models.train import run, smoke, validate_config


class FakeDataset:
    def __init__(self, augment):
        self.augment = augment
        self.max_instances_index = 0
        self.image_ids = [17]
        self.augmentation_calls = 0

    def __getitem__(self, index):
        if self.augment:
            self.augmentation_calls += 1
        return torch.ones(3, 16, 12), {"labels": torch.tensor([1]), "boxes": torch.tensor([[1., 1., 7., 9.]]),
                                      "masks": torch.ones(1, 16, 12, dtype=torch.uint8)}

    def sample_metadata(self, index):
        return {"image_id": 17, "instance_count": 1}


class FakeDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        if self.training:
            return {"loss_classifier": self.weight.square()}
        return [{"boxes": torch.empty(0, 4), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.int64),
                 "masks": torch.empty(0, 1, 16, 12)}]


class TrainingRunTests(unittest.TestCase):
    def config(self):
        return json.loads(Path('models/training_config.json').read_text(encoding='utf-8'))

    def test_smoke_does_not_advance_training_augmentation(self):
        training, validation = FakeDataset(True), FakeDataset(False)
        with tempfile.TemporaryDirectory() as directory, \
             patch('models.train.build_model', return_value=FakeDetector()), \
             patch('models.train.model_summary', return_value={}), \
             patch('models.train.torch.cuda.is_available', return_value=False):
            result = smoke(self.config(), training, validation, torch.device('cpu'), Path(directory), lambda message: None)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(training.augmentation_calls, 0)
            self.assertTrue(training.augment)
            self.assertTrue((Path(directory)/'smoke.json').exists())

    def test_missing_cuda_records_failure_without_claiming_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root/'config.json'
            config_path.write_text(json.dumps(self.config()), encoding='utf-8')
            with patch('models.train.torch.cuda.is_available', return_value=False), \
                 patch('models.train.subprocess.check_output', return_value='test-freeze\n'), \
                 patch('models.train.build_model', side_effect=AssertionError('model should not load')), \
                 redirect_stdout(io.StringIO()):
                status = run(config_path, root/'run')
            self.assertEqual(status, 2)
            report = json.loads((root/'run/run.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['completed_epochs'], 0)
            self.assertIn('CUDA is not available', report['error'])
            self.assertFalse((root/'run/checkpoint_best.pt').exists())

    def test_run_contract_rejects_wrong_classes_seed_or_unsafe_loader(self):
        config = self.config()
        validate_config(config)
        for key, value in [('seed', 7), ('class_names', ['calculator'])]:
            changed = deepcopy(config); changed[key] = value
            with self.assertRaises(ValueError): validate_config(changed)
        changed = deepcopy(config); changed['training']['num_workers'] = 2
        with self.assertRaises(ValueError): validate_config(changed)


if __name__ == '__main__':
    unittest.main()
