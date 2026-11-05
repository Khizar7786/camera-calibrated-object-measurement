"""End-to-end joint correction, preservation and unsafe-geometry refusal."""

from contextlib import redirect_stdout
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from preprocessing.coco_masks import decode_segmentation, encode_mask, mask_geometry
from preprocessing.prepare_dataset import PreparationError, file_hash, run, source_snapshot, write_review


class PreparationTests(unittest.TestCase):
    def fixture(self, workspace):
        root = workspace / "dataset"
        (root / "Raw").mkdir(parents=True)
        (root / "splits").mkdir()
        geometry = {"schema_version": 1,
                    "export": {"size": [80, 80], "resize": "stretch", "crop_xywh": None,
                               "rotation_cw_degrees": 0, "source": "synthetic full-frame stretch"},
                    "camera_frame_rotations": {"160x240": {"rotation_cw_degrees": 0, "source": "known"},
                                               "240x160": {"rotation_cw_degrees": 90, "source": "known clockwise rotation"}},
                    "verification": {"max_rgb_rmse": 12.0, "min_rgb_correlation": 0.97}}
        calibration = {"schema_version": 1, "model": "opencv_pinhole", "coordinate_system": "stored_pixels_full_frame",
                       "camera_matrix": [[180.0, 0, 80], [0, 185.0, 120], [0, 0, 1]],
                       "distortion_coefficients": [0.09, -0.02, 0.001, 0.002, 0.004], "image_size": [160, 240]}
        settings_path, camera_path = workspace / "geometry.json", workspace / "camera.json"
        settings_path.write_text(json.dumps(geometry), encoding="utf-8")
        camera_path.write_text(json.dumps(calibration), encoding="utf-8")
        sums = []
        for index, split in enumerate(("train", "valid", "test")):
            (root / split).mkdir()
            export_name, original_name = f"image_{split}.jpg", f"original_{split}.png"
            size = (240, 160) if split == "valid" else (160, 240)
            mask = np.zeros((80, 80), np.uint8)
            mask[12:68, 15:65] = 1
            mask[30:44, 32:45] = 0  # A hole must survive RLE roundtrips.
            restored = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST_EXACT)
            original = np.repeat((restored * 255)[:, :, None], 3, axis=2)
            export = cv2.resize(original, (80, 80), interpolation=cv2.INTER_AREA)
            self.assertTrue(cv2.imwrite(str(root / "Raw" / original_name), original))
            self.assertTrue(cv2.imwrite(str(root / split / export_name), export))
            document = {"info": {"fixture": True}, "categories": [{"id": 1, "name": "calculator"}],
                        "images": [{"id": 10 + index, "file_name": export_name, "width": 80, "height": 80,
                                    "extra": {"name": original_name}}],
                        "annotations": [{"id": 100 + index, "image_id": 10 + index, "category_id": 1, "iscrowd": 0,
                                         "segmentation": encode_mask(mask), "bbox": [0, 0, 1, 1], "area": 1}]}
            annotation = root / split / "_annotations.coco.json"
            annotation.write_text(json.dumps(document), encoding="utf-8")
            listing = root / "splits" / f"{split}.txt"
            listing.write_text(f"{split}/{export_name}\n", encoding="utf-8")
            sums.extend([f"{file_hash(annotation)}  {split}/_annotations.coco.json",
                         f"{file_hash(listing)}  splits/{split}.txt"])
        (root / "splits" / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
        return root, workspace / "corrected", camera_path, settings_path, geometry

    def test_joint_correction_preserves_inputs_membership_and_instance_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output, camera, geometry, _ = self.fixture(Path(directory))
            before = source_snapshot(root)
            with redirect_stdout(io.StringIO()):
                status = run(root, output, camera, geometry)
            self.assertEqual(status, 0)
            self.assertEqual(source_snapshot(root), before)
            report = json.loads((output / "run.json").read_text())
            self.assertEqual((report["processed_images"], report["processed_instances"], report["failed_images"]), (3, 3, 0))
            self.assertFalse(report["ready_for_training"])
            self.assertFalse(report["manual_review_complete"])
            for split in ("train", "valid", "test"):
                self.assertEqual((root / "splits" / f"{split}.txt").read_bytes(), (output / "splits" / f"{split}.txt").read_bytes())
                old = json.loads((root / split / "_annotations.coco.json").read_text())
                new = json.loads((output / split / "_annotations.coco.json").read_text())
                image, annotation = new["images"][0], new["annotations"][0]
                self.assertEqual((image["width"], image["height"]), (160, 240))
                self.assertEqual(image["id"], old["images"][0]["id"])
                self.assertEqual(annotation["id"], old["annotations"][0]["id"])
                self.assertEqual(annotation["image_id"], image["id"])
                self.assertEqual(annotation["category_id"], 1)
                self.assertNotEqual(annotation["segmentation"], old["annotations"][0]["segmentation"])
                mask = decode_segmentation(annotation["segmentation"], 160, 240)
                box, area = mask_geometry(mask)
                self.assertEqual(annotation["bbox"], box)
                self.assertEqual(annotation["area"], area)
                self.assertGreater(area, 1)
                record = next(row for row in report["images"] if row["split"] == split)
                self.assertEqual(record["review_flags"], [])
                png = cv2.imread(str(output / record["instances"][0]["mask_file"]), cv2.IMREAD_GRAYSCALE)
                np.testing.assert_array_equal(png != 0, mask != 0)
                corrected = cv2.imread(str(output / image["measure_it_source"]["export_file"]))
                foreground = corrected.mean(axis=2) > 127
                iou = np.logical_and(foreground, mask).sum() / np.logical_or(foreground, mask).sum()
                self.assertGreater(iou, 0.99, "Image foreground and corrected mask must remain aligned")
                self.assertTrue((output / record["overlay"]).is_file())
            self.assertTrue((output / "review.html").exists())

    def test_unknown_rotation_blocks_image_and_publishes_only_partial_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output, camera, config, settings = self.fixture(Path(directory))
            del settings["camera_frame_rotations"]["240x160"]
            config.write_text(json.dumps(settings), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                status = run(root, output, camera, config)
            self.assertEqual(status, 2)
            self.assertFalse((output / "valid" / "_annotations.coco.json").exists())
            self.assertTrue((output / "valid" / "_annotations.partial.coco.json").exists())
            self.assertFalse((output / "valid" / "image_valid.jpg").exists())
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["failed_images"], 1)
            self.assertEqual(report["status"], "incomplete_do_not_train")
            self.assertTrue(report["source_files_unchanged"])

    def test_unverified_crop_or_content_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output, camera, config, _ = self.fixture(Path(directory))
            cv2.imwrite(str(root / "train" / "image_train.jpg"), np.zeros((80, 80, 3), np.uint8))
            with redirect_stdout(io.StringIO()):
                status = run(root, output, camera, config)
            self.assertEqual(status, 2)
            report = json.loads((output / "run.json").read_text())
            self.assertIn("geometry did not verify", report["splits"]["train"]["errors"][0]["error"])
            self.assertFalse((output / "train" / "_annotations.coco.json").exists())

    def test_original_paths_and_existing_outputs_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output, camera, config, _ = self.fixture(Path(directory))
            before = source_snapshot(root)
            with self.assertRaises(PreparationError):
                run(root, root / "train", camera, config)
            output.mkdir()
            (output / "existing.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(PreparationError):
                run(root, output, camera, config)
            self.assertEqual(source_snapshot(root), before)
            self.assertEqual((output / "existing.txt").read_text(), "preserve")

    def test_source_change_during_run_prevents_final_coco_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root, output, camera, config, _ = self.fixture(Path(directory))
            calls = 0

            def changed_snapshot(path):
                nonlocal calls
                calls += 1
                snapshot = source_snapshot(path)
                if calls > 1:
                    snapshot["concurrent_change"] = "changed"
                return snapshot

            with patch("preprocessing.prepare_dataset.source_snapshot", side_effect=changed_snapshot), redirect_stdout(io.StringIO()):
                status = run(root, output, camera, config)
            self.assertEqual(status, 2)
            report = json.loads((output / "run.json").read_text())
            self.assertFalse(report["source_files_unchanged"])
            self.assertEqual(report["status"], "incomplete_do_not_train")
            for split in ("train", "valid", "test"):
                self.assertFalse((output / split / "_annotations.coco.json").exists())
                self.assertTrue((output / split / "_annotations.partial.coco.json").exists())

    def test_review_gallery_and_csv_record_explicit_user_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = {"split": "train", "source_export": "train/example.jpg", "output_image": "train/example.jpg",
                   "overlay": "train/overlays/example.jpg", "instance_count": 2, "status": "processed",
                   "review_status": "passed", "review_flags": ["multiple_instances"], "reviewed_by": "user",
                   "reviewed_at_utc": "2026-11-05T00:00:00+00:00", "flag_resolution": "accepted_by_user",
                   "review_note": "All images passed user review."}
            write_review(output, [row])
            gallery = (output / "review.html").read_text(encoding="utf-8")
            self.assertIn("1 of 1 processed images passed manual review", gallery)
            self.assertIn("review: passed", gallery)
            self.assertIn("accepted_by_user", gallery)
            self.assertNotIn("pending review", gallery)
            with (output / "manual_review.csv").open(newline="", encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))[0]
            self.assertEqual(saved["review_status"], "passed")
            self.assertEqual(saved["reviewed_by"], "user")
            self.assertEqual(saved["review_flags"], "multiple_instances")
            self.assertEqual(saved["flag_resolution"], "accepted_by_user")


if __name__ == "__main__":
    unittest.main()
