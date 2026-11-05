"""Calibration guards and metric correctness using independent synthetic truth."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration.calibrate import board_points, load_settings, reprojection_errors, run
from calibration.camera import CalibrationError, CameraCalibration


class CalibrationTests(unittest.TestCase):
    def settings(self):
        return {"inner_corners": [9, 7], "square_size_mm": 20.0,
                "square_size_source": "synthetic test", "corner_count_source": "synthetic test",
                "image_size": [640, 480], "minimum_views": 20, "detection_max_dimension": 800}

    def create_run(self, root, count):
        settings = self.settings()
        config = root / "board.json"
        config.write_text(json.dumps(settings), encoding="utf-8")
        inputs = root / "raw"
        inputs.mkdir()
        for index in range(count):
            self.assertTrue(cv2.imwrite(str(inputs / f"view_{index:02}.png"), np.zeros((480, 640, 3), np.uint8)))
        return config, inputs, root / "results"

    def synthetic_observations(self):
        objects = board_points(self.settings())
        matrix = np.array([[700.0, 0, 320.0], [0, 680.0, 240.0], [0, 0, 1]])
        distortion = np.array([-0.08, 0.02, 0.001, -0.001, 0.005])
        observations = []
        for index in range(20):
            rotation = np.array([-0.35 + index * 0.035, -0.25 + (index % 5) * 0.12, index * 0.015])
            translation = np.array([-100.0 + (index % 4) * 20, -75.0 + (index % 3) * 20, 650.0 + index * 12])
            points, _ = cv2.projectPoints(objects, rotation, translation, matrix, distortion)
            observations.append((points, "synthetic known camera"))
        return matrix, observations

    def test_board_measurement_and_minimum_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.json"
            for key, value in [("square_size_mm", 0), ("minimum_views", 19), ("inner_corners", [8, 6])]:
                settings = self.settings()
                settings[key] = value
                path.write_text(json.dumps(settings), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(CalibrationError):
                    load_settings(path)

    def test_rms_is_per_corner_euclidean_error_not_norm_divided_by_count(self):
        objects = board_points(self.settings())
        matrix = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
        rotation, translation = np.zeros(3), np.array([-80.0, -60.0, 800.0])
        projected, _ = cv2.projectPoints(objects, rotation, translation, matrix, np.zeros(5))
        observed = projected + np.array([[[3.0, 4.0]]], dtype=np.float32)
        errors, total = reprojection_errors([objects], [observed], matrix, np.zeros(5), [rotation], [translation])
        self.assertAlmostEqual(errors[0], 5.0, places=5)
        self.assertAlmostEqual(total, 5.0, places=5)

    def test_nineteen_successes_and_wrong_orientation_refuse_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, inputs, output = self.create_run(root, 19)
            cv2.imwrite(str(inputs / "wrong_orientation.png"), np.zeros((640, 480, 3), np.uint8))
            corners = np.mgrid[0:9, 0:7].T.reshape(-1, 1, 2).astype(np.float32) * 20 + 100
            with patch("calibration.calibrate.detect_corners", return_value=(corners, "test")), \
                 patch("calibration.calibrate.cv2.calibrateCameraExtended") as fit, redirect_stdout(io.StringIO()):
                result = run(config, inputs, output)
            fit.assert_not_called()
            self.assertEqual(result, 2)
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["usable_count"], 19)
            self.assertEqual(report["status"], "insufficient_views")
            self.assertEqual(report["views"][-1]["status"], "rejected_geometry")
            self.assertFalse((output / "camera.json").exists())
            self.assertFalse((output / "raw_calibration.npz").exists())

    def test_successful_fit_recovers_known_intrinsics_and_saves_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, inputs, output = self.create_run(root, 20)
            expected, corners = self.synthetic_observations()
            with patch("calibration.calibrate.detect_corners", side_effect=corners), \
                 patch("calibration.calibrate.save_previews", return_value=[]), redirect_stdout(io.StringIO()):
                result = run(config, inputs, output)
            self.assertEqual(result, 0)
            calibration = CameraCalibration.load(output / "camera.json")
            np.testing.assert_allclose(calibration.camera_matrix, expected, rtol=1e-3, atol=0.05)
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(report["status"], "calibrated")
            self.assertEqual(report["usable_count"], 20)
            self.assertLess(report["global_rms_px"], 0.001)
            with np.load(output / "raw_calibration.npz", allow_pickle=False) as raw:
                self.assertEqual(raw["refined_image_points_px"].shape, (20, 63, 1, 2))
                self.assertEqual(raw["object_points_mm"].shape, (20, 63, 3))
                self.assertEqual(len(raw["opencv_per_view_errors"]), 20)
            self.assertTrue((output / "views.csv").exists())
            self.assertTrue((output / "console.txt").exists())

    def test_nonempty_output_is_not_overwritten_with_stale_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            config, inputs, output = self.create_run(Path(directory), 1)
            output.mkdir()
            camera = output / "camera.json"
            camera.write_text("existing fit", encoding="utf-8")
            with self.assertRaisesRegex(CalibrationError, "not empty"):
                run(config, inputs, output)
            self.assertEqual(camera.read_text(), "existing fit")

    def test_unreadable_view_is_reported_and_does_not_abort_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            config, inputs, output = self.create_run(Path(directory), 1)
            with patch("calibration.calibrate.Path.read_bytes", side_effect=PermissionError("input denied")), \
                 redirect_stdout(io.StringIO()):
                result = run(config, inputs, output)
            self.assertEqual(result, 2)
            report = json.loads((output / "run.json").read_text())
            self.assertEqual(len(report["views"]), 1)
            self.assertEqual(report["views"][0]["status"], "failed")
            self.assertIn("input denied", report["views"][0]["reason"])
            self.assertFalse((output / "camera.json").exists())

    def test_final_evidence_write_failure_does_not_publish_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            config, inputs, output = self.create_run(Path(directory), 20)
            _, observations = self.synthetic_observations()
            original_write = Path.write_text

            def fail_calibrated_report(path, text, *args, **kwargs):
                if path.name == "run.json" and json.loads(text).get("status") == "calibrated":
                    raise OSError("simulated final report write failure")
                return original_write(path, text, *args, **kwargs)

            with patch("calibration.calibrate.detect_corners", side_effect=observations), \
                 patch("calibration.calibrate.save_previews", return_value=[]), \
                 patch("calibration.calibrate.Path.write_text", new=fail_calibrated_report), \
                 redirect_stdout(io.StringIO()):
                result = run(config, inputs, output)
            self.assertEqual(result, 2)
            self.assertFalse((output / "camera.json").exists())
            self.assertFalse((output / "camera.json.pending").exists())
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "fit_failed")


if __name__ == "__main__":
    unittest.main()
