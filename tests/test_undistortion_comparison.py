"""Raw bypass isolation, per-frame geometry and honest paired denominators."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration.camera import Undistorter
from inference import measure as normal
from measurement import raw_inference
from measurement.compare_undistortion import paired_statistics, validate_result
from measurement.evaluate_placements import measurement_row


class UndistortionComparisonTests(unittest.TestCase):
    def fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        calibration = {"schema_version": 1, "model": "opencv_pinhole",
                       "coordinate_system": "stored_pixels_full_frame", "image_size": [400, 500],
                       "camera_matrix": [[500, 0, 200], [0, 500, 250], [0, 0, 1]],
                       "distortion_coefficients": [0.2, -0.1, 0, 0, 0]}
        camera_path = root / "camera.json"
        camera_path.write_text(json.dumps(calibration))
        config = json.loads(Path("inference/config.json").read_text())
        config["calibration"] = str(camera_path)
        config["camera"].update(frame_rotations_cw={"400x500": 0}, expected_exif_model=None)
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        frame = np.full((500, 400, 3), 40, np.uint8)
        frame[20:120, 20:120] = 250
        frame[180:340, 200:260] = 10
        image = root / "original.png"
        cv2.imwrite(str(image), frame)
        return root, image, config_path, frame

    def test_bypass_requires_explicit_acknowledgement_and_normal_rejects_flag(self):
        with self.assertRaises(ValueError):
            raw_inference.run(Path("unused"), Path("unused"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            raw_inference.main(["unused", "--output", "unused"])
        self.assertEqual(raised.exception.code, 2)
        with patch("sys.argv", ["inference.measure", "unused", "--experimental-skip-undistortion"]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            normal.main()
        self.assertEqual(raised.exception.code, 2)

    def test_raw_detector_and_segmenter_receive_same_unremapped_frame_and_raw_geometry(self):
        root, image, config, frame = self.fixture()
        seen = []
        original_detector = normal.detect_reference
        def detector(actual, settings):
            np.testing.assert_array_equal(actual, frame)
            seen.append("reference")
            return original_detector(actual, settings)
        def segmenter(actual, *_):
            np.testing.assert_array_equal(actual, frame)
            seen.append("mask")
            mask = np.zeros((500, 400), np.uint8)
            mask[180:340, 200:260] = 1
            return mask, 0.99, {"corrected_size": [400, 500], "mask_restoration": "corrected full frame"}
        with patch.object(Undistorter, "apply", side_effect=AssertionError("Raw must not remap")), \
                patch.object(normal, "detect_reference", side_effect=detector), \
                patch.object(normal, "load_segmenter", return_value=(None, None, {"sha256": "stub"})), \
                patch.object(normal, "segment_calculator", side_effect=segmenter), redirect_stdout(io.StringIO()):
            code = raw_inference.run(image, root / "raw", config, experimental_skip_undistortion=True, device_name="cpu")
        result = json.loads((root / "raw/result.json").read_text())
        self.assertEqual(code, 0)
        self.assertEqual(seen, ["reference", "mask"])
        self.assertFalse(result["undistortion_applied"])
        self.assertEqual(result["coordinate_system"], "raw_distorted_full_frame")
        self.assertIn("corners_raw_px", result["reference"])
        self.assertNotIn("corners_corrected_px", result["reference"])
        geometry = json.loads((root / "raw/geometry.json").read_text())
        self.assertIn("contours_raw_px", geometry)
        self.assertNotIn("rectangle_corrected_px", geometry)
        self.assertAlmostEqual(result["width_mm"], 29.5, delta=0.2)
        self.assertAlmostEqual(result["height_mm"], 79.5, delta=0.2)

    def test_raw_missing_reference_keeps_refusal_and_null_dimensions(self):
        root, image, config, frame = self.fixture()
        frame[20:120, 20:120] = 40
        cv2.imwrite(str(image), frame)
        mask = np.zeros((500, 400), np.uint8)
        mask[180:340, 200:260] = 1
        transform = {"corrected_size": [400, 500], "mask_restoration": "corrected full frame"}
        with patch.object(normal, "load_segmenter", return_value=(None, None, {"sha256": "stub"})), \
                patch.object(normal, "segment_calculator", return_value=(mask, 0.99, transform)), redirect_stdout(io.StringIO()):
            code = raw_inference.run(image, root / "missing", config, experimental_skip_undistortion=True, device_name="cpu")
        result = json.loads((root / "missing/result.json").read_text())
        self.assertEqual((code, result["status"]), (2, "reference_not_found"))
        self.assertIsNone(result["width_mm"])
        self.assertIsNone(result["height_mm"])
        self.assertTrue((root / "missing/mask.png").exists())

    def row(self, identifier, prediction):
        case = {"image_id": identifier, "filename": identifier + ".jpg", "sha256": identifier,
                "previously_used_for_development": False}
        reading = {"hand_width_mm": 70, "hand_height_mm": 160, "reading_source": "ruler"}
        result = {"status": "success" if prediction else "reference_not_found",
                  "width_mm": prediction[0] if prediction else None, "height_mm": prediction[1] if prediction else None}
        return measurement_row(case, reading, result)

    def test_different_coverage_uses_intersection_not_unpaired_means_for_change(self):
        corrected = [self.row("a", (72, 162)), self.row("b", (170, 260))]
        raw = [self.row("a", (71, 164)), self.row("b", None)]
        pairs, report = paired_statistics({"corrected": corrected, "raw": raw})
        self.assertEqual(report["paired_valid_count"], 1)
        self.assertEqual(report["all_valid_per_mode"]["corrected"]["dimensions"]["width"]["mae_mm"], 51)
        self.assertEqual(report["paired_valid_only"]["corrected"]["dimensions"]["width"]["mae_mm"], 2)
        self.assertEqual(report["paired_error_changes"]["width"]["mean_absolute_error_change_raw_minus_corrected_mm"], -1)
        self.assertEqual(report["paired_error_changes"]["height"]["mean_absolute_error_change_raw_minus_corrected_mm"], 2)
        self.assertIsNone(pairs[1]["width_absolute_error_raw_minus_corrected_mm"])

    def test_no_valid_pair_gives_missing_changes_and_zero_pooled_denominator(self):
        _, report = paired_statistics({"corrected": [self.row("a", None)], "raw": [self.row("a", (70, 160))]})
        self.assertEqual(report["paired_valid_count"], 0)
        self.assertIsNone(report["paired_error_changes"]["width"]["mean_absolute_error_change_raw_minus_corrected_mm"])
        self.assertIsNone(report["paired_valid_only"]["raw"]["dimensions"]["height"]["mae_mm"])
        self.assertEqual(report["pooled_paired_dimensions"]["dimension_count"], 0)

    def test_pair_rejects_different_inputs_or_hand_readings(self):
        a, b = self.row("a", (70, 160)), self.row("a", (70, 160))
        b["input_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            paired_statistics({"corrected": [a], "raw": [b]})
        b = deepcopy(a)
        b["hand_width_mm"] = 75
        with self.assertRaises(ValueError):
            paired_statistics({"corrected": [a], "raw": [b]})

    def test_raw_results_cannot_silently_carry_corrected_corner_coordinates(self):
        config = {"checkpoint_sha256": "frozen", "reference": {"width_mm": 50, "height_mm": 50}}
        result = {"input": {"sha256": "a"}, "undistortion_applied": False,
                  "coordinate_system": "raw_distorted_full_frame",
                  "reference": {"coordinate_system": "raw_distorted_full_frame", "width_mm": 50,
                                "height_mm": 50, "corners_corrected_px": [[1, 2]]},
                  "checkpoint": {"loaded": True, "sha256": "frozen"},
                  "model_transform": {"score_cutoff": 0.5, "mask_probability_threshold": 0.5, "aspect_preserving": True}}
        with self.assertRaisesRegex(ValueError, "Raw output must not contain corrected"):
            validate_result("raw", {"sha256": "a"}, result, config)
