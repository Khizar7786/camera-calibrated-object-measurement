"""Known planar geometry, plain-paper detection and explicit CLI failure paths."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from calibration.camera import Undistorter
from inference.measure import load_configuration, run
from inference.reference import detect_reference, order_square_corners
from measurement.planar import MeasurementGeometryError, measure_mask_contour_mm


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads(Path("inference/config.json").read_text())

    def test_perspective_then_metric_rectangle_recovers_known_rotated_object(self):
        reference_mm = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], np.float64)
        # Non-rectangular silhouette with a known 75 x 160 mm minimum rectangle.
        object_mm = np.array([[80, 20], [155, 20], [155, 180], [100, 180], [100, 100], [80, 100]], np.float64)
        angle = np.deg2rad(27)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        object_mm = object_mm @ rotation.T
        plane_to_image = np.array([[3.0, 0.3, 150], [0.2, 2.5, 110], [0.002, 0.001, 1]], np.float64)
        ref_px = cv2.perspectiveTransform(reference_mm[None], plane_to_image)[0]
        contour_px = cv2.perspectiveTransform(object_mm[None], plane_to_image)[0]
        result = measure_mask_contour_mm(contour_px, ref_px, 50, 50)
        self.assertAlmostEqual(result["width_mm"], 75, places=3)
        self.assertAlmostEqual(result["height_mm"], 160, places=3)
        for shift in range(4):
            cyclic = measure_mask_contour_mm(contour_px, np.roll(ref_px, shift, axis=0), 50, 50)
            self.assertAlmostEqual(cyclic["width_mm"], 75, places=3)
            self.assertAlmostEqual(cyclic["height_mm"], 160, places=3)

    def test_invalid_reference_and_degenerate_mask_never_produce_dimensions(self):
        reference = [[10, 10], [110, 10], [110, 110], [10, 110]]
        contour = [[150, 100], [250, 100], [250, 300], [150, 300]]
        for width in [0, -50, float("nan"), True]:
            with self.assertRaises(MeasurementGeometryError):
                measure_mask_contour_mm(contour, reference, width, 50)
        with self.assertRaises(MeasurementGeometryError):
            measure_mask_contour_mm(contour, [[0, 0]] * 4, 50, 50)
        with self.assertRaises(MeasurementGeometryError):
            measure_mask_contour_mm([[0, 0], [1, 1], [2, 2]], reference, 50, 50)

    def scene(self, paper=True, two=False, rotated=False):
        image = np.full((500, 400, 3), 40, np.uint8)
        if paper:
            if rotated:
                points = cv2.boxPoints(((100., 100.), (100., 100.), 27.)).astype(np.int32)
                cv2.fillConvexPoly(image, points, (250, 250, 250))
            else:
                image[20:120, 20:120] = 250
        if two:
            image[20:120, 260:360] = 250
        image[180:340, 200:260] = 10
        return image

    def test_unmarked_square_outer_boundary_refines_after_preview_scaling(self):
        reference = deepcopy(self.config["reference"])
        reference["detection"]["max_detection_dimension"] = 400
        result = detect_reference(self.scene(), reference)
        self.assertEqual(result["status"], "found")
        corners = np.array(result["corners_corrected_px"])
        np.testing.assert_allclose(corners, [[19.5, 19.5], [119.5, 19.5], [119.5, 119.5], [19.5, 119.5]], atol=0.5)
        self.assertEqual(result["candidate_count"], 1)

    def test_rotated_square_and_corner_order(self):
        result = detect_reference(self.scene(rotated=True), self.config["reference"])
        self.assertEqual(result["status"], "found")
        corners = order_square_corners(np.array(result["corners_corrected_px"])[[2, 0, 3, 1]])
        self.assertGreater(cv2.contourArea(corners, oriented=True), 0)
        self.assertEqual(int(np.argmin(corners.sum(axis=1))), 0)

    def test_missing_and_ambiguous_paper_are_not_found(self):
        self.assertEqual(detect_reference(self.scene(paper=False), self.config["reference"])["status"], "not_found")
        duplicate = detect_reference(self.scene(two=True), self.config["reference"])
        self.assertEqual(duplicate["status"], "not_found")
        self.assertEqual(duplicate["candidate_count"], 2)
        self.assertIn("ambiguous", duplicate["failure_reason"])

    def test_bright_region_recovers_paper_when_woodgrain_breaks_edge_contour(self):
        rows, _ = np.indices((500, 400))
        noise = np.random.default_rng(42).normal(0, 3, (500, 400))
        background = np.clip(150 + 80 * np.sin(rows * 0.35) + noise, 0, 235).astype(np.uint8)
        # Keep the four corner neighbourhoods well-defined while wood grain
        # still breaks/joins edge contours along the sides. This isolates the
        # boundary-proposal regression from corner fitting on arbitrary texture.
        for x, y in [(90, 170), (209, 170), (209, 289), (90, 289)]:
            background[y - 7:y + 8, x - 7:x + 8] = 150
        image = np.repeat(background[..., None], 3, axis=2)
        image[170:290, 90:210] = 245
        edge_only = deepcopy(self.config["reference"])
        edge_only["detection"]["region_thresholds"] = []
        self.assertEqual(detect_reference(image, edge_only)["status"], "not_found")
        recovered = detect_reference(image, self.config["reference"])
        self.assertEqual(recovered["status"], "found")
        self.assertEqual(recovered["candidate_count"], 1)
        self.assertEqual(recovered["candidate_generation_method"], "intensity_threshold_240")
        np.testing.assert_allclose(recovered["corners_corrected_px"],
                                   [[89.5, 169.5], [209.5, 169.5], [209.5, 289.5], [89.5, 289.5]], atol=2)

    def fixture(self, paper=True, wrong_size=False, measured=True):
        calibration = {"schema_version": 1, "model": "opencv_pinhole", "coordinate_system": "stored_pixels_full_frame",
                       "camera_matrix": [[400, 0, 200], [0, 400, 250], [0, 0, 1]],
                       "distortion_coefficients": [0, 0, 0, 0, 0], "image_size": [400, 500]}
        camera = self.root / "camera.json"
        camera.write_text(json.dumps(calibration))
        config = deepcopy(self.config)
        config["calibration"] = str(camera)
        config["camera"].update(frame_rotations_cw={"400x500": 0}, expected_exif_model=None)
        config["reference"]["physically_measured"] = measured
        config["reference"]["measurement_source"] = "Synthetic unit fixture; not a physical accuracy experiment"
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        image = self.root / "original.png"
        cv2.imwrite(str(image), self.scene(paper) if not wrong_size else np.zeros((100, 100, 3), np.uint8))
        return image, path

    def fake_loader(self, found=True):
        class Detector(torch.nn.Module):
            def forward(self, images):
                mask = torch.zeros((1, 1, 500, 400))
                mask[:, :, 180:340, 200:260] = 1
                return [{"scores": torch.tensor([0.99 if found else 0.1]), "labels": torch.tensor([1]),
                         "masks": mask, "boxes": torch.tensor([[0., 0., 399., 499.]])}]
        training = {"model": {"min_size": 400, "max_size": 500}, "training": {"mixed_precision": False}}
        return Detector().eval(), training, {"sha256": "synthetic-stub", "epoch": 2}

    def run_fixture(self, name, image, config, found=True):
        output = self.root / name
        with patch("inference.measure.load_segmenter", return_value=self.fake_loader(found)), redirect_stdout(io.StringIO()):
            code = run(image, output, config, device_name="cpu")
        result = json.loads((output / "result.json").read_text())
        return code, result, output

    def test_success_uses_mask_not_detector_box_and_keeps_corrected_overlay(self):
        image, config = self.fixture()
        original_apply = Undistorter.apply
        with patch.object(Undistorter, "apply", autospec=True, side_effect=original_apply) as applied:
            code, result, output = self.run_fixture("success", image, config)
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["undistortion_applied"])
        self.assertGreaterEqual(applied.call_count, 2)
        self.assertAlmostEqual(result["width_mm"], 29.5, delta=0.2)
        self.assertAlmostEqual(result["height_mm"], 79.5, delta=0.2)
        self.assertEqual(cv2.imread(str(output / "overlay.jpg")).shape[:2], (500, 400))
        self.assertEqual(result["metric_geometry"]["conversion_function"], "measurement.planar.measure_mask_contour_mm")

    def test_object_not_found_has_null_dimensions_and_saved_overlay(self):
        image, config = self.fixture()
        code, result, output = self.run_fixture("no_object", image, config, found=False)
        self.assertEqual((code, result["status"]), (2, "object_not_found"))
        self.assertIsNone(result["width_mm"])
        self.assertIsNone(result["height_mm"])
        self.assertTrue((output / "overlay.jpg").exists())

    def test_reference_not_found_has_null_dimensions_and_object_score(self):
        image, config = self.fixture(paper=False)
        code, result, output = self.run_fixture("no_reference", image, config)
        self.assertEqual((code, result["status"]), (2, "reference_not_found"))
        self.assertIsNone(result["width_mm"])
        self.assertIsNone(result["height_mm"])
        self.assertGreater(result["detection_confidence"], 0.5)
        self.assertTrue((output / "mask.png").exists())

    def test_unmeasured_scale_is_refused(self):
        image, config = self.fixture(measured=False)
        _, result, _ = self.run_fixture("unmeasured", image, config)
        self.assertEqual(result["status"], "reference_size_required")
        self.assertIsNone(result["width_mm"])

    def test_wrong_resolution_is_rejected_before_segmentation(self):
        image, config = self.fixture(wrong_size=True)
        with patch("inference.measure.load_segmenter") as model, redirect_stdout(io.StringIO()):
            run(image, self.root / "bad_geometry", config, device_name="cpu")
        model.assert_not_called()
        result = json.loads((self.root / "bad_geometry/result.json").read_text())
        self.assertEqual(result["status"], "invalid_camera_geometry")
        self.assertFalse(result["undistortion_applied"])
        self.assertIsNone(result["width_mm"])

    def test_no_raw_bypass_or_unpaired_reference_override(self):
        _, config = self.fixture()
        with self.assertRaises(ValueError):
            load_configuration(config, width_mm=50)
        data = json.loads(config.read_text()); data["skip_undistortion"] = True
        config.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            load_configuration(config)


if __name__ == "__main__":
    unittest.main()
