"""Explicit source/export geometry, categorical recovery and provenance checks."""

import copy
import unittest

import cv2
import numpy as np

from preprocessing.geometry import GeometryError, GeometryTransform


class GeometryTests(unittest.TestCase):
    def settings(self, raw_size=(8, 6), export_size=(4, 3), camera_rotation=0):
        return {
            "schema_version": 1,
            "export": {
                "size": list(export_size), "resize": "stretch", "crop_xywh": None,
                "rotation_cw_degrees": 0, "source": "known export settings",
            },
            "camera_frame_rotations": {
                f"{raw_size[0]}x{raw_size[1]}": {
                    "rotation_cw_degrees": camera_rotation,
                    "source": "explicit capture confirmation",
                },
            },
            "verification": {"max_rgb_rmse": 12.0, "min_rgb_correlation": 0.97},
            "overrides": {},
        }

    def test_declared_clockwise_camera_rotation_is_shared_by_image_and_mask(self):
        transform = GeometryTransform.from_settings(self.settings(camera_rotation=90), (8, 6), (4, 3), "capture.jpg")
        image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        mask = np.zeros((3, 4), dtype=bool)
        mask[0, 0] = True
        restored = transform.restore_mask(mask)
        expected = np.zeros((8, 6), dtype=np.uint8)
        expected[0:2, 4:6] = 1
        np.testing.assert_array_equal(restored, expected)
        np.testing.assert_array_equal(transform.prepare_image(image), np.rot90(image, -1))
        self.assertEqual(transform.canonical_size, (6, 8))
        self.assertEqual(restored.dtype, np.uint8)
        self.assertTrue(transform.describe()["annotation_coverage"]["full_raw_frame"])

    def test_crop_and_export_rotation_are_reversed_before_camera_rotation(self):
        settings = self.settings(export_size=(2, 4), camera_rotation=90)
        settings["export"].update(crop_xywh=[2, 1, 4, 2], rotation_cw_degrees=90)
        transform = GeometryTransform.from_settings(settings, (8, 6), (2, 4), "crop.jpg")
        # The left column of the 4x2 raw crop becomes the top row after CW rotation.
        export_mask = np.zeros((4, 2), dtype=np.uint8)
        export_mask[0, :] = 1
        restored = transform.restore_mask(export_mask)
        expected = np.zeros((8, 6), dtype=np.uint8)
        expected[2, 3:5] = 1
        np.testing.assert_array_equal(restored, expected)
        coverage = transform.describe()["annotation_coverage"]
        self.assertFalse(coverage["full_raw_frame"])
        self.assertAlmostEqual(coverage["raw_fraction"], 8 / 48)
        self.assertIn("not verified background", coverage["outside_crop"])
        self.assertTrue(coverage["manual_review_required"])

    def test_per_source_export_override_merges_without_mutating_settings(self):
        settings = self.settings()
        settings["overrides"] = {"changed.jpg": {"export": {"rotation_cw_degrees": 180, "source": "per-image record"}}}
        before = copy.deepcopy(settings)
        changed = GeometryTransform.from_settings(settings, (8, 6), (4, 3), "changed.jpg")
        normal = GeometryTransform.from_settings(settings, (8, 6), (4, 3), "normal.jpg")
        self.assertEqual(changed.export_rotation_cw_degrees, 180)
        self.assertTrue(changed.override_used)
        self.assertEqual(normal.export_rotation_cw_degrees, 0)
        self.assertEqual(settings, before)

    def test_noninteger_mask_scale_uses_exact_nearest_pixel_centres(self):
        settings = self.settings(raw_size=(3, 1), export_size=(2, 1))
        transform = GeometryTransform.from_settings(settings, (3, 1), (2, 1), "thin.jpg")
        restored = transform.restore_mask(np.array([[0, 1]], dtype=np.uint8))
        # INTER_NEAREST would put the transition one pixel too far right here.
        np.testing.assert_array_equal(restored, np.array([[0, 1, 1]], dtype=np.uint8))

    def test_missing_camera_rule_and_invalid_geometry_are_rejected(self):
        settings = self.settings()
        invalid = []
        for crop in ([0, 0, 9, 6], [-1, 0, 4, 3], [0, 0, 0, 2], [0.0, 0, 4, 3], [False, 0, 4, 3]):
            value = copy.deepcopy(settings)
            value["export"]["crop_xywh"] = crop
            invalid.append(value)
        for key, value in (("rotation_cw_degrees", 45), ("rotation_cw_degrees", True), ("resize", "fit"), ("size", [3, 4])):
            candidate = copy.deepcopy(settings)
            candidate["export"][key] = value
            invalid.append(candidate)
        missing = copy.deepcopy(settings)
        missing["camera_frame_rotations"] = {}
        invalid.append(missing)
        missing_angle = copy.deepcopy(settings)
        del missing_angle["camera_frame_rotations"]["8x6"]["rotation_cw_degrees"]
        invalid.append(missing_angle)
        for candidate in invalid:
            with self.subTest(settings=candidate), self.assertRaises(GeometryError):
                GeometryTransform.from_settings(candidate, (8, 6), (4, 3), "capture.jpg")

    def test_bad_sizes_masks_and_verification_thresholds_are_rejected(self):
        transform = GeometryTransform.from_settings(self.settings(), (8, 6), (4, 3), "capture.jpg")
        for mask in (np.zeros((4, 3), np.uint8), np.full((3, 4), 255, np.uint8), np.zeros((3, 4), float), np.zeros((3, 4, 1), np.uint8)):
            with self.subTest(shape=mask.shape), self.assertRaises(GeometryError):
                transform.restore_mask(mask)
        with self.assertRaises(GeometryError):
            transform.prepare_image(np.zeros((8, 6, 3), np.uint8))
        with self.assertRaises(GeometryError):
            transform.verify_export(np.zeros((6, 8, 3), np.uint8), np.zeros((4, 3, 3), np.uint8))
        for key, value in (("max_rgb_rmse", float("nan")), ("max_rgb_rmse", -1), ("min_rgb_correlation", 1.1), ("min_rgb_correlation", True)):
            settings = self.settings()
            settings["verification"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(GeometryError):
                GeometryTransform.from_settings(settings, (8, 6), (4, 3), "capture.jpg")

    def test_verification_checks_declared_geometry_without_searching_rotations(self):
        settings = self.settings(raw_size=(80, 60), export_size=(40, 30))
        transform = GeometryTransform.from_settings(settings, (80, 60), (40, 30), "capture.jpg")
        image = np.random.default_rng(42).integers(0, 256, (60, 80, 3), dtype=np.uint8)
        exported = cv2.resize(image, (40, 30), interpolation=cv2.INTER_AREA)
        result = transform.verify_export(image, exported)
        self.assertTrue(result["verified"])
        self.assertEqual(result["best_renderer"], "opencv_area")
        self.assertEqual(result["rgb_rmse"], 0)
        self.assertAlmostEqual(result["rgb_correlation"], 1)
        rotated_export = cv2.rotate(exported, cv2.ROTATE_180)
        mismatch = transform.verify_export(image, rotated_export)
        self.assertFalse(mismatch["verified"])
        self.assertIn("no alternative geometry", mismatch["reason"])
        self.assertEqual(len(mismatch["candidates"]), 2)

    def test_verification_uses_declared_crop_and_export_rotation(self):
        settings = self.settings(raw_size=(80, 60), export_size=(20, 30))
        settings["export"].update(crop_xywh=[10, 5, 60, 40], rotation_cw_degrees=270)
        transform = GeometryTransform.from_settings(settings, (80, 60), (20, 30), "crop.jpg")
        image = np.random.default_rng(2).integers(0, 256, (60, 80, 3), dtype=np.uint8)
        export = cv2.resize(np.rot90(image[5:45, 10:70]), (20, 30), interpolation=cv2.INTER_AREA)
        self.assertTrue(transform.verify_export(image, export)["verified"])

    def test_untextured_image_cannot_verify_provenance(self):
        transform = GeometryTransform.from_settings(self.settings(), (8, 6), (4, 3), "blank.jpg")
        result = transform.verify_export(np.zeros((6, 8, 3), np.uint8), np.zeros((3, 4, 3), np.uint8))
        self.assertFalse(result["verified"])
        self.assertIsNone(result["rgb_correlation"])


if __name__ == "__main__":
    unittest.main()
