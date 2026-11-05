"""Shared undistortion contracts, including unsafe image-geometry failures."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from calibration.camera import (
    CalibrationError,
    CameraCalibration,
    Undistorter,
    read_image,
    undistort,
)


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.size = (128, 96)
        self.matrix = np.array([[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]])
        self.calibration = CameraCalibration(self.matrix, np.zeros(5), self.size)

    def payload(self):
        return {
            "schema_version": 1,
            "model": "opencv_pinhole",
            "coordinate_system": "stored_pixels_full_frame",
            "image_size": list(self.size),
            "camera_matrix": self.matrix.tolist(),
            "distortion_coefficients": [0.0] * 5,
        }

    def test_zero_distortion_keeps_every_pixel_and_full_frame(self):
        image = np.random.default_rng(42).integers(0, 256, (96, 128, 3), dtype=np.uint8)
        result = undistort(image, self.calibration)
        self.assertEqual(result.shape, image.shape)
        np.testing.assert_array_equal(result, image)
        np.testing.assert_array_equal(Undistorter(self.calibration).apply(image), result)

    def test_reversed_orientation_and_resized_images_are_rejected(self):
        for shape in ((128, 96, 3), (48, 64, 3), (96, 127, 3)):
            with self.subTest(shape=shape), self.assertRaisesRegex(
                CalibrationError, "resolution/orientation mismatch"
            ):
                undistort(np.zeros(shape, dtype=np.uint8), self.calibration)

    def test_nearest_neighbour_does_not_invent_mask_labels(self):
        calibration = CameraCalibration(
            self.matrix, np.array([-0.18, 0.04, 0.002, -0.001, 0.0]), self.size
        )
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[20:70, 20:60] = 2
        mask[30:80, 65:110] = 17
        result = undistort(mask, calibration, interpolation=cv2.INTER_NEAREST)
        self.assertEqual(result.shape, mask.shape)
        self.assertTrue(set(np.unique(result)).issubset({0, 2, 17}))
        self.assertFalse(np.array_equal(result, mask))

    def test_unsupported_interpolation_and_mask_dtypes_are_rejected(self):
        image = np.zeros((96, 128), dtype=np.uint8)
        with self.assertRaises(CalibrationError):
            undistort(image, self.calibration, interpolation=cv2.INTER_CUBIC)
        with self.assertRaisesRegex(CalibrationError, "dtype"):
            undistort(image.astype(np.int32), self.calibration)

    def test_load_accepts_declared_full_frame_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            path.write_text(json.dumps(self.payload()), encoding="utf-8")
            loaded = CameraCalibration.load(path)
        self.assertEqual(loaded.image_size, self.size)
        np.testing.assert_array_equal(loaded.camera_matrix, self.matrix)
        np.testing.assert_array_equal(loaded.distortion_coefficients, np.zeros(5))

    def test_load_rejects_unsupported_or_incomplete_geometry_contracts(self):
        changes = (
            {"schema_version": 2},
            {"schema_version": True},
            {"model": "fisheye"},
            {"coordinate_system": "cropped_roi"},
            {"image_size": [128.0, 96]},
            {"image_size": [128, 0]},
            {"distortion_coefficients": [0.0] * 6},
            {"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 1, 1]]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            for change in changes:
                with self.subTest(change=change):
                    payload = self.payload()
                    payload.update(change)
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises(CalibrationError):
                        CameraCalibration.load(path)
            payload = self.payload()
            del payload["camera_matrix"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CalibrationError):
                CameraCalibration.load(path)

    def test_nonfinite_or_nonphysical_parameters_are_rejected(self):
        for focal in (0, -1, np.inf, np.nan):
            with self.subTest(focal=focal):
                matrix = self.matrix.copy()
                matrix[0, 0] = focal
                with self.assertRaises(CalibrationError):
                    CameraCalibration(matrix, np.zeros(5), self.size)
        with self.assertRaises(CalibrationError):
            CameraCalibration(self.matrix, [0, 0, 0, 0, np.inf], self.size)

    def test_image_reader_preserves_stored_dimensions_and_bgr_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normal.png"
            Image.new("RGB", (12, 8), (10, 20, 30)).save(path)
            image = read_image(path)
        self.assertEqual(image.shape, (8, 12, 3))
        np.testing.assert_array_equal(image[0, 0], [30, 20, 10])

    def test_exif_rotation_is_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (12, 8)).save(path, exif=exif)
            with self.assertRaisesRegex(CalibrationError, "EXIF orientation 6"):
                read_image(path)

    def test_unreadable_input_is_reported_as_calibration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jpg"
            path.write_bytes(b"not an image")
            with self.assertRaises(CalibrationError):
                read_image(path)
            with self.assertRaises(CalibrationError):
                read_image(Path(directory) / "missing.jpg")


if __name__ == "__main__":
    unittest.main()
