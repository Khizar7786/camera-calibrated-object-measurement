"""COCO mask topology and regenerated geometry, independent of source metadata."""

import json
import unittest

import numpy as np
from pycocotools import mask as mask_api

from preprocessing.coco_masks import (
    AnnotationError,
    decode_segmentation,
    encode_mask,
    mask_geometry,
)


class CocoMaskTests(unittest.TestCase):
    def test_compressed_rle_roundtrip_preserves_holes_and_disconnected_pieces(self):
        mask = np.zeros((31, 37), dtype=np.uint8)
        mask[3:24, 4:27] = 1
        mask[8:17, 10:19] = 0
        mask[26:29, 30:34] = 1
        encoded = encode_mask(mask)
        self.assertIsInstance(encoded["counts"], str)
        self.assertEqual(encoded["size"], [31, 37])
        json.dumps(encoded)  # The output must be directly serializable in COCO.
        np.testing.assert_array_equal(decode_segmentation(encoded, 37, 31), mask)
        np.testing.assert_array_equal(decode_segmentation(
            {**encoded, "counts": encoded["counts"].encode("ascii")}, 37, 31), mask)

    def test_uncompressed_rle_uses_fortran_pixel_order(self):
        mask = decode_segmentation({"size": [3, 4], "counts": [2, 3, 4, 2, 1]}, 4, 3)
        expected = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0], dtype=np.uint8).reshape((3, 4), order="F")
        np.testing.assert_array_equal(mask, expected)

    def test_polygon_parts_are_unioned_with_official_rasterization(self):
        polygons = [[2, 2, 8, 2, 8, 9, 2, 9], [12, 10, 16, 10, 16, 14, 12, 14]]
        decoded = decode_segmentation(polygons, 20, 18)
        expected = np.zeros((18, 20), dtype=np.uint8)
        expected[2:9, 2:8] = 1
        expected[10:14, 12:16] = 1
        np.testing.assert_array_equal(decoded, expected)

    def test_bbox_and_area_are_recomputed_from_actual_pixels(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[5:9, 7:13] = 1
        mask[6, 8] = 0
        bbox, area = mask_geometry(mask)
        self.assertEqual(bbox, [7.0, 5.0, 6.0, 4.0])
        self.assertEqual(area, 23)
        mask[15, 23] = 1
        bbox, area = mask_geometry(mask)
        self.assertEqual(bbox, [7.0, 5.0, 17.0, 11.0])
        self.assertEqual(area, 24)

    def test_empty_and_full_frame_masks_have_exact_geometry(self):
        empty = decode_segmentation([], 11, 7)
        self.assertEqual(mask_geometry(empty), ([0.0, 0.0, 0.0, 0.0], 0))
        full = np.ones((7, 11), dtype=bool)
        self.assertEqual(mask_geometry(full), ([0.0, 0.0, 11.0, 7.0], 77))
        np.testing.assert_array_equal(decode_segmentation(encode_mask(full), 11, 7), full)

    def test_rle_mismatched_dimensions_and_invalid_counts_are_rejected(self):
        invalid = [
            {"size": [4, 3], "counts": [12]},
            {"size": [3, 4], "counts": []},
            {"size": [3, 4], "counts": [11]},
            {"size": [3, 4], "counts": [13]},
            {"size": [3, 4], "counts": [1, -1, 12]},
            {"size": [3, 4], "counts": [1.0, 11.0]},
            {"size": [3, 4], "counts": [True, 11]},
            {"size": [3, 4], "counts": ""},
            {"size": [3, 4], "counts": "P"},  # Unterminated continuation.
            {"size": [3, 4], "counts": "z"},  # Outside compressed alphabet.
            {"size": [3, 4], "counts": "é"},
            {"size": [3, 4], "counts": "="},  # Expands to 13 pixels.
        ]
        for segmentation in invalid:
            with self.subTest(segmentation=segmentation), self.assertRaises(AnnotationError):
                decode_segmentation(segmentation, 4, 3)

    def test_invalid_polygon_shapes_and_coordinates_are_rejected(self):
        for segmentation in [None, [1, 2, 3], [[0, 0, 1, 1]],
                             [[0, 0, 1, 0, 1]], [[0, 0, 1, 0, np.nan, 1]],
                             [[0, 0, 1, 0, True, 1]], [[0, 0, 1, 0, 1e12, 1]]]:
            with self.subTest(segmentation=segmentation), self.assertRaises(AnnotationError):
                decode_segmentation(segmentation, 4, 3)

    def test_nonbinary_or_noninteger_instances_are_not_silently_converted(self):
        for mask in [np.zeros((3, 4, 1), np.uint8), np.zeros((3, 4), np.float32),
                     np.full((3, 4), 255, np.uint8), np.ones((0, 4), np.uint8)]:
            with self.subTest(shape=mask.shape), self.assertRaises(AnnotationError):
                encode_mask(mask)

    def test_invalid_or_excessive_dimensions_are_rejected_before_allocation(self):
        for width, height in [(True, 3), (4, 0), (-4, 3), (4.0, 3), (100_000, 100_000)]:
            with self.subTest(size=(width, height)), self.assertRaises(AnnotationError):
                decode_segmentation([], width, height)

    def test_official_random_masks_validate_and_roundtrip(self):
        rng = np.random.default_rng(42)
        for height, width in [(1, 1), (3, 4), (23, 49), (100, 80)]:
            for fraction in [0.0, 0.05, 0.5, 0.95, 1.0]:
                mask = (rng.random((height, width)) < fraction).astype(np.uint8)
                official = mask_api.encode(np.asfortranarray(mask))
                with self.subTest(size=(height, width), fraction=fraction):
                    np.testing.assert_array_equal(decode_segmentation(official, width, height), mask)


if __name__ == "__main__":
    unittest.main()
