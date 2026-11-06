"""Regression checks for honest physical-error aggregation and coverage."""

import unittest

from measurement.evaluate_placements import measurement_row, summary


class MeasurementEvaluationTests(unittest.TestCase):
    def row(self, index, prediction=(71, 164), new=True):
        case = {"image_id": f"case_{index:02d}", "filename": f"image_{index}.jpg", "sha256": str(index),
                "previously_used_for_development": not new}
        reading = {"hand_width_mm": "70", "hand_height_mm": "160", "reading_source": "confirmed ruler reading"}
        result = {"status": "success" if prediction else "reference_not_found",
                  "width_mm": prediction[0] if prediction else None, "height_mm": prediction[1] if prediction else None,
                  "detection_confidence": 0.99, "failure_reason": None if prediction else "Missing paper",
                  "outputs": {"overlay": "overlay.jpg"}}
        return measurement_row(case, reading, result)

    def test_refusal_remains_in_coverage_but_never_becomes_zero_error(self):
        rows = [self.row(i) for i in range(9)] + [self.row(9, None)]
        report = summary(rows)
        self.assertEqual(report["accepted_count"], 9)
        self.assertEqual(report["refused_count"], 1)
        self.assertEqual(report["dimensions"]["width"]["mae_mm"], 1)
        self.assertEqual(report["dimensions"]["height"]["mae_mm"], 4)
        self.assertEqual(report["dimensions"]["height"]["mean_percentage_error"], 2.5)
        self.assertIsNone(rows[-1]["width_absolute_error_mm"])
        self.assertFalse(report["mo1"]["full_pass"])

    def test_all_refused_produces_missing_statistics_not_fake_zero(self):
        report = summary([self.row(i, None) for i in range(10)])
        self.assertEqual(report["coverage"], 0)
        self.assertIsNone(report["dimensions"]["width"]["mae_mm"])
        self.assertIsNone(report["dimensions"]["height"]["worst_percentage_error"])
        self.assertFalse(report["mo1"]["accepted_only_numeric_targets_met"])

    def test_original_targets_need_full_coverage_and_worst_error(self):
        rows = [self.row(i, (70, 160)) for i in range(10)]
        self.assertTrue(summary(rows)["mo1"]["full_pass"])
        rows[-1] = self.row(9, (79, 160))
        report = summary(rows)
        self.assertLess(report["dimensions"]["width"]["mae_mm"], 3)
        self.assertEqual(report["dimensions"]["width"]["worst_absolute_error_mm"], 9)
        self.assertFalse(report["mo1"]["full_pass"])

    def test_reused_development_scene_does_not_prove_ten_new_placements(self):
        rows = [self.row(i, (70, 160), new=i != 0) for i in range(10)]
        report = summary(rows)
        self.assertEqual(report["new_scenes_only"]["scene_count"], 9)
        self.assertTrue(report["mo1"]["accepted_only_numeric_targets_met"])
        self.assertFalse(report["mo1"]["ten_new_valid_placements_available"])
        self.assertFalse(report["mo1"]["full_pass"])

    def test_incomplete_or_numeric_refusal_results_cannot_hide_missing_scenes(self):
        with self.assertRaises(ValueError):
            summary([self.row(i) for i in range(9)])
        case = {"image_id": "x", "filename": "x.jpg", "sha256": "x", "previously_used_for_development": False}
        reading = {"hand_width_mm": 70, "hand_height_mm": 160, "reading_source": "ruler"}
        with self.assertRaises(ValueError):
            measurement_row(case, reading, {"status": "reference_not_found", "width_mm": 70, "height_mm": 160})
