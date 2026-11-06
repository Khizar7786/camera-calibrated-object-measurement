"""Replay saved masks/geometry and independently check paired error statistics; no model inference."""

from datetime import datetime, timezone
from decimal import Decimal
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT))
from measurement.planar import measure_mask_contour_mm


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-9), (actual, expected)


def verify():
    declaration = json.loads((ROOT / "manifest.json").read_text())
    baseline_path = PROJECT / declaration["baseline_manifest"]
    assert sha(baseline_path) == declaration["baseline_manifest_sha256"]
    baseline = json.loads(baseline_path.read_text())
    for name, digest in declaration["source_sha256"].items():
        assert sha(PROJECT / name) == digest, name
    old_checksums_path = PROJECT / declaration["baseline_all_artifacts_checksum_file"]
    assert sha(old_checksums_path) == declaration["baseline_all_artifacts_checksum_file_sha256"]
    for name, digest in json.loads(old_checksums_path.read_text()).items():
        assert sha(old_checksums_path.parent / name) == digest, name
    out = ROOT / "results"
    report = json.loads((out / "summary.json").read_text())
    rows = json.loads((out / "per_mode.json").read_text())
    pairs = json.loads((out / "paired.json").read_text())
    errors, percentages, effective, changes = {}, {}, set(), {}
    original_control_matches, mask_differences = [], []
    for mode in ("corrected", "raw"):
        errors[mode], percentages[mode] = {}, {}
        assert len(rows[mode]) == len(baseline["cases"]) == 10
        for case, row in zip(baseline["cases"], rows[mode]):
            identifier = case["image_id"]
            assert identifier == row["image_id"]
            folder = out / mode / identifier
            result = json.loads((folder / "result.json").read_text())
            assert sha(baseline_path.parent / case["input"]) == case["sha256"] == result["input"]["sha256"]
            assert result["checkpoint"]["loaded"] and result["checkpoint"]["sha256"] == declaration["checkpoint_sha256"]
            assert result["configuration"]["file_sha256"] == declaration["inference_configuration_sha256"]
            effective.add(result["configuration"]["effective_sha256"])
            assert result["undistortion_applied"] is (mode == "corrected")
            expected_frame = "undistorted_full_frame_P_equals_K" if mode == "corrected" else "raw_distorted_full_frame"
            assert result["coordinate_system"] == result["reference"]["coordinate_system"] == expected_frame
            assert datetime.fromisoformat(result["started_at_utc"]) > datetime.fromisoformat(declaration["declared_before_comparison_at_utc"])
            for filename in ("overlay.jpg", "mask.png", "corrected.jpg" if mode == "corrected" else "raw_frame.jpg"):
                with Image.open(folder / filename) as image:
                    assert list(image.size) == case["stored_size"] == [2296, 4080]
            if mode == "corrected":
                old = baseline_path.parent / "results/cases" / identifier
                old_result = json.loads((old / "result.json").read_text())
                assert all(result[key] == old_result[key] for key in ("status", "width_mm", "height_mm", "detection_confidence"))
                assert sha(folder / "mask.png") == sha(old / "mask.png")
                original_control_matches.append(identifier)
            else:
                assert "corners_corrected_px" not in result["reference"]
                if sha(folder / "mask.png") != sha(out / "corrected" / identifier / "mask.png"):
                    mask_differences.append(identifier)
            if result["status"] == "success":
                key = "corners_corrected_px" if mode == "corrected" else "corners_raw_px"
                corners = result["reference"][key]
                mask = cv2.imread(str(folder / "mask.png"), cv2.IMREAD_GRAYSCALE)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = np.concatenate([c.reshape(-1, 2) for c in contours])
                geometry = measure_mask_contour_mm(contour, corners, 50, 50)
                errors[mode][identifier], percentages[mode][identifier] = {}, {}
                for dimension in ("width", "height"):
                    close(result[dimension + "_mm"], geometry[dimension + "_mm"])
                    hand = Decimal(str(row["hand_" + dimension + "_mm"]))
                    error = abs(Decimal(str(result[dimension + "_mm"])) - hand)
                    percentage = 100 * error / hand
                    close(row[dimension + "_absolute_error_mm"], error)
                    close(row[dimension + "_percentage_error"], percentage)
                    errors[mode][identifier][dimension] = float(error)
                    percentages[mode][identifier][dimension] = float(percentage)
            else:
                assert not row["accepted"]
                for dimension in ("width", "height"):
                    assert result[dimension + "_mm"] is row[dimension + "_absolute_error_mm"] is row[dimension + "_percentage_error"] is None
    assert len(effective) == 1
    paired = sorted(set(errors["raw"]) & set(errors["corrected"]))
    assert paired == report["paired_valid_ids"]
    assert report["paired_valid_count"] == len(paired)
    for group in ("all_valid_per_mode", "paired_valid_only"):
        for mode in errors:
            ids = list(errors[mode]) if group == "all_valid_per_mode" else paired
            record = report[group][mode]
            assert record["accepted_count"] == len(ids)
            for d in ("width", "height"):
                absolute = [errors[mode][i][d] for i in ids]
                percentage = [percentages[mode][i][d] for i in ids]
                expected = {"mae_mm": statistics.mean(absolute), "median_absolute_error_mm": statistics.median(absolute),
                            "worst_absolute_error_mm": max(absolute), "mean_percentage_error": statistics.mean(percentage),
                            "median_percentage_error": statistics.median(percentage), "worst_percentage_error": max(percentage)}
                for name, value in expected.items():
                    close(record["dimensions"][d][name], value)
    for d in ("width", "height"):
        delta = [errors["raw"][i][d] - errors["corrected"][i][d] for i in paired]
        changes[d] = statistics.mean(delta)
        close(report["paired_error_changes"][d]["mean_absolute_error_change_raw_minus_corrected_mm"], statistics.mean(delta))
        close(report["paired_error_changes"][d]["median_absolute_error_change_raw_minus_corrected_mm"], statistics.median(delta))
        change_record = report["paired_error_changes"][d]
        close(change_record["minimum_absolute_error_change_mm"], min(delta))
        close(change_record["maximum_absolute_error_change_mm"], max(delta))
        percentage_delta = [percentages["raw"][i][d] - percentages["corrected"][i][d] for i in paired]
        close(change_record["mean_percentage_error_change_points"], statistics.mean(percentage_delta))
        assert change_record["skipping_worse_count"] == sum(x > 1e-9 for x in delta)
        assert change_record["skipping_better_count"] == sum(x < -1e-9 for x in delta)
        assert change_record["effectively_equal_count"] == sum(abs(x) <= 1e-9 for x in delta)
        for row in pairs:
            if row["image_id"] in paired:
                close(row[d + "_absolute_error_raw_minus_corrected_mm"], errors["raw"][row["image_id"]][d] - errors["corrected"][row["image_id"]][d])
            else:
                assert row[d + "_absolute_error_raw_minus_corrected_mm"] is None
    pooled = {mode: statistics.mean(errors[mode][i][d] for i in paired for d in ("width", "height")) for mode in errors}
    assert report["pooled_paired_dimensions"]["dimension_count"] == 2 * len(paired)
    for mode in pooled:
        close(report["pooled_paired_dimensions"][mode + "_mae_mm"], pooled[mode])
        assert report["all_valid_per_mode"][mode]["refused_count"] == 10 - len(errors[mode])
        close(report["all_valid_per_mode"][mode]["coverage"], len(errors[mode]) / 10)
    close(report["pooled_paired_dimensions"]["raw_minus_corrected_mae_mm"], pooled["raw"] - pooled["corrected"])
    for csv_name, expected_rows in (("per_mode.csv", rows["corrected"] + rows["raw"]), ("paired.csv", pairs)):
        with (out / csv_name).open(newline="", encoding="utf-8") as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == len(expected_rows)
        for saved, expected in zip(csv_rows, expected_rows):
            for key, value in expected.items():
                if value is None:
                    assert saved[key] == "", (csv_name, key)
                elif isinstance(value, float):
                    close(saved[key], value)
                else:
                    assert saved[key] == str(value), (csv_name, key)
    for name, digest in json.loads((out / "artifacts_sha256.json").read_text()).items():
        assert sha(out / name) == digest, name
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "mode_runs_verified": 20,
              "all_prior_placement_artifacts_preserved": True, "corrected_controls_exactly_match_prior_results_and_masks": original_control_matches,
              "raw_and_corrected_effective_configuration_identical": True, "checkpoint_sha256": declaration["checkpoint_sha256"],
              "own_frame_geometry_reconstructed_from_each_saved_mask_and_reference": True,
              "independent_decimal_errors_and_all_summary_statistics_match": True,
              "paired_valid_ids": paired, "paired_valid_count": len(paired),
              "paired_mae_change_skipping_correction_mm": changes,
              "different_mask_file_ids": mask_differences,
              "mask_file_difference_interpretation": "Different frame pixels; not an aligned-mask IoU or attribution to semantic segmentation alone",
              "all_source_configuration_input_and_output_hashes_verified": True,
              "tests_passed_before_inference": declaration["preflight_tests"]["passed"],
              "verification_command": ".venv\\Scripts\\python.exe -m measurement.paired_undistortion_10.verify_results"}
    # Replay is read-only: preserve the committed verification timestamp/hash.
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
