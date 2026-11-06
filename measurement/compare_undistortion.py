"""Paired end-to-end raw-versus-corrected ablation on frozen measurement scenes."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

from .evaluate_placements import PROJECT, load_manifest, measurement_row, sha256, summary, write_json


MODES = {"corrected": ("inference.measure", True, "undistorted_full_frame_P_equals_K"),
         "raw": ("measurement.raw_inference", False, "raw_distorted_full_frame")}


def mode_statistics(rows):
    dimensions = summary(rows, expected_count=len(rows))["dimensions"] if rows else {
        d: {"accepted_dimension_count": 0, "mae_mm": None, "median_absolute_error_mm": None,
            "worst_absolute_error_mm": None, "mean_percentage_error": None,
            "median_percentage_error": None, "worst_percentage_error": None} for d in ("width", "height")}
    count = sum(r["accepted"] for r in rows)
    return {"scene_count": len(rows), "accepted_count": count, "refused_count": len(rows) - count,
            "coverage": count / len(rows) if rows else None, "dimensions": dimensions}


def paired_statistics(by_mode):
    index = {mode: {r["image_id"]: r for r in rows} for mode, rows in by_mode.items()}
    ids = list(index["corrected"])
    if len(ids) != len(by_mode["corrected"]) or len(index["raw"]) != len(by_mode["raw"]) \
            or set(ids) != set(index["raw"]):
        raise ValueError("Both modes must contain the same unique scene IDs, including refusals.")
    paired, rows = [], []
    for identifier in ids:
        corrected, raw = index["corrected"][identifier], index["raw"][identifier]
        if corrected["input_sha256"] != raw["input_sha256"] or any(
                corrected[f"hand_{d}_mm"] != raw[f"hand_{d}_mm"] for d in ("width", "height")):
            raise ValueError("A pair must use the same original image and hand readings.")
        valid = corrected["accepted"] and raw["accepted"]
        row = {"image_id": identifier, "filename": corrected["filename"], "paired_valid": valid,
               "corrected_status": corrected["status"], "raw_status": raw["status"],
               "corrected_reason": corrected["reason"], "raw_reason": raw["reason"]}
        for d in ("width", "height"):
            row[f"hand_{d}_mm"] = corrected[f"hand_{d}_mm"]
            for mode, source in (("corrected", corrected), ("raw", raw)):
                for key in (f"predicted_{d}_mm", f"{d}_absolute_error_mm", f"{d}_percentage_error"):
                    row[f"{mode}_{key}"] = source[key]
            row[f"{d}_prediction_raw_minus_corrected_mm"] = raw[f"predicted_{d}_mm"] - corrected[f"predicted_{d}_mm"] if valid else None
            row[f"{d}_absolute_error_raw_minus_corrected_mm"] = raw[f"{d}_absolute_error_mm"] - corrected[f"{d}_absolute_error_mm"] if valid else None
            row[f"{d}_percentage_error_raw_minus_corrected_points"] = raw[f"{d}_percentage_error"] - corrected[f"{d}_percentage_error"] if valid else None
        for mode, source in (("corrected", corrected), ("raw", raw)):
            row[f"{mode}_detection_confidence"] = source["detection_confidence"]
            row[f"{mode}_result"] = source["result"]
            row[f"{mode}_overlay"] = source["overlay"]
        rows.append(row)
        if valid:
            paired.append(identifier)
    paired_modes = {mode: mode_statistics([index[mode][identifier] for identifier in paired]) for mode in MODES}
    changes = {}
    for d in ("width", "height"):
        delta = [r[f"{d}_absolute_error_raw_minus_corrected_mm"] for r in rows if r["paired_valid"]]
        percentage = [r[f"{d}_percentage_error_raw_minus_corrected_points"] for r in rows if r["paired_valid"]]
        changes[d] = {"paired_dimension_count": len(delta),
                      "mean_absolute_error_change_raw_minus_corrected_mm": statistics.mean(delta) if delta else None,
                      "median_absolute_error_change_raw_minus_corrected_mm": statistics.median(delta) if delta else None,
                      "minimum_absolute_error_change_mm": min(delta) if delta else None,
                      "maximum_absolute_error_change_mm": max(delta) if delta else None,
                      "mean_percentage_error_change_points": statistics.mean(percentage) if percentage else None,
                      "skipping_worse_count": sum(x > 1e-9 for x in delta),
                      "skipping_better_count": sum(x < -1e-9 for x in delta),
                      "effectively_equal_count": sum(abs(x) <= 1e-9 for x in delta)}
    pooled = {mode: [index[mode][identifier][f"{d}_absolute_error_mm"] for identifier in paired for d in ("width", "height")]
              for mode in MODES}
    pooled_means = {mode: statistics.mean(values) if values else None for mode, values in pooled.items()}
    report = {"supplied_scene_count": len(ids), "paired_valid_count": len(paired), "paired_valid_ids": paired,
              "unpaired_ids": [i for i in ids if i not in paired],
              "all_valid_per_mode": {mode: mode_statistics(by_mode[mode]) for mode in MODES},
              "paired_valid_only": paired_modes, "paired_error_changes": changes,
              "delta_sign": "raw absolute error minus corrected absolute error; positive means skipping worsened error",
              "error_denominators": "Per-mode statistics use that mode's accepted count; paired statistics use identical jointly accepted IDs. Refusals are missing, never zero.",
              "pooled_paired_dimensions": {"label": "Optional pooled MAE over width and height, not a per-dimension MAE",
                                           "dimension_count": 2 * len(paired), **{f"{mode}_mae_mm": value for mode, value in pooled_means.items()},
                                           "raw_minus_corrected_mae_mm": pooled_means["raw"] - pooled_means["corrected"] if paired else None}}
    return rows, report


def validate_result(mode, case, result, config):
    _, undistorted, frame = MODES[mode]
    if result["input"]["sha256"] != case["sha256"] or result["undistortion_applied"] is not undistorted:
        raise ValueError("Result does not match its original image or declared correction mode.")
    if result["coordinate_system"] != frame or result["reference"]["coordinate_system"] != frame:
        raise ValueError("Reference and mask must share the mode's own coordinate system.")
    if not result["checkpoint"]["loaded"] or result["checkpoint"]["sha256"] != config["checkpoint_sha256"]:
        raise ValueError("Every mode must run the same frozen checkpoint, including reference refusals.")
    if result["reference"]["width_mm"] != config["reference"]["width_mm"] \
            or result["reference"]["height_mm"] != config["reference"]["height_mm"]:
        raise ValueError("Physical reference dimensions differ between modes.")
    transform = result["model_transform"]
    if transform["score_cutoff"] != 0.5 or transform["mask_probability_threshold"] != 0.5 or not transform["aspect_preserving"]:
        raise ValueError("Frozen score/mask settings or input geometry changed.")
    if mode == "raw" and "corners_corrected_px" in result["reference"]:
        raise ValueError("Raw output must not contain corrected reference coordinates.")


def compare(manifest_path, output, device):
    manifest_path, output = manifest_path.resolve(), output.resolve()
    declaration = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_path = PROJECT / declaration["baseline_manifest"]
    if sha256(baseline_path) != declaration["baseline_manifest_sha256"]:
        raise ValueError("Baseline scene declaration changed.")
    baseline, readings, config = load_manifest(baseline_path)
    def verify_sources():
        for name, digest in declaration["source_sha256"].items():
            if sha256(PROJECT / name) != digest:
                raise ValueError(f"Experiment source changed: {name}")
    verify_sources()
    if output.exists():
        raise ValueError("Use a fresh output directory; previous evidence is never overwritten.")
    output.mkdir(parents=True)
    by_mode, commands = {mode: [] for mode in MODES}, []
    effective_ids, transform_sizes = set(), set()
    declared_hash = sha256(manifest_path)
    for case in baseline["cases"]:
        image = baseline_path.parent / case["input"]
        for mode, (module, _, _) in MODES.items():
            destination = output / mode / case["image_id"]
            command = [sys.executable, "-m", module, str(image.relative_to(PROJECT)),
                       "--config", baseline["frozen"]["inference_configuration"],
                       "--output", str(destination.relative_to(PROJECT)), "--device", device]
            if mode == "raw":
                command.append("--experimental-skip-undistortion")
            commands.append(subprocess.list2cmdline(command))
            execution = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, encoding="utf-8")
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "console.txt").write_text(execution.stdout + execution.stderr, encoding="utf-8")
            write_json(destination / "execution.json", {"command": command, "returncode": execution.returncode})
            result = json.loads((destination / "result.json").read_text(encoding="utf-8"))
            if execution.returncode not in (0, 2) or (execution.returncode == 0) != (result["status"] == "success"):
                raise ValueError("CLI exit code disagrees with result; preserve and inspect partial evidence.")
            validate_result(mode, case, result, config)
            effective_ids.add(result["configuration"]["effective_sha256"])
            transform_sizes.add(tuple(result["model_transform"]["input_size"]))
            row = measurement_row(case, readings[case["image_id"]], result)
            row.update(mode=mode, coordinate_system=result["coordinate_system"],
                       result=f"{mode}/{case['image_id']}/result.json", overlay=f"{mode}/{case['image_id']}/overlay.jpg")
            by_mode[mode].append(row)
            print(f"{case['image_id']} {mode}: {result['status']}", flush=True)
    verify_sources()
    load_manifest(baseline_path)
    if sha256(manifest_path) != declared_hash or len(effective_ids) != 1 or len(transform_sizes) != 1:
        raise ValueError("Declaration or effective configuration/geometry changed between modes.")
    pairs, report = paired_statistics(by_mode)
    write_json(output / "per_mode.json", by_mode)
    write_json(output / "paired.json", pairs)
    write_json(output / "summary.json", report)
    for name, rows in (("per_mode.csv", [r for mode in MODES for r in by_mode[mode]]), ("paired.csv", pairs)):
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_json(output / "run.json", {"status": "complete", "manifest_sha256": declared_hash,
                                    "baseline_manifest_sha256": declaration["baseline_manifest_sha256"],
                                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "command": subprocess.list2cmdline([sys.executable, "-m", "measurement.compare_undistortion", *sys.argv[1:]]),
                                    "effective_configuration_sha256": next(iter(effective_ids)),
                                    "checkpoint_sha256": config["checkpoint_sha256"], "device": device,
                                    "model_or_threshold_changes": False, "modes_run_afresh": True,
                                    "shared_conversion": "measurement.planar.measure_mask_contour_mm"})
    write_json(output / "artifacts_sha256.json", {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    compare(args.manifest, args.output, args.device)


if __name__ == "__main__":
    main()
