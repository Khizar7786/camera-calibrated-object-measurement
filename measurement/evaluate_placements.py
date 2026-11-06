"""Run the unchanged one-image CLI on a frozen, hand-measured placement manifest."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent.parent


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def positive(value, label):
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite and positive.")
    return number


def measurement_row(case, reading, result):
    """A refusal contributes coverage failure and missing errors, never zero error."""
    row = {"image_id": case["image_id"], "filename": case["filename"],
           "input_sha256": case["sha256"], "new_scene": not case["previously_used_for_development"],
           "hand_width_mm": positive(reading["hand_width_mm"], "Hand width"),
           "hand_height_mm": positive(reading["hand_height_mm"], "Hand height"),
           "predicted_width_mm": None, "predicted_height_mm": None,
           "width_absolute_error_mm": None, "height_absolute_error_mm": None,
           "width_percentage_error": None, "height_percentage_error": None,
           "detection_confidence": result.get("detection_confidence"),
           "accepted": result["status"] == "success", "status": result["status"],
           "reason": result.get("failure_reason") or "", "reading_source": reading["reading_source"],
           "result": f"cases/{case['image_id']}/result.json",
           "overlay": f"cases/{case['image_id']}/overlay.jpg" if "overlay" in result.get("outputs", {}) else ""}
    if row["hand_width_mm"] > row["hand_height_mm"]:
        raise ValueError("Hand width must describe the shorter physical side.")
    for dimension in ("width", "height"):
        prediction = result.get(f"{dimension}_mm")
        if row["accepted"]:
            prediction = positive(prediction, f"Predicted {dimension}")
            error = abs(prediction - row[f"hand_{dimension}_mm"])
            row.update({f"predicted_{dimension}_mm": prediction,
                        f"{dimension}_absolute_error_mm": error,
                        f"{dimension}_percentage_error": 100 * error / row[f"hand_{dimension}_mm"]})
        elif prediction is not None:
            raise ValueError("A refused CLI result must not contain numeric dimensions.")
    return row


def summary(rows, expected_count=10):
    if len(rows) != expected_count:
        raise ValueError(f"Expected exactly {expected_count} results, including refusals.")
    accepted = [row for row in rows if row["accepted"]]

    def dimension_stats(dimension, subset):
        absolute = [r[f"{dimension}_absolute_error_mm"] for r in subset]
        percentage = [r[f"{dimension}_percentage_error"] for r in subset]
        return {"accepted_dimension_count": len(subset),
                "mae_mm": statistics.mean(absolute) if absolute else None,
                "median_absolute_error_mm": statistics.median(absolute) if absolute else None,
                "worst_absolute_error_mm": max(absolute) if absolute else None,
                "mean_percentage_error": statistics.mean(percentage) if percentage else None,
                "median_percentage_error": statistics.median(percentage) if percentage else None,
                "worst_percentage_error": max(percentage) if percentage else None}

    dimensions = {d: dimension_stats(d, accepted) for d in ("width", "height")}
    numeric_targets = bool(accepted) and all(
        dimensions[d]["mae_mm"] <= 3 and dimensions[d]["worst_absolute_error_mm"] <= 8
        for d in dimensions)
    fresh = [r for r in rows if r["new_scene"]]
    fresh_accepted = [r for r in fresh if r["accepted"]]
    coverage = len(accepted) == expected_count
    complete_new_baseline = len(fresh) >= 10 and len(fresh_accepted) == len(fresh)
    full_pass = coverage and numeric_targets and complete_new_baseline
    return {"scope": "Repeatability of one physical calculator across photographed placements; not product generalization",
            "supplied_scene_count": len(rows), "accepted_count": len(accepted),
            "refused_count": len(rows) - len(accepted), "coverage": len(accepted) / len(rows),
            "error_denominator": "Accepted dimensions only, separately for width and height; refusals are missing, not zero",
            "dimensions": dimensions,
            "new_scenes_only": {"scene_count": len(fresh), "accepted_count": len(fresh_accepted),
                                "refused_count": len(fresh) - len(fresh_accepted),
                                "dimensions": {d: dimension_stats(d, fresh_accepted) for d in dimensions}},
            "mo1": {"targets": {"width_mae_mm_max": 3, "height_mae_mm_max": 3,
                                  "worst_dimension_error_mm_max": 8, "required_new_valid_placements": 10},
                    "all_supplied_scenes_measured": coverage,
                    "accepted_only_numeric_targets_met": numeric_targets,
                    "ten_new_valid_placements_available": complete_new_baseline,
                    "full_pass": full_pass, "verdict": "met" if full_pass else "not_met"}}


def table_markdown(rows, prefix=""):
    lines = ["| Image ID / input | Hand W / H (mm) | Predicted W / H (mm) | Absolute W / H error (mm) | W / H error (%) | Confidence | Accepted/refused | Reason / overlay |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    def pair(row, first, second):
        return "missing" if row[first] is None else f"{row[first]:.3f} / {row[second]:.3f}"
    for row in rows:
        input_link = f"{prefix}../inputs/{row['filename']}"
        label = row["image_id"] + (" (reused demo)" if not row["new_scene"] else "")
        reason = row["reason"].replace("|", "\\|").replace("\n", " ") or "None"
        if row["overlay"]:
            reason += f"; [overlay]({prefix}{row['overlay']})"
        confidence = "missing" if row["detection_confidence"] is None else f"{row['detection_confidence']:.6f}"
        lines.append(f"| [{label}]({input_link}) | {row['hand_width_mm']:g} / {row['hand_height_mm']:g} | "
                     f"{pair(row, 'predicted_width_mm', 'predicted_height_mm')} | "
                     f"{pair(row, 'width_absolute_error_mm', 'height_absolute_error_mm')} | "
                     f"{pair(row, 'width_percentage_error', 'height_percentage_error')} | {confidence} | "
                     f"{'Accepted' if row['accepted'] else 'Refused'} ({row['status']}) | {reason} |")
    return "\n".join(lines) + "\n"


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if manifest["schema_version"] != 1 or len(cases) != 10:
        raise ValueError("Schema 1 and exactly ten supplied placements are required.")
    if len({c["image_id"] for c in cases}) != 10 or len({c["sha256"] for c in cases}) != 10:
        raise ValueError("Placement IDs and input content must be unique.")
    if any(not c["split_disjoint_verified"] for c in cases):
        raise ValueError("Verify separation from train/validation/test before running.")
    frozen = manifest["frozen"]
    for name, digest in frozen["files_sha256"].items():
        if sha256(PROJECT / name) != digest:
            raise ValueError(f"Frozen configuration/source/data changed: {name}")
    config = json.loads((PROJECT / frozen["inference_configuration"]).read_text(encoding="utf-8"))
    if sha256(PROJECT / config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint SHA-256 mismatch.")
    readings_path = path.parent / manifest["hand_readings"]
    if sha256(readings_path) != manifest["hand_readings_sha256"]:
        raise ValueError("Hand readings changed after the experiment declaration.")
    with readings_path.open(newline="", encoding="utf-8") as stream:
        readings = list(csv.DictReader(stream))
    if len(readings) != 10 or {r["image_id"] for r in readings} != {c["image_id"] for c in cases}:
        raise ValueError("Every placement needs exactly one pre-recorded hand-reading row.")
    readings = {r["image_id"]: r for r in readings}
    for case in cases:
        image = (path.parent / case["input"]).resolve()
        if not image.is_relative_to(path.parent.resolve()) or sha256(image) != case["sha256"]:
            raise ValueError("Input path escapes the experiment or differs from the frozen photo.")
        reading = readings[case["image_id"]]
        positive(reading["hand_width_mm"], "Hand width")
        positive(reading["hand_height_mm"], "Hand height")
    return manifest, readings, config


def evaluate(manifest_path, output, device):
    manifest_path, output = manifest_path.resolve(), output.resolve()
    manifest, readings, config = load_manifest(manifest_path)
    if output.exists():
        raise ValueError("Use a fresh output directory; recorded evidence is never overwritten.")
    output.mkdir(parents=True)
    rows, commands = [], []
    manifest_hash = sha256(manifest_path)
    for case in manifest["cases"]:
        case_output = output / "cases" / case["image_id"]
        image = manifest_path.parent / case["input"]
        command = [sys.executable, "-m", "inference.measure", str(image.relative_to(PROJECT)),
                   "--config", manifest["frozen"]["inference_configuration"],
                   "--output", str(case_output.relative_to(PROJECT)), "--device", device]
        commands.append(subprocess.list2cmdline(command))
        execution = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, encoding="utf-8")
        case_output.mkdir(parents=True, exist_ok=True)
        (case_output / "console.txt").write_text(execution.stdout + execution.stderr, encoding="utf-8")
        write_json(case_output / "execution.json", {"command": command, "returncode": execution.returncode})
        result = json.loads((case_output / "result.json").read_text(encoding="utf-8"))
        if execution.returncode not in (0, 2) or (execution.returncode == 0) != (result["status"] == "success"):
            raise ValueError("CLI exit status disagrees with its saved result; inspect raw evidence.")
        if result["input"]["sha256"] != case["sha256"]:
            raise ValueError("CLI input hash mismatch.")
        if result["status"] == "success" and (not result["undistortion_applied"] or
                result["checkpoint"].get("sha256") != config["checkpoint_sha256"]):
            raise ValueError("Successful result did not use the frozen corrected pipeline.")
        rows.append(measurement_row(case, readings[case["image_id"]], result))
        print(f"{case['image_id']} {case['filename']}: {result['status']}", flush=True)
    # Catch changes to declared readings, files or source while the batch was running.
    load_manifest(manifest_path)
    if sha256(manifest_path) != manifest_hash:
        raise ValueError("Manifest changed during execution.")
    statistics_record = summary(rows)
    with (output / "per_instance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", statistics_record)
    write_json(output / "per_instance.json", rows)
    (output / "per_instance_table.md").write_text(table_markdown(rows), encoding="utf-8")
    (output / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_json(output / "run.json", {"status": "complete", "manifest": str(manifest_path.relative_to(PROJECT)),
                                    "manifest_sha256": manifest_hash, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                    "evaluation_source_sha256": sha256(__file__), "device": device,
                                    "command": subprocess.list2cmdline([sys.executable, "-m", "measurement.evaluate_placements", *sys.argv[1:]]),
                                    "checkpoint_sha256": config["checkpoint_sha256"],
                                    "threshold_or_model_changes": False, "summary": "summary.json"})
    write_json(output / "artifacts_sha256.json", {str(p.relative_to(output)): sha256(p)
                                                for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps(statistics_record, indent=2, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    evaluate(args.manifest, args.output, args.device)


if __name__ == "__main__":
    main()
