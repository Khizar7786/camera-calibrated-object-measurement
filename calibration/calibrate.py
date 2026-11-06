"""Fit one explicitly chosen stored-pixel camera geometry; retain every outcome.

Run from the repository root: python -m calibration.calibrate
The output directory must be empty so a failed run cannot expose a stale fit.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

from .camera import CalibrationError, CameraCalibration, Undistorter, read_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-9)


def load_settings(path: Path) -> dict[str, Any]:
    settings = json.loads(path.read_text(encoding="utf-8"))
    for key in ("inner_corners", "image_size"):
        values = settings.get(key)
        if not isinstance(values, list) or len(values) != 2 or any(
            type(value) is not int or value <= 0 for value in values
        ):
            raise CalibrationError(f"{key} must contain two positive integers.")
    columns, rows = settings["inner_corners"]
    if min(columns, rows) < 7 or max(columns, rows) < 9:
        raise CalibrationError("The assessment requires at least 7 x 9 inner corners.")
    size = settings.get("square_size_mm")
    if isinstance(size, bool) or not isinstance(size, (float, int)) or not np.isfinite(size) or size <= 0:
        raise CalibrationError("square_size_mm must be a positive measured length.")
    for key in ("square_size_source", "corner_count_source"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise CalibrationError(f"{key} must describe the measured/verified board setting.")
    minimum = settings.get("minimum_views", 20)
    if type(minimum) is not int or minimum < 20:
        raise CalibrationError("minimum_views cannot be less than 20.")
    maximum = settings.get("detection_max_dimension", 1600)
    if type(maximum) is not int or maximum < 320:
        raise CalibrationError("detection_max_dimension must be an integer >= 320.")
    settings["minimum_views"] = minimum
    settings["detection_max_dimension"] = maximum
    return settings


def board_points(settings: dict[str, Any]) -> np.ndarray:
    columns, rows = settings["inner_corners"]
    points = np.zeros((columns * rows, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    points *= settings["square_size_mm"]
    return points


def detect_corners(gray: np.ndarray, pattern: tuple[int, int], maximum: int) -> tuple[np.ndarray | None, str]:
    """Detect on a bounded preview; refine in original full-resolution pixels."""
    height, width = gray.shape
    scale = min(1.0, maximum / max(width, height))
    small = cv2.resize(gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(small, pattern, flags)
    method = "findChessboardCorners + full-resolution cornerSubPix"
    if not found:
        found, corners = cv2.findChessboardCornersSB(small, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE)
        method = "findChessboardCornersSB + full-resolution cornerSubPix"
    if not found or corners is None:
        return None, "neither detector found the complete board"
    corners = np.asarray(corners, dtype=np.float32)
    if scale < 1:
        # OpenCV resize uses pixel-centre mapping, with independently rounded sizes.
        factors = np.array([width / small.shape[1], height / small.shape[0]], dtype=np.float32)
        corners = (corners + 0.5) * factors - 0.5
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    if refined is None or len(refined) != pattern[0] * pattern[1] or not np.isfinite(refined).all():
        return None, "corner refinement did not produce a finite complete grid"
    xy = refined.reshape(-1, 2)
    if np.any(xy < 0) or np.any(xy[:, 0] >= width) or np.any(xy[:, 1] >= height):
        return None, "refined corner outside the stored frame"
    return refined, method


def reprojection_errors(object_points: list[np.ndarray], image_points: list[np.ndarray],
                        matrix: np.ndarray, distortion: np.ndarray,
                        rotations: tuple, translations: tuple) -> tuple[list[float], float]:
    """RMS Euclidean residual per detected corner, in original-image pixels."""
    errors = []
    total_squared = 0.0
    total_corners = 0
    for objects, observed, rotation, translation in zip(object_points, image_points, rotations, translations):
        projected, _ = cv2.projectPoints(objects, rotation, translation, matrix, distortion)
        squared = np.square(observed.reshape(-1, 2).astype(np.float64) - projected.reshape(-1, 2)).sum(axis=1)
        errors.append(float(np.sqrt(squared.mean())))
        total_squared += float(squared.sum())
        total_corners += len(squared)
    return errors, float(np.sqrt(total_squared / total_corners))


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise CalibrationError(f"Cannot encode {path}.")
    path.write_bytes(encoded.tobytes())


def corner_preview(image: np.ndarray, corners: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    scale = min(1.0, 1000 / max(image.shape[:2]))
    preview = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
    factors = np.array([preview.shape[1] / image.shape[1], preview.shape[0] / image.shape[0]], dtype=np.float32)
    points = (corners + 0.5) * factors - 0.5
    cv2.drawChessboardCorners(preview, pattern, points, True)
    return preview


def save_previews(output: Path, accepted: list[dict], camera: CameraCalibration) -> list[str]:
    folder = output / "before_after"
    folder.mkdir()
    selected = sorted({0, len(accepted) // 2, len(accepted) - 1})
    paths = []
    mapper = Undistorter(camera)
    for index in selected:
        row = accepted[index]
        source = read_image(row["path"])
        corrected = mapper.apply(source)
        stem = Path(row["filename"]).stem
        write_image(folder / f"{stem}.before.jpg", source)
        write_image(folder / f"{stem}.after.jpg", corrected)
        scale = min(1.0, 1000 / source.shape[0])
        dimensions = (round(source.shape[1] * scale), round(source.shape[0] * scale))
        left = cv2.resize(source, dimensions, interpolation=cv2.INTER_AREA)
        right = cv2.resize(corrected, dimensions, interpolation=cv2.INTER_AREA)
        header = np.full((60, dimensions[0] * 2, 3), 255, dtype=np.uint8)
        cv2.putText(header, "RAW", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        cv2.putText(header, "UNDISTORTED: same K, full frame", (dimensions[0] + 12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
        cv2.putText(header, "Display preview only; saved before/after files retain original dimensions.", (12, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)
        preview = np.vstack([header, np.hstack([left, right])])
        name = f"before_after/{stem}.comparison.jpg"
        write_image(output / name, preview)
        paths.append(name)
    return paths


def run(config: Path, input_dir: Path, output: Path) -> int:
    settings = load_settings(config)
    files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise CalibrationError(f"No calibration images in {input_dir}.")
    if output.exists() and any(output.iterdir()):
        raise CalibrationError(f"Output directory is not empty: {output}. Use a fresh output path for each run.")
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = output / "detected_corners"
    diagnostics.mkdir()
    log = []

    def emit(message: str) -> None:
        print(message, flush=True)
        log.append(message)
        (output / "console.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    size = tuple(settings["image_size"])
    pattern = tuple(settings["inner_corners"])
    emit(f"Board: {pattern[0]} x {pattern[1]} inner corners; measured squares {settings['square_size_mm']:g} mm")
    emit(f"Geometry: {size[0]} x {size[1]} stored pixels; no EXIF rotation, crop, resize or mixed-orientation fit")
    emit(f"Inputs: {len(files)}; required usable views: {settings['minimum_views']}")
    object_template = board_points(settings)
    objects, points, rows, accepted = [], [], [], []
    for path in files:
        row = {"filename": path.name, "path": path.as_posix(), "sha256": None,
               "width": None, "height": None, "status": "failed", "reason": "", "corners": 0,
               "detector": "", "rms_px": None}
        try:
            row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            image = read_image(path)
            row["height"], row["width"] = image.shape[:2]
            if (row["width"], row["height"]) != size:
                row["status"] = "rejected_geometry"
                row["reason"] = f"stored size {row['width']} x {row['height']} differs from required {size[0]} x {size[1]}"
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                corners, method = detect_corners(gray, pattern, settings["detection_max_dimension"])
                row["detector"] = method
                if corners is None:
                    row["reason"] = "complete inner-corner grid not detected/refined"
                else:
                    write_image(diagnostics / f"{path.stem}.corners.jpg", corner_preview(image, corners, pattern))
                    row.update(status="usable", corners=len(corners), reason="complete grid refined at full resolution")
                    objects.append(object_template.copy())
                    points.append(corners)
                    accepted.append(row)
        except (CalibrationError, cv2.error, OSError) as exc:
            row["reason"] = str(exc)
        rows.append(row)
        emit(f"{row['status'].upper()}: {path.name}: {row['reason']}")

    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "calibration.calibrate", "--config", str(config), "--input", str(input_dir), "--output", str(output)],
        "settings": settings,
        "environment": {"python": sys.version, "opencv": cv2.__version__, "numpy": np.__version__,
                        "platform": platform.platform(), "processor": platform.processor(), "threads": 1, "seed": 42},
        "input_count": len(files), "usable_count": len(accepted), "rejected_count": len(files) - len(accepted),
        "views": rows, "status": "insufficient_views",
        "metric_definition": "sqrt(sum squared 2D corner residuals / number of corners), full-resolution pixels",
        "limitations": ["Reprojection fit is not evidence of millimetre measurement accuracy.",
                        "Focus and zoom lock are user-confirmed; EXIF cannot prove them.",
                        "Board flatness/rigid support are not independently verified.",
                        "Same-dimension rotations, crops or changed optics cannot be detected from array shape.",
                        "No views are dropped based on reprojection error."],
    }

    def save_report() -> None:
        (output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        fields = ["filename", "width", "height", "status", "reason", "corners", "detector", "rms_px", "sha256"]
        with (output / "views.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    emit(f"Detection summary: {len(accepted)} usable / {len(files)} input; {len(files) - len(accepted)} rejected/failed")
    save_report()
    if len(accepted) < settings["minimum_views"]:
        emit("CALIBRATION REFUSED: fewer than the required 20 (or configured higher minimum) usable views. No camera parameters saved.")
        return 2
    try:
        rms, matrix, distortion, rotations, translations, std_intrinsics, std_extrinsics, cv_errors = cv2.calibrateCameraExtended(
            objects, points, size, None, None, flags=0, criteria=CRITERIA)
        per_view, recomputed_rms = reprojection_errors(objects, points, matrix, distortion, rotations, translations)
        if not np.isfinite(rms) or not np.isclose(rms, recomputed_rms, rtol=1e-3, atol=1e-3):
            raise CalibrationError("Non-finite fit or independently recomputed RMS disagrees with OpenCV.")
        if not np.allclose(per_view, np.asarray(cv_errors).reshape(-1), rtol=1e-3, atol=1e-3):
            raise CalibrationError("Per-view RMS disagrees with OpenCV's raw per-view output.")
        for row, error in zip(accepted, per_view):
            row["rms_px"] = error
            emit(f"RMS_PX: {row['filename']}: {error:.9f}")
        camera_data = {
            "schema_version": 1, "model": "opencv_pinhole", "coordinate_system": "stored_pixels_full_frame",
            "camera_matrix": matrix.tolist(), "distortion_coefficients": distortion.reshape(-1).tolist(),
            "distortion_order": ["k1", "k2", "p1", "p2", "k3"], "image_size": list(size),
            "board": {key: settings[key] for key in ("inner_corners", "square_size_mm", "square_size_source", "corner_count_source")},
            "global_rms_px": float(rms), "recomputed_rms_px": recomputed_rms,
            "minimum_views": settings["minimum_views"], "usable_views": len(accepted), "input_views": len(files),
            "per_view_errors": [{"filename": row["filename"], "rms_px": row["rms_px"]} for row in accepted],
            "undistortion": {"output_camera_matrix": "same as camera_matrix (P=K)", "crop": False, "resize": False,
                             "output_image_size": list(size), "border": "constant zero; invalid pixels are not measurements"},
            "camera_settings": {key: settings.get(key) for key in ("camera", "focus_and_zoom_locked", "capture_settings_source", "board_flatness_and_rigid_backing")},
            "calibration_flags": 0, "seed": 42,
        }
        camera = CameraCalibration(matrix, distortion, size, camera_data)
        np.savez_compressed(output / "raw_calibration.npz", camera_matrix=matrix, distortion_coefficients=distortion,
                            rotation_vectors=np.asarray(rotations), translation_vectors=np.asarray(translations),
                            intrinsic_standard_deviations=std_intrinsics, extrinsic_standard_deviations=std_extrinsics,
                            opencv_per_view_errors=cv_errors, global_rms_px=np.asarray(rms), image_size=np.asarray(size),
                            object_points_mm=np.asarray(objects), refined_image_points_px=np.asarray(points),
                            filenames=np.asarray([row["filename"] for row in accepted]), flags=np.asarray(0))
        previews = save_previews(output, accepted, camera)
        report.update(status="calibrated", global_rms_px=float(rms), recomputed_rms_px=recomputed_rms,
                      per_view_median_px=float(np.median(per_view)), per_view_worst_px=max(per_view),
                      mo2_target_px=1.0, mo2_met=bool(rms <= 1.0), before_after_previews=previews)
        emit(f"GLOBAL_RMS_PX: {rms:.9f}; independently recomputed: {recomputed_rms:.9f}")
        emit(f"PER_VIEW_RMS_PX: median {np.median(per_view):.9f}; worst {max(per_view):.9f}")
        emit(f"CAMERA_MATRIX: {matrix.tolist()}")
        emit(f"DISTORTION_COEFFICIENTS [k1,k2,p1,p2,k3]: {distortion.reshape(-1).tolist()}")
        emit(f"MO-2 (RMS <= 1.0 px, >=20 usable views): {'MET' if rms <= 1.0 else 'MISSED'}")
        emit("Millimetre accuracy: NOT EVALUATED. Reprojection RMS alone does not establish it.")
        emit(f"Saved parameters, per-view outcomes, raw fit and before/after evidence under {output}")
        save_report()
        # Publish loadable parameters only after every other successful artifact.
        pending = output / "camera.json.pending"
        pending.write_text(json.dumps(camera_data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pending.replace(output / "camera.json")
        return 0
    except (CalibrationError, cv2.error, OSError, ValueError) as exc:
        (output / "camera.json.pending").unlink(missing_ok=True)
        report.update(status="fit_failed", error=str(exc))
        emit(f"CALIBRATION FAILED: {exc}")
        save_report()
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("calibration/board.json"))
    parser.add_argument("--input", type=Path, default=Path("calibration/raw"))
    parser.add_argument("--output", type=Path, default=Path("calibration/results"))
    args = parser.parse_args(argv)
    try:
        return run(args.config, args.input, args.output)
    except (CalibrationError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
