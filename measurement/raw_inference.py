"""EXPERIMENT ONLY: measure a distorted original frame without undistortion.

Normal inference remains inference.measure and always corrects the camera image.
This separate entry point deliberately bypasses only the lens remap, for an
end-to-end ablation on the same frozen images, model, reference and thresholds.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch

from inference import measure as normal


FRAME = "raw_distorted_full_frame"


def detect_raw_reference(frame, settings):
    # The frozen detector is run afresh ON THIS RAW FRAME. Rename its legacy
    # normal-inference labels; no corrected corners are read or transformed.
    reference = normal.detect_reference(frame, settings)
    reference["corners_raw_px"] = reference.pop("corners_corrected_px")
    reference["coordinate_system"] = FRAME
    reference["corner_refinement"] = reference["corner_refinement"].replace("corrected-image", "raw-image")
    return reference


def raw_geometry(contour, corners, width, height):
    # Explicit experimental exception to the normal corrected-frame contract:
    # BOTH contour and freshly detected reference corners are raw pixels here.
    # The mathematical conversion itself is the exact same frozen function.
    geometry = normal.measure_mask_contour_mm(contour, corners, width, height)
    geometry["homography_raw_pixels_to_mm"] = geometry.pop("homography_corrected_pixels_to_mm")
    geometry["rectangle_raw_px"] = geometry.pop("rectangle_corrected_px")
    geometry["coordinate_system"] = FRAME
    return geometry


def save_raw_overlay(output, frame, mask, result):
    overlay = frame.copy()
    if mask is not None:
        selected = mask != 0
        overlay[selected] = (0.65 * overlay[selected] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 200, 0), 3)
        normal.save_image(output / "mask.png", mask * 255)
        result["outputs"]["mask"] = "mask.png"
    corners = result.get("reference", {}).get("corners_raw_px")
    if corners is not None:
        points = np.rint(corners).astype(np.int32)
        cv2.polylines(overlay, [points], True, (255, 150, 0), 4)
        for index, point in enumerate(points):
            cv2.putText(overlay, str(index), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3)
    if result["status"] == "success":
        rectangle = np.clip(np.rint(result["metric_geometry"]["rectangle_raw_px"]), -1e6, 1e6).astype(np.int32)
        cv2.polylines(overlay, [rectangle], True, (0, 0, 255), 4)
        label = f"width={result['width_mm']:.2f} mm height={result['height_mm']:.2f} mm"
    else:
        label = result["status"] + ": " + str(result["failure_reason"])
    cv2.putText(overlay, "EXPERIMENT ONLY: RAW, UNDISTORTION BYPASSED", (25, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 3)
    cv2.putText(overlay, label[:110], (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 3)
    normal.save_image(output / "overlay.jpg", overlay)
    result["outputs"]["overlay"] = "overlay.jpg"


def run(image_path, output, config_path=Path("inference/config.json"), *,
        experimental_skip_undistortion=False, device_name="cuda", command=None):
    if experimental_skip_undistortion is not True:
        raise ValueError("Explicit experimental_skip_undistortion=True is required; normal inference always undistorts.")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh output directory; existing evidence is never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    if command is not None:
        (output / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    result = {"schema_version": 1, "status": "initializing", "width_mm": None, "height_mm": None,
              "detection_confidence": None, "confidence_definition": "Calculator detection score; not dimensional accuracy",
              "failure_reason": None, "reference": {}, "checkpoint": {}, "outputs": {},
              "input": {"path": str(image_path)}, "undistortion_applied": False,
              "coordinate_system": FRAME, "experimental_only": True,
              "experiment": "Deliberate undistortion bypass; not normal inference or a replacement calibration",
              "started_at_utc": datetime.now(timezone.utc).isoformat()}
    frame = mask = None
    try:
        config = normal.load_configuration(config_path)
        result["reference"] = deepcopy(config["reference"])
        result["checkpoint"] = {"path": config["checkpoint"], "sha256": config["checkpoint_sha256"], "loaded": False}
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        result["configuration"] = {"path": str(config_path), "file_sha256": normal.sha256(config_path),
                                   "effective_sha256": hashlib.sha256(canonical).hexdigest()}
        normal.write_json(output / "effective_configuration.json", config)
        result["input"]["sha256"] = normal.sha256(image_path)
        calibration_path = normal.PROJECT / config["calibration"]
        camera = normal.CameraCalibration.load(calibration_path)
        result["calibration"] = {"path": str(calibration_path), "sha256": normal.sha256(calibration_path),
                                 "image_size": list(camera.image_size), "use": "Geometry validation only; remap deliberately bypassed"}
        frame, info = normal.camera_frame(image_path, camera, config)
        result["input"].update(info)
        normal.save_image(output / "raw_frame.jpg", frame)
        result["outputs"]["raw_frame"] = "raw_frame.jpg"
        normal.seed_everything(42)
        torch.set_num_threads(4)
        if device_name == "cuda" and not torch.cuda.is_available():
            raise normal.InferenceFailure("model_error", "CUDA unavailable; choose --device cpu explicitly if required.")
        device = torch.device(device_name)
        normal.write_json(output / "environment.json", normal.environment())
        result["reference"] = detect_raw_reference(frame, config["reference"])
        model, training, checkpoint_info = normal.load_segmenter(config, device)
        result["checkpoint"] = {**checkpoint_info, "loaded": True}
        mask, confidence, transform = normal.segment_calculator(frame, model, training, device)
        transform["raw_size"] = transform.pop("corrected_size")
        transform["mask_restoration"] = transform["mask_restoration"].replace("corrected full frame", "raw full frame")
        transform["coordinate_system"] = FRAME
        result.update(detection_confidence=confidence, model_transform=transform)
        if mask is None or not np.any(mask):
            raise normal.InferenceFailure("object_not_found", "No nonempty calculator mask at frozen score >=0.5.")
        reference = result["reference"]
        if reference["status"] != "found":
            raise normal.InferenceFailure("reference_not_found", reference["failure_reason"])
        if reference.get("physically_measured") is not True:
            raise normal.InferenceFailure("reference_size_required", "Physically measure both outer paper edges and supply their mm sizes.")
        if np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]]):
            raise normal.InferenceFailure("invalid_object_geometry", "Calculator mask touches the original camera-frame border.")
        corners = np.asarray(reference["corners_raw_px"], dtype=np.float64)
        if np.any(corners < 0) or np.any(corners[:, 0] > frame.shape[1] - 1) or np.any(corners[:, 1] > frame.shape[0] - 1):
            raise normal.InferenceFailure("invalid_reference_geometry", "Reference corners fall outside the raw full frame.")
        # Raw source pixels have no remap-created invalid border. Bounds and
        # frame-touch checks are otherwise the same as the corrected command.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
        geometry = raw_geometry(contour, corners, reference["width_mm"], reference["height_mm"])
        normal.write_json(output / "geometry.json", {"contours_raw_px": [c.reshape(-1, 2).tolist() for c in contours], **geometry})
        result.update(status="success", width_mm=geometry["width_mm"], height_mm=geometry["height_mm"],
                      metric_geometry={key: value for key, value in geometry.items() if key != "contour_mm"})
        result["outputs"]["geometry"] = "geometry.json"
    except normal.CalibrationError as exc:
        result.update(status="invalid_camera_geometry", failure_reason=str(exc))
    except normal.MeasurementGeometryError as exc:
        result.update(status="invalid_planar_geometry", failure_reason=str(exc))
    except normal.InferenceFailure as exc:
        result.update(status=exc.status, failure_reason=str(exc))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as exc:
        result.update(status="inference_error", failure_reason=str(exc))
    if result["status"] != "success":
        result.update(width_mm=None, height_mm=None)
    if frame is not None:
        save_raw_overlay(output, frame, mask, result)
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    normal.write_json(output / "result.json", result)
    source_paths = [normal.PROJECT / name for name in ["calibration/camera.py", "inference/measure.py",
                    "inference/reference.py", "models/model.py", "measurement/planar.py", "measurement/raw_inference.py"]]
    normal.write_json(output / "source_hashes.json", {p.relative_to(normal.PROJECT).as_posix(): normal.sha256(p) for p in source_paths})
    print(json.dumps({key: result[key] for key in ("experimental_only", "undistortion_applied", "coordinate_system",
                     "status", "width_mm", "height_mm", "detection_confidence", "failure_reason")}), flush=True)
    return 0 if result["status"] == "success" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--experimental-skip-undistortion", action="store_true", required=True,
                        help="Required acknowledgement: bypass lens correction for this experiment only")
    parser.add_argument("--config", type=Path, default=Path("inference/config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    command = [sys.executable, "-m", "measurement.raw_inference", *(sys.argv[1:] if argv is None else argv)]
    return run(args.image, args.output, args.config, experimental_skip_undistortion=args.experimental_skip_undistortion,
               device_name=args.device, command=command)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
