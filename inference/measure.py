"""Measure a calculator in one original camera image; undistortion is mandatory."""

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
from PIL import Image
import torch

from calibration.camera import CameraCalibration, CalibrationError, Undistorter, read_image
from measurement.planar import MeasurementGeometryError, measure_mask_contour_mm
from models.data import seed_everything
from models.model import build_model
from models.train import environment, sha256
from .reference import detect_reference, validate_reference


PROJECT = Path(__file__).resolve().parent.parent


class InferenceFailure(ValueError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_configuration(path, width_mm=None, height_mm=None):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or config.get("skip_undistortion", False):
        raise InferenceFailure("invalid_configuration", "Schema 1 and mandatory undistortion are required.")
    config = deepcopy(config)
    reference = config["reference"]
    if (width_mm is None) != (height_mm is None):
        raise InferenceFailure("invalid_configuration", "Supply both physically measured reference edges.")
    if width_mm is not None:
        reference.update(width_mm=width_mm, height_mm=height_mm, physically_measured=True,
                         measurement_source="User-supplied physically measured outer paper edges through CLI")
    if reference.get("physically_measured") is True:
        for key in ("width_mm", "height_mm"):
            value = reference.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value <= 0:
                raise InferenceFailure("invalid_configuration", "Measured reference edges must be finite positive mm values.")
        if not reference.get("measurement_source"):
            raise InferenceFailure("invalid_configuration", "Record the reference measurement source.")
    validate_reference(reference)
    return config


def camera_frame(image_path, camera, config):
    image = read_image(image_path)  # Reject EXIF auto-rotation before decoding.
    with Image.open(image_path) as original:
        model = original.getexif().get(272)
    expected = config["camera"].get("expected_exif_model")
    if model and expected and str(model).strip() != expected:
        raise CalibrationError(f"EXIF camera model {model!r} differs from calibrated profile {expected!r}.")
    width, height = image.shape[1], image.shape[0]
    rotation = config["camera"]["frame_rotations_cw"].get(f"{width}x{height}")
    if type(rotation) is not int or rotation not in (0, 90):
        raise CalibrationError(f"Unconfigured original resolution/orientation {width} x {height}; no resize/crop guess.")
    if rotation == 90:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if (image.shape[1], image.shape[0]) != camera.image_size:
        raise CalibrationError("Explicit frame normalization does not match the calibration size.")
    return image, {"stored_size": [width, height], "exif_camera_model": str(model) if model else None,
                   "camera_frame_rotation_cw": rotation, "camera_frame_size": list(camera.image_size),
                   "capture_condition": config["camera"]["capture_condition"]}


def load_segmenter(config, device):
    checkpoint_path = PROJECT / config["checkpoint"]
    digest = sha256(checkpoint_path)
    if digest != config["checkpoint_sha256"]:
        raise InferenceFailure("checkpoint_error", "Checkpoint SHA-256 differs from configured frozen weights.")
    training_path = PROJECT / config["training_configuration"]
    if sha256(training_path) != config["training_configuration_sha256"]:
        raise InferenceFailure("checkpoint_error", "Saved training configuration hash differs from the frozen ID.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if checkpoint["training_config"] != training or checkpoint["class_names"] != ["background", "calculator"]:
        raise InferenceFailure("checkpoint_error", "Checkpoint metadata/configuration does not match.")
    thresholds = training["validation"]
    if thresholds["precision_recall_score_threshold"] != 0.5 or thresholds["mask_threshold"] != 0.5:
        raise InferenceFailure("checkpoint_error", "Unexpected frozen calculator score/mask cutoff.")
    model = build_model(checkpoint["model_config"], pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    info = {"path": str(checkpoint_path), "sha256": digest, "epoch": checkpoint["epoch"],
            "training_configuration_sha256": config["training_configuration_sha256"],
            "classes": checkpoint["class_names"], "versions_at_training": checkpoint["versions"]}
    del checkpoint
    return model.to(device).eval(), training, info


def segment_calculator(corrected, model, training, device):
    height, width = corrected.shape[:2]
    settings = training["model"]
    scale = min(settings["min_size"] / min(width, height), settings["max_size"] / max(width, height))
    size = (round(width * scale), round(height * scale))
    rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().div_(255).to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                               enabled=device.type == "cuda" and training["training"]["mixed_precision"]):
        output = model([tensor])[0]
    scores = output["scores"].detach().cpu().float().numpy()
    labels = output["labels"].detach().cpu().numpy()
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise InferenceFailure("model_error", "Model returned invalid detection scores.")
    candidates = np.flatnonzero((labels == 1) & (scores >= 0.5))
    meta = {"input_size": list(size), "corrected_size": [width, height], "aspect_preserving": True,
            "score_cutoff": 0.5, "mask_probability_threshold": 0.5,
            "selection": "Highest-score calculator instance at frozen operating cutoff",
            "mask_restoration": "Threshold at model-input coordinates, then INTER_NEAREST_EXACT to corrected full frame"}
    if not len(candidates):
        return None, None, meta
    selected = int(candidates[np.argmax(scores[candidates])])
    probability = output["masks"][selected, 0].detach().cpu().float().numpy()
    if probability.shape != (size[1], size[0]) or not np.isfinite(probability).all() \
            or np.any((probability < 0) | (probability > 1)):
        raise InferenceFailure("model_error", "Model mask is invalid or uses unexpected coordinates.")
    mask = cv2.resize((probability >= 0.5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST_EXACT)
    return mask, float(scores[selected]), {**meta, "selected_output_index": selected}


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}.")


def save_overlay(output, corrected, mask, result):
    overlay = corrected.copy()
    if mask is not None:
        selected = mask != 0
        overlay[selected] = (0.65 * overlay[selected] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 200, 0), 3)
        save_image(output / "mask.png", mask * 255)
        result["outputs"]["mask"] = "mask.png"
    corners = result.get("reference", {}).get("corners_corrected_px")
    if corners is not None:
        points = np.rint(corners).astype(np.int32)
        cv2.polylines(overlay, [points], True, (255, 150, 0), 4)
        for index, point in enumerate(points):
            cv2.putText(overlay, str(index), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 3)
    if result["status"] == "success":
        rectangle = np.clip(np.rint(result["metric_geometry"]["rectangle_corrected_px"]), -1e6, 1e6).astype(np.int32)
        cv2.polylines(overlay, [rectangle], True, (0, 0, 255), 4)
        label = f"width={result['width_mm']:.2f} mm  height={result['height_mm']:.2f} mm"
    else:
        label = result["status"] + ": " + str(result["failure_reason"])
    # Full-frame overlay: no resize/crop or change to the corrected coordinate system.
    cv2.putText(overlay, label[:110], (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 3)
    save_image(output / "overlay.jpg", overlay)
    result["outputs"]["overlay"] = "overlay.jpg"


def run(image_path, output, config_path=Path("inference/config.json"), *, width_mm=None, height_mm=None, device_name="cuda", command=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh output directory; existing evidence is never overwritten.")
    output.mkdir(parents=True, exist_ok=True)
    if command is not None:
        (output / "command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    result = {"schema_version": 1, "status": "initializing", "width_mm": None, "height_mm": None,
              "detection_confidence": None, "confidence_definition": "Calculator detection score; not dimensional accuracy",
              "failure_reason": None, "reference": {}, "checkpoint": {}, "outputs": {},
              "input": {"path": str(image_path)}, "undistortion_applied": False,
              "coordinate_system": "undistorted_full_frame_P_equals_K", "started_at_utc": datetime.now(timezone.utc).isoformat()}
    corrected = mask = None
    try:
        config = load_configuration(config_path, width_mm, height_mm)
        result["reference"] = deepcopy(config["reference"])
        result["checkpoint"] = {"path": config["checkpoint"], "sha256": config["checkpoint_sha256"], "loaded": False}
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        result["configuration"] = {"path": str(config_path), "file_sha256": sha256(config_path),
                                   "effective_sha256": hashlib.sha256(canonical).hexdigest()}
        write_json(output / "effective_configuration.json", config)
        result["input"]["sha256"] = sha256(image_path)
        calibration_path = PROJECT / config["calibration"]
        camera = CameraCalibration.load(calibration_path)
        result["calibration"] = {"path": str(calibration_path), "sha256": sha256(calibration_path),
                                 "image_size": list(camera.image_size)}
        frame, frame_info = camera_frame(image_path, camera, config)
        result["input"].update(frame_info)
        mapper = Undistorter(camera)
        corrected = mapper.apply(frame)  # Mandatory on every normal inference path.
        result["undistortion_applied"] = True
        save_image(output / "corrected.jpg", corrected)
        result["outputs"]["corrected_image"] = "corrected.jpg"
        seed_everything(42)
        torch.set_num_threads(4)
        if device_name == "cuda" and not torch.cuda.is_available():
            raise InferenceFailure("model_error", "CUDA unavailable; choose --device cpu explicitly if required.")
        device = torch.device(device_name)
        write_json(output / "environment.json", environment())
        result["reference"] = detect_reference(corrected, config["reference"])
        model, training, checkpoint_info = load_segmenter(config, device)
        result["checkpoint"] = {**checkpoint_info, "loaded": True}
        mask, confidence, transform_info = segment_calculator(corrected, model, training, device)
        result.update(detection_confidence=confidence, model_transform=transform_info)
        if mask is None or not np.any(mask):
            raise InferenceFailure("object_not_found", "No nonempty calculator mask at frozen score >=0.5.")
        if result["reference"]["status"] != "found":
            raise InferenceFailure("reference_not_found", result["reference"]["failure_reason"])
        reference = result["reference"]
        if reference.get("physically_measured") is not True:
            raise InferenceFailure("reference_size_required", "Physically measure both outer paper edges and supply their mm sizes.")
        valid = mapper.apply(np.ones(corrected.shape[:2], dtype=np.uint8), interpolation=cv2.INTER_NEAREST)
        if np.any((mask != 0) & (valid == 0)) or np.any(mask[[0, -1], :]) or np.any(mask[:, [0, -1]]):
            raise InferenceFailure("invalid_object_geometry", "Calculator mask touches the frame or invalid undistortion border.")
        corner_array = np.array(reference["corners_corrected_px"], dtype=np.float64)
        if np.any(corner_array < 0) or np.any(corner_array[:, 0] > corrected.shape[1] - 1) \
                or np.any(corner_array[:, 1] > corrected.shape[0] - 1):
            raise InferenceFailure("invalid_reference_geometry", "Reference corners fall outside the corrected full frame.")
        footprint = np.zeros_like(valid)
        cv2.fillConvexPoly(footprint, np.rint(corner_array).astype(np.int32), 1)
        if np.any((footprint != 0) & (valid == 0)):
            raise InferenceFailure("invalid_reference_geometry", "Reference overlaps an invalid undistortion border.")
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
        geometry = measure_mask_contour_mm(contour, corner_array, reference["width_mm"], reference["height_mm"])
        write_json(output / "geometry.json", {"contours_corrected_px": [c.reshape(-1, 2).tolist() for c in contours], **geometry})
        result.update(status="success", width_mm=geometry["width_mm"], height_mm=geometry["height_mm"],
                      metric_geometry={k: v for k, v in geometry.items() if k != "contour_mm"})
        result["outputs"]["geometry"] = "geometry.json"
    except CalibrationError as exc:
        result.update(status="invalid_camera_geometry", failure_reason=str(exc))
    except MeasurementGeometryError as exc:
        result.update(status="invalid_planar_geometry", failure_reason=str(exc))
    except InferenceFailure as exc:
        result.update(status=exc.status, failure_reason=str(exc))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, cv2.error) as exc:
        result.update(status="inference_error", failure_reason=str(exc))
    if result["status"] != "success":
        result.update(width_mm=None, height_mm=None)
    if corrected is not None:
        save_overlay(output, corrected, mask, result)
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "result.json", result)
    write_json(output / "source_hashes.json", {str(p.relative_to(PROJECT)): sha256(p) for p in
               [PROJECT / "calibration/camera.py", PROJECT / "models/model.py", PROJECT / "measurement/planar.py",
                Path(__file__).resolve(), PROJECT / "inference/reference.py"]})
    print(json.dumps({key: result[key] for key in ("status", "width_mm", "height_mm", "detection_confidence", "failure_reason")}), flush=True)
    return 0 if result["status"] == "success" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="One uncropped original camera image, not a stretched export")
    parser.add_argument("--config", type=Path, default=Path("inference/config.json"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--reference-width-mm", type=float, help="Physically measured outer paper-square width in mm")
    parser.add_argument("--reference-height-mm", type=float, help="Physically measured outer paper-square height in mm")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output = args.output or Path("inference/runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if output.exists() and any(output.iterdir()):
        parser.error("Output contains evidence; choose a fresh directory.")
    command = [sys.executable, "-m", "inference.measure", *sys.argv[1:]]
    code = run(args.image, output, args.config, width_mm=args.reference_width_mm,
               height_mm=args.reference_height_mm, device_name=args.device, command=command)
    return code


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
