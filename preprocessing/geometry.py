"""Explicit export-coordinate recovery and camera-frame normalization.

The declared export operation is crop, clockwise quarter-turn, then a stretch
resize. Restoring categorical masks reverses those operations before applying
the separately declared camera-frame rotation. No camera intrinsics are inferred
or changed here; the shared undistorter validates the resulting camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image


class GeometryError(ValueError):
    """Missing, invalid, or incompatible source/export geometry."""


def _integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _size(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise GeometryError(f"{name} must be [width, height].")
    if any(not _integer(v) or v <= 0 for v in value):
        raise GeometryError(f"{name} must contain two positive integers.")
    return int(value[0]), int(value[1])


def _quarter_turn(value: Any, name: str) -> int:
    if not _integer(value) or value not in (0, 90, 180, 270):
        raise GeometryError(f"{name} must be an explicit clockwise 0, 90, 180 or 270.")
    return int(value)


def _source(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeometryError(f"{name} must explain the declared geometry's source.")
    return value


def _rotate(array: np.ndarray, clockwise_degrees: int) -> np.ndarray:
    codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if clockwise_degrees == 0:
        return np.ascontiguousarray(array).copy()
    return cv2.rotate(array, codes[clockwise_degrees])


def _image(array: Any, expected: tuple[int, int], name: str) -> np.ndarray:
    if (
        not isinstance(array, np.ndarray)
        or array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[2] != 3
    ):
        raise GeometryError(f"{name} must be a uint8 BGR image with three channels.")
    actual = (array.shape[1], array.shape[0])
    if actual != expected:
        raise GeometryError(f"{name} size {actual} differs from declared {expected}.")
    return array


@dataclass(frozen=True)
class GeometryTransform:
    """One validated source-to-export mapping and explicit camera rotation."""

    raw_size: tuple[int, int]
    export_size: tuple[int, int]
    crop_xywh: tuple[int, int, int, int]
    export_rotation_cw_degrees: int
    camera_rotation_cw_degrees: int
    export_source: str
    camera_rotation_source: str
    source_key: str
    override_used: bool
    max_rgb_rmse: float
    min_rgb_correlation: float

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        raw_size: tuple[int, int],
        export_size: tuple[int, int],
        source_key: str,
    ) -> GeometryTransform:
        if not isinstance(settings, Mapping):
            raise GeometryError("Geometry settings must be an object.")
        if type(settings.get("schema_version")) is not int or settings["schema_version"] != 1:
            raise GeometryError("Geometry schema_version must be 1.")
        raw_size = _size(raw_size, "raw_size")
        export_size = _size(export_size, "export_size")
        if not isinstance(source_key, str) or not source_key:
            raise GeometryError("source_key must be a nonempty source identity.")
        export = settings.get("export")
        overrides = settings.get("overrides", {})
        if not isinstance(export, Mapping) or not isinstance(overrides, Mapping):
            raise GeometryError("export and overrides must be objects.")
        export = dict(export)
        override_used = source_key in overrides
        if override_used:
            override = overrides[source_key]
            if not isinstance(override, Mapping) or set(override) - {"export"}:
                raise GeometryError("A source override may contain only an export object.")
            if not isinstance(override.get("export"), Mapping):
                raise GeometryError("A source override must contain an export object.")
            export.update(override["export"])
        allowed = {"size", "resize", "crop_xywh", "rotation_cw_degrees", "source"}
        if set(export) - allowed:
            raise GeometryError("Unsupported export fields: " + ", ".join(sorted(set(export) - allowed)))
        configured_size = _size(export.get("size"), "export.size")
        if configured_size != export_size:
            raise GeometryError(
                f"Configured export size {configured_size} differs from actual/COCO {export_size}."
            )
        if export.get("resize") != "stretch":
            raise GeometryError("Only explicitly declared stretch resizing is supported.")
        export_rotation = _quarter_turn(export.get("rotation_cw_degrees"), "export.rotation_cw_degrees")
        export_source = _source(export.get("source"), "export.source")
        crop = export.get("crop_xywh")
        if crop is None:
            crop = (0, 0, *raw_size)
        if (
            not isinstance(crop, (list, tuple))
            or len(crop) != 4
            or any(not _integer(v) for v in crop)
        ):
            raise GeometryError("crop_xywh must be null or four integer coordinates.")
        x, y, width, height = (int(v) for v in crop)
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > raw_size[0] or y + height > raw_size[1]:
            raise GeometryError("crop_xywh must be positive and lie within the raw image.")
        rotations = settings.get("camera_frame_rotations")
        if not isinstance(rotations, Mapping):
            raise GeometryError("camera_frame_rotations must explicitly cover the raw size.")
        size_key = f"{raw_size[0]}x{raw_size[1]}"
        rule = rotations.get(size_key)
        if not isinstance(rule, Mapping):
            raise GeometryError(f"No explicit camera-frame rotation rule for {size_key}.")
        camera_rotation = _quarter_turn(rule.get("rotation_cw_degrees"), "camera rotation_cw_degrees")
        camera_source = _source(rule.get("source"), "camera rotation source")
        verification = settings.get("verification", {})
        if not isinstance(verification, Mapping):
            raise GeometryError("verification must be an object.")
        limits = []
        for name, default, upper in (("max_rgb_rmse", 12.0, 255.0), ("min_rgb_correlation", 0.97, 1.0)):
            value = verification.get(name, default)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
                raise GeometryError(f"verification.{name} must be numeric.")
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise GeometryError(f"verification.{name} must be a finite real number.") from exc
            if not math.isfinite(value) or not 0 <= value <= upper:
                raise GeometryError(f"verification.{name} must be finite and between 0 and {upper}.")
            limits.append(value)
        return cls(
            raw_size, export_size, (x, y, width, height), export_rotation,
            camera_rotation, export_source, camera_source, source_key,
            override_used, limits[0], limits[1],
        )

    @property
    def canonical_size(self) -> tuple[int, int]:
        if self.camera_rotation_cw_degrees in (90, 270):
            return self.raw_size[1], self.raw_size[0]
        return self.raw_size

    @property
    def coverage_is_full_frame(self) -> bool:
        return self.crop_xywh == (0, 0, *self.raw_size)

    def describe(self) -> dict[str, Any]:
        x, y, width, height = self.crop_xywh
        return {
            "source_key": self.source_key,
            "raw_size": list(self.raw_size),
            "export_size": list(self.export_size),
            "canonical_size": list(self.canonical_size),
            "export": {
                "operation_order": ["crop", "clockwise_quarter_turn", "stretch_resize"],
                "crop_xywh": [x, y, width, height],
                "rotation_cw_degrees": self.export_rotation_cw_degrees,
                "resize": "stretch",
                "source": self.export_source,
                "override_used": self.override_used,
            },
            "camera_frame": {
                "rotation_cw_degrees": self.camera_rotation_cw_degrees,
                "source": self.camera_rotation_source,
                "intrinsics_modified": False,
            },
            "annotation_coverage": {
                "full_raw_frame": self.coverage_is_full_frame,
                "raw_fraction": width * height / (self.raw_size[0] * self.raw_size[1]),
                "outside_crop": "none" if self.coverage_is_full_frame else "unlabelled; zero fill is not verified background",
                "manual_review_required": True,
            },
            "mask_reconstruction": "INTER_NEAREST_EXACT; exported raster quantization is not recoverable",
        }

    def prepare_image(self, raw_bgr: np.ndarray) -> np.ndarray:
        """Normalize the full raw image using only the declared camera rotation."""
        raw_bgr = _image(raw_bgr, self.raw_size, "raw image")
        return _rotate(raw_bgr, self.camera_rotation_cw_degrees)

    def restore_mask(self, export_binary: np.ndarray) -> np.ndarray:
        """Restore an export mask to the canonical frame without inventing labels."""
        if not isinstance(export_binary, np.ndarray) or export_binary.ndim != 2:
            raise GeometryError("Export mask must be a two-dimensional binary array.")
        if export_binary.dtype.kind not in "bui" or not np.isin(export_binary, [0, 1]).all():
            raise GeometryError("Export mask must contain binary integer/bool values 0 and 1.")
        if (export_binary.shape[1], export_binary.shape[0]) != self.export_size:
            raise GeometryError("Export mask size differs from the declared export size.")
        x, y, width, height = self.crop_xywh
        rotated_size = (height, width) if self.export_rotation_cw_degrees in (90, 270) else (width, height)
        resized = cv2.resize(export_binary.astype(np.uint8), rotated_size, interpolation=cv2.INTER_NEAREST_EXACT)
        crop_mask = _rotate(resized, (-self.export_rotation_cw_degrees) % 360)
        raw_mask = np.zeros((self.raw_size[1], self.raw_size[0]), dtype=np.uint8)
        raw_mask[y:y + height, x:x + width] = crop_mask
        return _rotate(raw_mask, self.camera_rotation_cw_degrees)

    def verify_export(self, raw_bgr: np.ndarray, exported_bgr: np.ndarray) -> dict[str, Any]:
        """Check declared coordinate provenance; never search for another geometry."""
        raw_bgr = _image(raw_bgr, self.raw_size, "raw image")
        exported_bgr = _image(exported_bgr, self.export_size, "export image")
        x, y, width, height = self.crop_xywh
        crop = raw_bgr[y:y + height, x:x + width]
        rotated = _rotate(crop, self.export_rotation_cw_degrees)
        renderers = {
            "opencv_area": cv2.resize(rotated, self.export_size, interpolation=cv2.INTER_AREA),
            "pillow_bilinear": np.asarray(
                Image.fromarray(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)).resize(self.export_size, Image.Resampling.BILINEAR)
            )[:, :, ::-1],
        }
        target = exported_bgr.astype(np.float64).reshape(-1)
        target_centered = target - target.mean()
        target_norm = float(np.linalg.norm(target_centered))
        candidates = []
        for renderer, candidate in renderers.items():
            pixels = candidate.astype(np.float64).reshape(-1)
            rmse = float(np.sqrt(np.mean((pixels - target) ** 2)))
            centered = pixels - pixels.mean()
            denominator = float(np.linalg.norm(centered)) * target_norm
            correlation = None if denominator <= 1e-12 else float(np.clip(np.dot(centered, target_centered) / denominator, -1.0, 1.0))
            passed = correlation is not None and rmse <= self.max_rgb_rmse and correlation >= self.min_rgb_correlation
            candidates.append({
                "renderer": renderer, "rgb_rmse": rmse,
                "rgb_correlation": correlation, "verified": bool(passed),
            })
        candidates.sort(key=lambda c: (not c["verified"], c["rgb_rmse"]))
        best = candidates[0]
        verified = best["verified"]
        if verified:
            reason = "Declared crop/rotation/stretch matches the exported image within fixed thresholds."
        elif best["rgb_correlation"] is None:
            reason = "Insufficient pixel variation to verify coordinate provenance."
        else:
            reason = "Declared export geometry fails the fixed RMSE/correlation thresholds; no alternative geometry was inferred."
        return {
            "verified": verified,
            "best_renderer": best["renderer"],
            "rgb_rmse": best["rgb_rmse"],
            "rgb_correlation": best["rgb_correlation"],
            "reason": reason,
            "thresholds": {"max_rgb_rmse": self.max_rgb_rmse, "min_rgb_correlation": self.min_rgb_correlation},
            "candidates": candidates,
        }
