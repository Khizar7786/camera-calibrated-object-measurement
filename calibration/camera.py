"""Strict full-frame camera geometry shared by preprocessing and inference.

Images use their stored pixel grid: origin at the upper left, x to the right,
y down. No EXIF rotation, resizing, cropping or intrinsic rescaling is implicit.
Undistortion retains both the calibrated image size and camera matrix (P = K).
Pixels outside the source image receive the caller's constant border value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


class CalibrationError(ValueError):
    """Invalid calibration, unreadable input, or incompatible image geometry."""


@dataclass(frozen=True)
class CameraCalibration:
    """Validated OpenCV pinhole intrinsics for one stored-pixel resolution."""

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: tuple[int, int]
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        try:
            matrix = np.array(self.camera_matrix, dtype=np.float64, copy=True)
            distortion = np.array(
                self.distortion_coefficients, dtype=np.float64, copy=True
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationError("Camera parameters must be numeric arrays.") from exc
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise CalibrationError("camera_matrix must be a finite 3 x 3 array.")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise CalibrationError("Camera focal lengths must be positive.")
        if not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0, atol=1e-12):
            raise CalibrationError("The camera matrix bottom row must be [0, 0, 1].")
        if distortion.ndim == 2 and 1 in distortion.shape:
            distortion = distortion.reshape(-1)
        if (
            distortion.ndim != 1
            or distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()
        ):
            raise CalibrationError(
                "distortion_coefficients must contain 4, 5, 8, 12 or 14 finite values."
            )
        try:
            size = tuple(self.image_size)
        except TypeError as exc:
            raise CalibrationError("image_size must be [width, height].") from exc
        if len(size) != 2 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in size
        ):
            raise CalibrationError("image_size must contain two positive integers.")
        matrix.setflags(write=False)
        distortion.setflags(write=False)
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", distortion)
        object.__setattr__(self, "image_size", (int(size[0]), int(size[1])))

    @classmethod
    def load(cls, path: str | Path) -> CameraCalibration:
        """Load schema v1, rejecting unsupported models and coordinate systems."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CalibrationError(f"Cannot read calibration file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise CalibrationError("Calibration JSON must be an object.")
        if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
            raise CalibrationError("Unsupported calibration schema_version; expected 1.")
        if data.get("model") != "opencv_pinhole":
            raise CalibrationError("Unsupported calibration model; expected opencv_pinhole.")
        if data.get("coordinate_system") != "stored_pixels_full_frame":
            raise CalibrationError(
                "Unsupported coordinate_system; expected stored_pixels_full_frame."
            )
        required = ("camera_matrix", "distortion_coefficients", "image_size")
        missing = [name for name in required if name not in data]
        if missing:
            raise CalibrationError("Missing calibration fields: " + ", ".join(missing))
        return cls(
            camera_matrix=data["camera_matrix"],
            distortion_coefficients=data["distortion_coefficients"],
            image_size=data["image_size"],
            metadata=data,
        )


def read_image(path: str | Path) -> np.ndarray:
    """Read BGR pixels without EXIF auto-rotation; reject non-normal orientation.

    The resolution guard in :func:`undistort` then checks the stored dimensions.
    A missing EXIF orientation is treated as already encoded normally. Neither
    metadata nor dimensions can prove that focus/zoom or a same-size rotation
    remained unchanged; capture consistency must also be verified by the user.
    """
    path = Path(path)
    try:
        with Image.open(path) as source:
            orientation = source.getexif().get(274, 1)
            if orientation != 1:
                raise CalibrationError(
                    f"{path}: EXIF orientation {orientation} is unsupported. "
                    "Use a verified stored-pixel coordinate system; no automatic rotation is applied."
                )
        encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    except CalibrationError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, cv2.error) as exc:
        raise CalibrationError(f"Cannot read image {path}: {exc}") from exc
    if image is None or image.size == 0:
        raise CalibrationError(f"Cannot decode image {path}.")
    return image


class Undistorter:
    """Cache one mapping for repeated images and corresponding categorical masks."""

    def __init__(self, calibration: CameraCalibration) -> None:
        if not isinstance(calibration, CameraCalibration):
            raise CalibrationError("Expected a validated CameraCalibration instance.")
        self.calibration = calibration
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            None,
            calibration.camera_matrix,
            calibration.image_size,
            cv2.CV_32FC1,
        )

    def apply(
        self,
        image: np.ndarray,
        *,
        interpolation: int = cv2.INTER_LINEAR,
        border_value: int | float | tuple = 0,
    ) -> np.ndarray:
        """Keep the full frame; pass INTER_NEAREST for categorical mask labels."""
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise CalibrationError("Input must be a two- or three-dimensional NumPy image.")
        actual_size = (image.shape[1], image.shape[0])
        if actual_size != self.calibration.image_size:
            raise CalibrationError(
                f"Image resolution/orientation mismatch: got {actual_size[0]} x {actual_size[1]}, "
                f"calibrated for {self.calibration.image_size[0]} x {self.calibration.image_size[1]}. "
                "No resizing, rotation or intrinsic rescaling is performed."
            )
        if image.ndim == 3 and image.shape[2] == 0:
            raise CalibrationError("Input image must have at least one channel.")
        if image.dtype not in (
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("int16"),
            np.dtype("float32"),
            np.dtype("float64"),
        ):
            raise CalibrationError(
                "Unsupported image dtype; categorical masks should use uint8 or uint16."
            )
        if interpolation not in (cv2.INTER_LINEAR, cv2.INTER_NEAREST):
            raise CalibrationError("Use INTER_LINEAR for images or INTER_NEAREST for categorical masks.")
        try:
            return cv2.remap(
                image,
                self._map_x,
                self._map_y,
                interpolation=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=border_value,
            )
        except cv2.error as exc:
            raise CalibrationError(f"Undistortion failed: {exc}") from exc

    __call__ = apply


def undistort(
    image: np.ndarray,
    calibration: CameraCalibration,
    *,
    interpolation: int = cv2.INTER_LINEAR,
    border_value: int | float | tuple = 0,
) -> np.ndarray:
    """Undistort one full frame using the same geometry for images and masks.

    Reuse :class:`Undistorter` to avoid rebuilding maps in a preprocessing or
    inference loop. This function never chooses a new camera matrix or crops to
    an optimal ROI. Use INTER_NEAREST for categorical masks and retain any
    invalid-border handling explicitly in the calling stage.
    """
    return Undistorter(calibration).apply(
        image, interpolation=interpolation, border_value=border_value
    )
