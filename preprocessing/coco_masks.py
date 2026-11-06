"""COCO instance-mask decoding and lossless regenerated mask annotations.

Polygon rasterization, RLE encoding, area and bbox semantics use the official
COCO API. Corrected instances are encoded as compressed RLE, which retains
holes and disconnected components that polygon contour export could lose.
See https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools.
"""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any

import numpy as np
from pycocotools import mask as mask_api


# Original phone frames contain about 9.4 million pixels. Bound allocations
# before passing malformed or implausible dimensions into the native decoder.
MAX_MASK_PIXELS = 100_000_000


class AnnotationError(ValueError):
    """An instance segmentation cannot be interpreted in its declared image."""


def _dimensions(width: int, height: int) -> tuple[int, int]:
    values = (width, height)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or value <= 0 for value in values):
        raise AnnotationError("Mask width and height must be positive integers.")
    width, height = int(width), int(height)
    if width * height > MAX_MASK_PIXELS:
        raise AnnotationError(f"Mask exceeds the {MAX_MASK_PIXELS:,}-pixel allocation limit.")
    return width, height


def _check_runs(runs: list[int], pixels: int) -> None:
    if not runs:
        raise AnnotationError("RLE counts must not be empty.")
    if len(runs) > pixels + 1:
        raise AnnotationError("RLE has more runs than a canonical binary pixel sequence.")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
           or value < 0 or value > pixels for value in runs):
        raise AnnotationError("RLE run lengths must be nonnegative integers within the image size.")
    if sum(int(value) for value in runs) != pixels:
        raise AnnotationError("RLE run lengths do not cover the declared image exactly.")


def _compressed_runs(counts: str | bytes, pixels: int) -> tuple[bytes, list[int]]:
    """Validate COCO's signed five-bit delta stream before native decoding.

    Validation is important: the native decoder expands runs directly into its
    allocated image. Incorrect totals must never reach that decoder.
    """
    try:
        encoded = counts.encode("ascii") if isinstance(counts, str) else bytes(counts)
    except (UnicodeError, TypeError, ValueError) as exc:
        raise AnnotationError("Compressed RLE counts must be ASCII text/bytes.") from exc
    if not encoded or len(encoded) > 6 * (pixels + 1):
        raise AnnotationError("Compressed RLE is empty or exceeds the possible run-stream size.")
    runs: list[int] = []
    offset = 0
    total = 0
    while offset < len(encoded):
        value = 0
        shift = 0
        while True:
            if offset >= len(encoded):
                raise AnnotationError("Compressed RLE ends in an incomplete run.")
            code = encoded[offset] - 48
            offset += 1
            if code < 0 or code > 63:
                raise AnnotationError("Compressed RLE contains a character outside the COCO alphabet.")
            value |= (code & 31) << shift
            shift += 5
            if shift > 30:
                raise AnnotationError("Compressed RLE run exceeds native integer bounds.")
            if not (code & 32):
                if code & 16:
                    value -= 1 << shift
                break
        # COCO encodes counts from index three as differences from index i-2.
        if len(runs) > 2:
            value += runs[-2]
        if value < 0 or value > pixels:
            raise AnnotationError("Compressed RLE resolves to an invalid run length.")
        total += value
        if total > pixels:
            raise AnnotationError("Compressed RLE exceeds its declared image size.")
        runs.append(value)
        if len(runs) > pixels + 1:
            raise AnnotationError("Compressed RLE contains too many runs.")
    _check_runs(runs, pixels)
    return encoded, runs


def _rle(segmentation: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    size = segmentation.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise AnnotationError("RLE size must be [height, width].")
    declared_width, declared_height = _dimensions(size[1], size[0])
    if (declared_width, declared_height) != (width, height):
        raise AnnotationError("RLE size does not match the COCO image dimensions.")
    counts = segmentation.get("counts")
    if isinstance(counts, list):
        _check_runs(counts, width * height)
        try:
            return mask_api.frPyObjects({"size": [height, width], "counts": counts}, height, width)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnnotationError(f"Cannot encode uncompressed COCO RLE: {exc}") from exc
    if isinstance(counts, (str, bytes)):
        encoded, _ = _compressed_runs(counts, width * height)
        return {"size": [height, width], "counts": encoded}
    raise AnnotationError("RLE counts must be an integer list or compressed ASCII string.")


def decode_segmentation(segmentation: Any, width: int, height: int) -> np.ndarray:
    """Decode one COCO polygon set or RLE into an H x W uint8 0/1 mask.

    Empty polygon sets produce an empty mask; callers must explicitly decide
    whether empty instances are acceptable. Polygon pieces are unioned, while
    input and output RLE preserve holes and disconnected components exactly.
    """
    width, height = _dimensions(width, height)
    if isinstance(segmentation, dict):
        encoded = _rle(segmentation, width, height)
    elif isinstance(segmentation, list):
        if not segmentation:
            return np.zeros((height, width), dtype=np.uint8)
        polygons = []
        for polygon in segmentation:
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 6 or len(polygon) % 2:
                raise AnnotationError("Each polygon needs at least three complete x/y vertices.")
            if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                   or not np.isfinite(value) for value in polygon):
                raise AnnotationError("Polygon coordinates must be finite numbers.")
            # Prevent pathological native polygon allocations while allowing
            # modest out-of-frame coordinates, which COCO clips normally.
            limit = 4 * max(width, height) + 1024
            if any(abs(value) > limit for value in polygon):
                raise AnnotationError("Polygon coordinates are implausibly far outside the image.")
            polygons.append([float(value) for value in polygon])
        try:
            pieces = mask_api.frPyObjects(polygons, height, width)
            encoded = mask_api.merge(pieces)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnnotationError(f"Cannot rasterize COCO polygons: {exc}") from exc
    else:
        raise AnnotationError("Segmentation must be a COCO polygon list or RLE object.")
    try:
        decoded = mask_api.decode(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnnotationError(f"Cannot decode COCO segmentation: {exc}") from exc
    if decoded.shape != (height, width):
        raise AnnotationError("Decoded mask has an unexpected shape.")
    return np.ascontiguousarray(decoded, dtype=np.uint8)


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise AnnotationError("An instance mask must be a two-dimensional NumPy array.")
    _dimensions(mask.shape[1], mask.shape[0])
    if not (np.issubdtype(mask.dtype, np.integer) or mask.dtype == np.bool_):
        raise AnnotationError("An instance mask must use integer or Boolean values.")
    if not np.all((mask == 0) | (mask == 1)):
        raise AnnotationError("An instance mask must contain only binary 0/1 values.")
    return np.asfortranarray(mask, dtype=np.uint8)


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    """Return JSON-ready compressed COCO RLE without changing mask topology."""
    encoded = mask_api.encode(_binary_mask(mask))
    return {"size": [int(value) for value in encoded["size"]],
            "counts": encoded["counts"].decode("ascii")}


def mask_geometry(mask: np.ndarray) -> tuple[list[float], int]:
    """Calculate fresh COCO [x, y, width, height] and pixel area from a mask."""
    encoded = mask_api.encode(_binary_mask(mask))
    return mask_api.toBbox(encoded).astype(float).tolist(), int(mask_api.area(encoded))
