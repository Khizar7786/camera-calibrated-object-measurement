"""The single image-contour to millimetre conversion used by normal inference."""

import cv2
import numpy as np


class MeasurementGeometryError(ValueError):
    """Degenerate reference, contour or planar projection."""


def measure_mask_contour_mm(contour_px, reference_corners_px, reference_width_mm, reference_height_mm):
    """Map actual corrected-image contour points to mm, THEN fit a rectangle.

    Reference corners are TL, TR, BR, BL around the outer paper boundary.
    The unmarked square has no coded orientation; cyclic rotations preserve
    its geometry because the physically measured edges are equal.
    Both arrays must already use the undistorted full-frame P=K coordinates.
    All external components of the selected predicted mask may be concatenated.
    This assumes paper and measured outline are coplanar; it cannot verify that.
    """
    try:
        width, height = float(reference_width_mm), float(reference_height_mm)
        contour = np.asarray(contour_px, dtype=np.float64).reshape(-1, 2)
        corners = np.asarray(reference_corners_px, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise MeasurementGeometryError("Reference sizes and contour/corners must be numeric.") from exc
    if isinstance(reference_width_mm, (bool, np.bool_)) or isinstance(reference_height_mm, (bool, np.bool_)) \
            or not np.isfinite([width, height]).all() or min(width, height) <= 0:
        raise MeasurementGeometryError("Physically measured reference edges must be finite and positive.")
    if corners.shape != (4, 2) or not np.isfinite(corners).all() \
            or not cv2.isContourConvex(corners.astype(np.float32)) \
            or abs(cv2.contourArea(corners.astype(np.float32))) <= 1e-6:
        raise MeasurementGeometryError("Reference must have four ordered, distinct, convex corners.")
    if len(contour) < 3 or not np.isfinite(contour).all():
        raise MeasurementGeometryError("A finite actual mask contour with at least three points is required.")
    target = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    if not np.isfinite(transform).all():
        raise MeasurementGeometryError("Reference produced a nonfinite planar transform.")
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError as exc:
        raise MeasurementGeometryError("Reference produced a singular planar transform.") from exc

    def project(points, matrix):
        homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
        denominator = homogeneous[:, 2]
        tolerance = np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(homogeneous)))) * 100
        if np.any(np.abs(denominator) <= tolerance) or (np.min(denominator) < 0 < np.max(denominator)):
            raise MeasurementGeometryError("Object/reference projection crosses the planar horizon.")
        result = homogeneous[:, :2] / denominator[:, None]
        if not np.isfinite(result).all():
            raise MeasurementGeometryError("Planar projection is nonfinite.")
        return result

    contour_mm = project(contour, transform)
    if abs(cv2.contourArea(cv2.convexHull(contour_mm.astype(np.float32)))) <= 1e-8:
        raise MeasurementGeometryError("Projected object contour is degenerate.")
    # Deliberately fit in millimetres, never fit a pixel box then scale its sides.
    rectangle = cv2.minAreaRect(contour_mm.astype(np.float32))
    sides = sorted(float(side) for side in rectangle[1])
    if not np.isfinite(sides).all() or sides[0] <= 0:
        raise MeasurementGeometryError("Projected object has no positive metric rectangle.")
    rectangle_mm = cv2.boxPoints(rectangle).astype(np.float64)
    rectangle_px = project(rectangle_mm, inverse)
    return {"width_mm": sides[0], "height_mm": sides[1],
            "homography_corrected_pixels_to_mm": transform.tolist(),
            "reference_corners_mm": target.tolist(), "contour_mm": contour_mm.tolist(),
            "rectangle_mm": rectangle_mm.tolist(), "rectangle_corrected_px": rectangle_px.tolist(),
            "conversion_function": "measurement.planar.measure_mask_contour_mm",
            "width_definition": "shorter metric rectangle side", "height_definition": "longer metric rectangle side"}
