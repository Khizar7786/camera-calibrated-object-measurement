"""Detect the outer boundary of the user's unmarked paper square."""

from collections import Counter

import cv2
import numpy as np


def validate_reference(reference):
    if reference.get("type") != "paper_square":
        raise ValueError("Reference must be the measured unmarked paper square.")
    if not np.isclose(reference["width_mm"], reference["height_mm"], atol=1e-9):
        raise ValueError("The configured paper square must have equal measured edges.")


def order_square_corners(points):
    """Clockwise corners starting at minimum x+y; square rotations are equivalent."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    if cv2.contourArea(points, oriented=True) < 0:
        points = points[::-1]
    return np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)


def detect_reference(corrected, reference):
    validate_reference(reference)
    settings = reference["detection"]
    height, width = corrected.shape[:2]
    factor = min(1.0, settings["max_detection_dimension"] / max(width, height))
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    preview = cv2.resize(corrected, size, interpolation=cv2.INTER_AREA) if factor < 1 else corrected
    gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, settings["canny_low"], settings["canny_high"])
    edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = [(contour, "canny") for contour in edge_contours]
    generated = {"canny": len(edge_contours)}
    # Texture can join/break Canny edges. Filled bright-region contours provide
    # independent boundary proposals, subject to every existing acceptance gate.
    for threshold in settings.get("region_thresholds", []):
        if type(threshold) is not int or not 0 < threshold < 255:
            raise ValueError("Paper region thresholds must be integer intensities in (0,255).")
        _, region = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        region_contours, _ = cv2.findContours(region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        method = f"intensity_threshold_{threshold}"
        generated[method] = len(region_contours)
        contours.extend((contour, method) for contour in region_contours)
    candidates = []
    rejected = Counter()
    for contour, method in contours:
        polygon = cv2.approxPolyDP(contour, settings["polygon_epsilon_fraction"] * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            rejected["not_convex_quadrilateral"] += 1
            continue
        corners = order_square_corners(polygon)
        area = abs(cv2.contourArea(corners))
        if not settings["min_area_fraction"] * gray.size <= area <= settings["max_area_fraction"] * gray.size:
            rejected["area"] += 1
            continue
        if np.any(corners < 3) or np.any(corners[:, 0] >= size[0] - 3) or np.any(corners[:, 1] >= size[1] - 3):
            rejected["frame_boundary"] += 1
            continue
        vectors = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(vectors, axis=1)
        if min(lengths) <= 0 or max(lengths) / min(lengths) > settings["max_side_ratio"]:
            rejected["side_ratio"] += 1
            continue
        cosines = np.abs(np.sum(vectors * np.roll(vectors, 1, axis=0), axis=1) / (lengths * np.roll(lengths, 1)))
        if max(cosines) > settings["max_angle_cosine"]:
            rejected["corner_angle"] += 1
            continue
        footprint = np.zeros_like(gray)
        cv2.fillConvexPoly(footprint, np.rint(corners).astype(np.int32), 255)
        inside = cv2.erode(footprint, np.ones((5, 5), np.uint8)) != 0
        ring = (cv2.dilate(footprint, np.ones((11, 11), np.uint8)) != 0) & (footprint == 0)
        if not inside.any() or not ring.any():
            rejected["empty_inside_or_surrounding_ring"] += 1
            continue
        mean, std = float(gray[inside].mean()), float(gray[inside].std())
        contrast = mean - float(np.median(gray[ring]))
        if mean < settings["min_inside_gray"]:
            rejected["inside_brightness"] += 1
            continue
        if std > settings["max_inside_gray_std"]:
            rejected["inside_nonuniformity"] += 1
            continue
        if contrast < settings["min_inside_outside_contrast"]:
            rejected["inside_outside_contrast"] += 1
            continue
        quality = contrast / (1.0 + std)
        candidate = {"corners": corners, "area": area, "mean_gray": mean, "gray_std": std,
                     "contrast": contrast, "quality": quality, "method": method}
        duplicate = None
        for index, old in enumerate(candidates):
            intersection, _ = cv2.intersectConvexConvex(corners, old["corners"])
            if intersection / max(area, old["area"]) >= 0.9:
                duplicate = index
                break
        if duplicate is None:
            candidates.append(candidate)
        elif (quality, area) > (candidates[duplicate]["quality"], candidates[duplicate]["area"]):
            # Prefer a uniform interior with a clear outer contrast transition.
            # Largest-area alone can include a connected patch of bright table.
            candidates[duplicate] = candidate
    result = {**reference, "status": "not_found", "corners_corrected_px": None,
              "coordinate_system": "undistorted_full_frame_P_equals_K", "candidate_count": len(candidates),
              "detection_preview_size": list(size), "generated_contours": generated,
              "rejected_contour_counts": dict(rejected),
              "corner_order": "Clockwise TL, TR, BR, BL; cyclic square rotations are equivalent",
              "corner_refinement": "Full corrected-image cornerSubPix, win=5, iterations=30, epsilon=0.01",
              "failure_reason": None}
    if len(candidates) != 1:
        result["failure_reason"] = ("No isolated bright unmarked square boundary passed the configured checks"
                                    if not candidates else "Multiple plausible paper squares; reference is ambiguous")
        return result
    candidate = candidates[0]
    corners = candidate["corners"].astype(np.float64)
    corners[:, 0] = (corners[:, 0] + 0.5) * width / size[0] - 0.5
    corners[:, 1] = (corners[:, 1] + 0.5) * height / size[1] - 0.5
    if np.any(corners < 6) or np.any(corners[:, 0] >= width - 6) or np.any(corners[:, 1] >= height - 6):
        result["failure_reason"] = "Paper boundary is too close to the frame for corner refinement"
        return result
    refined = cv2.cornerSubPix(cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY), np.ascontiguousarray(corners, dtype=np.float32).reshape(4, 1, 2),
                              (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))[:, 0]
    if not cv2.isContourConvex(refined) or not np.isfinite(refined).all():
        result["failure_reason"] = "Refined paper boundary is invalid"
        return result
    result.update(status="found", corners_corrected_px=order_square_corners(refined).tolist(),
                  candidate_mean_gray=candidate["mean_gray"], candidate_gray_std=candidate["gray_std"],
                  candidate_inside_outside_contrast=candidate["contrast"],
                  candidate_generation_method=candidate["method"],
                  candidate_photometric_quality=candidate["quality"])
    return result
