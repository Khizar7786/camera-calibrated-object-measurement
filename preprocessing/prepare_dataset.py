"""Decode, restore, rotate and jointly undistort original images/instance masks.

Original export files are read-only. Every output annotation is derived from
the corresponding corrected mask; no source segmentation/bbox/area is copied.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import html
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np

from calibration import CalibrationError, CameraCalibration, Undistorter, read_image
from .coco_masks import AnnotationError, decode_segmentation, encode_mask, mask_geometry
from .geometry import GeometryError, GeometryTransform


SPLITS = ("train", "valid", "test")
PALETTE = ((40, 190, 255), (220, 90, 40), (110, 230, 80), (210, 70, 210), (20, 200, 200))


class PreparationError(ValueError):
    """Input provenance or geometry is incomplete; outputs are not training-ready."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    pending = path.with_name(path.name + ".pending")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def safe_name(value, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value or "/" in value or "\\" in value or value in (".", ".."):
        raise PreparationError(f"{field} must be a single safe filename.")
    return value


def raw_filename(image: dict) -> str:
    name = safe_name(image.get("file_name"), "COCO image filename")
    explicit = image.get("extra", {}).get("name")
    match = re.fullmatch(r"(IMG_\d{8}_\d{6})_jpg\.rf\.[0-9a-f]{32}\.jpg", name)
    derived = match[1] + ".jpg" if match else None
    if explicit:
        explicit = safe_name(explicit, "COCO original filename")
    if explicit and derived and explicit != derived:
        raise PreparationError(f"Conflicting original filename references for {name}.")
    if not explicit and not derived:
        raise PreparationError(f"No verifiable original filename reference for {name}.")
    return explicit or derived


def validate_inputs(root: Path) -> tuple[dict, dict, dict]:
    """Freeze checks are mandatory; a changed original export is not accepted."""
    checksum_file = root / "splits" / "SHA256SUMS.txt"
    checksums = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or file_hash(target) != digest:
            raise PreparationError(f"Original freeze checksum does not match: {relative}")
        checksums[relative] = digest
    documents, lists, capture_owners = {}, {}, {}
    for split in SPLITS:
        expected_keys = (f"splits/{split}.txt", f"{split}/_annotations.coco.json")
        if not all(key in checksums for key in expected_keys):
            raise PreparationError(f"Freeze lacks manifest/annotation checksum for {split}.")
        paths = (root / "splits" / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        if len(paths) != len(set(paths)):
            raise PreparationError(f"Duplicate frozen paths in {split}.")
        document = json.loads((root / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        images, annotations = document["images"], document["annotations"]
        names = [safe_name(image["file_name"], "COCO filename") for image in images]
        image_ids = [image["id"] for image in images]
        annotation_ids = [annotation["id"] for annotation in annotations]
        if any(type(identifier) is not int for identifier in image_ids + annotation_ids):
            raise PreparationError(f"COCO IDs must be integers in {split}.")
        if len(names) != len(set(names)) or len(image_ids) != len(set(image_ids)) or len(annotation_ids) != len(set(annotation_ids)):
            raise PreparationError(f"Duplicate COCO names or IDs in {split}.")
        if set(paths) != {f"{split}/{name}" for name in names}:
            raise PreparationError(f"Frozen list differs from original COCO membership in {split}.")
        categories = {category["id"] for category in document["categories"]}
        for annotation in annotations:
            if annotation["image_id"] not in image_ids or annotation["category_id"] not in categories:
                raise PreparationError(f"Unknown image/category reference in {split} annotation {annotation['id']}.")
            if "keypoints" in annotation:
                raise PreparationError("Keypoint coordinates are not supported by this instance-mask pipeline.")
        for image in images:
            raw = raw_filename(image)
            if raw in capture_owners:
                raise PreparationError(f"A raw capture occurs more than once/across splits: {raw}")
            capture_owners[raw] = split
            if not (root / "Raw" / raw).is_file() or not (root / split / image["file_name"]).is_file():
                raise PreparationError(f"Missing original/export file for {image['file_name']}.")
        documents[split], lists[split] = document, paths
    return documents, lists, checksums


def source_snapshot(root: Path) -> dict[str, str]:
    sources = []
    for name in (*SPLITS, "Raw", "splits"):
        sources.extend(path for path in (root / name).rglob("*") if path.is_file())
    readme = root / "README.roboflow.txt"
    if readme.is_file():
        sources.append(readme)
    return {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(sources)}


def encode_image(image: np.ndarray, extension: str, quality: int = 95) -> bytes:
    parameters = [cv2.IMWRITE_JPEG_QUALITY, quality] if extension.lower() in (".jpg", ".jpeg") else []
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise PreparationError(f"Image encoding failed for {extension}.")
    return encoded.tobytes()


def draw_instance(overlay: np.ndarray, mask: np.ndarray, box: list, identifier: int, index: int) -> None:
    color = PALETTE[index % len(PALETTE)]
    selected = mask.astype(bool)
    overlay[selected] = (0.6 * overlay[selected] + 0.4 * np.asarray(color)).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 3)
    x, y, width, height = [round(value) for value in box]
    cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), color, 2)
    cv2.putText(overlay, f"instance {identifier}", (max(0, x), max(25, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.6, overlay.shape[1] / 1600), color, 2)


def process_image(image: dict, annotations: list[dict], split: str, root: Path, output: Path,
                  settings: dict, camera: CameraCalibration, mapper: Undistorter) -> tuple[dict, list[dict], dict]:
    filename = safe_name(image["file_name"], "COCO filename")
    source_key = f"{split}/{filename}"
    original_name = raw_filename(image)
    raw_path, export_path = root / "Raw" / original_name, root / source_key
    original, exported = read_image(raw_path), read_image(export_path)
    declared = (image["width"], image["height"])
    if any(type(value) is not int or value <= 0 for value in declared) or (exported.shape[1], exported.shape[0]) != declared:
        raise PreparationError(f"COCO/export dimensions disagree for {source_key}.")
    transform = GeometryTransform.from_settings(settings, (original.shape[1], original.shape[0]), declared, source_key)
    verification = transform.verify_export(original, exported)
    if not verification["verified"]:
        raise PreparationError(f"Declared raw/export geometry did not verify for {source_key}: {verification}")
    normalized = transform.prepare_image(original)
    if (normalized.shape[1], normalized.shape[0]) != camera.image_size:
        raise PreparationError(f"Explicitly normalized size does not match calibration for {source_key}.")
    corrected = mapper.apply(normalized)
    overlay = corrected.copy()
    regenerated, instances, mask_payloads = [], [], []
    for index, annotation in enumerate(annotations):
        if type(annotation.get("iscrowd", 0)) is not int or annotation.get("iscrowd", 0) not in (0, 1):
            raise PreparationError(f"Invalid iscrowd in annotation {annotation['id']}.")
        decoded = decode_segmentation(annotation["segmentation"], declared[0], declared[1])
        if not decoded.any():
            raise PreparationError(f"Original annotation {annotation['id']} has an empty mask.")
        aligned = transform.restore_mask(decoded)
        remapped = mapper.apply(aligned, interpolation=cv2.INTER_NEAREST)
        box, area = mask_geometry(remapped)
        if area == 0:
            raise PreparationError(f"Annotation {annotation['id']} became empty after geometry correction; review required.")
        encoded = encode_mask(remapped)
        # Do not silently substitute old segmentation, box or area on any path.
        result = deepcopy(annotation)
        result.update(segmentation=encoded, bbox=box, area=area)
        regenerated.append(result)
        mask_name = f"masks/image_{image['id']}_instance_{annotation['id']}.png"
        mask_payloads.append((output / split / mask_name, encode_image(remapped * 255, ".png")))
        instances.append({"annotation_id": annotation["id"], "category_id": annotation["category_id"],
                          "mask_file": f"{split}/{mask_name}", "area": area, "bbox": box,
                          "source_area": annotation.get("area"), "source_bbox": annotation.get("bbox")})
        draw_instance(overlay, remapped, box, annotation["id"], index)
    corrected_image = deepcopy(image)
    corrected_image.update(width=camera.image_size[0], height=camera.image_size[1])
    corrected_image["measure_it_source"] = {"export_file": source_key, "raw_file": f"Raw/{original_name}",
                                            "coordinate_system": "stored_pixels_full_frame"}
    overlay_relative = f"{split}/overlays/{Path(filename).stem}.overlay.jpg"
    payloads = [(output / source_key, encode_image(corrected, Path(filename).suffix)),
                (output / overlay_relative, encode_image(overlay, ".jpg", quality=90)), *mask_payloads]
    for path, data in payloads:
        pending = path.with_name(path.name + ".pending")
        pending.write_bytes(data)
        pending.replace(path)
    record = {"split": split, "source_export": source_key, "source_raw": f"Raw/{original_name}",
              "image_id": image["id"], "output_image": source_key, "overlay": overlay_relative,
              "source_export_sha256": file_hash(export_path), "source_raw_sha256": file_hash(raw_path),
              "output_image_sha256": file_hash(output / source_key), "transform": transform.describe(),
              "geometry_verification": verification, "instances": instances,
              "instance_count": len(instances), "status": "processed", "review_status": "pending",
              "review_flags": ["multiple_instances"] if len(instances) > 1 else []}
    if not transform.coverage_is_full_frame:
        record["review_flags"].append("cropped_annotation_coverage")
    return corrected_image, regenerated, record


def write_review(output: Path, records: list[dict]) -> None:
    with (output / "manual_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "source_export", "output_image", "overlay", "instance_count", "review_status", "review_flags",
                                                   "reviewed_by", "reviewed_at_utc", "flag_resolution", "review_note"])
        writer.writeheader()
        for row in records:
            selected = {key: row.get(key, "") for key in writer.fieldnames}
            selected["review_flags"] = ";".join(row.get("review_flags", []))
            writer.writerow(selected)
    sections = []
    for split in SPLITS:
        cards = []
        for row in records:
            if row["split"] != split or row["status"] != "processed":
                continue
            overlay = html.escape(row["overlay"], quote=True)
            image = html.escape(row["output_image"], quote=True)
            title = html.escape(row["source_export"])
            flags = html.escape(", ".join(row["review_flags"]) or "none")
            review_status = html.escape(row.get("review_status", "pending"))
            resolution = html.escape(row.get("flag_resolution", "not yet reviewed"))
            reviewer = html.escape(row.get("reviewed_by", ""))
            cards.append(f'<article><a href="{overlay}"><img loading="lazy" src="{overlay}" alt="Corrected mask overlay"></a>'
                         f'<p>{title}</p><p>{row["instance_count"]} instances; review: {review_status}. Reviewer: {reviewer}</p>'
                         f'<p>Flags: {flags}; resolution: {resolution}</p>'
                         f'<a href="{image}">Corrected image</a></article>')
        sections.append(f'<h2>{split}</h2><section>{"".join(cards)}</section>')
    processed = [row for row in records if row["status"] == "processed"]
    passed = sum(row.get("review_status") == "passed" for row in processed)
    summary = f'{passed} of {len(processed)} processed images passed manual review.'
    instructions = ('User approval is recorded in review_approval.json. Existing flags and all instances are retained; '
                    'approval does not imply model training or measured millimetre accuracy.'
                    if processed and passed == len(processed) else
                    'Check outer boundaries, missing/duplicate instances, holes and clipping. '
                    'Edit review status only after inspection; a status edit alone does not repair labels.')
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Corrected dataset manual review</title>'
            '<style>body{font:16px system-ui;margin:32px;background:#f4f4f1;color:#222}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}'
            'article{background:white;padding:12px;border:1px solid #ddd}img{width:100%;height:420px;object-fit:contain;background:#222}p{overflow-wrap:anywhere}</style>'
            f'<h1>Corrected images and instance masks</h1><p>{summary}</p><p>{instructions}</p>'
            '<p>COCO boxes and areas were regenerated from the corresponding corrected masks.</p>'
            + ''.join(sections) + '</html>')
    (output / "review.html").write_text(page, encoding="utf-8")


def run(root: Path, output: Path, calibration_path: Path, geometry_path: Path) -> int:
    root, output = root.resolve(), output.resolve()
    if output == root or any(output == (root / name).resolve() or output.is_relative_to((root / name).resolve()) for name in (*SPLITS, "Raw", "splits")):
        raise PreparationError("Output must not overwrite or sit within the original export/raw/split directories.")
    if output.exists() and any(output.iterdir()):
        raise PreparationError("Output directory is not empty; use a fresh path to preserve every prior run.")
    documents, frozen_lists, source_checksums = validate_inputs(root)
    original_hashes = source_snapshot(root)
    camera = CameraCalibration.load(calibration_path)
    settings = json.loads(geometry_path.read_text(encoding="utf-8"))
    mapper = Undistorter(camera)
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for folder in (output / split, output / split / "masks", output / split / "overlays"):
            folder.mkdir(exist_ok=True)
    (output / "splits").mkdir()
    for split in SPLITS:
        shutil.copyfile(root / "splits" / f"{split}.txt", output / "splits" / f"{split}.txt")
    write_json(output / "source_snapshot.json", original_hashes)
    shutil.copyfile(calibration_path, output / "calibration.json")
    write_json(output / "geometry.json", settings)
    valid_pixels = mapper.apply(np.ones((camera.image_size[1], camera.image_size[0]), np.uint8), interpolation=cv2.INTER_NEAREST)
    (output / "valid_pixels.png").write_bytes(encode_image(valid_pixels * 255, ".png"))
    log = []

    def emit(message):
        print(message, flush=True)
        log.append(message)
        (output / "console.txt").write_text("\n".join(log) + "\n", encoding="utf-8")

    report = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "command": [sys.executable, "-m", "preprocessing.prepare_dataset", "--input", str(root), "--output", str(output),
                          "--calibration", str(calibration_path), "--geometry", str(geometry_path)],
              "software": {name: metadata.version(name) for name in ("numpy", "opencv-python", "Pillow", "pycocotools")},
              "calibration_sha256": file_hash(calibration_path), "geometry_sha256": file_hash(geometry_path),
              "source_freeze_checksums": source_checksums, "splits": {}, "images": [],
              "manual_review_complete": False, "ready_for_training": False,
              "coordinate_system": "stored_pixels_full_frame", "image_size": list(camera.image_size),
              "image_encoding": "JPEG quality 95, no resize after canonical normalization; original files unchanged",
              "mask_encoding": "binary PNG 0/255 and JSON COCO compressed RLE 0/1",
              "mask_interpolation": "nearest-neighbour for export restoration and lens remapping",
              "limitations": ["432 x 432 annotations cannot recover original subpixel boundary precision.",
                              "All original instances are retained, including potentially incorrect/duplicate labels for manual review.",
                              "Corrected boxes/areas are computed from corrected mask pixels, not source metadata.",
                              "Physical rotation mapping is user-confirmed; matching dimensions alone would not establish it.",
                              "No calibration refit, model training or mm-accuracy evaluation occurs here."]}
    emit(f"Frozen split counts: { {split: len(frozen_lists[split]) for split in SPLITS} }")
    emit("Original export is read-only; every image and decoded mask follows the same explicit geometry and undistortion.")
    pending_documents = {}
    for split in SPLITS:
        document = documents[split]
        annotations_by_image = {}
        for annotation in document["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
        corrected_images, corrected_annotations, errors = [], [], []
        for image in document["images"]:
            source_key = f"{split}/{image['file_name']}"
            try:
                new_image, new_annotations, record = process_image(image, annotations_by_image.get(image["id"], []), split, root, output, settings, camera, mapper)
                corrected_images.append(new_image)
                corrected_annotations.extend(new_annotations)
                report["images"].append(record)
                emit(f"PROCESSED {source_key}: {len(new_annotations)} regenerated instances; manual review pending")
            except (PreparationError, AnnotationError, GeometryError, CalibrationError, cv2.error, OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({"source_export": source_key, "error": str(exc)})
                report["images"].append({"split": split, "source_export": source_key, "status": "failed", "error": str(exc), "review_status": "blocked"})
                emit(f"FAILED {source_key}: {exc}")
        result = deepcopy(document)
        result.update(images=corrected_images, annotations=corrected_annotations)
        result["measure_it_preprocessing"] = {"coordinate_system": "stored_pixels_full_frame", "calibration_sha256": report["calibration_sha256"],
                                              "frozen_split": split, "manual_review_status": "pending", "all_source_instances_retained": not errors}
        annotation_name = "_annotations.coco.json" if not errors else "_annotations.partial.coco.json"
        pending_documents[split] = result
        report["splits"][split] = {"input_images": len(document["images"]), "processed_images": len(corrected_images),
                                   "input_instances": len(document["annotations"]), "processed_instances": len(corrected_annotations), "errors": errors,
                                   "annotations": f"{split}/{annotation_name}"}
    report["source_files_unchanged"] = source_snapshot(root) == original_hashes
    report["frozen_split_lists_byte_identical"] = all((root / "splits" / f"{split}.txt").read_bytes() == (output / "splits" / f"{split}.txt").read_bytes() for split in SPLITS)
    report["processed_images"] = sum(summary["processed_images"] for summary in report["splits"].values())
    report["processed_instances"] = sum(summary["processed_instances"] for summary in report["splits"].values())
    report["failed_images"] = sum(len(summary["errors"]) for summary in report["splits"].values())
    complete = report["source_files_unchanged"] and report["frozen_split_lists_byte_identical"] and report["failed_images"] == 0
    for split, document in pending_documents.items():
        summary = report["splits"][split]
        publish = report["source_files_unchanged"] and report["frozen_split_lists_byte_identical"] and not summary["errors"]
        annotation_name = "_annotations.coco.json" if publish else "_annotations.partial.coco.json"
        summary["annotations"] = f"{split}/{annotation_name}"
        write_json(output / split / annotation_name, document)
    report["status"] = "processed_pending_manual_review" if complete else "incomplete_do_not_train"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_review(output, report["images"])
    emit(f"SUMMARY: {report['processed_images']} corrected images, {report['processed_instances']} regenerated instances, {report['failed_images']} failures")
    emit(f"Original files unchanged: {report['source_files_unchanged']}; frozen split lists byte-identical: {report['frozen_split_lists_byte_identical']}")
    emit("MANUAL REVIEW PENDING: open review.html and inspect overlays; no labels are automatically approved.")
    write_json(output / "run.json", report)
    return 0 if complete else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset/corrected"))
    parser.add_argument("--calibration", type=Path, default=Path("calibration/results/camera.json"))
    parser.add_argument("--geometry", type=Path, default=Path("preprocessing/geometry.json"))
    args = parser.parse_args(argv)
    try:
        return run(args.input, args.output, args.calibration, args.geometry)
    except (PreparationError, GeometryError, CalibrationError, AnnotationError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
