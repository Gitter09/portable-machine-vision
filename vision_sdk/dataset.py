"""Import one COCO ZIP and build the exact images the board model will see."""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from .config import VisionConfig
from .preprocessing import Box, preprocessing_settings


class DatasetError(ValueError):
    pass


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and "\\" not in name


def inspect_coco_zip(archive: Path) -> dict:
    """Validate data before extracting any files; return a compact import report."""
    try:
        with zipfile.ZipFile(archive) as zipped:
            members = {member.filename: member for member in zipped.infolist() if not member.is_dir()}
            if len(members) != len([member for member in zipped.infolist() if not member.is_dir()]):
                raise DatasetError("ZIP contains duplicate filenames")
            if any(not _safe_member(name) for name in members):
                raise DatasetError("ZIP contains an unsafe file path")
            if "annotations/instances.json" not in members:
                raise DatasetError("Add annotations/instances.json to the ZIP")
            if len(members) > 20000 or sum(item.file_size for item in members.values()) > 20 * 1024**3:
                raise DatasetError("Dataset exceeds the first-release limit of 20 GB or 20,000 files")
            data = json.loads(zipped.read("annotations/instances.json"))
            if not isinstance(data, dict):
                raise DatasetError("COCO JSON needs a top-level object")
            images = data.get("images")
            categories = data.get("categories")
            annotations = data.get("annotations")
            if not isinstance(images, list) or not isinstance(categories, list) or not isinstance(annotations, list):
                raise DatasetError("COCO JSON needs images, categories, and annotations lists")
            if len(images) < 2 or not categories:
                raise DatasetError("Add at least two annotated images and one object class")
            class_ids = [item.get("id") for item in categories]
            class_names = [item.get("name") for item in categories]
            if any(not isinstance(value, int) for value in class_ids) or len(set(class_ids)) != len(class_ids) or any(not isinstance(name, str) or not name.strip() for name in class_names):
                raise DatasetError("Every class needs a unique ID and a name")
            if len(set(class_names)) != len(class_names):
                raise DatasetError("Class names must be unique")
            by_id = {}
            for item in images:
                image_id, filename = item.get("id"), item.get("file_name")
                if not isinstance(image_id, int) or image_id in by_id or not isinstance(filename, str) or not _safe_member(filename):
                    raise DatasetError("Each image needs a unique ID and a safe filename")
                member_name = filename if filename.startswith("images/") else f"images/{filename}"
                if Path(member_name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    raise DatasetError(f"Image type is not supported: {member_name}")
                if member_name not in members:
                    raise DatasetError(f"Image is missing: {member_name}")
                blob = zipped.read(member_name)
                decoded = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    raise DatasetError(f"Cannot open image: {member_name}")
                height, width = decoded.shape[:2]
                if width * height > 100_000_000:
                    raise DatasetError(f"Image is larger than 100 megapixels: {member_name}")
                if width != item.get("width") or height != item.get("height"):
                    raise DatasetError(f"Image dimensions disagree with COCO JSON: {member_name}")
                by_id[image_id] = {"filename": member_name, "width": width, "height": height,
                                   "sha256": hashlib.sha256(blob).hexdigest()}
            labels = defaultdict(list)
            seen_annotations = set()
            for item in annotations:
                annotation_id = item.get("id")
                image_id = item.get("image_id")
                category_id = item.get("category_id")
                box = item.get("bbox")
                if not isinstance(annotation_id, int) or not isinstance(image_id, int):
                    raise DatasetError("Every annotation needs integer IDs")
                if annotation_id in seen_annotations or image_id not in by_id or category_id not in class_ids:
                    raise DatasetError("An annotation has an unknown image/class or repeated ID")
                seen_annotations.add(annotation_id)
                if not isinstance(box, list) or len(box) != 4:
                    raise DatasetError("Each annotation needs a [x, y, width, height] box")
                try:
                    x, y, width, height = map(float, box)
                except (TypeError, ValueError) as error:
                    raise DatasetError("Box coordinates must be numbers") from error
                image = by_id[image_id]
                if not all(np.isfinite(v) for v in (x, y, width, height)) or x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image["width"] + 1 or y + height > image["height"] + 1:
                    raise DatasetError(f"Box is outside image {image['filename']}")
                labels[image_id].append({"class_id": class_ids.index(category_id), "bbox": [x, y, width, height]})
            if not labels:
                raise DatasetError("No object boxes found; raw images alone cannot start training")
            return {"images": by_id, "labels": dict(labels), "classes": class_names,
                    "boxes": len(annotations), "negative_images": len(images) - len(labels)}
    except zipfile.BadZipFile as error:
        raise DatasetError("The selected file is not a valid ZIP") from error


def import_coco_zip(archive: Path, destination: Path) -> dict:
    report = inspect_coco_zip(archive)
    if destination.exists():
        raise FileExistsError(f"Dataset already exists: {destination}")
    destination.mkdir(parents=True)
    (destination / "images").mkdir()
    exported = {}
    with zipfile.ZipFile(archive) as zipped:
        for image_id, image in report["images"].items():
            suffix = Path(image["filename"]).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
                raise DatasetError(f"Unsupported image type: {suffix}")
            name = f"image_{image_id}{suffix}"
            (destination / "images" / name).write_bytes(zipped.read(image["filename"]))
            exported[str(image_id)] = {**image, "stored_name": name}
    clean = {key: value for key, value in report.items() if key != "images"}
    clean["images"] = exported
    (destination / "manifest.json").write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")
    return clean


def _map_box_agnostic(box: list[float], preprocessor, metadata: dict) -> tuple[float, float, float, float] | None:
    """Map a source-frame label into model space with the forward transform.

    `forward_map` is the exact inverse of the `inverse_map` used on detections,
    kept beside it in the preprocessor so the two cannot drift apart. A label
    that loses more than a sixth of itself to the model-input edge is dropped:
    a sliver teaches a truncated shape, and an unlabelled object teaches
    background.
    """
    x, y, width, height = box
    mapped = preprocessor.forward_map(Box(x, y, width, height), metadata, clip=False)
    area = max(0.0, mapped.width) * max(0.0, mapped.height)
    x1, y1 = max(0.0, mapped.x), max(0.0, mapped.y)
    x2, y2 = min(640.0, mapped.x2), min(640.0, mapped.y2)
    clipped = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0 or clipped / area < 0.85:
        return None
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return ((x1 + x2) / 1280, (y1 + y2) / 1280, (x2 - x1) / 640, (y2 - y1) / 640)


def _map_box_fixed_reference(box: list[float], metadata: dict) -> tuple[float, float, float, float] | None:
    if metadata.get("affine_M") is None:
        return None
    x, y, width, height = box
    points = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float32).reshape(-1, 1, 2)
    matrix = np.asarray(metadata["affine_M"], dtype=np.float32)
    points = cv2.transform(points, matrix).reshape(-1, 2)
    roi = metadata["roi"]
    points[:, 0] = (points[:, 0] - roi["x"]) * metadata["scale"] + metadata["pad_left"]
    points[:, 1] = (points[:, 1] - roi["y"]) * metadata["scale"] + metadata["pad_top"]
    x1, y1 = np.maximum(points.min(axis=0), 0)
    x2, y2 = np.minimum(points.max(axis=0), 640)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return ((x1 + x2) / 1280, (y1 + y2) / 1280, (x2 - x1) / 640, (y2 - y1) / 640)


def build_dataset_preprocessor(config: VisionConfig):
    """The preprocessor training will use, built from the same config as serving."""
    from .preprocessing import resolve_preprocessor

    return resolve_preprocessor(
        config.preprocessor,
        reference_image=config.reference_image or None,
        roi=config.roi,
        profile_path=config.preprocessing_profile or None,
        flatfield_lut=config.flatfield_lut or None,
    )


_CAPTURE_STAMP = re.compile(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})")
BURST_SECONDS = 30.0


def _captured_seconds(filename: str) -> float | None:
    match = _CAPTURE_STAMP.search(filename)
    if not match:
        return None
    try:
        return datetime(*(int(value) for value in match.groups()), tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _leak_free_groups(images: dict) -> list[list[str]]:
    """Group frames that must stay in one split.

    Exact duplicates share bytes, and frames captured within one burst share a
    scene; a split that separates either measures memorisation instead of
    accuracy. The capture timestamp comes from the camera filename
    (img_YYYYMMDD_HHMMSS_seq); frames without one fall back to their hash.
    """
    ids = list(images)
    parent = {image_id: image_id for image_id in ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        parent[find(left)] = find(right)

    by_hash: dict[str, list[str]] = defaultdict(list)
    for image_id in ids:
        by_hash[images[image_id]["sha256"]].append(image_id)
    for items in by_hash.values():
        for other in items[1:]:
            union(items[0], other)

    timed = [(seconds, image_id) for image_id in ids
             if (seconds := _captured_seconds(str(images[image_id].get("filename", "")))) is not None]
    timed.sort()
    for (first, left), (second, right) in zip(timed, timed[1:]):
        if second - first <= BURST_SECONDS:
            union(left, right)

    groups: dict[str, list[str]] = defaultdict(list)
    for image_id in ids:
        groups[find(image_id)].append(image_id)
    return list(groups.values())


def prepare_training_dataset(imported: Path, output: Path, config: VisionConfig) -> dict:
    """Condition through the board preprocessor, then write YOLO training files.

    Training images are produced by exactly the code path that will run at
    inference. That parity is the single most valuable property here, so this
    function never takes a shortcut around the preprocessor.

    The product-agnostic path emits one training image per located part, because
    serving conditions each part on its own and a scene view shrinks a cap hole
    below what the model can resolve. The fixed-reference path keeps its single
    calibrated window.
    """
    from .preprocessing import PartNotFoundError

    agnostic = config.preprocessor.strip().lower() == "product_agnostic"
    if not agnostic and (config.roi is None or not config.reference_image):
        raise DatasetError("Set a reference image and inspection area before training")
    if agnostic and not config.preprocessing_profile and config.roi is None:
        raise DatasetError(
            "The product-agnostic preprocessor locates the part itself, but a "
            "fallback window is still required; set an inspection area as fallback_roi"
        )

    manifest = json.loads((imported / "manifest.json").read_text(encoding="utf-8"))
    preprocessor = build_dataset_preprocessor(config)
    if output.exists():
        raise FileExistsError(f"Training output already exists: {output}")
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    groups = _leak_free_groups(manifest["images"])
    random.Random(42).shuffle(groups)
    val_count = max(1, round(len(groups) * 0.2))
    val_ids = {image_id for group in groups[:val_count] for image_id in group}
    ids = list(manifest["images"])
    rejected = []
    class_counts = {"train": [0] * len(manifest["classes"]), "val": [0] * len(manifest["classes"])}

    def emit(image_id: str, split: str, suffix: str, model_image, lines: list[str]) -> None:
        stem = f"image_{image_id}{suffix}"
        cv2.imwrite(str(output / "images" / split / f"{stem}.png"), model_image)
        (output / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    for image_id in ids:
        record = manifest["images"][image_id]
        source = imported / "images" / record["stored_name"]
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        split = "val" if image_id in val_ids else "train"
        labels = manifest["labels"].get(image_id, manifest["labels"].get(str(image_id), []))

        if agnostic:
            try:
                instances = preprocessor.preprocess_instances(frame)
            except PartNotFoundError as error:
                rejected.append({"image": record["filename"], "reason": str(error)})
                continue
            except Exception as error:
                rejected.append({"image": record["filename"], "reason": f"{type(error).__name__}: {error}"})
                continue
            if not instances:
                rejected.append({"image": record["filename"], "reason": "no part was located"})
                continue
            assigned: dict[int, list[dict]] = defaultdict(list)
            for label in labels:
                box_x, box_y, box_w, box_h = label["bbox"]
                centre_x, centre_y = box_x + box_w / 2, box_y + box_h / 2
                for index, instance in enumerate(instances):
                    box = instance["box"]
                    if box["x"] <= centre_x <= box["x"] + box["width"] and box["y"] <= centre_y <= box["y"] + box["height"]:
                        assigned[index].append(label)
                        break
                else:
                    rejected.append({"image": record["filename"], "reason": "a label was outside every located part"})
            for index, instance in enumerate(instances):
                accepted = assigned.get(index, [])
                if not accepted:
                    continue
                lines = []
                for label in accepted:
                    mapped = _map_box_agnostic(label["bbox"], preprocessor, instance["metadata"])
                    if mapped is None:
                        continue
                    class_counts[split][label["class_id"]] += 1
                    lines.append(f"{label['class_id']} " + " ".join(f"{value:.8f}" for value in mapped))
                if lines:
                    emit(image_id, split, f"_i{index:02d}", instance["model_bgr"], lines)
            continue

        try:
            _, model_image, metadata = preprocessor.preprocess(frame)
        except PartNotFoundError as error:
            rejected.append({"image": record["filename"], "reason": str(error)})
            continue
        except Exception as error:
            rejected.append({"image": record["filename"], "reason": f"{type(error).__name__}: {error}"})
            continue
        if metadata.get("affine_M") is None:
            rejected.append({"image": record["filename"], "reason": "the frame could not be aligned to the reference"})
            continue
        lines = []
        for label in labels:
            mapped = _map_box_fixed_reference(label["bbox"], metadata)
            if mapped is None:
                continue
            class_counts[split][label["class_id"]] += 1
            lines.append(f"{label['class_id']} " + " ".join(f"{value:.8f}" for value in mapped))
        emit(image_id, split, "", model_image, lines)
    if any(count == 0 for count in class_counts["train"]) or any(count == 0 for count in class_counts["val"]):
        shutil.rmtree(output)
        raise DatasetError("Every class needs examples in both training and validation images after alignment; add more varied images")
    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text("path: " + output.resolve().as_posix() + "\ntrain: images/train\nval: images/val\nnames:\n" +
                            "".join(f"  {index}: {json.dumps(name)}\n" for index, name in enumerate(manifest["classes"])), encoding="utf-8")
    report = {"classes": manifest["classes"], "counts": class_counts, "rejected": rejected,
              "preprocessing": preprocessing_settings(
                  config.preprocessor, config.preprocessing_profile or None, config.flatfield_lut or None
              ),
              "dataset_yaml": str(dataset_yaml)}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
