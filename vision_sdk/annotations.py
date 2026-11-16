"""Small, local annotation workspace that exports the SDK's COCO contract."""

from __future__ import annotations

import json
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .config import VisionStore


class AnnotationError(ValueError):
    pass


class AnnotationWorkspace:
    def __init__(self, store: VisionStore):
        self.root = store.root / "annotations"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, project_id: str) -> Path:
        uuid.UUID(project_id)
        path = self.root / project_id
        if not path.is_dir():
            raise AnnotationError("Image set not found")
        return path

    def create(self, class_names: list[str]) -> dict:
        if isinstance(class_names, str) or not isinstance(class_names, (list, tuple)):
            raise AnnotationError("Enter the object names as a list")
        entries = list(class_names)
        if any(not isinstance(name, str) or not name.strip() for name in entries):
            raise AnnotationError("Every object name must be a non-empty string")
        names = [name.strip() for name in entries]
        if not names or any(not name for name in names) or len(set(names)) != len(names):
            raise AnnotationError("Enter distinct names for the objects you want to find")
        project_id = str(uuid.uuid4())
        path = self.root / project_id
        (path / "images").mkdir(parents=True)
        manifest = {"project_id": project_id, "classes": names, "images": []}
        self._save(path, manifest)
        return manifest

    @staticmethod
    def _save(path: Path, manifest: dict) -> None:
        target = path / "manifest.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)

    def get(self, project_id: str) -> dict:
        return json.loads((self._path(project_id) / "manifest.json").read_text(encoding="utf-8"))

    def add_image(self, project_id: str, blob: bytes) -> dict:
        if not blob or len(blob) > 40 * 1024 * 1024:
            raise AnnotationError("Choose a camera image smaller than 40 MB")
        image = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (2256, 3200):
            raise AnnotationError("Each image must come from the 3200 × 2256 camera profile")
        path = self._path(project_id)
        manifest = self.get(project_id)
        image_id = len(manifest["images"]) + 1
        filename = f"image_{image_id}.png"
        if not cv2.imwrite(str(path / "images" / filename), image):
            raise AnnotationError("Could not save the camera image")
        record = {"id": image_id, "file_name": filename, "width": 3200, "height": 2256,
                  "reviewed": False, "boxes": []}
        manifest["images"].append(record)
        self._save(path, manifest)
        return record

    def save_boxes(self, project_id: str, image_id: int, boxes: list[dict]) -> dict:
        manifest = self.get(project_id)
        record = next((item for item in manifest["images"] if item["id"] == image_id), None)
        if record is None:
            raise AnnotationError("Image not found in this set")
        normalized = []
        for box in boxes:
            try:
                class_id = int(box["class_id"])
                x, y, width, height = (float(box[key]) for key in ("x", "y", "width", "height"))
            except (KeyError, TypeError, ValueError) as error:
                raise AnnotationError("Each box needs a class and four coordinates") from error
            if not 0 <= class_id < len(manifest["classes"]) or not all(np.isfinite(v) for v in (x, y, width, height)) or x < 0 or y < 0 or width < 2 or height < 2 or x + width > 3200 or y + height > 2256:
                raise AnnotationError("A box is outside the image or has an unknown class")
            normalized.append({"class_id": class_id, "x": x, "y": y, "width": width, "height": height})
        record["boxes"] = normalized
        record["reviewed"] = True
        self._save(self._path(project_id), manifest)
        return record

    def export(self, project_id: str) -> Path:
        path = self._path(project_id)
        manifest = self.get(project_id)
        if len(manifest["images"]) < 2 or any(not item["reviewed"] for item in manifest["images"]):
            raise AnnotationError("Review every image and include at least two before exporting")
        annotations = []
        for image in manifest["images"]:
            for box in image["boxes"]:
                annotations.append({"id": len(annotations) + 1, "image_id": image["id"],
                                    "category_id": box["class_id"] + 1,
                                    "bbox": [box["x"], box["y"], box["width"], box["height"]]})
        if not annotations:
            raise AnnotationError("Draw at least one object box before training")
        missing = [name for index, name in enumerate(manifest["classes"])
                   if not any(box["class_id"] == index for image in manifest["images"] for box in image["boxes"])]
        if missing:
            raise AnnotationError("Draw at least one box for every object name: " + ", ".join(missing))
        coco = {"images": [{key: image[key] for key in ("id", "file_name", "width", "height")}
                           for image in manifest["images"]],
                "categories": [{"id": index + 1, "name": name} for index, name in enumerate(manifest["classes"])],
                "annotations": annotations}
        archive = path / "annotated_images.zip"
        temporary = path / "annotated_images.tmp.zip"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as zipped:
            zipped.writestr("annotations/instances.json", json.dumps(coco, indent=2))
            for image in manifest["images"]:
                zipped.write(path / "images" / image["file_name"], f"images/{image['file_name']}")
        temporary.replace(archive)
        return archive
