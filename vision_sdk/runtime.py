"""Stable inspection operations over the existing SiMa board protocol."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import VisionStore
from .board import BoardConnection
from .light import LightSession, port_present
from .preprocessing.state_counts import summarise


class VisionClient:
    def __init__(self, store: VisionStore | None = None):
        self.store = store or VisionStore()

    def health(self) -> dict:
        config = self.store.load()
        result = {"board": False, "camera": False, "light_port": port_present(config.light_port),
                  "reference": bool(config.reference_image and Path(config.reference_image).is_file()),
                  "area": config.roi is not None, "base_weights": bool(config.base_weights and Path(config.base_weights).is_file())}
        try:
            with BoardConnection(config.board_host, config.board_port, timeout=2) as board:
                result["board"] = True
                result["camera"] = bool(board.ready.get("camera")) and board.ready.get("camera_ip") == config.camera_ip
                result["board_release"] = board.ready.get("release_id")
                result["board_classes"] = board.ready.get("classes")
        except (OSError, RuntimeError):
            pass
        return result

    def capture_and_inspect(self) -> dict:
        config = self.store.load()
        with LightSession(config.light_port):
            with BoardConnection(config.board_host, config.board_port, timeout=60) as board:
                running = board.ready.get("release_id")
                if config.active_release and running != config.active_release:
                    raise RuntimeError(
                        "The board is running a different model version than the one activated on this PC; "
                        "activate the model again before inspecting"
                    )
                capture = board.capture()
        body = capture.header
        detections = body.get("detections", [])
        return {"detections": detections,
                "summary": body.get("summary") or summarise(detections),
                "instance_count": body.get("instance_count"),
                "possibly_merged": body.get("possibly_merged", 0),
                "preprocessor": body.get("preprocessor", {}),
                "image_jpeg": capture.image_jpeg,
                "source_width": body.get("source_width"), "source_height": body.get("source_height"),
                "model_version": running or config.active_release, "health": body.get("health", {})}

    def inspect_image(self, image: bytes) -> dict:
        """Inspect an uploaded image with the active release's own preprocessor.

        This is for setup and model review. The preprocessor is built from the
        release's vision_setup.json, so a product-agnostic release is reviewed
        with the same profile and flat-field map that the board runs.
        """
        if not image:
            raise ValueError("Image is empty")
        config = self.store.load()
        release_dir = self.store.root / "releases" / config.active_release
        model_path = release_dir / "model.pt"
        if not config.active_release or not model_path.is_file():
            raise RuntimeError("Activate a model before inspecting images")
        setup_path = release_dir / "vision_setup.json"
        setup = json.loads(setup_path.read_text(encoding="utf-8")) if setup_path.is_file() else {"preprocessor": "fixed_reference_realtime"}
        frame = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Image cannot be opened")

        def release_asset(name) -> str | None:
            if not name:
                return None
            path = Path(name)
            if not path.is_absolute():
                path = release_dir / path
            return str(path) if path.is_file() else None

        try:
            from ultralytics import YOLO
            from .preprocessing.base import Box
            from .preprocessing.factory import resolve_preprocessor
        except ImportError as error:
            raise RuntimeError("Install the training extras to inspect uploaded images on the PC") from error
        name = str(setup.get("preprocessor", "fixed_reference_realtime"))
        if name != "product_agnostic" and release_asset("reference.png") is None:
            raise RuntimeError("The active release has no reference image")
        preprocessor = resolve_preprocessor(
            name,
            reference_image=release_asset("reference.png"),
            roi=tuple(setup["roi"]) if "roi" in setup else config.roi,
            profile_path=release_asset(setup.get("preprocessing_profile")),
            flatfield_lut=release_asset(setup.get("flatfield_lut")),
        )
        _, model_image, metadata = preprocessor.preprocess(frame, require_alignment=True)
        prediction = YOLO(str(model_path)).predict(source=model_image, imgsz=640, verbose=False)[0]
        detections = []
        for box in prediction.boxes:
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0])
            source = preprocessor.inverse_map(Box(x1, y1, x2 - x1, y2 - y1), metadata)
            class_id = int(box.cls[0])
            detections.append({"class_id": class_id, "class_name": prediction.names[class_id],
                               "confidence": float(box.conf[0]), "box_source": source.as_dict()})
        return {"model_version": config.active_release, "preprocessor": name,
                "detections": detections, "summary": summarise(detections),
                "source_width": frame.shape[1], "source_height": frame.shape[0]}
