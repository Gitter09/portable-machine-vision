"""Loopback setup app. It never writes PLC outputs."""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import VisionConfig, VisionStore
from .annotations import AnnotationError, AnnotationWorkspace
from .dataset import DatasetError, inspect_coco_zip
from .runtime import VisionClient
from .light import LightError, LightSession
from .lease import LeaseController
from .release import ReleaseError, ReleaseManager
from .release_package import ReleasePackageError, export_release, import_release
from .training import TrainingError, TrainingManager


LOGGER = logging.getLogger(__name__)


def create_app(store: VisionStore | None = None, mode: str = "trainer") -> FastAPI:
    if mode not in {"trainer", "runtime"}:
        raise ValueError("SDK mode must be trainer or runtime")
    store = store or VisionStore()
    store.ensure()
    client = VisionClient(store)
    manager = TrainingManager(store)
    releases = ReleaseManager(store)
    annotations = AnnotationWorkspace(store)
    lease = LeaseController(store)
    session_lock = threading.RLock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if mode == "runtime":
            lease.start()
        try:
            yield
        finally:
            if mode == "runtime":
                try:
                    with session_lock:
                        lease.stop()
                        releases.stop_active_worker()
                except (ReleaseError, OSError) as error:
                    LOGGER.warning("SDK worker could not be stopped during app exit: %s", error)

    app = FastAPI(title="Demeter Vision Setup", lifespan=lifespan)
    app.state.mode = mode
    static = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    @app.get("/api/state")
    def state():
        config = store.load()
        return {"mode": mode, "config": asdict(config), "health": client.health() if mode == "runtime" else {},
                "releases": sorted([path.name for path in (store.root / "releases").iterdir() if (path / "manifest.json").is_file()]),
                "jobs": manager.list_jobs()[:5] if mode == "trainer" else []}

    def require_mode(expected: str) -> None:
        if mode != expected:
            raise HTTPException(403, f"This action belongs on the {expected} computer")

    @app.post("/api/session/quit")
    def quit_sdk():
        require_mode("runtime")
        try:
            with session_lock:
                result = releases.stop_active_worker()
                lease.stop()
        except (ReleaseError, OSError) as error:
            raise HTTPException(409, str(error)) from error
        server = getattr(app.state, "uvicorn_server", None)
        if server is not None:
            threading.Timer(0.2, lambda: setattr(server, "should_exit", True)).start()
        return {**result, "message": "SDK camera released. It is safe to start the existing machine application."}

    @app.post("/api/session/start")
    def start_sdk_session():
        require_mode("runtime")
        try:
            with session_lock:
                return releases.start_active_worker()
        except (ReleaseError, OSError) as error:
            raise HTTPException(409, str(error)) from error

    @app.put("/api/config")
    async def save_config(request: Request):
        try:
            values = await request.json()
            if not isinstance(values, dict):
                raise ValueError("Send the configuration as an object")
            if {"active_release", "previous_release"} & set(values):
                raise ValueError("The active model can only change through activation or rollback")
            with session_lock:
                config = VisionConfig.from_dict({**asdict(store.load()), **values})
                store.save(config)
            return {"config": asdict(config)}
        except (TypeError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/reference")
    async def upload_reference(request: Request):
        require_mode("trainer")
        blob = await request.body()
        if not blob or len(blob) > 40 * 1024 * 1024:
            raise HTTPException(400, "Choose a camera image smaller than 40 MB")
        image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "This image cannot be opened")
        height, width = image.shape[:2]
        if (width, height) != (3200, 2256):
            raise HTTPException(400, "The current SiMa camera profile requires a 3200 × 2256 reference image")
        path = store.root / "references" / f"{uuid.uuid4().hex}.png"
        cv2.imwrite(str(path), image)
        config = store.load()
        config.reference_image = str(path)
        config.roi = None
        store.save(config)
        return {"reference": str(path), "width": width, "height": height}

    @app.get("/api/reference")
    def reference():
        require_mode("trainer")
        path = Path(store.load().reference_image)
        if not path.is_file():
            raise HTTPException(404, "Upload a reference image first")
        return FileResponse(path, media_type="image/png")

    @app.post("/api/capture")
    def capture():
        require_mode("runtime")
        try:
            result = client.capture_and_inspect()
            image = result.pop("image_jpeg")
            result["preview"] = "data:image/jpeg;base64," + base64.b64encode(image).decode("ascii")
            return result
        except Exception as error:
            raise HTTPException(503, f"Board capture is unavailable: {error}") from error

    @app.post("/api/light/check")
    def check_light():
        require_mode("runtime")
        try:
            with LightSession(store.load().light_port):
                pass
            return {"light": True}
        except (LightError, OSError) as error:
            raise HTTPException(503, f"Light did not answer: {error}") from error

    @app.post("/api/train")
    async def train(request: Request):
        require_mode("trainer")
        path = store.root / "jobs" / f"upload_{uuid.uuid4().hex}.zip"
        size = 0
        try:
            with path.open("wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 20 * 1024**3:
                        raise HTTPException(413, "Dataset ZIP exceeds 20 GB")
                    handle.write(chunk)
            if size == 0:
                raise HTTPException(400, "Choose a COCO ZIP to upload")
            report = inspect_coco_zip(path)
            job_id = manager.start(path)
            return {"job_id": job_id, "images": len(report["images"]), "classes": report["classes"]}
        except (DatasetError, TrainingError) as error:
            raise HTTPException(400, str(error)) from error
        finally:
            path.unlink(missing_ok=True)

    @app.get("/api/dataset/template")
    def dataset_template():
        require_mode("trainer")
        buffer = io.BytesIO()
        examples = []
        annotations = []
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for index, (name, center) in enumerate((("example_1.png", (1100, 1000)), ("example_2.png", (1800, 1000))), 1):
                image = np.full((2256, 3200, 3), (34, 40, 46), dtype=np.uint8)
                x, y = center
                cv2.rectangle(image, (x, y), (x + 160, y + 140), (90, 155, 220), -1)
                ok, encoded = cv2.imencode(".png", image)
                if not ok:
                    raise HTTPException(500, "Could not create example ZIP")
                zipped.writestr(f"images/{name}", encoded.tobytes())
                examples.append({"id": index, "file_name": name, "width": 3200, "height": 2256})
                annotations.append({"id": index, "image_id": index, "category_id": 1, "bbox": [x, y, 160, 140]})
            zipped.writestr("annotations/instances.json", json.dumps({"images": examples, "annotations": annotations,
                                                                        "categories": [{"id": 1, "name": "example_object"}]}, indent=2))
            zipped.writestr("READ_ME.txt", "This ZIP shows the required folder and COCO annotation format. Replace the example images, class name, and boxes with your own camera images and annotations. The colored squares are examples, not a useful training dataset.\n")
        return Response(buffer.getvalue(), media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="demeter-coco-example.zip"'})

    @app.post("/api/annotations")
    async def create_annotations(request: Request):
        require_mode("trainer")
        try:
            body = await request.json()
            return annotations.create(body.get("classes", []))
        except (AnnotationError, ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/annotations/{project_id}")
    def annotation_project(project_id: str):
        require_mode("trainer")
        try:
            return annotations.get(project_id)
        except (AnnotationError, ValueError) as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/api/annotations/{project_id}/images")
    async def add_annotation_image(project_id: str, request: Request):
        require_mode("trainer")
        try:
            return annotations.add_image(project_id, await request.body())
        except (AnnotationError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/annotations/{project_id}/images/{image_id}")
    def annotation_image(project_id: str, image_id: int):
        require_mode("trainer")
        try:
            project = annotations.get(project_id)
            image = next((item for item in project["images"] if item["id"] == image_id), None)
            if image is None:
                raise AnnotationError("Image not found")
            return FileResponse(annotations._path(project_id) / "images" / image["file_name"], media_type="image/png")
        except (AnnotationError, ValueError) as error:
            raise HTTPException(404, str(error)) from error

    @app.put("/api/annotations/{project_id}/images/{image_id}")
    async def save_annotation_boxes(project_id: str, image_id: int, request: Request):
        require_mode("trainer")
        try:
            body = await request.json()
            if not isinstance(body.get("boxes"), list):
                raise AnnotationError("Send a list of object boxes")
            return annotations.save_boxes(project_id, image_id, body["boxes"])
        except (AnnotationError, ValueError, TypeError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/annotations/{project_id}/export")
    def export_annotations(project_id: str):
        require_mode("trainer")
        try:
            return FileResponse(annotations.export(project_id), media_type="application/zip",
                                filename="demeter-annotated-images.zip")
        except (AnnotationError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/annotations/{project_id}/train")
    def train_annotations(project_id: str):
        require_mode("trainer")
        try:
            return {"job_id": manager.start(annotations.export(project_id))}
        except (AnnotationError, TrainingError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        require_mode("trainer")
        try:
            return manager.status(job_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, "Training job not found") from error

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        require_mode("trainer")
        try:
            prior = manager.status(job_id)
            if prior.get("phase") not in {"failed", "interrupted"}:
                raise HTTPException(409, "This job has not stopped")
            archive = store.root / "jobs" / job_id / "dataset.zip"
            return {"job_id": manager.start(archive)}
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, "Training job not found") from error
        except TrainingError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/jobs/{job_id}/compile")
    def compile_job(job_id: str):
        require_mode("trainer")
        try:
            return manager.start_compile(job_id)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(404, "Training job not found") from error
        except TrainingError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/releases/{release_id}")
    def release(release_id: str):
        try:
            uuid.UUID(release_id)
        except ValueError as error:
            raise HTTPException(404, "Model version not found") from error
        path = store.root / "releases" / release_id / "manifest.json"
        if not path.is_file():
            raise HTTPException(404, "Model version not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/releases/{release_id}/review/{image_name}")
    def review_image(release_id: str, image_name: str):
        manifest = release(release_id)
        if (image_name not in manifest.get("review_images", []) or Path(image_name).name != image_name
                or not image_name.endswith(".png")):
            raise HTTPException(404, "Review image not found")
        review_dir = (store.root / "releases" / release_id / "review").resolve()
        path = (review_dir / image_name).resolve()
        if path.parent != review_dir or not path.is_file():
            raise HTTPException(404, "Review image not found")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/releases/{release_id}/export")
    def export_model_package(release_id: str):
        require_mode("trainer")
        try:
            return FileResponse(export_release(store, release_id), media_type="application/zip",
                                filename=f"demeter-model-{release_id}.zip")
        except (ReleasePackageError, ValueError) as error:
            raise HTTPException(400, str(error)) from error

    @app.post("/api/releases/import")
    async def import_model_package(request: Request):
        require_mode("runtime")
        path = store.root / "releases" / f".upload_{uuid.uuid4().hex}.zip"
        size = 0
        try:
            with path.open("wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 4 * 1024**3:
                        raise HTTPException(413, "Model package exceeds 4 GB")
                    handle.write(chunk)
            return import_release(store, path)
        except (ReleasePackageError, ValueError) as error:
            raise HTTPException(400, str(error)) from error
        finally:
            path.unlink(missing_ok=True)

    @app.post("/api/releases/{release_id}/activate")
    def activate(release_id: str):
        require_mode("runtime")
        try:
            with session_lock:
                return releases.activate(release_id)
        except (ReleaseError, ValueError, OSError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/releases/{release_id}/stage")
    def stage(release_id: str):
        require_mode("runtime")
        try:
            return releases.stage(release_id)
        except (ReleaseError, ValueError, OSError) as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/rollback")
    def rollback():
        require_mode("runtime")
        previous = store.load().previous_release
        if not previous:
            raise HTTPException(409, "No previous SDK model is available")
        try:
            with session_lock:
                return releases.activate(previous)
        except (ReleaseError, ValueError, OSError) as error:
            raise HTTPException(409, str(error)) from error

    return app
