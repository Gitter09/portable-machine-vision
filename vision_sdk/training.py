"""Opinionated, local YOLO26s training job."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2

from .atomic import write_json
from .config import VisionStore
from .compiler import CompileError, compile_candidate
from .dataset import DatasetError, import_coco_zip, prepare_training_dataset
from .model import export_six_heads
from .preprocessing import preprocessing_settings


class TrainingError(RuntimeError):
    pass


RUNNING_PHASES = {"queued", "checking_images", "preparing_images", "training", "quantizing", "validating", "exporting", "compiling"}


def _process_alive(pid: int) -> bool:
    """Whether a recorded job owner still runs, without signalling it.

    On Windows any non-console signal sent through os.kill terminates the target,
    so the liveness probe must use OpenProcess instead.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class TrainingManager:
    def __init__(self, store: VisionStore):
        self.store = store
        self._lock = threading.Lock()
        self._running = False
        jobs = self.store.root / "jobs"
        if jobs.is_dir():
            for status_path in jobs.glob("*/status.json"):
                try:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    if status.get("phase") not in RUNNING_PHASES:
                        continue
                    owner = status.get("owner")
                    if isinstance(owner, int) and owner != os.getpid() and _process_alive(owner):
                        continue
                    self._status(status_path.parent.name, phase="interrupted",
                                 error="The app closed before training finished. Start this job again.")
                except (OSError, ValueError):
                    continue

    def _status(self, job_id: str, **values) -> None:
        path = self.store.root / "jobs" / job_id / "status.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"job_id": job_id}
        previous.update(values)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def status(self, job_id: str) -> dict:
        uuid.UUID(job_id)
        path = self.store.root / "jobs" / job_id / "status.json"
        if not path.is_file():
            raise FileNotFoundError(job_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_jobs(self) -> list[dict]:
        jobs = self.store.root / "jobs"
        result = []
        for status_path in jobs.glob("*/status.json"):
            try:
                result.append(json.loads(status_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda job: job.get("started_at", 0), reverse=True)

    def start(self, archive: Path) -> str:
        with self._lock:
            if self._running:
                raise TrainingError("A training job is already running on this PC")
            self._running = True
        try:
            self.store.ensure()
            job_id = str(uuid.uuid4())
            job_dir = self.store.root / "jobs" / job_id
            job_dir.mkdir()
            saved_archive = job_dir / "dataset.zip"
            shutil.copy2(archive, saved_archive)
            self._status(job_id, phase="queued", started_at=time.time(), owner=os.getpid())
            thread = threading.Thread(target=self._run, args=(job_id, saved_archive), daemon=True, name=f"vision-train-{job_id[:8]}")
            thread.start()
            return job_id
        except Exception:
            with self._lock:
                self._running = False
            raise

    def _run(self, job_id: str, archive: Path) -> None:
        job_dir = self.store.root / "jobs" / job_id
        try:
            config = self.store.load()
            if not config.base_weights or not Path(config.base_weights).is_file():
                raise TrainingError("Choose the YOLO26s starting weights in Setup")
            if config.preprocessor.strip().lower() == "product_agnostic":
                if not config.flatfield_lut and not (config.preprocessing_profile and "flatfield" in Path(config.preprocessing_profile).read_text(encoding="utf-8")):
                    raise TrainingError(
                        "The product-agnostic preprocessor needs a flat-field map captured from the "
                        "empty fixture. Run tools/capture_flatfield.py and set it in Setup."
                    )
            try:
                import torch
                from ultralytics import YOLO
            except ImportError as error:
                raise TrainingError("Install the training extras: pip install 'demeter-vision-sdk[train]'") from error
            if not torch.cuda.is_available():
                raise TrainingError("An NVIDIA GPU and working CUDA installation are required for training")
            self._status(job_id, phase="checking_images")
            imported = self.store.root / "datasets" / job_id
            manifest = import_coco_zip(archive, imported)
            self._status(job_id, phase="preparing_images", classes=manifest["classes"], images=len(manifest["images"]))
            prepared = job_dir / "prepared"
            preparation = prepare_training_dataset(imported, prepared, config)
            self._status(job_id, phase="training", preparation=preparation)
            model = YOLO(config.base_weights)
            # Parts are placed by hand, so their orientation in a frame is
            # arbitrary. Ultralytics defaults degrees to 0.0, which would train a
            # model that only works when the operator happens to place the part
            # square to the camera. Rotating the labelled image rotates the box
            # with it, so this costs no annotation effort. Full turns are allowed
            # deliberately: a 180 degree error is a real placement, and the
            # cavities and caps are not distinguishable by upright-ness.
            result = model.train(data=str(prepared / "dataset.yaml"), epochs=50, batch=-1,
                                 imgsz=640, device=0, project=str(job_dir / "training"), name="yolo26s",
                                 degrees=config.rotation_degrees,
                                 workers=2, plots=False, save=True, exist_ok=False)
            best = Path(result.save_dir) / "weights" / "best.pt"
            if not best.is_file():
                raise TrainingError("Training finished but did not produce best.pt")
            quantization = "fp32"
            if config.qat_epochs > 0:
                self._status(job_id, phase="quantizing", baseline=str(best))
                qat_model = YOLO(str(best))
                qat_result = qat_model.train(data=str(prepared / "dataset.yaml"), epochs=config.qat_epochs, batch=-1,
                                             imgsz=640, device=0, project=str(job_dir / "qat"), name="yolo26s_qat",
                                             quantize=8, lr0=0.001, lrf=0.1, degrees=config.rotation_degrees,
                                             workers=2, plots=False, save=True, exist_ok=False)
                qat_best = Path(qat_result.save_dir) / "weights" / "best.pt"
                if not qat_best.is_file():
                    raise TrainingError("QAT finished but did not produce best.pt")
                best = qat_best
                quantization = "qat_int8"
            self._status(job_id, phase="validating")
            trained = YOLO(str(best))
            if quantization == "qat_int8":
                metrics = trained.val(data=str(prepared / "dataset.yaml"), split="val", device=0, plots=False, quantize=8)
            else:
                metrics = trained.val(data=str(prepared / "dataset.yaml"), split="val", device=0, plots=False)
            release = self.store.root / "releases" / job_id
            release.mkdir()
            shutil.copy2(best, release / "model.pt")
            preview_dir = release / "review"
            preview_dir.mkdir()
            previews = []
            for image_path in sorted((prepared / "images" / "val").glob("*.png"))[:6]:
                prediction = trained.predict(source=str(image_path), imgsz=640, device=0, verbose=False)[0]
                target = preview_dir / image_path.name
                cv2.imwrite(str(target), prediction.plot())
                previews.append(target.name)
            self._status(job_id, phase="exporting")
            onnx_path = Path(trained.export(format="onnx", imgsz=640, dynamic=False, simplify=False, opset=17, device=0))
            shutil.copy2(onnx_path, release / "model.onnx")
            export_six_heads(release / "model.onnx", release / "model_sima_6out.onnx", len(manifest["classes"]))
            release_manifest = {"version": job_id, "classes": manifest["classes"],
                                "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
                                "dataset_sha256": _hash_file(archive), "review_images": previews,
                                "quantization": quantization,
                                "preprocessing": preprocessing_settings(
                                    config.preprocessor,
                                    config.preprocessing_profile or None,
                                    config.flatfield_lut or None,
                                ),
                                "state": "trained_not_compiled"}
            if config.reference_image and Path(config.reference_image).is_file():
                shutil.copy2(config.reference_image, release / "reference.png")
            elif config.preprocessor.strip().lower() != "product_agnostic":
                raise TrainingError("Set a reference image in Setup for this preprocessor")
            (release / "labels.json").write_text(json.dumps(manifest["classes"], indent=2) + "\n", encoding="utf-8")
            setup = {"preprocessor": config.preprocessor}
            if config.roi is not None:
                setup["roi"] = list(config.roi)
            if config.preprocessing_profile:
                shutil.copy2(config.preprocessing_profile, release / "preprocessing_profile.json")
                setup["preprocessing_profile"] = "preprocessing_profile.json"
                release_manifest["preprocessing"]["preprocessing_profile"] = "preprocessing_profile.json"
            flatfield_source = config.flatfield_lut
            if not flatfield_source and config.preprocessing_profile:
                try:
                    profile_data = json.loads(Path(config.preprocessing_profile).read_text(encoding="utf-8"))
                    candidate = profile_data.get("flatfield_lut") or ""
                    flatfield_source = candidate if Path(candidate).is_file() else ""
                except (OSError, ValueError):
                    flatfield_source = ""
            if flatfield_source:
                shutil.copy2(flatfield_source, release / "flatfield.npz")
                setup["flatfield_lut"] = "flatfield.npz"
                release_manifest["preprocessing"]["flatfield_lut"] = "flatfield.npz"
            (release / "vision_setup.json").write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
            write_json(release / "manifest.json", release_manifest)
            if config.compiler_image:
                self._status(job_id, phase="compiling", release=job_id)
                try:
                    release_manifest["board_build"] = compile_candidate(release, prepared, config.compiler_image)
                    release_manifest["state"] = "compiled_not_board_verified"
                except CompileError as error:
                    release_manifest["compile_error"] = str(error)
                write_json(release / "manifest.json", release_manifest)
            self._status(job_id, phase="needs_board_check" if release_manifest["state"] == "compiled_not_board_verified" else "needs_board_compile",
                         release=job_id, metrics={"map50": release_manifest["map50"], "map50_95": release_manifest["map50_95"]},
                         note=release_manifest.get("compile_error"))
        except Exception as error:
            self._status(job_id, phase="failed", error=str(error))
        finally:
            self._status(job_id, owner=None)
            with self._lock:
                self._running = False

    def start_compile(self, job_id: str) -> dict:
        """Re-run only the SiMa build for a job that already trained."""
        with self._lock:
            if self._running:
                raise TrainingError("A training job is already running on this PC")
            self._running = True
        try:
            uuid.UUID(job_id)
            self.status(job_id)
            config = self.store.load()
            if not config.compiler_image:
                raise TrainingError("Set the SiMa compiler image in Setup")
            release = self.store.root / "releases" / job_id
            prepared = self.store.root / "jobs" / job_id / "prepared"
            if not (release / "manifest.json").is_file() or not prepared.is_dir():
                raise TrainingError("The release or its prepared images are missing; train this job again")
            self._status(job_id, phase="compiling", release=job_id, compile_error=None)
            thread = threading.Thread(target=self._run_compile, args=(job_id,), daemon=True, name=f"vision-compile-{job_id[:8]}")
            thread.start()
        except Exception:
            with self._lock:
                self._running = False
            raise
        return {"job_id": job_id, "phase": "compiling"}

    def _run_compile(self, job_id: str) -> None:
        release = self.store.root / "releases" / job_id
        prepared = self.store.root / "jobs" / job_id / "prepared"
        try:
            config = self.store.load()
            manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
            try:
                manifest["board_build"] = compile_candidate(release, prepared, config.compiler_image)
                manifest["state"] = "compiled_not_board_verified"
                manifest.pop("compile_error", None)
            except CompileError as error:
                manifest["compile_error"] = str(error)
            write_json(release / "manifest.json", manifest)
            self._status(job_id,
                         phase="needs_board_check" if manifest["state"] == "compiled_not_board_verified" else "needs_board_compile",
                         release=job_id, note=manifest.get("compile_error"))
        except Exception as error:
            self._status(job_id, phase="failed", error=str(error))
        finally:
            with self._lock:
                self._running = False


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
