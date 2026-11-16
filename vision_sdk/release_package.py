"""Portable, hash-checked handoff from the GPU trainer to the Admin runtime."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
import zipfile
from pathlib import Path

from .config import VisionStore


FORMAT = "demeter-vision-release-1"
REQUIRED = {"manifest.json", "yolov8_stage1_mla.elf", "yolov8_mpk.json",
            "labels.json", "vision_setup.json"}
#: Present only for some preprocessors: a fixed-reference release ships the
#: reference image, a product-agnostic release ships its profile and map.
OPTIONAL_ASSETS = {"reference.png", "preprocessing_profile.json", "flatfield.npz"}
MAX_BYTES = 4 * 1024**3


class ReleasePackageError(ValueError):
    pass


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def export_release(store: VisionStore, release_id: str) -> Path:
    uuid.UUID(release_id)
    release = store.root / "releases" / release_id
    manifest_path = release / "manifest.json"
    if not manifest_path.is_file():
        raise ReleasePackageError("Model version not found")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != release_id or manifest.get("state") != "compiled_not_board_verified":
        raise ReleasePackageError("Only a compiled, unactivated SiMa candidate can be sent to the Admin PC")
    setup_path = release / "vision_setup.json"
    if not setup_path.is_file():
        raise ReleasePackageError("Compiled model package is missing a required file")
    try:
        setup = json.loads(setup_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleasePackageError(f"vision_setup.json cannot be read: {error}") from error
    agnostic = str(setup.get("preprocessor", "")).strip().lower() == "product_agnostic"
    if agnostic:
        if not (release / "flatfield.npz").is_file():
            raise ReleasePackageError("A product-agnostic release needs its flat-field map")
    elif not (release / "reference.png").is_file():
        raise ReleasePackageError("A fixed-reference release needs its reference image")
    names = set(REQUIRED)
    for optional in OPTIONAL_ASSETS:
        if (release / optional).is_file():
            names.add(optional)
    for name in manifest.get("review_images", []):
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".png"):
            raise ReleasePackageError("Review image name is invalid")
        if not (release / "review" / name).is_file():
            raise ReleasePackageError("Review image is missing")
        names.add(f"review/{name}")
    if any(not (release / name).is_file() for name in names):
        raise ReleasePackageError("Compiled model package is missing a required file")
    hashes = {name: _digest(release / name) for name in sorted(names)}
    archive = release / f"demeter-model-{release_id}.zip"
    temporary = archive.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zipped:
        zipped.writestr("package.json", json.dumps({"format": FORMAT, "release_id": release_id,
                                                      "files": hashes}, indent=2))
        for name in sorted(names):
            zipped.write(release / name, name)
    temporary.replace(archive)
    return archive


def import_release(store: VisionStore, archive: Path) -> dict:
    store.ensure()
    staging = store.root / "releases" / f".incoming_{uuid.uuid4().hex}"
    try:
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            if len(members) > 20 or len({item.filename for item in members}) != len(members):
                raise ReleasePackageError("Model package has too many or duplicate files")
            if sum(item.file_size for item in members) > MAX_BYTES:
                raise ReleasePackageError("Model package exceeds the 4 GB limit")
            if "package.json" not in zipped.namelist() or zipped.getinfo("package.json").file_size > 1024 * 1024:
                raise ReleasePackageError("Model package manifest is missing or too large")
            package = json.loads(zipped.read("package.json"))
            if not isinstance(package, dict):
                raise ReleasePackageError("This is not a compiled Demeter model package")
            release_id_value = package.get("release_id")
            if not isinstance(release_id_value, str):
                raise ReleasePackageError("This is not a compiled Demeter model package")
            release_id = str(uuid.UUID(release_id_value))
            files = package.get("files")
            allowed = REQUIRED | OPTIONAL_ASSETS
            if package.get("format") != FORMAT or not isinstance(files, dict) or not REQUIRED.issubset(files):
                raise ReleasePackageError("This is not a compiled Demeter model package")
            if set(zipped.namelist()) != set(files) | {"package.json"}:
                raise ReleasePackageError("Model package file list does not match its manifest")
            if any(name not in allowed and (not name.startswith("review/") or
                   Path(name).name != name.removeprefix("review/") or not name.endswith(".png")) for name in files):
                raise ReleasePackageError("Model package contains an unexpected file path")
            destination = store.root / "releases" / release_id
            if destination.exists():
                raise ReleasePackageError("This model version is already on the Admin PC")
            staging.mkdir()
            for name, expected in files.items():
                if not isinstance(expected, str) or len(expected) != 64:
                    raise ReleasePackageError("Model package checksum is invalid")
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
                if _digest(target) != expected:
                    raise ReleasePackageError(f"Model package checksum failed: {name}")
        manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != release_id or manifest.get("state") != "compiled_not_board_verified":
            raise ReleasePackageError("Model package version or build state is invalid")
        review = manifest.get("review_images", [])
        if not isinstance(review, list):
            raise ReleasePackageError("Model package review list is invalid")
        for name in review:
            if (not isinstance(name, str) or Path(name).name != name
                    or not name.endswith(".png") or f"review/{name}" not in files):
                raise ReleasePackageError("Model package review list is invalid")
        if json.loads((staging / "labels.json").read_text(encoding="utf-8")) != manifest.get("classes"):
            raise ReleasePackageError("Model package class labels do not match")
        staging.rename(destination)
        return {"release": release_id, "classes": manifest["classes"], "state": manifest["state"]}
    except (zipfile.BadZipFile, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ReleasePackageError(f"Model package cannot be opened: {error}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)
