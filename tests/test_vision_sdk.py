from __future__ import annotations

import json
import socket
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from vision_sdk.app import create_app
from vision_sdk.annotations import AnnotationError, AnnotationWorkspace
from vision_sdk.board import BoardConnection, BoardError
from vision_sdk.board_runtime.camera_lease import CameraLease
from vision_sdk.compiler import CompileError
from vision_sdk.config import VisionConfig, VisionStore
from vision_sdk.dataset import DatasetError, _leak_free_groups, import_coco_zip, inspect_coco_zip, prepare_training_dataset
from vision_sdk.preprocessing import build_flatfield, save_flatfield
from vision_sdk.release import ReleaseError, ReleaseManager
from vision_sdk.release_package import ReleasePackageError, export_release, import_release
from vision_sdk.runtime import VisionClient
from vision_sdk.training import TrainingError, TrainingManager


def make_archive(path: Path, *, bad_box: bool = False, unsafe: bool = False) -> None:
    image = np.full((32, 48, 3), 130, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    data = {
        "images": [{"id": 1, "file_name": "a.png", "width": 48, "height": 32},
                   {"id": 2, "file_name": "b.png", "width": 48, "height": 32}],
        "categories": [{"id": 7, "name": "screw"}],
        "annotations": [{"id": 11, "image_id": 1, "category_id": 7,
                         "bbox": [45, 2, 8, 8] if bad_box else [2, 3, 8, 9]}],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("annotations/instances.json", json.dumps(data))
        archive.writestr("images/a.png", encoded.tobytes())
        archive.writestr("images/b.png", encoded.tobytes())
        if unsafe:
            archive.writestr("../escape.txt", "bad")


def test_coco_import_preserves_explicit_negative_image(tmp_path: Path):
    archive = tmp_path / "dataset.zip"
    make_archive(archive)
    report = inspect_coco_zip(archive)
    assert report["classes"] == ["screw"]
    assert report["negative_images"] == 1
    imported = import_coco_zip(archive, tmp_path / "imported")
    assert imported["labels"][1][0]["class_id"] == 0
    assert (tmp_path / "imported" / "images" / "image_2.png").is_file()


@pytest.mark.parametrize("bad_box,unsafe", [(True, False), (False, True)])
def test_coco_import_rejects_bad_box_or_unsafe_zip(tmp_path: Path, bad_box: bool, unsafe: bool):
    archive = tmp_path / "dataset.zip"
    make_archive(archive, bad_box=bad_box, unsafe=unsafe)
    with pytest.raises(DatasetError):
        inspect_coco_zip(archive)


def test_local_app_rejects_unverified_model_activation(tmp_path: Path):
    store = VisionStore(tmp_path)
    app = create_app(store)
    with TestClient(app) as client:
        response = client.get("/api/state")
        assert response.status_code == 200
        assert response.json()["config"]["active_release"] == ""
        bad = client.post("/api/train", content=b"not a zip")
        assert bad.status_code == 400
        template = client.get("/api/dataset/template")
        assert template.status_code == 200
        template_path = tmp_path / "template.zip"
        template_path.write_bytes(template.content)
        assert inspect_coco_zip(template_path)["classes"] == ["example_object"]
        assert client.post("/api/session/quit").status_code == 403
    with TestClient(create_app(store, mode="runtime")) as client:
        assert client.get("/api/state").json()["mode"] == "runtime"
        assert client.post("/api/train", content=b"dataset").status_code == 403
        assert client.post("/api/session/quit").json()["stopped"] is True


def test_release_package_moves_compiled_candidate_between_hosts(tmp_path: Path):
    trainer = VisionStore(tmp_path / "trainer")
    runtime = VisionStore(tmp_path / "runtime")
    trainer.ensure()
    release_id = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    release = trainer.root / "releases" / release_id
    release.mkdir()
    (release / "manifest.json").write_text(json.dumps({"version": release_id,
        "state": "compiled_not_board_verified", "classes": ["screw"], "review_images": []}))
    (release / "labels.json").write_text(json.dumps(["screw"]))
    for name in ("yolov8_stage1_mla.elf", "yolov8_mpk.json", "reference.png"):
        (release / name).write_bytes(b"candidate-test-data")
    (release / "vision_setup.json").write_text(json.dumps({"preprocessor": "fixed_reference_realtime"}))
    archive = export_release(trainer, release_id)
    imported = import_release(runtime, archive)
    assert imported["release"] == release_id
    assert (runtime.root / "releases" / release_id / "yolov8_stage1_mla.elf").is_file()
    assert not (runtime.root / "releases" / release_id / "model.pt").exists()
    with pytest.raises(ReleasePackageError, match="already"):
        import_release(runtime, archive)
    api_runtime = VisionStore(tmp_path / "api_runtime")
    with TestClient(create_app(trainer, mode="trainer")) as client:
        assert client.get(f"/api/releases/{release_id}/export").status_code == 200
        assert client.post("/api/releases/import", content=archive.read_bytes()).status_code == 403
    with TestClient(create_app(api_runtime, mode="runtime")) as client:
        assert client.post("/api/releases/import", content=archive.read_bytes()).json()["release"] == release_id
        assert client.get(f"/api/releases/{release_id}/export").status_code == 403


def test_release_package_rejects_changed_artifact(tmp_path: Path):
    store = VisionStore(tmp_path / "trainer")
    store.ensure()
    release_id = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    release = store.root / "releases" / release_id
    release.mkdir()
    (release / "manifest.json").write_text(json.dumps({"version": release_id,
        "state": "compiled_not_board_verified", "classes": ["screw"], "review_images": []}))
    (release / "labels.json").write_text(json.dumps(["screw"]))
    for name in ("yolov8_stage1_mla.elf", "yolov8_mpk.json", "reference.png"):
        (release / name).write_bytes(b"test")
    (release / "vision_setup.json").write_text(json.dumps({"preprocessor": "fixed_reference_realtime"}))
    original = export_release(store, release_id)
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(changed, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"different" if name == "yolov8_stage1_mla.elf" else source.read(name))
    with pytest.raises(ReleasePackageError, match="checksum failed"):
        import_release(VisionStore(tmp_path / "runtime"), changed)


def test_release_bundle_contains_board_runtime_and_model_contract(tmp_path: Path):
    store = VisionStore(tmp_path)
    store.ensure()
    release_id = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    release = tmp_path / "releases" / release_id
    release.mkdir()
    (release / "manifest.json").write_text(json.dumps({"version": release_id, "state": "compiled_not_board_verified", "classes": ["a", "b"]}))
    for name in ("yolov8_stage1_mla.elf", "yolov8_mpk.json", "labels.json", "reference.png"):
        (release / name).write_bytes(b"test fixture")
    (release / "vision_setup.json").write_text(json.dumps({"preprocessor": "fixed_reference_realtime"}))
    bundle = ReleaseManager(store).prepare_bundle(release_id)
    assert (bundle / "persistent_board_worker.py").is_file()
    assert (bundle / "camera_lease.py").is_file()
    assert (bundle / "triggered_camera.py").is_file()
    assert (bundle / "fault_evidence.py").is_file()
    assert (bundle / "preprocessing" / "fixed_ref_prep.py").is_file()
    assert json.loads((bundle / "vision_release.json").read_text())["classes"] == ["a", "b"]
    assert "yolov8_stage1_mla.elf" in json.loads((bundle / "bundle_manifest.json").read_text())


def test_sdk_worker_uses_separate_port_and_migrates_old_state():
    assert VisionConfig().board_port == 5005
    assert VisionConfig.from_dict({"board_port": 5004}).board_port == 5005


def test_sdk_camera_lease_expires_without_valid_heartbeat():
    release_id = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    lease = CameraLease("127.0.0.1", "127.0.0.1", release_id, 0)
    lease.start()
    try:
        lease.deadline = time.monotonic() - 1
        assert lease.expired()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
            channel.sendto(b"wrong-release", ("127.0.0.1", lease.port))
            time.sleep(0.05)
            assert lease.expired()
            channel.sendto(release_id.encode("ascii"), ("127.0.0.1", lease.port))
        deadline = time.monotonic() + 1
        while lease.expired() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not lease.expired()
    finally:
        lease.close()


def test_sdk_quit_never_stops_foreign_worker(tmp_path: Path, monkeypatch):
    store = VisionStore(tmp_path)
    config = store.load()
    config.active_release = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    store.save(config)
    known_release = tmp_path / "releases" / config.active_release
    known_release.mkdir()
    (known_release / "manifest.json").write_text(json.dumps({"state": "board_verified"}))
    stopped = []

    class FakeBoard:
        ready = {"release_id": "different-release"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def shutdown(self):
            stopped.append(True)

    monkeypatch.setattr("vision_sdk.release.BoardConnection", lambda *_args, **_kwargs: FakeBoard())
    with pytest.raises(ReleaseError, match="different worker"):
        ReleaseManager(store).stop_active_worker()
    assert stopped == []


def test_sdk_activation_refuses_running_old_worker(monkeypatch):
    class OpenPort:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr("vision_sdk.release.socket.create_connection", lambda *_args, **_kwargs: OpenPort())
    with pytest.raises(ReleaseError, match="existing machine worker is still running"):
        ReleaseManager._require_old_worker_stopped(VisionConfig())


def test_sdk_activation_checks_board_over_ssh_when_old_port_times_out(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    checks = []
    monkeypatch.setattr("vision_sdk.release.socket.create_connection", timeout)
    monkeypatch.setattr(ReleaseManager, "_connection", staticmethod(lambda config: ("sima@192.168.1.20", config.board_host)))
    monkeypatch.setattr(ReleaseManager, "_ssh", staticmethod(lambda *args, **kwargs: checks.append(args)))
    ReleaseManager._require_old_worker_stopped(VisionConfig())
    assert len(checks) == 1
    assert "sport = :5004" in checks[0][1]


def test_sdk_activation_does_not_ignore_failed_ssh_check(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("timed out")

    def failed_check(*args, **kwargs):
        raise ReleaseError("Board command failed")

    monkeypatch.setattr("vision_sdk.release.socket.create_connection", timeout)
    monkeypatch.setattr(ReleaseManager, "_connection", staticmethod(lambda config: ("sima@192.168.1.20", config.board_host)))
    monkeypatch.setattr(ReleaseManager, "_ssh", staticmethod(failed_check))
    with pytest.raises(ReleaseError, match="port 5004"):
        ReleaseManager._require_old_worker_stopped(VisionConfig())


def test_sdk_shutdown_requires_camera_release_ack(monkeypatch):
    board = BoardConnection("127.0.0.1")
    monkeypatch.setattr(board, "_send", lambda *_args: "request-1")
    monkeypatch.setattr(board, "_receive", lambda: ({"request_id": "request-1", "type": "stopped",
                                                    "body": {"camera_released": True}}, b""))
    board.shutdown()
    monkeypatch.setattr(board, "_receive", lambda: ({"request_id": "request-1", "type": "stopped",
                                                    "body": {"camera_released": False}}, b""))
    with pytest.raises(BoardError, match="did not confirm"):
        board.shutdown()


def test_local_annotation_workspace_exports_valid_coco_zip(tmp_path: Path):
    workspace = AnnotationWorkspace(VisionStore(tmp_path))
    project = workspace.create(["screw"])
    image = np.full((2256, 3200, 3), 100, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    first = workspace.add_image(project["project_id"], encoded.tobytes())
    second = workspace.add_image(project["project_id"], encoded.tobytes())
    with pytest.raises(AnnotationError):
        workspace.export(project["project_id"])
    workspace.save_boxes(project["project_id"], first["id"],
                         [{"class_id": 0, "x": 100, "y": 200, "width": 60, "height": 70}])
    workspace.save_boxes(project["project_id"], second["id"], [])
    archive = workspace.export(project["project_id"])
    report = inspect_coco_zip(archive)
    assert report["classes"] == ["screw"]
    assert report["negative_images"] == 1
    with pytest.raises(AnnotationError):
        workspace.save_boxes(project["project_id"], first["id"],
                             [{"class_id": 0, "x": 3199, "y": 100, "width": 10, "height": 10}])


def test_annotation_api_creates_and_recovers_project(tmp_path: Path):
    with TestClient(create_app(VisionStore(tmp_path))) as client:
        created = client.post("/api/annotations", json={"classes": ["screw"]})
        assert created.status_code == 200
        project_id = created.json()["project_id"]
        assert client.get(f"/api/annotations/{project_id}").json()["images"] == []
        assert client.get(f"/api/annotations/{project_id}/export").status_code == 400
        assert client.post("/api/annotations", json={"classes": ["screw", "screw"]}).status_code == 400
        assert client.post("/api/annotations", json={"classes": "screw"}).status_code == 400


def test_release_package_supports_product_agnostic_assets(tmp_path: Path):
    store = VisionStore(tmp_path)
    store.ensure()
    release_id = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    release = store.root / "releases" / release_id
    release.mkdir()
    (release / "manifest.json").write_text(json.dumps({"version": release_id,
        "state": "compiled_not_board_verified", "classes": ["cavity_full"], "review_images": []}))
    (release / "labels.json").write_text(json.dumps(["cavity_full"]))
    (release / "vision_setup.json").write_text(json.dumps({"preprocessor": "product_agnostic",
        "preprocessing_profile": "preprocessing_profile.json", "flatfield_lut": "flatfield.npz"}))
    for name in ("yolov8_stage1_mla.elf", "yolov8_mpk.json", "preprocessing_profile.json", "flatfield.npz"):
        (release / name).write_bytes(b"asset")
    archive = export_release(store, release_id)
    runtime = VisionStore(tmp_path / "runtime")
    imported = import_release(runtime, archive)
    assert imported["release"] == release_id
    assert (runtime.root / "releases" / release_id / "flatfield.npz").is_file()
    assert not (runtime.root / "releases" / release_id / "reference.png").exists()


def test_release_package_rejects_malformed_manifest(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("package.json", json.dumps(["not", "a", "dict"]))
    with pytest.raises(ReleasePackageError):
        import_release(VisionStore(tmp_path / "runtime"), archive)


def test_config_put_cannot_write_activation_state(tmp_path: Path):
    with TestClient(create_app(VisionStore(tmp_path))) as client:
        assert client.put("/api/config", json={"camera_ip": "192.168.1.10"}).status_code == 200
        response = client.put("/api/config", json={"active_release": "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"})
        assert response.status_code == 400


def test_new_config_fields_are_validated():
    assert VisionConfig().camera_mode == "5fps"
    assert VisionConfig().qat_epochs == 0
    for bad in ({"camera_mode": "60fps"}, {"qat_epochs": -1}):
        with pytest.raises(ValueError):
            VisionConfig.from_dict(bad)


def test_count_semantics_have_one_implementation():
    from vision_sdk.inference.instance_detection import summarise as host_summarise
    from vision_sdk.preprocessing.state_counts import summarise

    assert host_summarise is summarise


def test_capture_fails_closed_when_board_runs_a_different_model(tmp_path: Path, monkeypatch):
    import types

    store = VisionStore(tmp_path)
    config = store.load()
    config.active_release = "release-one"
    store.save(config)
    state = {"release_id": "release-two"}

    class FakeLight:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeBoard:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @property
        def ready(self):
            return {"release_id": state["release_id"]}

        def capture(self):
            return types.SimpleNamespace(header={"detections": [], "image_jpeg": b"jpeg"}, image_jpeg=b"jpeg")

    monkeypatch.setattr("vision_sdk.runtime.BoardConnection", FakeBoard)
    monkeypatch.setattr("vision_sdk.runtime.LightSession", FakeLight)
    client = VisionClient(store)
    with pytest.raises(RuntimeError, match="different model version"):
        client.capture_and_inspect()

    state["release_id"] = "release-one"

    class CompletingBoard(FakeBoard):
        def capture(self):
            return types.SimpleNamespace(
                header={"detections": [{"class_name": "cavity_empty", "confidence": 0.9}],
                        "summary": {"count": 1}, "instance_count": 1, "source_width": 3200, "source_height": 2256},
                image_jpeg=b"jpeg")

    monkeypatch.setattr("vision_sdk.runtime.BoardConnection", CompletingBoard)
    result = client.capture_and_inspect()
    assert result["model_version"] == "release-one"
    assert result["summary"] == {"count": 1}
    assert result["detections"][0]["class_name"] == "cavity_empty"


def test_wait_ready_binds_the_class_list(tmp_path: Path, monkeypatch):
    store = VisionStore(tmp_path)
    config = store.load()

    class FakeBoard:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @property
        def ready(self):
            return {"release_id": "release-one", "classes": ["cavity_full", "cap_ok"]}

    monkeypatch.setattr("vision_sdk.release.BoardConnection", FakeBoard)
    ReleaseManager._wait_ready(config, "release-one", ["cavity_full", "cap_ok"])
    with pytest.raises(ReleaseError, match="different classes"):
        ReleaseManager._wait_ready(config, "release-one", ["cavity_full"])


def test_board_bundle_is_reverified_before_activation(tmp_path: Path, monkeypatch):
    store = VisionStore(tmp_path)
    store.ensure()
    release_id = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    release = store.root / "releases" / release_id
    (release / "board_bundle" / release_id).mkdir(parents=True)
    (release / "manifest.json").write_text(json.dumps({"version": release_id, "state": "staged_not_verified"}))
    (release / "board_bundle" / release_id / "bundle_manifest.json").write_text(json.dumps({"worker.py": "abc"}))
    manager = ReleaseManager(store)
    monkeypatch.setattr(ReleaseManager, "_ssh", staticmethod(lambda *_args, **_kwargs: "abc  worker.py"))
    manager._verify_board_bundle("sdk@board", release_id)
    monkeypatch.setattr(ReleaseManager, "_ssh", staticmethod(lambda *_args, **_kwargs: "deadbeef  worker.py"))
    with pytest.raises(ReleaseError, match="checksum"):
        manager._verify_board_bundle("sdk@board", release_id)


@pytest.mark.parametrize("candidate_ready", [False, True])
def test_activation_uses_readiness_without_requiring_a_sample(tmp_path: Path, monkeypatch, candidate_ready):
    import types

    store = VisionStore(tmp_path)
    store.ensure()
    previous, candidate = "aaaa10aa-0000-4000-8000-000000000001", "bbbb10bb-0000-4000-8000-000000000002"
    for release_id, state in ((previous, "board_verified"), (candidate, "staged_not_verified")):
        release = store.root / "releases" / release_id
        release.mkdir(parents=True)
        (release / "manifest.json").write_text(json.dumps({"version": release_id, "state": state, "classes": ["cavity_full"]}))
    config = store.load()
    config.active_release = previous
    store.save(config)

    board = {"running": previous, "shutdowns": 0}

    class FakeBoard:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @property
        def ready(self):
            return {"release_id": board["running"], "classes": ["cavity_full"]}

        def shutdown(self):
            board["shutdowns"] += 1
            board["running"] = None

        def capture(self):
            raise AssertionError("Activation must not require a part or sample capture")

    class FakeSocket:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def connect(self, _address):
            pass

        def getsockname(self):
            return ("192.168.1.17", 4444)

    launches = []
    waits = []

    def fake_wait(_config, release_id, _classes=None):
        waits.append(release_id)
        if release_id == candidate and not candidate_ready:
            raise ReleaseError("candidate never became ready")

    monkeypatch.setattr("vision_sdk.release.BoardConnection", FakeBoard)
    monkeypatch.setattr("vision_sdk.release.socket", types.SimpleNamespace(AF_INET=2, SOCK_DGRAM=2, socket=FakeSocket))
    monkeypatch.setattr(ReleaseManager, "_require_old_worker_stopped", staticmethod(lambda _config: None))
    monkeypatch.setattr(ReleaseManager, "_connection", staticmethod(lambda _config: ("sdk@board", "192.168.1.20")))
    monkeypatch.setattr(ReleaseManager, "_verify_board_bundle", lambda self, _target, _release_id: None)
    monkeypatch.setattr(ReleaseManager, "_ssh", lambda self, *_args, **_kwargs: "")
    monkeypatch.setattr(ReleaseManager, "_launch", lambda self, _target, _config, _admin_ip: launches.append(1))
    monkeypatch.setattr(ReleaseManager, "_wait_ready", staticmethod(fake_wait))

    if candidate_ready:
        result = ReleaseManager(store).activate(candidate)
        assert result["active_release"] == candidate
        assert store.load().active_release == candidate
        assert json.loads((store.root / "releases" / candidate / "manifest.json").read_text())["activation_check"] == "worker_ready_release_and_classes"
        assert launches == [1]
        assert waits == [candidate]
    else:
        with pytest.raises(ReleaseError, match="previous model restored"):
            ReleaseManager(store).activate(candidate)
        assert store.load().active_release == previous
        assert launches == [1, 1]
        assert waits == [candidate, previous]
    assert board["shutdowns"] == 1


def test_leak_free_groups_put_a_burst_and_a_duplicate_together():
    images = {
        "1": {"filename": "img_20261001_100000_0001.png", "sha256": "a"},
        "2": {"filename": "img_20261001_100005_0002.png", "sha256": "b"},
        "3": {"filename": "img_20261001_100020_0003.png", "sha256": "c"},
        "4": {"filename": "img_20261002_090000_0004.png", "sha256": "c"},
        "5": {"filename": "camera_roll_5.png", "sha256": "d"},
        "6": {"filename": "camera_roll_6.png", "sha256": "d"},
    }
    groups = sorted((sorted(group) for group in _leak_free_groups(images)))
    assert groups == [["1", "2", "3", "4"], ["5", "6"]]


def test_job_ownership_decides_interrupted_marking(tmp_path: Path, monkeypatch):
    store = VisionStore(tmp_path)
    store.ensure()
    alive_job, dead_job = "aaaa10aa-0000-4000-8000-000000000011", "aaaa10aa-0000-4000-8000-000000000012"
    for job_id, owner in ((alive_job, 424242), (dead_job, 424243)):
        folder = store.root / "jobs" / job_id
        folder.mkdir(parents=True)
        (folder / "status.json").write_text(json.dumps({"job_id": job_id, "phase": "training", "owner": owner}))
    monkeypatch.setattr("vision_sdk.training._process_alive", lambda pid: pid == 424242)
    TrainingManager(store)
    alive_status = json.loads((store.root / "jobs" / alive_job / "status.json").read_text())
    dead_status = json.loads((store.root / "jobs" / dead_job / "status.json").read_text())
    assert alive_status["phase"] == "training"
    assert dead_status["phase"] == "interrupted"


def test_compile_retry_rebuilds_without_retraining(tmp_path: Path, monkeypatch):
    store = VisionStore(tmp_path)
    store.ensure()
    job_id = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    job = store.root / "jobs" / job_id
    (job / "prepared" / "images" / "train").mkdir(parents=True)
    (job / "prepared" / "images" / "train" / "a.png").write_bytes(b"x")
    (job / "status.json").write_text(json.dumps({"job_id": job_id, "phase": "needs_board_compile"}))
    release = store.root / "releases" / job_id
    release.mkdir(parents=True)
    (release / "manifest.json").write_text(json.dumps({"version": job_id, "state": "trained_not_compiled"}))
    config = store.load()
    config.compiler_image = "sima/compiler:1"
    store.save(config)

    calls = []

    def working_compile(release_dir, prepared, image):
        calls.append(image)
        return {"compiler_image": image, "artifacts": {"yolov8_stage1_mla.elf": "abc"}}

    monkeypatch.setattr("vision_sdk.training.compile_candidate", working_compile)
    manager = TrainingManager(store)
    client = TestClient(create_app(store))
    response = client.post(f"/api/jobs/{job_id}/compile")
    assert response.status_code == 200
    deadline = time.time() + 15
    status = {}
    while time.time() < deadline:
        status = json.loads((job / "status.json").read_text())
        if status.get("phase") != "compiling":
            break
        time.sleep(0.05)
    assert status["phase"] == "needs_board_check"
    manifest = json.loads((release / "manifest.json").read_text())
    assert manifest["state"] == "compiled_not_board_verified"
    assert manifest["board_build"]["artifacts"] == {"yolov8_stage1_mla.elf": "abc"}
    assert calls == ["sima/compiler:1"]

    def failing_compile(release_dir, prepared, image):
        raise CompileError("docker said no")

    manifest["state"] = "trained_not_compiled"
    (release / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr("vision_sdk.training.compile_candidate", failing_compile)
    manager.start_compile(job_id)
    deadline = time.time() + 15
    while time.time() < deadline:
        status = json.loads((job / "status.json").read_text())
        if status.get("phase") != "compiling":
            break
        time.sleep(0.05)
    assert status["phase"] == "needs_board_compile"
    assert "docker said no" in (status.get("note") or "")


def test_review_image_route_refuses_traversal(tmp_path: Path):
    store = VisionStore(tmp_path)
    store.ensure()
    release_id = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    release = store.root / "releases" / release_id
    release.mkdir(parents=True)
    (release / "manifest.json").write_text(json.dumps({"version": release_id, "state": "board_verified",
        "classes": ["cavity_full"], "review_images": ["..\\..\\..\\secret.txt"]}))
    (tmp_path / "secret.txt").write_text("secret")
    with TestClient(create_app(store)) as client:
        response = client.get(f"/api/releases/{release_id}/review/..%5C..%5C..%5Csecret.txt")
        assert response.status_code == 404


def test_product_agnostic_prepare_emits_one_image_per_located_part(tmp_path: Path):
    """Serving conditions each part on its own, so training must too."""

    def rig(width: int = 1021, height: int = 720) -> "np.ndarray":
        ys, xs = np.mgrid[0:height, 0:width]
        radius = np.sqrt(((ys - height / 2) / (height / 2)) ** 2 + ((xs - width / 2) / (width / 2)) ** 2)
        base = np.interp(radius, [0.0, 0.2, 0.4, 0.6, 0.8, 1.6], [96, 88, 70, 48, 30, 20])
        frame = np.dstack([base, base, base]).astype(np.uint8)
        frame[: int(height * 0.15), :] = (150, 158, 170)
        return frame

    def with_parts(frame: "np.ndarray", boxes: list[tuple[int, int, int, int]]) -> "np.ndarray":
        out = frame.copy()
        for x, y, width, height in boxes:
            out[y : y + height, x : x + width] = 205
        return out

    layouts = [
        (1, [(120, 180, 140, 120), (600, 300, 150, 150)]),
        (2, [(200, 100, 130, 130), (700, 400, 160, 160)]),
    ]
    images, annotations = [], []
    annotation_id = 0
    with zipfile.ZipFile(tmp_path / "dataset.zip", "w") as archive:
        for image_id, boxes in layouts:
            frame = with_parts(rig(), boxes)
            ok, encoded = cv2.imencode(".png", frame)
            assert ok
            archive.writestr(f"images/{image_id}.png", encoded.tobytes())
            images.append({"id": image_id, "file_name": f"{image_id}.png", "width": 1021, "height": 720})
            for index, (x, y, width, height) in enumerate(boxes):
                annotation_id += 1
                annotations.append({"id": annotation_id, "image_id": image_id,
                                    "category_id": index + 1, "bbox": [x, y, width, height]})
        archive.writestr("annotations/instances.json", json.dumps({
            "images": images,
            "categories": [{"id": 1, "name": "cavity_full"}, {"id": 2, "name": "cavity_empty"}],
            "annotations": annotations,
        }))

    flatfield_path = save_flatfield(build_flatfield([rig() for _ in range(3)]), tmp_path / "flatfield.npz")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({"name": "general", "crop_mode": "auto",
                                        "stage_presence": True, "presence_threshold": 0.18}))
    import_coco_zip(tmp_path / "dataset.zip", tmp_path / "imported")
    config = VisionConfig(preprocessor="product_agnostic", preprocessing_profile=str(profile_path),
                          flatfield_lut=str(flatfield_path))
    report = prepare_training_dataset(tmp_path / "imported", tmp_path / "prepared", config)

    prepared = tmp_path / "prepared"
    produced = sorted((prepared / "images").glob("*/*.png"))
    assert len(produced) == 4, f"expected one image per located part, got {len(produced)}"
    for image_path in produced:
        split = image_path.parent.name
        lines = (prepared / "labels" / split / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1, f"{image_path.name} should carry exactly its own part label"
    assert report["counts"]["train"] == [1, 1]
    assert report["counts"]["val"] == [1, 1]


def test_start_refuses_a_release_that_is_not_board_verified(tmp_path: Path):
    store = VisionStore(tmp_path)
    store.ensure()
    release_id = "f4a3a74f-3093-4299-8c9d-1eaf19c39b62"
    release = store.root / "releases" / release_id
    release.mkdir(parents=True)
    (release / "manifest.json").write_text(json.dumps({"version": release_id, "state": "staged_not_verified"}))
    config = store.load()
    config.active_release = release_id
    store.save(config)
    with pytest.raises(ReleaseError, match="not a board-verified"):
        ReleaseManager(store).start_active_worker()


def test_export_six_heads_finds_heads_without_exact_node_names(tmp_path: Path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from vision_sdk.model import export_six_heads

    class_count = 4
    nodes, outputs, initializers = [], [], []
    for kind, prefix, channels in (("bbox", "cv2", 4), ("class_prob", "cv3", class_count)):
        for side, stride in ((80, 8), (40, 16), (20, 32)):
            sanitized = f"w_{prefix}_{side}"
            initializers.append(numpy_helper.from_array(np.zeros((channels, 3, 1, 1), dtype=np.float32), name=sanitized))
            node_name = f"backbone/{prefix}.{side}/{prefix}.{side}.2/Conv"
            nodes.append(helper.make_node("Conv", ["images", sanitized], [f"{prefix}_{side}"],
                                          name=node_name, strides=[stride, stride], pads=[0, 0, 0, 0]))
            outputs.append(helper.make_tensor_value_info(f"{prefix}_{side}", TensorProto.FLOAT, [1, channels, side, side]))
    input_tensor = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])
    model = helper.make_model(helper.make_graph(nodes, "synthetic", [input_tensor], outputs, initializer=initializers),
                              opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    source = tmp_path / "synthetic.onnx"
    onnx.save(model, source)
    destination = export_six_heads(source, tmp_path / "six.onnx", class_count)
    extracted = onnx.load(str(destination))
    names = {output.name: [dim.dim_value for dim in output.type.tensor_type.shape.dim]
             for output in extracted.graph.output}
    assert names["bbox_0"] == [1, 4, 80, 80]
    assert names["bbox_2"] == [1, 4, 20, 20]
    assert names["class_prob_1"] == [1, 4, 40, 40]
