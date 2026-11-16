"""Isolated SiMa release staging and guarded activation."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path

from .atomic import write_json
from .board import BoardConnection, BoardError
from .config import VisionConfig, VisionStore


BOARD_ROOT = "/home/sima/demeter-vision"


class ReleaseError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ReleaseManager:
    def __init__(self, store: VisionStore):
        self.store = store

    def _paths(self, release_id: str) -> tuple[Path, dict]:
        uuid.UUID(release_id)
        root = self.store.root / "releases" / release_id
        path = root / "manifest.json"
        if not path.is_file():
            raise ReleaseError("Model version does not exist")
        return root, json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _connection(config: VisionConfig) -> tuple[str, str]:
        try:
            ipaddress.IPv4Address(config.board_host)
            ipaddress.IPv4Address(config.camera_ip)
        except ValueError as error:
            raise ReleaseError("Enter valid IPv4 addresses for the board and camera") from error
        if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,30}", config.board_ssh_user):
            raise ReleaseError("Board SSH user is invalid")
        if not shutil.which("ssh") or not shutil.which("scp"):
            raise ReleaseError("OpenSSH client is required on this PC")
        return f"{config.board_ssh_user}@{config.board_host}", config.board_host

    @staticmethod
    def _run(argv: list[str], timeout: int = 120) -> str:
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ReleaseError(f"Board transfer did not finish: {error}") from error
        if result.returncode:
            raise ReleaseError((result.stderr or result.stdout or "Board command failed").strip()[:500])
        return result.stdout.strip()

    @staticmethod
    def _ssh(target: str, command: str, timeout: int = 120) -> str:
        return ReleaseManager._run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", target, command], timeout)

    def _verify_board_bundle(self, target: str, release_id: str) -> None:
        """Re-verify the staged board copy before it is allowed to run.

        Checksums ran when the release was staged, but the board copy can be
        replaced or corrupted afterwards. The symlink swap is the last point at
        which the bytes on the board can still be checked against the bundle.
        """
        root, _ = self._paths(release_id)
        manifest_path = root / "board_bundle" / release_id / "bundle_manifest.json"
        if not manifest_path.is_file():
            raise ReleaseError("This release has no staged bundle; send it to the board first")
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        remote = f"{BOARD_ROOT}/releases/{release_id}"
        for filename, digest in expected.items():
            actual = self._ssh(target, f"sha256sum {remote}/{filename}").split()[0]
            if actual != digest:
                raise ReleaseError(f"Board copy of {filename} failed checksum verification")

    @staticmethod
    def _worker_sources() -> tuple[Path, Path]:
        directory = Path(__file__).resolve().parent / "board_runtime"
        worker = directory / "persistent_board_worker.py"
        detector = directory / "live_camera_benchmark.py"
        if worker.is_file() and detector.is_file():
            return worker, detector
        raise ReleaseError("Board worker files are missing from this SDK installation")

    def prepare_bundle(self, release_id: str) -> Path:
        root, manifest = self._paths(release_id)
        if manifest.get("state") not in {"compiled_not_board_verified", "staged_not_verified", "board_verified"}:
            raise ReleaseError("Compile this model for SiMa before sending it to the board")
        bundle = root / "board_bundle" / release_id
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True, exist_ok=True)
        worker, detector = self._worker_sources()
        for source in (worker, detector, root / "yolov8_stage1_mla.elf", root / "yolov8_mpk.json",
                       root / "labels.json", root / "vision_setup.json",
                       Path(__file__).resolve().parent / "board_runtime" / "camera_lease.py",
                       Path(__file__).resolve().parent / "board_runtime" / "fault_evidence.py",
                       Path(__file__).resolve().parent / "board_runtime" / "triggered_camera.py"):
            if not source.is_file():
                raise ReleaseError(f"Release file is missing: {source.name}")
            shutil.copy2(source, bundle / source.name)
        reference = root / "reference.png"
        if reference.is_file():
            shutil.copy2(reference, bundle / "reference.png")
        from .preprocessing import fixed_ref_prep

        source_dir = Path(fixed_ref_prep.__file__).parent
        target_dir = bundle / "preprocessing"
        target_dir.mkdir(exist_ok=True)
        for source in source_dir.glob("*.py"):
            shutil.copy2(source, target_dir / source.name)
        # A preprocessing profile and a flat-field map are data, not code, so the
        # *.py glob above does not pick them up. The worker resolves them relative
        # to the release root, so they are copied under exactly the names that
        # vision_setup.json records; an absolute trainer-PC path would not exist
        # on the board.
        setup = json.loads((root / "vision_setup.json").read_text(encoding="utf-8"))
        for key in ("preprocessing_profile", "flatfield_lut"):
            name = setup.get(key)
            if not name:
                continue
            asset = root / name
            if not asset.is_file():
                raise ReleaseError(f"Preprocessing asset is missing: {name}")
            shutil.copy2(asset, bundle / asset.name)
        (bundle / "vision_release.json").write_text(json.dumps({"release_id": release_id, "classes": manifest["classes"]}) + "\n", encoding="utf-8")
        hashes = {str(path.relative_to(bundle)).replace("\\", "/"): _sha256(path)
                  for path in bundle.rglob("*") if path.is_file()}
        (bundle / "bundle_manifest.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
        return bundle

    def stage(self, release_id: str) -> dict:
        config = self.store.load()
        target, _ = self._connection(config)
        root, manifest = self._paths(release_id)
        bundle = self.prepare_bundle(release_id)
        remote_release = f"{BOARD_ROOT}/releases/{release_id}"
        if config.active_release == release_id:
            raise ReleaseError("The active SDK release cannot be replaced while it is running")
        self._ssh(target, f"mkdir -p {BOARD_ROOT}/releases")
        self._run(["scp", "-q", "-r", "-o", "BatchMode=yes", str(bundle), f"{target}:{BOARD_ROOT}/releases/"], 300)
        expected = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
        for filename in expected:
            actual = self._ssh(target, f"sha256sum {remote_release}/{filename}").split()[0]
            if actual != expected[filename]:
                raise ReleaseError(f"Board copy of {filename} failed checksum verification")
        manifest["state"] = "staged_not_verified"
        write_json(root / "manifest.json", manifest)
        return {"release": release_id, "state": "staged_not_verified"}

    def _launch(self, target: str, config: VisionConfig, admin_ip: str) -> None:
        current = f"{BOARD_ROOT}/current"
        command = (f"( cd {current} && exec setsid nohup python3 persistent_board_worker.py "
                   f"--root {current} --bind {config.board_host} --admin-ip {admin_ip} "
                   f"--camera {config.camera_ip} --camera-mode {config.camera_mode} "
                   f"--port {config.board_port} --lease-port {config.board_port + 1} ) "
                   f"> {BOARD_ROOT}/worker.log 2>&1 < /dev/null &")
        self._ssh(target, command, timeout=20)

    @staticmethod
    def _require_port_free(config: VisionConfig, port: int) -> None:
        target, _ = ReleaseManager._connection(config)
        command = (f"listeners=$(ss -H -ltn 'sport = :{port}') || exit 2; "
                   'test -z "$listeners"')
        try:
            ReleaseManager._ssh(target, command, timeout=15)
        except ReleaseError as error:
            raise ReleaseError(f"Board port {port} is occupied or its status could not be checked over SSH: {error}") from error

    @staticmethod
    def _require_old_worker_stopped(config: VisionConfig) -> None:
        try:
            with socket.create_connection((config.board_host, 5004), timeout=2):
                raise ReleaseError("The existing machine worker is still running. Close the machine application and wait for it to stop before activating the SDK")
        except ConnectionRefusedError:
            return
        except TimeoutError:
            ReleaseManager._require_port_free(config, 5004)
        except OSError as error:
            raise ReleaseError(f"Could not verify that the existing machine worker is stopped: {error}") from error

    def stop_active_worker(self) -> dict:
        """Stop only a release known to this SDK installation, never a foreign worker."""
        config = self.store.load()
        known = set()
        for path in (self.store.root / "releases").glob("*/manifest.json"):
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("state") in {"staged_not_verified", "board_verified"}:
                    known.add(path.parent.name)
            except (OSError, ValueError):
                continue
        if not known:
            return {"stopped": True, "worker_was_running": False}
        try:
            with BoardConnection(config.board_host, config.board_port, timeout=15) as board:
                if board.ready.get("release_id") not in known:
                    raise ReleaseError("A different worker is on the SDK port; it was not stopped")
                board.shutdown()
        except ConnectionRefusedError:
            return {"stopped": True, "worker_was_running": False}
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((config.board_host, config.board_port), timeout=1):
                    time.sleep(0.25)
            except ConnectionRefusedError:
                return {"stopped": True, "worker_was_running": True}
            except OSError as error:
                raise ReleaseError(f"Cannot confirm that the SDK worker released the camera: {error}") from error
        raise ReleaseError("SDK worker did not stop within 15 seconds; do not start the existing application yet")

    def start_active_worker(self) -> dict:
        """Resume the last verified SDK model after returning from the old app."""
        config = self.store.load()
        if not config.active_release:
            raise ReleaseError("Activate a verified SDK model before starting inspection")
        _, manifest = self._paths(config.active_release)
        if manifest.get("state") != "board_verified":
            raise ReleaseError("The active model is not a board-verified release; activate it again")
        self._require_old_worker_stopped(config)
        try:
            with BoardConnection(config.board_host, config.board_port, timeout=3) as board:
                if board.ready.get("release_id") != config.active_release:
                    raise ReleaseError("A different worker is on the SDK port; it was not replaced")
                return {"started": True, "already_running": True, "release": config.active_release}
        except ConnectionRefusedError:
            pass
        except TimeoutError:
            self._require_port_free(config, config.board_port)
        target, _ = self._connection(config)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect((config.board_host, 22))
                admin_ip = probe.getsockname()[0]
        except OSError as error:
            raise ReleaseError(f"Cannot determine this PC's board-network address: {error}") from error
        self._ssh(target, f"ln -s releases/{config.active_release} {BOARD_ROOT}/current.next && mv -Tf {BOARD_ROOT}/current.next {BOARD_ROOT}/current")
        self._launch(target, config, admin_ip)
        self._wait_ready(config, config.active_release, manifest.get("classes"))
        return {"started": True, "already_running": False, "release": config.active_release}

    @staticmethod
    def _wait_ready(config: VisionConfig, release_id: str, classes: list[str] | None = None) -> None:
        deadline = time.monotonic() + 90
        last = "no response"
        while time.monotonic() < deadline:
            try:
                with BoardConnection(config.board_host, config.board_port, timeout=3) as board:
                    if board.ready.get("release_id") != release_id:
                        raise ReleaseError("The board started a different model version")
                    if classes is not None and list(board.ready.get("classes") or []) != list(classes):
                        raise ReleaseError(
                            "The board worker reports different classes than this release; the compiled "
                            "model does not match its labels"
                        )
                    return
            except (OSError, BoardError) as error:
                last = str(error)
                time.sleep(1)
        raise ReleaseError(f"New board worker did not become ready: {last}")

    def activate(self, release_id: str) -> dict:
        root, manifest = self._paths(release_id)
        if manifest.get("state") not in {"staged_not_verified", "board_verified"}:
            raise ReleaseError("Send this candidate to the board before activation")
        config = self.store.load()
        self._require_old_worker_stopped(config)
        target, _ = self._connection(config)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect((config.board_host, 22))
                admin_ip = probe.getsockname()[0]
        except OSError as error:
            raise ReleaseError(f"Cannot determine this PC's board-network address: {error}") from error
        # Verify the bytes on the board before anything is stopped, so a corrupted
        # or replaced staged copy cannot become current.
        self._verify_board_bundle(target, release_id)
        previous = config.active_release
        try:
            with BoardConnection(config.board_host, config.board_port, timeout=15) as board:
                if not previous or board.ready.get("release_id") != previous:
                    raise ReleaseError("Another application owns the board camera. Stop it before activating this SDK model")
                board.shutdown()
        except ConnectionRefusedError:
            if previous:
                raise ReleaseError("The previous SDK worker is missing; check board state before activation")
        except TimeoutError:
            self._require_port_free(config, config.board_port)
        # The lease heartbeat carries config.active_release and the new worker only
        # renews on its own id, so point the config at the candidate before the
        # worker starts; a slow activation then cannot let the lease expire.
        # Failure paths restore it in _rollback_activation.
        config.active_release = release_id
        self.store.save(config)
        try:
            self._ssh(target, f"ln -s releases/{release_id} {BOARD_ROOT}/current.next && mv -Tf {BOARD_ROOT}/current.next {BOARD_ROOT}/current")
            self._launch(target, config, admin_ip)
            self._wait_ready(config, release_id, manifest.get("classes"))
        except Exception as error:
            try:
                self._rollback_activation(target, config, previous, admin_ip, release_id)
            except ReleaseError as rollback_error:
                raise ReleaseError(f"Board check failed and the previous worker did not restart: {rollback_error}") from error
            if previous:
                raise ReleaseError(f"Board check failed; previous model restored: {error}") from error
            raise ReleaseError(f"Board check failed; no previous SDK model was available: {error}") from error
        config.previous_release = previous
        self.store.save(config)
        manifest["state"] = "board_verified"
        manifest["activation_check"] = "worker_ready_release_and_classes"
        write_json(root / "manifest.json", manifest)
        return {"active_release": release_id, "previous_release": previous}

    def _rollback_activation(self, target: str, config: VisionConfig, previous: str, admin_ip: str, failed_release: str) -> None:
        try:
            with BoardConnection(config.board_host, config.board_port, timeout=15) as board:
                if board.ready.get("release_id") == failed_release:
                    board.shutdown()
        except (OSError, BoardError):
            pass
        config.active_release = previous
        failure: ReleaseError | None = None
        try:
            if previous:
                self._ssh(target, f"ln -s releases/{previous} {BOARD_ROOT}/current.next && mv -Tf {BOARD_ROOT}/current.next {BOARD_ROOT}/current")
                self._launch(target, config, admin_ip)
                _, previous_manifest = self._paths(previous)
                self._wait_ready(config, previous, previous_manifest.get("classes"))
        except ReleaseError as error:
            failure = error
        finally:
            self.store.save(config)
        if failure is not None:
            raise failure
