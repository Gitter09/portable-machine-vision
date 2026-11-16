"""Compile the IC/terminal six-head export on the Admin PC's ModelSDK container.

Mirrors tools/compile_combined_on_admin.py: stage the graph, the worker and the
calibration frames on the Admin PC, run the quantize+compile inside the cached
SiMa image, then pull the ELF and MPK back and verify their hashes.

The input side is passed through to the worker, which otherwise assumes 640.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import paramiko

REPO = Path(__file__).resolve().parents[1]
IMAGE = "783709528641.dkr.ecr.us-west-2.amazonaws.com/vdp-cli/modelsdk:2.0.0_Palette_SDK_master_B240"
REMOTE_ROOT = "C:/Users/Admin/Desktop/ICTerminalBuilds"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def connect() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    known = Path.home() / ".ssh" / "known_hosts"
    if known.is_file():
        client.load_host_keys(str(known))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(os.environ.get("ADMIN_SSH_HOST", "172.21.30.222"),
                   username=os.environ.get("ADMIN_SSH_USER", "admin"),
                   password=os.environ["ADMIN_SSH_PASSWORD"],
                   look_for_keys=False, allow_agent=False, timeout=20)
    client.get_transport().set_keepalive(30)
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int = 120) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    result = stdout.read().decode("utf-8", errors="replace").strip()
    error = stderr.read().decode("utf-8", errors="replace").strip()
    if stdout.channel.recv_exit_status():
        raise RuntimeError(f"{command}: {error or result}")
    return result


def run_retry(client: paramiko.SSHClient, command: str, attempts: int = 6, timeout: int = 120) -> str:
    """Retry the calls Docker Desktop intermittently fails with a 500."""
    last = None
    for attempt in range(attempts):
        try:
            return run(client, command, timeout=timeout)
        except RuntimeError as error:
            last = error
            if "500 Internal Server Error" not in str(error):
                raise
            time.sleep(10 * (attempt + 1))
    raise last


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True, help="folder holding model_sima_6out.onnx")
    parser.add_argument("--graph", default="model_sima_6out.onnx")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--input-side", type=int, default=1280)
    parser.add_argument("--timeout-hours", type=float, default=3.0)
    arguments = parser.parse_args()

    graph = (arguments.export / arguments.graph).resolve()
    if not graph.is_file():
        raise SystemExit(f"missing graph: {graph}")
    candidates = sorted(arguments.calibration.glob("*.png"))
    if len(candidates) < 50:
        raise SystemExit(f"need at least 50 calibration images, found {len(candidates)}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job = f"ic_terminal_{digest(graph)[:12]}_{stamp}"
    remote = f"{REMOTE_ROOT}/{job}"
    container = f"demeter_{job}"
    local = arguments.export / "remote-builds" / job
    local.mkdir(parents=True, exist_ok=True)
    worker = REPO / "vision_sdk" / "vendor_compile_worker.py"

    client = connect()
    try:
        image = json.loads(run_retry(client, f'docker image inspect "{arguments.image}"'))[0]
        print("Compiler image:", image["Id"], flush=True)

        sftp = client.open_sftp()
        try:
            sftp.stat(REMOTE_ROOT)
        except FileNotFoundError:
            sftp.mkdir(REMOTE_ROOT)
        sftp.mkdir(remote)
        sftp.mkdir(remote + "/calibration")
        remote_graph = remote + "/model_sima_6out.onnx"
        sftp.put(str(graph), remote_graph)
        sftp.put(str(worker), remote + "/vendor_compile_worker.py")
        for index, source in enumerate(candidates, start=1):
            sftp.put(str(source), remote + "/calibration/" + source.name)
            if index % 50 == 0:
                print(f"Calibration transfer: {index}/{len(candidates)}", flush=True)
        with sftp.open(remote + "/run_compile.sh", "w") as handle:
            handle.write(
                "#!/bin/sh\n"
                f"export SIMA_INPUT_SIDE={arguments.input_side}\n"
                "python3 -u /work/vendor_compile_worker.py > /work/compile.log 2>&1\n"
                "status=$?\necho $status > /work/compile.exit\nexit $status\n"
            )

        transferred = run(client, f'powershell -NoProfile -Command "(Get-FileHash -LiteralPath \'{remote_graph}\' -Algorithm SHA256).Hash"')
        if transferred.lower() != digest(graph):
            raise RuntimeError("transferred graph hash differs from the local export")

        metadata = {
            "compiler_image": arguments.image, "compiler_image_id": image["Id"],
            "compiler_image_digests": image.get("RepoDigests", []), "host": "admin",
            "remote_directory": remote, "container": container,
            "source_onnx": arguments.graph, "source_onnx_sha256": digest(graph),
            "input_side": arguments.input_side,
            "calibration_images": len(candidates),
            "calibration": {source.name: digest(source) for source in candidates},
            "worker_sha256": digest(worker),
        }
        (local / "job.json").write_text(json.dumps(metadata, indent=2) + "\n")

        run_retry(client, f'docker run -d --name {container} "{arguments.image}" sleep 86400')
        run(client, f"docker exec {container} mkdir -p /work")
        graph_windows = remote_graph.replace("/", "\\")
        run(client, f'docker cp "{graph_windows}" {container}:/work/model_sima_6out.onnx')
        for name in ("vendor_compile_worker.py", "calibration", "run_compile.sh"):
            windows = (remote + "/" + name).replace("/", "\\")
            run(client, f'docker cp "{windows}" {container}:/work/{name}')
        run(client, f"docker exec -d {container} sh /work/run_compile.sh")
        print("Compilation started:", remote, flush=True)

        deadline = time.monotonic() + arguments.timeout_hours * 3600
        while time.monotonic() < deadline:
            try:
                data = run(client, f"docker exec {container} cat /work/compile.log")
                (local / "compile.log").write_text(data, encoding="utf-8")
                lines = data.splitlines()
                print("Progress:", " | ".join(lines[-3:])[-1400:], flush=True)
            except RuntimeError:
                pass
            status_text = run(client, f'docker exec {container} sh -c "if [ -f /work/compile.exit ]; then cat /work/compile.exit; else echo RUNNING; fi"')
            if status_text == "RUNNING":
                time.sleep(20)
                continue
            if int(status_text):
                raise RuntimeError(f"SiMa compiler exited {status_text}; see {local / 'compile.log'}")
            break
        else:
            raise TimeoutError(f"compilation still running after {arguments.timeout_hours} h: {container}")

        sftp.mkdir(remote + "/compiled")
        run(client, f"docker exec {container} ls /work/compiled/yolov8_stage1_mla.elf")
        windows_compiled = (remote + "/compiled").replace("/", "\\")
        run(client, f'docker cp {container}:/work/compiled/. "{windows_compiled}"')

        artifacts = {}
        for name in ("yolov8_stage1_mla.elf", "yolov8_mpk.json"):
            target = local / name
            sftp.get(remote + "/compiled/" + name, str(target))
            remote_hash = run(client, f'powershell -NoProfile -Command "(Get-FileHash -LiteralPath \'{remote}/compiled/{name}\' -Algorithm SHA256).Hash"')
            if target.stat().st_size == 0 or remote_hash.lower() != digest(target):
                raise RuntimeError(f"invalid or corrupted compiled artifact: {name}")
            artifacts[name] = digest(target)
        if (local / "yolov8_stage1_mla.elf").read_bytes()[:4] != b"\x7fELF":
            raise RuntimeError("compiled artifact does not have an ELF header")
        json.loads((local / "yolov8_mpk.json").read_text())

        metadata["artifacts"] = artifacts
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        for name in artifacts:
            (arguments.export / name).write_bytes((local / name).read_bytes())
        (arguments.export / "compilation.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (local / "job.json").write_text(json.dumps(metadata, indent=2) + "\n")
        sftp.close()
        run(client, f"docker stop {container}")
        print("SUCCESS: compiled ELF and MPK downloaded and hash verified.", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
