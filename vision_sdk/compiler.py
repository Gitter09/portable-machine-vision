"""Invoke the provisioned SiMa compiler with an immutable candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


class CompileError(RuntimeError):
    pass


def compile_candidate(release: Path, prepared: Path, image_name: str) -> dict:
    if not image_name:
        raise CompileError("A provisioned SiMa compiler image is required")
    if not shutil.which("docker"):
        raise CompileError("Docker is required to run the provisioned SiMa compiler")
    if not (release / "model_sima_6out.onnx").is_file():
        raise CompileError("The six-head ONNX graph is missing from this release")
    candidates = sorted((prepared / "images" / "train").glob("*.png"))
    if not candidates:
        raise CompileError("No training images are available for SiMa calibration")
    # Evenly spaced across the sorted set: lexical truncation would always take
    # image_1, image_10, image_100... and skew calibration toward one batch.
    limit = 200
    if len(candidates) > limit:
        step = len(candidates) / float(limit)
        calibration = [candidates[int(index * step)] for index in range(limit)]
    else:
        calibration = candidates
    if len(calibration) < 50:
        raise CompileError("At least 50 usable training images are required for SiMa calibration")
    work = release / "board_build"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()
    (work / "calibration").mkdir()
    shutil.copy2(release / "model_sima_6out.onnx", work / "model_sima_6out.onnx")
    shutil.copy2(Path(__file__).with_name("vendor_compile_worker.py"), work / "vendor_compile_worker.py")
    for image in calibration:
        shutil.copy2(image, work / "calibration" / image.name)
    command = ["docker", "run", "--rm", "--mount", f"type=bind,source={work.resolve()},target=/work",
               image_name, "python3", "/work/vendor_compile_worker.py"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=7200, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CompileError(f"SiMa compiler did not finish: {error}") from error
    (work / "compile.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise CompileError(f"SiMa compiler failed. See {work / 'compile.log'}")
    compiled = work / "compiled"
    required = [compiled / "yolov8_stage1_mla.elf", compiled / "yolov8_mpk.json"]
    if any(not path.is_file() for path in required):
        raise CompileError("SiMa compiler output is incomplete")
    artifacts = {}
    for path in required:
        target = release / path.name
        shutil.copy2(path, target)
        artifacts[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"compiler_image": image_name, "artifacts": artifacts, "calibration_images": len(calibration)}
