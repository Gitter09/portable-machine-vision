"""Record provenance for the compiled 1280 model."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from analysis_paths import DATA

EXPORT = DATA / "export-1280"
REPO = Path(r"C:\Users\DELL\Desktop\d-meter\demeter-systems-mv-full")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    calibration = sorted(path.name for path in (EXPORT / "calibration").glob("*.png"))
    record = {
        "schema_version": 1,
        "job": "ic_terminal_94d18f801d4f_20261112_071940",
        "compiler_image": "783709528641.dkr.ecr.us-west-2.amazonaws.com/vdp-cli/modelsdk:2.0.0_Palette_SDK_master_B240",
        "compiler_image_id": "sha256:c5f1ff46cb773403a0f223e828b3e7b46006321e34318d378aed5f16814adefd",
        "host": "admin-172.21.30.222",
        "source_onnx": "model_1280_sima_6out.onnx",
        "source_onnx_sha256": sha(EXPORT / "model_1280_sima_6out.onnx"),
        "fused_onnx_sha256": sha(EXPORT / "model_1280.onnx"),
        "worker_sha256": sha(REPO / "vision_sdk" / "vendor_compile_worker.py"),
        "input_side": 1280,
        "calibration_images": len(calibration),
        "calibration_manifest_sha256": hashlib.sha256(
            json.dumps(calibration).encode()).hexdigest(),
        "artifacts": {
            "yolov8_stage1_mla.elf": sha(EXPORT / "yolov8_stage1_mla.elf"),
            "yolov8_mpk.json": sha(EXPORT / "yolov8_mpk.json"),
        },
        "elf_bytes": (EXPORT / "yolov8_stage1_mla.elf").stat().st_size,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "note": "compile.exit was 0. A quoting bug in the driver's post-build guard "
                "wrongly reported a missing ELF; the artifacts were then fetched and "
                "hash-verified against the container.",
    }
    (EXPORT / "compilation.json").write_text(json.dumps(record, indent=2) + "\n")
    print("ELF sha256 :", record["artifacts"]["yolov8_stage1_mla.elf"])
    print("MPK sha256 :", record["artifacts"]["yolov8_mpk.json"])
    print("graph      :", record["source_onnx_sha256"])
    print("elf bytes  :", format(record["elf_bytes"], ","))


if __name__ == "__main__":
    main()
