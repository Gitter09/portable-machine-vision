"""Installation configuration and local state paths."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class VisionConfig:
    board_host: str = "192.168.1.20"
    board_port: int = 5005
    board_ssh_user: str = "sima"
    camera_ip: str = ""
    runtime_setup_complete: bool = False
    light_port: str = "COM1"
    reference_image: str = ""
    roi: tuple[int, int, int, int] | None = None
    base_weights: str = ""
    compiler_image: str = ""
    active_release: str = ""
    previous_release: str = ""
    # Preprocessing. `preprocessor` defaults to the screw rig's fixed-reference
    # path so an existing setup keeps behaving identically until this is changed
    # deliberately. See docs/decisions/log.md D-03 and D-10.
    preprocessor: str = "fixed_reference_realtime"
    preprocessing_profile: str = ""
    flatfield_lut: str = ""
    # Training. Parts are placed by hand, so frame orientation is arbitrary and the
    # model is trained to tolerate it. 180 is the honest range: a part upside down
    # is a placement that really happens, and neither cavities nor caps have an
    # upright that means anything. See docs/decisions/log.md D-30.
    rotation_degrees: float = 180.0
    # Quantization-aware fine-tuning epochs; 0 disables the QAT stage.
    qat_epochs: int = 0
    #: Camera transport for the SDK board worker: 5fps, 10fps-jumbo or
    #: 10fps-jumbo-trigger.
    camera_mode: str = "5fps"

    @classmethod
    def from_dict(cls, value: dict) -> "VisionConfig":
        if value.get("board_port") == 5004:
            value = {**value, "board_port": 5005}  # migrate pre-isolation SDK state
        allowed = set(cls.__dataclass_fields__)
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown setup fields: {', '.join(sorted(unknown))}")
        if value.get("roi") is not None:
            roi = tuple(int(n) for n in value["roi"])
            if len(roi) != 4 or roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0:
                raise ValueError("Inspection area must have x, y, width, and height")
            value = {**value, "roi": roi}
        config = cls(**value)
        if not 1 <= config.board_port <= 65534:
            raise ValueError("Board port is invalid")
        if not str(config.preprocessor).strip():
            raise ValueError("Preprocessor name cannot be empty")
        if config.camera_mode not in {"5fps", "10fps-jumbo", "10fps-jumbo-trigger"}:
            raise ValueError("Camera mode must be 5fps, 10fps-jumbo or 10fps-jumbo-trigger")
        if config.qat_epochs < 0:
            raise ValueError("QAT epochs cannot be negative")
        return config


class VisionStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.environ.get("DEMETER_VISION_HOME") or Path.home() / "DemeterVision").resolve()

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("datasets", "jobs", "releases", "references", "annotations"):
            (self.root / name).mkdir(exist_ok=True)

    def load(self) -> VisionConfig:
        path = self.root / "config.json"
        return VisionConfig.from_dict(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else VisionConfig()

    def save(self, config: VisionConfig) -> None:
        self.ensure()
        path = self.root / "config.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
