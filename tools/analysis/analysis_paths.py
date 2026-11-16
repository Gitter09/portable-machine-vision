"""Shared data root for the analysis scripts.

The training data lives outside the workspace, beside it:

    d-meter/IC_TERMINAL_TRAIN/     <- datasets, runs, exports, candidate tree
    d-meter/portable-vision/       <- this workspace

Set ``IC_TERMINAL_TRAIN`` to override the location. Everything that used to be a
hard-coded absolute path now goes through this module, so the scripts run from
anywhere and on any account.
"""

from __future__ import annotations

import os
from pathlib import Path

# tools/analysis/analysis_paths.py -> tools/analysis -> tools -> portable-vision -> d-meter
_DEFAULT = Path(__file__).resolve().parents[3] / "IC_TERMINAL_TRAIN"

DATA = Path(os.environ.get("IC_TERMINAL_TRAIN", _DEFAULT)).resolve()

#: Where the compiled artefacts and the four-label file live.
EXPORT = DATA / "export-yolo11"
CANDIDATE = DATA / "candidate"
DATASET = DATA / "dataset"
RUNS = DATA / "runs"
LABELS = Path(__file__).resolve().parents[2] / "config" / "labels-ic-terminal.txt"

#: Board HTTP endpoints used by the measurement scripts.
BOARD_FEED = "http://172.21.30.120:8080/frame.jpg"
BOARD_PAGE = "http://172.21.30.120:8080/"

if __name__ == "__main__":
    print(f"data root : {DATA}  (exists: {DATA.is_dir()})")
    print(f"export    : {EXPORT}")
    print(f"candidate : {CANDIDATE}")
    print(f"labels    : {LABELS}  (exists: {LABELS.is_file()})")
