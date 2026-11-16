# Portable Vision Workspace

Self-contained workspace for the **portable Demeter machine-vision station**: a
SiMa Modalix board that inspects ICs and screw terminals on a rotating plate and
streams detections to a host.

This folder was split out of the fixed AOI repository on 2026-11-14. Everything
here runs without the fixed station: the board release layer, the board runtime,
the IC/terminal model pipeline, and the tests that cover them.

- **Start here:** [`openwiki/quickstart.md`](openwiki/quickstart.md)
- **Debugging case study (plain English):** [`PORTABLE_VISION_DEBUGGING_CASE_STUDY.md`](PORTABLE_VISION_DEBUGGING_CASE_STUDY.md)
- **Why files moved or were copied:** [`MIGRATION.md`](MIGRATION.md)
- **The fixed station, which is a different machine:** `../demeter-systems-mv-full/`

## Clone-only portable setup (Mac M3)

Runtime/model/native library and OFFLINE Mac/Windows dependency wheels now live
INSIDE `deployment/portable-v1/`. No sibling data/MODEL/fixed-station directory is
required for this launch path. No external artifact download or LFS needed.

```sh
python3.11 setup-host.py
./local/host-env/bin/python -B run-portable.py --board-host 192.168.1.20
```

Native arm64 Python3.11 is required. Setup/environment/output stay inside the
clone under ignored `local/`; SSH password is prompted, never committed. Board IP
is launch configuration, receiver auto-selected. Existing isolated switch and
compatible Mac Ethernet address needed. Actual Mac test pending.
[Setup details and platform prerequisites](deployment/README.md).

The Windows production selection below is historical/local to the original PC;
use the clone-only launcher above on Mac.

## Production-selected pipeline

The owner-approved operational default is **`portable-vision-production-20261116-r2`**:
~15–16 model inspections/sec with independent ~30FPS camera video, frame-matched
browser overlays, pose reuse, feature ROI and C++8 input packing.

```powershell
.\run-production.ps1
```

Selection: `config/production-pipeline.json`. Launches verified frozen package in
its dedicated environment, not mutable experiment sources. Sessions remain bounded
(default/max300s); no boot service or continuous supervisor added yet. Previous
fallbacks/r33 remain untouched. [Deployment, rollback and frontend handoff](openwiki/operations/production-pipeline-20261116.md).
Owner selection is not independent production-accuracy/latency qualification.

## Layout

```
portable-vision/
  runtime/launcher/   worker.py, support.py, launch.py, verify.py — the code the board runs
  vision_sdk/         fixed station's SDK (copied); its board_runtime/ is the screw rig
  tools/              IC/terminal model pipeline
  tools/analysis/     measurement and diagnostic scripts
  config/             labels and credentials
  tests/              tests, including the preserved buggy config fixture
  openwiki/           documentation for this workspace
```

## Quick start

```powershell
# use an environment that already has pytest, onnx, opencv, fastapi, paramiko
python -m pytest tests\test_ic_terminal_configs.py tests\test_ic_terminal_dataset.py -q
```

This workspace ships no virtual environment of its own; see
[`openwiki/reference/tests.md`](openwiki/reference/tests.md).

The board itself is reached over SSH at `172.21.30.120` (also `192.168.1.20`),
user `sima`. Releases are staged under `/media/nvme/portable-release-*`. See
[`openwiki/operations/deployment.md`](openwiki/operations/deployment.md).

## Live annotated view (diagnostic r33)

On the receiver workstation, with Paramiko and FFplay available:

```powershell
python tools/portable_live_view.py --seconds 600 --output ../IC_TERMINAL_TRAIN/live-review-01
```

Use a **new output basename each run**. Press **Q** in the viewer or **Ctrl+C**
in the terminal to stop both viewer and board job. Maximum duration is bounded;
this is not a production service. Confidence remains diagnostic **0.05**.

[Operator instructions and physical test protocol](openwiki/guides/live-view-and-labelled-check.md)
· [Historical labelled results and remaining roadblocks](openwiki/operations/r33-labelled-evaluation-20261114.md)
· [Live benchmark baseline and specimen-swap protocol](openwiki/operations/benchmark-baseline-20261115.md)

## What this workspace does not contain

The fixed station's operator HMI, PLC control loop, production inference service
and printed manual stay in `../demeter-systems-mv-full/`. The two systems share
the `vision_sdk` package by copy, not by import; see `MIGRATION.md` for why.
