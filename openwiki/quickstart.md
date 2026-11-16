---
type: guide
title: Portable Station Quickstart
description: What this workspace is, how to run it, and what to read before changing it.
status: active
tags: [quickstart, orientation]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/vision_sdk/release.py
  - repo://portable-vision/tools/build_ic_terminal_configs.py
  - repo://portable-vision/tools/compile_ic_terminal_on_admin.py
---

# Portable station quickstart

## What this system is

A single SiMa Modalix board inspecting ICs and screw terminals on a plate that
turns under a fixed camera. The board decodes detections and reports them; it
does not decide pass or fail. The host decides.

Four classes, fixed order:

```
0 IC_Fail
1 Terminal_pass
2 IC_Pass
3 Terminal_Fail
```

**Fault scope: missing components only**, confirmed by the owner. `IC_Fail`
and `Terminal_Fail` mean missing IC and missing terminal; the pass classes mean
present. Legacy class names/IDs stay fixed; solder/electrical defects are not
being evaluated.

A dataset audit confirmed historical `Terminal_Fail` annotations (43 training,
11 validation, 10 test boxes), correcting an earlier handover error. Keep class
IDs fixed. Historical labels do not establish current-camera acceptance.

## Current operational default

Owner-selected release is **portable-vision-production-20261116-r2**, started with
repo `run-production.ps1`. `config/production-pipeline.json` pins the frozen package,
native library and dedicated environment. It uses coordinate/browser + pose reuse
+ feature ROI + C++8, not old full-frame r33. Default/max300s sessions remain until
frontend lifecycle work; no autostart service added. Old releases/fallbacks intact.
[Production deployment and limits](operations/production-pipeline-20261116.md).

## The pieces

| Piece | Lives in | Role |
|---|---|---|
| Model tooling | `tools/` | Build the dataset, cut the six-head export, derive board configs, drive the compiler |
| SDK | `vision_sdk/` | Release manager, board worker, compiler driver, preprocessing contracts |
| Board | `172.21.30.120` | Runs staged releases under `/media/nvme/portable-release-*` |
| Camera | `rtsp://192.168.1.10:5005/routecam` | 1280×720 H.264; reachable only from the board |
| Compiler | Admin PC `172.21.30.222` | ModelSDK 2.0.0 container, started through the `DockerStart` task |

## The pipeline in one line per step

```
frames → tools/ic_terminal_dataset.py → train (Ultralytics) → tools/export_modalix_heads.py
      → tools/build_ic_terminal_configs.py → tools/compile_ic_terminal_on_admin.py
      → stage on board → run → detections.jsonl
```

Each step and its contracts: [model-pipeline](architecture/model-pipeline.md).

## Run the tests

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

They need `fastapi`, `httpx`, `opencv-python`, `numpy`, `onnx`, `paramiko`. The
workspace does not vendor a virtual environment; point `-m pytest` at one that
has them.

What the tests do and do not prove: [reference/tests](reference/tests.md).

## Before you touch the board

**Read [operations/status](operations/status.md) first.** It records what is staged,
what is running, and which configs on the board are which — state that is not in the
code.

Two facts cause most wasted hours:

1. **The camera and the inference pipeline cannot run at once.** Both need the
   hardware decoder. Stop the capture feed, run the pipeline, restart the feed.
2. **Release hashes are checked at run time.** If you rewrite a config after
   staging, the run aborts with `release asset hash differs`. Re-stage.

Deployment procedure: [operations/deployment](operations/deployment.md).

## Current state, honestly

Release **r33** now sends annotated 1280×720 video to this workstation at about
29.5 fps. Captured native detections agree with independent Python decoding after
repairing input and output tile layouts. This is a working **diagnostic** stream
at confidence 0.05, not a labelled-accuracy or production acceptance sign-off.
For repeatable start/stop and the known-good/bad trial protocol, use
[live viewing](guides/live-view-and-labelled-check.md). A 31-image historical
labelled test is complete. The first live trial found clipping; the owner
repositioned the PCB and visually accepted framing. A read-only two-minute
[good-part baseline](operations/benchmark-baseline-20261115.md) now measures
**29.98 fps**, but prediction consistency and viewer drops remain poor/unqualified.
The bad-IC/good-terminal baseline also sustains **30.01 fps**, but reports an
unwanted IC_Pass candidate in **11.2%** of frames (not a false-acceptance rate).
The owner-confirmed **IC-missing + terminal-missing** case is also captured at
**29.98 fps**; it still emits IC_Pass candidates in **29.2%** of frames. Three
available board states are now recorded; these are not site-level accuracy scores. Legacy candidate configs still need regeneration with the corrected
tool; do not reuse their tile settings. Read [current status](operations/status.md) for the
release, viewing clip, evidence and remaining quality/shutdown work.
