---
type: reference
title: Tool Command Line
description: Flags for the four model-pipeline tools in this workspace.
status: active
tags: [cli, tooling, reference]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/tools/ic_terminal_dataset.py
  - repo://portable-vision/tools/export_modalix_heads.py
  - repo://portable-vision/tools/build_ic_terminal_configs.py
  - repo://portable-vision/tools/compile_ic_terminal_on_admin.py
---

# Tool command line

All four tools are run as modules from this workspace root. Paths below use the
workspace-relative form; the training data lives outside the repository, usually
at `C:\Users\<user>\Desktop\d-meter\IC_TERMINAL_TRAIN`.

## `tools/ic_terminal_dataset.py`

Builds the four-class dataset from capture runs.

| Flag | Meaning |
|---|---|
| `--run NAME=DIR` | One capture run; repeat per run. Required |
| `--output DIR` | Dataset root. Required |
| `--mode {crop,letterbox}` | Transform. Default `crop` |
| `--crop-size N` | Square window side, crop mode |
| `--crop-centre X,Y` | Plate axis, crop mode |
| `--letterbox-size N` | Model square, letterbox mode. Default 640 |
| `--crop-margin F` | Window margin factor. Default 1.10 |
| `--quarantine RUN:FRAME` | Drop a known-bad frame; repeat as needed |
| `--blocks-per-run N` | Split granularity. Default 6 |
| `--guard N` | Frames dropped between blocks. Default 3 |
| `--force` | Overwrite an existing output directory |

## `tools/export_modalix_heads.py`

Cuts a raw six-head ONNX into the decoded or pre-sigmoid form the board consumes.

| Flag | Meaning |
|---|---|
| `--raw PATH` | Six-head ONNX with raw heads. Required |
| `--output PATH` | Written graph. Required |
| `--model-size N` | Model square. Default 1280 |
| `--classes N` | Class count. Default 4 |

## `tools/build_ic_terminal_configs.py`

Derives the three board configs from the compiler's MPK.

| Flag | Meaning |
|---|---|
| `--mpk PATH` | `yolov8_mpk.json`. Required |
| `--template-dir DIR` | Base JSONs. Required |
| `--output-dir DIR` | Where the three configs are written. Required |
| `--model-size N` | Default 1280; validated against the MPK |
| `--source WxH` | Camera frame size. Default `1280x720` |
| `--classes N` | Default 4 |
| `--labels-file PATH` | One label per line, class order. Required |
| `--elf-name NAME` | Default `yolov8_stage1_mla.elf` |
| `--threshold F` | Default 0.25 |
| `--nms-iou F` | Default 0.45 |
| `--topk N` | Default 100 |
| `--decode-type TYPE` | Overrides the template; `yolov8` for DFL |
| `--box-channels N` | 64 for a DFL head, 4 for decoded outputs |
| `--sigmoid {true,false}` | Sets `sigmoid_on_probabilities` |

The tool exits non-zero rather than writing a config pair whose geometry
disagrees, whose grids contradict `--model-size`, or whose compiler input/output
tile metadata is missing. Input/output tiles are derived from the MPK;
`input_*` logical frame geometry is distinct from physical `slice_*` geometry.

## `tools/compile_ic_terminal_on_admin.py`

Runs the ModelSDK container on the Admin PC and pulls the artefacts back.

| Flag | Meaning |
|---|---|
| `--export DIR` | Folder holding the graph and receiving the artefacts. Required |
| `--graph NAME` | Graph filename. Default `model_sima_6out.onnx` |
| `--calibration DIR` | Folder of calibration images; at least 50. Required |
| `--image TAG` | Compiler image |
| `--input-side N` | Model square passed to the worker. Default 1280 |
| `--timeout-hours F` | Abort threshold. Default 3.0 |

Credentials come from the environment: `ADMIN_SSH_PASSWORD`, optionally
`ADMIN_SSH_HOST` and `ADMIN_SSH_USER`.

## `tools/portable_live_view.py`

Host-side pinned r33 viewer. Required `--output DIR` must be new locally and have
a new basename on the board. `--seconds F` defaults to 600, finite and positive.
`--headless --frames N` uses bounded FFmpeg decode (default 900) instead of
FFplay. Optional `--release` and `--credentials` select explicit paths; defaults
are pinned r33 and this repository's credential file. Never put passwords on the
command line. Q/player exit or Ctrl+C stops its own board launcher; no auto-restart
or service installation. See [operator guide](../guides/live-view-and-labelled-check.md)
for dependencies, evidence and failure handling.

## `tools/analysis/benchmark_live_session.py`

Read-only observer of an existing session; never starts/stops the pipeline.
Required: `--session <host .session.json>`, `--case <confirmed-state>`,
`--expected <canonical-labels...>`, `--output <new-directory>`.
Defaults: `--seconds 120`, `--interval 2`; finite seconds >= interval >=0.25.
Optional `--credentials` defaults to this repo's credential file. CPU/health/
viewer sampling and exact encoder/detection cohort scoring are saved with raw
log windows. Keep specimen unchanged; confirm swaps before each measurement.
See [baseline results and limits](../operations/benchmark-baseline-20261115.md).

## Exit behaviour

All four tools return non-zero on contract violations and print the reason to
stderr. None of them writes a partial artefact before validating its inputs.
