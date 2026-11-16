---
type: operations
title: Experiment Log
description: What was tried on the portable board, in order, with numbers and verdicts.
status: active
tags: [experiments, history, decode]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T12:00:00Z
sources:
  - repo://portable-vision/openwiki/CHANGELOG.md
  - board://ic-terminal-verify-r24/detections.jsonl
---

# Experiment log

Chronological record of what was attempted on this board, so the same dead ends
are not re-entered. Verdicts only; mechanics live in the architecture pages.

## 2026-11-12 — packaging and the decoder contract

| Attempt | Result | Verdict |
|---|---|---|
| YOLO26n at 1280 | Head is 4-channel anchor-free; neither pass-through nor DFL | Rejected |
| YOLO11n at 1280, raw six heads | Trains and exports; parity to Ultralytics 0.0002 px | Adopted as the graph |
| In-graph `Sigmoid` or decode at 1280 | 0.6–0.78 fps | Rejected for speed |
| DFL on the board (`decode_type: yolov8`) | 27–30 fps | Adopted for speed, **decode correctness unproven** |
| Releases `r10`–`r19` | Staged, produced no usable boxes | Superseded |
| `boxdecoder.json` from the seed template | Runtime rejects `debug` as string, `cpu`, `graph_name`, `next_cpu` | Rejected; `r8`'s schema is the accepted one |

## 2026-11-13 — camera, then the geometry bug

### Camera work (this consumed most of the day)

| Measurement | Result |
|---|---|
| Board at old distance | ~21×15 px IC; miss risk |
| After moving closer | ~40×33 px IC; sweep clipped the top edge |
| Re-aim instead of backing off | 0/40 frames clipped, 11 px top / 17 px bottom clearance |
| Focus, wrong-way turn | 26.7 (training baseline 57.1) |
| Focus, best | 50.0 median over 40 frames |
| Camera-free decode harness (`filesrc` + `jpegdec`) | Fails: `meta not found` |
| Camera-free harness (`filesrc` + `h264parse`) | Fails: `input buffer is NULL` |
| Second board on the plate | Removed; it corrupted both framing and tracking |

Verdict: the camera is settled. Do not move it. There is **no** camera-free way to
run the decode chain; the preprocessing node requires decoder metadata.

### The MLA geometry bug

Found by diffing against release `r8`'s configs, then confirmed against the
compiler's `tuple_get_item_*` output shapes.

| Tensor | Compiler | Config had |
|---|---|---|
| bbox 160² | 64 ch | 4 ch |
| bbox 80² | 64 ch | 4 ch @ 160² |
| bbox 40² | 64 ch | 4 ch @ 80² |
| class 160²/80²/40² | 4 ch each | shifted down a level |

Buffer sizes did **not** match and nothing compared them: the unpack node declared
67,200 bytes of tensor geometry, the decoder declared 2,284,800, and the MLA buffer
is 2,688,000 (compiler padding). The run started anyway. Fixed in
`build_ic_terminal_configs.py`; a guard now refuses mismatched pairs.

Verdict: real bug, real improvement (uniform ~300 px squares → IC-sized boxes),
**not a fix** — the numbers stayed far from the host.

### Decode variants, all measured on release `r24`

| Variant | Frames | Boxes/frame | Max score | Negative h | Verdict |
|---|---|---|---|---|---|
| Standard (grouped, RGB, conf 0.05) | 1181 | 0.4 | 0.217 | 32.7% | baseline |
| Interleaved box/class order | 886 | 0.66 | 0.413 | 32.8% | inconclusive, not adopted |
| Class scales negated | 711 | 1.9 | 0.141 | **77%** | rejected |
| BGR instead of RGB | 734 | 0.21 | 0.127 | 35.8% | rejected (single class only) |

Also established: the two coarse class tensors are quantised with
`zero_point = 127`, so their sigmoid can never exceed **0.5**. Any host-vs-board
score comparison must account for that before calling the difference a defect.

### Vendor tool investigation

| Finding | Evidence |
|---|---|
| `tool-model-to-pipeline` does **not** generate the configs; it copies them from the compiler's temp directory | `pipeline/pipeline_create.py` |
| Its surgeon decodes DFL in-graph and emits 4-channel boxes + sigmoided classes | `surgeons/surgeon_yolov8.py` |
| Its own working sample is grouped (three box tensors, then three class tensors) with `input_* == slice_*` | `Wood_simaaisrc` configs |
| The tool's venv is bound to the Admin PC, so it must be run there | `pyvenv.cfg` |

Verdict: the vendor's canonical path is pre-decoded boxes with
`decode_type: "yolo"` — exactly what `r8` proved. Our DFL design is the outlier.

## 2026-11-14 — workspace split

The portable work moved to this workspace. See [CHANGELOG](../CHANGELOG.md) and
[MIGRATION.md](../../MIGRATION.md). No board behaviour changed. State repairs
during the move: restored the board's candidate configs from the rejected BGR
variant back to RGB, and collapsed three capture supervisors to one.

## Things that look like progress but are not

- A run that writes thousands of detections but ends `"state": "error"`.
- A board run judged without a host reference on the same scene.
- Any score comparison above 0.5 without accounting for the class zero points.
- A recapture before the decode path is settled.
