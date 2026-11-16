---
type: guide
title: The IC/Terminal Model
description: The four-class contract, dataset rules, and the geometry that makes it work.
status: active
tags: [dataset, labels, camera, framing]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/tools/ic_terminal_dataset.py
  - repo://portable-vision/openwiki/CHANGELOG.md
---

# The IC/terminal model

## Classes

Fixed order, never renumbered:

| id | label | What it is |
|---|---|---|
| 0 | `IC_Fail` | IC missing |
| 1 | `Terminal_pass` | Screw terminal present |
| 2 | `IC_Pass` | IC present |
| 3 | `Terminal_Fail` | Screw terminal missing |

**Owner-confirmed scope (2026-11-15): missing components only.** All fault
specimens are missing ICs and/or terminals; no solder, electrical or other defect
types are included. Keep the legacy label names and IDs unchanged. A `Pass`
label here does not certify joint quality or electrical function.

**Handover correction (2026-11-14):** v4-1280 does contain `Terminal_Fail`:
43 training, 11 validation and 10 test boxes, backed by the terminal-absent COCO
export. Class names/IDs match the training YAML and FP32 model metadata. This
historical absence example does not establish coverage of every terminal defect
or performance at the current camera position. See
[labelled evaluation](../operations/r33-labelled-evaluation-20261114.md).

Scenes containing `IC_Pass` and `Terminal_Fail` together are out of scope and
have not been captured.

## The plate and the camera

The plate turns under a fixed camera. One board per plate; a second board left on
the plate corrupts both annotation and framing measurements.

Camera facts that matter:

| Property | Value |
|---|---|
| Stream | `rtsp://192.168.1.10:5005/routecam` |
| Native size | 1280×720, H.264 |
| Reachable from | The board only |
| Focus | Manual, fixed-focus lens |

## Framing, as measured on 2026-11-13

| Quantity | Value |
|---|---|
| Board footprint on the plate | ~160–205 px |
| Orbit radius | ~270–290 px |
| Clipped frames, 40-frame sweep | **0** |
| Clearance: top / bottom | 11 px / 17 px |
| Clearance: left / right | 255 px / 364 px |

The sweep filled **96%** of the frame height at that checkpoint. The
2026-11-15 owner-confirmed good-part trial instead shows **actual top/bottom
clipping** with the current placement. Old clearance numbers must not be reused
for a newly positioned board. Move the specimen inward with the plate stopped,
then verify a full rotation; see [trial evidence](../operations/good-part-trial-20261115.md).
Treat the camera and plate mounts as fixed. If either moves, re-measure before capturing.

The camera was deliberately re-aimed rather than moved back, so the IC stayed at
roughly **24 px** at 1280 input — the size that made the from-scratch crop
experiment viable.

## Focus

Laplacian variance is a crude but useful proxy. Reference points from the same
scene:

| State | Sharpness |
|---|---|
| Training frames (old position) | 57.1 |
| Before adjustment | 42.7 |
| After a wrong-way turn | 26.7 |
| **Current, 40-frame median** | **50.0** |

Anything below ~40 means the lens was turned the wrong way. Judge on a median over
several seconds, not a single frame; a rotating plate changes the number with pose.

## Dataset rules

Defined by `tools/ic_terminal_dataset.py` and enforced by its tests:

1. **Split by capture session, never by frame.** Adjacent frames are correlated.
2. **Guard band between blocks.** Default 3 frames are dropped.
3. **Cover the revolution.** Every split must see every sector, or the model
   learns pose instead of defect.
4. **Quarantine known-bad frames** rather than deleting them silently; the report
   stays reproducible.
5. **Hashes are recorded** so a dataset can be traced back to its frames.

## Scale is a hard boundary

The following historical measurements at the changed distance predate the r33
layout repair. They were confounded by integration faults and do **not** by
themselves prove a current camera-transfer failure:

| Input size | Detections at the new distance |
|---|---|
| 1280 | 0 |
| 960 | 0 |
| 800 | 2 |
| 640 | 2 |

Moving the camera changes the input distribution. Hold it fixed for a controlled
known-specimen trial before deciding whether recapture/retraining is necessary.
Re-measuring framing first is cheaper than re-annotating. The historical labelled
test is not a substitute for that current-camera trial.

## Current model

- Architecture YOLO11n, trained at 1280 on `dataset/v4-1280`.
- Held-out: `IC_Fail` precision 1.000, recall 0.928; mAP50 0.977.
- That result belongs to the **old camera distance**. It is evidence the pipeline
  works, not evidence the model is deployable.
