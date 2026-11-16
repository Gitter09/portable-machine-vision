---
type: architecture
title: Model Pipeline
description: From captured frames to a staged board release, and the contract at each hand-off.
status: active
tags: [dataset, training, export, compile, pipeline]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/tools/ic_terminal_dataset.py
  - repo://portable-vision/tools/export_modalix_heads.py
  - repo://portable-vision/tools/build_ic_terminal_configs.py
  - repo://portable-vision/tools/compile_ic_terminal_on_admin.py
---

# Model pipeline

```
capture runs
   │  tools/ic_terminal_dataset.py
   ▼
dataset/  (YOLO layout, session-aware split)
   │  Ultralytics training
   ▼
best.pt
   │  ultralytics export → six raw heads
   ▼
yolo11n_six.onnx
   │  tools/export_modalix_heads.py  (optional: decoded variant)
   ▼
yolo11n_modalix_6out.onnx
   │  tools/compile_ic_terminal_on_admin.py  (ModelSDK container)
   ▼
yolov8_stage1_mla.elf + yolov8_mpk.json
   │  tools/build_ic_terminal_configs.py
   ▼
0_preproc.json, 0_process_mla.json, boxdecoder.json
   │  vision_sdk release manager
   ▼
staged board release → detections.jsonl
```

## 1. Dataset

`tools/ic_terminal_dataset.py` builds the four-class dataset from capture runs.

- **Split by session, not by frame.** Frames from one capture run are correlated;
  splitting inside a run leaks.
- **Guard band.** `--guard` frames are dropped between blocks so adjacent frames
  cannot straddle splits.
- **Two transforms.** `letterbox` halves the source resolution to the model
  square; `crop` takes a measured square window around the plate axis.
- **Coverage reporting.** The tool reports per-split coverage of every run and
  full-revolution sector coverage, and hashes inputs.

Rules and exclusions for the label set: [ic-terminal-model](../guides/ic-terminal-model.md).

## 2. Training

Ultralytics, no scale augmentation. The practical consequence, learned the hard
way: a model trained at one camera distance does not transfer to another. Moving
the camera from its training distance is a dataset invalidation event, not a
retuning event.

### Cropped-input candidate (draft annotation stage)

The frozen colour-seed/local-Canny crop preserves all reviewed targets on 42
current development images, but the old FP32 detector loses IC detections on
those enlarged inputs. Owner requested [cropped training-box proposals](../guides/cropped-training-box-review.md)
for manual review before training. This is a separate candidate pipeline, not a
change to the historical dataset builder or running r33 preprocessing.

Projected site polygons are only training-box seeds; present body/pins versus
empty footprint/pads need explicit review. All 84 boxes start pending. The current
batch has only two present examples per pass class; do not silently random-split
these correlated images or train from stale draft YOLO files. No final YAML/split
or training is created by the review builder. After review, fill coverage and
finalize labels; keep crop/resize/normalization consistent across candidate
training, calibration and serving. Existing weights/releases remain untouched.

### Class coverage and augmentation ordering

Owner accepted the first 42 crop-box proposals; a [second historical TRAIN-only
review batch](../guides/training-batch02-and-augmentation.md) adds 32 both-present
and 16 IC-missing/terminal-present images, all initially pending. Potential pool
is 90 correlated originals (class counts 56/50/34/40), not independent validation.
Historical validation/test membership and files remain untouched.

Finalize original labels and capture/specimen-aware splits BEFORE augmentation.
Stochastic, label-aware augmentation belongs only on training inputs (preferably
on the fly); validation/test and calibration use deterministic serving-compatible
crop/resize/normalization. No extra CLAHE/sharpening or numeric augmentation
schedule selected yet. Old `train_yolo11.py` still targets v4 and disables synthetic
geometry; a candidate needs a separate explicit dataset/config/run, not a blind
rerun or overwrite of existing weights.

### Prepared train-only crop pool (2026-11-15)

Owner completed batch02 box review; [preprocessing/augmentation preparation](../guides/prepared-training-pool-20261115.md)
now contains 90 approved parents assigned TRAIN, 450 derived 1280-square PNGs and
900 boxes. Four extras per base: 180°, exposure, mild contained rotation/scale,
mild model-space blur. Labels move with pixels; no clipping/drop. Preprocessing
matches candidate crop/RGB/resize/pad114/normalization contract; PNGs are uint8,
loader applies RGB/255 once. No CLAHE/sharpening added. Baked pool requires extra
online augmentation disabled, not duplicated by old trainer defaults.

No validation/test or final YAML is fabricated; training_ready=false pending
separate reviewed validation/evaluation plan. Do not turn these development
parents/siblings into validation or alias train as val. No optimizer, weight,
calibration or release change. Current r33 runtime remains full-frame and would
need separately validated matching crop serving before deploying a crop model.

## 3. Export

The plain Ultralytics export ends in a single `[1, 84, N]` output. The board
needs six.

`tools/export_modalix_heads.py` cuts the graph at the head and rewrites it to emit:

```
bbox_0..2      [1, 64, H/8, H/16, H/32]   raw DFL bins, 4 sides × 16
class_prob_0..2 [1, 4,  H/8, H/16, H/32]   raw logits, pre-sigmoid
```

Parity with the host is checked before compiling. The last verified cut matched
Ultralytics to **0.0002 model pixels** and **0.000000 score**.

## 4. Compile

`tools/compile_ic_terminal_on_admin.py` runs the ModelSDK container on the Admin
PC (`172.21.30.222`), with:

- image `783709528641.dkr.ecr.us-west-2.amazonaws.com/vdp-cli/modelsdk:2.0.0_Palette_SDK_master_B240`
- at least 50 calibration images (`--calibration`), 127 were used for the current
  build
- `--input-side 1280`

The script verifies the transferred graph's SHA-256 before compiling, pulls back
the ELF and MPK, verifies their hashes, and writes `compilation.json`.

Docker Desktop on the Admin PC must be started through the `DockerStart` scheduled
task. It cannot be started from a non-interactive SSH session.

## 5. Configs

`tools/build_ic_terminal_configs.py` derives all three configs from the compiler's
`yolov8_mpk.json`. It never invents numbers.

| Config | Contract |
|---|---|
| `0_preproc.json` | 1280×720 NV12 → 1280×1280 RGB, aspect preserved, centre padding |
| `0_process_mla.json` | `input_*` is logical tensor geometry; `slice_*` is compiler physical tile geometry—not necessarily equal |
| `boxdecoder.json` | `decode_type`, depths, grids, dequant, labels, threshold |

The generator raises rather than writing an inconsistent pair:

- grids that contradict `--model-size`
- tensor depths that contradict `--box-channels` / `--classes`
- unpack logical/tile geometry that differs from decoder logical/tile geometry
- missing compiler input/output tile metadata (CLI refuses to inherit template tiles)

For the current raw model, preprocessing tiles are HWC `[1280,64,3]`.
The three box-head output tiles are `[2,160,64]`, `[1,80,64]`, `[1,40,64]`;
class-head tiles are `[160,160,4]`, `[80,80,4]`, `[40,40,4]`. Logical frame
heights remain 160/80/40. Equating these tile heights with frame heights was a
second independent layout defect, repaired in r33 together with stale input tiles.

That last guard exists because it was missing: the unpack node kept a template's
640-model geometry (grid 80, depth 4 — 67,200 bytes of declared tensor geometry)
while the compiled graph emits 64-channel heads on grid 160 (2,284,800 bytes, in a
2,688,000-byte padded buffer). Nothing compared those numbers, so the decoder read
the buffer with the wrong layout and still reported "detections".

## 6. Stage and verify

Staging, running and measuring: [deployment](../operations/deployment.md).

A release is acceptable only when, on the same scene, the board reproduces the
host's detections within quantisation tolerance **and** holds ≥25 fps. r33 now
has a 29.5 fps annotated stream and same-frame native/Python decoder agreement;
full labelled accuracy, long-run recovery and shutdown acceptance remain open.
