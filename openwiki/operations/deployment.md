---
type: operations
title: Deploying to the Portable Board
description: Stage a release, run it, measure it, and hand the camera back.
status: active
tags: [deployment, board, verification]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/vision_sdk/release.py
  - repo://portable-vision/vision_sdk/board.py
---

# Deploying to the portable board

## Current operational default

Owner-selected **portable-vision-production-20261116-r2** now launches via repo
`run-production.ps1` and `config/production-pipeline.json`, using the immutable
coordinate/browser/pose/feature-ROI/C++ package and dedicated host environment.
The following native r33 stage/run instructions remain historical/reference;
do NOT substitute crop weights into that full-frame graph.
[Current deployment, bounds and rollback](production-pipeline-20261116.md).
No automatic board service/autostart changes were made.

## Before you start

- Board reachable: `ssh sima@172.21.30.120` (password in the team vault; not in
  this repo).
- Camera reachable **from the board**: `ping 192.168.1.10`.
- The capture feed stopped. It holds the hardware decoder.
- Configs regenerated after the last edit. Staging re-hashes every asset.

## Repeatable viewing of already staged r33

Use the host-side [live-view start/stop guide](../guides/live-view-and-labelled-check.md)
for a bounded window or machine-checkable network decode. It runs the pinned
release without changing configs or services and refuses an occupied board.
The [historical labelled test](r33-labelled-evaluation-20261114.md) now measures
FP32/compiled detections against annotations; current-camera physical acceptance
still needs known specimens and an operator.

## 1. Regenerate the configs

```powershell
python tools\build_ic_terminal_configs.py `
  --mpk ..\IC_TERMINAL_TRAIN\export-yolo11\yolov8_mpk.json `
  --template-dir ..\IC_TERMINAL_TRAIN\template-r8 `
  --output-dir ..\IC_TERMINAL_TRAIN\candidate\etc `
  --model-size 1280 --source 1280x720 --classes 4 `
  --labels-file config\labels-ic-terminal.txt `
  --decode-type yolov8 --box-channels 64 --sigmoid true `
  --threshold 0.05
```

`--threshold 0.05` matches the currently staged candidate and is the right value
for decode diagnostics. The CLI default 0.25 is not a measured production
recommendation: choose the operating threshold from suitable validation data,
then qualify it on known specimens. See [D9](../architecture/decisions.md) for
the original bench/operating distinction.

Note the file names: the compiler writes `yolov8_mpk.json` into the export folder,
while the candidate tree carries the same content as `etc/mpk.json`. They are the
same document under two names.

The tool fails loudly if compiler geometry/tiles are missing or disagree with
the flags. It derives preprocessing HWC tiles `[1280,64,3]` and output tiles
separately from logical frame shapes. Box-head slice heights are **2/1/1**, not
160/80/40. Do not copy template tile sizes or force `slice_* == input_*`.
The r33 stream fixed both layout faults; missing tile metadata now causes a
failure rather than silently inheriting old values.

The legacy candidate directories still contain stale tile configs. Regenerate
before staging another release, then upload all three files. Pinned r33 is the
currently tested native-layout candidate; it is still diagnostic conf 0.05.

## 2. Upload and stage

Upload the three configs (and the ELF if it changed) to
`/media/nvme/ic-terminal-candidate`, then:

```sh
cd /media/nvme/ic-terminal-candidate/tools
python3 launch.py stage /media/nvme/ic-terminal-candidate \
        /media/nvme/portable-release-YYYYMMDD-rNN \
        --config /media/nvme/ic-terminal-candidate/deployment.json
```

Staging prints `{"staged_release": ..., "assets": 7, "plugins": 16}`.

For the owner-selected workstation receiver (currently **172.21.30.13:5010**),
r30 and r31 were staged with a separate deployment JSON:
`/media/nvme/ic-terminal-receiver-deployment-r30.json`. Pass that file as
`--config` when intentionally targeting this workstation. The candidate's own
`deployment.json` still names the old 192.168.1.50 receiver. Confirm workstation
IP/reachability before staging; do not edit a pinned release to change its target.

## 3. Run

```sh
python3 /media/nvme/portable-release-.../tools/launch.py run \
        /media/nvme/portable-release-... --diagnostic \
        --seconds 60 --output /media/nvme/ic-terminal-verify-rNN
```

`--diagnostic` uses a local `fakesink` and captures raw buffers, so nothing is
sent to a receiver. It is the right mode for bench work, but **does not prove
normal network delivery**: that sink disables clock pacing. Omitting the flag
uses the real UDP sink, whose clock pacing exposed a source/pipeline time-base
mismatch in r30. r31 fixes that conversion and has passed a 600-frame live decode
test on this workstation. Finite-run shutdown can still truncate the final frame;
see [current status](status.md#r31-receiving-and-reference-comparison-results).
r33 additionally repairs input/output layouts and has a 29.5 fps annotated stream
with native/Python same-frame box agreement. Labelled detection accuracy remains
unaccepted; do not accept a release merely because video arrives or its worker
reports no errors. The saved annotated clip and exact tests are in
[current status](status.md#r33-first-working-annotated-native-stream).

Check the run, not just the detection count:

```sh
python3 -c "import json;d=json.load(open('/media/nvme/ic-terminal-verify-rNN/worker.json'));print(d['state'],d['frames'],d['errors'])"
```

A run that prints thousands of detections and still says `"state": "error"` has
not qualified. See [troubleshooting](troubleshooting.md#the-timing-defect).

## 4. Judge the decode

Never judge a board run by itself. Compare it with the host model on the same
scene. The host reference is a short Ultralytics script kept with the training
data, outside this workspace:

```python
from ultralytics import YOLO
model = YOLO(r"..\IC_TERMINAL_TRAIN\runs\train-yolo11n-1280\weights\best.pt")
result = model.predict(image_path, imgsz=1280, conf=0.05, verbose=False)[0]
```

Run it over a spread of frames from one rotation, then compare:

The comparison that matters:

| Metric | Host reference | Board |
|---|---|---|
| Boxes per frame at conf 0.05 | ~5.8 | measured |
| Max score | up to 0.99 | capped by class zero point |
| Box sizes | ~22×21 px | measured |

Remember the 0.5 ceiling on the two coarse class levels: a board score of 0.5 is
not the same as a host score of 0.5.

## 5. Restart the camera feed

The camera must be released before anything else can use it:

```sh
nohup /tmp/feed_supervisor.sh > /tmp/supervisor.out 2>&1 < /dev/null & disown
```

The supervisor restarts the capture if it dies and prunes old frames, so it can
run for days. Confirm with `ls /tmp/live | wc -l` and by fetching
`http://172.21.30.120:8080/`.

## Release hygiene

- Stage, then run. Never edit a config inside a staged release.
- Keep release numbers monotonic.
- Record the release id with any measurement you publish.
- A staged release is a candidate, not an accepted deployment, until its
  detections match the host on a known scene.
