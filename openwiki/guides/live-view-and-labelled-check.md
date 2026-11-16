# Live annotated view and a controlled good/bad check

This is the portable station's **diagnostic** r33 stream, confidence **0.05**.
It is not a production inspection service. Read [current status](../operations/status.md)
before using the shared board. Do not run the capture feed simultaneously.

## Start and stop

From this repository on the receiver workstation (172.21.30.13), with Python,
Paramiko and FFplay on PATH:

```powershell
python tools/portable_live_view.py --seconds 600 `
  --output ../IC_TERMINAL_TRAIN/live-review-01
```

- A window opens with live boxes and labels. **Q** in the window, or **Ctrl+C**
  in the terminal, stops the viewer and its board launcher.
- Ten minutes is the default maximum. `--seconds` changes this finite bound;
  there is no automatic restart or installed background service.
- Choose a **new output directory and basename** each time. Existing board
  evidence is never reused or overwritten.
- The wrapper targets pinned `/media/nvme/portable-release-20261114-r33`, not
  stale candidate configs. Normal release verification still runs.
- It refuses an occupied board, a changed pinned deployment hash, an unavailable
  receiver address/port, or an existing board output. It does **not** kill another
  job, stop the capture supervisor, change the receiver target or restart services.
- If the workstation address changes, stage a new correctly targeted release.
  Do not edit r33 or bypass checks.

The script starts the receiver before the board. It retains the remote launcher
PID, checks its exact release/output arguments before signalling it, and waits
for board shutdown. It terminates only its own local viewer subprocess. SSH loss
can prevent confirmed cleanup; inspect the board before another run. The finite
run limit bounds an orphan but is not a recovery service.

### A machine-checkable receiver test

```powershell
python tools/portable_live_view.py --headless --frames 300 --seconds 35 `
  --output ../IC_TERMINAL_TRAIN/live-check-01
```

This replaces the window with a bounded FFmpeg decode. Success means
`decoded_frames: 300`, `receiver_natural_exit: 0`, `launcher_exit: 0` and
`errors: []` in `result.json`. It tests actual network video, not a local fakesink.

Local evidence includes receiver logs, SDP, pinned release inventory, board
health/worker/supervisor JSON, and a final `result.json`. Detailed board evidence
stays under `/media/nvme/ic-terminal-live-<output-basename>`.

For a windowed run, `receiver_natural_exit: null` means the wrapper stopped the
player after board completion or Ctrl+C. Windows reports that termination as exit
1; this is recorded separately from an unexpected player failure. It is not a
clean natural player exit or proof of lossless stream draining.

## What has actually been tested

The 2026-11-14 trials exercised 300-frame headless completion, windowed duration
completion, programmatically requested Ctrl+C, and immediate restart followed by
150-frame headless completion. A final source revision also passed a 60-frame
headless smoke test. All bounded decoders and board launchers exited 0. A final windowed run showed advancing FFplay video-clock statistics.
It also reported 96 dropped frames; desktop playback smoothness/latency is **not**
qualified. That run overlapped host-side ONNX evaluation, so the drop cause was
not isolated. No human-observed visual acceptance or manual Q-key test is claimed.

## Controlled live good/bad trial — requires an operator

Remote software cannot place parts or independently certify their physical state.
The historical labelled test is complete. The owner subsequently supplied a
good specimen: the [first current-camera trial](../operations/good-part-trial-20261115.md)
found top/bottom clipping during rotation and false fail labels. Stop the plate
and move that same board inward before repeating the good-state trial. Known
bad-state testing and broader specimen coverage still need the operator.

1. Keep the current camera mount, focus, lighting and framing fixed. Have the
   owner confirm each specimen's good/bad condition independently of predictions.
2. Start with the three historically represented states:
   - IC present + terminal present;
   - IC missing + terminal present;
   - IC missing + terminal missing.
   The owner confirmed these are **missing-component faults only**; no solder,
   electrical or other defect types are evaluated.
   Good IC + failed terminal was not represented in the audited three runs.
3. Run one state at a time. Record specimen ID, intended class for each site,
   run/output name, start/end times and a full plate rotation. Do not use model
   predictions as labels. Include several distinct specimens, not just many
   adjacent frames of one board.
4. Preserve source images with independently drawn boxes for numerical scoring;
   a screenshot of already drawn predictions alone is not sufficient ground truth.
5. Count missed defects, wrong classes and alarms on good parts. Report these
   separately from bounding-box overlap. Agree the acceptance criteria before
   choosing confidence or adapting the model.
6. Stop the run and check evidence before changing specimens or starting capture.
   A separate source-capture pipeline needs exclusive decoder access.

Do not tune on the historical test set, silently relax the IoU criterion, or
retrain until the small-IC annotation/localisation issue has been reviewed.

## Benchmark an already-running view

After owner-confirmed framing, use the **read-only**
[live baseline protocol](../operations/benchmark-baseline-20261115.md) rather than
starting another camera pipeline. It measures a fixed warm window and matches
native detections to encoder-admitted frame IDs. Leave the confirmed specimen
unchanged during the window; stop the plate before swaps and confirm state before
measuring again. Performance, class consistency and actual inspection accuracy
are reported separately. All three available states (both present, IC missing/
terminal present, both missing) are captured. The owner confirmed missing-only
fault scope. Optional good-state repeat and IC-present/terminal-missing are not
measured; never infer the physical state from model output.

## Historical labelled evaluation

[Results and limitations](../operations/r33-labelled-evaluation-20261114.md).
The reproducible prepare/inference/evaluate commands are in that report. They
use the actual compiled model via the verified software-transform diagnostic;
they do not replay images through the native live camera graph.
