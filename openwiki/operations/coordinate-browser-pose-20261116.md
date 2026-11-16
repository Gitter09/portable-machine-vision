# Coordinate-only browser rendering + bounded PCB pose reuse — 2026-11-16

Owner requested (1) publish coordinates/render in browser, (2) reuse PCB pose with
re-localization when unreliable. Explicitly NO favourable-region gating or C++ fusion.
Both frozen fallback packages remain unchanged and pass integrity checks.

## What is implemented

- Original compressed camera video remains independent of latest-only inference.
- Board coordinate mode skips drawing and ALL inspection JPEG writes. A separate
  bounded publisher (queue1, replace-old) hashes immutable source NV12 bytes and
  sends result coordinates/metadata to workstation172.21.30.230:5012. Video remains
  RTP5010. No passwords/images are sent on metadata channel.
- Host receives original video, depacketizes/decode H264 with PyAV, creates browser
  JPEGs on the WORKSTATION, and keeps at most90 unique decoded images. Browser
  live pane is MJPEG; browser JS draws result quads/text on its own canvas.
- **Do not confuse this with stale overlays on live video**: second pane uses ONLY
  the exactly matched inspected frame. Native decoder PTS are absent, so matching
  uses SHA-256 of full decoded NV12 pixel bytes, not guessed timestamps or frame
  order. Missing hash match, >0.5s stale metadata, or disconnection clears overlays.
  Current uninspected video has no overlaid old decisions. No interpolated verdict.
- Source raw images are never shipped separately from the board: host already
  decoded that same camera frame. Optional host-only audits retain matched raw
  frames; production browser transport does not need the audit.

Pose reuse is NOT a static position lookup table. `pcb_pose_tracker.py` uses
forward/backward LK flow within the PCB seed and RANSAC similarity transform,
then validates inlier fraction, error, scale/rotation/translation, source bounds
and blue substrate evidence. Full localization runs at least every0.5s, on >0.25s
step gaps, and immediately on tracking confidence failure. Insufficient features
leave tracking unavailable, not an invented pose. Failure/removal clears state and
falls back to full locator/UNKNOWN. Fresh model inference still runs on each
accepted crop; pose tracking never asserts component state.

## Measurements

Sequential live runs, not identical frames or independent accuracy trials:

| Configuration | Whole-run inspections/s | Located / unknown |
|---|---:|---:|
| Prior optimized snapshot pipeline |7.99|479/9|
| Coordinate-only, asynchronous publication40s |9.33|377/6|
| Coordinate + pose reuse40s |12.22|502/1|
| Final60s, headless Edge render/audit |**12.59**|**761/8**|

Final:769 inspections in61.07s;652 tracked updates/117 full attempts; one tracking
confidence failure led to re-localization, not stale pose reuse. All769 coordinate
messages published/received, no publication replacement, all769 matched decoded
source hashes. Receiver decoded1761 frames,30.01 frames/s between arrivals,0 RTP
sequence gaps; p99 gap69ms/max703ms (includes startup/source burstiness), so not a
zero-jitter/optical-latency/long-soak guarantee. Board forwarding28.84FPS includes
startup/shutdown. Browser canvas screenshot demonstrates actual JS box rendering.
No real browser-render FPS or physical camera-to-display latency measured.

Tracked located frames averaged54.0ms processing,23.5ms pose stage; full located
refreshes averaged165.4ms,134.8ms pose stage (full localization + feature seeding).
This is a remaining optimization target, not hidden by aggregate FPS.

Initial synchronous hashing smoke was7.64FPS; moving hashing/network to publisher
improved critical-path throughput. Early receiver runs had packet gaps;4MB receive
buffer was added, final run had none. All failed/intermediate evidence retained.

## Validation / limits

- Final31 raw host audit frames: hash matches corresponding board input EXACTLY;
  25 tracked samples. Fresh full-selector seed coverage by estimated crop minimum
  0.99999997. This is **geometry diagnostics**, not human target-site preservation,
  detection precision/recall or production tracking acceptance.
- Synthetic controls cover translation/rotation, large jump, stale time, periodic
  refresh and board removal. RTP FU-A and missing-packet fail-closed behaviour,
  exact-match/stale gating and bounded latest-only publication tested.
- Headless Edge screenshot `browser.png` shows matched/tracked canvas boxes and
  separate live pane. Full suite340 passed, two existing dependency warnings.
- Model, quantization, calibration, confidence/NMS and class IDs unchanged. Tracking
  changes crop pose versus full-per-frame localization: no claim of pixel-identical
  model inputs or unchanged live accuracy. Real occlusion/glare/multiple-object and
  independent specimen qualification still required.
- Frozen slow22-file and optimized24-file package verifiers both passed. No r33,
  installed vendor/SDK, service or fixed-station changes. No C++ fusion or region gate.
- Bounded jobs completed;10:33 UTC read-only process check found no camera graphs.

## Files / run

Runtime: `latest_frame_stream.py` coordinate flags, `coordinate_publisher.py`,
`pcb_pose_tracker.py`. Host: `run_coordinate_browser.py`, `browser_coordinate_view.py`,
`coordinate_browser.html`. Local browser HTTP binds127.0.0.1 only. New separate host
environment `../IC_TERMINAL_TRAIN/browser-view-env` uses requirements-browser-view.txt
(PyAV16.1, NumPy2.2.6, OpenCV4.12, Paramiko3.5.1); existing environments unchanged.

```
../IC_TERMINAL_TRAIN/browser-view-env/Scripts/python.exe tools/analysis/run_coordinate_browser.py \
  --output ../IC_TERMINAL_TRAIN/<NEW-unique-name> --seconds 180 --track-pose
```

Use `--headless` for unattended measurement; omit `--track-pose` for coordinate-only
ablation. `--audit-inputs` saves occasional matched raw frames on HOST only.
Bounds20..300s, occupied-board refusal and unique evidence directories enforced.
Closing browser does not immediately stop board: bound applies. No other receiver
may occupy5010/5012. Old pipeline remains available via frozen fallback launchers.

Evidence roots: `coordinate-browser-20261116-{smoke,coordinates,tracking,final}/`.
Final worker/receiver JSON, published/inspection JSONL, source snapshots,
geometry-audit-final.json, stage-summary.json and browser.png retained.
