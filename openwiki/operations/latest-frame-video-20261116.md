# Smooth camera video + latest-frame inspection — 2026-11-16

## Owner-selected order

1. Smooth video with independent latest-frame inference (this experiment).
2. Favourable-region inspection, after owner judges this view.
3. Profiling and optimizations later. Neither region gating nor tracking is enabled.

## Architecture

`runtime/launcher/latest_frame_stream.py`, `latest_frame_graph.py`, and host
`tools/analysis/run_latest_frame_view.py` implement an isolated bounded diagnostic.

```
RTSP H264 -> depay/parse -> compressed tee
  |-> bounded queue -> RTP -> approved172.21.30.230:5010 -> FFplay LIVE video
  `-> bounded queue -> ONE hardware decoder -> appsink(max1,drop=true)
        -> frozen hybrid crop/pad114 -> software transforms -> compiled MLA
        -> inverse-map boxes onto THAT inspection frame -> immutable JPEG/result
        -> loopback-only browser snapshot panel
```

Compressed original camera video is forwarded, not CPU-reencoded or redrawn.
Inference pulls a latest available decoded sample only when free. No accumulation
of old inspection jobs and no inference-controlled video cadence. Compressed
queues are bounded but not leaky (do not deliberately discard H264 reference frames).
Native graph threads forward video without Python per-frame probe callbacks.

Live video intentionally has NO boxes. Latest result boxes appear only on the
corresponding inspected snapshot in a separate panel. Panel shows source pull time,
processing time, received age, UNKNOWN, STALE after2s and DISCONNECTED on fetch
failure. Received age is not camera-to-display latency or proof of recent part state.
An old result never establishes the state of the current uninspected live frame.
No tracking, interpolation, favourable-region filtering or production verdict.

## Evidence

`../IC_TERMINAL_TRAIN/latest-frame-stream-20261116-01/`:
- Board exit0;70 inspections,68 located/2 unknown over41.48s (~1.69 inference FPS).
-1184 forwarded access units (~28.55FPS over whole startup/shutdown-inclusive run).
- Received clip independently counted/decoded:600 frames,1280x720,30/1 advertised
  rate,19.958s container duration. FFmpeg decode exit0, with one nonmonotonic DTS
  warning in null muxing. This is not zero-loss, jitter/latency or long-soak acceptance.
- No hardware encoder/CPU video reencode. Model, crop policy and thresholds unchanged.

`../IC_TERMINAL_TRAIN/latest-frame-view-20261116-01/` is the subsequent180s
owner-viewing session. `session.json` records actual controller/receiver PIDs,
start time and loopback panel URL. At observation: running, panel HTTP200,
latest snapshot sequence19, located. Subsequently completed:324 inspections,
316 located/8 unknown,1.79 inference FPS,29.67 forwarded video FPS, no errors.
Bounded job stopped by itself; check current
session/process state before another camera job. Closing FFplay alone does not
immediately stop board inference; the bound still applies. r33 remains untouched.

## Use

From portable repo, using GPU environment only for dependency availability:

```
../MODEL/.venv/Scripts/python.exe tools/analysis/run_latest_frame_view.py \
  --output <NEW-evidence-directory> --seconds 180 --receiver-host 172.21.30.230
```

For measured recording use `--headless` with >=30s (600-frame receiver target).
Host verifies local receiver assignment and refuses existing outputs/occupied board.
Receiver starts before camera. Only its own receiver process is terminated; board
has a self-bound plus timeout. Original releases/services/vendor SDK are not changed.
The HTTP server binds127.0.0.1 only. Deployed source snapshots and all inference
JSONL are retained. Snapshot filenames are immutable and path-checked on download.

## Limitations / next

This improves video cadence, NOT detector throughput or accuracy. Earlier extra
present predictions, crop unknowns and quantization sensitivity remain relevant.
Keep fresh independent-state validation pending. First judge visual usability;
then separately design favourable-region triggering and its evidence-age policy.
