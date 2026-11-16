---
type: architecture
title: Portable Runtime
description: The board pipeline, the launcher that ships it, and how a release is staged and started.
status: active
tags: [runtime, board, gstreamer, release, launcher]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T15:00:00Z
sources:
  - repo://portable-vision/runtime/launcher/worker.py#L125-L135
  - repo://portable-vision/runtime/launcher/support.py#L184-L205
  - repo://portable-vision/runtime/launcher/launch.py
---

# Portable runtime

## Selected operational release

As of2026-11-16, owner-selected default is `portable-vision-production-20261116-r2`.
Repo `run-production.ps1` reads pinned `config/production-pipeline.json`, verifies
manifest/native flags and invokes the frozen package in a dedicated environment.
Backend is latest-frame coordinate/browser + pose + feature ROI + C++8. Old r33
is preserved, not the new default. Each launch creates a new verified board job;
no global board symlink/service is changed. Default/max300s until frontend
lifecycle work. [Release/verification/handoff](../operations/production-pipeline-20261116.md).

## Opt-in summit host supervision (not board-qualified)

The standalone dashboard now discovers running loopback receivers from trusted
local `session.json` records and includes bounded client/proxy recovery. Root
`run-summit.py` optionally launches sequential frozen `run-portable.py` sessions;
each retains its20..300s board bound, hash checks and decoder-busy refusal.
Renewal requires child exit plus terminal host/worker evidence. Confirmed-ended
errors have bounded retries; missing cleanup evidence stops for manual review.
A per-checkout OS lock prevents simultaneous supervisors; low host disk stops
renewal without deleting evidence. Prompt-once credentials use a permission-
restricted temporary profile, removed on ordinary exit. No SDK/frozen package,
model, board service or default launcher changes. Offline policy/HTTP/state tests
are not hardware acceptance. [Operation and limits](../operations/summit-runbook.md).

## Two different runtimes live near this workspace

This distinction is the most expensive one to get wrong, so it comes first.

| Runtime | Where it lives | What it drives |
|---|---|---|
| **Portable launcher** (this station) | `runtime/launcher/` — `worker.py`, `support.py`, `launch.py`, `verify.py` | RTSP camera → 1280×720 → six-head model |
| **SDK board worker** (fixed station) | `vision_sdk/board_runtime/` | GigE Aravis camera → 3200×2256 Bayer → screw model |

`vision_sdk/board_runtime/persistent_board_worker.py` contains `aravissrc`,
`screw_count` and a hard-coded `192.168.1.17`. It never builds an RTSP graph and
knows nothing about `routecam` or `simaaidecoder`. It is the **fixed AOI
station's** worker, copied here with the shared SDK.

The graph the portable releases actually run is built in
`runtime/launcher/worker.py`. That file did not exist in the repository before
2026-11-14; it lived only in the candidate tree outside it.

## The machine

| Thing | Value |
|---|---|
| Board SSH | `172.21.30.120`, also `192.168.1.20`, user `sima` |
| SDK release root | `/home/sima/demeter-vision` (`vision_sdk/release.py`) |
| Launcher releases | `/media/nvme/portable-release-YYYYMMDD-rNN` |
| Worker port | `5005` is the SDK default; the launcher takes `--port` |
| Camera lease port | worker port + 1 |
| Camera | `rtsp://192.168.1.10:5005/routecam`, 1280×720 H.264 |

## The graph

From `runtime/launcher/worker.py`:

```
rtspsrc name=camera
  ! rtph264depay wait-for-keyframe=true
  ! h264parse
  ! video/x-h264,parsed=true,stream-format=byte-stream,alignment=au
  ! simaaidecoder name=decoder sima-allocator-type=2 next-element=CVU
  ! tee name=source
  ! video/x-raw
  ! simaaiprocesscvu name=simaai_preprocess num-buffers=5   config=0_preproc.json
  ! simaaiprocessmla name=simaai_process_mla                config=0_process_mla.json
  ! simaaiboxdecode  name=simaai_boxdecode                  config=boxdecoder.json
  ! application/vnd.simaai.tensor
  ! overlay
  source. ! video/x-raw ! overlay.
  overlay ! simaaiencoder name=encoder enc-bitrate=4000
          ! h264parse ! rtph264pay name=payloader ! udpsink   (fakesink with --diagnostic)
```

Three facts that cost time if forgotten:

1. **The decoder is exclusive.** A second pipeline (for example the JPEG capture
   feed used for focus checks) cannot open the camera while a worker runs.
2. **`videocrop` cannot sit between `simaaidecoder` and `simaaiprocesscvu`.** It
   fails with `Pad push YUV output buffer failed`.
3. **`simaaiprocesscvu` needs decoder metadata.** Feeding it from `filesrc` +
   `jpegdec` fails with `meta not found`; from `filesrc` + `h264parse` it fails
   with `input buffer is NULL`. There is no camera-free way to exercise the decode
   chain.

## Latest-frame video experiment (separate diagnostic)

`latest_frame_stream.py`/`latest_frame_graph.py` now split compressed camera H264:
one bounded branch forwards original video to the approved receiver; another
uses the sole hardware decoder and latest-only appsink for slower crop/MLA work.
No old boxes are drawn on current video. Results appear on their own timestamped
snapshot via a loopback browser panel. Initial received clip:600 decoded720p
frames at advertised30FPS while inference~1.69FPS. This is not native crop
throughput, tracking, favourable-region gating or production acceptance. r33's
worker is unchanged. See [experiment](../operations/latest-frame-video-20261116.md)
for bounds, active-session evidence, caveats and next stages.

## Opt-in latest-frame performance candidate

The independent-video worker now records per-inference stage times. Host flags
`--optimized --fast-transforms` select portable-only exact LUT packing and sparse
class-gated DFL decode plus removal of unused crop-ablation work. Installed SDK
and frozen fallback are unchanged. Startup compares against SDK byte/tensor and
reference prediction outputs; unsupported layouts/parity failures abort before
camera start. Default remains original path. Final60s diagnostic achieved7.99
inspections/s with approximately30FPS received video; this is not production
accuracy or optical latency acceptance. See
[profiling/equivalence report](../operations/latest-frame-performance-20261116.md).

## Coordinate-only publication and pose reuse (opt-in)

The candidate adds `--coordinates-only`/`--track-pose`. Board skips all drawing and
JPEG writes; latest-only publisher hashes raw NV12 and sends coordinates to host
5012 while video5010 remains independent. Host decodes camera video and draws
canvas quads on EXACT hash-matched inspection frames, separate from live video;
missing/stale matches clear overlays. This handles native decoder missing PTS
without timestamp guessing or stale live-box placement.

LK/RANSAC pose updates reuse previous PCB geometry between full localizations
(max0.5s) with confidence/appearance/geometry/time guards; failures re-localize or
become UNKNOWN. Model still inspects each accepted crop. Final60s:12.59 inspections/s,
30.01 host decoded frames/s,769/769 matches. Not independent accuracy or tracker
qualification. Both fallbacks/r33 untouched; no region gate or C++ fusion.
[Protocol, evidence and limitations](../operations/coordinate-browser-pose-20261116.md).

## C++ input-fusion prototype (opt-in, not packaged)

`native/fused_input.cpp`/`fused_input.py` fuse uint8 BGR channel order, accepted LUT
quantization and compiler input packing into one output buffer. No resize/pad,
model or numerical threshold changes. Worker requires explicit library SHA, ABI/
layout checks and byte-exact startup preflight before camera start. `--cpp-library`
with `--cpp-kernel parallel8` enables it; no library retains accepted OpenCV path.
Eight-worker OpenMP pool has passive waiting scoped to this child process only.

All31 prepared inputs match reference bytes in six kernels; naive scalar/NEON
versions were slower and rejected. Selected live input stage8.92→3.81ms; observed
whole-run12.44→13.34 inspections/s, host video~30FPS. This is sequential short-run
performance, not production accuracy/latency/soak acceptance. Frozen latest fallback
unchanged. [Evidence/build details](../operations/cpp-input-fusion-20261116.md).

## Tracker feature ROI (opt-in)

`--feature-roi` with `--track-pose` restricts GFTT compute to detected seed bounds
plus8px halo, keeping identical full polygon mask/corner parameters/global coords.
Not an inspection-region gate. Ordered points matched on31 board audited frames;
full refresh feature cost~93→4.3ms live. On the same C++8 input path, final60s
achieved974 actual model inspections/61.05s (15.95/s),30.03 host decoded video FPS,
all974 hashes matched. Reliability/refresh thresholds unchanged; no production
quality/latency guarantee. Default and all frozen fallbacks unchanged.
[Parity, audits and limitations](../operations/tracker-feature-roi-20261116.md).

## Worker protocol

The worker probes named pads and writes JSONL evidence:

| Probe | What it enforces |
|---|---|
| `decoder:src` | Caps must be 1280×720, else `unsupported_camera_geometry` |
| `simaai_boxdecode:src` | Frame identity present, buffer size within `4 + 24×topk`, boxes parse |
| `encoder:src` | **Frame admission** — this is where PTS is read and `health.observe()` runs |
| `output:sink` | RTP payload sanity and marker/time progress |

Native stages can leave `GstBuffer.pts` unset while retaining a timestamp and
frame identity in `GstSimaMeta`. `worker.py` records the **decoder output's** meta
timestamp, converts it to pipeline running time using `support.SourceClock`,
then stores it in `support.FrameTimestampCache`, keyed by `(stream_id, frame_id)`.
Only decoder output inserts entries; boxdecode output cannot overwrite or
recreate them. Encoder output consumes the matching entry, even if it already
has a valid PTS, and restores missing PTS before admission and RTP packetisation.
This carries source metadata; output arrival time is never used as a substitute.

`SourceClock` anchors the first decoder metadata timestamp to
`pipeline.get_clock().get_time() - pipeline.get_base_time()`. That one fixed
offset applies to all later frames, preserving source intervals exactly rather
than replacing them with delivery/encoding jitter. Invalid clock values,
overflow or a source reset before the initial epoch fail the run; subsequent
nonadvancing output remains rejected by `Health`. No silent re-anchoring occurs.
Health/run JSON includes `source_clock` origins, offset and conversion method.
This establishes a usable presentation-time origin, **not optical capture time**.

The cache is thread-safe and holds at most **512** outstanding frames. Completed
entries are removed; at capacity the oldest pending entry is evicted. A missing
entry never borrows another frame's timestamp: if encoder PTS is also absent,
`Health` records `missing_media_time` (and non-diagnostic output is dropped).
Invalid source identity/time or conflicting duplicate source times stop the run.
`Gst.CLOCK_TIME_NONE`, booleans and negative timestamps are invalid source times.
Failed buffer timestamp assignment also stops the run.

`health.json` and `worker.json` include `timestamp_cache`: capacity, pending,
peak pending, stored, consumed, misses, and evictions. Frames still in flight at
shutdown can leave a small pending count; these are discarded with the run, not
retained across sessions.

The initial `r28` trial processed 426 timed frames after a hardware-service
restart. The bounded, stream-aware `r29` passed two 300-second diagnostics:
8,978 and 8,983 frames respectively, all validly timed, matching RTP marker
counts, no errors/misses/evictions, peak pending 30. The second run measured
worker RSS of 61,252–61,636 KiB after warm-up. These are short stability and
local RTP integrity checks, not proof of camera-clock origin, real receiver
playback, detection accuracy, or all-day operation; application qualification
remains unestablished.

**Clock conversion applied in r31:** r29/r30 carried approximately board uptime,
which made real `udpsink sync=true` wait on the first future-dated packet. Local
`fakesink sync=false` tests had hidden the mismatch. The fixed-offset conversion
above allows real clock-paced delivery without disabling sink scheduling.
A bounded workstation FFmpeg test decoded 600 live 1280×720 frames and exited 0;
a separate complete RTP capture decoded 1,716 frames offline. Detection accuracy
and human-observed playback are still not established.

Finite runs still set the graph to NULL without an EOS/drain handshake. Encoder,
RTP marker and receiver frame counts can differ at the tail, and one capture
explicitly reports `incomplete_final_frame`. Clock repair does not establish
lossless shutdown or full application acceptance.
See [current status](../operations/status.md) for evidence and
[troubleshooting](../operations/troubleshooting.md#the-timing-defect) for the
original failure.

## Release staging

Two mechanisms exist and they do not share a directory:

| Mechanism | Entry point | Writes to |
|---|---|---|
| Launcher | `runtime/launcher/launch.py stage <candidate> <release>` | `/media/nvme/portable-release-*` |
| SDK release manager | `vision_sdk/release.py` | `/home/sima/demeter-vision` |

The IC/terminal work uses the **launcher**. `vision_sdk/release.py` belongs to the
fixed station's flow.

`launch.py stage` copies the candidate tree, records a SHA-256 per asset and writes
`release.json`. `launch.py run` re-verifies those hashes before starting the
worker, then starts it under `setsid nohup`.

### The hash rule

Rewriting a config after staging produces:

```
ValueError: release asset hash differs: etc/0_preproc.json
```

That is the contract, not corruption. Re-stage after any edit.

## Worker artefacts

| File | Contents |
|---|---|
| `detections.jsonl` | One record per frame: session, frame id, stream, detections |
| `frames.jsonl` | Frame ids and timestamps |
| `packets.bin` | Raw buffer captures (tensor payloads, not video) |
| `worker.json` | Final state, frame counts, `valid_timed_frames`, error list |
| `health.json` | Periodic status snapshots |

`worker.json` is the authority on qualification. A run can write thousands of
detections and still be `"state": "error"`.

## Detections are not verdicts

The board reports boxes, scores and class ids. Pass/fail policy lives on the host.
