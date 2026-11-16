---
type: operations
title: Troubleshooting
description: Camera symptoms, decoder contention, the decode defect and the timing defect.
status: active
tags: [troubleshooting, defects, camera]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/runtime/launcher/worker.py
  - board://ic-terminal-verify-r24/worker.json
---

# Troubleshooting

## The camera

### The feed is frozen but frames keep arriving

Two different faults look identical in a browser:

| Symptom | Cause | Check |
|---|---|---|
| New files appear, image never changes | **Scene** is static — turntable stopped | Compare two frames 3 s apart; zero changed pixels means the plate is not turning |
| No new files, image frozen | **Transport** dead — camera or link | `ping 192.168.1.10` from the board |

Measured example: 91 frames in 3 s with distinct hashes, and 0 pixels changed
between frames 4 s apart. The feed was healthy; the turntable had stopped.

### The camera is unreachable

`ping 192.168.1.10` from the board returning 100% loss is a physical fault:
power, cable, or switch port. Nothing in software recovers it. The capture
supervisor keeps retrying and will reconnect on its own once the link returns.

## Decoder contention

The hardware decoder is exclusive. Symptoms when two users collide:

- `simaaidecoder` never reaches `PLAYING`
- a run exits in ~3 s with `exit_code: 1`
- the capture feed stops producing frames while a pipeline runs

Stop one before starting the other. A run pauses nothing automatically.

## Native GStreamer failures

| Message | Meaning |
|---|---|
| `Pad push YUV output buffer failed` | An element between the decoder and `simaaiprocesscvu` cannot accept its buffer, e.g. `videocrop` or a bare `fakesink` |
| `Please check readme to use metadata information, meta not found` | `simaaiprocesscvu` was fed from something other than the hardware decoder, e.g. `filesrc ! jpegdec` |
| `[CRITICAL] input buffer is NULL` | `simaaiprocesscvu` was fed from `filesrc ! h264parse`; the decoder's metadata is missing |

There is currently **no way to exercise the full decode chain without the
camera**. A JPEG-over-RTSP or file-based harness was attempted on 2026-11-13 and
failed at the preprocessing node for exactly these reasons.

## Release and hash errors

```
ValueError: release asset hash differs: etc/0_preproc.json
```

The staged release's recorded hash does not match its file. Usually a config was
regenerated after staging, or two staging runs interleaved. Delete the release
directory and stage again.

```
ReleaseError: This release has no staged bundle; send it to the board first
```

The manifest says `compiled_not_board_verified`. Stage before starting.

## The decode defect

**Status: two layout faults isolated and repaired in r33.** Short annotated video
and same-frame native/Python decoder agreement now pass; labelled accuracy,
operating threshold and long-run acceptance remain open. The r24 measurements
below are historical evidence, not r33's current behaviour.

### Evidence (release r24, 1181 frames, conf 0.05)

| Metric | Board | Host on the same scene |
|---|---|---|
| Boxes per frame | 0.4 | 5.8 |
| Max score | 0.217 | 0.986 |
| Median box | 32×30 px | 22×21 px |
| Negative heights | 32.7% | n/a |

At conf 0.5 the board returns **zero** detections across 1785 frames while the
host returns several. Negative heights come from `box_clip_boundaries` on boxes
predicted in the letterbox padding: an origin above the frame yields a negative
extent, which the board reports as an unsigned value near 2³².

### What was fixed on 2026-11-13

The MLA unpack node was configured with a 640-model template geometry (grid 80,
depth 4) while the compiled graph emits 64-channel DFL heads on grid 160. The
physical buffer allocation did not enforce the declared logical tensor geometry,
so the pipeline ran and decoded garbage. The generator
now writes `input_*` from the compiler's tensor shapes and refuses mismatched
pairs. Output changed from uniform ~300 px squares to IC/terminal-sized boxes —
an improvement, not a fix.

### Hypotheses already rejected

| Variant | Result |
|---|---|
| BGR instead of RGB | max score 0.127, single class — worse |
| Negated class dequant scales | 77% negative heights — much worse |
| Interleaved box/class tensor order | max 0.413, same negatives — inconclusive, not adopted |

The vendor's own shipped pipeline lists three box tensors then three class
tensors, which is what we already had.

### Known quantisation ceiling

The two coarse class levels are quantised with `zero_point = 127`, so their
logits can only be ≤ 0 and their sigmoid can never exceed **0.5**. Any comparison
of board and host scores must account for that before it is called a defect.

### Confirmed repair: input packing and output tile interpretation

A raw-buffer probe of r31 identified native input tiles `[32,128,3]` versus
compiler tiles `[1280,64,3]`. Replaying the captured input with the same ELF
reproduced the captured output byte-for-byte on two frames. Repacking only the
same pixels into compiler order changed about 79.3% of byte positions and
restored meaningful predictions. This is direct evidence, not a colour/threshold
hypothesis. The companion untiled quantized RGB tensor verifies tile semantics.

Input-only r32 restored scores but native box sizes still disagreed with Python
on the **same captured output**. Logical output heights 160/80/40 are not tile
heights: the compiler uses 2/1/1-row box-head tiles. r33 sets preprocessor tiles
and MLA/boxdecoder slice geometry from the MPK while keeping logical geometry
separate. Eight detections across two r33 frames agree with Python within
0.473 pixels / 1.50e-8 score; the later annotated network run records no negative
heights among 6,431 diagnostic boxes. No model, training or threshold change.

The config generator and geometry guard now preserve this distinction and refuse
missing tile metadata. Do not restore `slice_* == input_*` as a supposed safety
rule: that would recreate the output-layout fault. Raw `yolov8` decoding is
supported and works under this contract; repairing the failed vendor conversion
is no longer a prerequisite for the first working stream.

Evidence, clip and limitations: [current status](status.md). Never generalize
these short tests into labelled precision/recall or all-day reliability.

## The timing defect

**Status: bounded storage and source-to-running-time conversion applied (r31).
Real clock-paced delivery and bounded live decoding pass; detection accuracy,
shutdown-tail integrity and full application qualification remain unestablished.**

Before the repair, runs ended with:

```
"state": "error", "valid_timed_frames": 0
"errors": ["missing_media_time", "nonadvancing_rtp_time"]
```

Frames flowed (~29 fps) and detections were written, but no frame was counted as
validly timed. This is separate from detection accuracy.

### Where it comes from

Frame admission reads PTS on `encoder:src` in `runtime/launcher/worker.py`.
Board probes found valid buffer PTS at `decoder:sink`, but `CLOCK_TIME_NONE` at
`decoder:src` and downstream native stages. `GstSimaMeta` still carried an
advancing timestamp and frame/stream identity. Thus `Health.observe()` recorded
`missing_media_time`, and the RTP output had nonadvancing time markers.

### Applied repair and checks

`support.FrameTimestampCache` carries the decoder-output metadata timestamp by
`(stream_id, frame_id)` to the encoder-output probe. Encoder output consumes the
entry and restores PTS only if missing. This preserves native PTS when present;
it never substitutes output arrival time or weakens the health gate.

The cache is thread-safe, bounded to 512 pending frames, and exposes depth,
high-water mark, consumption, misses and evictions in worker/health JSON.
Invalid source metadata or conflicting duplicate source times stop the run.
Evicted or missing entries do not borrow another frame's time; absent encoder
PTS still causes `missing_media_time`.

An earlier draft (`r28`) never deleted entries and keyed only by frame number.
Do not reuse it for long runs. `r29` fixes both issues and has 34 offline
regression tests. Two 300-second diagnostics processed 8,978 and 8,983 frames,
all validly timed, with identical RTP marker counts, peak pending 30, no cache
misses/evictions/errors, clean shutdown, and passing raw RTP integrity checks.
Memory was measured on the second run; see [current status](status.md) for the
reports and limits of this evidence.

A clean completed worker reports `state: stopped`, **not `ok`**. Check
`errors: []`, `shutdown_complete`, supervisor exit/reap status, matching frame and
valid-timed counts, plus `verify.py run` and `verify.py media`. The run verifier
still withholds full readiness until accuracy, pairing, recovery and receiver
playback are proven; a passing timing check alone is not application acceptance.

### Real UDP output stalls although local diagnostics pass

r30 targets the selected workstation (172.21.30.13:5010). A real network test
observed one encoder/output-sink buffer, zero received UDP packets, then
`output_timeout`, despite 305 source/detection frames. The metadata timestamp
is approximately board uptime (179,755 s), while `udpsink sync=true` schedules
using pipeline running time. Thus the first packet waits far into the future.
The diagnostic `fakesink sync=false` hides that scheduling mismatch.

A synthetic `appsrc ! udpsink` test reproduced no delivery with absolute
board time and delivery with pipeline-relative time under the default clock
pacing. Disabling pacing also delivered, but is not the preferred repair.
r31 now applies a fixed offset from the first decoder timestamp to pipeline
running time and preserves later source intervals. Sink pacing remains enabled;
a bounded workstation receiver decoded 600 live frames and exited 0. See
[current status](status.md#r31-receiving-and-reference-comparison-results).
Do not lower detection thresholds, retrain the model, relax health gates or blame
H.264 to fix this zero-packet condition. FFmpeg's missing-dimensions error here
means it received no video, not that received video failed decoding.

### Receiver DTS warnings versus source RTP integrity

The first r31 FFmpeg harness rounded decoded timestamps to a 30 Hz output time
base and reported repeated DTS values. A second test used the RTP 90 kHz time
base (`-enc_time_base 1:90000`), passthrough frame timing, and an explicit
600-frame limit. It decoded live and exited cleanly without those warnings.
Do not discard source timing or force constant fps just to silence a test-harness
warning. The nominal-30fps MP4 is for viewing only, not timing qualification.

A direct NULL shutdown can still cut off the final RTP frame or queued video.
The second r31 raw capture explicitly fails `verify.py media` with
`incomplete_final_frame`, despite its successful 600-frame live decode. EOS/drain
handling is still outstanding; do not hide this by trimming the capture before
claiming end-to-end integrity.

## Recovering a stuck state

```sh
pkill -f feed_supervisor; pkill gst-launch-1.0        # stop capture
pgrep -af 'gst-launch|worker.py'                       # confirm nothing holds the decoder
```

Then re-stage and run. If the board's `/tmp` is full, `rm -f /tmp/live/*.jpg` and
`rm -rf /tmp/decode-test`.
