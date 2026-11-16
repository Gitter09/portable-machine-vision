# Crop-aware compiled streaming diagnostic — 2026-11-16

## Scope

Owner authorized compiled replay, then end-to-end streaming. A separate diagnostic
uses `runtime/launcher/crop_stream.py` and `crop_geometry.py`; it does NOT alter
r33 or the native runtime, copied SDK, installed vendor code or fixed station.
This is NOT a 30fps production implementation.

Graph: one RTSP hardware decoder → latest-frame appsink (bounded/drop) → NV12/BGR
→ frozen colour-seed/local-Canny crop → resize/pad114 → RGB/255 once → installed
software quantize/tessellate → actual MLA → software unpack/dequantize → class-aware
NMS (.05/.45/top100) → inverse homography to source-frame quadrilaterals → software
OpenH264/RTP → approved receiver172.21.30.13:5010. Unknown crop emits UNKNOWN with
no inference or stale boxes. Separate capture and output graphs share no decoder.

`run_crop_stream_diagnostic.py` is receiver-first, bounded15..300seconds, refuses
occupied board graphs and existing outputs, and only terminates its own receiver.
It now also refuses a selected receiver IP that cannot bind locally BEFORE board
work; `--receiver-host` must be explicitly selected/authorized if destination changes.

## Trials

- `../IC_TERMINAL_TRAIN/crop-stream-diagnostic-20261116-01/`: retained failed first
  attempt. Board OpenCV4.6 lacks NMSBoxesBatched. Added equivalent per-class NMSBoxes
  in portable diagnostic code only, not vendor/SDK changes. Six geometry/NMS tests
  pass against reference including cross-class overlaps, empty input and transforms.
- `...-02/`: worker completed70 frames,69 located/1 unknown, no worker errors,
  46.33seconds, **1.51 processed FPS** including snapshot overhead. Per-frame
  processing timings are retained in frames.jsonl. This does NOT
  establish receiver/render FPS or physical latency. Software path remains slow.
- Receiver failed to obtain H264 dimensions/frames. PC's current IP is
  **172.21.30.230**, not approved172.21.30.13. No authorization assumed to retarget.
  End-to-end network video is therefore **NOT passed**. Both bounded jobs stopped;
  final process inspection found no camera/worker/feed graphs. r33 untouched.
- OpenH264 printed one EncodeFrame cmInitParaError warning. Encoder delivery is
  not qualified; investigate that warning and verify packet counts/receiver decode
  at the correct endpoint rather than assuming the IP mismatch is the only issue.

## Evidence and quality caveats

Actual board fixture/scripts, frame JSONL and three source/model/head snapshots are
retained in `...-02/`, with downloaded deployed scripts in `source-snapshot/`.
Reconstruction using each actual board crop quad reproduced all three model images
pixel-exact and source-to-model transforms within floating-point tolerance. Local
and board crop polygons agree within.000184px after cyclic corner ordering; two
crops differ by180° corner start between OpenCV versions, consistent with known
unresolved semantic orientation (not missing pixels). No canonical orientation fix
or independent pose qualification is claimed.

Three matched live snapshot FP32/compiled outputs agree on main IC_Fail/Terminal_Fail
predictions; third FP32 sample has an additional low-score IC_Fail not retained in
compiled output. This is not exact model equivalence or ground-truth accuracy.
Live class presence: IC_Pass4/70 (scores.076/.187/.500/.146), Terminal_pass1/70
(score.076); IC_Fail and Terminal_Fail each69/70. No operator-state ground truth
was collected for these70 frames. In particular do NOT claim unwanted pass-class
predictions solved from historical validation. Need inspect matched source frames.

Full repository suite307 passed, two existing dependency deprecation warnings.

## Approved endpoint repeat — bounded video delivered

Owner approved172.21.30.230:5010. Trials03–06 remain as failure evidence:
03 refused old endpoint allowlist;04/05 still no media detected after retarget;
06 exposed a parse error in new instrumentation, fixed before07. Synthetic
camera-free H264 file and10-frame RTP decode both succeeded (evidence in04).

Trial07 added explicit appsrc200ms buffer duration plus encoded H264 file tap:
43 processed/encoded frames,41 located/2 unknown,31.35s,1.37 processed FPS.
Receiver remuxed30 frames, then a separate FFmpeg decode exited0 and decoded all30
with0 offline duplicates/drops. This is NOT a no-loss live-network measurement.
`received-detections.png` visually shows missing-component boxes mapped to rotated
source sites; `received-first.png` shows explicit UNKNOWN, never stale boxes.
Clip: `../IC_TERMINAL_TRAIN/crop-stream-diagnostic-20261116-07/received.mkv`.
OpenH264 warning alone did not prevent synthetic or final delivery; exact causal
fix not isolated because buffer duration/instrumentation changed together.
Deployed scripts downloaded to `source-snapshot/`; later RTP-counter buffer-list
support is untested on board and does not reinterpret trial07's partial counter.
All jobs stopped, no graphs at08:31 UTC.308 tests passed, two existing warnings.

## Next

The endpoint approval and initial receiver decode are complete. Retain failures
and successful clip. Next collect operator-confirmed states/rotations and
investigate extra present predictions, then optimize the slow software path. Investigate extra present predictions on matched frames without
post-hoc threshold changes. Optimize native crop/packing/encoding later; do not
advertise this software diagnostic as native30fps streaming.
