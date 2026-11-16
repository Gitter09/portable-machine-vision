---
type: operations
title: Current Status
description: The live state of the portable station — what is staged, what is running, what is next.
status: active
tags: [status, state, handover]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T12:00:00Z
sources:
  - board://portable-release-20261113-r24
  - board://ic-terminal-candidate/etc
---

# Current status

**Read this before touching the board.** It records state that is not derivable
from the code, and the drift that has already bitten once.

Last updated: 2026-11-16.

## GitHub handoff completed

Initialized main and pushed component-based commits to
`https://github.com/Gitter09/portable-machine-vision.git` using configured owner
identity, no co-author trailers or historical date fabrication. Credentials/local
selectors ignored, known-secret scan clean. Fresh isolated local clone verifies55
payload files and18 offline wheels, root launcher help passes. Mac guide:
`MAC_TESTING.md`. Actual Mac execution remains owner's validation task.

## Owner requirement: CLONE-ONLY runtime

All runtime assets and offline wheels copied INTO repository `deployment/portable-v1`
(55 verified files/~210MiB uncompressed). Root `setup-host.py` creates
`local/host-env` from those wheels; root `run-portable.py --board-host <IP>` uses
that payload/example profile, prompts for SSH secret and writes to `local/runs`.
No sibling data/MODEL/fixed-station runtime dependency on this path. All bundled
files<100MiB and explicitly unignored for ordinary Git, no LFS/external download.
Original Windows selectors/old evidence paths are historical, not Mac launch path.
Platform Python3.11/OS/board SDK remain prerequisites. Mac execution/fresh clone
validation and Git initialization/chunked push still pending.
[Clone setup](../../deployment/README.md).

## Mac/private-link portable package BUILT; actual Mac test pending

Package `../IC_TERMINAL_TRAIN/releases/portable-vision-mac-link-20261116-v1.zip`,
188.7MB/55 verified files, SHA256
`4dc3ee22c73c8ecf93449b9877df5a63264424b08240e2bcac6e8e456a8ac809`.
Includes Linux native library/model and18 offline wheels (9 Windows +9 Mac arm64),
platform-neutral launcher/setup and private-link instructions. No credentials.
Ready-state exact-owned-worker SIGINT stop passed (worker/session stopped, host0),
immediate restart passed. Frozen package Windows smoke also passed:566 decoded
frames/~30.11FPS,479 metadata/471 matched AT ARRIVAL,0 RTP gaps/errors. All UNKNOWN
without PCB: not inference speed/quality validation.380 tests pass/two warnings;
portable55-file/current production42-file hashes pass unchanged after test.

Actual M3 install/macOS Ethernet/firewall/browser test STILL required. Board network
settings unchanged. Production selector remains accepted Windows r2; no automatic
promotion of untested Mac candidate. [Setup/carry instructions](mac-private-link-20261116.md).
Git initialization/chunked commits/push remain pending as separate task.

## Mac/private-link portability preparation history

Owner has M3 Air and reports macOS27.0; existing switch confirmed. Read-only
board inspection: sole Ethernet end0 has BOTH192.168.1.20/24 and172.21.30.120/24;
camera192.168.1.10 RTSP OPTIONS200. Network settings/services NOT changed.
Mac can join isolated switch with compatible private Ethernet address; summit
router not needed. Actual Mac setup/stream test still pending.

Experimental source now route-selects host IPv4, binds SSH to that address,
checks SSH identity via pinned prior board key, configures camera/board through
external profile, and removes fixed receiver allowlist in favour of private IPv4.
Windows auto-route video test747/747 matched,~30 decodedFPS but ALL747 UNKNOWN
(no PCB), so NOT an inference/performance validation. Native-arm64 CPython3.11
Mac wheels downloaded offline, not executed on Mac. `run-portable.py` and host
setup helper added; no new portable package/promoted selection yet.

Owned-worker SIGINT/stop cleanup prototype caught startup-before-ready edge case;
initial test stopped during SDK preflight, not camera run. Missing-file download
handling fixed to avoid empty status artifacts; stop/restart retest still pending.
Production r2/frozen fallbacks unchanged. Git initialization/push still pending;
local secrets ignored and wiki plaintext passwords removed BEFORE planned import.

## Production owner stream test — 2026-11-16 12:18 UTC

Started300s run through DEFAULT `run-production.ps1` (~12:23 UTC end), selected
frozen r2, not experimental source. Session
`../IC_TERMINAL_TRAIN/production-runs/coordinate-pose-20261116-121859-188/`.
Panel `http://127.0.0.1:58078/`; verified running/matched frame15, crop located.
Check bound/session before new camera graph; may expire before reading. No release
changes, await owner feedback.

## CURRENT operational production selection

**`portable-vision-production-20261116-r2`**, owner approved. Start repo
`run-production.ps1`; selection `config/production-pipeline.json` pins immutable
package/manifest/native hash and isolated `production-envs/...-r2` Python.
Feature ROI + pose + C++8 + coordinate-only browser are frozen together. Package
`../IC_TERMINAL_TRAIN/releases/portable-vision-production-20261116-r2.zip`,119.2MB/
42 files, includes offline dependencies/native binary/source; no credentials.

Packaged r2 smoke through production launcher:468 actual model calls,15.09/s
including startup,30.06 host decoded FPS,468/468 matches,0 RTP gaps/errors/unknowns.
Prior full confirmation15.95/s. Selected only AFTER packaged smoke/source/hash
verification; prior selection retained. Dedicated offline environment/pip check
pass;364 tests pass. Latest fallback40-file verifier still passes unchanged.

**Sessions remain bounded300s, not a new always-on service.** Smoke completed;
active selection does not mean currently streaming. No r33/vendor/SDK/service/
fixed-station changes. Owner acceptance is operational, not independent accuracy/
optical-latency/soak qualification. Frontend lifecycle/integration is next.
[Production release and rollback guide](production-pipeline-20261116.md).

Earlier experiment sections below are historical; they do not override this selection.

## Crop candidate compilation — 2026-11-15 18:12 UTC

Owner authorized training before validation preparation. Separate candidate
`../IC_TERMINAL_TRAIN/crop-candidate-20261115-164105/` completed 100 epochs;
final `last.pt` exported, raw/fused FP32 parity passed on three states.
Compilation completed **18:04:22 UTC** (started 17:48:57 UTC; about 15m25s).
ModelSDK 2.0.0/gen2 MLA INT8 produced hash-verified ELF/MPK. MPK source checksum
matches raw ONNX `a488c6f7f6c0510d87b0f22de25cc2c10381e63a222a84c582c4569948e673dc`.
Offline `board-configs/` regenerated from THIS MPK: input tile1280/64/3,
DFL64/class4 grids160/80/40, 2,688,000-byte physical output, raw-logit sigmoid.
These are tensor-contract drafts, **not a deployment-ready crop graph**: generated
preprocessing still describes full-frame input; serving must implement the frozen
hybrid crop plus pad114/RGB/255 consistently with training/calibration. No board
execution, native parity, performance or held-out accuracy qualification yet.
Existing r33/live state unchanged and not rechecked. Calibration used90 base
TRAIN views only. Historical CSV val-loss placeholders are not measurements.

## Reserved historical validation review prepared

`../IC_TERMINAL_TRAIN/cropped-validation-review-20261115/review.html` contains
31 native crops/62 UNREVIEWED transferred boxes from all32 reserved historical
VAL sources. `val-present_003540` is quarantined: no plausible grouped substrate;
include it in the crop-coverage denominator, not a silent successful exclusion.
Class counts22/20/9/11. Frozen hybrid crop, deterministic1280 resize/pad114,
no augmentation, no test consumption. All31 model inputs independently regenerated
pixel-exact and all62 draft model labels exact; five new guard tests passed.
Returned root `cropped-validation.review-export.json` validated: all31 images/
62 boxes reviewed by H, zero box corrections; source hashes, identity, classes,
geometry, original split and transforms match. Export SHA-256
`a70a1cec9d1199099d4334fa45a6ddaea64da8b736be288d505cdb8c6d351190`;
audit `review-validation.json` in the pack. No scoring or evaluation-ready YAML yet. Source recordings overlap training groups: diagnostic, not independent
production accuracy. No board changes.

## Reviewed crop validation FP32 evaluation

Final gallery/report: `../IC_TERMINAL_TRAIN/crop-validation-evaluation-20261115-final/`.
Fixed diagnostic confidence.05/NMS.45/match IoU.50/top100, no tuning or test use.
New FP32 candidate: **59TP/3FP/3FN out of62 reviewed boxes**, precision/recall95.16%
conditional on31 located crops. Per class: IC_Fail22/22, Terminal_pass20/20,
IC_Pass9/9, Terminal_Fail8/11. Three terminal-missing predictions have the correct
class but insufficient localization overlap (.4611/.4811/.4048) at frames
005001/005274/005728. No incorrect present-class detections on reviewed missing
states at these settings. Crop coverage remains31/32 (96.875%); unknown is not pass.
Old model ON THE SAME CROPS:0TP/6FP/62FN, not its full-frame runtime baseline.
Historical correlated validation is NOT production accuracy or compiled parity.
Raw outputs, reviewed snapshot, per-frame failures and overlays retained.
Eight focused tests passed. Visually inspected all three failure overlays:
new Terminal_Fail boxes cover the actual empty terminal footprint but are tighter
than the reviewed boxes, which include surrounding PCB/background. Scores
.928/.974/.914; all three predicted boxes are contained in their reviewed boxes.
This supports a box-extent mismatch, not an observed missed missing-state or
wrong-object prediction in these three frames. Ground truth and strict metrics
remain unchanged; do not retrospectively loosen IoU or relabel to improve score.
No board access/deployment. Next: qualify compiled outputs/serving-compatible
crop, plus fresh sessions; these three frames alone do not justify retraining.

## Compiled crop replay — 2026-11-16

Portable board process preflight at07:39 UTC found no camera/worker/feed graphs.
Isolated `/media/nvme/compiled-crop-replay-20261116` fixture ran31 exact reviewed
model inputs on actual MLA with installed SDK software transforms, exit0; no
camera decoder, live release, service or vendor changes. Host evidence
`../IC_TERMINAL_TRAIN/compiled-crop-replay-20261116/`; scored report/gallery
`../IC_TERMINAL_TRAIN/compiled-crop-evaluation-20261116/`. Verified downloaded
output/input hashes and compiled ELF/MPK lineage. Fixed .05/.45/.50/top100.
FP32:59TP/3FP/3FN; compiled:58TP/3FP/4FN, precision95.08%, recall93.55% on
62 reviewed boxes. Coverage31/32 historical sources; correlated, not production
accuracy.61/62 FP32 retained boxes match compiled at IoU>=.8023. Differences:
- present003712: IC_Pass localization IoU.5778→.4957 (score.9977→.9840).
- terminal_absent005001: Terminal_Fail localization.4611→.5007, score.9275→.4158.
- terminal_absent005456: Terminal_Fail dropped; FP32 retained score.5326, compiled
  maximum class probability anywhere.016625, below unchanged.05 threshold.
Thus real quantization sensitivity remains, not exact FP32 equivalence. No
wrong-present predictions on reviewed missing states. Mean MLA3.80ms, software
input/output transforms106.36/98.64ms: NOT native streaming FPS. Next implement
matching hybrid crop/pad114 preprocessing plus reverse coordinate mapping in an
isolated runtime candidate, then native parity/end-to-end stream. Do not run
these crop-trained weights through r33's full-frame graph or deploy as qualified.

## Crop streaming diagnostic — 2026-11-16

See [bounded crop-stream trial](crop-stream-diagnostic-20261116.md). Separate
software crop/MLA/inverse-coordinate-overlay implementation processed70 camera
frames,69 located/1 unknown, no worker errors, **1.51 processed FPS** (not native
30fps). Three source/model/head snapshots reconstructed pixel-exact using actual
board crop quads; geometry/NMS tests and full suite307 passed. OpenCV4.6-compatible
per-class NMS implemented in portable diagnostic only. Extra present-class
predictions persist in live frame logs; no physical-state accuracy claim.
Owner approved **172.21.30.230:5010**. Trial `crop-stream-diagnostic-20261116-07`
proves bounded end-to-end video:43 processed/encoded frames,41 located/2 unknown
in31.35s (1.37FPS);30 received frames independently decoded, FFmpeg exit0.
Received rotated boxes and UNKNOWN overlay visually checked. Explicit appsrc
buffer duration and encoded-file tap were added after failed retries; root cause
was not isolated. OpenH264 warning also occurs in successful synthetic tests.
All trials stopped;08:31 UTC inspection found no worker/camera/feed graphs.
Full suite308 passed. Still a slow diagnostic, not native30fps/accuracy acceptance.
Clip: `...-07/received.mkv`; deployed sources preserved in `source-snapshot/`.
Later RTP buffer-list counter fix is not exercised in trial07.
No r33, service, installed vendor, copied SDK or fixed-station changes.

## Smooth video / independent latest inference — 2026-11-16

Owner selected smooth-video/latest-frame first, favourable-region inspection next,
profiling/optimization later. Implemented separate compressed-camera passthrough
and sole-decoder/latest-only AI branch; boxes shown ONLY on their corresponding
inspection snapshot in loopback browser panel, never stale overlays on live video.
Initial `latest-frame-stream-20261116-01`:70 inspections/68 located/2 unknown,
~1.69 inference FPS;1184 video access units in41.48s including startup/shutdown.
Received600-frame720p clip advertises30FPS, independent decode exit0 (one null-mux
DTS warning). Video smoothness improved, not inference throughput or acceptance.

Owner viewing session `../IC_TERMINAL_TRAIN/latest-frame-view-20261116-01/` started
with180s bound and has now **completed**:324 inspections/316 located/8 unknown,
1.79 inference FPS and29.67 forwarded video FPS, no worker errors. Actual PIDs/
start/panel URL in session.json; HTTP200 and updating results were verified.
Check current processes before starting any other camera graph. Endpoint172.21.30.230:5010, r33 untouched.313 tests passed.
[Details and caveats](latest-frame-video-20261116.md). No favourable-region gate,
tracking, confidence tuning or profiling optimizations implemented yet.

## Frozen latest-frame fallback packaged

Selected package: `../IC_TERMINAL_TRAIN/fallbacks/latest-frame-20261116-v2/` plus
ZIP/.sha256, ~31.4MB,22 manifested files. Frozen successful owner-view runtime,
compiled ELF/MPK, reference ONNX, panel/launcher and evidence; no credentials.
`run.ps1` verifies hashes and uses frozen local fixture, fresh board/output dirs,
external credentials/dependencies; no old board replay copy required. Default180s.
Offline extraction/help/integrity/PowerShell checks passed;317 tests passed.
No board access/restart during packaging; restore path not newly end-to-end tested.
Do not modify this package for experiments. Preliminary v1 retained, use v2.
[Restore instructions](latest-frame-fallback-20261116.md).

## Profiled optimized candidate — 2026-11-16

[Performance report](latest-frame-performance-20261116.md): baseline1.65/s to
**7.99 inspections/s** over61.08s (488 processed,479 located/9 unknown),~4.85x.
Video remains independent;600 received720p frames decoded at advertised30FPS,
board forwarding28.73FPS including startup/shutdown. Two null-mux DTS warnings;
not a zero-loss, render-latency or soak claim. No worker errors. Final bounded job
stopped;09:43 UTC check found no graphs.

Opt-in `--optimized --fast-transforms`: exact OpenCV LUT quantization/packing,
class-first sparse unpack/dequant/DFL and only selected crop-method work. Startup
SDK byte/tensor/reference decode checks passed;31 matched reviewed replay retained
sets/classes/scores/boxes exact,32 VAL crop-selection records exact.326 tests pass.
No threshold/model/calibration/SDK/vendor changes. Fallback v2 reverified unchanged;
default remains unoptimized. No favourable-region gate yet. Further bottlenecks:
crop selection~42ms and JPEG publication~31ms per located frame in final run.

## Owner optimized live viewing — 2026-11-16 09:53 UTC

Owner requested viewing before further changes. Started optimized candidate with
300s bound (~09:58 UTC end), approved172.21.30.230:5010; FFplay smooth camera and
separate snapshot panel `http://127.0.0.1:61821/`. Session/evidence:
`../IC_TERMINAL_TRAIN/optimized-live-view-20261116-095325/`. Verified running,
HTTP200, snapshot58 located, processing112.8ms at observation. Check session and
processes before any other camera graph; may expire before reading. No further
experiments until owner feedback. Diagnostic candidate, not production acceptance;
fallback/r33 unchanged.

## Optimized fallback packaged — 2026-11-16

Owner accepted optimized viewing. New selected fallback
`../IC_TERMINAL_TRAIN/fallbacks/latest-frame-optimized-20261116-v2/` and ZIP/SHA,
24 verified files/~31.4MB. Exact owner-view board sources and model, explicit
optimized/fast-transform flags, independent of future experiment files. Earlier
slower fallback untouched and reverified.328 tests; clean extraction/help/PS parse
passed. Credentials excluded; no new stream during packaging.

Found owner host-controller Windows JSON replacement lock: host finally closed
receiver/browser server early; session.json remains stale running. Board completed
its300s bound independently; recovered report2820 iterations/1928 located/892
unknown,29.77 forwarded video FPS, no worker errors. Loop rate9.37 includes cheap
unknowns, not a new model benchmark. Packaged host-only retry fix preserves stale
snapshot and retries instead of crashing; tested but not new end-to-end deployed.
[Details](optimized-fallback-20261116.md). No active owner preview from that job.

## Coordinate browser + bounded pose reuse — 2026-11-16

Owner requested browser coordinates and pose reuse, no favourable-region gate or
C++ yet. Implemented board JPEG/overlay removal, off-thread latest-only SHA/UDP
publication (5012), host camera decode (5010), browser canvas on EXACT matched
NV12 frames. Live camera remains separate; no stale boxes on current uninspected
video. Unmatched/stale/disconnected clears overlays. Browser rendering verified
with headless Edge screenshot. New host browser-view-env; existing envs unchanged.

Coordinate-only40s9.33 inspections/s; tracking40s12.22/s; final60s **12.59/s**
(769 processed/761 located/8 unknown),~1.58x prior8FPS candidate.652 tracked/
117 full attempts; one low-confidence event re-localized. All769 metadata results
matched decoded frame hashes; host decoded30.01FPS, zero final RTP sequence gaps.
P99 arrival gap69ms/max703ms: not a zero-jitter or optical-latency guarantee.
31 host raw-frame audits (25 tracked) agree with source hashes and preserve fresh
selector blue seed coverage>=.99999997; NOT human site/accuracy qualification.
Tracking changes crop pose and needs real stress validation.340 tests passed;
both fallback hashes pass unchanged. No model/threshold/calibration/SDK changes.
All jobs ended;10:33 UTC check found no graphs. See
[complete results](coordinate-browser-pose-20261116.md).

## Owner coordinate viewer — 2026-11-16 10:43 UTC

Opened300s browser trial (~10:48 UTC end):
`../IC_TERMINAL_TRAIN/coordinate-owner-view-20261116-104346/`.
Panel `http://127.0.0.1:62098/`, live camera left/exact-matched inspection right,
pose reuse enabled, receiver172.21.30.230:5010/5012. Verified running and matched
frame4/located. Check session/processes before another graph; bound may expire.
No further changes pending owner feedback; fallback/r33 unchanged.

## LATEST fallback: browser coordinates + pose tracking

Owner accepted view. Frozen package:
`../IC_TERMINAL_TRAIN/fallbacks/coordinate-browser-pose-20261116-v1/` and ZIP/SHA,
119.6MB/40 files. Exact accepted runtime/browser sources, compiled model, evidence,
9 pinned offline host wheels, setup/run launchers, no credentials. Fresh extraction,
isolated OFFLINE dependency install/H264 codec/help/reverification and PowerShell
parsers pass;340 tests pass. Older fallbacks reverified unchanged. No new board
run or C++ fusion during packaging; C++ remains next separate experiment.
[Restore guide](coordinate-fallback-20261116.md).

Owner coordinate view300s subsequently completed:4077 loops/3862 located/215
unknown,3317 tracked/760 full,13 confidence re-localizations, no worker errors.
Original session-in-package records state at packaging, not later completion.
No owner preview from that completed bounded session; verify processes before
new graph. Latest accepted configuration is~12.6 inspections/s/~30 video baseline,
not a new production accuracy claim.

## C++ input-fusion candidate — 2026-11-16

Implemented portable-only C ABI/ctypes fusion: prepared BGR→exact quantization LUT
→compiler tiles, no intermediate RGB/quantized images. No model/resize/pad or
threshold change. Naive single-thread/NEON slower; bounded OpenMP8 selected.
Six implementations byte-identical on31 prepared reviewed inputs; all256/random
patterns, output margin guards and invalid lengths pass. Explicit library SHA/
ABI/layout and startup parity required; no C++ flag = existing fast default.

Camera-free8.70→3.24ms. Sequential live input stage8.92→3.81ms and whole-run
12.44→13.34 inspections/s (~7.3% observed); host decode~30FPS,549/549 matched
metadata frames, no RTP gaps/worker errors. Short sequential trials, not guaranteed
uplift, independent accuracy or optical latency.346 tests pass. Frozen40/24-file
fallback checks pass unchanged; prototype NOT packaged/default. No SDK/vendor/
r33/service/fixed-station changes. Jobs stopped;11:15 UTC process check clear.
[Full results](cpp-input-fusion-20261116.md). Source/build artifacts and rejected
kernel measurements preserved; no output-side fusion/tracker rewrite yet.

## Tracker feature ROI candidate — 2026-11-16

Target achieved in bounded confirmation: **15.95 actual model inspections/s**
(974 located/0 unknown over61.05s), host decoded30.03FPS,974/974 exact frame
matches,0 RTP gaps/errors. Uses prior C++8 input kernel in BOTH live arms;
baseline13.22/s, first ROI15.65 processed/15.31 actual model calls/s (14 unknown).
Sequential scene trials, not guaranteed throughput/production accuracy.

Feature computation only inside PCB seed bounds+8px halo, same full mask and
80/.01/3/block3 detector parameters/global points, no inspection-region gate.
31 board camera audits ordered points exact; live seeding93→4.3ms. Final39 host
raw audits ordered points equal and fresh-seed coverage>=.999999956, not human
site/accuracy qualification. Same tracking guards and0.5s refresh threshold.

Opt-in `--feature-roi --track-pose`, defaults/fallbacks unchanged; builder refuses
unfrozen ROI/native flags.353 tests pass/two existing warnings;40/24-file fallback
verifiers pass. No model/calibration/threshold/SDK/r33/service/fixed-station edits.
All runs stopped;11:39 UTC process check clear. [Full report](tracker-feature-roi-20261116.md).

## Owner feature-ROI viewer — 2026-11-16 11:44 UTC

Opened300s browser trial (~11:49 UTC end), feature ROI + C++8 + pose reuse:
`../IC_TERMINAL_TRAIN/feature-roi-owner-view-20261116-114404/`.
Panel `http://127.0.0.1:54588/`; verified running/matched frame10, crop located,
59ms processing. Left live video/right exact-matched inspected frame for owner
quality/perceived-delay review. Processing time is NOT optical latency. Approved
receiver172.21.30.230:5010/5012, fallbacks/r33 unchanged. Verify session/expiry
before any other camera graph; await owner feedback, no further changes.

## Ready to use

| Thing | State |
|---|---|
| Camera `192.168.1.10` | up, answering from the board |
| Capture feed | **Stopped at last board check**. Preview occupied the decoder then; offline work has not rechecked live state. Verify exclusive decoder ownership before any new graph |
| Feed page | `http://172.21.30.120:8080/` |
| Legacy board candidate configs | RGB / conf **0.05**, grouped 64-channel DFL, but **stale tiles**. Do not reuse without regeneration. Corrected trial is r33 |
| Workstation receiver | Coordinate/pose owner view **completed300s bound**, no worker errors. Target **172.21.30.230:5010/5012**; latest fallback packaged, no new stream during packaging. Check processes before next graph |
| Portable launcher | `runtime/launcher/` in this workspace — the code the releases actually run |

The threshold in the candidate tree is 0.05 because the decode investigation needs
the full score distribution. The CLI defaults to 0.25; the example in
[deployment](deployment.md) explicitly passes `--threshold 0.05` for diagnostics.
Keep the distinction deliberate. Do not ship 0.05.

## What is staged on the board

| Release | Meaning |
|---|---|
| `r8` | Historical YOLOv8m at 640, in-graph decode; first proven box path (27 fps). r33 now also has controlled native/Python box agreement |
| `r20`–`r22` | 1280 DFL experiments, before the geometry fix |
| `r23` | 1280 DFL, first geometry fix. Failed to stage/run cleanly (asset hash) |
| `r24` | 1280 DFL, geometry fix. Ran 1181 frames — the measurement baseline |
| `r25` | 1280 DFL, **staged with the rejected BGR variant**. Do not reuse or trust |
| `r26` | Timing baseline: 1321 frames, zero valid timed frames; timing errors remain |
| `r27` | First timestamp patch: 879 frames, zero valid timed frames. Later baseline retry refused: `etc/0_preproc.json` hash differs. Do not bypass verification |
| `r28` | Initial timestamp patch: short success, but unbounded and frame-only storage; superseded by r29 |
| `r29` | Bounded, stream-aware timestamp storage. Two 300-second local diagnostics pass; real clock-paced output is not proven by these tests |
| `r30` | Workstation destination selected; absolute-time PTS blocked real UDP delivery. Superseded by r31 |
| `r31` | Clock conversion fixed; input/output tile layouts still wrong. Historical diagnostic baseline |
| `r32` | Input tiles corrected only. Scores restored, output box geometry still mismatched |
| `r33` | Input/output tiles from compiler. Native/Python same-frame agreement and **29.5 fps annotated stream** to workstation; diagnostic conf 0.05, not accuracy-qualified |

Run artefacts for `r24` exist locally under `..\IC_TERMINAL_TRAIN\verify-r24`;
`verify-r10`–`verify-r22` exist on the board. `verify-r24/worker.json` records
`state: error`, `frames: 885`, `valid_timed_frames: 0`.

## The measurement that defines the defect

Release `r24`, 1181 frames, conf 0.05:

| Metric | Board | Host, same scene |
|---|---|---|
| Boxes per frame | 0.4 | 5.8 |
| Max score | 0.217 | 0.986 |
| Median box | 32×30 px | 22×21 px |
| Negative heights | 32.7% | n/a |

At conf 0.5 the board returns zero detections across 1785 frames.

## Open decisions

1. **Native raw DFL now works in r33.** Controlled buffer captures show native
   box agreement with the independent decoder after both layout fixes. Completing
   a canonical vendor export is optional, not a prerequisite for the first stream.
   Labelled historical scoring is now available below; live physical inspection
   acceptance is still open. Do not reuse the old full-frame tile assumption.
2. **Timing — five-minute diagnostic trials pass, not full acceptance.** The
   initial `r28` test passed 426 frames after restarting `simaai-appcomplex`,
   `decoder`, and `encoder` (2026-11-14 10:06 UTC). Earlier startup aborted with
   `Unable to connect to the server from dispatcher`; the exact failed service
   was not isolated. The safer `r29` then passed two 300-second diagnostics
   without another service restart. r31 subsequently passed a bounded live
   decode test after clock conversion (details below). Human-observed playback,
   recovery, optical timing and all-day stability remain unverified;
   `application_qualification` remains `not_established`.

## Dataset state

`dataset/v4-1280` was captured at the **old** camera distance. The camera has
since been re-aimed. Earlier 0-at-1280/2-at-640 observations predate the layout
repair and do not establish current camera-transfer failure on their own.
All 31 historical test images have now been scored against FP32 and the compiled
model (57/62 matched labels each). This is not current-camera acceptance.
A fresh known-specimen trial needs an operator before deciding on retraining.
The dataset does contain all four classes, including 43/11/10 Terminal_Fail
train/validation/test boxes—correcting the earlier missing-class handover.

Excluded frames, already recorded: `frame-003643`, `frame-005871` (mislabeled).

## Known drift to watch

- **Two candidate directories exist.** `..\IC_TERMINAL_TRAIN\candidate\etc` on the
  workstation was the standard RGB configs; the board's
  `/media/nvme/ic-terminal-candidate/etc` was left holding the rejected BGR
  variant after the last experiment. It has been restored to RGB (verified
  2026-11-14), but this is exactly how a rejected experiment gets re-run by
  accident. After any variant test, restore the candidate directory.
- **Releases are immutable in intent, not in fact.** Staging re-hashes files but
  does not notice a recycled release number.
- **Multiple capture supervisors can accumulate.** Repeated test restarts left
  three supervisors and seven capture processes competing for the decoder. Check
  with:

  ```sh
  pgrep -af feed_supervisor | grep -v pgrep | wc -l   # expect 1
  pgrep -x gst-launch-1.0 | wc -l                     # expect 1
  ```

## How to resume

```sh
# 1. free the decoder
for p in $(pgrep -f "^/bin/sh /tmp/feed_supervisor.sh"); do kill $p; done
pkill -x gst-launch-1.0

# 2. confirm, then stage and run (see operations/deployment.md)
pgrep -x gst-launch-1.0 | wc -l      # expect 0
```

```sh
# 3. afterwards, hand the camera back
nohup sh /tmp/feed_supervisor.sh > /tmp/supervisor.out 2>&1 < /dev/null & disown
```

## Timestamp safety review and applied repair

An offline exercise of the actual `worker.py` probe found two defects in the
provisional timestamp map:

- After 10,000 simulated completed frames, all 10,000 timestamp entries remained.
  Encoder lookup never removes them, and there is no capacity limit.
- The key is only the frame number. In a simulated overlapping-stream case,
  stream A frame 1 received stream B frame 1's timestamp (200 instead of 100).
  This demonstrates a matching weakness, not an observed live-board stream mix-up.

With approval, repaired these issues in `runtime/launcher/{support,worker}.py`:
keys are `(stream_id, frame_id)`, encoder lookup consumes entries, outstanding
storage is limited to 512, and only decoder output supplies entries. Invalid or
conflicting source metadata stops the run; missing entries never invent time.
Added 34 timestamp regression tests; the full suite passes **94 tests**.

### r29 board evidence

Model, configuration and deployment inputs were checked against r28 before
staging; unchanged. Only launcher code changed. Existing releases were not
modified, and no checksum gate was bypassed.

| Metric | First 300-second request | Second 300-second request |
|---|---|---|
| Board output directory | `/media/nvme/ic-terminal-verify-r29-soak` | `/media/nvme/ic-terminal-verify-r29-soak-memory` |
| Frames / valid timed frames / RTP markers | 8,978 / 8,978 / 8,978 | 8,983 / 8,983 / 8,983 |
| Steady observed output rate | 30.14 fps | 30.16 fps |
| Errors, warnings, cache misses, evictions | None | None |
| Cache peak / capacity | 30 / 512 | 30 / 512 |
| Cache pending at shutdown | 1 in-flight frame | 2 in-flight frames |
| Shutdown | Clean, exit 0 | Clean, exit 0 |

Both raw captures pass `verify.py media` and both frame logs pass the
`run_integrity_passed` checks in `verify.py run`. Full readiness deliberately
remains false: numerical accuracy, image/result pairing and real receiver
playback were not tested.

The first memory sampler used `/proc/<pid>/task/<pid>/children`, which this kernel
does not provide, so **it collected no memory evidence**. The second run used
`ps -eo pid=,ppid=,rss=,args=` to select the worker by launcher parent PID and
sample RSS every five seconds. After the first 30 seconds, RSS was
61,252–61,636 KiB (first to last increase: 384 KiB). This excludes the original
per-frame cache accumulation; it does not prove zero growth over hours or days.

Small evidence files are copied locally to
`../IC_TERMINAL_TRAIN/verify-r29-soak` and `verify-r29-soak-memory`, including
`timestamp-soak.json`, worker/supervisor/health reports, frame/detection logs,
release inventory, and media/run verifier results. Large `packets.bin` captures
remain on the board. The run's local `fakesink` sends no network video.

## Receiver test — workstation selected; output clock mismatch found

The pinned r29 deployment sends video to **192.168.1.50:5010**, but that address
currently fails neighbour resolution on the board (`ip neigh`: `FAILED`) and
does not answer ping. Starting the network run unchanged would not test playback
on this workstation.

This workstation currently has **172.21.30.13**, not 192.168.1.50. The board can
ping it, and a temporary UDP listener on 172.21.30.13:5010 received a 30-byte probe
from 172.21.30.120. FFmpeg/FFplay 8.1 are available locally. This proves a small
UDP datagram can arrive, not that an H.264/RTP stream can decode or play smoothly.
The temporary listener is closed; no firewall, IP, deployment or runtime settings
were changed, and no inference pipeline was started during this preflight.

The owner selected **this workstation** as receiver and explicitly reaffirmed
that correct model detections—not just video delivery—are the end goal. Staged
r30 using `/media/nvme/ic-terminal-receiver-deployment-r30.json`, changing only
receiver host to 172.21.30.13. The candidate deployment remains 192.168.1.50;
existing releases were not modified. Original model/config asset hashes and
launcher hashes match r29.

### Real network test and remaining timing blocker

A temporary Python UDP receiver bound 172.21.30.13:5010, captured raw packets,
and forwarded them to FFmpeg on loopback port 5012 for simultaneous decode and
recording. A 20-second non-diagnostic r30 request ended after about 17 seconds:

- Decoder and boxdecode each processed 305 buffers.
- Encoder and output-sink probes each observed only one buffer.
- Workstation received **zero packets**; FFmpeg could not determine video
  dimensions because no video arrived. This is not evidence of corrupt H.264.
- Worker recorded `output_timeout`, 1 valid timed frame, no RTP frame markers,
  cache pending 304, clean shutdown, supervisor exit 1.

The carried metadata timestamp was 179,755.773 seconds, approximately the board
monotonic uptime, not elapsed pipeline time. `udpsink` defaults to `sync=true`
and waits until pipeline time reaches a buffer's presentation time. The local
`fakesink sync=false` used by diagnostic runs never exercised that wait, which
is why those tests passed without proving real video delivery.

A separate board-only `appsrc ! udpsink` experiment (no camera/model) confirmed:

| Timestamp supplied | Sink clock pacing | Packet within 750 ms |
|---|---|---|
| Absolute board time (~179,945 s) | `sync=true` (default) | No |
| Pipeline-relative time (~0.05 s) | `sync=true` | Yes |
| Absolute board time | `sync=false` | Yes |

With approval, applied fixed-offset conversion in r31: the first source metadata
time anchors to the pipeline running time at first decoder output. Later frame
spacing is preserved exactly. Sink pacing remains enabled; no confidence/model/
decode settings changed. Clock origin/offset are recorded in health/run JSON.
The full automated suite now passes **111 tests** (51 portable timestamp tests).
Candidate launcher copies were uploaded and hash-verified; original model/config
source asset hashes still match r30. Existing releases remain untouched.

Evidence: board `/media/nvme/ic-terminal-verify-r30-receiver`, local
`../IC_TERMINAL_TRAIN/verify-r30-receiver`, including receiver/FFmpeg logs,
board reports, raw empty capture, deployment/release inventory, and
`sink-clock-test.json`. Temporary FFmpeg and sockets were closed; no capture or
inference pipeline remains running.

### r31 receiving and reference-comparison results

Receiver is this workstation, 172.21.30.13:5010. A temporary Python UDP listener
captures packets and forwards them to FFmpeg on loopback 5012. Two real-output
runs used r31 without further runtime changes:

- **60-second request:** board processed 1,776 source/detection frames and
  admitted 1,717 encoded frames, all timed, no reported errors. Receiver captured
  24,869 packets / 1,716 complete RTP frame markers; raw media verification
  passed with no issues. Reconstructed H.264 decodes to exactly 1,716 frames,
  1280×720. This run's initial dual-output FFmpeg harness rounded decoded time
  to a 30 Hz time base and logged repeated-DTS warnings; stopping it after the
  sender ended required termination and left its MKV incomplete. That MKV is
  **not** the accepted archive. Raw capture/reconstruction is the valid evidence.
- **35-second request, bounded decoder:** board admitted 969 timed frames,
  968 RTP markers, no reported errors, clean exit. FFmpeg used the 90 kHz RTP
  time base and a 600-frame limit: **600 live frames decoded, exit 0, no forced
  termination or repeated-DTS warnings**. The continued packet capture contains
  968 complete frames plus an incomplete final frame. The media verifier correctly
  fails with `incomplete_final_frame`; this limitation is not suppressed.
- Both board frame logs pass `run_integrity_passed`; neither grants full
  application readiness. No human-observed live playback test was performed.

The finite-run worker switches directly to NULL rather than draining EOS, so it
can discard queued source/encoder data or stop mid-packetised frame. Shutdown
integrity is a separate outstanding issue, not fixed by clock conversion.

Evidence: board `/media/nvme/ic-terminal-verify-r31-receiver{,-bounded}`, local
`../IC_TERMINAL_TRAIN/verify-r31-receiver{,-bounded}`. Local files include SDP,
receiver/FFmpeg/progress reports, board reports, packet captures and verifier
results. `received.h264` in the first directory reconstructs all 1,716 frames.
`viewing-30fps.mp4` is a complete **viewing-only** copy with nominal 30 fps
regenerated timestamps; do not use it as original timing evidence.

**Detection gap remains:** the first run produced 1,092 boxes over 1,776 detection
frames; 285 heights (26.1%) are negative under the documented signed reading.
The reference used `export-yolo11/yolo11n_fused.onnx` via ONNX Runtime CPU
(RGB /255, square 1280 letterbox with padding 114, conf 0.05, class-aware NMS
IoU 0.7). It shares 174 same-named initializers with the six-head export, all
identical. This is FP32 exported-model inference, not execution of `best.pt`.

Of 20 deliberately selected, evenly spread returned frames with no board
boxes, the reference produced detections in **17**, totalling **121 boxes**.
Frame 300 alone has zero board boxes versus five reference detections.
`reference-comparison.json` records model hash, settings, frame indices and
predictions; `reference-frame-*.png` visualises the reference boxes.

This is evidence of a board/reference mismatch, **not measured accuracy**:
selection favoured board misses, returned video is lossy, reference detections
are not labelled ground truth, and index-to-frame-id alignment relies on the
complete RTP capture and contiguous encoder identities rather than independently
proven optical pairing. Quantized and floating-point confidence scores are not
directly equivalent. No threshold, decoder or model changes were made.

## Vendor comparison — conversion bookkeeping blocker isolated

Consulted the owner's [SiMa documentation](https://docs.sima.ai/index.html),
specifically graph surgery, boxdecode and model-accuracy/accelerator-mode pages;
see [inference backends](../architecture/inference-backends.md#vendor-contract-comparison-2026-11-14)
for links and contract details. Compatibility remains the leading hypothesis,
not proof that training quality is sufficient.

Read the installed dedicated YOLO11 surgeon on the Admin PC. It rewrites the
attention block as well as the detection heads. Ran it unmodified against our
original `yolo11n_fused.onnx` in isolated directory
`/work/portable_vendor_reference_20261114` of the existing portable compiler
container `demeter_ic_terminal_8b4b5b8de783_20261112_153756`. It failed at
`simplify(model)` before completing the six-output conversion: an attention
consumer precedes its producer. No tensor producer is missing.

An offline **in-memory diagnostic only** established:

- Node dependency ordering alone is insufficient: old inferred tensor sizes
  then conflict with the rewritten attention layout.
- Dependency ordering **plus removing stale `value_info` metadata** passes full
  ONNX validation and executes in ONNX Runtime.
- For received `sample-300.png`, maximum difference from the original model is
  **0.000274658 pixels** in model-frame box coordinates and **7.152557e-7** in
  class probabilities. This validates only the intermediate attention rewrite
  on one input—not final-head surgery, quantization or board correctness.

The installed vendor tool, saved model files, existing compiler jobs and board
releases were not patched. Only a separate diagnostic workspace and output
artifacts were added. No new compile, model deployment, inference run or training
was performed. Proposed adapter work needs approval before proceeding.

Evidence: workstation `../IC_TERMINAL_TRAIN/vendor-reference-20261114/`
(`run_surgeon.py`, failed `surgery.log`/`surgery.exit`, original intermediate
`yolo11n_fused_mod.onnx`, `ordering-diagnosis.json`); corresponding Admin-PC job
folder `C:/Users/Admin/Desktop/ICTerminalBuilds/portable_vendor_reference_20261114`.

## Portable Python API diagnostic — software preparation succeeds

The owner approved borrowing the fixed station's **execution pattern**, not its
model or detection settings. Added `tools/analysis/portable_tensor_parity.py`:
prepare identical RGB float32 inputs, use the portable r31 ELF/MPK through SiMa's
Python API, then decode/compare six numerical outputs on the workstation. The
portable model, four class IDs, 1280 geometry and 64-value DFL heads are retained.
No fixed-station camera/ROI/640-grid/four-distance decoder is reused.

Prepared three inputs: received `sample-300.png`, validation
`present_003540.png` and `absent_010026.png` (neither validation file occurs in
this compilation's calibration list). Input SHA-256:
`be76597517fa3b7a729650d722fdb97e01fa7e956c6d12d4d807f623657884d6`.
The archive and manifest were transferred and byte-hash verified. New CPU
regression tests cover portable DFL math, layouts, labels and preparation;
full suite **123 passed**.

The CPU-only diagnostic decoder matches the original FP32 ONNX on all three
inputs within **0.000244141 model pixels** and **1.371664e-7 probability**;
reference detections at conf 0.05 / NMS 0.7 are 5, 3 and 2 respectively. This
validates the diagnostic's decoding logic, not device accuracy.

Two bounded board invocations (the second adding stage/config evidence) both
aborted with exit **134 / SIGABRT**, `*** stack smashing detected ***`. The
instrumented invocation locates failure precisely at:

```
SimaQuantTess(parser).configure({})
```

The model runner had **not been constructed**, the ELF had not been loaded by
that invocation, and no input sample had reached inference. No output tensors
or hardware parity report were obtained. The native quantizer configuration
specifies a 1280×1280×3 float input (19,660,800 bytes), INT8 tessellated output
(4,915,200 bytes), quant scale 255.12458585747225, zero point -128, compiler-derived
64×1×3 tiles. The exact native crash cause is not established; do not attribute
it to model accuracy, classes, a size limit or bad calibration without evidence.

Installed APIs are present: `sima.operations` in `/usr/lib/python3.11/dist-packages`,
`mpk_parser` and `ev_transforms.transforms` in `/usr/local/lib/python3.11/dist-packages`.
Plugin/MLA runtime packages report 2.1.3. The installed software-transform module
exposes `quantize`, `tessellation`, `detessellation`, and `dequantize`. This
provided the alternative route below, now implemented with compiler-defined
layout/scale/rounding rather than guessed native-CVU defaults.

Evidence: board `/media/nvme/ic-terminal-tensor-parity-20261114/`, local
`../IC_TERMINAL_TRAIN/tensor-parity-20261114/` (prepared input archive/manifest,
`reference-only.json`, `board.log`, `locate.log`, exit codes,
`board-locate/quantizer-config.json`, and `environment.json`). No release,
calibration, model, installed SiMa code or service configuration changed. No
service restart was performed; the three hardware services remain active and
no diagnostic/capture/inference process remains running. Hardware service health
beyond those status checks has not been requalified after the abort.

### Successful software-transform test (same compiled model)

With approval, extended the diagnostic to call installed SDK software
`quantize`, `tessellation`, `detessellation` and `dequantize`, leaving only
`SimaModelRunner` on the accelerator. It derives the chain directly from MPK
plugin dependencies, byte sizes and parameters. Input/output SDK tile/inverse
roundtrips pass before execution, including the class heads' physical 16-channel
padding. No native CVU quantizer/dequantizer is instantiated in this mode.
`--transforms software` is the default; native mode is explicit reproduction
only. Added eight tests (total **131 passed**).

The same three prepared inputs, unchanged input archive, r31 ELF/MPK, weights,
class IDs and calibration were used. The software-mode board process exited 0,
returning exactly the six expected float heads for each input. ELF/MPK/raw ONNX
compilation lineage and transferred archive/output hashes pass comparison checks.

| Input | FP32 boxes at conf 0.05 | Compiled + Python boxes at conf 0.05 | FP32 / hardware max score |
|---|---:|---:|---|
| Recent `sample-300.png` | 5 | 9 | 0.819 / 0.572 |
| Validation `present_003540.png` | 3 | 3 | 0.928 / 0.762 |
| Validation `absent_010026.png` | 2 | 2 | 0.904 / 0.802 |

Using class-aware greedy matching, all **8** reference boxes with score ≥0.3
have hardware counterparts at the diagnostic score threshold 0.05 with IoU
≥0.5. Actual matched overlaps are **0.855–0.965**. This is three-example
reference agreement, **not labelled-ground-truth accuracy**: the current image
still has extra low-score predictions and quantized confidence differs materially.
It proves that the unchanged compiled model can produce meaningful IC/terminal
predictions under the inspectable software preparation/output path. It does not
prove full precision/recall or eliminate quantization/calibration quality work.

MLA calls measured 3.7–4.2 ms; software input preparation took 108–115 ms and
output conversion 96–101 ms per input, excluding complete diagnostic overhead
and DFL/NMS. No live fps claim follows. This path isolates correctness, not
production throughput. No camera, GStreamer preprocessing or boxdecode plugin
participated in the test.

A static audit found differing layout descriptions: compiler input tessellation
`slice_shape=[1280,64,3]`, live preprocessor `tile_width=128, tile_height=32`, and
previous native quantizer-generated `tile_width=64, tile_height=1`. These
observations justify investigating layout handling, but **do not yet prove the
first failing live stage**; plugin parameter meanings and actual bytes must be
checked before changing a configuration. No live tiles were altered.

Evidence additionally includes `board-software/{board.json,outputs.npz,software-transform-contract.json}`,
`software.log`, `software.exit`, `comparison-software.json`,
`reference-agreement.json`, `layout-audit.json`, and `input_*-comparison.jpg` in
the local diagnostic directory. Board copies are under the matching
`board-software` directory. All diagnostic and camera/inference processes are
stopped; no production release, vendor installation or fixed-station file changed.

## r33: first working annotated native stream

The owner approved isolating the first live divergence and fixing that stage.
A bounded probe captured two matching frames at decoder, preprocessor and MLA
output, later extended to boxdecoder output. No old release was edited.

### Controlled evidence and repair

- r31 preprocessor output is two 4,915,200-byte regions: tiled input and companion
  untiled quantized RGB. Untiling with native HWC `[32,128,3]` reproduces the
  companion **100%** on both frames. Compiler HWC `[1280,64,3]` interpretation
  matches only ~20.7%. The mismatch is therefore real, not merely differently
  named parameters.
- Replaying each captured tiled input through the same ELF reproduces captured
  raw MLA output **byte-for-byte**. Repacking only those same quantized pixels
  to compiler order changes ~79.3% of byte positions and restores predictions.
  No new float preprocessing, model, weights, calibration or confidence change.
- Input-only r32 uses compiler input tiles. Ten-second diagnostic: 281 timed
  encoder frames, 1,987 boxes across 282 detection frames, max score 0.907,
  no negative heights/errors. But same-frame native box sizes still disagree
  with software decoding of the very same captured MLA output.
- r33 also sets MLA and boxdecoder physical slices from compiler metadata:
  box heads HWC `[2,160,64]`, `[1,80,64]`, `[1,40,64]`; class heads unchanged
  `[160,160,4]`, `[80,80,4]`, `[40,40,4]`. Logical tensor geometry remains
  160/80/40 with depths 64/64/64/4/4/4. On two captured r33 frames all eight
  native detections match class-aware software decoding within **0.473 pixels**
  and **1.50e-8 score**, consistent with integer-coordinate rounding.

`tools/build_ic_terminal_configs.py` now reads compiler input and output tiles,
not template values. The geometry guard compares logical tensors and physical
slices separately rather than falsely requiring `slice_* == input_*`. The CLI
rejects missing tile metadata before writing configs. Four regressions added;
full suite **135 passed**. A real regeneration in
`../IC_TERMINAL_TRAIN/native-layout-generated-20261114` produces the tested layouts.
The native DFL path now works; vendor export repair is not a prerequisite for
this first stream. Do not reintroduce full-frame box tiles as a geometry guard.

### Workstation video test

r33, 45-second requested run to 172.21.30.13:5010:

| Check | Result |
|---|---|
| Encoded frames / valid timed frames | 1,267 / 1,267 |
| Complete RTP frames received | 1,266 (18,307 packets) |
| Observed encoder frame rate | **29.52 fps** |
| Bounded live FFmpeg decode | **900 frames, exit 0**, no forced decoder stop |
| Raw RTP integrity verifier | Passed, no issues |
| Detection frames / boxes | 1,326 / 6,431 |
| Negative heights | **0** |
| Board outcome | Clean exit 0, no errors |

FFplay was also launched as a workstation preview, but it did not auto-exit and
was terminated after the finite test. Do not claim human-observed visual or
clean-player-shutdown acceptance from that run. The bounded FFmpeg decode and
saved annotated video are the independent receiving evidence.

**Open the annotated viewing clip:**
`../IC_TERMINAL_TRAIN/verify-r33-receiver/annotated-viewing-30fps.mp4`
(1,266 frames, 1280×720). `annotated-sample-600.png` is a still. MP4 timestamps
are regenerated at nominal 30 fps **for viewing only**, not latency evidence.
Raw `packets.bin`, verifier reports and worker logs remain available. The one
admitted-but-not-complete output frame and queued source tail still demonstrate
that finite-run EOS/drain handling is not accepted as lossless.

This is a working **debug** stream at threshold **0.05**, intentionally unchanged.
Class/position fidelity against an independent decoder is now demonstrated on
captured frames. Labelled historical precision/recall has subsequently been
measured below. Current-camera class quality, the operating threshold, recovery
and a long soak still need acceptance. The subsequent dataset audit also corrects
the earlier claim that Terminal_Fail had no captured examples. Do not use visible boxes alone as proof of correct labels.

### Artifact and candidate boundaries

- r32/r33 were staged from separate board experiment trees
  `/media/nvme/ic-terminal-{input,full}-layout-candidate-20261114`.
  The legacy `/media/nvme/ic-terminal-candidate/etc` and workstation's original
  candidate were deliberately not overwritten; their tiles are stale. Regenerate
  with the corrected tool before future staging, or explicitly run pinned r33.
- Receiver remains pinned to the approved workstation through the separate
  receiver deployment JSON. The legacy candidate deployment still names .50.
- Evidence directories locally and on board: `native-buffers-20261114`,
  `native-layout-control-20261114`, `native-buffers-r32-20261114`,
  `native-buffers-r33-20261114`, `verify-r32-input-layout`, `verify-r33-receiver`
  (board names prefixed `ic-terminal-`). Includes exact capture metadata,
  controlled replay, decoder comparisons, network logs and annotated artifacts.
- No installed vendor code, weights, classes, calibration, shared SDK, fixed
  station, or old release changed. All preview/receiver/board processes are stopped.

## Operator flow and first labelled test (2026-11-14 follow-up)

Owner approved repeatable viewing and known-good/bad evaluation. Added host-side
`tools/portable_live_view.py`: receiver first, pinned release, finite duration,
Q/Ctrl+C/receiver exit stops its own launcher, PID-identity checks before signal,
and refusal on busy board/reused output/wrong receiver binding. No services or
release changes. See [start/stop guide](../guides/live-view-and-labelled-check.md).

Hardware trials: 300 decoded frames, windowed duration stop, programmatically
requested Ctrl+C, immediate restart and 150 decoded frames. Both bounded FFmpeg
receivers and every board launcher exited 0 without wrapper errors. Final FFplay
clock advanced; it logged 96 dropped frames while host ONNX evaluation overlapped.
Desktop zero-drop playback, latency and human-observed visual acceptance remain
unqualified. All trials are stopped; the feed remains stopped.

`portable_labelled_eval.py` scored **all 31 historical test images / 62 boxes**
using fixed diagnostic conf 0.05, NMS IoU 0.45/top100, match IoU 0.50. Compiled
model: **57 TP / 16 FP / 5 FN, precision 78.1%, recall 91.9%**. FP32: 57/20/5,
precision 74.0%, recall 91.9%. Both fail the same five small IC_Fail boxes on
localisation, with correct-class nearby predictions (compiled IoU 0.334–0.439).
The other classes match all their historical test labels, on small correlated
samples—not a production accuracy claim. No confidence or model change.

A raw-annotation/dataset/training-metadata audit found actual Terminal_Fail
examples; the earlier no-examples claim was stale. Current-camera controlled
physical testing was blocked at that checkpoint; the owner subsequently supplied
a good specimen for the 2026-11-15 trial below. Historical frames share three runs with training; this is not a new
physical-part generalisation test or full native camera replay.

Full report, reproduction, hashes and failure crops:
[r33 labelled evaluation](r33-labelled-evaluation-20261114.md).
Evidence: `../IC_TERMINAL_TRAIN/acceptance-20261114/`, board matching
`/media/nvme/ic-terminal-acceptance-20261114/`; live-flow evidence under
`live-view-20261114-*`. Full suite now **153 passed**, two existing warnings.
Shared SDK, fixed station, vendor install, weights and calibration unchanged.

## First owner-confirmed good-part trial (2026-11-15)

Owner placed a good part with both IC and terminal present. Expected classes:
IC_Pass and Terminal_pass. No configuration, model, threshold or service change.

Initial 45-second live run: 1,266 admitted frames, 1,325 detection records;
both pass labels in every record, 19 records with very weak IC_Fail labels.
The limited movement did not establish full-rotation behaviour.

A later moving-part recording decoded 1,800 frames cleanly and saved ~60 seconds
of annotated video. Across 1,896 detection records, both pass labels occur in
1,283 (67.7%); 634 (33.4%) contain a fail label. These are descriptive class
presence counts, **not precision/recall**: images have no per-frame box labels,
records include unencoded tail frames, and the component leaves the image.

**Primary physical roadblock:** saved frames directly show the board clipped at
the bottom and top of its orbit. A blue-mask screen finds 44/183 sampled video
frames near a vertical border (heuristic, not an exact clipping rate). Old
zero-clipping measurements do not cover this placement. False IC_Fail boxes also
appear on plate reflections and other PCB regions. No threshold was adjusted.

Two exact source/preprocess/MLA/native-box captures preserve current-view
reference evidence. Python decoding agrees with native retained boxes; FP32
itself misses the weak terminal on one captured view, while compiled output adds
weak false fail boxes. This is not proof of a new layout defect or a reason to
retrain before fixing framing.

See [full good-part trial and next physical action](good-part-trial-20261115.md).
Viewing copy: `../IC_TERMINAL_TRAIN/good-part-20261115-rotation/capture/annotated-viewing.mp4`;
clipping stills: `framing-edge-0.png`, `framing-edge-180.png` in the same directory.
All players/pipelines stopped. Same good specimen should be moved inward with
the plate stopped, then retested without moving the camera, before bad states.

## Earlier continuous owner preview — stopped for approved capture (05:54 UTC session)

Owner confirmed the good PCB was repositioned and requested continuous viewing.
Started the existing host wrapper in the background with `--seconds 14400` and
pinned r33; no model/config/service change. **Leave this run active** while the
owner checks complete rotations. It will end on Q/player exit or its four-hour
limit (approximately 09:54 UTC), not automatically restart.

- Host controller PID at startup: **31520**; remote launcher/worker observed as
  **2961476 / 2961477**. PIDs are observations, not permission to signal blindly;
  recheck exact release/output identity before any remote stop.
- Local evidence: `../IC_TERMINAL_TRAIN/continuous-view-20261115-055448/`;
  adjacent `.log` and `.session.json` record detached wrapper startup.
- Board output:
  `/media/nvme/ic-terminal-live-continuous-view-20261115-055448`.
- Startup check: **state live, running true, media_ready true**, 3,543 admitted
  frames, all validly timed, no board errors. FFplay clock advanced beyond 116 s;
  it also reported frame drops, so no zero-drop/smoothness qualification.
- This is an annotated live preview, not a new recorded good-part scoring run.
  Repositioning has not yet been independently requalified for full visibility.
  Do not assume the previous cropped-run counts apply to this placement.
- An additional SSH command-channel inspection ended in EOF after listing the
  active jobs; it did not request a stop. A subsequent SFTP check verified exact
  launcher identity and **6,907 valid timed frames, live/media-ready, no errors**.
  Observe current health/player state before diagnosing recovery or starting
  another graph.

## Baseline case A captured — good repositioned PCB (2026-11-15)

Owner visually accepted framing and requested: **baseline → improve detection →
production streaming UI**. Added `benchmark_live_session.py`, a read-only
observer of the existing warm r33 stream. It does not stop/restart the preview
or open another decoder. Sampling/analysis unit runs were outside the window.

Two-minute window, exact encoder cohort **41,451–45,049**:

- **3,599 frames / 120.04 s = 29.98 fps** (delivery-span rate 30.00 fps).
  All 3,599 validly timed, zero encoder-ID gaps, 3,599 RTP marker increase,
  no board errors/non-ready samples or cache misses/evictions.
- Worker CPU **44.0% of one core** average; board-wide **3.91% across 16 cores**.
  Worker RSS **62,616–62,624 KiB**, end-minus-start 0. Not total native-service
  memory, accelerator utilization or a long-run leak qualification.
- Encoder delivery-gap p99 **151.6 ms**, max **162.4 ms**; FFplay drop counter
  increased **287**. Mean throughput is good; pacing/desktop losses remain
  unqualified. This does not measure RTP loss, rendered FPS or optical latency.
- For the same admitted cohort, both pass labels occur in **2,845 / 3,599 (79.0%)**;
  unwanted fail labels in **1,441 (40.0%)**; 92 empty results. These are raw
  class-presence counts, **not precision/recall or part rejection rates**.
  No independent per-frame/site boxes or host verdict rule yet.

Evidence: `../IC_TERMINAL_TRAIN/benchmark-20261115-good-inward-01/`, including
raw log byte windows, 61 health/proc/viewer samples, pinned inventory, full report
and identical offline re-computation. [Baseline protocol/results](benchmark-baseline-20261115.md).
Full suite **169 passed**, two existing warnings. Stream intentionally left
running; model/threshold/calibration/ROI/SDK/vendor services unchanged.

## Baseline case B captured — bad IC + good terminal (2026-11-15)

Owner confirmed "good terminal bad ic board present now." Measured the same
120-second read-only window without restarting the stream or changing settings.
Owner has since clarified that **all faults are missing components only**:
case B is IC missing + terminal present, not a solder/electrical defect.

- **3,603 frames / 120.05 s = 30.01 fps**, all validly timed; zero encoder-ID
  gaps/board errors/non-ready samples/cache misses or evictions. RTP marker
  delta 3,603, maximum sampled pending timestamps 61.
- Worker CPU **44.0% of one core**, board-wide **4.19% across 16 cores**.
  Worker RSS **63,276–63,284 KiB**, end-minus-start 0; 660 KiB higher starting
  RSS than A is not explained or sufficient evidence of a leak.
- Encoder delivery p99 **150.6 ms**, max **155.1 ms**; FFplay reported **334**
  additional drops. Throughput remains stable; smooth/zero-drop delivery does not.
- Expected IC_Fail present **93.6%**, Terminal_pass **85.7%**, both **80.4%**.
  Unexpected IC_Pass appears in **403/3,603 (11.2%)**, score up to **0.428**;
  Terminal_Fail in 4.2%. Any unexpected class: 15.0%; empty: 1.1%.
  This is not false-acceptance rate or box accuracy: expected/wrong classes can
  coexist and boxes are not independently labelled at the inspection site.

Exact admitted cohort **107,091–110,693**, same pipeline session as A. Evidence:
`../IC_TERMINAL_TRAIN/benchmark-20261115-bad-ic-good-terminal-01/`; full report,
61 samples, raw log windows and identical offline metric re-computation retained.
[Updated A/B comparison](benchmark-baseline-20261115.md#case-b-bad-ic-good-terminal-01).
No source code, model, threshold, calibration, ROI or service change; existing
169-test source baseline unchanged. Preview intentionally left running.

## Baseline case C captured — both components missing (2026-11-15)

Owner confirmed terminal missing + IC missing, and clarified that **no other
fault types** are included. Legacy IC_Fail/Terminal_Fail labels mean missing;
pass labels mean present, not joint/electrical quality certification.

Same release/session and 120-second read-only protocol:

- **3,599 frames / 120.04 s = 29.98 fps**, all validly timed, zero encoder-ID
  gaps/errors/non-ready samples/cache misses or evictions; marker delta 3,599.
- Worker CPU **44.6% of one core**, board-wide **4.00% across 16 cores**;
  RSS **63,388–63,396 KiB**, +8 KiB within window, not leak qualification.
- Encoder delivery p99/max **151.1/154.6 ms**; FFplay reported **276** drops.
- IC_Fail appears in **96.9%**, Terminal_Fail **84.7%**, both **83.9%**.
  Unexpected **IC_Pass: 1,051/3,599 (29.2%)**, max score **0.413**;
  **Terminal_pass: 0**; empty results 1.9%. Expected/wrong classes can coexist;
  these remain class-presence counts, not site-level accuracy/false acceptance.

Exact cohort **146,830–150,428**. Evidence:
`../IC_TERMINAL_TRAIN/benchmark-20261115-ic-missing-terminal-missing-01/`,
including 61 samples, raw windows, unchanged release inventory, exact offline
re-computation and `specimen-confirmation.json` preserving fault scope.
[Three-state comparison](benchmark-baseline-20261115.md#case-c-ic-missing-terminal-missing-01).
No code/model/threshold/ROI/service change; existing 169-test baseline unchanged.
Preview intentionally left running. Three available physical states captured;
optional good-A drift repeat and IC-present/terminal-missing are not measured.

Main quality concern for the next investigation: IC_Pass candidates survive on
missing-IC boards (11.2% in B, 29.2% in C). Locate them against exact source/site
annotations and FP32 before choosing threshold, ROI or training changes.
Current both-missing board can remain in place for that diagnostic. Do not infer
physical state from predictions or include swap/changeover frames in a benchmark.

## Matched false-pass diagnosis completed (2026-11-15)

Owner approved pause/capture/restore. Captured 40 periodic frames across four
stages (160 buffers) on the both-missing board, with stream/ID/caps/hash checks.
Compiler untiling matches companion RGB on all frames; raw/fused FP32 parity
passes. No model, calibration, threshold, ROI, pinned release or service change.

Five sampled unwanted IC_Pass boxes: four on the black pin header, one on a small
silver/grey PCB component, not the intended missing-IC footprint. All five match
independent compiled-output decoding. FP32 retains one; compiled probabilities
raise four weaker same-anchor IC_Pass candidates above 0.05. This supports
wrong-object confusion amplified by compiled/quantized-network score differences,
not a recurrence of tile corruption. Exact internal quantisation cause is unknown.

All sampled false-pass scores are 0.057–0.128; the baseline's stronger 0.413
case was not captured, so findings do not explain every error or select a fix.
Five IC_Fail retained-set differences also remain after Python NMS, while raw
candidate coordinates match; do not claim identical all-class suppression.

CPU packaging hit its 120-second bound after 39 archives. Resume recomputed
and verified existing arrays against raw captures, rebuilt the partial last file,
and completed; no additional neural/camera execution. Full suite **190 passed**.
[Detailed findings and evidence](false-ic-pass-diagnosis-20261115.md).

**Preview restored and left RUNNING**, started **08:27:29 UTC**, limit ~12:27 UTC.
Host PID 38940; remote launcher/worker 2962180/2962181 at observation. New session:
`../IC_TERMINAL_TRAIN/continuous-view-20261115-after-parity.session.json`;
board output `/media/nvme/ic-terminal-live-continuous-view-20261115-after-parity`.
Verified live/media-ready, 68,955 valid timed frames, no errors. Old 05:54 session
is stopped and cannot be used for new live benchmarks. Q stops current preview.

## Offline pose/site experiment completed (owner approved, 2026-11-15)

Tested 42 original source images: 40 both-missing and two adjacent both-present
views from before inward repositioning. No matched source images for the third
state; no independent pose/site labels or held-out session. No board connection,
preview interruption, model/settings/service/release change.

Blue-silhouette + dark-header orientation produces 36 candidate poses, six unknown.
Correction from enlarged annotation review: earlier four reversal claims were
wrong; 243/272/678/1084 are not reversed. Frame 678 fits only lower substrate.
Both present views fail. PCB-scoped
ORB/RANSAC produces 13 plausible candidates (including its reference), 29 unknown.
An initial unscoped ORB trial incorrectly aligned stationary background; preserved
as failed evidence, not counted as success. Neither method is safe to deploy.

A counterfactual class/site centre gate removes five sampled unwanted IC_Pass
boxes, but clean output is NOT proof of improvement: scoped ORB loses most
missing-component candidates to unknown alignment, and colour can fit only part
of the PCB. No precision/recall claim, no one-result-per-site verdict logic applied.
See [experiment report](site-pose-experiment-20261115.md) and
`../IC_TERMINAL_TRAIN/site-pose-experiment-20261115/selected-review-corrected.jpg`.
12 helper tests added; full suite **202 passed**, two existing warnings.
Live state was not rechecked during offline work; previous preview bound applies.

## Draft PCB/site labels prepared (owner approved, 2026-11-15)

Prepared 42 image drafts (126 polygons), **zero independently approved**. Open
`../IC_TERMINAL_TRAIN/site-label-draft-20261115/review-pack-final/review.html`.
Includes exact source copies, PCB corners/direction, IC/terminal footprint
polygons, editable browser review, LabelMe alternatives, seven comparison pages
and 10 high-priority images. All existing images are development-only. The pack
includes 40 both-missing and two adjacent both-present; third state is missing.

Geometry is explicitly provisional: machine seeds for most, seven manual board
proposals for failures/partial fit, three retained after reinspection. Site
polygons are template-projected. Human can correct/approve/flag then export JSON;
no training export. [Review instructions](../guides/pcb-site-label-review.md).

Annotation preparation exposed the earlier mistaken four-reversal report; current
report corrected, old changelog entries untouched and correction appended. Only
`review-pack-final` is current; earlier assembly attempts are superseded.
No board connection, preview interruption, model/runtime/release change. Full
suite **218 passed**, isolated browser smoke 13 checks pass; actual labels remain
0/42 approved. Live health was not rechecked; previous preview duration applies.

## Owner-reviewed labels received; Canny crop experiment completed

Repo-root `annotations.review-export.json` validates against exact source images
and the original draft: **42 human-reviewed**, 16 PCB quads and all 42 site records
changed; IDs, hashes, classes/state, splits, convexity/bounds match. Export is
preserved unchanged and copied into experiment snapshots. Original draft remains
historical/pending; current human review is a separate annotation version.

Tested three Canny pairs (30/90, 50/150, 80/200) and 0/1/2 closing iterations.
Best exploratory row, 30/90 + one closing, preserves reviewed PCB and both sites
in **28/42** images. **14 wrong crops**, all with zero reviewed-target overlap,
including both present views; example failures select a fixed mounting bracket.
Successful crops retain median 2.73% of frame area (~97% excluded). This is NOT
held-out detector precision/recall and does not make the crop selector safe.
Canny rectangle orientation still has 180-degree ambiguity. No deployment.

[Full report](canny-crop-experiment-20261115.md); visual examples:
`../IC_TERMINAL_TRAIN/canny-crop-20261115-close1/selected-review.jpg`.
17 new helper tests; full suite **235 passed**, two existing warnings. No board
connection/stream interruption, model inference, runtime/config/release change.
Live health not rechecked; previous preview bound remains applicable.

## Hybrid PCB-colour/Canny crop experiment completed (owner approved)

Same reviewed images, no annotation-based positive selection. Colour-gated Canny
rejects all bracket mistakes but leaves 14 unknown. Grouped colour seeds alone
preserve 40/42, clipping terminal geometry on both present views. Colour seeds +
local Canny extent preserve reviewed PCB/IC/terminal in **42/42** (strict vertex
containment also passes), no wrong crop/unknown in this DEVELOPMENT sample.
Median crop area **2.95%** (~97% source area excluded). This is not held-out
performance or detector precision/recall. All 42 synthetic board-removed controls
return unknown; no real empty-plate false-positive qualification.

No manual plate ROI; local edge extent constrained around colour seed, not GT.
180-degree semantic orientation and glare remain unresolved; header still appears
in crop, so its false IC_Pass may persist. Only two adjacent good views and no
third-state source data. Cropped-input neural accuracy/runtime remain unmeasured.
[Report](hybrid-crop-experiment-20261115.md); compare images:
`../IC_TERMINAL_TRAIN/hybrid-crop-20261115/selected-review.jpg`.

14 new helper tests; full suite **249 passed**, two existing warnings; source,
review snapshots, coverage and summaries independently checked. No board access,
stream interruption, model/training/calibration/runtime/config/release change.
Live health not rechecked; previous preview bound remains applicable.

## Original-versus-crop FP32 detection images ready for owner review

Owner requested the detection images and will decide further improvements after
review. Folder `../IC_TERMINAL_TRAIN/detection-review-20261115/`; open `index.html`.
42 comparisons, 42 full originals, 42 original ROI zooms, 42 cropped detection
images and seven sheets. All images including empty predictions retained.

84 CPU FP32 inferences, same pinned reference, confidence .05/NMS .45/top100.
No site filter, enhancement, box hiding or verdict. Native crop pixels used, not
512px viewing preview; exact crop regeneration and image/review/model hashes
checked. **NOT native/compiled board detections or quantized-input replay.**

Original inputs: 200 boxes (151 IC_Fail, 2 Terminal_pass, 5 IC_Pass, 42
Terminal_Fail). Cropped: 34 boxes, **all Terminal_Fail**, no IC detections at fixed
settings; 21/42 empty, including both present views. Cleaner imagery has NOT
established better inspection. Scale/orientation/context changes are not isolated.
[Report](crop-detection-review-20261115.md). Full raw outputs and coordinates kept.

Nine new helper tests; full suite **258 passed**, two existing warnings; raw-output
re-decoding/image counts/gallery links verified. No board access, stream/model/
calibration/training/runtime/release change. Live health not rechecked.

## Cropped training-box draft ready for manual review (owner approved)

Owner observed recall loss, discussed retraining on cropped inputs and requested
annotations for manual review before training. Prepared **42 crops / 84 boxes**,
all newly pending: source-site polygons projected then converted to axis-aligned
training-box proposals. Not predictions; source-site approval not carried over.

Open `../IC_TERMINAL_TRAIN/cropped-training-label-review-20261115/review.html`.
Resize/move rectangles, review/flag, then EXPORT (not autosaved) as
**`cropped-training.review-export.json`** to repo root. Do not overwrite older
`annotations.review-export.json`. [Instructions](../guides/cropped-training-box-review.md).
Seven static sheets and per-image raw/proposal comparisons also available.

Coverage is **40 IC_Fail / 2 Terminal_pass / 2 IC_Pass / 40 Terminal_Fail**.
Only 40 both-missing and two adjacent both-present; more present/third-state
rotations and independent sessions needed before meaningful four-class training
and validation. No data.yaml or final split; `draft-yolo-DO-NOT-TRAIN` labels are
provisional, NOT updated by browser edits. Final labels must be regenerated from
review export. Every row remains training-ineligible; no training started.

11 new helper tests; full suite **269 passed**, two existing warnings, browser
smoke 14 checks. Exact crops/source snapshots and label coordinate round trips
verified. No board access, stream/model/calibration/runtime/release change.
Live health not rechecked; previous preview duration applies.

## First batch accepted; class-balancing batch02 prepared

Owner says first-batch boxes were all accurate. Recorded verbal approval of its
unchanged draft SHA in batch02 `first-batch-owner-confirmation.json`; no browser
export or edits invented. If geometry was edited, obtain actual export before
finalizing. Original first draft untouched; training still not started.

New review: `../IC_TERMINAL_TRAIN/cropped-training-label-review-batch02-20261115/review.html`.
**48 images / 96 pending boxes**: 32 both-present, 16 IC-missing/terminal-present,
selected from HISTORICAL TRAIN ONLY data, not fresh current-camera captures.
Export as **`cropped-training-batch02.review-export.json`**. Same editor; no autosave.

84 historical train candidates passed frozen crop: 76 accepted, eight unknown
quarantined, not forced. Quotas selected by farthest orbit angle, both states
span all 12 thirty-degree bins. Historical validation/test images not consumed;
source/model hashes have zero exact holdout overlap. Original train/test share
recordings, so this is NOT independent-part validation. Original labels projected,
not detector predictions; all new boxes require review (some become loose).

Potential combined class counts after approval: **56/50/34/40** for 90 images/
180 boxes. Better balance, still correlated and missing independent current-camera
holdout coverage. [Review and augmentation plan](../guides/training-batch02-and-augmentation.md).
Split originals by capture/specimen BEFORE augmentation; random, label-aware
augmentation TRAIN ONLY, same deterministic crop/resize/normalization in serving.
No CLAHE/sharpening or numeric augmentation/training config added now.

10 new helper tests; full suite **279 passed**, two existing warnings. Browser
smoke 14 checks, all new labels pending. Exact hashes/labels/train-only membership
verified; frozen source helpers retained. No board/model/training/runtime/release
change. Live state unverified during offline work; earlier finite bound may expire.

## Reviewed training-only preprocessing/augmentation complete

Owner returned `cropped-training-batch02.review-export.json`; all 48 rows reviewed,
identity/state/classes/bounds valid. Corrected IC at hist-absent_006413 used from
export. First 42 use exact-hash-bound unchanged-draft verbal approval. Source
files/old datasets untouched. All **90 parents assigned TRAIN before augmentation**.

Prepared `../IC_TERMINAL_TRAIN/augmented-training-pool-20261115/`; open index.html.
**450 lossless 1280-square PNGs, 450 YOLO files, 900 boxes**, five views/parent:
base, 180°, exposure, mild rotation/scale, mild model-space blur. Padding114,
aspect preserved; uint8 PNGs still need loader RGB/255 once. No CLAHE/sharpening/
hue/mosaic. Geometry/class checked; no target clipping/drop. Additional online
augmentation should be disabled for this baked pool; options provided, no run.

Base class counts 56/50/34/40; augmented 280/250/170/200. This is **90 correlated
originals, NOT 450 independent examples**. All source/capture/variant transforms,
parameters/seeds/hashes retained; all 450 pixels/labels regenerated exactly and
900 YOLO round trips verified. [Prepared-pool guide](../guides/prepared-training-pool-20261115.md).

**Training not started; training_ready=false.** No validation/test folders or
train-as-val alias/YAML invented. Need separate validation/evaluation decision
before candidate training. Current live graph remains full-frame, not this crop
pipeline. 14 helper tests added; full suite **293 passed**, two existing warnings.
No board/stream/model/calibration/compilation/runtime/release/SDK change; live
health not rechecked.

## Next actions (require agreement before further changes)

1. Review augmentation previews and choose separate validation plan: preferably
   new current-camera full rotations across all three states, or explicitly
   limited reserved historical validation prepared/reviewed without joining train.
   Then finalize candidate data/config/run and train separately; old v4 trainer
   and weights remain untouched. Never split augmented siblings or use train as
   validation. Deployment still needs real negatives and runtime/unknown handling;
   prepared augmentation is not model accuracy evidence.
   Stronger unwanted IC_Pass (baseline max 0.413) also remains uncaptured.
   Keep preview untouched unless an agreed capture requires the exclusive decoder.
2. Choose confidence on suitable validation data after the physical/annotation
   checks; do not tune on the historical test set or ship diagnostic 0.05.
3. Address player startup drops, EOS/drain, disconnect/recovery, latency and a
   longer soak separately; regenerate/promote stale candidate configs explicitly.

The capture feed remains stopped at this checkpoint; restarting it will occupy
hardware needed by any further inference test.

The history of how this state was reached: [experiments](experiments.md).
