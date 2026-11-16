# Owner-selected production pipeline — 2026-11-16

## Current operational default

**Release `portable-vision-production-20261116-r2`** is selected in
`config/production-pipeline.json`. Start from repo:

```powershell
.\run-production.ps1
# bounded unattended test:
.\run-production.ps1 -Seconds 40 -Headless
```

Default300s/max300s sessions are deliberately retained. This is NOT an unattended
boot service or24/7 supervisor. Closing browser alone does not immediately stop
board; the bound applies. Frontend lifecycle/stop/status/retention work is next.
No auto-restart of arbitrary processes, no displacement of another camera graph.

Production selection means owner-approved operational baseline, NOT an independently
qualified accuracy/optical-latency/soak claim. Missing components only; existing
model confidence/quantization and tracking limits remain documented.

## Frozen package and dependencies

Folder/ZIP: `../IC_TERMINAL_TRAIN/releases/portable-vision-production-20261116-r2`.
ZIP119,184,765bytes,42 manifested files. SHA-256:
`21b3dcc61a4992a7b842cf9f70cb394be647060ace74c1417bdec75422aae211`.
Manifest SHA-256:
`82386a4629856e7219651e531ccfce37ac070056b1e0c5153eb137c3f8fe48ba`.

Includes accepted model ELF/MPK/reference ONNX, actual owner-view runtime/helpers,
C++8 compiled library AND source, feature ROI, exact-frame browser renderer,
proof/audits,9 pinned offline Windows/Python3.11 wheels and setup/run/verifier.
Credentials EXTERNAL, never archived. Native library SHA:
`da2f8d81121aec34955fb6dd23df1bb9180e7151f5badc4a59cc1594b32f5182`.
Existing board OS libgomp.so.1/SiMa SDK remain prerequisites; no vendor installs.

Dedicated isolated environment:
`../IC_TERMINAL_TRAIN/production-envs/portable-vision-production-20261116-r2/`.
Created using packaged setup.ps1 and installed OFFLINE from shipped wheels;
pip check passes. No modification of experiment/training/fixed-station environments.
If moved, run setup.ps1 with compatible CPython3.11 x64, then run.ps1 with external
-Credentials/-Python overrides. Both launchers verify integrity first.

## What is promoted

- Independent original camera H264 video (~30 decoded FPS).
- Latest-only actual model inspection (~15–16/s observed).
- Coordinate-only asynchronous publication; host matches exact decoded NV12 hashes.
- Browser draws only on the corresponding inspected frame, not stale live video.
- Bounded LK/RANSAC pose reuse and unchanged confidence/refresh guards.
- Same80-corner feature detector/mask, PCB-derived8px-halo compute ROI.
- C++ parallel8 exact input packing, SHA/ABI/byte-parity checked before camera start.
- Existing model, calibration, confidence/NMS and four-class contract unchanged.

No favourable-region gate, tracking-verdict inference or dynamic confidence tuning.

## Promotion and actual verification

`tools/package_coordinate_fallback.py --production` now freezes native artifacts,
feature/native flags and versioned dedicated host environment. Production package
runtime/helper/library bytes match the accepted viewer and completed feature-ROI
proof; host launch argument validation is the only entrypoint packaging adaptation.
`run-production.ps1` verifies selected manifest/role/mode/receiver before invoking
frozen package, not mutable experimental runtime source.

`tools/promote_production_pipeline.py` refuses activation unless packaged smoke:
completed/no worker or receiver errors, correct model/native hashes and all flags,
full metadata matching, per-file package integrity and deployed-source hashes.
Writes selection atomically and keeps prior selection snapshots in
`../IC_TERMINAL_TRAIN/production-staging/history/`.

r1 was staging validation; retained, not edited. r2 corrects inherited README
wording/default duration and has its OWN dedicated environment/packaged smoke.
No inference code difference between staged r1 and selected r2.

- Packaged r1 smoke40s:639 model calls,15.56/s,30.04 host decoded FPS,
 639/639 matches,0 RTP gaps/errors.
- Selected r2 smoke30s:468 model calls,15.09/s including startup/shutdown,
 30.06 host decoded FPS,468/468 matches,0 RTP gaps/errors,0 unknown crops.
  Evidence `../IC_TERMINAL_TRAIN/production-runs/coordinate-pose-20261116-120302-226/`.
- Prior60s performance proof:974 actual model calls/61.05s=15.95/s.
-42-file verification, archive CRC, offline dependency restore and pip check pass.
  Eleven production promotion/launcher guard tests added;364 total tests pass,
  two existing dependency warnings. Earlier40-file fallback reverified unchanged.

These are bounded live tests, not independent-part accuracy, browser-render FPS,
zero jitter, physical latency or long soak. All smoke jobs stopped cleanly; current
selection being active does not imply a background camera job is running.
12:16 UTC read-only process inspection found no active camera/worker/feed graphs.

## Deployment identity / rollback

Each run uploads this frozen package into a NEW verified board job and records
source/model/library hashes in evidence. Host selection is authoritative. No board
current-symlink/service switch was made; old pinned r33 remains untouched. Do NOT
use old full-frame r33 graph with these crop-trained weights.

Earlier packages stay immutable and directly runnable for manual rollback:
- `fallbacks/coordinate-browser-pose-20261116-v1` (pre-C++/feature-ROI).
- `fallbacks/latest-frame-optimized-20261116-v2` (snapshot diagnostic).
- staged production r1 also remains with its prior selector history.

Do not just point production config at an older fallback: production launch guards
require this native/feature profile. Use older package's own run.ps1 explicitly,
with exclusive camera ownership; promotion of another operational release needs
its own matching proof/config. No automatic silent rollback.

## Frontend handoff

Build frontend against this SELECTED immutable backend, not tools experiment copies.
Browser HTTP currently binds127.0.0.1/ephemeral port, recorded in each session.json:
`/live.mjpg`, `/inspection.json`, `/frames/<sha>.jpg`, `/stats.json`. Exact-match,
UNKNOWN/stale/disconnected and separate-live-vs-inspected semantics must survive.
API auth/network exposure, stable port/process supervision, operator stop and
continuous-operation/retention policy are explicitly NOT implemented in this step.
