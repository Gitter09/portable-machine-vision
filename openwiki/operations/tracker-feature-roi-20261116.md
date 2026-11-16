# PCB feature-tracker ROI optimization — 2026-11-16

Owner requested tracker optimization toward15–16 inspections/s. New opt-in
`--feature-roi` restricts ONLY feature detection computation to the already found
PCB's bounding rectangle plus8px filter halo. This is NOT favourable-region
inspection: localization still examines full camera frame and all plate positions
remain eligible. No fewer corners or relaxed thresholds/refresh intervals.

## Algorithm and equivalence

`pcb_pose_tracker.seed_features` uses the SAME full-frame polygon mask,80 corners,
qualityLevel.01/minDistance3/blockSize3. Optional grayscale ROI/mask views reduce
processed area; local coordinates are translated back to camera coordinates.
8px context halo preserves detector/filter/NMS neighbourhoods. Border-clipped
halo retains camera-edge behaviour. Featureless remains unavailable, never a
fabricated pose. Existing LK forward/backward/RANSAC/appearance/motion guards and
0.5s full-refresh threshold/>0.25s stale-step rules unchanged. The refresh check
runs on the next sampled frame, not a hard real-time0.5s deadline.

Stage instrumentation now separately records full localization and feature seed
cost/count/area in pose metadata. Default remains full-frame feature computation.
Host rejects --feature-roi without --track-pose before board access/output creation.

Camera-free BOARD benchmark31 audited source NV12 frames (hash checked): ordered
feature point arrays EXACT for full-frame vs ROI, not merely similar counts.
Interleaved warmed timing:81.05ms→2.46ms mean (81.09→2.21ms median). No camera/MLA
used. ROI covers~3% of image on live samples; full mask still created for exactness.

## Live comparison

All runs use identical C++8 input library SHA
`da2f8d81121aec34955fb6dd23df1bb9180e7151f5badc4a59cc1594b32f5182`.
Same model, crop policy, confidence/NMS, latest-only queues, browser matching.
Sequential rotating scenes, NOT identical-frame controlled/independent accuracy trials.

| Trial | Processed/s | Located actual model calls/s | Unknown |
|---|---:|---:|---:|
| Baseline40s, feature full frame |13.22|13.22|0/544|
| ROI40s |15.65|15.31|14/642|
| ROI confirmation60s |**15.95**|**15.95**|**0/974**|

Final974 model inspections in61.05s including startup/shutdown;864 tracked updates/
110 full attempts, one tracking-confidence fallback, no worker errors. Every974
metadata result published/received and source-hash matched at arrival.0 final RTP
sequence gaps, host decoded30.03FPS; p99 arrival gap63ms/max610ms includes startup/
source burstiness, not zero-jitter or physical latency acceptance. Board forwarding
28.78FPS includes startup/shutdown. Browser-render FPS not measured.

Live full-refresh feature seeding93.03ms baseline→4.31ms final. Full localization
itself stays~42–43ms; parameters unchanged. First ROI trial had10 metadata messages
not matched AT ARRIVAL despite no RTP gaps (can precede host decode); unmatched
coordinates remain fail-closed until a matching image exists. Do not claim those
trials all displayed every inference. Final974/974 immediate matching succeeded.

## Validation and limits

- Final39 host raw-frame audits: exact board-input hash; full-vs-ROI ordered
  feature coordinates equal on all39. Fresh-selector seed coverage by estimated
  tracking crop >=.999999956. This is geometry diagnostic, NOT reviewed component
  site coverage/detection precision/recall or stress qualification.
- Seven new tests: exact point/order equality under rotated masks, camera-edge
  halo, featureless input, disappearance cache clear and CLI guard. Existing
  rotation/translation/jump/stale/reacquisition tests still pass.353 total tests,
  two existing deprecation warnings.
- Model, MLA/input packing, calibration, label order, confidence, crop size/
  orientation policy and tracking reliability gates unchanged. Faster sampling
  changes which live frames are inspected, so no invented live quality equivalence.
- Latest40-file and prior24-file fallback verifiers pass unchanged. No package
  published; existing builder refuses ROI proof until launch flags explicitly
  frozen (also still refuses unsupported native-library packaging).
- Bounded jobs stopped;11:39 UTC read-only process check found no graphs. No r33,
  installed vendor/SDK, clock, service or fixed-station changes.

## Evidence / invocation

`../IC_TERMINAL_TRAIN/feature-seed-roi-20261116/benchmark.json`,
`feature-roi-live-20261116-{baseline,optimized,final}/` contain source snapshots,
worker/receiver/pose/frame/published/preflight JSON and raw host audits.
`feature-roi-summary-20261116.json` aggregates measurements and final audit.

```
../IC_TERMINAL_TRAIN/browser-view-env/Scripts/python.exe tools/analysis/run_coordinate_browser.py \
  --output ../IC_TERMINAL_TRAIN/<NEW-unique-trial> --seconds 60 --headless \
  --track-pose --feature-roi \
  --cpp-library ../IC_TERMINAL_TRAIN/cpp-input-fusion-20261116-parallel/libportable_input.so \
  --cpp-kernel parallel8
```

Omit --headless for browser. Receiver172.21.30.230:5010/5012,20..300s bound,
no concurrent camera graph. No favourable-region gating, model tuning or new
output fusion introduced. Candidate achieved target in a short run, not production
acceptance or a sustained throughput guarantee across parts/lighting/occlusions.
