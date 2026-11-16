# Matched-frame diagnosis of unwanted IC_Pass — 2026-11-15

## Scope and outcome

Owner approved briefly pausing the preview to inspect the **both-missing** PCB.
No model, calibration, threshold, ROI, pinned release or installed service changed.
This is a diagnostic sample, not an accuracy benchmark or a completed quality fix.

**Finding:** in the five sampled unwanted IC_Pass predictions, four boxes land on
the black pin header and one on a small silver/grey PCB component—not the intended
missing-IC footprint. These visual identifications come from reviewing exact source
crops; independently drawn inspection-site boxes are not yet available.

The native box decoder and independent compiled-output decoder agree on all five.
FP32 also produces one of them above 0.05. For the other four, the compiled network
raises weak IC_Pass probabilities above the diagnostic cutoff. This supports
wrong-object confusion plus compiled/quantized-network confidence differences,
not recurrence of the input/output tile-layout fault.

## Controlled capture and verification

- Stopped only the identity-checked launcher belonging to the owner's preview.
  Its wrapper completed shutdown and stopped its own player normally.
- Captured **40 periodic frame IDs: 40 + 29*i, i=0..39**, without score-based
  selection, using unchanged r33 diagnostic graph. For every ID, retained decoder
  pixels, preprocessor bytes, raw MLA output and native detections: **160 buffers**.
- Recorded frame/stream IDs, caps and SHA-256 for every buffer. Verified pinned
  release assets/tools and identical cross-stage frame membership.
- Compiler untiling reproduces the companion quantized RGB exactly on all 40.
- Decoded captured MLA outputs using installed SDK software transforms, without
  rerunning neural inference. Compared original FP32 and raw-cut FP32 on the same
  captured quantized RGB dequantized to float. This is **not** the original
  pre-quantization float image.
- Raw/fused FP32 decoding agrees within **0.000244141 model pixels** and
  **1.45e-7 probability**. Artifact/compilation lineage and class metadata pass.
- Preview restarted immediately after capture, before CPU packaging/reference work.

## Sampled unwanted pass candidates

| Frame | Visual region | FP32 IC_Pass at same anchor | Compiled IC_Pass |
|---|---|---:|---:|
| 359 | Black pin header | 0.0143 | **0.0715** |
| 388 | Black pin header | 0.0407 | **0.0715** |
| 794 | Black pin header | 0.0309 | **0.0575** |
| 823 | Black pin header | **0.0795** | **0.1275** |
| 939 | Small silver/grey PCB component | 0.0370 | **0.0575** |

All are stride-8 anchors. FP32 already ranks IC_Pass highest among the four
classes at those anchors, but four scores are below 0.05 and are suppressed.
The one FP32 retained box matches the native one at IoU **0.973**. Thus this is
not merely drawing a correctly detected missing IC with the wrong overlay text.

These confidence shifts are not proof of an incorrect calibration scale or a
specific faulty layer. Their internal cause is not isolated. No recalibration,
weight change, threshold selection or retraining was attempted.

### Decoder/postprocessing limits

Across all classes, **198/203** native boxes at score >=0.05 match Python's retained
compiled detections at IoU >=0.8, max coordinate error **0.500 px**, max score
error **5.96e-8**. The other five are IC_Fail retained-set differences: matching
raw compiled candidates exist within **0.484 px**, but Python NMS does not retain
those same candidates. Exact native/Python suppression behaviour is not isolated.
Do not claim perfect all-class NMS parity. All five IC_Pass boxes do match.
Native marginal scores around 0.04871 are retained in evidence but excluded from
this strict-0.05 parity comparison, not silently removed from production output.

## Important limits and proposed next step

Only **5/40 sampled frames** contain native IC_Pass; FP32 contains it in 1/40.
These are not estimates of the earlier 29.2% baseline frequency. Capture copying
changes timing, and the periodic sample covers different moments/poses.

The sample's highest unwanted score is **0.128**. The baseline reached **0.413**;
those stronger cases were **not captured here**. Do not claim this explains every
false pass or that raising the threshold solves the problem.

Recommended next action: a bounded capture targeting stronger unwanted IC_Pass
candidates, then review site labels and hard-negative header/passive examples on
current-camera data. Use an independently labelled validation set to distinguish
FP32 model confusion from compiled confidence changes before selecting a data,
calibration, training or site-filtering fix. No such changes are approved/applied
by this investigation. Leave the both-missing specimen available.

## Operational handover and roadblock

CPU packaging timed out after 39/40 archives at 120 seconds; original raw buffers
were complete. Added explicit resume that recomputes arrays from those raw buffers
and requires exact equality before reusing each complete archive. All 39 passed;
the incomplete final archive was rebuilt atomically. No new camera/MLA run or
repeated native crash was involved. Resume completed with a larger bounded timeout.

**Preview left RUNNING:** started **08:27:29 UTC**, four-hour bound to about
**12:27 UTC**. Host wrapper PID **38940**, remote launcher/worker
**2962180 / 2962181** at observation. Recheck exact identity before signalling.
Session file: `../IC_TERMINAL_TRAIN/continuous-view-20261115-after-parity.session.json`.
Board output: `/media/nvme/ic-terminal-live-continuous-view-20261115-after-parity`.
Verified live/media-ready with **68,955 valid timed frames**, no errors. Q stops
viewer and its owning board job. Earlier 05:54 preview/session is stopped; do not
use it for new live benchmarks. Capture feed remains stopped.

## Evidence and source

Local root: `../IC_TERMINAL_TRAIN/live-false-pass-20261115/`.
Board root: `/media/nvme/ic-terminal-false-pass-20261115/`.

- `decoded/`: 40 matched input/head archives, exact source NV12 frames and hashes.
- `comparison/comparison.json`: full FP32/compiled/native results and anchor scores.
- **`comparison/false-pass-contact-sheet.jpg`**: labelled five-case visual comparison.
- `comparison/false-pass-<frame>-0.jpg`: individual enlarged source/FP32/native crops.
- `comparison/visual-findings.json`, `raw-candidate-parity-review.json`: region
  review and the five retained-set mismatch checks.
- Capture, timeout/resume, previous-preview outcome and restored-health logs retained.

Tools: [periodic buffer capture](../../tools/analysis/inspect_native_buffers.py),
[decode and comparison](../../tools/analysis/live_false_pass_parity.py).
Capture options used: `--start-frame 40 --frame-step 29 --samples 40 --seconds 60`.
Decode uses `--release`, `--capture`, new `--output`; explicit `--resume` verifies
existing arrays rather than trusting old files. Compare uses `--decoded`,
`--original`, `--raw`, `--compilation`, new `--output`.

Full suite **190 passed**, two existing dependency warnings. Added 11 sampling
and 10 diagnostic helper/resume tests; no native hardware is mocked into a claim
of accuracy. Shared SDK, fixed station and installed vendor implementation untouched.
