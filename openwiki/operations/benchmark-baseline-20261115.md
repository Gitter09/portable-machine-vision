# Portable pipeline baseline — 2026-11-15

## Purpose and frozen conditions

Owner requested a baseline before detection-quality work, followed later by an
actual streaming UI. The owner has now explicitly confirmed **missing-component
faults only**: IC_Fail = missing IC, Terminal_Fail = missing terminal; the pass
classes mean present, not electrical/joint-quality certification. The first case is **one known-good PCB with IC and terminal
present**, repositioned inward. Owner visually accepted the framing; this is not
a new quantitative source-image clearance measurement.

- **Release:** pinned r33, native camera → model → boxes → overlay → H.264/RTP.
- **Input/output:** 1280×720; model input 1280×1280.
- **Confidence:** unchanged diagnostic **0.05**, native outputs retained as-is,
  including marginal scores near 0.04871. No post-hoc filtering/tuning.
- **Viewer:** existing FFplay on workstation 172.21.30.13:5010, left running.
- **Observation:** read-only SFTP sampling every two seconds over approximately
  two minutes of the already-warm stream. No restart, extra decoder or model run.
- **Specimen:** owner-confirmed good state; no independent per-frame/site boxes,
  and no automatic host good/bad verdict was evaluated.

## Case A: good-inward-01

### Pipeline and timing

| Measurement | Observed |
|---|---:|
| Health-counter window | **120.04 s** |
| Encoder frames in window | **3,599** |
| Encoder throughput, health window | **29.98 fps** |
| Encoder throughput, exact delivery span | **30.00 fps** |
| Validly timed frames | **3,599 / 3,599** |
| Encoder frame-ID gaps | **0** |
| RTP marker-counter increase | **3,599** |
| Observed board errors / non-ready samples | **0 / 0** |
| Timestamp-cache misses / evictions | **0 / 0** |
| Maximum sampled pending timestamps | **60** (capacity 512) |
| Encoder delivery interval p50 / p95 | **39.5 / 53.4 ms** |
| Encoder delivery interval p99 / maximum | **151.6 / 162.4 ms** |
| FFplay reported drop-counter increase | **287** |
| FFplay video-clock advance | **120.06 s** |

Average throughput meets the earlier ≥25 fps target in this window. This does
**not** establish uniformly paced 30 fps presentation: delivery is bursty and
FFplay reports drops. The median interval can exceed 33.3 ms while the mean rate
is 30 fps because other intervals are shorter/bunched.

The 287 is FFplay's observed drop-counter delta, **not measured RTP packet loss**
or an exact rendered-FPS count. No claim that the desktop displayed every frame.
Likewise, interarrival intervals/valid PTS are **not optical end-to-end latency**.

### Board resource use

| Measurement | Observed |
|---|---:|
| Worker CPU, mean / max, 100% = one logical core | **44.0% / 45.7%** |
| Whole-board CPU, mean / max, normalized across 16 logical CPUs | **3.91% / 5.10%** |
| Worker RSS, min / max | **62,616 / 62,624 KiB** (~61.15 MiB) |
| Worker RSS, end minus start | **0 KiB** |

Worker RSS/CPU does not include all separate native service allocations. The
whole-board CPU includes other processes and the SSH sampler. Low CPU usage does
not establish accelerator headroom. MLA utilization, power and hardware memory
were not measured. A two-minute flat RSS trace is not an all-day leak test.

### Native prediction consistency on the known-good board

Results below use the **same 3,599 encoder-admitted frame IDs**, joined to native
detection records by session + stream + frame ID. Encoder and detection cohort
counts match exactly. This excludes unencoded tail frames from the encoder
cohort, but does not pair frames to client pixels or provide box ground truth.

| Observation | Encoder frames | Fraction |
|---|---:|---:|
| IC_Pass present | 3,326 | **92.4%** |
| Terminal_pass present | 2,988 | **83.0%** |
| Both expected labels present | 2,845 | **79.0%** |
| At least one unwanted fail label | 1,441 | **40.0%** |
| No predictions | 92 | **2.6%** |

Fail labels and pass labels can coexist. IC_Fail appears in 1,423 frames,
maximum score 0.319; Terminal_Fail appears in 37, maximum score 0.198. Most
IC_Fail candidates are weak (median score 0.064), but not all are confined to
0.05. Terminal_pass has multiple candidates in 1,490 frames; IC_Pass in 284.
These counts expose clutter and inconsistency, not distinct physical parts.

**Do not call 79% accuracy or 40% part rejection rate.** Presence somewhere in
the image does not prove correct localisation, and extra fail boxes may lie on
background or other PCB regions. The part is owner-confirmed good, but the
inspection-site/box annotations and host decision rule are not established.

## Case B: bad-ic-good-terminal-01

Owner then confirmed **“good terminal bad ic board present now.”** At capture
the physical fault type was unspecified; the owner has since clarified that
**all faults are missing only**, so case B is **IC missing + terminal present**. Same repositioned
setup, running session, release, confidence and read-only 120-second protocol;
no stream restart or settings change.

| Measurement | Case A: good IC | Case B: bad IC |
|---|---:|---:|
| Window / encoded frames | 120.04 s / 3,599 | **120.05 s / 3,603** |
| Encoder throughput | 29.98 fps | **30.01 fps** |
| Encoder frame-ID gaps / board errors | 0 / 0 | **0 / 0** |
| Worker CPU, mean of one core | 44.0% | **44.0%** |
| Whole-board CPU, mean across 16 cores | 3.91% | **4.19%** |
| Worker RSS min–max, KiB | 62,616–62,624 | **63,276–63,284** |
| RSS end-minus-start within each window | 0 | **0** |
| Encoder delivery p99 / maximum | 151.6 / 162.4 ms | **150.6 / 155.1 ms** |
| FFplay reported drop delta | 287 | **334** |

Case B has 3,603 validly timed frames and matching RTP marker increase, no cache
misses/evictions or non-ready samples. Maximum sampled pending cache entries: 61.
Worker RSS starts 660 KiB higher than case A, but is flat within B; the cause of
that between-window change is not isolated and this is not a leak qualification.

Prediction counts for the exact **3,603-frame** admitted cohort:

| Observation on bad IC + good terminal | Frames | Fraction |
|---|---:|---:|
| Expected IC_Fail present | 3,371 | **93.6%** |
| Expected Terminal_pass present | 3,088 | **85.7%** |
| Both expected labels present | 2,896 | **80.4%** |
| Unexpected IC_Pass present | **403** | **11.2%** |
| Unexpected Terminal_Fail present | 150 | **4.2%** |
| Either unexpected label | 541 | **15.0%** |
| No predictions | 40 | **1.1%** |

Expected/unexpected labels may coexist. IC_Pass score reaches **0.428** on this
owner-confirmed bad board, so unwanted pass candidates are not all at the
minimum diagnostic score. Without site/box ground truth or a host verdict rule,
**11.2% is not a measured false-acceptance rate**. It is a signal to investigate
where those pass boxes lie and what physical defect is represented.

There are 10,951 IC_Fail candidates, with multiple candidates in 2,704 frames;
these are not 10,951 independent defects. IC_Fail max score is 0.907. No scores
were post-filtered, no threshold chosen, no labels changed and no model tuned.
The 80.4% expected-pair presence is not 80.4% inspection accuracy or a fair
ranking against case A's different physical state.

B evidence: `../IC_TERMINAL_TRAIN/benchmark-20261115-bad-ic-good-terminal-01/`,
with the same report/raw-window/sample files and identical offline re-computation.
Exact cohort IDs **107,091–110,693**, same source session as A. Existing 169-test
source baseline is unchanged; no new unit run was needed for this measurement.

## Case C: ic-missing-terminal-missing-01

Owner confirmed **both components missing** and clarified that no other defect
types are in scope. Same native session, r33 inventory, confidence and read-only
120-second protocol as A/B; no graph restart or detector changes.

| Measurement | Case C |
|---|---:|
| Window / encoder frames | **120.04 s / 3,599** |
| Encoder throughput | **29.98 fps** (delivery-span 30.00) |
| Valid timed frames / RTP marker increase | **3,599 / 3,599** |
| Encoder-ID gaps / board errors / non-ready samples | **0 / 0 / 0** |
| Cache misses / evictions | **0 / 0** |
| Worker CPU mean, one-core units | **44.6%** |
| Board CPU mean, 16-core normalized | **4.00%** |
| Worker RSS min–max, KiB / end-minus-start | **63,388–63,396 / +8** |
| Encoder delivery p99 / maximum | **151.1 / 154.6 ms** |
| FFplay reported drop delta | **276** |

Prediction presence in the exact 3,599-frame admitted cohort:

| Observation on both-missing board | Frames | Fraction |
|---|---:|---:|
| Expected IC_Fail | 3,488 | **96.9%** |
| Expected Terminal_Fail | 3,050 | **84.7%** |
| Both expected missing labels | 3,021 | **83.9%** |
| Unexpected IC_Pass | **1,051** | **29.2%** |
| Unexpected Terminal_pass | **0** | **0.0%** |
| No predictions | 67 | **1.9%** |

The absence of Terminal_pass candidates is encouraging for this single missing-
terminal specimen, not proof of universal terminal-presence accuracy. IC_Pass
still appears when the IC is missing, up to **0.413** confidence (median 0.075).
Those boxes must be localized against independently labelled inspection sites
before calling this a 29.2% false-acceptance rate. Both expected labels can
coexist with the unwanted pass label.

There are 18,842 IC_Fail candidates, with multiple candidates in 3,306 frames;
this is clutter, not thousands of independent missing ICs. No scores were
post-filtered and no model, ROI or threshold fix was attempted.

Evidence: `../IC_TERMINAL_TRAIN/benchmark-20261115-ic-missing-terminal-missing-01/`.
Exact cohort **146,830–150,428**, same source session as A/B. Report, 61 samples,
raw log windows, original specimen clarification and identical offline metric
re-computation retained. Existing 169-test source baseline unchanged; no new
unit run for this measurement.

### Three-state comparison

| Owner-confirmed physical state | Encoder fps | Expected pair present | Unwanted IC_Pass |
|---|---:|---:|---:|
| A: both present | 29.98 | 79.0% | IC_Pass is expected |
| B: IC missing, terminal present | 30.01 | 80.4% | **11.2%** |
| C: IC missing, terminal missing | 29.98 | 83.9% | **29.2%** |

These are different specimens/states, not controlled proof that removing the
terminal causes IC errors. Throughput is stable in all three cases; IC-side
localisation/class confusion/clutter is a clear next investigation. Missing ICs
still producing pass candidates merits attention before a host acceptance rule.
There is no baseline for IC present + terminal missing; an optional repeat A
for drift has not been captured. Three available states, not exhaustive accuracy
or reliability qualification.

## What these baseline cases establish

1. The native graph sustains approximately 30 fps with no observed encoder-ID
   gaps, timestamp errors/cache misses or board failures in this warm window.
2. Memory is stable over this short observation.
3. Delivery pacing and desktop drops remain measurable limitations even without
   a concurrent host ONNX evaluation in the window. Their cause is not isolated;
   do not blame either the model or player alone.
4. Good-part predictions are not dependable inspection decisions yet. Detection
   improvement needs measured visible-site misses, false labels, duplicates and
   class confusion—not just a higher confidence threshold.
5. No model, calibration, ROI, threshold, pinned release or service was changed.
   The benchmark does not interfere with the owner's continuing preview.

## Case sequence and swap protocol

| Case | Expected native classes | State |
|---|---|---|
| A: good IC + good terminal | IC_Pass, Terminal_pass | **Captured above** |
| B: IC missing + terminal present | IC_Fail, Terminal_pass | **Captured; owner clarified missing-only faults** |
| C: IC missing + terminal missing | IC_Fail, Terminal_Fail | **Captured above** |
| A repeat | IC_Pass, Terminal_pass | Suggested after swaps to check drift |

For each swap: stop the plate before handling it, keep camera/focus/lighting and
inward radial placement consistent, confirm the actual fault and readiness, then
measure the same 120-second warm window. Keep the live preview running. Do not
label a changeover window or infer specimen condition from model output.

Only missing components are in scope; do not claim solder/electrical defect
coverage or relabel other faults as equivalent. Record the actual condition. Raw expected-class presence can be artificially
boosted by background false positives, especially in bad states. Site-level
annotation will be needed before reporting precision/recall or false acceptance.

Keep this pipeline/performance protocol fixed while improving detection. Choose
thresholds using suitable validation data, not this baseline by trial and error.
The production video UI follows the detection-quality work; FFplay remains a
diagnostic viewer, not that UI.

## Reproduction

Tool: [`tools/analysis/benchmark_live_session.py`](../../tools/analysis/benchmark_live_session.py).
It observes an **already-running** session and never starts/stops the stream.
It refuses a wrong worker PID identity, changed process/session epoch, missing
clock tick metadata, incomplete log cohort or initially non-ready pipeline.

```powershell
python tools/analysis/benchmark_live_session.py `
  --session ../IC_TERMINAL_TRAIN/continuous-view-20261115-055448.session.json `
  --case good-inward-01 --expected IC_Pass Terminal_pass `
  --seconds 120 --interval 2 `
  --output <new-evidence-directory>
```

Change `--case` and `--expected` **only after** owner confirmation of a swap.
Use a fresh output directory. The session file records process identities; PID
reuse is checked against the live worker's exact arguments and process start
clock. CPU ticks come from Linux `AT_CLKTCK`, not a guessed frequency.

## Evidence and verification

`../IC_TERMINAL_TRAIN/benchmark-20261115-good-inward-01/`:

- `benchmark.json`: complete measurements and limitations.
- `samples.jsonl`: health, Linux process/system counters and viewer snapshots.
- `raw-window-frames.jsonl`, `raw-window-detections.jsonl`: overlapping byte-window
  copies of original append-only evidence. Only the exact admitted cohort is scored.
- `release.json`: pinned inventory, with inventory fingerprint in the report.
- `analysis-validation.json`: original report fingerprint, observer fingerprint
  and successful offline re-computation of identical metrics.

Case B uses the matching files under
`../IC_TERMINAL_TRAIN/benchmark-20261115-bad-ic-good-terminal-01/`.
Case C uses `../IC_TERMINAL_TRAIN/benchmark-20261115-ic-missing-terminal-missing-01/`,
including `specimen-confirmation.json` for the owner's fault-scope clarification.

Exact cohort: encoder frame IDs **41,451–45,049**, session
`3d046019-0531-4d13-afe9-78c1a214a924`, stream `decoder`.

Added 16 benchmark regressions: proc/auxv parsing, CPU units, percentile units,
partial log boundaries, stream/frame matching, missing/duplicate cohort refusal,
weak native scores, viewer counters, and summary CPU/RSS/leading-ID-gap math.
Full suite: **169 passed**, two existing dependency deprecation warnings. Unit
runs occurred outside the measured window. The baseline numbers were verified
again offline; no repeat live measurement was substituted.
