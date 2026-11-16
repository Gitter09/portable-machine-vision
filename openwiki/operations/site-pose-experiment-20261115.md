# Offline PCB pose/site experiment — 2026-11-15

## Decision

**Do not deploy either tested method.** Site-aware inspection remains promising,
but localization must be made reliable first. Removing boxes when alignment is
wrong or unavailable is not an improvement in recall or inspection accuracy.

Owner approved an offline feasibility experiment, not a live detector change.
No board connection, graph start/stop, release change, threshold adjustment,
training, calibration or service operation occurred. Existing preview was not
interrupted; its previously recorded four-hour bound remains applicable. Live
status was not rechecked during this offline experiment.

## Inputs and controls

42 original, unannotated 1280x720 source images, hash-pinned in
`../IC_TERMINAL_TRAIN/site-pose-experiment-20261115/manifest.json`:

- **40 both-missing** images from the previous periodic matched-buffer capture.
- **2 both-present** images from the earlier exact-buffer probe. These are adjacent
  views from before the owner moved the PCB inward, not a full good-board rotation.
- **No IC-missing/terminal-present source images** available in these captures.
  That baseline retains predictions but no matched source pictures.

Saved annotated video/images were deliberately not used for feature extraction:
rendered model boxes could become alignment landmarks. Methods consume original
pixels only. Native boxes are used afterwards for a counterfactual gate, not for
locating the board or defining its orientation.

Reference is frame 823, chosen as a clear, correctly oriented both-missing view.
Provisional IC/terminal polygons are hand-chosen normalized substrate coordinates,
not independently drawn ground truth. The reference is included in the 42 images;
this is exploratory same-capture feasibility, **not held-out validation**.

## Methods and measured results

| Method | Candidate poses | Unknown | Observed failure |
|---|---:|---:|---|
| Blue substrate silhouette + neutral-dark header direction | 36/42 | 6/42 | Partial-substrate fit at 678; both present images unknown. Earlier four reversal claims withdrawn after enlarged reinspection |
| PCB-restricted ORB + RANSAC affine alignment | 13/42 | 29/42 | Insufficient RANSAC support on 29; both present images unknown |

### Colour/outline method

Threshold blue substrate, close small gaps, find a plausible elongated board,
rectify its rectangle and resolve 180 degrees from the neutral-dark header end.
No missing-component cue is deliberately used to determine orientation.

36 candidate poses are all on the both-missing images. Unknown IDs:
`185, 214, 649, 1113`, plus both present images. A visual review of all canonical
panels originally led to a mistaken claim of four reversed directions. **Correction
on preparing enlarged labels:** individual canonical/source views at 243, 272,
678 and 1084 do not show those reversals. Frame 678 instead fits only the lower
substrate. No candidate poses have been measured against independent corner/site
labels; 36 must not be reported as 36 correct alignments. See
[annotation preparation and correction](../guides/pcb-site-label-review.md).

Present components split/obscure the visible blue substrate. Some other views
also fail the conservative silhouette criteria, or fit only part of the substrate.
The earlier claim that dark-end direction reversed these four views is withdrawn.
Independent orientation error remains unmeasured. These findings are not proof
that colour localization can never work with a better method.

### ORB/RANSAC method

Reference features restricted to the PCB neighbourhood; provisional IC and
terminal sites excluded from the reference feature mask. Source features use a
blue-neighbourhood mask. ORB operates on a 2x view, maps points back to source
coordinates, and uses partial-affine RANSAC, minimum support, spatial spread,
scale and image-boundary checks. Upsampling adds no new camera detail.

13 accepted IDs: `301, 330, 359, 388, 417, 446, 475, 736, 765, 794, 823, 852, 881`.
Canonical views look plausible, but no quantitative corner-error claim is made.
One is the reference itself. All 29 refusals have insufficient RANSAC support.
The small PCB and repetitive features are possible contributors, not an isolated
root cause. This does not establish that every ORB configuration would fail.

**Important exploratory failure preserved:** an initial ORB reference mask covered
all blue regions in the full frame. It locked onto stationary background features
and misleadingly reported 38 candidate poses, often on the empty plate. Visual
contact sheets exposed this immediately. Restricting reference features to the
PCB removed that mistake but revealed poor usable coverage. Keep `trial-02-orb`
as negative evidence; its reported 38 is NOT an alignment success count.

## Counterfactual box filtering — not deployed

Using only mapped class-appropriate site polygons, retain a native box if its
centre lies inside its expected site. Do not create verdicts, choose the strongest
box or enforce one result per site. Unknown pose retains nothing and is counted
separately; it must become an UNKNOWN inspection, never an implicit pass/fail.

Raw native boxes across the 40 both-missing images, including marginal scores:

| Class | Before | Colour gate retains | Scoped ORB gate retains |
|---|---:|---:|---:|
| IC_Fail | 174 | 103 | 35 |
| Terminal_pass | 0 | 0 | 0 |
| IC_Pass | 5 | 0 | 0 |
| Terminal_Fail | 39 | 39 | 11 |

All five sampled unwanted IC_Pass boxes lie outside the colour method's mapped
IC site, on located frames. Scoped ORB excludes four on located frames; the fifth
is on an unknown frame. **Do not credit that fifth to successful localization.**
For scoped ORB, 113 IC_Fail and 28 Terminal_Fail boxes are rejected because pose
is unknown. This loss of inspection coverage makes its clean-looking output
unacceptable, even though unwanted pass boxes disappear.

Removing 71 IC_Fail candidates in the colour method is not 71 corrected errors:
some may be duplicates/background, some may be lost valid evidence. Nine of those
are on unknown frames. No independent site/box labels exist to classify the rest.
The confirmed partial-board fit and unvalidated coverage disqualify production use. These numbers
are neither precision/recall nor a site-level false-acceptance rate.

## Evidence and reproducibility

Root: `../IC_TERMINAL_TRAIN/site-pose-experiment-20261115/`.

- **`selected-review-corrected.jpg`**: plausible sites, blur/glare and unknown
  views. Original `selected-review.jpg` has a mistaken reversal caption; retained
  as history, not current evidence of reversal.
- `final-color/`, `final-scoped-orb/`: results JSON, full-frame overlays and
  canonical contact sheets; final runs share the recorded script hash.
- `visual-review.json`: initial assistant review, including withdrawn reversal
  claims; `visual-review-correction.json` explicitly supersedes those claims.
- `manifest.json`: source paths, state, identities and hashes.
- `experiment-source.py`: frozen final source, matching final results SHA-256.
- `trial-*`: exploratory history, not overwritten by final reruns. Early trial
  script revisions were not archived; final runs are the reproducible versions.

Run with the sibling environment supplying OpenCV/NumPy:

```powershell
python tools/analysis/portable_site_experiment.py `
  --manifest ../IC_TERMINAL_TRAIN/site-pose-experiment-20261115/manifest.json `
  --comparison ../IC_TERMINAL_TRAIN/live-false-pass-20261115/comparison/comparison.json `
  --output <new-directory>
```

Add `--reference ../IC_TERMINAL_TRAIN/live-false-pass-20261115/comparison/source-823.png`
for PCB-restricted ORB/RANSAC. Outputs must be new directories. The helper has
no board credentials, model inference or live-pipeline dependency. Recorded host
invocation timings are not board throughput/latency benchmarks.

12 new helper tests cover synthetic rotation/direction, ambiguous boards, missing
header cues, site-coordinate round trips, class-specific gating, unknown handling,
and restriction of reference features away from background. Initial synthetic
fixture exceeded the intended area bound; resized the fixture, not the algorithm.
Full suite: **202 passed**, two existing dependency deprecation warnings; compile
checks pass. Tests establish helper behaviour, not real-world alignment accuracy.

## Recommended next step

Build a small **independently labelled PCB-pose/site validation set**, adding full
rotation source images for both-present and IC-missing/terminal-present states
and separate capture sessions. Review all apparent successes as well as failures.
Measure corner/site error, wrong-direction rate and unknown coverage per state.

Then choose a robust locator: improved geometric/template landmarks, a trained
PCB/landmark detector, or (if operationally acceptable) a fiducial rigidly attached
to the board/carrier. Do not assume the fixed station's reference or illumination
LUT transfers to this camera. Only after localization is dependable should we
validate site-aware filtering/classification and detector precision/recall.

No arbitrary live suppression, confidence increase or training change is justified
by this experiment. The stream remains unchanged and still diagnostic.
