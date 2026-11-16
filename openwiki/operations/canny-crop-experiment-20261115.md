# Canny-based PCB cropping — 2026-11-15

## Outcome

Owner requested Canny cropping before further preprocessing. **Useful boundary
signal, not yet a safe standalone crop selector. No live change deployed.**

The strongest result among nine exploratory configurations preserves the reviewed
PCB and both target polygons in **28/42** images. The other **14** select a fixed
mounting bracket, with **zero overlap** with all three reviewed polygons. Both
saved both-present views fail. This is a crop-localization result, **not detector
precision/recall**, and is not held-out performance.

In the 28 successful crops, median crop area is **2.73% of the full frame**—about
**97% of the original image area is excluded**. Cropping therefore does remove
substantial irrelevant surroundings when the correct object is selected.

## Owner-reviewed annotations received

Repo-root `annotations.review-export.json` was read and preserved unchanged.
All **42** records are marked reviewed with a named reviewer; checks against the
original draft confirm source IDs/hashes, image dimensions, capture groups,
state, class mapping, split, corner order, finite convex polygons and image bounds.
**16 PCB quads and all 42 site records differ** from the original draft.

Human-reviewed geometry is now available for development comparisons. The
original draft remains unmodified/pending; imported review is a separate version.
No training export, train/test split or release activation occurred. Existing
coverage remains 40 both-missing, two adjacent both-present from earlier framing,
zero matched third-state sources. New independent sessions are still needed.

Every experiment retains an exact export snapshot and its SHA-256. Do not replace
these snapshots with a subsequent review export; make a new experiment directory.

## Method

Each source image independently follows:

1. Convert BGR source pixels to grayscale.
2. Gaussian smooth, 5x5, sigma 1.2.
3. Canny with L2 gradient, threshold pairs **30/90, 50/150, 80/200**.
4. Optional 3x3 morphological closing, **0, 1 or 2 iterations**.
5. Find contours (including nested contours); filter plausible board-sized,
   elongated rectangles. Rank by rectangularity, internal edge density and area.
   No colour mask, manual image ROI, reviewed polygon or detector box enters
   this selection. Deduplicate overlapping contours of the same candidate;
   refuse spatially distinct near-ties or padded crops outside the source image.
6. Expand chosen rectangle by **16 source pixels per side**. Warp the original
   **colour pixels**, not the edge image, into a long-axis-horizontal crop.
7. Save native-size crop and aspect-preserving **512x256 letterboxed preview**.
   Preview size is not a change to the detector's input contract.

Only after selecting a rectangle, compare it with owner-reviewed PCB/IC/terminal
polygons. Diagnostic preservation requires >=98% of **each** polygon retained;
also record strict all-vertices-inside coverage. Both counts agree in these runs.
This is not a certified acceptance threshold. A returned `located` status means
an edge candidate was selected, not that PCB identity has been verified.

Canny's rectangle does not resolve header-versus-terminal direction: **180-degree
semantic orientation remains unresolved**. Upscaling makes the crop easier to
view; it cannot recover detail not captured by the camera.

## Results and exploratory sequence

First predefined threshold sweep used two closing iterations. It exposed repeated
bracket selection. Then tested zero/one closing iteration to check whether edge
joining contributed to lost PCB contours. All earlier results were kept. Final
canonical two-iteration rerun matches the initial counts and shares the frozen
script hash with the two ablations. No configuration was silently promoted.

| Closing iterations | Canny pair | Selected | All targets preserved | Wrong target / clipped | Unknown |
|---:|---|---:|---:|---:|---:|
| 0 | 30/90 | 18 | 13 | 5 | 24 |
| 0 | 50/150 | 18 | 13 | 5 | 24 |
| 0 | 80/200 | 8 | 3 | 5 | 34 |
| 1 | 30/90 | 42 | **28** | **14** | 0 |
| 1 | 50/150 | 42 | 27 | 15 | 0 |
| 1 | 80/200 | 33 | 16 | 17 | 9 |
| 2 | 30/90 | 42 | 26 | 16 | 0 |
| 2 | 50/150 | 42 | 27 | 15 | 0 |
| 2 | 80/200 | 33 | 15 | 18 | 9 |

For the strongest development result (30/90, one closing iteration): 28/40
both-missing preserved; **0/2 both-present**. Failed IDs are 185, 214, 243, 272,
620, 649, 678, 707, 1084, 1113, 1142, 1171 and both present images.
All 14 have zero overlap with the reviewed PCB and sites, rather than merely
small edge clipping. Several saved crops/edge overlays were inspected directly;
they show the fixed bracket instead of the PCB. Do not interpret 42 selected
rectangles as 42 found boards. These are development images used for exploration,
so choosing the best row is not an independent validation result.

## Evidence and repeat

- `../IC_TERMINAL_TRAIN/canny-crop-20261115-close1/selected-review.jpg`:
  original image, Canny/candidate/review overlay, RGB crop—success and failures.
- `...-close1/low/review-page-01.jpg` through page-06: all 42 previews, including
  failed crops. `crop-*.png` are native-size original-colour warps;
  `rescaled-*.png` are padded previews; `edges-*.png` contain only Canny output.
- Roots `canny-crop-20261115-close0`, `...-close1`, `...-close2-final` hold complete
  results, snapshots, all presets and frame-level overlap measurements. The
  initial `canny-crop-20261115` run stays as historical evidence.
- Frozen final source: `...-close2-final/experiment-source.py`.
- `...-close1/analysis-validation.json` verifies shared script/review hashes,
  exact export snapshots and numerical summaries across final runs.

```powershell
python tools/analysis/portable_canny_crop.py `
  --review annotations.review-export.json `
  --original ../IC_TERMINAL_TRAIN/site-label-draft-20261115/review-pack-final/annotations.draft.json `
  --images ../IC_TERMINAL_TRAIN/site-label-draft-20261115/review-pack-final `
  --close-iterations 1 --output <new-directory>
```

Default closing is two iterations; allowed ablations are 0/1/2. Output directories
must be new. Reviewed polygons are used only for evaluation/green overlay, not
selection or crop generation. Source images and review JSON stay unchanged.

17 new helper tests cover rotation/crop mapping, nested-contour deduplication,
blank/ambiguous scenes, coverage arithmetic, ablation bounds and review integrity.
Full suite **235 passed**, two existing dependency warnings; compilation passes.
No board connection, stream interruption, installed SDK/vendor/fixed-system
change, model inference, confidence/ROI/config/release change or training.
Live health not checked during offline work. No board-speed/latency claim.

## Recommended next experiment

**Completed with owner approval:** [colour-seed/local-Canny follow-up](hybrid-crop-experiment-20261115.md)
preserves all reviewed targets in 42/42 development images. The older results
above remain the unchanged Canny-only baseline; no live deployment occurred.

**Combine Canny boundary evidence with PCB identity evidence**, such as substrate
colour/appearance and grouping nearby PCB fragments. Canny alone cannot tell a
rectangular bracket from a PCB. A justified plate-interior constraint may also
exclude fixed hardware, but must not clip the full rotation or present components.
Do not choose a per-image crop by its match to the reviewed labels.

Require preservation of reviewed geometry, explicit unknown on ambiguity, and
new-state/held-out-session checks before deployment. Once cropping is reliable,
evaluate subsequent image conditioning and FP32/compiled detection on that input
path. Cropping/rescaling changes the model's input distribution; a cleaner picture
alone does not establish improved precision/recall or justify changing calibration.
