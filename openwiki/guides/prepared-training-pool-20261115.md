# Reviewed crop training pool: preprocessing and augmentation

Owner supplied batch02 review and requested augmentation/preprocessing before
training. **Preparation complete; no training started.**

Open `../IC_TERMINAL_TRAIN/augmented-training-pool-20261115/index.html` for all
90 labelled augmentation strips. `START-HERE.md` explains data and readiness.

## Review consolidation

- First batch: 42 crops/84 boxes, unchanged-draft owner conversation approval
  verified against exact SHA and image/box counts. No browser export invented.
- Batch02: 48 crops/96 boxes, all named-human-reviewed in repo-root
  `cropped-training-batch02.review-export.json`. Contract, image identity, state,
  target/class IDs, box bounds and unique IDs validated against its original draft.
- One correction—IC box at `hist-absent_006413`—is used from the returned review,
  not from stale provisional YOLO text. All input files remain unchanged.
- Exact source crop PNG hashes verified; duplicate parents refused. Original
  historical train-only membership retained. Capture/source lineage preserved.

All **90 parents assigned TRAIN before augmentation**, not randomly divided into
adjacent-frame splits. Four correlated capture/run groups remain; these are not
90 independent physical boards. Validation/test must come from separate sources.

## Deterministic input processing

The frozen hybrid native RGB crop is already produced in the source packs. No
new crop tuning/re-estimation, CLAHE, sharpening, hue correction or illumination
LUT was introduced. Native crop → bilinear aspect-preserving resize → centred
padding **114** → **1280x1280** uint8 PNG. Actual rounded x/y scales transform
box coordinates. Loader still must convert to RGB float and divide by 255 once;
normalized float values are not incorrectly saved into an 8-bit image file.

This is the candidate's input contract, not the currently running full-frame r33
input path. Candidate training, calibration and future serving must match; do not
drop crop-trained weights into the unchanged full-frame graph and assume parity.

## Reproducible train-only augmentation

Five views per parent, with fixed root seed **20261115** and a distinct stable
SHA-derived seed per parent/variant:

| View | Transformation |
|---|---|
| base | Deterministic input processing only |
| rotate180 | Exact native 180-degree rotation; handles crop direction ambiguity |
| exposure | Native contrast 0.90–1.10 / brightness −10..+10, then resize/pad; padding remains 114 |
| geometry | −8..+8° rotation and 85–98% canvas occupancy; full rotated source fits |
| mild_blur | 3x3 Gaussian sigma 0.6–1.0 AFTER model resize, not on tiny native IC features |

Labels transformed alongside pixels; all four box corners mapped then enveloped
as axis-aligned boxes. Out-of-frame targets cause failure, never silent clipping
or dropping. Class IDs/meaning unchanged. No mosaic, arbitrary target-cutting
crop, mirror, shear, hue shifts or additional enhancement.

These are baked images, not five independent examples each. At training time
**disable additional online augmentation** to avoid stacking transforms; saved
`recommended-loader-overrides.json` records compatible options but selects no
model, epoch schedule, device or run. Old trainer remains unchanged/not invoked.

## Counts and verification

| Class | Original boxes | Five-view total |
|---|---:|---:|
| IC_Fail | 56 | 280 |
| Terminal_pass | 50 | 250 |
| IC_Pass | 34 | 170 |
| Terminal_Fail | 40 | 200 |

**450 training PNGs / 450 label files / 900 boxes**. `native-parents/` preserves
all 90 exact input crops; `previews/` and gallery show all 90 five-view strips.
`manifest.json` records parent/capture/split, variant, parameters, seed, image/label
hashes and native→model transform for every derived file. `provenance/` contains
exact review/draft/approval snapshots; `source-snapshot/` freezes builder/helpers.

Independent output validation regenerated **all 450 images pixel-for-pixel**,
recomputed every label exactly, checked all 900 normalized-label round trips
within 1e-5 pixels, and verified source/review/script hashes and gallery links.
No validation/test directory, train-as-val alias or YAML was invented.

14 new helper tests; full suite **293 passed**, two existing warnings; compilation
passes. Tests cover determinism, bounds/classes, 180°/letterbox mapping, clipping
refusal, per-parent RNG, corrected review use, identity/class/review gates and
exact-hash binding of verbal approval.

## Training readiness / next decision

**`training_ready: false`** in `training-readiness.json`: no separate validation
set or approved evaluation plan is supplied. All existing parents were already
development examples; never move their augmented descendants into validation or
point `val` at `train`. Augmentation cannot replace new states/specimens/captures.

Preferred next step: capture/review separate current-camera full rotations across
the three available board states. Reserved historical validation can alternatively
be prepared/reviewed for an explicitly limited diagnostic evaluation, kept out of
training; it shares recordings with historical train and does not establish
independent-part production accuracy. Do not automatically copy its unreviewed
labels into a supposedly approved validation set.

After the validation decision, finalize explicit data/config/run names and start
a separate candidate—not the old v4 trainer/run. Use representative unaugmented,
serving-compatible inputs for later calibration with traceable lineage. No model,
calibration, compilation, runtime/release/SDK/vendor/fixed-station change occurred.
No board connection/preview interruption; live health not rechecked.

## Repeat

```powershell
python tools/analysis/prepare_augmented_training_pool.py `
  --first ../IC_TERMINAL_TRAIN/cropped-training-label-review-20261115 `
  --second ../IC_TERMINAL_TRAIN/cropped-training-label-review-batch02-20261115 `
  --review cropped-training-batch02.review-export.json `
  --output <new-directory>
```

Output must be new. Data/preprocessing/augmentation are reproducible under the
recorded OpenCV version; recipe policy is frozen in the manifest, not an unrecorded
random export. Owner can inspect augmentation before any optimizer is launched.
