# Second crop-label batch and augmentation plan

## First batch accepted

Owner said the first crop-label batch was done and all boxes were accurate.
Recorded **conversation approval of the unchanged first draft**, tied to its exact
SHA-256, in batch02 `first-batch-owner-confirmation.json`. Original draft is not
rewritten and no browser export/reviewer click is invented. No first-batch crop
export was found in the repo root. If geometry was edited, obtain the actual
export before dataset finalization; unchanged verbal approval is recorded as such.
Training eligibility remains false pending data/split/config QA.

## Next review folder

Open `../IC_TERMINAL_TRAIN/cropped-training-label-review-batch02-20261115/review.html`.
**48 native cropped images / 96 NEW unreviewed box proposals**:

- **32 both-present**, adding IC_Pass and Terminal_pass.
- **16 IC-missing / terminal-present**, adding IC_Fail and Terminal_pass.

The first batch already contains 40 Terminal_Fail examples, so the next quota
prioritizes present classes rather than adding more both-missing frames.

| Class | First batch | Batch02 pending | Potential total |
|---|---:|---:|---:|
| IC_Fail | 40 | 16 | 56 |
| Terminal_pass | 2 | 48 | 50 |
| IC_Pass | 2 | 32 | 34 |
| Terminal_Fail | 40 | 0 | 40 |

After approval, 90 images/180 boxes form a better-balanced development pool,
**not 90 independent boards or a validated production dataset**.

## Source selection and preprocessing

Only historical `dataset/v4-1280` **train** images from runs `present`/`absent`
are considered: 42 each. Validation/test image files are never consumed or
reassigned. Manifest ancestor/processed-image hashes must not overlap holdouts;
training image files and label files are fingerprinted. Raw ancestor hashes are
recorded from the original manifest, not claimed to be newly recovered raw files.

These are **older-camera recordings**, not fresh current-camera captures.
Verified the known scale-1 1280-square letterbox with constant-128 padding, removed
exactly 280 rows from top and bottom to recover unchanged 1280x720 content pixels.
No resizing or guessing the plate window. Historical YOLO y coordinates are
un-padded by the same 280 pixels.

Frozen hybrid crop accepts **76/84 candidates**. Eight unknowns are logged and
visualized in `quarantine/`, not forced into manually guessed crops. Among accepted
sources, deterministic farthest-angle sampling selects 32 present and 16 absent;
uses source label centres for sampling angle ONLY, not crop localization. Both
sets span all twelve 30-degree orbit bins; largest gaps 33.88° / 31.86°. Selection
is by class and crop availability/angle, **not best detector results or label
coverage**. The exclusions and all selections are retained in `selection.json`.

Historical train labels are projected through pixel-derived source-to-crop
transforms and enveloped as axis-aligned boxes. They may become loose after
rotation: check target tightness and consistency against the approved first batch.
All new boxes start pending, with source/run/frame/hash/split provenance. Historical
train/test share source recordings despite exact-image hash separation; this batch
does not establish independently held-out part/session performance.

## Review and return

Same rectangle editor: select target, resize/move, review/flag with notes.
Use first-batch accepted geometry conventions consistently. No detections/model
predictions used as labels. Eight static review pages and 48 enlarged raw/proposal
views available. Flag glare/ambiguous geometry rather than inventing certainty.

**Export before closing**; changes are not file-autosaved. Return
**`cropped-training-batch02.review-export.json`** to repo root. The filename is
batch-specific; do not overwrite previous site/first-batch exports. Resume the
same page from this export. Browser edits do not update provisional YOLO text;
final labels must be regenerated and validated from review data.

## Batch02 review returned and preparation completed

Owner returned all 48 reviewed rows, correcting one IC box. The [prepared training-
only pool](prepared-training-pool-20261115.md) now has 90 parents / 450 baked views,
with deterministic preprocessing and explicit train-only augmentation. No training
has started; separate validation/evaluation decision remains. Proposal ordering
below is retained as rationale; exact executed recipe is in the prepared-pool guide.

## Answer: is the first batch enough? What augmentation/preprocessing?

**42 heavily correlated images with only two pass examples per class are not
enough for reliable four-class training/evaluation.** Augmentation helps diversity
but does not create new independent specimens, states or illumination conditions.
This second batch improves balance; independent current-camera validation is
still needed, particularly full rotations of the present/mixed states.

Recommended ordering, **not executed yet**:

1. Finalize reviewed original labels and source/capture/specimen-aware splits.
   Keep historical holdouts untouched; don't randomly split adjacent derived crops.
2. Freeze deterministic input processing across training, validation, calibration
   and serving: verified hybrid crop, aspect-preserving model resize/padding,
   colour conversion and normalization. No untested CLAHE, sharpening or imported
   fixed-station illumination LUT is added simply because training is planned.
3. Apply **label-aware augmentation only to the training branch**, preferably
   on the fly: rotation including 180° (crop direction is ambiguous), modest
   exposure/contrast, blur and scale variation. Geometric transforms must update
   boxes and avoid silently clipping/removing targets or changing class meaning.
   Do not use hue changes/extreme mosaics/reflections indiscriminately.
4. Validation/test get only deterministic serving preprocessing, not stochastic
   augmentation; never put augmented siblings across splits. Keep calibration
   representative of actual serving inputs, with independent lineage checks.
5. Train a separate candidate only after review/coverage/config agreement.
   Existing `train_yolo11.py` still points to the old v4 dataset and deliberately
   disables geometric augmentation; it was **not changed or launched**. A new
   candidate run/config must not overwrite the old run (`exist_ok=True` there).

No numeric augmentation hyperparameters or training schedule are selected by this
annotation step. Any run on this small correlated pool alone is a development
experiment, not proof of production recall/precision.

## Tooling and verification

`prepare_next_training_batch.py --dataset <v4-1280> --first-batch <original-draft>
--output <new>` validates known transform/class/split/hash contracts, uses frozen
hybrid policy, selects quotas by circular angle, transfers labels and writes the
review pack. No training YAML/final split/augmentation generated. Inputs unchanged.

10 helper tests cover lossless unpadding, invalid padding/labels, circular angle
selection, deterministic quotas, insufficient-source refusal and source/model
holdout-hash guards. Full suite **279 passed**, two existing warnings; compile
checks pass. Isolated browser smoke 14 checks, actual new labels remain pending.
Source/crop hashes, selected train-only membership, no cross-batch duplicate crop,
YOLO coordinates and all class counts verified. No board connection, preview
interruption, weights/calibration/runtime/release/SDK/vendor/fixed-station change.
Live health not checked during this offline work.
