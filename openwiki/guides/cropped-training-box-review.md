# Cropped training-box proposals and manual review

Owner requested annotations for manual review, with training afterwards. Prepared
**42 native cropped images / 84 boxes**, all newly **pending**. No training started.

Open `../IC_TERMINAL_TRAIN/cropped-training-label-review-20261115/review.html`.
The local page requires no server/install and includes original-pixel toggle,
axis-aligned rectangle resizing/moving, fixed target/class identity, review/flag,
notes and JSON export/resume. Review actions do **not** autosave files.

## Geometry and class contract

Transferred owner-reviewed **source-site polygons** through saved source-to-crop
homographies, then proposed their axis-aligned bounding rectangles. These are
label seeds, not model predictions and not automatically approved tight boxes.
Previous source-site review is deliberately reset for this new training task.

Proposed training-box rules for owner review:

- **2 IC_Pass:** visible IC body plus its own pins; exclude unrelated neighbours.
- **1 Terminal_pass:** visible terminal housing, not wires/header/other parts.
- **0 IC_Fail:** expected empty IC mounting footprint including its pads.
- **3 Terminal_Fail:** expected empty terminal footprint including holes/pads.

Keep class IDs/order fixed and missing-only semantics. Header/passives are not
new targets; they remain hard-negative background. If glare/occlusion prevents
reliable geometry, flag the image rather than treating an inferred label as
certain. Site-derived seeds may need enlargement to include pads/pins or body
extent; no unseen component boundaries are claimed as ground truth.

## Owner workflow

1. Enter reviewer name; select IC or Terminal.
2. Drag corner handles to resize, or inside selected box to move. Axis alignment
   and image bounds are maintained. Editing geometry resets review to pending.
3. Check both boxes and class/state; mark reviewed or flag uncertainty with notes.
   IDs are readonly because specimen presence was already owner-confirmed; report
   suspected state mistakes rather than silently relabelling the specimen.
4. **Export review JSON before closing.** Save
   **`cropped-training.review-export.json`** to the repo root when ready.
   It is a DIFFERENT filename/schema from `annotations.review-export.json`:
   do not overwrite the previous source-site review.
5. Resume by importing the export into the same page. Imports reject changed
   source/pack identity, specimen state, target/class IDs and invalid rectangles.

Alternative: return image IDs/notes for assistant geometry correction. Seven
`review-sheets/` JPEGs and 42 `review-images/` raw/proposal comparisons provided.

## First-batch owner confirmation and next batch

Owner subsequently confirmed all first-batch boxes accurate. Approval of the
unchanged draft is recorded separately (conversation provenance, not a fabricated
browser export). [Batch02 and augmentation plan](training-batch02-and-augmentation.md)
now supplies 48 historical TRAIN-only class-balancing images for new manual
review. First draft files remain unchanged; training remains blocked pending QA.

## Training gate and coverage

| Class | Draft boxes |
|---|---:|
| IC_Fail | 40 |
| Terminal_pass | **2** |
| IC_Pass | **2** |
| Terminal_Fail | 40 |

This is the first correlated development batch, **not a complete final training
set**. 40 both-missing views and two adjacent both-present views; no third-state
capture coverage. Add representative present/third-state full rotations and
separate sessions/specimens for independent validation. Do not randomly split
adjacent frames or reuse the historical test set for tuning.

Every row and pack remain training-ineligible even after browser approval. No
`data.yaml`, train/validation/test split, training process, weight or calibration
change is generated. After export, validate/correct boxes, fill coverage and
explicitly finalize labels/splits before training a separate candidate. Any run
on these 42 alone would be a limited development experiment, not production proof.

`draft-yolo-DO-NOT-TRAIN/` contains the INITIAL normalized YOLO preview, clearly
marked provisional. Browser edits update only exported JSON, **not** those text
files. Regenerate final labels from the returned reviewed JSON; never train on
stale draft text. No global training export/enable button is provided.

Input crops are byte-identical to the frozen hybrid outputs, no extra image
conditioning. Serving/training/calibration crop, resizing and normalization must
be consistent; this preparation does not change the running board graph.

## Tooling and evidence

`tools/analysis/prepare_cropped_training_review.py` accepts `--review`, `--original`,
`--images`, `--hybrid`, `--output <new>`. Validates source review/hashes, exact crop
pixel regeneration/transforms, presence/class mapping and label bounds before
creating output. No model inference/board access. Template:
`tools/analysis/cropped_training_review.html`.

Pack includes draft JSON, images, classes, provisional YOLO labels, seven pages,
source/hybrid snapshots, frozen builder/template, browser smoke and validation.
84 labels round-trip between pixel/YOLO coordinates within **1e-5 pixels**;
source/crop/review hashes verified; actual approved count remains zero.

11 new helper tests; full suite **269 passed**, two existing warnings. Python
compile, JavaScript syntax and isolated Edge smoke (14 checks) pass. Browser test
approval/export is synthetic/in-memory and never changes real annotation files.
No board connection/preview interruption, model/runtime/release/SDK/vendor or
fixed-station changes; live health not rechecked.
