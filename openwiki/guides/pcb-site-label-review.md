# PCB / inspection-site draft labels and owner review

## Owner-reviewed export received

Owner exported `annotations.review-export.json` to the repo root. All **42**
records are human-reviewed; source/contract/geometry validation passes. 16 PCB
quads and all 42 site records changed. This file is the current corrected
annotation version, separate from the unchanged original draft described below.
It is still development data, not a held-out validation set or training export.
Exact snapshots/hashes are preserved in the [Canny experiment](../operations/canny-crop-experiment-20261115.md).
Do not overwrite completed experiment snapshots when making later corrections.

## Original draft pack

**Open** `../IC_TERMINAL_TRAIN/site-label-draft-20261115/review-pack-final/review.html`
in a local browser. It is self-contained except for the adjacent `images/`
folder; no server, upload, board connection or new installation is needed.
Browser smoke-tested in isolated headless Edge, including image load, navigation,
review/reset, dragging coordinates, invalid-import refusal and valid resume.

There are **42 image drafts, 126 polygons, zero independently approved labels**:
40 both-missing and two adjacent both-present. Each image has four ordered PCB
corners, header-to-terminal direction, IC footprint and terminal footprint.
10 are high priority. All inputs and proposal provenance are hash-pinned.

This is **not** a completed labelled validation set: current images are all
**development-only**, because the prior methods were explored on them. Need full
rotation source views for both-present and IC-missing/terminal-present, separate
held-out capture sessions and independent geometry review. Same-specimen sessions
do not establish generalization to independently manufactured parts.

## Annotation contract

Coordinates use the original 1280x720 image; no letterbox/padding coordinates.
Corner order: `header_top`, `terminal_top`, `terminal_bottom`, `header_bottom`.
Top/bottom refer to canonical board coordinates with header on the left, not to
the top of the camera image. Winding is positive in image x/y coordinates.
Outline the **substrate**, excluding protruding pins and component overhangs.

IC/terminal polygons describe their **expected mounting footprints**, even when
empty. They are not tight visible-body bounding boxes and must not silently
replace the historical four-class YOLO labels. Presence state is inherited from
owner-confirmed specimens, not inferred from detector output. Class IDs remain
0 missing IC, 1 present terminal, 2 present IC, 3 missing terminal.

Proposal provenance is explicit:

- 32 images: unchanged machine-seeded substrate geometry from colour experiment.
- 7: assistant manually outlined provisional substrate quads from enlarged source
  grids (four refused missing views, two present views, partial-fit frame 678).
- 3: original machine geometry retained after assistant reinspection disproved
  earlier reversal claims (243, 272, 1084).
- All 84 site polygons: projected from provisional template coordinates, **not**
  independently traced. Obscured corners and footprint geometry need review.

Every row is pending, reviewer null, `training_eligible=false`, and
`geometry_independently_reviewed=false`. No independent acceptance is claimed by
an assistant visual pass or by reusing the method being evaluated.

## Owner workflow

1. Open the HTML; enter reviewer name. Start with **High priority only** if useful.
2. Toggle **None — original pixels** to avoid overlay anchoring. Then check cyan
   substrate corners, direction toward terminal, yellow IC and magenta terminal.
3. Drag vertices to adjust geometry, or simply report image IDs/notes to the
   assistant for correction. Moving PCB corners does not move sites automatically.
4. Mark reviewed only after checking both pose and sites; otherwise flag uncertain.
   Any vertex edit resets that image to pending. Concave/out-of-bounds proposals
   cannot be approved. Named review is recorded but never enables training.
5. **Export review JSON** before closing. Return the downloaded
   `annotations.review-export.json` for validation; reload it into the same page
   to resume. Browser edits stay in memory until export.

Seven static pages under `review-sheets/` show original crops beside proposals.
`review-images/<id>.jpg` provides individual enlarged comparisons. Separate
`labelme/<id>.json` files are an alternative editor format; edits in one format
do not update the other. Do not mix them without explicit merge/validation.

Dataset `START-HERE.md` has the same short review instructions. The source copies
are byte-identical PNGs without rendered detections. No live pipeline was touched
or checked during preparation; prior preview safety duration still applies.

## Correction to the earlier experiment report

Closer individual canonical/source inspection **withdraws the earlier four
180-degree reversal claims**. The previous contact sheet was misread. Frames
243/272/678/1084 have the original direction, not reversed direction.
However, frame 678's colour fit covers only the **lower part** of the substrate;
that is a real localization error. Counts remain 36 candidates/six unknown for
colour and 13 candidates/29 unknown for scoped ORB. Neither is validated for
production. This correction is appended to the changelog; old entries remain.

Old evidence stays intact. `visual-review-correction.json` and
`selected-review-corrected.jpg` under `site-pose-experiment-20261115/` supersede the
wrong reversal list/caption, not the captured pixels or numerical experiment data.
The draft pack's first two assembly attempts remain marked **superseded**; use
only `review-pack-final` (first attempt temporarily applied wrong flips).

## Build and test

`tools/analysis/prepare_portable_site_labels.py` accepts `--manifest`,
`--proposals`, `--overrides`, `--output <new>`. Validates all input hashes, IDs,
state mappings and convex/in-bounds geometry before creating output. Copies source
PNGs byte-for-byte, writes draft JSON/LabelMe, seven review sheets and browser UI
from `tools/analysis/portable_site_review.html`. Never exports training labels.

Evidence root: `../IC_TERMINAL_TRAIN/site-label-draft-20261115/`, including inputs,
frozen builder/template, manual grids, browser smoke results and version history.
16 helper tests added; full suite **218 passed**, two existing deprecation
warnings. Python compilation and JavaScript syntax pass. Isolated Edge smoke has
13 passing checks, exit 0; its synthetic review action was reset and never saved
to annotation data. Actual pack remains **0/42 approved**.
