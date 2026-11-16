# First current-camera known-good trial — 2026-11-15

## Specimen and unchanged settings

The owner confirmed: **“placed a good part with both terminal & IC present.”**
Expected classes are therefore `IC_Pass` and `Terminal_pass`. This confirms the
specimen state, not independently drawn boxes in every frame. One physical part
was tested; these are not independent repeated-part trials.

Used pinned **r33**, diagnostic confidence **0.05**, unchanged weights,
calibration, native preprocessing/decoding and receiver target. No ROI, threshold,
model or service changes were made. Tests and players are stopped afterward;
the separate capture feed remains stopped.

## Tests performed

### 1. Initial 45-second requested windowed run

- Board admitted **1,266 frames**, launcher exit 0, no wrapper errors.
- **1,325 detection records**: both expected pass classes occur in every record.
- `IC_Fail` occurs in **19 records**, scores approximately 0.049–0.068.
- Best IC_Pass box position changes only about two pixels; this is a limited-view
  result, **not proof across a complete rotation**.
- Duplicate pass boxes occur. Presence of both class names is not equivalent to
  precisely locating exactly one IC and exactly one terminal.

### 2. Bounded exact-buffer probe

Captured decoder pixels, preprocessed input, raw MLA output and native boxes for
matching frame IDs **20 and 21** in a separate diagnostic run. Decoder NV12
images and their native boxes are retained. This is not a throughput run and is
not temporally paired to the first or subsequent video run.

Two small IC_Fail boxes appear on reflections on the white plate, away from the
physical PCB. The actual part receives IC_Pass and Terminal_pass labels, with
weaker terminal confidence at this view.

A software decoding of the **same captured MLA tensors** agrees with native
box coordinates up to normal rounding for retained predictions. Native output
also retains a marginal score about 0.04871, below the Python diagnostic's strict
0.05 filter; this near-threshold difference is recorded, not silently removed
from the native counts.

FP32 on the dequantized companion input sees IC_Pass on both frames; it sees
Terminal_pass on frame 21 but not frame 20 at 0.05. This reference uses the exact
quantized input values converted back to float, not the original pre-quantization
image. The compiled outputs have weak extra IC_Fail predictions. Therefore:

- This is not evidence of a new wholesale box-layout failure.
- Weak detections/quantisation differences still matter.
- The reference itself is not consistently detecting the terminal at that view.
- Two frames do not settle full-rotation model accuracy or its underlying cause.

### 3. Recorded moving-part run

A separate receive/decode run recorded about **60 seconds** of annotated video:

| Check | Result |
|---|---:|
| Bounded FFmpeg decoded frames / exit | **1,800 / 0** |
| Board admitted frames / launcher exit | **1,837 / 0** |
| Detection records | **1,896** |
| Archive H.264 dimensions | 1280×720 |
| Complete archive video frames | **1,825** |
| Archive duration | 59.995 s |

The first FFmpeg output is a bounded decoder; the second independently copies
approximately 60 seconds to MKV. Their counts are not expected to be identical.
Detection records include pipeline tail frames that were not admitted by the
encoder. Do **not** claim one-to-one video/detection pairing from file index.
The MP4 viewing copy is a stream-copy remux, not a new inference run.

Descriptive class-presence counts across the **1,896 detection records**:

| Observation | Records |
|---|---:|
| Both pass labels present | **1,283 (67.7%)** |
| IC_Pass absent | 158 |
| Terminal_pass absent | 508 |
| Any fail label present | **634 (33.4%)** |
| No predictions | 41 |

IC_Fail appears in 629 records, maximum score **0.345**. Terminal_Fail appears
in 11, maximum score **0.115**; some records contain both fail classes. These
are unwanted fail-class outputs for the owner-confirmed good specimen, but no
host inspection verdict/reject logic was exercised.

These are **not precision/recall or usable inspection-yield estimates**: some
parts leave the image, box-level ground truth is absent, duplicate/background
boxes exist, and the records are correlated views of one part.

## Main roadblock: clipping during rotation

Saved frames directly show the PCB leaving the **bottom and top of the 720-pixel
image**. At some angles part of the terminal/PCB is outside the field of view.
Consequently, loss of a pass label at those angles cannot fairly be attributed
to the detector alone.

A supporting blue-PCB mask screen sampled every tenth archived frame: **44 of
183** candidates touch or approach a vertical border. This is a heuristic framing
screen, not annotated ground truth or an exact clipping rate. The saved images,
not the mask count alone, establish actual clipping.

This supersedes the older framing measurement of zero clipped frames: that was
a different placement/checkpoint. Do not assume an arbitrary new radial position
on the plate meets the old camera-clearance measurement.

False IC_Fail labels also occur on plate glare and on other PCB regions. Fixing
framing will not automatically remove those extras. Raising confidence alone
cannot recover a component outside the image, and no threshold was selected.

## Next controlled action

**With the plate stopped**, move the **same good PCB inward toward the hub** so
its entire outline, IC and terminal stay visible through a complete revolution.
Do not move the camera, change focus or substitute a bad specimen yet. Confirm
readiness, then repeat the known-good run before progressing to bad states.

After framing is verified, compare the remaining visible-part misses and false
fail labels to the reference on matched source images. Agree inspection-site
and box annotations with the owner before any ROI/training/threshold decision.

## Evidence

All local experiment data is under `../IC_TERMINAL_TRAIN/`:

- `good-part-20261115-live/`: first live-run results and full native detections.
- `good-part-20261115-probe/`: matching frame 20/21 raw buffers, `source-20.png`,
  `annotated-20.png`, `decoded.npz`, `same-frame-reference.json` and decoding script.
- `good-part-20261115-rotation/run_recorded.py`: exact one-off FFmpeg/wrapper recipe.
- `good-part-20261115-rotation/capture/`:
  - **`annotated-viewing.mp4`**, original `annotated.mkv`, archive metadata;
  - **`framing-edge-0.png`** (bottom cut-off) and **`framing-edge-180.png`**
    (top cut-off), plus other framing samples;
  - `class-summary.json`, `framing-screen.json`, `specimen.json`, command/logs,
    native detections, encoder records and board health.

Board evidence: `/media/nvme/ic-terminal-live-good-part-20261115-live`,
`/media/nvme/ic-terminal-good-part-20261115`, and
`/media/nvme/ic-terminal-live-capture` (the recorded recipe used local output
basename `capture`). No pinned release contents were modified.
