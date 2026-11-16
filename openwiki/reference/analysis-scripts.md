---
type: reference
title: Analysis Scripts
description: The measurement and diagnostic scripts used on this station, and what each answers.
status: active
tags: [analysis, measurement, scripts]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T12:00:00Z
sources:
  - repo://portable-vision/tools/analysis/analysis_paths.py
---

# Analysis scripts

`tools/analysis/` holds the bench scripts written while building this station.
They are not part of any release; they measure things.

## Coordinate-only browser / pose-reuse experiment

`run_coordinate_browser.py --output <new-dir> --seconds 60 --headless --track-pose`
runs bounded, receiver-first coordinate5012/video5010 inspection using the separate
`browser-view-env` and pinned requirements-browser-view.txt. Host PyAV decodes raw
camera stream; browser overlays coordinates ONLY on exact NV12-hash-matched frames.
`--audit-inputs` stores occasional raw frames on host, not board snapshots.
`browser_coordinate_view.py` supplies bounded frame matching and RTP reassembly;
`coordinate_browser.html` clears stale/unmatched overlays. See
[measurements/limits](../operations/coordinate-browser-pose-20261116.md).
This is not a frozen production release, tracker accuracy acceptance or single-pane
30FPS detection overlay. Both fallback packages remain unchanged.

## C++ input fusion benchmark

`benchmark_fused_input.py` runs camera-free on the board against fingerprinted
prepared inputs:31-frame byte parity, all256-level/random patterns, poisoned buffer
margins, invalid-length guards, then interleaved warmed timing. Requires separate
portable C++ library/wrapper, not vendor edits. Live host `--cpp-library`/
`--cpp-kernel parallel8` selects the prototype explicitly and verifies library SHA.
No flag keeps accepted path. See [native build and results](../operations/cpp-input-fusion-20261116.md).
Latest frozen fallback is still the non-C++ coordinate/pose package.

## Tracker feature ROI benchmark

`benchmark_feature_roi.py --inputs <audited-json/nv12-dir> --output <NEW-dir>`
compares full-frame vs8px-halo ROI GFTT on the board, requiring exact ordered
feature arrays before interleaved timing. No camera graph or model inference.
Live `run_coordinate_browser.py --track-pose --feature-roi` selects the candidate;
reported15.95/s confirmation also used explicit C++8 input library. No flag keeps
full-frame seeding. [Results/commands](../operations/tracker-feature-roi-20261116.md).

## Data root

All of them resolve paths through `analysis_paths.py`:

```python
from analysis_paths import DATA        # d-meter/IC_TERMINAL_TRAIN by default
```

Override with the `IC_TERMINAL_TRAIN` environment variable. The labels file moved
into the workspace: `config/labels-ic-terminal.txt`.

## Camera and framing

| Script | Answers |
|---|---|
| `check_focus.py` | Full-frame Laplacian sharpness vs training frames |
| `check_focus2.py`, `check_focus3.py` | Same, for a specific adjustment, plus scene motion |
| `sample_focus.py` | Median sharpness over several seconds (use this, not one frame) |
| `measure_new_view.py` | Detections and the plate silhouette at a new camera position |
| `measure_colour.py` | Blue-PCB and green-terminal component sizes |
| `check_framing.py` | Whether the board touches a frame edge across sampled frames |
| `measure_sweep.py`, `measure_sweep3..6.py` | Orbit radius, swept extent, per-edge clearance |
| `trace_orbit.py` | Orbit traced from model detections instead of colour |

Framing numbers produced by these are recorded in
[guides/ic-terminal-model](../guides/ic-terminal-model.md).

## Decode and detection judging

| Script | Answers |
|---|---|
| `benchmark_live_session.py` | Read-only warm live-session window: encoder FPS/gaps, board/worker CPU/RSS, FFplay drop delta and exact admitted-frame class-presence cohort; never starts/stops a graph |
| `judge_r24.py` | Box counts, scores, class histogram from a run's `detections.jsonl` |
| `judge_signed.py` | Same, but interprets negative extents as signed (the correct reading) |
| `host_reference.py` | Host model detections on one live frame at several thresholds |
| `host_stats.py` | Host detection statistics over a spread of real frames |
| `rebuild_frames.py` | Parses a run's `packets.bin` records |

**Always use `judge_signed.py`.** Extents arrive as unsigned values; `4294967174`
is `-122`, and the unsigned reading makes a broken box look valid.

### Read-only live benchmark

`benchmark_live_session.py --session <existing .session.json> --case <owner-confirmed-state>
--expected <canonical-labels...> --seconds 120 --interval 2 --output <new-directory>`
observes a running non-diagnostic pipeline without interrupting the viewer.
It identity-checks the worker, reads health/proc counters over SFTP, and saves
original log byte windows. Performance/prediction scoring joins **encoder-admitted**
frame IDs to native detections by session/stream/frame; missing or duplicate
cohort evidence fails the observation rather than silently changing denominators.

CPU ticks are discovered from Linux auxv; worker CPU uses one-core units and
system CPU is normalized over all logical cores. Worker RSS excludes separate
native services. FFplay drop counters are not network loss or rendered FPS;
delivery gaps/valid PTS are not optical latency. Class presence is not box
precision/recall or part verdicts. No cold-start/power/MLA-utilization measurement.
Keep the specimen unchanged during each window and confirm swaps explicitly.

See [baseline results and specimen sequence](../operations/benchmark-baseline-20261115.md).

## Model and compiler inspection

| Script | Answers |
|---|---|
| `inspect_unpack.py` | The compiler's per-tensor output shapes and dequant from the MPK |
| `analyze_mpk.py` | Plugin chain summary of an MPK |
| `inspect_heads.py`, `inspect_heads11.py` | Head shapes of an exported ONNX |
| `inspect_export.py` | Export sanity: inputs, outputs, dtypes |
| `parity.py`, `parity_1280.py` | Host-vs-graph decode parity |
| `verify_decoded.py`, `verify_yolo11.py` | Decoded export verification |
| `range_check.py` | Activation range probe |
| `plugin_chain.py` | Ordered plugin list for a build |
| `write_provenance.py` | Provenance record for an export |
| `portable_tensor_parity.py` | Portable four-class, 1280 DFL diagnostic: identical prepared inputs, direct SiMa Python execution, host/reference comparison |
| `portable_labelled_eval.py` | Prepare the complete historical test split with hash/calibration checks; score FP32 and actual compiled outputs against labels at fixed settings |
| `inspect_native_buffers.py` | Board-only bounded capture of decoder/preprocessor/MLA/boxdecode buffers; default 20/21 or explicit periodic IDs with per-buffer caps/hashes |
| `live_false_pass_parity.py` | Decode exact captured tensors and compare native/Python/FP32 boxes plus same-anchor scores; explicit verified resume, no neural rerun |
| `portable_site_experiment.py` | Offline original-image PCB/site feasibility: blue silhouette/header direction or PCB-restricted ORB/RANSAC; provisional site polygons and counterfactual gating, not production filtering |
| `prepare_portable_site_labels.py` | Hash-checked source-copy/draft-label/review pack builder; explicit proposal provenance, manual overrides, LabelMe, comparison sheets and local editable HTML; never training ground truth |
| `portable_canny_crop.py` | Validates human review, runs full-frame Canny/contour crop ablations without label-based selection, then measures reviewed-target retention; saves edges and original-colour crops |
| `portable_hybrid_crop.py` | Offline colour-gated Canny, grouped-colour control and colour-seed/local-edge crop comparison; source-only selection, reviewed-target containment and explicit synthetic no-board controls |
| `portable_crop_detection_review.py` | Pinned FP32 original-versus-native-crop inference, fixed settings, full-frame and equal-ROI annotated comparison/gallery; all raw predictions/coordinate transforms retained, no site gating |
| `prepare_cropped_training_review.py` | Projects reviewed sites into verified native crops, seeds axis-aligned four-class boxes, creates editable review pack/provisional YOLO preview; no training YAML/split/enable |
| `prepare_next_training_batch.py` | Historical TRAIN-only class/angle-balanced crop review batch, exact unpadding/hash guards, frozen crop, quarantine unknowns, transfer labels as pending; no held-out images or training |
| `prepare_augmented_training_pool.py` | Consolidates reviewed native crop boxes, assigns all parents TRAIN, generates fixed reproducible five-view preprocessing/augmentation plus labels/gallery/lineage; no validation/YAML/training invented |
| `compare_native_buffers.py` | Controlled replay of captured model-input bytes, then repack only the same pixels to compiler layout |

### Portable identical-input tensor diagnostic

`portable_tensor_parity.py` takes explicit paths rather than the legacy `DATA`
default. It is **not** part of any board release and does not use the fixed
station's model, classes, ROI, 640 grids or four-distance box decoder. Its NumPy
decoder handles three grouped 64-channel DFL heads (16 bins per side), four class
logits, and strides 8/16/32 at the portable input size. It rejects wrong shapes,
nonfinite outputs and guessed layouts.

```powershell
python tools/analysis/portable_tensor_parity.py prepare `
  --images <image1.png> <image2.png> --output <new-input-directory>
```

Transfer `inputs.npz`, `inputs.json` and the script to a new diagnostic directory
on the **portable** board. Ensure no camera feed or inference run is active.
Run with an external timeout; native SDK failures can abort the process rather
than raise Python exceptions:

```sh
timeout -k 5s 100s python3 -u portable_tensor_parity.py board --transforms software \
  --release /media/nvme/portable-release-20261114-r31 \
  --inputs inputs.npz --output <new-board-output-directory>
```

The board command verifies release asset/input fingerprints and saves six float
heads. **Software transforms are now the default**: input quantization/tiling and
output unpacking/untiling/dequantization call installed
`ev_transforms.transforms` functions with the pinned MPK's parameters. Only
`SimaModelRunner` performs neural execution on the accelerator. The code follows
plugin dependencies and physical byte sizes, including padded class heads;
it does not replace compiler tiles with guessed defaults or the SDK parser's
native-CVU configuration. It verifies software tile/inverse roundtrips on the
input and synthetic output heads before model execution.

`software-transform-contract.json` and stage markers record that route. Native
`--transforms native` is retained **for explicit reproduction only**, not as a
fallback: its prior `SimaQuantTess.configure({})` invocation aborted with native
`stack smashing detected` before model construction. Do not repeat that abort
as part of routine tests. Neither mode alters releases or installed services.

The software mode returned all six heads for the three prepared examples on
2026-11-14. This is a diagnostic path, not a production-speed implementation:
software input and output transforms took roughly 203–217 ms together,
excluding full DFL/NMS work. After a successful board run, copy its output
directory back, then:

```powershell
python tools/analysis/portable_tensor_parity.py compare `
  --inputs <inputs.npz> --board <board-output-directory> `
  --raw <yolo11n_six.onnx> --original <yolo11n_fused.onnx> `
  --compilation <compilation.json> --output <new-report.json>
```

Comparison checks input/output hashes and ELF/MPK/raw-ONNX compilation lineage.
It uses the same four-class DFL decoder on reference and hardware tensors, plus
class-aware NMS at confidence 0.05 / IoU 0.7. These are diagnostic settings, not
accuracy or deployment acceptance. Each output directory/report must be new.

### Native buffer layout diagnosis

`inspect_native_buffers.py --release <pinned-release> --output <new-directory>`
runs that release's unchanged diagnostic graph; defaults remain eight seconds
and frames 20/21. Optional `--start-frame`, `--frame-step`, `--samples` (1..64)
and `--seconds` (<=120) select a deterministic periodic sample. Exact per-stage
completeness, duplicate counts, stream identity, caps and per-buffer hashes are
recorded, and pinned launcher hashes are checked. It is a board-only
measurement, not a qualification or throughput run. It reuses the pinned
launcher's graph/support imports; large synchronous buffer copies affect timing.
The board must be idle. Use an external process timeout and new evidence paths.

For controlled replay, place `compare_native_buffers.py` and
`portable_tensor_parity.py` together on the board, then run:

```sh
python3 compare_native_buffers.py --release <same-release> \
  --capture <captured-directory> --output <new-result-directory>
```

This validates the two-buffer preprocessing contract, tests tile reconstruction
against companion RGB, replays the captured bytes through the same ELF, then
changes only packing order. It preserves native output copies before subsequent
calls because native allocation reuse must not corrupt the comparison. Never
apply a guessed tile setting to live configs based only on a textual mismatch.
Actual r31 replay was byte-identical; r33 native/Python detection matching is
recorded in [current status](../operations/status.md).

### Historical labelled accuracy check

`portable_labelled_eval.py prepare` accepts `--dataset`, `--compilation` and a new
`--output`. It selects all test images, records label fingerprints, and refuses
train/validation source-hash or compiler-calibration name/hash overlap.
`evaluate` accepts `--prepared`, downloaded `--board`, `--original` FP32 ONNX,
`--compilation` and a new `--output`. It checks hashes, class metadata and compiled
artifact lineage. Fixed settings: conf 0.05, NMS IoU 0.45/top100, match IoU 0.50;
class-aware confidence-ordered one-to-one matching. It saves complete predictions,
per-class TP/FP/FN/precision/recall, a class-agnostic confusion matrix and all
three-panel comparison images. This is not mAP or a confidence-selection tool.

The 31-image historical result and reproduction are in
[the evaluation report](../operations/r33-labelled-evaluation-20261114.md).
Those images share three source runs with training; this does not qualify the
current camera. The actual compiled ELF runs through the verified software
transforms, not native camera-image replay.

For repeatable viewing, the host tool is `tools/portable_live_view.py`, outside
this analysis folder. See [operator instructions](../guides/live-view-and-labelled-check.md).

### Matched false-pass comparison

`live_false_pass_parity.py decode --release ... --capture ... --output <new>`
uses installed SDK software transforms only, verifies input tile inverse against
the companion RGB, and saves one input/head/source archive per frame. It does
not construct a model runner or execute inference. Explicit `--resume` recomputes
arrays and demands exact equality before reusing archives; incomplete writes
are rebuilt atomically, changed valid data is refused.

Host `compare --decoded ... --original ... --raw ... --compilation ... --output
<new>` checks hashes/lineage and raw/fused FP32 parity, compares captured native
and compiled-Python boxes, recovers exact anchors, and saves source/crop images
with FP32-versus-compiled class scores. FP32 receives dequantized captured input,
not the pre-quantization float image. It is not labelled accuracy or tuning.
[40-frame findings and limits](../operations/false-ic-pass-diagnosis-20261115.md).

### PCB pose/site feasibility

`portable_site_experiment.py --manifest <hashed-source-manifest> --comparison
<matched-comparison.json> --output <new-directory>` runs blue-substrate/header
localization. Optional `--reference <source.png>` runs PCB-restricted ORB/RANSAC
with reference target sites excluded from feature matching. It writes per-frame
pose/unknown evidence, source overlays, canonical contact sheets and a hypothetical
class/site-centre gate. No board access or model inference. Source images must
not have rendered detections; site polygons are provisional, not ground truth.
Unknown alignment is not a pass or a fail. [The 42-image experiment](../operations/site-pose-experiment-20261115.md)
found both methods unready for deployment; reduced box counts are not accuracy.

### Draft PCB/site annotation pack

`prepare_portable_site_labels.py --manifest <source-manifest> --proposals
<experiment-results> --overrides <manual-quads.json> --output <new-directory>`
validates hashes, IDs, confirmed state mapping and geometry before writing. Exact
PNG copies, ordered board corners/direction, provisional site polygons, LabelMe,
comparison sheets and local `review.html`; all pending and training-ineligible.
Template `portable_site_review.html` has no server/network dependency; drag/edit,
flag/review, export/resume JSON. Separate LabelMe and browser edits do not sync.
[Owner review and current pack](../guides/pcb-site-label-review.md).

### Canny crop experiment

`portable_canny_crop.py --review <owner-export.json> --original <original-draft.json>
--images <review-pack-directory> --close-iterations 0|1|2 --output <new>` validates
review/source identity, class/state/split and geometry before writing. Sweeps
three Canny threshold pairs; optional edge closing, contour geometry/texture
ranking, deduplication and ambiguity checks select a padded rectangle without
review labels. Saves original-colour native crops and aspect-preserving 512x256
previews, not an edge-image neural input. Semantic orientation is unresolved.
Reviewed polygons enter only post-selection coverage/overlays. Snapshot export
and source hash retained. No inference, board access or deployment.
[42-image results and unsafe bracket crops](../operations/canny-crop-experiment-20261115.md).

### PCB colour + local Canny crop

`portable_hybrid_crop.py --review <owner-export> --original <draft> --images
<review-pack-directory> --output <new>` validates review through the Canny helper.
Fixed three-arm comparison: colour-gated existing Canny candidates, grouped-colour
seeds alone, grouped seed + bounded local Canny extent. No reviewed polygons or
manual camera ROI enter positive selection; review labels only evaluate crops
and generate explicitly synthetic board-removed negative controls. Saves masks,
edges, per-arm RGB crops/overlays/contact sheets, source/crop transforms, coverage,
policy and input/helper/script hashes. No model inference or live access.
[42-image outcome and limits](../operations/hybrid-crop-experiment-20261115.md).

### Original-versus-crop FP32 detection review

`portable_crop_detection_review.py --review <owner-export> --original <draft>
--images <review-pack> --hybrid <hybrid-results-directory> --model <pinned-fused.onnx>
--output <new>` validates review, source/model hashes and exact regenerated crop
pixels/transforms, then runs both branches using RGB /255/centred 1280 padding.
Fixed confidence .05, NMS .45, top100; no site filter or verdict. Saves full
originals, projected original ROI zooms, cropped detections, side-by-side gallery,
raw outputs, tensor fingerprints and inverse crop/source geometry. CPU FP32 only,
not native board/quantized parity. All 42 cases included; owner review pending.
[Review folder and findings](../operations/crop-detection-review-20261115.md).

### Cropped training-box draft review

`prepare_cropped_training_review.py --review <site-export> --original <site-draft>
--images <source-pack> --hybrid <hybrid-results> --output <new>` validates exact
crop pixels/transforms and class mapping, projects reviewed site polygons and
proposes AABBs. Creates native PNGs, all-pending box JSON, classes, clearly marked
provisional YOLO labels, seven sheets and `review.html` from
`cropped_training_review.html`. Browser keeps rectangles axis-aligned, resets
approval after edits, and exports/resumes **cropped-training.review-export.json**
(not the older site-export filename). No autosave, inference, board access or
training/split/data.yaml. [Contract, review and coverage gate](../guides/cropped-training-box-review.md).

### Next class-balancing training review batch

`prepare_next_training_batch.py --dataset <v4-1280> --first-batch <crop-draft>
--output <new>` permits only historical present/absent TRAIN sources, guards
source/model holdout hashes and exact scale-1 padding, uses frozen hybrid crop,
quarantines unknowns, then selects 32/16 by circular farthest-angle sampling.
Historical boxes are transformed as pending proposals, not auto-approved. Writes
batch-specific rectangle review/export, audit, frozen policy/provenance and first-
batch unchanged verbal-confirmation record; no train YAML/split/augmentation/run.
[Coverage, approval and augmentation sequencing](../guides/training-batch02-and-augmentation.md).

### Reviewed train-only preprocessing and augmentation

`prepare_augmented_training_pool.py --first <batch1-pack> --second <batch2-pack>
--review <batch2-reviewed.json> --output <new>` validates first-batch hash-bound
conversation approval, batch02 named review and exact sources/classes/boxes.
All parents TRAIN before derivation. Base 1280 letterbox + four fixed variants,
transformed boxes and no silent clipping/drop; saves PNGs/YOLO, complete manifest,
all-parent previews and exact review provenance. RGB/255 remains loader work.
No train-as-val, held-out images, model or training call. Baked augmentation means
additional online transforms should be disabled; no data.yaml until validation
plan agreed. [Recipe, data and readiness](../guides/prepared-training-pool-20261115.md).

## Config variants used in experiments

| Script | Produces |
|---|---|
| `make_bgr.py` | Preproc with `output_img_type: BGR` (rejected) |
| `make_signed.py` | Negated class dequant scales (rejected) |
| `make_interleaved.py` | Interleaved box/class tensor order (inconclusive) |

These exist so the rejected experiments are reproducible, not so they can be
shipped. Their verdicts are in [experiments](../operations/experiments.md).

## Training and export

| Script | Role |
|---|---|
| `train_yolo11.py` | The YOLO11n training run |
| `export_yolo11.py`, `export_1280.py` | Six-head exports at 1280 |
| `run_spike.py` | Board spike harness |

`rewrite_paths.py` is the one-off transformer that made these scripts
data-root-relative; it is kept for reference and is safe to ignore.

## Warning

These scripts write into the data folder (`newset/`, `sweep*/`, `verify-*/`). They
overwrite. Read one before running it if the output matters.
