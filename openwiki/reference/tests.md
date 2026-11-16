---
type: reference
title: Tests
description: What this workspace tests, how to run them, and what they do not prove.
status: active
tags: [tests, verification]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T16:00:00Z
sources:
  - repo://portable-vision/tests/test_ic_terminal_configs.py
  - repo://portable-vision/tests/test_ic_terminal_geometry_guard.py
  - repo://portable-vision/tests/test_ic_terminal_dataset.py
  - repo://portable-vision/tests/test_portable_timestamps.py
---

# Tests

## Summit recovery tests (no board)

```sh
python3 -m pytest tests/test_dashboard_server.py tests/test_dashboard_recovery.py tests/test_summit_supervisor.py -q
node --test tests/dashboard.test.cjs tests/recovery.test.cjs
python3 -B deployment/portable-v1/verify.py
```

These use pytest/standard-library loopback fixtures and Node's built-in test
runner, not the model dependencies. They cover bounded HTTP/image failures,
malformed/stale packets, cancellation/watchdog/source-switch behavior, receiver
session/port discovery, cross-session frame rejection, preview recovery,
concurrency/serving limits, supervision locks/retry/safety policy and temporary
credential permissions/cleanup. They do not constitute browser-rendered, Mac-wheel,
physical-network, decoder, SSH, actual renewal or soak qualification. See
[onsite acceptance](../operations/summit-runbook.md).

## Running

The workspace ships **no virtual environment**. The system interpreter does not
have `cv2`, `fastapi`, `onnx` or `paramiko`, so three of the four suites fail at
collection if you use it.

Use an environment that has them. On this machine that is the sibling
repository's:

```powershell
..\demeter-systems-mv-full\.venv\Scripts\python.exe -m pytest tests -q
```

Anything else needs `pip install -r requirements-dev.txt` first.

Current state: **293 tests**. 30 IC/terminal tooling tests (14 dataset, 10 configs,
6 geometry guard), 51 portable timestamp tests, 20 portable tensor-diagnostic
tests, 18 operator/evaluation tests, 16 live-benchmark tests, 11 native-sampling
and 10 matched false-pass/resume tests, 12 offline PCB pose/site helper tests,
16 draft-label pack tests, 17 Canny crop/review-integrity tests, 14 hybrid crop
helper tests, 9 original/crop FP32 review-helper tests, 11 cropped training-box
helper tests, 10 next-batch selection/leakage tests, 14 train-only augmentation
helper tests, plus 34 copied SDK tests covering the
board worker and camera eligibility.

## What each suite covers

### Train-only augmentation preparation

`test_augmented_training_pool.py` covers reproducibility of all five variants,
label bounds/classes, exact base/180° mapping, source immutability, target-clipping
refusal, parent/variant-specific seed, use of changed reviewed boxes, review/source/
class/duplicate guards and binding verbal approval to exact first-draft bytes.
14 helpers plus actual independent regeneration of all 450 PNGs/labels; no claim
of new independent examples or trained accuracy. [Recipe/readiness](../guides/prepared-training-pool-20261115.md).

### Historical next-batch selection

`test_next_training_batch.py` verifies exact scale-1 letterbox removal, bad padding/
label refusal, unpadded label coordinates, circular-angle distance/deterministic
farthest selection, insufficient-source refusal and exclusion of historical
validation/test via split and source/model hash guards. Ten tests do not establish
independent-part validation; older train/test recordings overlap. New batch's
browser smoke has 14 checks; synthetic class-tamper test was corrected to actually
change the first present-class ID, not reassign it to its existing value.
[Batch02 and augmentation ordering](../guides/training-batch02-and-augmentation.md).

### Cropped training-box proposal helpers

`test_cropped_training_review.py` covers four-corner projection/AABB, refusal to
silently clip outside targets, degenerate/invalid boxes, YOLO normalization,
missing/present fixed class mapping, resetting prior geometry approval for the
new training-box task, source immutability and review rendering. Eleven helper
tests do not approve labels or authorize training. Separate isolated browser
smoke has 14 checks for resizing/approval reset/identity-class import/export.
[Manual review/training gate](../guides/cropped-training-box-review.md).

### Original-versus-crop FP32 review helpers

`test_portable_crop_detection_review.py` covers RGB/NCHW normalization and centred
padding, exact inverse rounded x/y scales, fixed confidence/class-aware NMS,
padding-only/empty boxes, four-corner transforms, immutable source rendering and
outside-ROI display. Nine CPU helper tests, not hardware accuracy qualification.
Actual evidence includes 84 ONNX runs and exact raw-output re-decoding; [owner
review report](../operations/crop-detection-review-20261115.md).

### Offline hybrid crop helpers

`test_portable_hybrid_crop.py` covers neutral bracket rejection, four rotations,
colour-only clipping versus local-edge recovery of green overhang, grouped gaps,
multiple-substrate ambiguity, insufficient local edge support, truncated candidate
refusal, source immutability, separate synthetic board-removal controls, bounds
and unknown-versus-wrong summary accounting. These 14 tests do not imply generic
PCB recognition, held-out accuracy or board performance. [Development results](../operations/hybrid-crop-experiment-20261115.md).

### Offline Canny crop experiment

`test_portable_canny_crop.py` covers synthetic rotated crops, pixel-coordinate
mapping, no forced choice on blank/ambiguous scenes, nested-contour deduplication,
polygon coverage arithmetic and ablation bounds. Review tests reject changed
source pixels, states/classes/contracts/splits, duplicate or unreviewed records
and invalid geometry. These 17 helper tests do not establish real PCB identity:
[the actual experiment](../operations/canny-crop-experiment-20261115.md) still
selects fixed hardware in 14/42 images at its best development setting.

### Draft PCB/site annotation pack

`test_portable_site_labels.py` covers invalid/wrong-winding/out-of-bounds quads,
pending/non-training defaults, fixed class IDs and confirmed presence mapping,
manual proposals for unknown poses, coordinate/direction projection, LabelMe
paths/flags, exact image copy, computed coverage, offline HTML, refusal to reuse
output, tampered source hashes and unsafe/duplicate IDs. These 16 tests do not
approve labels. Separate isolated Edge smoke verifies review/navigation/dragging,
source-identity import checks and resume (13 checks); actual pack stays unreviewed.
[Review workflow and known limits](../guides/pcb-site-label-review.md).

### Offline PCB pose/site experiment

`test_portable_site_experiment.py` covers blank/ambiguous substrate, missing
header cues, four synthetic rotations, source/canonical site round trips,
class-specific site gating, unknown evidence preservation, and PCB-restricted
reference features. No accuracy or live performance is inferred from these
synthetic tests. [The real-image experiment](../operations/site-pose-experiment-20261115.md)
finds wrong orientation and poor coverage; neither tested method is deployed.

### Native sampling and matched false-pass diagnostic

`test_native_buffer_sampling.py` covers default 20/21 selection, periodic
prediction-independent IDs, bounded arguments and exact per-stage completeness.
`test_live_false_pass_parity.py` covers letterbox/source mapping, empty/disjoint
IoU, recovering the exact anchor despite repeated quantized scores, same-class
matching, explicit exact-array resume, refusal of changed archives and rebuilding
an incomplete archive. These are CPU helper tests; the 40-frame native evidence
and limitations are in [the diagnosis](../operations/false-ic-pass-diagnosis-20261115.md).

### `test_portable_live_benchmark.py`

Read-only benchmark calculations: Linux process-stat parsing with spaces in the
process name, real AT_CLKTCK extraction/refusal to guess, aggregate CPU guest
accounting, partial JSONL edges, exact session/stream/frame cohort matching,
missing/duplicate evidence rejection, weak native score retention, duplicate
class candidates, viewer counter interpretation, percentile units and summary
CPU/RSS/frame-ID-gap math (including the leading boundary). No board/display
access. Actual two-minute hardware result and limits are in the
[baseline report](../operations/benchmark-baseline-20261115.md).

### `test_portable_operator_tools.py`

Host-side live-view and labelled-scoring checks: finite durations, safe board
paths and shell quoting, actual UDP mode rather than `--diagnostic`, RTP/receiver
arguments, exact PID identity checks before stop, preflight refusal on a busy
board or reused board output without starting a receiver, annotation parsing,
IoU, confidence-ordered one-to-one matching, duplicates, class errors, missing
predictions and model-to-source coordinate clipping. These use mocks/pure Python;
board/FFplay/FFmpeg lifecycle and actual compiled inference are separate hardware
trials in the [evaluation report](../operations/r33-labelled-evaluation-20261114.md).
No test fabricates live good/bad ground truth.

### `test_ic_terminal_configs.py`

The config generator: input scale, zero point and MLA buffer size from the MPK;
dequantisation kept in tensor order; rejection of a missing tensor; depths taken
from the compiler rather than `--box-channels`; rejection when the compiler grids
contradict `--model-size`; the legacy inspection fallback when an MPK publishes
no shapes. New regressions cover compiler input/output tiles, actual CLI emission
of distinct logical and physical geometry, and refusing missing tile metadata
before writing files. The CLI no longer inherits template tile values.

### `test_ic_terminal_geometry_guard.py`

The guard that exists because of the shipped defect, exercised against the
**preserved bad config pair** in `tests/fixtures/unpack_640_bug/` — the original
`candidate/etc-before-fix` files, kept because they are the only artefact that
still contains the bug.

It asserts:

- the bad pair is rejected, and the message names the field that disagrees
- the numbers that went unchecked really are different: the unpack node declared
  **67,200** bytes of tensor geometry, the decoder declared **2,284,800**, and the
  MLA buffer is **2,688,000** (compiler padding)
- a depth-only mismatch is caught, not just a grid mismatch
- a matching pair passes
- matching physical tile geometry may differ from logical tensor geometry;
  mismatched tile configurations are rejected rather than forcing `slice_*`
  to equal `input_*`

### `test_ic_terminal_dataset.py`

Both transforms' box mappings, letterbox padding colour, crop clamping, rotation
axis recovery, guard band, per-split and per-sector coverage, class order. Geometry
only; no accuracy.

### `test_portable_timestamps.py`

The portable launcher's source-to-running-time clock conversion, bounded
timestamp cache, and the actual worker probe
callback with fake GStreamer buffers/pads (compiled from the callback's AST;
no `gi` or board plugins needed). Covers:

- fixed-offset clock conversion removes board uptime while preserving frame
  intervals despite arrival jitter; invalid clocks, sentinel values, epoch resets,
  overflow and unavailable pipeline clocks never invent usable time
- the actual encoder probe receives stream-relative PTS, not absolute board time
- 10,000 completed frames leave no retained labels, both directly and through
  the worker's encoder probe
- same frame numbers from different streams keep their own time labels
- capacity limits and oldest-entry eviction when frames never complete
- native encoder PTS is preserved, while its cache entry is still consumed
- boxdecode cannot recreate consumed entries or overwrite source times
- invalid/sentinel timestamps and identities, duplicate/conflicting source times
- zero time, out-of-order completion, concurrent access, and consume-once lookup
- missing/evicted labels never fabricate time; non-diagnostic missing time drops
  output

These tests exercise Python logic, not the native hardware's metadata handling.
The board evidence is in [current status](../operations/status.md).

### `test_portable_tensor_parity.py`

Portable-only CPU diagnostic contracts: known 16-bin DFL distributions recover
expected box positions/sizes; NHWC hardware and NCHW reference layouts decode
identically; four class IDs retain their order; the fixed station's 4-channel
box tensors are rejected. Tests cover malformed geometry, integer/nonfinite
outputs and incomplete tensors, plus RGB /255 conversion, centered padding 114,
input fingerprints and refusal to overwrite an existing preparation directory.

Software-transform tests additionally verify compiler-derived input tiles,
physical class-head padding/unpack boundaries, input/output byte counts, output
head ordering, and rejection of unsupported rounding, duplicate plugins and
multiple MLA segments. Fake SDK functions test dispatch and boundaries, not the
proprietary tessellation algorithm. On-board software preflight uses the actual
installed SDK and checks tile/inverse roundtrips before model execution.

These **20 tests do not execute SiMa libraries**. The native quantizer diagnostic
aborted before model execution; the alternative software-transform diagnostic
subsequently returned predictions successfully for three inputs. See
[current status](../operations/status.md) and the
[diagnostic commands](analysis-scripts.md#portable-identical-input-tensor-diagnostic).

### Copied SDK suites

`test_vision_sdk.py` and `test_sdk_triggered_camera.py` belong to the fixed
station's SDK. They are here because this workspace ships a copy of that SDK.

## What none of this proves

- That the board decodes correctly. That needs a pipeline run against the camera.
- That the complete portable launcher and native GStreamer graph work. Timestamp
  logic now has regression tests, but stage/run lifecycle and hardware behaviour
  are not covered by these unit tests.
- Accuracy, throughput, or full qualification. `r29` has passed short board timing
  trials, but detection accuracy, real receiver playback, recovery, optical latency,
  and an all-day soak remain unverified
  (see [current status](../operations/status.md)).
