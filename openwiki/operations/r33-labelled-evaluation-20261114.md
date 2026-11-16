# r33: operator-flow trials and historical labelled evaluation

**Date:** 2026-11-14. **Status:** repeatable live-view flow tested; historical
labelled evaluation complete; current-camera controlled physical trial blocked.
No model, threshold, calibration, pinned release or installed service changed.

**2026-11-15 follow-up:** the owner supplied a known-good part. The
[current-camera trial](good-part-trial-20261115.md) found clipping during rotation
and false fail labels. The blocked status above describes this report's earlier
checkpoint, not the latest availability of an operator.

## 1. Repeatable live viewing

New host tool: [`tools/portable_live_view.py`](../../tools/portable_live_view.py).
[Start/stop instructions and physical trial protocol](../guides/live-view-and-labelled-check.md).

| Trial | Observation |
|---|---|
| Headless, target 300 decoded frames | 300 decoded, FFmpeg exit 0; board 326 frames, launcher exit 0 |
| Windowed, 20-second limit | Board 536 frames, launcher exit 0; wrapper stopped its own player |
| Ctrl+C path, requested programmatically | Board 163 frames, launcher exit 0; own player stopped |
| Immediate restart, target 150 decoded frames | 150 decoded, FFmpeg exit 0; board 174 frames, launcher exit 0 |
| Final windowed run with player stats | Board 534 frames, launcher exit 0; FFplay clock advanced to ~18 seconds |
| Final tool revision, headless smoke test | 60 decoded, FFmpeg exit 0; board 82 frames, launcher exit 0 |

The final player reported **96 dropped frames**, while host ONNX evaluation was
running concurrently. The cause was not isolated. This demonstrates live viewer
operation, not a zero-drop desktop playback benchmark, latency qualification, or
human-observed visual sign-off. No claim of a manual Q-key test is made.

Each windowed player was terminated by its wrapper after the board stopped;
Windows returned 1 for that deliberate termination. This differs from a natural
FFmpeg exit 0. Existing finite-run EOS/drain limitations remain.

All trial `result.json` files report no wrapper errors. Logs and health evidence
are under `../IC_TERMINAL_TRAIN/live-view-20261114-{headless,window,interrupt,restart,final-window,final-headless}`.
Matching board outputs use `/media/nvme/ic-terminal-live-<basename>`.

## 2. Correct an important handover error

Earlier wiki pages said `Terminal_Fail` had no captured examples. The actual
v4-1280 training dataset and raw COCO annotations contradict that:

| Class | Train boxes | Validation boxes | Test boxes |
|---|---:|---:|---:|
| IC_Fail | 85 | 22 | 20 |
| Terminal_pass | 84 | 21 | 21 |
| IC_Pass | 42 | 10 | 11 |
| Terminal_Fail | **43** | **11** | **10** |

The training run's `args.yaml` points to v4-1280. Its data YAML, dataset report,
FP32 ONNX class metadata and COCO category names use the canonical class order.
The raw `instances_default.json` has 100 category-4 (`Terminal_Fail`) annotations;
the other exports describe the present and IC-absent runs. This is historical
label evidence, not confirmation of current specimens or broad defect coverage.

The dataset has `present`, `absent` and `terminal_absent` source runs. Do not
infer that it covers arbitrary solder-joint defects, different parts, or the
unrepresented good-IC/bad-terminal combination.

## 3. Evaluation method

- Used **all 31 existing test images**, 62 labelled boxes: 11 present, 10 absent,
  10 terminal-absent frames. No favourable subset was selected.
- Source-image fingerprints do not overlap train/validation. Test image names
  and stored-image hashes do not overlap compiler calibration records.
- These frames still share **three source captures with training**. They are not
  31 independent physical specimens or a newly captured production test.
- Compared original FP32 ONNX with the unchanged r33 compiled ELF, using identical
  RGB float input arrays. Compiled inference ran on the board, with the verified
  SDK software transforms and portable DFL decoder—not native camera replay.
- Checked image, label, prepared-input and output fingerprints, class contracts,
  and ELF/MPK compilation lineage. FP32 model fingerprint is in the JSON report.
- Locked diagnostic settings: confidence **0.05**, class-aware NMS IoU **0.45**,
  top 100, match IoU **0.50**. Matching is confidence-ordered, one-to-one and
  class-aware; duplicate/unmatched predictions count as false positives.
- Did not tune confidence, alter labels, relax matching criteria or retrain.
- Images are from the old camera setup. This does not establish live accuracy at
  today's camera position, nor test the entire native live preprocessing chain.

## 4. Results

These are fixed-threshold box metrics, **not mAP** or part-level inspection yield.

| Metric | Original FP32 | Compiled model + Python decode |
|---|---:|---:|
| Matched labels (true positives) | 57 / 62 | **57 / 62** |
| Unmatched predictions (false positives) | 20 | **16** |
| Unmatched labels (false negatives) | 5 | **5** |
| Precision | 74.0% | **78.1%** |
| Recall | 91.9% | **91.9%** |

Compiled per-class results:

| Class | Label count | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| IC_Fail | 20 | 15 | 16 | 5 | **48.4%** | **75.0%** |
| Terminal_pass | 21 | 21 | 0 | 0 | 100% | 100% |
| IC_Pass | 11 | 11 | 0 | 0 | 100% | 100% |
| Terminal_Fail | 10 | 10 | 0 | 0 | 100% | 100% |

Small, correlated historical samples do not justify a 100%-accuracy claim for
those classes. Fewer extra low-confidence compiled predictions do not establish
that quantisation generally improves accuracy.

### What the five misses actually mean

Both implementations fail the **same five** IC_Fail annotations at IoU 0.50:

- `absent_007948.png`
- `terminal_absent_001910.png`
- `terminal_absent_002001.png`
- `terminal_absent_002365.png`
- `terminal_absent_002456.png`

Each has a nearby prediction of the **correct class**. Compiled best same-class
box overlap is approximately **0.334–0.439**, with confidence **0.386–0.811**.
These are localisation mismatches under the fixed criterion, not five cases of
an entirely absent or wrong-class prediction. The mismatched prediction counts
as an FP and the annotation as an FN, as required by the scoring rule.

The other 11 unmatched compiled predictions have confidence no higher than
about **0.180**. Annotation completeness still matters when interpreting them
as physical false alarms. No threshold recommendation was selected from the test
results. Raising confidence alone cannot repair the five higher-score box
mismatches.

**Conclusion:** the main observed residual weakness is small failed-IC box
localisation plus weak extra predictions, already present in FP32—not a new
class flip or wholesale accuracy collapse introduced by the accelerator. Whether
the five small annotations or the predictions need correction requires review;
we have not established an annotation defect or a need to retrain.

## 5. Roadblocks and next controlled step

1. Nobody remotely operating this session can place or independently certify
   known good/bad specimens. A current-camera physical trial needs an operator.
2. Historical annotations do not validate today's camera framing, lighting or
   diverse defect types. Earlier claims of camera-transfer failure were measured
   before the layout repair and should not be treated as a controlled current
   diagnosis.
3. Review the five localisation crops with the owner before changing training or
   box criteria. Then perform the physical protocol in the linked runbook.
4. Desktop startup/frame drops and shutdown draining remain separate work; these
   trials are not production acceptance.

## 6. Reproduction and artifacts

Local evidence root: `../IC_TERMINAL_TRAIN/acceptance-20261114/`.
Board evidence: `/media/nvme/ic-terminal-acceptance-20261114/`.

```powershell
python tools/analysis/portable_labelled_eval.py prepare `
  --dataset ../IC_TERMINAL_TRAIN/dataset/v4-1280 `
  --compilation ../IC_TERMINAL_TRAIN/export-yolo11/compilation.json `
  --output <new-prepared-directory>
```

Transfer its `inputs/inputs.npz`, `inputs/inputs.json` and
`tools/analysis/portable_tensor_parity.py` to a new directory on the idle portable
board. Then, on the board:

```sh
timeout -k 5s 180s python3 -u portable_tensor_parity.py board \
  --transforms software --release /media/nvme/portable-release-20261114-r33 \
  --inputs inputs/inputs.npz --output <new-board-directory>
```

Copy board outputs back, then on the host with NumPy, OpenCV and ONNX Runtime:

```powershell
python tools/analysis/portable_labelled_eval.py evaluate `
  --prepared <prepared-directory> --board <downloaded-board-directory> `
  --original ../IC_TERMINAL_TRAIN/export-yolo11/yolo11n_fused.onnx `
  --compilation ../IC_TERMINAL_TRAIN/export-yolo11/compilation.json `
  --output <new-evaluation-directory>
```

Retained results:

- `prepared/selection.json`: label fingerprints, selection and overlap checks.
- `board/board.json`, `outputs.npz`, `software-transform-contract.json`: hardware
  outputs and execution contract; `board.exit` is 0.
- `evaluation-verified/evaluation.json`: complete per-frame predictions, ground
  truth, counts, confusion matrices and limitations.
- `evaluation-verified/*.jpg`: all 31 three-panel GT / FP32 / compiled comparisons.
- **`IC-localisation-review.jpg`**: enlarged five-failure comparison sheet.
- `localisation-review.json`: nearest same-class boxes, scores and overlaps for
  those failures. This was derived after scoring, not used to change the metric.

New host-side regression tests cover operator preflight refusal, shell quoting,
receiver parameters, PID identity checks, label parsing, one-to-one matching,
duplicates, wrong classes, missing detections and coordinate conversion. Native
execution and display lifecycle are separately documented hardware trials.
