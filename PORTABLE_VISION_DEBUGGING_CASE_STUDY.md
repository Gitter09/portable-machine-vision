# When a vision model runs fast but sees the wrong image

## A plain-English case study of the portable SiMa Modalix pipeline

**Investigation record:** 14 November 2026, through test release **r33**.

**Outcome:** a working annotated 1280×720 video stream at approximately **29.5 frames per second**, without retraining or replacing the compiled model. The investigation isolated and repaired two data-layout faults and separate video-timing faults. This was a successful engineering diagnostic, **not a final certification of detection accuracy or production reliability**.

**Follow-up:** [operator-flow and labelled evaluation results](openwiki/operations/r33-labelled-evaluation-20261114.md) now document repeatable viewing and a 31-image historical test. That audit also corrected the earlier handover claim that `Terminal_Fail` had no captured examples.

## 1. What we were building

The portable station uses a camera to watch electronic parts on a rotating plate. A SiMa Modalix board runs a YOLO11-based model, produces object detections, draws boxes and labels, and sends video to a workstation.

The model has four class IDs, in this fixed order:

| ID | Label |
|---|---|
| 0 | `IC_Fail` |
| 1 | `Terminal_pass` |
| 2 | `IC_Pass` |
| 3 | `Terminal_Fail` |

The owner subsequently confirmed that this task's faults are **missing components only**: the legacy `Fail` names mean an absent IC or terminal, and `Pass` means present. Solder quality and electrical faults are not evaluated.

These are predictions, not independently verified inspection verdicts. In particular, retaining a class name in the model does not establish that it has sufficient training examples.

The intended processing chain was:

```text
Camera video
  → decode compressed video into pixels
  → resize, pad, convert and pack the image
  → run the compiled neural network on the accelerator
  → unpack predictions and turn them into boxes
  → draw boxes and labels
  → encode and transmit annotated video
  → receive and display it on the workstation
```

Every arrow is an interface. The model can be correct while an interface before or after it is wrong.

## 2. What made the initial problem misleading

The board was not simply refusing to run. It accepted the compiled model and processed images at roughly 27–30 fps. It also returned detections, some with plausible-looking sizes.

But its predictions disagreed badly with the floating-point model running on the workstation. One earlier scene comparison recorded about **0.4 boxes per frame on the board versus 5.8 on the host**. Some reported box heights were negative when interpreted as signed integers. Confidence was also much lower.

There were separate video problems: frames could flow through the graph while timing checks failed, and later a real receiver got no video even though local diagnostic runs passed.

This looked like a model-compatibility or quantisation problem. Both were reasonable hypotheses, but neither was a complete diagnosis.

The important distinction was:

> A model executing successfully proves that the accelerator can perform calculations. It does not prove that it received the intended image or that its output was interpreted correctly.

## 3. Why the working fixed station mattered

The nearby fixed inspection station already used a YOLO26s-based model successfully. However, the portable station was not just that system with different weights.

The fixed station's implementation used SiMa's Python execution API and explicit Python/NumPy calculations around the accelerator. The portable station used a more tightly integrated GStreamer pipeline with native preprocessing and box-decoding plugins.

There were also model-specific differences:

| Interface | Fixed-station implementation | Portable implementation |
|---|---|---|
| Model input | 640×640 | 1280×1280 |
| Box outputs at each location | Four distance values | 64 values requiring additional decoding |
| Box conversion | Inspectable Python/NumPy | Native SiMa box-decoding plugin |
| Results/display transport | Inspection results and a JPEG | Continuous H.264/RTP video |

The portable model's 64 box values represent **16 possible distance bins for each of four box edges**. Decoding them involves probabilities and a weighted average, often called Distribution Focal Loss, or **DFL**, decoding. Copying the fixed station's four-distance calculation would have been incorrect.

The useful thing to borrow was the **execution pattern**, not its screw model, labels, camera geometry or decoder. That gave us a way to inspect the portable model's numerical outputs without relying on the live box-decoding plugin.

## 4. First separate video timing from detection correctness

### Missing time labels

Native processing stages could drop a frame's standard GStreamer presentation timestamp while preserving a SiMa metadata timestamp and frame ID. The worker looked for the standard timestamp, so it counted no valid timed frames. The video packet timestamps also failed to advance correctly.

We carried the preserved source timestamp forward, matching it by **stream ID and frame ID**, and restored it at encoder output when necessary.

That bookkeeping needed its own safeguards:

- Remove a stored timestamp when the encoded frame consumes it.
- Limit waiting entries to 512 so dropped frames cannot cause unlimited memory growth.
- Never borrow another stream's same-numbered frame.
- Report missing or invalid time instead of inventing it.

The unbounded-map and cross-stream collision defects were reproduced in offline tests; they were not evidence that a live cross-stream collision had occurred.

### The wrong clock origin

Restoring a timestamp was necessary but insufficient. The preserved metadata time was approximately **time since the board started**, while the video sender scheduled output using **time since the pipeline started**.

A timestamp around 179,755 seconds looked far into the future to a newly started pipeline. The sender waited on its first packet.

A small synthetic test confirmed the distinction: the same UDP sink delivered promptly with pipeline-relative timestamps, but not with absolute board-time timestamps under normal clock scheduling.

We applied one fixed clock offset when the first decoded frame arrived:

```text
presentation time = first pipeline time
                  + source timestamp − first source timestamp
```

This preserves the spacing between source frames. It does not replace each timestamp with its arrival time, and it does not measure optical capture latency.

The local diagnostic sink had clock scheduling disabled, which is why earlier local tests missed this failure. The real network sink exposed it. The receiver address also needed correction to target the selected workstation.

**Lesson:** local frame flow, real video delivery and correct detections are three separate things to verify.

## 5. Build an independent route to the compiled model

The next objective was to answer a narrower question:

> If the compiled model receives an image in exactly its expected format, can it produce meaningful predictions?

We prepared three fixed images once and transferred the exact same numerical input arrays to the board. File fingerprints checked that the inputs had not changed in transit.

The first direct Python attempt failed before neural inference. SiMa's native quantisation/tessellation helper aborted during configuration with `stack smashing detected`. Stage-by-stage logging established that the model runner had not even been constructed. This failure therefore supplied **no predictions and no evidence about model accuracy**. Its exact native cause was not established.

We bypassed that helper using the SDK's installed software transforms:

```text
Identical prepared image
  → software number-format conversion and packing
  → accelerator executes the unchanged compiled model
  → software unpacking and number-format conversion
  → portable-specific Python DFL decoding
```

Here, **quantisation** means representing floating-point values in a lower-precision integer format. **Tessellation** means arranging those values into the memory blocks expected by the accelerator. We used the compiler's recorded settings rather than making up replacement scales or layouts.

The alternative route returned all six expected tensors: three box heads and three class heads. On the three images, floating-point versus compiled/Python detection counts were **5/9, 3/3 and 2/2** at the diagnostic confidence threshold of 0.05.

All eight reference boxes with confidence at least 0.3 had same-class hardware counterparts at that lower diagnostic threshold, with box overlap of **85–97%**. Scores differed and the recent image had extra weak predictions. This established useful reference agreement, not ground-truth accuracy.

The software path was slow—roughly 203–217 ms for input/output conversion alone—but it gave us an inspectable diagnostic baseline. It was not intended to replace the fast live pipeline.

### A vendor-conversion detour

We also consulted SiMa's public documentation and tested its installed YOLO11 graph-conversion tool. The tool failed on operation ordering and stale tensor-size metadata during its attention rewrite. An in-memory cleanup let that intermediate graph execute with a close match to the original model.

That was a real obstacle to testing an alternative export, but **it was not established as the cause of the deployed model's bad detections**. Once the unchanged compiled model produced meaningful results through the independent route, completing that converter repair was no longer necessary to reach the first working stream.

## 6. The hardest part: prove what the live pipeline was actually passing

Configuration files contained suspicious differences, but a textual difference is not proof of a bug. Different APIs can use different names or conventions for the same layout.

We therefore captured actual buffers for matching frame IDs at:

1. Video decoder output.
2. Image preprocessor output.
3. Neural-network accelerator output.
4. Box-decoder output.

These were bounded diagnostic captures. Copying large buffers can disturb timing, so they were not used to claim throughput.

The decisive advantage was that the preprocessor exposed both a **tiled model-input region** and a companion **untiled quantized RGB region** for the same frame. This allowed us to check packing against the actual pixels, not a second camera image taken at a different time.

### Fault one: the model was being given the wrong memory order

The live preprocessor packed the image using tiles of:

```text
height × width × channels = 32 × 128 × 3
```

The compiled model expected:

```text
height × width × channels = 1280 × 64 × 3
```

Think of cutting a photograph into strips and writing the strips into a file. If the reader assumes a different strip size, the file can contain every original pixel and still reconstruct the wrong picture.

That was the dangerous property here: **the byte count was correct**. The accelerator had enough data and could calculate normally. The pixels were simply in the wrong positions for its memory contract.

We established causality with a controlled experiment:

1. Untiling the captured input with the live settings reproduced the companion RGB tensor **exactly**, confirming what the preprocessor had done.
2. Interpreting those bytes in the compiler's expected order matched only about 20.7% of positions.
3. Replaying the captured packed input through the same compiled model reproduced the captured live accelerator output **byte-for-byte** on two frames.
4. Repacking only the same quantized pixels into compiler order restored meaningful predictions.

About **79.3% of byte positions differed between the two arrangements**. This was a reordering of values, not a claim that 79.3% of pixel values had been numerically altered.

No weights, source pixel values, calibration or thresholds changed in that experiment. It isolated an input-layout fault.

### Fault two: output tiles were mistaken for complete tensors

Correcting the input alone improved confidence and detections, but the live boxes still had the wrong sizes compared with Python decoding of the **same captured accelerator output**.

The remaining error was a distinction between:

- **Logical shape:** the complete grid of predictions.
- **Physical tile shape:** how pieces of that grid are arranged in memory.

For example, the first box head logically has 160 rows, 160 columns and 64 channels. The compiler stores it in tiles only **two rows high**.

| Box head | Logical shape, H×W×C | Compiler tile, H×W×C |
|---|---|---|
| 0 | 160×160×64 | 2×160×64 |
| 1 | 80×80×64 | 1×80×64 |
| 2 | 40×40×64 | 1×40×64 |

The configuration generator had set physical slice dimensions equal to logical dimensions. That assumption happened to fit an earlier model's template, but not this compiled model.

A geometry guard also enforced that equality. It was intended to prevent inconsistent configurations, but encoded the wrong invariant. **A passing safety check is only as good as the assumption it checks.**

The repair kept logical `input_*` dimensions separate from physical `slice_*` dimensions. Both the accelerator-output description and box decoder now receive the compiler's actual tile settings.

For two captured frames after the repair, all eight native detections agreed with independent Python decoding within **0.473 pixels** and **1.50×10⁻⁸ confidence**. The coordinate difference is consistent with rounding floating-point boxes to integer pixels.

This comparison was between two decoders of the same compiled outputs—not a claim that quantized and floating-point neural networks are numerically identical.

## 7. Make the fix reproducible, not just a successful experiment

The final native-layout test release was **r33**. The input-only experiment was r32, which helped distinguish the two faults instead of changing everything at once.

We corrected the configuration generator to:

- Read input and output tile shapes from the compiler's **MPK JSON descriptor** (its model-package report).
- Stop inheriting preprocessing tile sizes from an older template.
- Validate logical geometry and physical tile geometry separately.
- Refuse missing tile metadata rather than silently guessing.

Regression tests cover the distinction and verify the actual generated files. The complete repository test suite finished with **135 tests passing**.

Old releases were not edited in place. Their asset fingerprints remained meaningful, and test releases were staged separately. The fixed station, installed vendor implementation, model weights, class order and calibration were left unchanged.

**Operational caution:** the old candidate directories still contain stale layouts. Future deployments must regenerate configurations using the corrected tool or explicitly use the tested pinned release. Fixing the source generator does not retroactively fix old files.

## 8. What the final video test demonstrated

The repaired native pipeline sent annotated video to the workstation during a 45-second requested run.

| Measurement | Result |
|---|---|
| Video format | H.264, 1280×720 |
| Observed encoder rate | **29.52 fps** |
| Frames admitted / validly timed | 1,267 / 1,267 |
| Complete RTP video frames received | 1,266 |
| Live frames decoded by bounded FFmpeg receiver | **900, clean exit** |
| Raw RTP integrity check | Passed |
| Negative heights in detection log | **0 of 6,431 boxes** |
| Board process outcome | Clean exit, no reported errors |

The saved annotated viewing clip contains all 1,266 complete received frames. Its nominal 30-fps timestamps were regenerated for convenient viewing; it is **not original timing or latency evidence**.

This was a finite test, not an always-on service. A separate preview player was launched but required termination after the test; we do not claim a fully accepted operator display lifecycle.

## 9. What remained at the first-stream checkpoint

The integration faults were repaired in the tested release. That did not automatically settle the product's inspection quality. The linked follow-up records subsequent progress; at this checkpoint, further work included:

- Labelled precision/recall testing on known good and bad parts.
- Choosing an operating confidence threshold; **0.05 was a diagnostic setting**.
- Checking class quality at the current camera position. A later dataset audit found historical `Terminal_Fail` examples, contrary to the earlier handover; current-camera qualification still requires known specimens.
- Assessing quantisation-related confidence changes and weak extra detections.
- Providing repeatable live-view start/stop controls.
- Draining queued frames cleanly at shutdown. The admitted-frame and completed-video counts still differ at the tail.
- Testing recovery, latency and longer-duration operation.

Matching a reference model proves agreement, not that the reference itself is always correct.

## 10. The broader engineering lesson

The most challenging work was not changing a neural network. It was making the hidden interfaces observable and proving which one first diverged.

The investigation succeeded by separating:

- Transport and clock errors from prediction errors.
- Neural-network calculations from surrounding image and box processing.
- Logical tensor dimensions from physical memory layout.
- A plausible compatibility hypothesis from a controlled, reproducible cause.
- A working demonstration from production acceptance.

**The model could run, the buffers could have the right length, the video could be fast, and the boxes could look plausible—while the system was still reading the wrong arrangement of numbers.** Capturing the exact data, replaying it, and changing only its packing turned that ambiguity into a testable explanation and a targeted fix.

## Technical references and evidence

### Source files in this repository

- [Configuration generator and geometry guard](tools/build_ic_terminal_configs.py)
- [Portable tensor diagnostic and Python DFL decoder](tools/analysis/portable_tensor_parity.py)
- [Native buffer capture](tools/analysis/inspect_native_buffers.py)
- [Controlled captured-input replay](tools/analysis/compare_native_buffers.py)
- [Portable runtime and timestamp handling](runtime/launcher/worker.py)
- [Timestamp cache and clock conversion](runtime/launcher/support.py)
- Tests: [configs](tests/test_ic_terminal_configs.py), [geometry guard](tests/test_ic_terminal_geometry_guard.py), [tensor diagnostic](tests/test_portable_tensor_parity.py), [timestamps](tests/test_portable_timestamps.py).

### Investigation and vendor documentation

- [Current operational state and detailed measurements](openwiki/operations/status.md)
- [Inference and memory-layout contracts](openwiki/architecture/inference-backends.md)
- [Append-only project history](openwiki/CHANGELOG.md)
- SiMa: [graph surgery](https://docs.sima.ai/pages/model-sdk/model_graph_surgery.html), [box decoding](https://docs.sima.ai/pages/api_reference/gst-plugins/simaaiboxdecode.html), [model accuracy and accelerator mode](https://docs.sima.ai/pages/palette/get_accuracy.html).

### Local experiment artifacts

These are retained beside the repository under `../IC_TERMINAL_TRAIN/`, not bundled into this Markdown document:

- `tensor-parity-20261114/`: identical inputs, compiled outputs and reference comparisons.
- `native-buffers-20261114/` and `native-layout-control-20261114/`: original buffer captures and byte-identical replay/repacking evidence.
- `native-buffers-r33-20261114/parity-summary.json`: native/Python decoder comparison.
- `verify-r33-receiver/`: receiver logs, packet capture and video evidence.
- **Viewing clip:** `verify-r33-receiver/annotated-viewing-30fps.mp4`.
- **Annotated still:** `verify-r33-receiver/annotated-sample-600.png`.
