---
type: architecture
title: Inference Backends and Decode Contracts
description: The six-head graph, the two decode paths, quantisation, and what is measured.
status: active
tags: [inference, onnx, decode, quantisation, throughput]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/tools/export_modalix_heads.py
  - repo://portable-vision/tools/build_ic_terminal_configs.py
  - board://genericboxdecode/src/boxdecode.cpp
---

# Inference backends and decode contracts

## The six-head graph

The compiler consumes a six-output ONNX; the board runs the resulting ELF.
Two variants have been measured, with only the raw variant fast so far:

| Variant | Outputs | Decode work |
|---|---|---|
| Raw six-head | `bbox_0..2` `[1,64,H,W]`, `class_prob_0..2` `[1,4,H,W]` pre-sigmoid | DFL on the board decoder |
| Decoded six-head | `bbox_0..2` `[1,4,H,W]` xywh, `class_*` post-sigmoid | none, pass-through |

At 1280 the raw heads run at **27–30 fps**. Earlier custom in-graph `Sigmoid`/
decode variants measured **0.6–0.78 fps**, motivating DFL in the board decoder.
Those numbers are not a benchmark of the complete vendor YOLO11 surgery path:
that path also rewrites the attention block and has not yet been compiled here.

## What the board decoder supports

`genericboxdecode` dispatches on `decode_type` in `boxdecoder.json`:

| `decode_type` | Box tensors | Softmax | Sigmoid on class tensor |
|---|---|---|---|
| `yolo` | pre-decoded xywh, 4 channels | no | no (classes must already be probabilities) |
| `yolov8` / `yolov9` / `yolov10` | DFL, 64 channels | yes, 16 bins per side | yes |

The decoder pairs tensors into heads as `(0,1)`, `(2,3)`, `(4,5)`. The vendor's
own working pipeline (its `Wood_simaaisrc` sample) lists **three box tensors and
then three class tensors**. Grouping is correct for this compiled graph, as
confirmed by the direct hardware/Python comparison. That sample happens to use
full-frame tiles (`input_* == slice_*`); this is **not a general requirement**.
Our 64-channel box heads have smaller compiler tiles, so forcing equality
corrupts decoding.

## Geometry the compiler actually produced

For the yolo11n 1280 MPK:

| Tensor | Shape |
|---|---|
| `bbox_0` | `[1,160,160,64]` |
| `bbox_1` | `[1,80,80,64]` |
| `bbox_2` | `[1,40,40,64]` |
| `class_0` | `[1,160,160,4]` |
| `class_1` | `[1,80,80,4]` |
| `class_2` | `[1,40,40,4]` |
| MLA output buffer | 2,688,000 bytes |

`build_ic_terminal_configs.py` now reads these from the compiler's
`tuple_get_item_N_detessellation_transform` plugins and refuses to write a config
pair whose unpack and decode nodes disagree.

## Dequantisation

Each tensor carries `(scale, zero_point)` from the compiler. The decoder
computes `(q - zero_point) / scale`.

| Tensor | scale | zero point | Representable range |
|---|---|---|---|
| `bbox_0` | 12.749 | −55 | −5.73 … 14.28 (signed INT8 DFL logits) |
| `bbox_1` | 13.362 | −47 | |
| `bbox_2` | 12.100 | 3 | |
| `class_0` | 17.161 | 88 | −12.6 … 2.27 → max sigmoid 0.906 |
| `class_1` | 26.358 | 127 | −9.67 … 0.0 → **max sigmoid 0.5** |
| `class_2` | 34.396 | 127 | −7.41 … 0.0 → **max sigmoid 0.5** |

The two coarse class levels cannot express a positive logit at all. Whatever the
model wants to say above 0.5 is clipped. This is a property of the calibration,
not of the decoder, and it is the reason a board score can never be compared
naively with a host score above 0.5.

## Measured throughput

| Build | Input | Decode | FPS |
|---|---|---|---|
| YOLOv8m, canonical | 640 | in-graph | 27 |
| YOLO11n, raw heads | 1280 | board DFL | 27–30 |
| YOLO11n, in-graph sigmoid/decode | 1280 | in-graph | 0.6–0.78 |

The 0.6 fps figure is suspiciously bad for a 4× pixel increase and has not been
re-investigated. If the canonical pre-decoded path can be made to hold ~25 fps at
1280, it removes the DFL question entirely.

## Vendor contract comparison (2026-11-14)

Official [boxdecode documentation](https://docs.sima.ai/pages/api_reference/gst-plugins/simaaiboxdecode.html)
explicitly supports **both** paths: `yolo` retrieves already-decoded coordinates;
`yolov8`/`yolov9`/`yolov10` decode cut-model outputs. Thus raw DFL is not inherently
unsupported. Its tensor layout, quantization and interpretation must still match
the compiled model and installed decoder. Do not infer incompatibility solely
from which path the vendor's example chooses.

The installed `model_to_pipeline/surgeons/surgeon_yolo11.py` (Admin PC,
`C:/Users/Admin/Sima_Projects/tool-model-to-pipeline/`) additionally rewrites
YOLO11 attention into 4D-friendly operations before producing three 4-channel
box tensors and three sigmoided class tensors. Public
[graph-surgery guidance](https://docs.sima.ai/pages/model-sdk/model_graph_surgery.html)
requires original/modified numerical comparison and clearing stale inferred
shape metadata.

Running that **unmodified** surgeon on our original 1280 FP32 export in an
isolated portable compiler-container directory fails before final-head surgery:
its attention rewrite leaves a consumer before its producer, and sorting alone
then exposes stale tensor-shape metadata. All referenced tensor producers exist.
An in-memory diagnostic that orders nodes and clears `value_info` passes full
ONNX validation; the intermediate attention rewrite matches the original on
`sample-300.png` within 0.000274658 model pixels and 7.152557e-7 class probability.
This is **not** the full six-output graph, quantized comparison, compiled-model
validation, or proof of the existing board detection defect's root cause.
An isolated compatibility adapter is proposed, not applied; the installed vendor
source and board releases were untouched.

SiMa's [accuracy guide](https://docs.sima.ai/pages/palette/get_accuracy.html)
also documents accelerator mode: send the same host-preprocessed input to the
compiled model, then apply host postprocessing to returned outputs. That could
separate compilation/quantization errors from camera/preprocessing/boxdecode
errors without relying on matching two live scenes. The linked sample is under
SDK1.7.0 while this compiler is 2.0.0, so verify the installed API before using it;
the public sample itself has not been run here.

A portable-specific direct Python diagnostic was subsequently added using the
fixed station's execution **pattern**, but retaining portable weights/classes,
1280 geometry and 64-channel DFL tensors. Its NumPy decoder agrees with original
FP32 output on three identical prepared inputs within 0.000244141 model pixels
and 1.371664e-7 probability. The installed native route aborts at
`SimaQuantTess.configure({})`, before model-runner construction or ELF execution.
That failed route returned no predictions.

The alternative software route now succeeds: installed `ev_transforms.transforms`
functions perform quantization/tiling and output unpacking/untiling/dequantization
using the original MPK's transform parameters and byte sizes; only
`SimaModelRunner` executes the network on hardware. SDK tile/inverse roundtrips
are checked before execution. The six returned heads retain portable 64-channel
DFL boxes and four class logits, not the fixed station's four-distance contract.

On the three identical prepared inputs, FP32 versus compiled/Python box counts
at confidence 0.05 were **5/9**, **3/3**, **2/2**. All eight FP32 reference boxes
with score ≥0.3 had same-class hardware counterparts at score ≥0.05, with IoU
0.855–0.965. Scores differ and the recent image has extra low-score predictions;
this is useful reference agreement, not labelled precision/recall or release
acceptance. The compiled model is capable of meaningful predictions; the live
pipeline's input/output handling needs isolation before blaming training.

Software input/output conversion alone took roughly 203–217 ms per input;
MLA calls took 3.7–4.2 ms. This is a correctness diagnostic, not a ≥25 fps path.
Compiler input tiles `[1280,64,3]` differ textually from live preprocessing tile
fields (128×32) and the failed parser-generated native quantizer (64×1).
Those static differences were subsequently tested against captured buffers,
confirming both layout faults and the r33 repair described below. No repair was
based solely on textual differences. See
[current status](../operations/status.md#portable-python-api-diagnostic--software-preparation-succeeds)
and [analysis commands](../reference/analysis-scripts.md#portable-identical-input-tensor-diagnostic).

Evidence: `../IC_TERMINAL_TRAIN/vendor-reference-20261114/{surgery.log,ordering-diagnosis.json}`.

## Layout faults isolated and repaired in r33

Two independent byte-layout faults, not a requirement to retrain/re-export, were
confirmed by captured same-frame buffers:

1. Preprocessing used HWC tiles `[32,128,3]`; compiler input expects
   `[1280,64,3]`. Untiling with the native settings reproduces the companion
   RGB tensor exactly; compiler interpretation misplaces about 79.3% of bytes.
   Directly replaying the captured packed input reproduces captured MLA output
   **byte-for-byte** on two frames. Repacking only those same quantized pixels
   to compiler layout restores meaningful detections.
2. Native output decoding used full logical frame heights as tile heights.
   Actual box-head tiles are `[2,160,64]`, `[1,80,64]`, `[1,40,64]`.
   Input-only r32 restored scores but still produced wrong box sizes. r33 fixes
   both MLA source caps and boxdecoder `slice_*` to the compiler tiles, while
   leaving logical `input_*` unchanged. On two captured r33 frames, eight native
   detections match independent software decoding within **0.473 px** and
   **1.50e-8 score** (normal integer box rounding).

`build_ic_terminal_configs.py` now derives both input and output tiling, checks
logical and tile consistency separately, and refuses missing tile metadata in
its CLI. It no longer inherits 128×32 preprocessing tiles or equates all
`slice_*` with `input_*`.

r33 delivered 1,266 complete annotated 1280×720 RTP frames to the workstation
at about **29.5 fps** in a short run. A live receiver decoded 900 frames and
exited cleanly. No negative heights were recorded among 6,431 diagnostic boxes.
This is a working debug stream at threshold 0.05—not labelled precision/recall,
production-threshold acceptance or long-run reliability qualification. Model,
classes, calibration, confidence threshold and ELF are unchanged. See
[current status](../operations/status.md) for evidence, the remaining limitations,
and the warning about legacy candidate configs.
