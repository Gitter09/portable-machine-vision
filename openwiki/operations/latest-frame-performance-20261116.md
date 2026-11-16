# Latest-frame profiling and optimization — 2026-11-16

Owner requested performance work now. Favourable-region gating remains OFF. The
frozen `fallbacks/latest-frame-20261116-v2` package is untouched and reverified.
This is a new opt-in source candidate, not a replacement of r33 or fallback.

## Measurements

All bounded sequential live trials use the same compiled ELF/MPK, camera graph,
latest-only appsink, crop policy, pad114 and diagnostic .05 confidence/.45 NMS.
Not frame-identical trials: rotating scene and unknown counts differ. Whole-run
rates include startup/shutdown. Stage means use located frames after first3 records.

| Trial | Inspections/sec | Located / unknown | Meaning |
|---|---:|---:|---|
| baseline40s |1.65|68/0|stage-instrumented prior path|
| opt1 |2.52|101/3|RGB conversion and score-gated DFL|
| opt2 |2.98|122/1|exact NumPy LUT physical transforms|
| opt3 |5.28|214/3|OpenCV LUT kernels instead of slow NumPy indexing|
| opt4 |7.80|308/12|selected-only crop work and sparse packed output decode|
| final60s |**7.99**|**479/9**|complete preflight,488 inspections/61.08s|

Final located inspections alone:479/61.08=7.84/s. Overall throughput improvement
~4.85x versus baseline; not an identical-scene randomized benchmark or optical
latency result. Final video forwarded1755 access units/61.08s (~28.73FPS including
startup/shutdown); received clip independently decoded600 frames at advertised
1280x720/30FPS, container19.934s. FFmpeg exit0; null mux reported two repeated-DTS
warnings. Not zero network loss, render-FPS, jitter or long-soak acceptance.

## Measured stage costs (ms, located frames)

| Stage | baseline mean | final mean |
|---|---:|---:|
| Crop selection |57.34|42.00|
| RGB conversion/normalization |100.25|~0 (fused into quantization)|
| Quantization/physical packing |109.22|10.59|
| MLA |3.75|4.12|
| Full unpack/dequantization |96.78|~0 (folded into sparse decode)|
| DFL/output decode |149.68|13.52|
| JPEG write |22.20|31.13|

Stage boundaries changed where operations are fused; compare combined costs, not
claim disappearance of work. Remaining major measured costs are crop localization
and JPEG snapshot generation. JPEG cost may reflect higher update/disk load.

## Implementation

`runtime/launcher/fast_crop_inference.py`, opt-in worker/controller flags
`--optimized --fast-transforms`:

- uint8 BGR→RGB via OpenCV; exact256-level LUT reproduces float32 RGB/255 followed
  by SDK float64 broadcast/TONEAREST quantization. No double normalization.
- Explicit MPK layout guards; vectorized input HWC1280/64/3 packing.
- INT8 dequantization LUTs exactly reproduce installed SDK float32 tensors.
- Class-score LUT/filter BEFORE bbox work; only qualifying anchors'64 DFL logits
  are unpacked/dequantized/decoded. Same .05 diagnostic cutoff, no confidence raise.
- Crop selector computes the chosen colour-seed/local-Canny path only; removes
  unused experiment candidates/ranking. Same full-frame grayscale/blur/Canny and
  blue grouping/quad fit; no new ROI, downscale, pose convention or size threshold.
- Stage timings in every inference JSONL. No change to compressed live-video path
  or separate timestamped result-panel semantics. Default remains unoptimized.

Installed vendor/SDK code is READ ONLY. Fast path refuses unsupported contracts.
Every fast startup compares full packed bytes and all six dequantized tensors to
installed SDK on synthetic patterns (all256 input levels/random raw output), then
compares sparse candidate scores and coordinates to reference decode. Any failure
aborts before starting the camera. Assertions passed in final board run; no
separate preflight JSON was emitted, source snapshot retains checks.

## Equivalence and evidence

-31 reviewed compiled replay frames: reconstruct packed INT8 values from saved
  dequantized heads, require recovered six tensors EXACT before comparing. All
  retained classes/scores/boxes match the original decoder EXACTLY on these inputs.
  Reconstructed padding is not claimed byte-identical to original raw padding.
-32 reserved historical VAL full frames: complete selected crop records identical,
  including unknown. Test images not consumed.
- Nine fast-path unit tests cover RGB equality, thresholds, empty output, layout
  rejection, all uint8 values, sparse/full prediction parity and unknown selector.
- Full suite326 passed, two existing warnings. Frozen fallback22-file check passed.
- No model, calibration, thresholds, labels, service, native r33, copied SDK or
  fixed-station changes. Frame queues still bounded/latest-only.

Evidence: `../IC_TERMINAL_TRAIN/latest-frame-profile-20261116-{baseline,opt1,opt2,opt3,opt4,final}/`.
Each includes deployed sources, logs, stage records, receiver/session evidence.
Aggregate: `../IC_TERMINAL_TRAIN/latest-frame-performance-20261116/{measurements,parity}.json`.
Final job stopped;09:43 UTC process check found no camera/worker/feed graphs.
Historical quality limitations, unwanted live present predictions and crop unknowns
remain; optimization is not new independent accuracy evidence.

## Run a new bounded optimized diagnostic

```
../MODEL/.venv/Scripts/python.exe tools/analysis/run_latest_frame_view.py \
  --output ../IC_TERMINAL_TRAIN/<NEW-unique-trial-directory> --seconds 60 \
  --headless --optimized --fast-transforms
```

Omit `--headless` for FFplay plus snapshot browser. Verify board is free and
approved172.21.30.230 locally assigned. Do not overwrite any prior output/package.
Next optional work: JPEG/panel transport and crop cost, or favourable-region
inspection as a separately measured behavioural experiment; neither enabled here.
