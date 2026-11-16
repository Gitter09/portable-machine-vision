# Original versus cropped FP32 detection review — 2026-11-15

## Owner handoff

Owner requested all detection images in a folder for personal review before
choosing further improvements. **Prepared; waiting for owner feedback.**

Folder: `../IC_TERMINAL_TRAIN/detection-review-20261115/`.
Open **`index.html`** for the local gallery; `START-HERE.md` explains the views.
No server, installation or board connection is required.

- `comparison/`: 42 side-by-side JPGs, original predictions in the crop-region
  zoom on the left, cropped-input predictions on the right.
- `original-full-frame/`: 42 complete original views, including off-ROI detections.
- `original-roi-zoom/`: 42 original predictions projected through the crop transform.
- `cropped/`: 42 cropped-input detection views.
- `contact-sheets/`: seven quick-review pages. Empty predictions are included.

All surviving boxes/scores are drawn with fixed class colours. No site filter,
strongest-per-site suppression, enhancement, automatic verdict or extra threshold
has been applied to make the views look cleaner. Rotated left-hand polygons are
original axis-aligned boxes projected into the rotated crop, not a mapping error.

## Controlled execution

**CPU FP32 reference only**, not quantized/native board output. Model SHA-256:
`b59eaab038d54ee4a623906366087544900c3570381e2abcf6b5cee7323290d6`.
ONNX input/output shape and canonical class metadata checked. Same unchanged model
and settings for both branches: confidence **0.05**, class-aware NMS IoU **0.45**,
top **100**, four-class argmax decoder from existing diagnostic helper.

All 42 owner-reviewed source images and saved hybrid crops included: **84 actual
ONNX inferences**. Revalidated source hashes/review identity; regenerated each
native-size crop from original pixels plus recorded transform and required exact
pixel equality with saved crop PNGs before running. No selective sample omissions.

Input processing for both branches: bilinear resize to fit 1280 square, centre
padding 114, BGR→RGB float32 /255, NCHW. The cropped branch consumes the **native
crop**, not the earlier 512x256 viewing preview. This is standard host reference
preparation, not byte-identical native CVU/quantized input replay. Actual rounded
x/y resize scales and padding are recorded and undone for box coordinates.

Full original boxes remain in JSON and full-frame images. The ROI zoom naturally
omits boxes entirely outside its display extent, but does not remove them from
inference results. Crop detections retain source-coordinate polygons via inverse
homography; do not replace a rotated quadrilateral by an incorrect tight box.

## Observed result — do not call this a quality improvement

| Class | Original-input boxes | Cropped-input boxes |
|---|---:|---:|
| IC_Fail | 151 | **0** |
| Terminal_pass | 2 | **0** |
| IC_Pass | 5 | **0** |
| Terminal_Fail | 42 | 34 |
| Total | 200 | 34 |

**21/42 cropped images have no retained detections**, including both both-present
views. No IC predictions survive fixed diagnostic settings on any cropped input.
Several crop detections are small Terminal_Fail candidates on other PCB features,
as shown in the saved comparisons. Fewer boxes is not better inspection here.
These are prediction counts, not labelled TP/FP/FN, precision/recall or a verdict.

Cropping demonstrably preserves reviewed geometry, but also changes model input
scale, orientation, interpolation and surrounding context. This experiment does
not isolate which change causes the failed detections. Do not conclude that a
confidence change, CLAHE, retraining or quantization fix is already established.
User will inspect the results before deciding the next experiment.

40 images are both-missing from one periodic capture; two are adjacent both-present
views from earlier framing. Third-state and independently held-out sessions remain
absent. FP32 results cannot be substituted for native compiled detection quality.

## Evidence / repeat

`detections.json` retains settings, class counts, all predictions, preprocessing
metadata/input tensor fingerprints, image/model/helper hashes and source/crop
transforms. `raw-predictions.npz` retains all 84 raw [1,8,33600] outputs, including
scores below cutoff. Exact reviewed-label and hybrid-result snapshots retained.
Frozen `experiment-source.py` matches the recorded script hash.

`analysis-validation.json` independently re-decodes every saved raw output and
requires exact detection equality, verifies model/review/raw/script fingerprints,
checks all 168 annotated JPGs and seven sheets, and validates gallery links.

```powershell
python tools/analysis/portable_crop_detection_review.py `
  --review annotations.review-export.json `
  --original ../IC_TERMINAL_TRAIN/site-label-draft-20261115/review-pack-final/annotations.draft.json `
  --images ../IC_TERMINAL_TRAIN/site-label-draft-20261115/review-pack-final `
  --hybrid ../IC_TERMINAL_TRAIN/hybrid-crop-20261115 `
  --model ../IC_TERMINAL_TRAIN/export-yolo11/yolo11n_fused.onnx `
  --output <new-directory>
```

Nine new tests cover RGB/normalization/padding, rounded-scale inverse mapping,
class-aware NMS, fixed settings, padding-only/empty detection handling, four-corner
crop/source projection, image immutability and off-ROI display behaviour.
Full suite **258 passed**, two existing warnings; compilation passes.

No board connection/preview interruption, model modification, calibration,
threshold tuning, training, runtime/release/SDK/vendor or fixed-station change.
Live health not checked; previously recorded preview bound remains applicable.
No further preprocessing or model changes pending the owner's review.
