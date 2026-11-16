---
type: architecture
title: Decisions and Rationale
description: Why the system is built this way, including the options that were rejected and why.
status: active
tags: [decisions, rationale, tradeoffs]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T12:00:00Z
sources:
  - repo://portable-vision/openwiki/operations/experiments.md
  - repo://portable-vision/openwiki/CHANGELOG.md
---

# Decisions and rationale

Each entry states the choice, the alternative, and the consideration that settled
it. Numbers are measured unless marked as estimates.

## D1. Six separate heads instead of one `[1,84,N]` output

**Chosen:** export six tensors — `bbox_0..2` and `class_0..2`.

**Alternative:** keep Ultralytics' single transposed output.

**Why:** the board decoder reads one box tensor and one score tensor per head.
A single packed output cannot be sliced into DFL bins without a decode stage the
board does not have.

## D2. YOLO11n instead of YOLO26n

**Chosen:** YOLO11n.

**Alternative:** YOLO26n, which was trained first.

**Why:** YOLO26's head is 4-channel anchor-free. The board supports either
pass-through (pre-decoded 4-channel) or DFL (64-channel). YOLO26's output is
neither, so it cannot be decoded on the board at all.

**Cost:** a retrain. Measured result on the old dataset: IC_Fail P 1.000,
R 0.928, mAP50 0.977.

## D3. 1280 input instead of a crop

**Chosen:** feed the whole 1280×720 frame, letterboxed to 1280×1280.

**Alternative:** crop a square window around the plate axis and feed 640.

**Why:** the crop route required `videocrop` between the decoder and the
preprocessing node, which fails with `Pad push YUV buffer failed`. Cropping in the
model tooling was possible but bound the model to one plate position and threw
away the outer field that the framing measurements depend on.

**Cost:** the IC arrives at ~24 px instead of a larger cropped size; accepted
because 1280 held the contract at ≥25 fps on the raw-head path.

## D4. DFL on the board instead of decoding in the graph

**Chosen:** raw 64-channel heads, `decode_type: "yolov8"`, decoder-side softmax.

**Alternative:** the vendor's canonical decoded head (4-channel boxes, sigmoided
classes, `decode_type: "yolo"`).

**Why:** in-graph decode or sigmoid at 1280 measured **0.6–0.78 fps**; raw heads
measured 27–30 fps. The 25 fps requirement forced the decoder to do the work.

**Known cost, unresolved:** the DFL path has never reproduced the host's
detections. This decision is the one currently under review. The 0.6 fps number
predates a canonical vendor export and deserves a re-test; if the canonical path
holds 25 fps at 1280, D4 should be reversed.

## D5. Re-aim the camera instead of backing it off

**Chosen:** keep the distance, tilt the aim so the sweep is centred.

**Alternative:** move the camera back ~20% to fit the sweep.

**Why:** the measured sweep overflowed the top edge while leaving 179 px unused at
the bottom and ~300 px on each side. Re-aiming recovered that slack at zero cost
to resolution. Result: 0/40 frames clipped, 11 px top / 17 px bottom clearance.

**Cost accepted:** the sweep fills 96% of frame height. The mounts are now
effectively fixed; any shift invalidates the framing and needs a re-measure.

## D6. Letterbox instead of stretch in preprocessing

**Chosen:** `aspect_ratio: true`, `padding_type: CENTER`, 1280×720 → 1280×1280.

**Alternative:** stretch to the model square.

**Why:** the model trains on letterboxed images. Stretching changes the aspect the
network sees and would invalidate every trained scale. The mapping back to source
coordinates (`upscale`, `margin_y = 280`) depends on this choice.

## D7. One copy of `vision_sdk` per workspace

**Chosen:** duplicate the package; keep the two copies in step by hand.

**Alternative:** import one shared installation from both workspaces.

**Why:** the portable release line must be able to move without a desktop-runtime
change breaking a board release. A shared import couples the two machines' release
cadence, which is what the split was meant to end.

**Cost:** fixes that belong to both must be applied twice. The changelog says so
when it happens.

## D8. Move only unreferenced files during the split

**Chosen:** move the IC/terminal tooling; copy the shared SDK.

**Why:** the fixed station's backend imports `vision_sdk.preprocessing`. Moving it
would have broken a running machine to tidy a folder. A repository-wide search
confirmed the moved files had no inbound references.

## D9. Threshold 0.05 during decode experiments

**Chosen:** run bench diagnostics at 0.05, not 0.25 or 0.5.

**Why:** at the current camera distance the model's scores on the live scene top
out around 0.2 on the board; a 0.25 threshold returns nothing and hides the
defect's shape. 0.05 exposes the distribution (including the noise in the
letterbox padding) which is what the experiments needed.

**Do not ship 0.05.** Production thresholds are 0.25 and above, and the current
candidate config carries 0.05 only because it is still under investigation.

## D10. Fixed class order with an empty class

**Chosen:** keep `3 Terminal_Fail` in the label file with no training examples.

**Why:** class ids are baked into the compiled graph and into every stored
detection. Removing the class and adding it later would renumber everything and
invalidate every historical result.

## D11. Recapture instead of augmenting scale

**Chosen:** accept that the existing 300-frame dataset is unusable and recapture
at the final camera position.

**Why:** the model has no scale augmentation, so it is bound to the training
distance. Measured at the new position: 0 detections at 1280, 2 at 640. The
alternative — a scale-jittered retrain — would need the new-position data it does
not have.

**Ordering constraint:** settle the decode path first. Recapturing before then
spends annotation effort on a model that cannot be validated.
