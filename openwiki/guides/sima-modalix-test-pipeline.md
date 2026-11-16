---
type: guide
title: What the Modalix Path Can Be Trusted With
description: Capability boundary for the portable board, stated as measurements rather than claims.
status: active
tags: [sima, modalix, capability, boundary]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/openwiki/CHANGELOG.md
  - board://portable-release-20261113-r24
---

# What the Modalix path can be trusted with

## Proven

| Claim | Evidence |
|---|---|
| The board runs a 1280 six-head graph at ~29 fps | measured over 1181 frames in 40 s |
| The 640 canonical path decodes correctly | release `r8`, YOLOv8m, in-graph decode, 27 fps |
| Release staging and hash verification work | every stage/run since `r10` |
| The compiler's geometry can be reproduced in configs | `build_ic_terminal_configs.py` + tests |
| The camera can be driven indefinitely from the board | supervisor + frame pruning, days-long design |

## Not proven

| Question | Current answer |
|---|---|
| Does the board reproduce the host's detections at 1280? | **No** — see the decode defect |
| Can the pre-decoded path hold ≥25 fps at 1280? | Unknown; 0.6 fps was measured for a different graph |
| Does a run ever qualify? | No — `missing_media_time`, `nonadvancing_rtp_time` |
| Is the accuracy acceptable on the production scene? | Unknown; the current model is from another camera distance |

## The boundary, in one sentence

The board is a proven *runner* and an unproven *decoder*. Treat throughput,
staging and camera handling as reliable; treat any detection it reports as
suspect until the decode defect is closed.

## Why the canonical path is the likely answer

The vendor's own pipeline never asks the board to do DFL. Its surgeon rewrites the
head into 4-channel decoded boxes with sigmoided class outputs, and its
`boxdecoder.json` therefore uses `decode_type: "yolo"` with depth 4. That is also
the only configuration observed to produce correct boxes on this hardware.

Our 1280 DFL design exists because in-graph decode measured 0.6 fps at 1280. That
number predates a canonical vendor export and deserves a re-test before the DFL
path is debugged any further.

## Reference: the fixed station

The fixed AOI screw station (`../demeter-systems-mv-full/`) uses the same SDK but a
different board, a different camera geometry and a production inference service.
Nothing measured here transfers to it, and its releases must not be staged on the
portable board.
