---
type: guide
title: Workspace split, 2026-11-14
description: What was moved out of the fixed repository, what was copied, and why.
status: active
tags: [migration, boundaries, portability]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://demeter-systems-mv-full/tools/ic_terminal_dataset.py
  - repo://demeter-systems-mv-full/tools/build_ic_terminal_configs.py
  - repo://demeter-systems-mv-full/vision_sdk/release.py
---

# Workspace split, 2026-11-14

The portable station lived inside the fixed AOI repository. On 2026-11-14 the
portable work was given its own workspace so the two machines can be changed,
released and reasoned about independently.

## Rule applied

- **Move** a file only when nothing in the fixed station references it.
- **Copy** a file when both machines use it.

Every move below was checked with a repository-wide search before it happened.

## Moved out of the fixed repository

These are portable-only. They have no inbound references from `backend/`,
`inference/`, `frontend/`, `deployment/` or the fixed station's tests.

| File | Why portable-only |
|---|---|
| `tools/ic_terminal_dataset.py` | Builds the four-class IC/terminal dataset |
| `tools/export_modalix_heads.py` | Cuts and rewrites the six-head export for the board decoder |
| `tools/build_ic_terminal_configs.py` | Derives `0_preproc.json`, `0_process_mla.json`, `boxdecoder.json` |
| `tools/compile_ic_terminal_on_admin.py` | Drives the ModelSDK container for this model |
| `tests/test_ic_terminal_dataset.py` | Covers the dataset tool |
| `tests/test_ic_terminal_configs.py` | Covers the config generator |

They now live at the same relative paths inside `portable-vision/`, so
`compile_ic_terminal_on_admin.py` still finds `vision_sdk/vendor_compile_worker.py`
via `Path(__file__).resolve().parents[1]`.

## Copied, because both machines use them

| Copied | Also used by the fixed station for |
|---|---|
| `vision_sdk/` (whole package) | Operator HMI backend and the Windows desktop runtime |
| `tests/test_vision_sdk.py` | SDK coverage (copied so this workspace is testable) |
| `tests/test_sdk_triggered_camera.py` | Board camera eligibility |
| `pyproject.toml` | Packaging |

The SDK is copied rather than imported because the two workspaces must be able to
move independently: a board release here must not be broken by a desktop-runtime
change there, and vice versa.

### What `vision_sdk` is — and is not

The copy is byte-identical to the fixed repository's, and it is **the fixed
station's code**. In particular `vision_sdk/board_runtime/` is the screw rig: an
Aravis GigE camera, 3200×2256 Bayer, `screw_count`, a hard-coded `192.168.1.17`,
and 80/40/20 head grids. It contains no `rtspsrc`, no `routecam`, no
`simaaidecoder`.

The RTSP/1280 pipeline that this station's releases run is **not** in
`vision_sdk`. It was recovered on 2026-11-14 from the candidate tree outside the
repository and now lives at `runtime/launcher/` (`worker.py`, `support.py`,
`launch.py`, `verify.py`).

So: a change to `vision_sdk/` here is a change to the fixed station, and a change
to `runtime/launcher/` is a change to this station. They are different machines.

### Consequence you must live with

`vision_sdk/` now exists twice:

- `portable-vision/vision_sdk/` — this workspace
- `demeter-systems-mv-full/vision_sdk/` — the fixed station

A change that belongs to both must be made in both, and the changelog entry says
so. A change that belongs to one must not leak into the other. The package is
currently identical; `MIGRATION.md` records the copy date.

## Deliberately left behind

Everything that serves the fixed AOI screw station: `backend/`, `frontend/`,
`inference/`, `deployment/runtime/`, `machine_vision_runtime/`, `manual/`,
`seed/`, the production release builders and their tests. None of it is needed to
build, stage or run the portable board.
