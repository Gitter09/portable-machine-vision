---
type: reference
title: Files and Paths
description: Where everything lives, in the workspace, on the board and in training data.
status: active
tags: [paths, reference]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/vision_sdk/release.py#L21
  - repo://portable-vision/tools/compile_ic_terminal_on_admin.py#L24
---

# Files and paths

## This workspace

| Path | Contents |
|---|---|
| `vision_sdk/` | Shared SDK copy: release manager, board worker, compiler driver, preprocessing |
| `runtime/launcher/` | **The portable launcher and worker** that staged releases actually run |
| `vision_sdk/board_runtime/` | Fixed station's GigE worker and camera helpers (copied with the SDK; not used here) |
| `tools/` | The four model-pipeline tools |
| `tests/` | Tests for the tools and for the copied SDK |
| `openwiki/` | This documentation |

## Training and build artefacts (outside the repository)

Kept beside this workspace, not in it, because they are large and machine-local:

| Path | Contents |
|---|---|
| `..\IC_TERMINAL_TRAIN\dataset\` | Built datasets (`v4-1280` is the current one) |
| `..\IC_TERMINAL_TRAIN\runs\` | Ultralytics training runs |
| `..\IC_TERMINAL_TRAIN\export-yolo11\` | Six-head exports, ELF, MPK, compile logs, `compilation.json` |
| `..\IC_TERMINAL_TRAIN\template-r8\` | The three base JSONs used as config templates |
| `..\IC_TERMINAL_TRAIN\candidate\` | The candidate tree that gets staged to the board |
| `config/labels-ic-terminal.txt` | The four labels, one per line, in class order |
| `config/credentials.json` | Hosts, logins and endpoints for scripting |

The candidate tree is what the board actually receives:

```
candidate/
  deployment.json
  etc/0_preproc.json
  etc/0_process_mla.json
  etc/boxdecoder.json
  etc/mpk.json                     (same document as export-yolo11/yolov8_mpk.json)
  share/processmla/yolov8_stage1_mla.elf
  share/overlay/labels
  tools/launch.py
  tools/support.py
  tools/worker.py
  tools/verify.py
```

The four launcher files also exist in this workspace at `runtime/launcher/`, which
is the source of record; the candidate tree holds the copies that get staged. The
`tools/` copies must be refreshed from `runtime/launcher/` before a stage if either
has changed.

## On the board

| Path | Contents |
|---|---|
| `/home/sima/demeter-vision` | SDK release root used by `vision_sdk/release.py` |
| `/media/nvme/portable-release-YYYYMMDD-rNN` | Launcher releases used by `launch.py stage`; `r8` and `r20`–`r25` exist |
| `/media/nvme/ic-terminal-candidate` | Upload target used before staging |
| `/media/nvme/ic-terminal-verify-rNN` | Run output: detections, frames, packets, health |
| `/usr/bin/gst-launch-1.0` | The pipeline is driven through GStreamer |

There are two staging mechanisms on the board and they do not share a directory:
the SDK's own release manager writes under `/home/sima/demeter-vision`, while the
release bundle's `launch.py` writes `/media/nvme/portable-release-*`. The IC/terminal
work uses the launcher path.

Diagnostics written during this work live under `/tmp` on the board
(`/tmp/live` for the focus feed, `/tmp/decode-test` for the decode harness) and
are not part of any release.

## On the Admin PC

| Path | Contents |
|---|---|
| `C:/Users/Admin/Desktop/ICTerminalBuilds/` | Compile job roots, one folder per build |
| `C:/Users/Admin/Desktop/ICTerminalBuilds/<job>/compiled/` | ELF and MPK, pulled back by the compile tool |
| ModelSDK image | `vdp-cli/modelsdk:2.0.0_Palette_SDK_master_B240` |

## Naming

- Releases are `portable-release-<date>-r<n>`, monotonically increasing.
- Compile jobs are `ic_terminal_<graph-hash-prefix>_<utc timestamp>`.
- Verification runs are `ic-terminal-verify-r<n>` matching the release.

Do not reuse a release number. A stage re-verifies hashes but does not detect that
a number was recycled.
