# Latest fallback: browser coordinates + PCB pose reuse

Owner accepted the browser view and requested freezing it before C++ experiments.

Selected folder/ZIP: `../IC_TERMINAL_TRAIN/fallbacks/coordinate-browser-pose-20261116-v1`.
ZIP SHA-256 `f7429b31870bb514a456e63a3d68d81399519ff9656abb9a6da8f8a7e2611945`.
119,633,044bytes (~119.6MB),40 manifested files, adjacent SHA sidecar.

## Contents and identity

Actual runtime/helper/browser sources match accepted owner-view snapshot AND the
completed60s proof snapshot exactly. Frozen compiled ELF/MPK, reference ONNX,
compiler lineage, screenshot/geometry audit/measurement evidence, browser UI,
PowerShell setup/run and verifier included. Only host entrypoint packaging change:
explicit local `--fixture` and external `--credentials` instead of workspace paths.

Coordinate-only/optimized/fast transforms and pose reuse are explicitly enabled.
Original camera video is separate from exact-hash-matched inspection overlays;
no stale live boxes, favourable-region gate or C++ fusion. Restores the~12.6
inspection FPS/~30 decoded video FPS configuration, not production acceptance.

Nine exact host dependency wheels (including transitive SSH/crypto packages) are
bundled for Windows x64 CPython3.11, installable OFFLINE. Python itself and board
SDK/GStreamer are prerequisites, not bundled. No external FFmpeg executable needed:
PyAV provides host codec libraries. Credentials deliberately excluded.

## Restore

In PowerShell in a freshly extracted package:

```
.\setup.ps1
.\run.ps1
```

Setup creates a NEW environment outside package in `fallback-envs/`; refuses to
reuse an existing environment. Run default180s/max300s, receiver172.21.30.230,
videoUDP5010/metadata5012 must be free. Browser HTTP is loopback-only.
Override `-Python`/`-Credentials` in run.ps1 if moved or using existing environment.
Outputs go outside package under `fallback-runs/`; source/model upload goes into
new board job with occupied-board refusal. Frozen packages must not be edited.

## Validation and state

- All40 hashes, ZIP CRC and SHA sidecar passed.
- ZIP extracted to fresh temporary directory. Created an isolated no-system-site-
  packages venv, installed ALL dependencies from bundled wheels with --no-index,
  loaded H264 codec, ran host launcher help, reverified hashes successfully.
- Both PowerShell scripts parsed successfully. Full suite340 tests passed/two
  existing dependency warnings. Older22-file and24-file fallback hashes pass.
- No board contact/new camera graph while packaging. Existing owner300s session
  completed naturally:4077 loops/3862 located/215 unknown,3317 tracked/760 full,
  13 tracking confidence failures, no worker errors. Session evidence inside ZIP
  is its state AT packaging, possibly running; not rewritten after completion.
  Higher13.54 loop rate is not controlled comparative accuracy/performance proof.
- New packaged upload/restore entrypoint has offline checks, not an additional
  end-to-end board run here. Previous frozen fallbacks untouched; no r33/service/
  SDK/vendor/fixed-station modifications. C++ fusion remains the next experiment,
  not part of this fallback.
