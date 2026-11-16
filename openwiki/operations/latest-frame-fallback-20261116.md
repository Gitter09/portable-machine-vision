# Frozen latest-frame fallback — 2026-11-16

Owner requested preserving the successful multirate pipeline before experiments.

**Use `../IC_TERMINAL_TRAIN/fallbacks/latest-frame-20261116-v2/`**, or its adjacent
ZIP plus `.zip.sha256`. Archive SHA-256:
`a8b74a1e96f856826cde7e6919a184282d0b26172a7e0be87c026ebb6a56a82a`.
Size31,421,798 bytes.22 manifested files, excluding manifest itself.

Package freezes runtime/helper files from the successful180s owner's actual source
snapshot: original unannotated camera video ~30FPS, latest-only compiled inference
~1.8FPS, separate timestamped inspection panel. Contains ELF/MPK/release inventory,
FP32 and raw reference ONNX, compiler lineage, owner-view metrics, frozen host
controller/panel, PowerShell launcher, verifier and instructions. No credentials.

`tools/package_latest_frame_fallback.py` builds new packages only, refuses reuse.
Preliminary v1 retained, v2 is selected: `-B` prevents Python bytecode from mutating
frozen contents. Default run outputs go outside package under `fallback-runs/`.
Do not use package folders as experiment output directories.

## Restore/run

Extract ZIP to a fresh folder. In PowerShell there:

```
.\run.ps1
# or
.\run.ps1 -Headless -Seconds 45
```

Defaults locate existing MODEL Python and external portable credentials by workspace
relative paths. After moving, supply `-Python` and `-Credentials` explicitly.
Receiver defaults to approved172.21.30.230:5010 and must be locally assigned.
Bound20..300s. Existing board SDK/plugins, host NumPy/OpenCV/Paramiko and FFmpeg/
FFplay are prerequisites, NOT bundled/frozen environments.

Verifier checks file set and SHA-256 before launch; ZIP SHA fingerprints archive,
not a publisher signature. The controller's new `--fixture` uploads frozen local
assets into a new board job, avoiding dependence on old replay directories or live
repo/export/model copies. Refuses occupied hardware; r33/services untouched.

## Packaging validation

- Per-file integrity and ZIP CRC passed.
- Fresh temporary ZIP extraction: verifier, launcher --help, verifier again passed
  without importing mutable project files or producing bytecode.
- PowerShell syntax passed. Four local-fixture completeness/tamper/path-escape tests
  added; full suite317 passed, two existing dependency warnings.
- No board connection or stream restart during packaging. The new restore/upload
  path is unit-tested but not re-exercised end-to-end here. Successful pipeline
  metrics are the earlier180s trial, not a fabricated new packaging deployment.

Future experiments must use new directories. Next owner plan remains favourable-
region inspection after evaluating this multirate view, then profiling/optimization.
