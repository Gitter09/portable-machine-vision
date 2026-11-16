# Optimized fallback package — 2026-11-16

Owner accepted the optimized live view and requested another fallback.

Selected: `../IC_TERMINAL_TRAIN/fallbacks/latest-frame-optimized-20261116-v2/`.
ZIP SHA-256: `cb770d073deec3fe60c6d0e79d930236d7e0222964f884253843870100a0d053`.
Archive31,427,722bytes,24 manifested files. Run `run.ps1` after extraction.

- Freezes EXACT board runtime/helpers from owner's `optimized-live-view-
  20261116-095325/source-snapshot`, compiled ELF/MPK, reference ONNX, host launcher,
  UI, provenance and evidence. Launcher explicitly passes `--optimized` AND
  `--fast-transforms`; cannot silently start the slow default path.
- Keeps earlier slower `latest-frame-20261116-v2` intact and reverified.
- Credentials excluded. Requires existing board SDK and host Python/dependencies/
  FFmpeg, approved172.21.30.230:5010 locally assigned. Default180s, max300s.
  New runs upload frozen local assets into unique board jobs, write outside the
  package and refuse occupied hardware. No r33/service/vendor/fixed-station edits.
- Preliminary optimized v1 retained. Use v2, which includes a host-only snapshot
  publishing fix and actual owner-view sources (not just prior profiling source).

## Windows panel failure found while packaging

Owner's host controller terminated early when the browser held `latest.json`
open and `os.replace` raised Windows PermissionError. Its finally block closed
receiver/server; board independently ran to its time bound. Original session.json
is stale `running`; original traceback preserved, not rewritten as success.
Read-only SSH recovered completed worker/log/frame evidence. Board:2820 iterations,
1928 located/892 unknown,301.05s,~29.77 forwarded video FPS, no worker errors.
The higher9.37 loop rate includes many cheap unknowns; do NOT replace the earlier
~8FPS benchmark with an implied quality/performance improvement.

Packaged host launcher retries transient reader locks; on persistent locks keeps
old snapshot/stale state and retries next poll, rather than crashing video. Existing
immutable snapshot image files are not downloaded over themselves. Two retry/
preservation tests added. This is not a model or board behaviour change.

## Validation

24-file SHA verification, ZIP CRC, fresh temporary extraction/help/reverification,
explicit optimized launcher flags and PowerShell parse all passed.328 tests pass,
two existing dependency warnings. Original slower package22-file verification
still passes. No new stream was launched during packaging; host retry fix was
unit-tested, not newly exercised in an end-to-end browser session here.

Original valid board180s/60s measurements and limits remain in performance report.
No production-accuracy, optical-latency, crop-coverage or soak acceptance implied.
