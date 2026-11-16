# C++ input fusion experiment — 2026-11-16

Owner authorized the first C++ fusion step. Implemented a separate input-only
prototype, NOT a replacement of the accepted browser/pose fallback. Default path,
weights, crop/resize/pad114, numerical lookup table, thresholds and video unchanged.
No output decoder/tracker rewrite or favourable-region gate.

## Implementation and correctness

`runtime/launcher/native/fused_input.cpp` + `fused_input.py`:
- One pass: BGR bytes → caller's exact quantization LUT → packed HWC1280/64/3 input.
  No full RGB/quantized intermediate images, floating-point math or interpolation.
- C ABI1/ctypes releases GIL. Python owns separate buffers; exact image geometry,
  contiguity/type, MPK layout, ABI/kernel availability and library SHA checked.
- Scalar, explicit AArch64 NEON table lookup and OpenMP2/4/8/16 alternatives.
  Disjoint output rows, implicit join before returning. No installed SDK changes.
- Startup compares all256-level pattern and random image bytes to accepted fast
  reference (whose preflight checks installed software SDK). Refuse mismatches
  before camera start. Native raw-byte/shape guards do not silently fall back.
- Camera-free tests: ALL31 prepared reviewed inputs reconstructed exactly from
  recorded float32 RGB inputs. Every tested implementation's4,915,200 packed bytes
  equals reference for every frame. Poisoned64-byte output prefix/suffix remained
  untouched, invalid input length rejected. No source/label/accuracy tuning.

## What did NOT work

First single-thread scalar/NEON implementations were substantially slower. Reordering
loops to read contiguous rows helped scalar, but not enough. The existing OpenCV
path already uses a16-thread TBB pool. C++ alone is not inherently faster.

All initial artifacts/results retained:
`../IC_TERMINAL_TRAIN/cpp-input-fusion-20261116/`, `...-rows/`, `...-parallel/`.

## Camera-free interleaved benchmark

Debian g++12.2/AArch64, -O3/-fopenmp/-fPIC/-shared/-Wall/-Wextra/-Werror, no fast-math.
libgomp.so.1 is existing OS dependency.60 warmed timings per method,3 rotating
prepared inputs. Compile command/compiler/ldd/source/library retained.

| Implementation | Mean input preparation ms |
|---|---:|
| Existing OpenCV/NumPy |8.70|
| Scalar C++ |18.46|
| Explicit NEON |29.29|
| OpenMP2 |9.54|
| OpenMP4 |5.15|
| OpenMP8 |**3.24**|
| OpenMP16 |3.09|

Selected8 workers rather than occupying16 for only0.15ms extra microbenchmark gain.
This limits THIS kernel's workers, not all pipeline threads. OMP_WAIT_POLICY=PASSIVE
and OMP_DYNAMIC=FALSE scoped to child process, no global tuning/clock changes.

## Full live pipeline trials

Two consecutive40s bounded trials, latest-only/coordinate/pose path, same model:

| Measurement | No C++ | C++8 |
|---|---:|---:|
| Whole-run inspections/s |12.44|**13.34**|
| Located/unknown |511/0|549/0|
| Mean input stage ms (after3 warmup records) |8.92|**3.81**|
| Mean frame processing ms |69.51|63.08|
| Host decoded arrival FPS |30.05|30.03|
| Exact metadata/frame hash matches |511/511|549/549|
| RTP sequence gaps / worker errors |0/0|0/0|

Observed~7.3% total throughput gain, not a guaranteed FPS uplift: live trials are
sequential rotating scenes, not paired identical frames. Full/tracked pose mixture
also differs. No independent accuracy, physical latency or soak acceptance claimed.
Host p99 delivery gap78ms both; max969ms/219ms includes startup/burstiness. No
zero-jitter claim. Video board counters include startup/shutdown and are lower
than30FPS; no fabricated render-FPS measurement.

Selected library SHA-256:
`da2f8d81121aec34955fb6dd23df1bb9180e7151f5badc4a59cc1594b32f5182`.
Worker checks this hash before loading and records kernel/preflight proof.

## Evidence and operation

- `cpp-fusion-live-20261116-{baseline,parallel8}/`: source/library snapshots,
  worker/receiver/frame/published/preflight JSON, no mutation of prior runs.
- `cpp-fusion-summary-20261116.json`: aggregate, hashes, microbenchmark and live metrics.
- Six host guard tests plus actual compiled tests on board; suite346 passed,
  two existing dependency warnings. Latest40-file and prior24-file fallback
  verifiers still pass. No frozen package changed or new fallback published.
- Both live jobs ended;11:15 UTC read-only process check found no active graphs.
  No r33/services/vendor/SDK/fixed-station edits. No board model compilation needed.

Use `run_coordinate_browser.py --cpp-library <AArch64-so> --cpp-kernel parallel8`
plus normal --track-pose/output/bound args. Omit library to keep accepted default.
Native source README documents build/benchmark. This is opt-in and not packaged;
existing coordinate fallback builder now refuses C++ proof rather than silently
omitting native library/kernel. Explicit packaging support is needed before any
future native fallback.

Next performance choices should follow updated profiling: remaining crop/feature
seeding and output stages may offer more than squeezing the last~4ms input stage.
No further optimization or viewer session was started without owner feedback.
