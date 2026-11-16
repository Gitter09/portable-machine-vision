# Portable-only C++ input fusion prototype

Not installed into vendor SDK or promoted into a fallback. The compiled function
reads already resized/padded contiguous1280x1280 BGR uint8, applies the caller's
EXACT256-byte INT8 lookup table, and writes compiler tiles HWC1280/64/3 directly.
No float rounding, interpolation, crop, model or calibration changes.

Build in a NEW board work directory (Debian g++12.2/AArch64):

```
g++ -O3 -std=c++17 -fopenmp -fPIC -shared -Wall -Wextra -Werror \
  -o libportable_input.so fused_input.cpp
```

Exports C ABI v1 (ctypes, GIL released), scalar, NEON and bounded1..16-worker OpenMP
kernels. No per-pixel intermediate image buffers. Python owns input/output/LUT;
output allocated per call. Never pass overlapping buffers; supported wrapper
allocates a separate destination. Null pointers and incorrect byte counts are refused at ABI.

Native NEON and scalar were slower than existing parallel OpenCV. OpenMP8 workers
won in actual board tests. Use OMP_WAIT_POLICY=PASSIVE / OMP_DYNAMIC=FALSE only
in the child benchmark/runtime process; no global environment or service edits.
libgomp.so.1 is an existing OS prerequisite, not bundled into SDK. No fast-math.

`fused_input.py` rejects noncontiguous/wrong shape/type, incompatible ABI and missing
requested kernel; library SHA is checked by worker BEFORE loading. Startup packed
bytes must match FastTransforms; its existing preflight also checks installed SDK.

Camera-free `tools/analysis/benchmark_fused_input.py` checks31 prepared input frames,
all256 levels/random patterns, poisoned destination margins and invalid lengths,
then interleaves60 warmed calls per implementation. Existing `inputs.npz` must be
fingerprint verified. Run with a time bound and NEW evidence output.

Opt-in live host invocation:

```
../IC_TERMINAL_TRAIN/browser-view-env/Scripts/python.exe tools/analysis/run_coordinate_browser.py \
  --output ../IC_TERMINAL_TRAIN/<NEW-trial> --seconds 40 --headless --track-pose \
  --cpp-library ../IC_TERMINAL_TRAIN/cpp-input-fusion-20261116-parallel/libportable_input.so \
  --cpp-kernel parallel8
```

No flag = original fast OpenCV/NumPy path. Do not replace libs in any existing
workdir/fallback. Current package builder rejects C++ candidate packaging until
library/kernel/dependencies are explicitly added. See operations C++ fusion report.
