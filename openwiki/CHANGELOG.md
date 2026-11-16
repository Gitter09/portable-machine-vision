# Changelog

Entries are appended, never edited, removed, reordered or squashed.

## 2026-11-14 - Split the portable station into its own workspace

- Category: Workspace separation.
- Decision/change: The portable IC/terminal work moved from
  `demeter-systems-mv-full/` to `d-meter/portable-vision/`. Four tools and two
  tests moved (`ic_terminal_dataset.py`, `export_modalix_heads.py`,
  `build_ic_terminal_configs.py`, `compile_ic_terminal_on_admin.py`,
  `test_ic_terminal_dataset.py`, `test_ic_terminal_configs.py`); `vision_sdk/`,
  `pyproject.toml` and `conftest.py` were copied because the fixed station uses
  them too. This wiki was written for the new workspace, with paths updated to
  match.
- Reasoning: The two machines were being changed in one repository with one wiki.
  The portable work needed its own release line, its own board notes and its own
  defects, and mixing them made both harder to reason about. Moving only
  unreferenced files keeps the fixed station intact; copying the shared SDK keeps
  this workspace runnable without it.
- Affected components: `portable-vision/{vision_sdk,tools,tests,openwiki}`. The
  fixed repository keeps `backend/`, `frontend/`, `inference/`, `deployment/`,
  `machine_vision_runtime/` and the production release builders.
- Verification: 21 tests pass in the new workspace for the two moved suites and
  34 for the two copied SDK suites. The moved compile tool still resolves
  `vision_sdk/vendor_compile_worker.py` through `parents[1]`, which now points at
  the workspace root.
- Backward compatibility: The fixed station is unaffected: no file it references
  was moved. `vision_sdk/` now exists twice and must be kept in step by hand;
  `MIGRATION.md` records the copy and the rule.

## 2026-11-13 - Give the MLA unpack node the geometry the compiler produced

- Category: Model tooling and board decode defect.
- Decision/change: `build_ic_terminal_configs.py` now writes the unpack node's
  `input_width`, `input_height` and `input_depth` from the compiler's tensor
  shapes. It previously wrote `output_*`, guarded by `if key in mla`; those keys do
  not exist in the node's schema, so the assignments were skipped and the
  template's 640-model geometry survived. The generator now also rejects an MPK
  whose grids contradict `--model-size`, whose depths contradict `--box-channels`,
  or whose unpack and decode geometry disagree.
- Reasoning: The unpack node advertised six 4-channel tensors on 80/40/20 grids
  while the compiled graph emits 64/64/64/4/4/4 on 160/80/40. The buffer total
  matched (2,688,000 bytes), so the pipeline prerolled and ran at 29 fps while the
  decoder read DFL bins as class scores. Output changed from uniform ~300 px
  squares to IC/terminal-sized boxes.
- Affected components: `tools/build_ic_terminal_configs.py`,
  `tests/test_ic_terminal_configs.py`.
- Verification: Compiler shapes for the yolo11n 1280 MPK are grids 160/80/40 and
  depths 64/64/64/4/4/4; the regenerated `0_process_mla.json` and
  `boxdecoder.json` agree on all six tensors where they previously disagreed. The
  defect is **not closed**: the board still reports 0.4 boxes per frame at max
  score 0.217 where the host reports 5.8 at 0.986.
- Backward compatibility: Config generation only. Releases `r20`-`r25` remain
  candidates, not accepted deployments.

## 2026-11-14 - Record live board state and the experiment history

- Category: Documentation.
- Decision/change: Added `operations/status.md` (what is staged, what is running,
  known drift, next actions) and `operations/experiments.md` (every decode variant
  tried, with frame counts, scores and verdicts). `AGENTS.md` and the quickstart
  now tell a reader to load the status page before touching the board.
- Reasoning: The architecture pages describe contracts, not state. After the split,
  the board's candidate directory still held the rejected BGR variant while the
  workstation copy held RGB, and three capture supervisors had accumulated — both
  invisible from the code and both capable of wasting a working day.
- Affected components: `openwiki/operations/status.md`,
  `openwiki/operations/experiments.md`, `openwiki/index.md`,
  `openwiki/quickstart.md`, `AGENTS.md`.
- Verification: The documented state was read back from the board on 2026-11-14:
  one supervisor, one capture, candidate configs RGB at threshold 0.05, camera up.
  Releases `r8` and `r20`-`r25` present, with `r25` recorded as staged from
  the rejected BGR variant.
- Backward compatibility: No code or deployment behaviour changed.

## 2026-11-14 - Close the handover gaps: rationale, credentials, analysis scripts

- Category: Documentation and workspace completeness.
- Decision/change: Added ``architecture/decisions.md`` (eleven decisions with the
  alternatives and the measurements that settled them),
  ``reference/credentials.md`` plus ``config/credentials.json`` (hosts, logins and
  endpoints in plain text at the owner's direction), and
  ``reference/analysis-scripts.md``. The 37 bench scripts that had been living in
  ``..\IC_TERMINAL_TRAIN`` moved to ``tools/analysis/`` and now resolve their data
  root through ``analysis_paths.py`` (``IC_TERMINAL_TRAIN`` overrides it) instead
  of a hard-coded absolute path. ``config/labels-ic-terminal.txt`` moved into the
  workspace and the deployment command was updated to match.
- Reasoning: An agent starting in this workspace could read the contracts and the
  live state but not the reasoning, the logins, or the scripts that produced the
  numbers it was told to trust. Rationale that only exists in a conversation is
  lost at the next handover, and a documented measurement nobody can reproduce is
  worth less than an undocumented one that can.
- Affected components: ``openwiki/architecture/decisions.md``,
  ``openwiki/reference/{credentials,analysis-scripts,files-and-paths,cli}.md``,
  ``openwiki/index.md``, ``openwiki/operations/deployment.md``, ``AGENTS.md``,
  ``config/``, ``tools/analysis/`` (37 scripts moved, plus ``analysis_paths.py``).
- Verification: All 37 scripts compile after the path rewrite
  (``py_compile``, 0 failures). ``analysis_paths.py`` resolves the data root and
  the labels file. ``judge_signed.py`` and ``inspect_unpack.py`` were run from the
  new location and produced the same output as before the move.
- Backward compatibility: Scripts now depend on ``analysis_paths.py`` being
  importable, which holds when they are run from ``tools/analysis``. The training
  folder no longer holds them; anything outside this workspace that called those
  paths by name will need updating, and nothing in the fixed repository did.

## 2026-11-14 - Correct the runtime claim, locate the timing defect, add the guard test

- Category: Documentation correction, workspace completeness, test coverage.
- Decision/change: The workspace now contains the runtime it documents.
  ``runtime/launcher/{worker,support,launch,verify}.py`` was copied in from
  ``..\IC_TERMINAL_TRAIN\candidate\tools``, where it had lived while the wiki
  described ``vision_sdk/board_runtime/persistent_board_worker.py`` instead. That
  file is the fixed station's screw rig (Aravis GigE, 3200x2256, ``screw_count``,
  hard-coded 80/40/20 grids) and builds no RTSP graph; it is retained only because
  it arrives with the copied SDK. ``architecture/runtime.md``, ``AGENTS.md``,
  ``MIGRATION.md``, ``reference/files-and-paths.md``,
  ``reference/tests.md`` and ``operations/troubleshooting.md`` were corrected to
  match. The timing defect's root cause is now located and documented with file and
  line references rather than a list of things to investigate. The dead
  ``conftest.py`` (it inserted ``inference/service``, which does not exist here) was
  removed. ``requirements-dev.txt`` records the dependencies and the interpreter
  that actually has them. The geometry guard was extracted as
  ``assert_geometry_matches()`` and is now covered by
  ``tests/test_ic_terminal_geometry_guard.py`` against the preserved bad config pair
  in ``tests/fixtures/unpack_640_bug/``.
- Reasoning: An audit found that the board-runtime half of the workspace was not
  the code the board runs, so its architecture page described a graph that is never
  built. Two defects were being tracked from the wrong file. Separately, a claim in
  the 2026-11-13 entry and in three pages — that the unpack and decoder buffer
  totals matched — is wrong: the unpack node declared 67,200 bytes of tensor
  geometry, the decoder 2,284,800, and the MLA buffer is 2,688,000 after compiler
  padding. Nothing compared them, which is the point; the corrected wording says so.
- Affected components: ``runtime/launcher/`` (new), ``conftest.py`` (removed),
  ``tests/test_ic_terminal_geometry_guard.py`` and ``tests/fixtures/unpack_640_bug/``
  (new), ``tools/build_ic_terminal_configs.py`` (guard extracted), the documentation
  pages listed above, ``README.md``, ``requirements-dev.txt``.
- Verification: 60 tests pass (26 IC/terminal tooling, 34 copied SDK). The guard
  test asserts the bad pair is rejected, names the field that disagrees, catches a
  depth-only mismatch, and passes a matching pair. ``runtime/launcher/`` compiles and
  its imports are standard library plus GStreamer; it has no automated tests here.
- Backward compatibility: No board behaviour changed. The candidate tree still holds
  the deployed copies of the launcher; ``runtime/launcher/`` is now the source of
  record and must be copied into the candidate tree before staging.

## 2026-11-14 - Trial timestamp carry-forward and recover board hardware services

- Category: Portable runtime diagnostics and operational recovery.
- Decision/change: Trial changes in `runtime/launcher/worker.py` track timestamp
  sources and carry `GstSimaMeta` timestamps by frame id to encoder output when
  native stages omit buffer PTS. Staged trials r27/r28 are not accepted deployments.
  With user approval, restarted the portable board's `simaai-appcomplex`,
  `decoder`, and `encoder` services and ran r28 once in diagnostic mode.
- Reasoning: Earlier runs had no valid timed frames. Later startup attempts
  aborted with `Unable to connect to the server from dispatcher`. A restart
  cleared that startup failure; the exact failed service is not isolated.
- Verification: `/media/nvme/ic-terminal-verify-r28-postrestart/worker.json`
  records 426 frames, 426 valid timed frames, 426 RTP frame markers, no errors,
  and clean shutdown after a 15-second diagnostic request. A separate r27
  baseline retry was blocked by an `etc/0_preproc.json` checksum mismatch, so
  it provides no current hardware baseline. No checksum guard was bypassed.
- Limitations: No detection-accuracy acceptance, real receiver test, long-run
  test, or frame-cache bound validation. Application qualification remains
  `not_established`. Capture feed and inference are stopped at this checkpoint.
- Affected components: `runtime/launcher/worker.py`,
  `openwiki/architecture/runtime.md`, `openwiki/operations/status.md`, portable
  board hardware services. Shared `vision_sdk/` and the fixed station untouched.

## 2026-11-14 - Identify timestamp-map safety issues before the longer trial

- Category: Runtime review evidence; no runtime or board change.
- Findings: An offline exercise of the actual `worker.py` probe retained all
  10,000 timestamp entries after 10,000 simulated completed frames. A separate
  overlapping-stream simulation assigned stream B's timestamp to stream A's
  same-numbered frame, because the map is keyed only by frame number.
- Decision: Pause the longer board trial and request approval for bounded,
  stream-aware, consume-on-encode storage with regression tests. The simulated
  stream collision is not evidence of a collision on the live board.
- Affected pages: `openwiki/operations/status.md`. Shared SDK and both boards
  were unchanged in this review.

## 2026-11-14 - Bound timestamp storage and separate stream identities (r29)

- Category: Portable runtime safety and board timing validation.
- Decision/change: Added thread-safe `FrameTimestampCache` in
  `runtime/launcher/support.py`, keyed by `(stream_id, frame_id)`, capped at 512
  pending entries, with oldest-entry eviction and consume-on-encode lookup.
  `worker.py` records only decoder-output metadata timestamps, consumes entries
  even when native encoder PTS exists, and reports cache counters in health/run
  evidence. Invalid/conflicting source metadata or timestamp assignment failure
  stops the run; missing labels never fabricate time or weaken admission gates.
- Reasoning: The trial map retained every completed frame and could confuse
  same-numbered frames from different streams. Single-source insertion also
  prevents late boxdecode output from recreating consumed entries.
- Verification: Added 34 tests in `tests/test_portable_timestamps.py`, covering
  cache contracts, concurrent access, and the actual probe callback with fake
  buffers/pads. Full suite: **94 passed**, two existing dependency deprecation
  warnings. Launcher source compiles. Uploaded and hash-verified repository
  launcher files, then staged new portable release `20261114-r29`; candidate
  model/config/deployment inputs were checked unchanged against r28. Existing
  releases and checksum checks were untouched.
- Board results: Two 300-second diagnostic requests yielded 8,978 and 8,983
  frames, every frame validly timed, matching RTP markers, no errors/warnings,
  no cache misses/evictions, peak pending 30/512, and clean exit 0. Raw RTP
  verification passed; frame-log run integrity passed. Second-run worker RSS
  after 30 seconds ranged 61,252–61,636 KiB (384 KiB first-to-last increase).
  First-run RSS sampler used an unsupported procfs interface and collected no
  memory evidence; a supported `ps` sampler was used for the repeat.
- Evidence: Board `/media/nvme/ic-terminal-verify-r29-soak{,-memory}`; small
  evidence copies at `../IC_TERMINAL_TRAIN/verify-r29-soak{,-memory}`. Raw packet
  captures remain on board. One/two source frames still in flight at shutdown
  account for the small final pending counts.
- Limitations: Local diagnostic sink only; no proof of real receiver playback,
  capture-clock semantics, image/result pairing, numerical accuracy, recovery,
  or all-day stability. Full application qualification remains unestablished.
  Capture feed and inference remain stopped at the checkpoint.
- Affected components/pages: `runtime/launcher/{support,worker}.py`,
  `tests/test_portable_timestamps.py`, `openwiki/architecture/runtime.md`,
  `openwiki/reference/tests.md`, `openwiki/operations/{status,troubleshooting}.md`.
  Shared `vision_sdk/` and the fixed station were not modified.

## 2026-11-14 - Check receiver reachability before changing the video destination

- Category: Network preflight evidence; no deployment/runtime change.
- Findings: r29 pins receiver `192.168.1.50:5010`; the board reports neighbour
  resolution `FAILED` for that address and receives no ping replies. The current
  workstation is `172.21.30.13`. It responds to board ping and a temporary local
  UDP listener on port 5010 received the exact 30-byte probe sent by the board
  (`172.21.30.120`). Local FFmpeg/FFplay 8.1 are available.
- Decision: Ask which receiver to use before staging a changed destination.
  A small UDP probe does not establish video decoding or normal playback.
- State: Probe socket closed; no firewall/IP changes, no release or candidate
  edits, no inference/capture startup. Fixed station untouched.
- Affected page: `openwiki/operations/status.md`.

## 2026-11-14 - Target workstation receiver and isolate the output clock mismatch (r30)

- Category: Portable receiver deployment trial and diagnostic evidence.
- Owner direction: Use this workstation as receiver; correct model detections,
  not merely receiving video, remain the end goal.
- Decision/change: Staged r30 for `172.21.30.13:5010` via a separate deployment
  JSON, preserving candidate deployment and existing releases. r29/r30 original
  model/config asset hashes and launcher hashes are identical. No model/decode
  settings or runtime source changed in this step.
- Real network test: A temporary workstation UDP listener forwarded incoming
  packets to FFmpeg for simultaneous decoding/recording. A 20-second request
  exited after about 17 seconds with `output_timeout`: 305 source/boxdecode
  buffers, one encoder buffer, one output-sink buffer, one valid timed frame,
  zero RTP markers, zero workstation packets, and clean worker shutdown.
  FFmpeg could not infer dimensions because no video arrived.
- Root-cause evidence: Metadata time is approximately board monotonic uptime,
  not pipeline running time. The real UDP sink defaults to clock pacing, so
  the first future-dated packet waits. The unpaced diagnostic sink did not test
  this. A separate synthetic board `appsrc ! udpsink` test delivered within
  750 ms with pipeline-relative PTS and default pacing, but not with absolute
  PTS; disabling pacing also delivered. No camera/model was used in that test.
- Proposed next step: Seek approval to convert carried source timestamps to
  pipeline running time while preserving frame spacing, then repeat real
  receiving/decoding and compare host/board detections on the same images.
  This clock conversion has not yet been applied.
- Evidence: Board `/media/nvme/ic-terminal-verify-r30-receiver`, workstation
  `../IC_TERMINAL_TRAIN/verify-r30-receiver`, including reports, receiver and
  FFmpeg logs, release/deployment JSON and `sink-clock-test.json`.
- Affected pages: `openwiki/operations/{status,deployment,troubleshooting}.md`,
  `openwiki/architecture/runtime.md`. Probe sockets/FFmpeg are stopped; capture
  and inference are stopped. No firewall/IP edits. Shared SDK/fixed station
  untouched.

## 2026-11-14 - Convert source clocks and verify workstation decoding (r31)

- Category: Portable runtime clock repair, receiver validation, detection diagnosis.
- Decision/change: With approval, added `SourceClock` to portable support and
  used it at decoder output to map source metadata time to pipeline running
  time. The first decoded frame establishes one fixed offset; later intervals
  remain unchanged. Clock origins/offset/method are recorded in run health.
  Invalid clocks, range failures or epoch resets do not invent time or silently
  re-anchor. Source-clock conversion precedes the existing bounded,
  stream/frame-keyed cache. Native valid encoder PTS is still preserved.
  Real sink clock pacing remains enabled.
- Tests/deployment: Added 17 clock regression cases, including the actual probe
  with an uptime-sized source timestamp. Full suite: **111 passed**, two existing
  dependency deprecation warnings. Source compiles. Uploaded/hash-verified
  launcher copies and staged r31 for workstation 172.21.30.13:5010. Original
  model/config source asset hashes match r30; existing releases and candidate
  deployment were not changed. No inference/decode thresholds were changed.
- Receiving evidence: A 60-second request yielded 1,717 timed encoder frames,
  1,716 received complete RTP frames (24,869 packets), clean board exit and no
  reported board errors. Raw media verification passed; reconstructed H.264
  decodes all 1,716 frames at 1280x720. The initial FFmpeg recording harness had
  coarse-time-base warnings and required termination, leaving an incomplete MKV;
  it is not the accepted archive. A repeat with a 600-frame live decoder limit
  and 90 kHz time base decoded **600 live frames**, exited 0 without forced stop
  or repeated-DTS warnings. The 35-second board run yielded 969 timed encoder
  frames, 968 RTP markers and clean exit.
- Remaining shutdown issue: The repeated run's full packet capture contains an
  incomplete final frame; its media verifier correctly fails that check. The
  worker stops directly to NULL without draining EOS. Do not claim lossless
  finite-run shutdown or full readiness. Human-observed playback, recovery and
  all-day stability are also unverified.
- Detection evidence: First run produced 1,092 boxes / 1,776 source frames;
  285 heights are negative under the signed reading. On 20 deliberately selected
  board-empty returned frames, the original decoded FP32 ONNX export produced
  detections on 17 (121 boxes total). Its 174 same-named initializers shared with
  the six-head export are identical. Inference used ONNX Runtime CPU, RGB /255,
  1280-square padding 114, conf 0.05 and class-aware NMS IoU 0.7. This is not
  checkpoint execution, unbiased accuracy measurement, ground truth or exact
  uncompressed numerical parity. Compression and index-based pairing limit the
  comparison. Quantized/reference confidence scores are not directly equivalent.
- Evidence locations: Board `/media/nvme/ic-terminal-verify-r31-receiver{,-bounded}`;
  workstation `../IC_TERMINAL_TRAIN/verify-r31-receiver{,-bounded}`. Includes raw
  captures, run/media verification, reference-comparison JSON and images.
  `viewing-30fps.mp4` is a complete viewing-only copy with regenerated timing,
  not original timing evidence.
- Proposed next work: Validate the vendor canonical pre-decoded model/output
  path against these misses before changing thresholds or retraining; add
  graceful shutdown/drain separately. Correct detections remain unaccepted.
- Affected files/pages: `runtime/launcher/{support,worker}.py`,
  `tests/test_portable_timestamps.py`, `openwiki/architecture/runtime.md`,
  `openwiki/reference/tests.md`, `openwiki/operations/{status,deployment,troubleshooting}.md`.
  All temporary receivers/FFmpeg and board pipelines are stopped. Shared SDK
  and fixed station untouched.

## 2026-11-14 - Consult SiMa contracts and isolate vendor YOLO11 conversion failure

- Category: Vendor-reference comparison; diagnostic only.
- Sources: Official SiMa graph-surgery, `simaaiboxdecode`, and get-accuracy
  documentation linked from `https://docs.sima.ai/index.html`; the installed
  Admin-PC `model_to_pipeline/surgeons/surgeon_yolo11.py` and helpers.
- Findings: Official boxdecode documentation supports both pre-decoded `yolo`
  and cut-model `yolov8` paths; raw DFL is not inherently unsupported. The
  dedicated YOLO11 surgeon rewrites attention as well as final heads, so old
  custom decoded-model FPS results do not benchmark that complete vendor path.
- Trial/result: Ran the unmodified installed surgeon on our original 1280 FP32
  export in an isolated directory of the existing portable ModelSDK container.
  It failed before final-head surgery: attention nodes are not dependency-ordered.
  All tensor producers exist. Sorting in memory then exposed stale inferred
  shapes; sorting plus clearing `value_info` passed full ONNX validation and
  executed in ONNX Runtime. On `sample-300.png`, intermediate attention-rewrite
  differences from the original were at most 0.000274658 model pixels and
  7.152557e-7 class probability.
- Limitations: Only the intermediate attention rewrite and one saved image were
  checked. No completed canonical six-output graph, compiler/quantized comparison
  or board run resulted. This converter failure is not established as the cause
  of the currently deployed DFL detection defect. Training quality is unproven.
  The SDK1.7.0 accelerator example linked from current documentation needs API
  compatibility verification against our 2.0.0 compiler before use.
- Proposed next step: Seek approval for an isolated compatibility adapter that
  orders nodes and clears stale metadata before vendor simplification; leave
  installed vendor source unchanged, then finish numerical/board comparison.
- Evidence: `../IC_TERMINAL_TRAIN/vendor-reference-20261114/` and matching Admin
  job directory; `surgery.log`, `surgery.exit`, original intermediate ONNX and
  `ordering-diagnosis.json`. Diagnostic wrapper `run_surgeon.py` lives with the
  experiment evidence, not in board releases.
- Affected pages: `openwiki/architecture/inference-backends.md`,
  `openwiki/operations/status.md`. No runtime/config/deployment/training changes;
  no existing compile job, installed vendor source, shared SDK or fixed station
  modifications. Board pipelines were not started.

## 2026-11-14 - Add portable identical-input Python diagnostic; isolate native quantizer abort

- Category: Portable diagnostic tooling and hardware evidence, not deployment.
- Owner direction: Borrow the fixed station's execution architecture only;
  retain the portable IC/terminal model, four class IDs, geometry and decoding.
- Change: Added `tools/analysis/portable_tensor_parity.py` with prepare/board/
  compare commands, byte-fingerprinted inputs/outputs, pinned-release asset and
  compilation-lineage checks, portable NumPy 16-bin DFL/sigmoid decoding,
  class-aware NMS and per-head numerical reports. Uses existing installed SiMa
  APIs, not the fixed station's model, ROI, 640 grids or 4-channel decoder.
  Generated configs and flushed stage markers locate native failures. Output
  directories/reports must be new; live runtime and installed services untouched.
- Offline verification: Added 12 tests; full suite **123 passed**, two existing
  dependency deprecation warnings. CPU decoder matches original FP32 ONNX on
  three prepared inputs within 0.000244141 model pixels / 1.371664e-7 probability;
  reference detection counts 5, 3 and 2. Inputs: recent received sample plus two
  validation images absent from the compilation's calibration list.
- Hardware result: Direct API attempt aborted with exit 134/SIGABRT and native
  `stack smashing detected`. A second bounded, instrumented invocation located
  the abort at `SimaQuantTess.configure({})`, before constructing the model runner,
  loading its ELF or feeding any image. No hardware heads were returned; hardware
  parity and model accuracy remain unmeasured by this diagnostic. Generated
  config and environment details are preserved. Exact native crash cause is
  unknown; no claim of a resolution limit, class problem or bad weights is made.
- Proposed next step: Verify and use installed SDK software quantize/tessellate/
  detessellate/dequantize operations from compiler metadata, leaving only neural
  execution on the accelerator. Seek approval before changing the diagnostic
  route; do not keep retrying the native abort.
- Evidence: `../IC_TERMINAL_TRAIN/tensor-parity-20261114/` and portable board
  `/media/nvme/ic-terminal-tensor-parity-20261114/`, including identical-input
  archive/manifest, CPU reference report, logs/exit codes, quantizer config and
  installed-module/package inventory. Hardware services report active; no
  remaining diagnostic/capture/inference processes. No service restart or
  post-abort inference-health qualification performed.
- Affected files/pages: New diagnostic and `tests/test_portable_tensor_parity.py`,
  `openwiki/reference/{analysis-scripts,tests}.md`,
  `openwiki/architecture/inference-backends.md`, `openwiki/operations/status.md`.
  No release/model/calibration changes; shared SDK and fixed station untouched.

## 2026-11-14 - Software preparation returns meaningful compiled-model predictions

- Category: Portable identical-input hardware diagnostic; not live deployment.
- Decision/change: With approval, added compiler-derived software transform
  support to `tools/analysis/portable_tensor_parity.py`. Installed SDK software
  functions quantize/tessellate the input and unpack/detessellate/dequantize
  output; only `SimaModelRunner` executes on the accelerator. MPK plugin
  dependencies, head names, physical byte sizes, padding, rounding and tile
  flags are explicit, not guessed from a template or native parser config.
  Actual SDK input/output roundtrips run before inference. Software mode is the
  default; the known-crashing native mode is explicit reproduction only, never
  a silent fallback. No proprietary implementation was copied into the tool.
- Verification: Eight additional boundary/dispatch tests; full suite **131
  passed**, two existing dependency deprecation warnings. Native library code
  is not covered by those unit tests; the board preflight/execution supplies
  separate hardware evidence. Source compiles.
- Hardware result: Same r31 ELF/MPK and same three input tensors as the previous
  failed native attempt. Software-mode process exited 0 and returned all six
  expected float heads for every sample. Input archive/output checksums and
  ELF/MPK/raw ONNX compilation lineage pass. At confidence 0.05 / class-aware
  NMS IoU 0.7, FP32 versus compiled/Python counts are 5/9, 3/3 and 2/2.
  Maximum scores are 0.819/0.572, 0.928/0.762 and 0.904/0.802 respectively.
- Interpretation/limits: All eight reference boxes with FP32 score ≥0.3 have
  class-matched compiled counterparts at diagnostic score ≥0.05; greedy IoU
  overlaps are 0.855–0.965. This is three-example reference agreement, not
  ground-truth accuracy, general precision/recall, numerical equality, or
  production acceptance. Extra low-score predictions and confidence changes
  remain. Software transforms are slow: about 203–217 ms combined per input,
  excluding complete DFL/NMS; MLA calls measured 3.7–4.2 ms. No live fps claim.
- Static audit: Compiler input tessellation is `[1280,64,3]`; live preprocessing
  tile fields are 128×32, while the failed native quantizer generated 64×1.
  These differ, but plugin semantics/actual bytes have not established which
  live stage diverges. No tile settings were substituted into a release.
- Proposed next work: Compare live input/output intermediates against this
  working format and locate the first mismatch before repairing native
  preprocessing/unpacking/box decoding. Validate labelled accuracy and speed
  separately; do not retrain or lower thresholds merely to mask integration.
- Evidence: Existing local/board `tensor-parity-20261114` experiment directories:
  `board-software/{board.json,outputs.npz,software-transform-contract.json}`,
  `software.log`, `software.exit`, `comparison-software.json`,
  `reference-agreement.json`, `layout-audit.json`, and annotated comparison JPGs.
- Affected files/pages: Diagnostic, `tests/test_portable_tensor_parity.py`,
  `openwiki/reference/{analysis-scripts,tests}.md`,
  `openwiki/architecture/inference-backends.md`, `openwiki/operations/status.md`.
  No weights, class IDs, calibration, releases, installed vendor code, services,
  shared SDK or fixed-station files changed. No pipelines remain running.

## 2026-11-14 - Isolate both layout faults and deliver annotated native video (r32/r33)

- Category: Portable layout correction, config generation and end-to-end video.
- Evidence/first divergence: Added bounded native-buffer capture and controlled
  replay tools (`inspect_native_buffers.py`, `compare_native_buffers.py`). Native
  preprocessing tiles `[32,128,3]` reconstruct the companion untiled RGB exactly,
  but compiler input tiles `[1280,64,3]` reinterpret ~79.3% of bytes incorrectly.
  Replaying captured packed input through the same ELF reproduces captured MLA
  output byte-for-byte on two frames. Repacking only those same quantized pixels
  into compiler order restores predictions. This establishes an input layout
  fault without changing weights, calibration, class IDs or source pixel values.
- Output fault: Input-only r32 restores scores but same-frame native box sizes
  still differ from independent decoding. r33 additionally uses compiler output
  tiles `[2,160,64]`, `[1,80,64]`, `[1,40,64]` for boxes; logical tensor frame
  heights stay 160/80/40. Eight native detections on two r33 captures agree with
  software decoding within 0.473 pixels and 1.50e-8 score. MLA source caps and
  boxdecoder slice settings are changed together; class-head tiles remain as
  compiled. No threshold change (still diagnostic 0.05).
- Source repair/tests: `build_ic_terminal_configs.py` derives preprocessing and
  output tiles from MPK rather than inheriting a template or equating tiles with
  logical frames. Geometry guard checks each contract separately. CLI refuses
  missing tile metadata before output writes. Four regressions added/updated;
  full suite **135 passed** with two existing dependency deprecation warnings.
  Both new board-only probes and tooling compile; native capture/replay tests
  are hardware evidence, not unit coverage. Real regeneration reproduces r33
  layout values in `native-layout-generated-20261114`.
- Deployment/result: r32 and r33 staged from separate experimental candidates;
  old releases and legacy candidate directories were not overwritten. A
  45-second r33 native run to workstation 172.21.30.13:5010 admitted 1,267 timed
  frames and delivered 1,266 complete annotated RTP frames (18,307 packets),
  about **29.52 fps**. Board exited 0 without errors; bounded live FFmpeg decoded
  900 frames and exited 0. Raw RTP verification passed. Detection log: 6,431 boxes
  across 1,326 source frames, zero negative heights.
- Viewing/limits: FFplay preview was launched but required termination after the
  finite test; no human-observed visual/clean-player-shutdown claim. Annotated
  viewing-only MP4 contains 1,266 frames at 1280x720 with regenerated nominal
  30-fps timestamps. Native/Python agreement is not ground-truth accuracy; the
  diagnostic threshold, uncertain class quality, queued shutdown tail, recovery
  and all-day stability still require acceptance. Legacy candidates still have
  wrong tiles and must be regenerated before staging. No continuous service is
  left running.
- Evidence: Local/board native buffer captures for r31/r32/r33, controlled layout
  replay, `verify-r32-input-layout`, and `verify-r33-receiver` (board names use
  `ic-terminal-` prefix). Final clip:
  `../IC_TERMINAL_TRAIN/verify-r33-receiver/annotated-viewing-30fps.mp4`;
  still: `annotated-sample-600.png`; exact r33 decoder comparison:
  `native-buffers-r33-20261114/parity-summary.json`.
- Documentation: Updated quickstart, model-pipeline/inference-backends, status,
  deployment/troubleshooting, CLI/tests/analysis references. Corrected signed
  INT8 representable-range arithmetic in the inference page (no quantization
  parameter change). Shared SDK, vendor installation and fixed station untouched.

## 2026-11-14 - Add an external-reader debugging case study

- Category: Documentation only.
- Change: Added `PORTABLE_VISION_DEBUGGING_CASE_STUDY.md` and linked it from
  `README.md`. Explains the portable station from first principles, separates
  video timing from detection/layout faults, describes the fixed-station-inspired
  diagnostic and its native-helper failure, and presents the captured-buffer,
  byte-identical replay and repacking evidence behind r33.
- Scope: Includes the technically difficult logical-versus-physical geometry
  distinction, the corrected guard assumption, results, reproducibility links,
  and explicit limits on accuracy, timing and production acceptance. The vendor
  converter detour is distinguished from the proven deployment faults.
- Verification: Measurements checked against the recorded r33 status/evidence;
  local Markdown links checked. No runtime, deployment, board or model changes.

## 2026-11-14 - Repeatable r33 live viewing and first held-out labelled test

- Category: Host operator tooling, measured evaluation and handover corrections.
- Approval: Owner requested repeatable live-view start/stop and known-good/bad
  evaluation. No threshold selection, retraining or release promotion performed.
- Operator flow: Added `tools/portable_live_view.py`. Starts FFplay or bounded
  FFmpeg before pinned r33; finite duration, receiver-exit/Ctrl+C cleanup, exact
  remote PID identity checks before signalling. Refuses an occupied board, reused
  board output, changed deployment hash, or unbindable receiver address/port.
  Never kills another job, edits releases or installs/restarts services.
- Hardware results: 300-frame headless decode exit 0; windowed duration completion;
  programmatically requested Ctrl+C; immediate restart and 150-frame decode exit
  0. Final revised tool passed 60-frame decode exit 0. All board launchers exited
  0 without wrapper errors. FFplay clock advanced, but logged 96 dropped frames
  during overlapping host ONNX work; cause not isolated. No human-observed
  visual/zero-drop/latency acceptance or manual Q-key test is claimed.
- Labelled test: Added `tools/analysis/portable_labelled_eval.py`. All 31 v4-1280
  test images (62 boxes), source-hash train/val disjoint and no calibration name/
  model-image-hash overlap. Input/output/label fingerprints, class metadata and
  ELF/MPK lineage checked. Actual unchanged ELF ran via verified SDK software
  transforms, not native camera replay. Conf 0.05, NMS IoU 0.45/top100, match IoU
  0.50 fixed before scoring; class-aware one-to-one matching, not mAP.
- Results: Compiled TP/FP/FN 57/16/5, precision 78.1%, recall 91.9%; FP32
  57/20/5, precision 74.0%, recall 91.9%. Both fail the same five IC_Fail
  annotations on localisation, with nearby correct-class predictions. Compiled
  overlap 0.334–0.439, scores 0.386–0.811; other 11 unmatched predictions are
  <=0.180. Whether annotations or model localisation need correction is not
  established. No criteria were relaxed or confidence chosen from the test set.
- Handover correction: Dataset, raw COCO and training metadata show Terminal_Fail
  annotations (43 train, 11 val, 10 test), contradicting earlier no-examples notes.
  Updated quickstart/model guide/status and linked the case study to the audit.
  Earlier camera-transfer failure claims were confounded by pre-repair layouts.
- Limits/roadblock: Historical frames share three captures with training, use the
  older camera view and do not represent diverse new specimens. Current-camera
  controlled good/bad placement requires an operator and independently confirmed
  parts. It is not completed or replaced by this historical test. EOS/drain,
  desktop startup drops, recovery and soak acceptance remain separate work.
- Tests/docs: 18 operator/scoring regressions; full suite **153 passed**, two
  existing warnings. Both new scripts compile. Added live-view runbook and
  `openwiki/operations/r33-labelled-evaluation-20261114.md`; updated README,
  deployment, CLI, analysis/tests references and status. Markdown links checked.
- Evidence: `../IC_TERMINAL_TRAIN/acceptance-20261114/` (board matching
  `ic-terminal-acceptance-20261114`), including `evaluation-verified/evaluation.json`,
  all 31 comparison images and `IC-localisation-review.jpg`; live-flow runs in
  `live-view-20261114-*`. Final r33 inventory, every asset and launcher fingerprint
  rechecked unchanged. No shared SDK, fixed station, weights, calibration, pinned
  configs or vendor install changes. All test pipelines/players stopped; capture
  feed remains stopped.

## 2026-11-15 - First owner-confirmed good-part live trial finds clipping

- Category: Controlled physical-state measurement and operational handover.
- Specimen: Owner confirmed a good part with both IC and terminal present;
  expected IC_Pass/Terminal_pass. One specimen, no per-frame independent boxes.
- Initial run: 45-second requested native r33 stream, 1,266 admitted frames,
  launcher exit 0. Both expected labels in all 1,325 detection records; 19
  records include weak IC_Fail scores about 0.049–0.068. Little movement means
  this was not full-rotation qualification.
- Recorded moving-part run: Bounded FFmpeg decoded 1,800 frames, exit 0; board
  admitted 1,837 frames, launcher exit 0. Independent ~60-second archive output
  contains 1,825 H.264 1280x720 frames. Of 1,896 detection records, both pass
  labels appear in 1,283 (67.7%); 634 (33.4%) contain a fail label. Not precision/
  recall: correlated views, no independent box annotations, duplicate/background
  boxes, clipped components and unencoded pipeline tail. No frame-index pairing
  is asserted between archive and detections.
- Finding: Saved images directly show the PCB leaving the top/bottom field of
  view. Blue-PCB heuristic screening flags 44/183 sampled video frames near a
  vertical border, not an exact clipping rate. Earlier clearance measurements
  do not cover this placement. False IC_Fail labels also occur on plate glare
  and other PCB regions; maximum moving-run IC_Fail score 0.345.
- Probe/reference: Captured matching decoder/preproc/MLA/native-box frames 20/21
  in a separate bounded probe. Decoded actual MLA tensors with installed SDK
  software transforms; native retained predictions agree up to rounding. Native
  also includes a marginal 0.04871 score below the strict Python 0.05 filter.
  FP32 on the dequantized companion input misses the weak terminal on one view;
  compiled output adds weak fail boxes. This does not prove a new layout defect
  or establish the cause of all rotating-view misses.
- Action/limits: Asked owner to stop the plate, move the SAME good board inward
  and verify whole-board visibility before another good run or any bad-state
  test. Camera, threshold, model and ROI unchanged; no speculative retraining.
- Evidence: `../IC_TERMINAL_TRAIN/good-part-20261115-{live,probe,rotation}/`.
  Rotation capture includes `annotated-viewing.mp4`, original MKV, framing edge
  stills, class counts, metadata/logs and exact one-off recording recipe. Board
  outputs: `ic-terminal-live-good-part-20261115-live`,
  `ic-terminal-good-part-20261115`, and `ic-terminal-live-capture` under NVMe.
- Docs: Added `openwiki/operations/good-part-trial-20261115.md`; updated current
  status, quickstart, operator/model guides and the earlier labelled report's
  follow-up. No repository runtime/test code, release, services, shared SDK,
  weights, calibration, vendor install or fixed station changed. Existing 153
  test result remains the prior source baseline, not a new test run. All probe,
  inference and viewer processes stopped; capture feed remains stopped.

## 2026-11-15 - Leave continuous annotated preview running for framing review

- Category: Owner-requested portable operation, no code/config changes.
- Owner confirmed the good board was repositioned and requested continuous video.
  Started existing `portable_live_view.py` in the background against pinned r33,
  approved workstation 172.21.30.13:5010, with an explicit **14,400-second** bound.
  Q/player exit stops its own board launcher; otherwise stops around 09:54 UTC.
  No service installation, automatic restart or unbounded daemon was introduced.
- Verification: Board live/media-ready, first 3,543 then 6,907 valid timed frames,
  no errors. FFplay clock advances; dropped frames remain recorded, not accepted
  as zero-drop playback. A transient additional SSH command-channel EOF was
  followed by successful SFTP health and exact launcher-argv verification.
- State: **Preview intentionally left RUNNING**, host PID 31520, remote launcher
  2961476/worker 2961477 at observation. PIDs must be identity-checked before
  signalling. Capture feed remains stopped and cannot run concurrently.
- Evidence: `../IC_TERMINAL_TRAIN/continuous-view-20261115-055448/` and adjacent
  wrapper `.log`/`.session.json`; board output
  `/media/nvme/ic-terminal-live-continuous-view-20261115-055448`.
- Limits/docs: Owner is observing framing; repositioning has not yet been
  independently scored or certified unclipped. Updated live state/next actions
  in `openwiki/operations/status.md`. Weights, threshold, pinned assets, services,
  shared SDK, fixed station and vendor installation unchanged.

## 2026-11-15 - Establish warm live baseline on the repositioned good PCB

- Category: Read-only benchmarking and baseline protocol. Owner visually accepted
  framing and requested baseline → detection improvements → production video UI.
- Tool: Added `tools/analysis/benchmark_live_session.py`. Observes the existing
  native UDP stream over SFTP without starting/stopping a graph or viewer.
  Identity-checks worker/session, discovers Linux AT_CLKTCK, samples proc/health/
  viewer counters and saves original log byte windows. Exact encoder-admitted
  cohort is joined by session/stream/frame; missing/duplicate evidence is refused.
- Good-state window: 3,599 encoder frames in 120.04 s, **29.98 fps** (delivery-span
  30.00), all validly timed, zero encoder-ID gaps, 3,599 RTP marker increase,
  no board errors/non-ready samples or cache misses/evictions. Cohort IDs
  41,451–45,049 in the existing `3d046019-0531-4d13-afe9-78c1a214a924` session.
- Resource/timing baseline: Worker CPU 44.0% mean of one core; whole-board CPU
  3.91% mean across 16 logical CPUs. Worker RSS 62,616–62,624 KiB, end-minus-start
  0. Encoder delivery p99 151.6 ms, max 162.4 ms. FFplay reported 287 additional
  drops. Good average throughput is not uniform pacing, packet-loss evidence,
  exact rendered FPS, optical latency, full native-service memory or a soak test.
- Prediction baseline: Same admitted cohort has both expected pass classes in
  2,845 frames (79.0%), unwanted fail labels in 1,441 (40.0%), 92 empty results.
  Raw weak native values and duplicate candidates retained. These are class
  presence counts, not precision/recall or part rejection/acceptance rates.
- Next state: Requested a known-bad IC + good terminal PCB if available. Owner
  must stop plate for swap, retain inward placement/camera, and confirm actual
  condition/readiness before another window. No bad-state results or detector
  changes are claimed. Preview intentionally remains running.
- Verification: Added 16 benchmark tests; full suite **169 passed**, two existing
  warnings; observer compiles. Tests run outside measured window. Offline
  re-computation, including corrected leading-boundary gap handling, reproduces
  every original metric exactly. Original report/raw window retained unchanged.
- Evidence/docs: `../IC_TERMINAL_TRAIN/benchmark-20261115-good-inward-01/`, including
  samples, raw frame/detection windows, release inventory, `benchmark.json` and
  `analysis-validation.json`. Added baseline report/protocol; updated README,
  quickstart, live-view guide, status and analysis/CLI/tests references.
- Boundaries: No pinned release, model, threshold, calibration, ROI, runtime,
  installed service, shared SDK, vendor implementation or fixed station changed.
  No source-image clearance, independent boxes, accelerator utilization, power
  or optical-latency measurement was fabricated. Capture feed stays stopped.

## 2026-11-15 - Baseline case B: bad IC with good terminal

- Owner confirmed the swapped board state. Actual IC defect type was not given;
  no assumption that the IC is missing or has a specific solder fault.
- Ran the unchanged observer for 120 seconds on the existing warm r33 preview,
  expected IC_Fail/Terminal_pass. No graph restart or settings change.
- Pipeline: 3,603 frames in 120.05 s, **30.01 fps**; all timed, zero encoder-ID
  gaps/errors/non-ready samples/cache misses or evictions; RTP marker delta
  3,603. Encoder delivery p99/max 150.6/155.1 ms; FFplay drop delta **334**.
- Resources: Worker mean CPU 44.0% of one core, board mean 4.19% across 16 cores.
  Worker RSS 63,276–63,284 KiB, end-minus-start 0. Starting RSS is 660 KiB above
  A, cause unisolated; neither a proven leak nor all-day stability qualification.
- Predictions on exact admitted cohort: IC_Fail 93.6%, Terminal_pass 85.7%, both
  80.4%. Unexpected IC_Pass **403/3,603 (11.2%)**, maximum score 0.428;
  Terminal_Fail 4.2%; either unexpected 15.0%; empty 1.1%. Raw class presence,
  not false-acceptance rate, part verdicts or independent box precision/recall.
- Evidence: `../IC_TERMINAL_TRAIN/benchmark-20261115-bad-ic-good-terminal-01/`;
  encoder IDs 107,091–110,693 in the same session as A. Full report, 61 samples,
  raw windows and exact offline re-computation retained; observer fingerprint
  matches the current source. Existing **169-test** source baseline unchanged;
  no new unit run or source-code changes for this measurement.
- Documentation: Updated A/B baseline report, live status, quickstart and
  operator guide. Next requested state is a bad terminal, with IC condition,
  physical defect and readiness explicitly confirmed before measurement.
- Preview intentionally left running. No weights, threshold, calibration, ROI,
  pinned release, runtime, services, shared SDK, vendor code or fixed-station
  changes. Capture feed remains stopped.

## 2026-11-15 - Baseline case C and missing-only fault-scope clarification

- Owner confirmed **IC missing + terminal missing**, and explicitly clarified
  that all fault specimens involve missing components only. No solder,
  electrical or other defects are in scope. Legacy class names/IDs unchanged:
  IC_Fail/Terminal_Fail mean missing, pass labels mean present. This also resolves
  the previously unspecified physical fault in case B as IC missing.
- Ran the same read-only 120-second observer on the unchanged live r33 session.
  3,599 frames in 120.04 s, **29.98 fps**, all timed; zero encoder-ID gaps,
  errors/non-ready samples/cache misses or evictions; marker delta 3,599.
- Resources/pacing: Worker CPU 44.6% of one core, board 4.00% across 16 cores;
  worker RSS 63,388–63,396 KiB, +8 KiB in-window. Delivery p99/max
  151.1/154.6 ms; FFplay drop delta 276. No optical-latency/rendered-FPS or
  long-term leak qualification.
- Native class presence: IC_Fail **96.9%**, Terminal_Fail **84.7%**, both
  **83.9%**; unwanted IC_Pass **1,051/3,599 (29.2%)**, max score 0.413;
  Terminal_pass **0**; empty 1.9%. Multiple candidates/expected and unexpected
  labels can coexist. Not site-level precision/recall or false-acceptance rate.
- Evidence: `../IC_TERMINAL_TRAIN/benchmark-20261115-ic-missing-terminal-missing-01/`;
  exact cohort 146,830–150,428, same inventory/session as A/B. Full report,
  61 samples, raw windows, fault-scope confirmation and identical offline
  re-computation retained. No original baseline files overwritten.
- Baseline now covers the three available states; optional good-A drift repeat
  and IC-present/terminal-missing are not measured. Proposed next quality step:
  locate unwanted IC_Pass boxes against matched source/site labels and FP32 on
  the current both-missing board before threshold/ROI/training changes.
- Docs: Updated three-state benchmark report, status, quickstart, model/operator
  guides and external case study to record missing-only semantics. No repository
  runtime/test code change or new unit run; existing 169-test baseline unchanged.
  No model, calibration, ROI, threshold, pinned release, service, shared SDK,
  vendor code or fixed-station changes. Preview intentionally left running;
  capture feed stays stopped.

## 2026-11-15 - Matched false-IC-pass diagnosis; preview restored

- Owner approved pause/capture/restore on the both-missing board. Stopped only
  the identity-checked original launcher; its wrapper/player shut down normally.
  Captured 40 periodic IDs (40+29*i) across decoder/preproc/MLA/boxdecode,
  160 buffers with caps, stream IDs and hashes. No score-selected capture or
  throughput claim; no model/settings/release change.
- Extended `inspect_native_buffers.py` with bounded periodic selection, exact
  completeness/duplicate checks and launcher/buffer fingerprints; default 20/21
  and eight seconds retained. Added `live_false_pass_parity.py` for captured
  software decoding, native/Python/FP32 comparison, exact anchor recovery and
  visual crops. Compiler input inverse matches companion RGB on all 40 frames;
  FP32 raw/fused parity <=0.000244141 px / 1.45e-7 probability.
- Finding: Five native IC_Pass candidates (four on the black pin header, one on
  another small PCB component), all matching independently decoded compiled
  boxes. FP32 retains one; four lower same-anchor scores (0.014–0.041) rise above
  0.05 in compiled output (0.057–0.071). One shared false candidate rises from
  0.0795 to 0.1275. Wrong-object confusion plus compiled/quantized-network score
  changes is supported for these cases, not recurrence of byte-layout corruption.
- Limits: No independent site boxes, no labelled accuracy/false-acceptance metric.
  Sample maximum 0.128 does not cover baseline's stronger 0.413 case. Of 203
  strict-score native boxes, 198 match Python's retained set; five IC_Fail
  differences have matching raw candidate coordinates but different retention.
  Exact suppression cause/internal quantization layer cause not isolated.
- Roadblock/recovery: CPU packaging timeout after 39 archives; raw capture intact.
  Explicit resume recomputes and requires exact existing-array equality, rebuilds
  incomplete archives atomically. All 39 verified, last rebuilt; no new camera
  or neural execution. No repeated native-helper crash.
- Preview restored BEFORE packaging/reference work and left RUNNING from 08:27 UTC
  with four-hour bound to ~12:27 UTC. Host PID 38940; remote 2962180/2962181 at
  observation. New `continuous-view-20261115-after-parity.session.json`; earlier
  05:54 session is stopped. Live health checked with 68,955 timed frames/no errors.
- Evidence: `../IC_TERMINAL_TRAIN/live-false-pass-20261115/`; board
  `/media/nvme/ic-terminal-false-pass-20261115/`. Comparison JSON, raw-candidate
  review, same-anchor scores and labelled `false-pass-contact-sheet.jpg` retained.
- Tests/docs: Added 21 sampling/helper/resume tests; full suite **190 passed**,
  two existing warnings; scripts compile. Added diagnosis report, updated status
  and analysis/tests references. No threshold, ROI, weights, calibration, service,
  shared SDK, vendor install or fixed-station changes. Stronger-case capture and
  site/hard-negative validation are proposed, not applied fixes.

## 2026-11-15 - Offline PCB pose/site feasibility experiment; do not deploy

- Owner approved offline investigation on captured images. Added
  `tools/analysis/portable_site_experiment.py`: original-image blue substrate and
  header orientation, optional PCB-restricted ORB/RANSAC, provisional site mapping,
  source overlays/canonical contact sheets and counterfactual class/site gating.
  No neural inference, board connection, live graph interruption or model,
  threshold, ROI, calibration, release, service or shared SDK change.
- Hash-pinned 42 unannotated source images: 40 both-missing, two adjacent
  both-present from earlier framing. Third state's matched source absent;
  reference included in sample and site polygons provisional. No independent
  pose labels, held-out validation or precision/recall claim.
- Blue/header method: 36 candidate poses, six unknown; at least four wrong
  180-degree directions (243/272/678/1084), both-present views unknown. Scoped
  ORB/RANSAC: 13 plausible candidates including reference, 29 insufficient-support
  refusals. An initial unscoped reference locked onto stationary background;
  retained failed trial and restricted reference to PCB. Neither method deployed.
- Counterfactual filtering removes five unwanted IC_Pass boxes, but scoped ORB
  loses most missing-component boxes to unknown pose; fifth false pass also lies
  on an unknown frame. This is NOT successful inspection or improved recall.
  Colour method's wrong direction independently disqualifies it from production.
- Evidence: `../IC_TERMINAL_TRAIN/site-pose-experiment-20261115/`, hashed manifest,
  final method results, frozen source, visual review and `selected-review.jpg`.
  Exploratory trials retained, not overwritten. Independent site/pose labels and
  full-rotation source data for other states proposed before further selection.
- Added 12 CPU helper tests (rotation, ambiguity, coordinate mapping, class-site
  gate, unknown handling, reference feature restriction). Initial synthetic board
  exceeded area bound; corrected fixture size, not algorithm. Full suite **202
  passed**, two existing warnings; compile checks pass. Added experiment report
  and updated status, analysis/tests references. Preview untouched; live health
  was not rechecked during this offline experiment.

## 2026-11-15 - Draft PCB/site labels and editable owner review; correction

- Owner approved preparation from saved images. Built
  `../IC_TERMINAL_TRAIN/site-label-draft-20261115/review-pack-final/`: 42 exact
  source PNG copies, 126 provisional polygons, ordered PCB corners/direction,
  owner-confirmed presence states, LabelMe alternatives, seven raw/overlay pages
  and offline editable review HTML. 40 both-missing, two adjacent both-present;
  third state absent. ALL pending, zero independently approved, development-only.
- Added `prepare_portable_site_labels.py` with hash/ID/state/geometry checks before
  output creation; source byte-copy verification, explicit manual override/proposal
  provenance and refusal to reuse output. `portable_site_review.html` supports
  vertex dragging, high-priority selection, named review/flag, export/resume JSON
  and changed-source import refusal. Geometry edits reset review; no training
  export. LabelMe/browser edits do not silently synchronize.
- CORRECTION to previous experiment entry (left intact): enlarged individual
  source/canonical inspection disproves four reversal claims at 243/272/678/1084.
  Earlier contact-sheet interpretation was wrong. Frame 678 has a real partial-
  substrate fit, not a reversed board. Numerical candidate/unknown counts remain
  unchanged; no pose accuracy claim. Current report/status corrected. Original
  evidence kept; added visual-review-correction.json and corrected review image.
- Seven manual substrate proposals cover four refusals, two present views and
  partial-fit 678; three high-priority original machine proposals retained after
  orientation recheck. Remaining 32 use machine seeds. All site geometry is a
  provisional template projection needing independent review, not ground truth.
  First two assembly drafts marked superseded (first briefly applied wrong flips);
  only review-pack-final is current. No draft used for training or evaluation.
- Tests: 16 new helper tests; full suite **218 passed**, two existing warnings;
  Python compile and JavaScript syntax pass. Isolated headless Edge smoke exit 0,
  13 checks for load/navigation/review/reset/drag/import/resume. Test review was
  transient/reset; actual annotations stay 0/42 approved. Initial browser harness
  had a duplicate local variable; fixed harness, not an application failure.
- Added review guide and dataset START-HERE, updated analysis/tests/status docs,
  froze source and retained review evidence. No board connection/camera capture,
  preview interruption, model/threshold/calibration/release/runtime/shared SDK or
  fixed-station changes. Live status not rechecked; previous safety bound applies.
  Owner review comes next; full rotations/independent held-out captures still needed.

## 2026-11-15 - Validate owner corrections and test Canny PCB cropping

- Owner supplied repo-root `annotations.review-export.json` and requested Canny
  cropping before further preprocessing. Validated all 42 human-reviewed records
  against original draft/source hashes: 16 PCB quads and all 42 site records
  changed; identity/classes/state/splits/convex bounds preserved. Original export
  and source images unchanged; exact hashed snapshots retained. Original draft
  remains historical, not silently overwritten with approved geometry.
- Added `tools/analysis/portable_canny_crop.py`: grayscale/Gaussian/Canny L2,
  optional closing, nested contours, size/aspect/rectangularity/edge-density
  ranking, duplicate suppression and ambiguity/bounds checks. Review polygons
  used only after selection for coverage, never to choose the object. Crop the
  original colour image, preserve aspect in 512x256 preview; no neural input or
  physical tensor-layout change. Header/terminal orientation remains unresolved.
- Three initial threshold pairs with two closing passes exposed bracket selection.
  Tested closing 0/1 to check edge-joining effects; preserved all nine settings.
  Canonical default rerun matches initial counts, shares frozen script hash with
  ablations. Best developmental setting 30/90 + one close: 28/42 preserve reviewed
  PCB and both sites >=98% (also strict vertex containment); 14 wrong crops have
  zero target overlap, including both present views. No held-out accuracy claim.
  In successful crops median area is 2.73% of source (~97% excluded). Edge
  detection is useful but does not identify PCB versus rectangular hardware.
- Evidence: `../IC_TERMINAL_TRAIN/canny-crop-20261115-close{0,1}` and
  `...-close2-final`, initial run preserved. close1 `selected-review.jpg` shows
  success plus bracket failures; all edge images/crops/overlays/results retained.
  Snapshots and common script hash independently verified in analysis-validation.
- Added 17 helper tests; full suite **235 passed**, two existing warnings;
  compilation passes. Added experiment report, updated owner-review guide,
  status and analysis/tests references. No board connection, stream interruption,
  model inference/training, calibration/threshold/ROI/release/runtime/SDK/vendor
  or fixed-station change. Live health not checked. Canny + PCB identity evidence
  is proposed next; no unsafe crop selector deployed.

## 2026-11-15 - Colour-seed/local-Canny crop comparison; 42/42 development containment

- Owner approved combining Canny with PCB colour/appearance checks. Added
  `tools/analysis/portable_hybrid_crop.py`, reusing unchanged Canny review/crop
  helpers. Three fixed arms: colour-gated Canny candidates, grouped substrate
  seeds only, and grouped seeds with bounded local Canny extent. Pixel-only
  selectors; reviewed geometry never selects positive crops. No manual plate ROI.
- Fixed HSV 90..140/S>=65/V>=30, >=500 blue pixels/>=15% support, 15px closing
  groups, geometric plausibility/ambiguity gates. Edge extent within 32 source
  pixels of seed, includes seed corners, then 16px padding. Original colour
  pixels cropped/rescaled; semantic 180-degree orientation not solved.
- Same 42 owner-reviewed sources: colour gate preserves 28, rejects 14 as unknown
  (no bracket selections); grouped colour preserves 40 but clips terminal on
  both present views (frame20 terminal coverage 87.25%); hybrid preserves **42/42**
  PCB plus both target polygons, strict vertex containment also passes. Median
  crop area 2.95% (~97% source excluded). No per-frame tuning/label-based ROI.
- Zero crops on 42 explicitly synthetic board-removed controls for every arm.
  Review geometry used only to construct these negatives and evaluate positives;
  artificial controls do not establish real empty-plate false-positive rate.
  Development-only, mostly one both-missing capture; two adjacent present views,
  no third-state/new-session coverage. No detector precision/recall claim. Glare
  and in-crop header confusion remain; no inference performed.
- Evidence `../IC_TERMINAL_TRAIN/hybrid-crop-20261115/`: original/old-Canny/colour/
  hybrid comparison, all 42 crop sheets, masks/edges/overlays, policy and transforms,
  immutable review snapshot, frozen script/helper. Independently recomputed exact
  per-frame coverage/summaries and verified source/review/helper hashes.
- Added 14 helper tests; full suite **249 passed**, two existing warnings;
  compilation passes. Added report, updated baseline follow-up, status and
  analysis/tests references. No board access, stream interruption, model/training,
  calibration/threshold/ROI/runtime/release/SDK/vendor or fixed-station changes.
  Live health not rechecked. Proposed next: original-versus-crop detector diagnostic,
  then independent-state/session/negative/runtime qualification before deployment.

## 2026-11-15 - Original-versus-crop FP32 detection review saved for owner

- Owner approved detection comparison and requested a folder for personal review
  before deciding further improvements. Added `portable_crop_detection_review.py`;
  ran all 42 original frames plus exact native hybrid crops (84 ONNX inferences)
  with unchanged pinned fused FP32 reference on CPU. Not compiled/native board
  output or quantized-input parity. No speculative enhancement/training change.
- Fixed confidence .05, class-aware NMS .45, top100; RGB /255, centred 1280-square
  padding 114 on both branches. Revalidated source/review/model/class/shape hashes,
  regenerated all native crop pixels/transforms exactly. Crop input is not the
  512x256 viewing preview. No site gate, strongest-per-site filter or verdict.
- Saved `../IC_TERMINAL_TRAIN/detection-review-20261115/`: index.html gallery,
  42 side-by-side comparisons, 42 full-frame originals, 42 original predictions
  projected into equal-ROI zooms, 42 cropped detection images, seven sheets and
  START-HERE. Full originals/JSON preserve off-ROI boxes; empty cases included.
- Observed original inputs 200 boxes (151/2/5/42 by fixed class order); crop inputs
  34, all Terminal_Fail. No IC detections survive fixed settings on cropped inputs;
  21/42 empty including both present views. Cleaner picture does NOT demonstrate
  improved inspection. Scale/orientation/context effects not isolated; no precision/
  recall or production verdict claimed. Await owner image-ID/observation feedback.
- Retained all 84 raw predictions, every score/box, actual resize/padding/tensor
  hashes and crop/source polygon transforms, exact review/hybrid snapshots and
  frozen source. Independently re-decoded saved raw outputs exactly, verified
  model/raw/review/script fingerprints, all 168 annotated JPGs and gallery links.
- Added nine CPU helper tests; full suite **258 passed**, two existing warnings;
  compilation passes. Added report, updated hybrid follow-up/status/analysis/tests
  docs. No board access, stream interruption, model/calibration/threshold tuning,
  training/runtime/release/SDK/vendor or fixed-station change. Live health not
  checked; previous preview duration applies. No further changes before owner review.

## 2026-11-15 - Cropped four-class training-box proposals for manual review

- Owner requested annotations for manual review, then training. Added
  `prepare_cropped_training_review.py` and `cropped_training_review.html`; created
  `../IC_TERMINAL_TRAIN/cropped-training-label-review-20261115/` with 42 exact
  native crop PNGs, 84 new pending box proposals, seven review sheets, per-image
  raw/proposal comparisons and editable rectangle review. No training started.
- Seeds are AABBs of projected OWNER-reviewed site polygons, not failed model
  predictions. Previous site approvals reset: body/pins versus empty footprint/
  pads is a new training-box review. Fixed missing-only IDs preserved. Box/crop
  bounds, source/class identity and exact crop regeneration verified before output.
- Browser supports axis-aligned resize/move, hide overlays, review/flag/notes,
  pending reset after edits and export/resume. Export filename deliberately
  `cropped-training.review-export.json`, distinct from old source-site export.
  No autosave; user must export. Changed identity/state/class imports refused.
- Coverage 40/2/2/40 by fixed class order; 40 correlated both-missing images and
  two adjacent both-present. Third state/new-session and present-class coverage
  must be filled before credible four-class retraining/evaluation. No random
  adjacent-frame split, final train/val/test split or data.yaml. Provisional
  `draft-yolo-DO-NOT-TRAIN` files are NOT updated by browser edits; regenerate
  final labels from returned review export. Pack/rows remain training-ineligible.
- Froze builder/template and source/hybrid snapshots, verified exact crop/review
  hashes and all 84 pixel↔YOLO round trips within 1e-5 pixels. Actual approved
  count remains zero. Isolated Edge smoke 14 checks, exit 0; synthetic review
  actions remain in memory, not dataset files. Python compile/JS syntax pass.
- Added 11 helper tests; full suite **269 passed**, two existing warnings. Added
  review guide/START-HERE; updated model pipeline/status/analysis/tests docs.
  No board access, stream interruption, weights/calibration/threshold/ROI/runtime/
  release/SDK/vendor or fixed-station changes. Live health not rechecked.
  Next: owner's new box review, coverage completion and explicit dataset finalization.

## 2026-11-15 - First crop-box batch accepted; historical TRAIN-only batch02 prepared

- Owner confirmed all first-batch boxes accurate and requested new classes plus
  augmentation/preprocessing discussion. Recorded unchanged-draft verbal approval
  against its hash in first-batch-owner-confirmation.json, not a fabricated browser
  export. Original draft untouched; if geometry edited, actual export still needed.
- Added `prepare_next_training_batch.py`. Historical v4-1280 TRAIN-only present/
  absent runs: verifies image/label/manifest hashes, scale-1 constant padding and
  class pairs; strips exactly 280 top/bottom rows, no resize. Holds val/test files
  and membership untouched; source/model exact holdout-hash overlap guarded.
- Frozen hybrid policy accepts 76/84 training candidates, eight unknown sources
  quarantined with audit/visuals rather than forced crops. Selected 32 both-present
  and 16 IC-missing/terminal-present via deterministic farthest orbit-angle quotas;
  both span all 12 thirty-degree bins (max gaps 33.88/31.86 degrees). No label-
  coverage/model-score selection; historical labels become pending projected boxes.
- New `../IC_TERMINAL_TRAIN/cropped-training-label-review-batch02-20261115/`:
  48 crops/96 boxes, eight sheets, source split/frame/hash/selection provenance,
  exact provisional YOLO, frozen full helper group and editable HTML. Export
  filename `cropped-training-batch02.review-export.json`; all new labels pending.
  New counts 16/48/32/0; potential combined 56/50/34/40, 90 originals/180 boxes.
  Older camera and shared-recording correlation remain; not independent validation.
- Explained preprocessing/augmentation ordering: freeze serving-compatible crop/
  resize/normalization, split original captures/specimens first, stochastic label-
  aware augmentation TRAIN ONLY. Rotation including 180/modest photometric/blur/
  scale are proposals, not applied/configured. No untested CLAHE/sharpening, YAML,
  final split or training. Old trainer still points to v4 and is not launched.
- 10 new helper tests; full suite **279 passed**, two existing warnings; compile
  and JS syntax pass. Isolated browser smoke 14 checks; initial harness tried to
  'change' class2 to class2, corrected harness to an actual different ID. App import
  guard works. Verified all image hashes/labels/train membership and no crop
  duplicates across batches; new real annotations remain pending.
- Added batch/augmentation guide and dataset START-HERE, updated pipeline/status/
  analysis/tests references. Ready table now explicitly says preview LAST VERIFIED:
  finite bound may have expired, offline work never rechecked live state. No board
  access, stream interruption, weights/calibration/runtime/release/SDK/vendor or
  fixed-station changes. Next: batch02 review, independent capture coverage and
  explicit candidate dataset/augmentation/training configuration.

## 2026-11-15 - Reviewed train-only preprocessing and augmentation prepared; no training

- Owner returned cropped-training-batch02.review-export.json and requested
  preprocessing/augmentation before training. Validated all 48 reviewed records,
  source/state/class/bounds/identity; used corrected IC box hist-absent_006413,
  not stale draft YOLO. First42 approval remains exact-hash-bound conversation
  provenance; no invented browser review. Originals/review files unchanged.
- Added prepare_augmented_training_pool.py. All 90 current development parents
  assigned TRAIN BEFORE derivation; no random neighbouring split, val/test folders,
  train-as-val alias or final YAML. New output
  ../IC_TERMINAL_TRAIN/augmented-training-pool-20261115/: 450 lossless 1280 PNGs,
  450 matching labels/900 boxes, native parent copies, 90 preview strips/gallery,
  lineage/transforms/seeds/policy/hash manifest and exact review provenance.
- Frozen deterministic resize/pad114, uint8 PNG (loader RGB/255 once). Four extras:
  180°, contrast .90–1.10/brightness ±10, rotation ±8°/occupancy .85–.98 with whole
  image contained, 3x3 sigma .6–1.0 blur AFTER model resize. Fixed seed20261115 per
  parent/variant. Labels move with pixels; clipping/drop refused. No CLAHE,
  sharpening, hue/mirrors/mosaic. Recommended loader flags disable double online
  augmentation, but no model/epochs/device/run selected.
- Base class counts56/50/34/40; augmented280/250/170/200. 450 views are NOT 450
  independent specimens. Training_ready=false: need separate validation/evaluation
  plan (preferred new current-camera captures; reserved historical val only a
  limited reviewed diagnostic). Current r33 graph still full-frame, not crop serving.
  No optimizer/training started or fake validation fabricated.
- Added14 helper tests; full suite **293 passed**, two existing warnings;
  compilation passes. Independently regenerated ALL450 image pixels and labels
  exactly, checked900 YOLO round trips within1e-5 pixels, source/review/script hashes
  and gallery links. Frozen source helpers retained. Added guide/START-HERE,
  updated pipeline/batch02/status/analysis/tests references. No board connection,
  stream interruption, weights/calibration/compilation/runtime/release/SDK/vendor
  or fixed-station changes. Live health not rechecked.

## 2026-11-15 — Train-now crop candidate compiled; offline tensor configs

- Owner-authorized separate candidate `crop-candidate-20261115-164105` completed
  100 epochs, final-checkpoint export and three-state raw/fused FP32 parity.
  Validation remains deferred; CSV val-loss placeholders are not evaluation.
- Compilation finished18:04:22 UTC, about15m25s after17:48:57 start. Confirmed
  ModelSDK2.0.0/gen2 MLA INT8,90 unaugmented TRAIN calibration views, raw-ONNX
  checksum in MPK, and locally verified downloaded ELF/MPK SHA-256.
- Generated offline board-configs from new MPK, not inherited quantization/tiles:
  input1280/64/3, DFL64/class4, grids160/80/40, physical output2,688,000 bytes;
  sigmoid enabled for raw logits. Full-frame preproc template is NOT crop serving:
  frozen hybrid crop/pad114 integration, native parity, throughput and held-out
  evaluation still required. No deployment, board access or shared SDK changes.

## 2026-11-15 — Reserved historical crop validation review pack

- Added `prepare_cropped_validation_review.py`: VAL-only hash/split guards,
  verified historical unpadding, frozen hybrid crop, deterministic resize/pad114;
  no augmentation, test access, prediction labels or inherited review approval.
- New `cropped-validation-review-20261115/`: all32 reserved sources audited,
  31 review crops/62 pending boxes (22/20/9/11), editable/exportable browser review,
  static sheets, deterministic model-input proposals, draft labels and provenance.
  `val-present_003540` unknown (no plausible grouped substrate) retained in
  quarantine/coverage denominator. Shared recordings limit independence.
- Five new tests passed; regenerated all31 model inputs pixel-exact and all62
  model labels exact. Human export/correction propagation required before scoring;
  no evaluation-ready YAML, evaluation claim, deployment or board access.

## 2026-11-15 — Reviewed historical crop validation FP32 comparison

- Validated owner export:31 reviewed images/62 boxes, reviewer H, no corrections;
  pack identity, source hashes/classes/state/split/transforms and geometry verified.
- Added `evaluate_cropped_validation.py` and fixed-settings final report/gallery
  `crop-validation-evaluation-20261115-final/`: new FP32 59TP/3FP/3FN, conditional
  precision/recall95.16%; old model on SAME CROPS0TP/6FP/62FN, not full-frame r33.
  All three unmatched predictions are correct Terminal_Fail class with IoU
  .4611/.4811/.4048 below.50; no wrong present-class predictions on reviewed
  missing states at diagnostic.05 confidence/.45 NMS/top100. No threshold tuning.
- Coverage31/32, unknown retained separately; historical shared recordings are
  correlated, NOT independent-session production accuracy or compiled parity.
  Raw outputs, review snapshot, per-image errors and comparisons preserved.
  Eight focused tests passed. No test images, board access or deployment.

## 2026-11-15 — Three terminal localization failures visually inspected

- Opened all three full comparison images (005001/005274/005728). New predictions
  cover the empty terminal footprint with tighter boxes than reviewed labels,
  which extend into surrounding PCB/background. Correct Terminal_Fail class,
  scores.928/.974/.914, contained-box IoUs.4611/.4811/.4048. Evidence supports
  box-extent mismatch, not observed missing-state/wrong-object confusion in these
  three cases; not a general accuracy claim. Kept approved labels, fixed IoU,
  strict59TP/3FP/3FN metrics and model unchanged. No board access/deployment.

## 2026-11-16 — Actual MLA replay and bounded crop-aware streaming diagnostic

- Actual portable-board replay of31 reviewed1280 inputs via installed software
  transforms completed exit0 in isolated fixture, hashes/lineage checked.
  FP3259TP/3FP/3FN vs compiled58TP/3FP/4FN;61/62 retained boxes agree at
  IoU>=.8023. One Terminal_Fail confidence drops.5326→maximum.016625;
  IC_Pass localization crosses below.50 IoU; another terminal crosses above.
  Historical correlated diagnostics, not independent production accuracy.
- Added separate crop_stream/crop_geometry portable diagnostic and receiver-first
  bounded controller. One RTSP decoder → frozen crop → pad114/RGB/255 → MLA →
  inverse box coordinates → software overlay/encoder/RTP. No r33/SDK/vendor edits.
  First failure (OpenCV4.6 missing batched NMS) retained; per-class NMS corrected.
  Second worker completed70 frames,69 located/1 unknown at1.51fps, not30fps.
- Network receive FAILED: PC now172.21.30.230, approved target still13:5010.
  Retargeting awaits owner approval; local-address preflight added. OpenH264
  warning also remains to investigate; no end-to-end video acceptance claimed.
  Live extra present predictions remain in logs, not claimed solved.
- Three matched snapshots model pixels reconstructed exactly from board quads;
  cross-platform corner ordering can reverse180° (known unresolved semantics).
  Six geometry/NMS tests added; full suite307 passed, two existing warnings.
  Both bounded trials stopped, no remaining graphs at final inspection. Source,
  tensors, logs, failures and reports preserved; see crop-stream diagnostic guide.

## 2026-11-16 — Approved receiver retarget and bounded crop video delivery

- Owner explicitly approved172.21.30.230:5010. Updated diagnostic allowlist with
  endpoint guard test, retaining local-IP preflight and r33 isolation.
- Preserved retries03–06. Added explicit appsrc buffer duration, encoded-file tap
  and counters; lower FFmpeg probing limits alone did not resolve prior failure.
  Synthetic file decode and10-frame network decode succeeded. No isolated cause
  claimed; OpenH264 warning appears even in successful synthetic trials.
- Trial07:43 camera frames processed/encoded,41 crops located/2 unknown,
  31.35s,1.37FPS.30 received annotated frames remuxed and independently decoded
  with exit0; checked rotated boxes/UNKNOWN on received images. Not30fps,
  production accuracy, physical latency or proof of zero live network loss.
- Successful clip and failed trials retained. All jobs stopped;08:31 UTC board
  inspection found no graphs. r33/services/SDK/fixed station untouched.
  Later counter handles RTP buffer lists (not re-exercised on board).308 tests
  passed, two existing warnings. Updated status and diagnostic report.

## 2026-11-16 — Smooth compressed video and independent latest-frame inference

- Owner selected multirate video first, favourable-region inspection later, then
  profiling/optimization. Added separate latest_frame_stream/graph diagnostic and
  receiver-first host controller/browser snapshot panel. Original H264 passthrough
  runs independently of single hardware-decoder/latest-only appsink MLA branch.
  No tracking, region gate, threshold change or stale boxes on current video.
- Result JPEGs immutable; panel labels snapshot timestamps/received age/processing,
  UNKNOWN/STALE/DISCONNECTED. Bounds20–300s, occupied-board refusal, approved local
  receiver preflight, owned-child cleanup and source/evidence snapshots retained.
- Headless trial:70 inspections,68 located/2 unknown,1.69 inference FPS;1184 video
  access units in41.48s.600 received720p frames advertise30FPS; independent decode
  exit0 with one null-mux DTS warning. No zero-loss/latency/accuracy claim.
- Launched separate180s FFplay+loopback-browser owner view; panel HTTP200 and
  updating snapshots checked. Verify expiry before new graphs. r33/SDK/vendor/
  services/fixed station untouched. Five new graph tests;313 total passed,
  two existing warnings. Updated architecture/status and new operations guide.

## 2026-11-16 — Latest-frame owner window completed

-180s owner-view bound completed cleanly:324 inspections,316 located/8 unknown,
  1.79 inference FPS,5384 forwarded access units/29.67FPS including startup and
  shutdown, no worker errors. Receiver closed by owned controller. These are
  forwarding/inference measurements, not optical/render latency or accuracy.
  Updated live status and experiment report; no further camera graph started.

## 2026-11-16 — Frozen latest-frame fallback package

- Owner requested fallback before experimentation. Created selected package
  `fallbacks/latest-frame-20261116-v2/` and ZIP/SHA sidecar (~31.4MB,22 files).
  Runtime/helpers from actual successful180s source snapshot; frozen ELF/MPK,
  reference ONNX, compiler lineage, host controller/panel and evidence included.
  Credentials and installed dependency environments intentionally excluded.
- Added package builder and local `--fixture`/external `--credentials` support:
  fallback uploads its own assets, not mutable old board fixture. Verifies before
  launching, unique job/output dirs, bound180s default. Bytecode disabled in wrapper
  to preserve package integrity; preliminary v1 retained, v2 is selected.
- File SHA/ZIP CRC, clean temporary extraction/help/reverification and PowerShell
  syntax passed. Four fixture guard tests added; full suite317 passed, two existing
  warnings. No board contact, restart, SDK/vendor/fixed-station changes; new restore
  upload path not end-to-end exercised during packaging. Added restore guide/status.

## 2026-11-16 — Profiled lossless latest-frame performance candidate

- Owner requested profiling/optimization now; added stage timing and opt-in
  --optimized/--fast-transforms without touching frozen fallback/default behaviour.
  Baseline:DFL149.68ms,quant/pack109.22ms,RGB100.25ms,unpack96.78ms,MLA3.75ms.
- Added portable exact256-level OpenCV LUT quant/dequant, explicit MPK layout
  packing/inverses, class-score-first sparse bbox/DFL decode and chosen crop method
  only. No threshold/model/calibration/ROI/label/SDK/vendor/fixed-station changes.
  Startup checks actual SDK packed bytes/six tensors and reference predictions;
  unsupported layout/parity fails before camera use. Candidate is opt-in.
- Sequential bounded trials1.65→2.52→2.98→5.28→7.80 inspections/s; final60s gives
  7.99/s (488 processed,479 located/9 unknown),~4.85x baseline including overhead.
  Not identical-frame live trials. Video independent:600 received720p frames
  independently decoded, advertised30FPS; board forwarding28.73FPS including
  startup/shutdown. Null mux two repeated-DTS warnings; no no-loss/latency claim.
-31 reviewed replay retained classes/scores/boxes exact after reconstructing INT8
  raw and requiring all dequantized heads exact;32 VAL crop records exact including
  unknown. Nine helper tests; suite326 passed/two existing warnings. Frozen v2
 22-file integrity passed unchanged. Raw evidence/source snapshots/profiling and
  parity reports preserved. No quality generalization or compiled confidence fix.
- Final jobs stopped,09:43 UTC process check clear. Updated status/runtime/report;
  favourable-region gate remains disabled. Main remaining stage costs include
  crop42ms and JPEG31ms. No services or pinned r33 changes.







## 2026-11-16 — Actual MLA replay of crop validation candidate

- Added isolated replay driver and compiled evaluator. Portable preflight found
  no active graphs; uploaded new31-input fixture, no camera use or live mutation.
  SDK software quantize/tile/unpack plus actual MLA completed exit0; verified
  downloaded outputs/inputs and ELF/MPK hashes. Evidence compiled-crop-replay-
  20261116 and compiled-crop-evaluation-20261116 (gallery/raw heads/per-image results).
- Fixed diagnostic settings: compiled58TP/3FP/4FN versus FP3259/3/3.61 of62 FP32
  retained boxes matched at IoU>=.8023. IC localization003712 crosses belowIoU.50;
  terminal005001 crosses above; terminal005456 score.5326 FP32 versus compiled
  class maximum.016625 loses detection. No threshold changes or wrong-present
  predictions on reviewed missing states. Historical data, coverage31/32; NOT
  production acceptance or exact quantized equivalence.
- Mean MLA3.80ms; software transforms106.36/98.64ms are diagnostic, NOT serving
  throughput. Crop/pad114 integration, reverse mapping, native parity and stream
  tests still required. No live release/service/vendor/shared-SDK changes.




## 2026-11-15 — Validation review export verified

- Root `cropped-validation.review-export.json` SHA-256
  `a70a1cec9d1199099d4334fa45a6ddaea64da8b736be288d505cdb8c6d351190`
  passes pack identity, class/target, reviewed status, geometry, split/transform
  and source image hash checks. All31 records/62 boxes reviewed by H; no box
  corrections. Wrote pack `review-validation.json`; original export unchanged.
- One unknown crop remains quarantined/in coverage denominator. Evaluation has
  not run; historical recording correlation still limits independence. No board
  access or deployment.

## 2026-11-16 — Optimized owner live viewing window

- Owner requested streaming before deciding next work. Started optimized candidate
  for300s at09:53 UTC, approved172.21.30.230:5010; FFplay plus loopback snapshot
  panel opened. Verified running/HTTP200, snapshot58 located,112.8ms processing.
  Session/evidence `optimized-live-view-20261116-095325`, bound~09:58 UTC.
  Diagnostic only; fallback/r33 unchanged. Await owner feedback before experiments.

## 2026-11-16 — Owner-approved optimized fallback packaged

- New selected `fallbacks/latest-frame-optimized-20261116-v2` ZIP/SHA,24 files,
  31.4MB; actual owner-view runtime/helpers, model, references, launcher/UI/evidence.
  Launcher explicitly restores optimized and fast-transform flags. Earlier slower
  fallback v2 untouched/reverified; preliminary optimized v1 retained, use v2.
- Packaging discovered host Windows PermissionError replacing JSON during browser
  reads. Host controller had closed receiver/server; board completed300s bound.
  Recovered actual worker/log/frame evidence read-only; stale session not rewritten.
  Board2820 loops,1928 located/892 unknown,29.77 forwarded video FPS; no errors.
  High loop rate includes unknowns, not new model quality/performance acceptance.
- Added host retry/keep-old-result behaviour and immutable image download reuse;
  no model, board pipeline, thresholds, crop or SDK changes. Two tests added,
  suite328 passed/two existing warnings. ZIP CRC,24-file hashes, fresh extraction/
  help/reverification, optimized flags and PowerShell parse passed. Credentials
  excluded. No new stream; retry fix not newly end-to-end tested. Updated docs.

## 2026-11-16 — Browser coordinate publication and bounded PCB pose reuse

- Owner requested steps1/2 only: no favourable-region gate or C++ fusion. Board
  coordinate mode skips drawing/JPEG generation. Latest-only asynchronous publisher
  hashes immutable NV12 and sends JSON5012; original camera RTP5010 unchanged.
  Host PyAV decoder/browser renders boxes only on exact pixel-hash-matched frames,
  separate live pane; stale/unmatched/disconnected clears canvas. Decoder has no
  valid Gst PTS: no frame-order/timestamp guess. New separate host dependency env.
- LK forward/backward + RANSAC pose candidate, confidence/geometry/blue evidence
  gates, max0.5s full refresh and immediate re-localization on failure/stale step.
  Fresh model inference per accepted crop. Tracking is not component-state proof.
- Coordinate-only9.33/s; tracking12.22/s; final60s12.59/s (769 loops,761 located/
  8 unknown).652 tracked/117 full attempts, one confidence failure recovered.
 769/769 metadata messages hash-matched source frames; host decode30.01FPS, zero
  final RTP gaps, p99 arrival69ms/max703ms. Not zero jitter/latency/soak acceptance.
-31 host raw audits/25 tracked preserve fresh-selector seed coverage>=.99999997;
  exact raw hashes verified. Geometry diagnostic, NOT human site/accuracy evidence.
  Headless Edge screenshot confirms canvas overlays.12 tests added, suite340 passed,
  two existing warnings. Both fallback22/24-file hashes pass unchanged.
- Bounded runs stopped;10:33 UTC board process check clear. No model/calibration/
  threshold/r33/service/vendor/SDK/fixed-station changes. Tracking crop changes
  still require independent quality/stress validation. Updated runtime/status/
  analysis docs and full experiment report; source/worker/receiver/audits retained.

## 2026-11-16 — Coordinate/pose owner viewer opened

- Owner requested detection/streaming viewer. Opened300s browser session at10:43
  UTC (~10:48 bound), `coordinate-owner-view-20261116-104346`, approved current
  receiver5010 video/5012 coordinates, pose reuse enabled. Verified running and
  matched frame4/located at loopback panel62098. Live and inspected panes remain
  explicit; no stale live overlays. Fallback/r33 unchanged; await owner feedback.

## 2026-11-16 — Latest coordinate/browser/pose fallback packaged

- Owner accepted current viewer and requested latest fallback before C++ fusion.
  Frozen `fallbacks/coordinate-browser-pose-20261116-v1` ZIP/SHA,119.6MB/40 files;
  exact accepted source snapshot and completed proof match. Compiled model, UI,
  tracking, async coordinates, references/evidence and9 pinned offline host wheels.
- Added dedicated package builder, setup/run wrappers and explicit fixture/external
  credentials options to host controller. Defaults restore pose/optimized coordinate
  mode; no mutable old board fixture or experiment source dependency. Credentials,
  Python interpreter and installed board SDK intentionally not bundled.
- ZIP integrity and fresh extraction verified; isolated venv restored every host
  dependency OFFLINE from bundled wheels, H264 codec loaded, help/reverification and
  PowerShell parsers passed. Suite340 tests/two existing warnings. Older22/24-file
  fallbacks reverified unchanged. No new board stream/deployment during packaging.
- Existing owner view completed300s naturally:4077 loops/3862 located/215 unknown,
  3317 tracked/760 full,13 confidence re-localizations, no worker errors. Packaged
  session is state-at-packaging; no retrospective rewrite. No new accuracy claim.
  Updated index/status/restore guide. C++ fusion remains next separate experiment.

## 2026-11-16 — C++ fused input prototype and measured modest live gain

- Owner authorized input fusion first. Added portable-only C ABI/ctypes kernel:
  BGR→exact256-level quantization LUT→compiler tiles, one output buffer; no model,
  numerical policy, resize/pad, calibration or threshold changes. Explicit SHA,
  ABI/layout/native kernel and startup byte parity required; no silent fallback.
- Built with board g++12.2/AArch64/OpenMP in new work dirs, no SDK/vendor installs.
  Scalar/NEON prototypes were slower (retained/rejected); OpenMP8 won. Interleaved
  warmed benchmark8.70→3.24ms,16 workers3.09ms not selected for tiny extra gain.
  Six kernels match ALL31 prepared inputs; all256/random patterns and poisoned
  buffer margins/invalid byte-length guards pass. Compile/source/library retained.
- Sequential bounded live trials input8.92→3.81ms, total12.44→13.34 inspections/s,
  ~7.3% observed; not controlled identical frames or guaranteed production uplift.
  Host decode~30FPS,549/549 exact frame matches, no RTP gaps/worker errors. No
  independent accuracy, zero-jitter, optical latency or soak claim.
- Six host guard tests; suite346 passed/two existing warnings. Frozen40/24-file
  fallbacks verified unchanged. C++ remains opt-in/not packaged; package builder
  refuses native proof until native artifacts/flags are explicitly supported.
  No r33/service/fixed-station changes. Both jobs stopped;11:15 UTC check clear.
  Updated runtime/status/reference/native README and detailed performance report.

## 2026-11-16 — PCB feature seeding ROI reaches target inspection cadence

- Owner requested tracker optimization. Added opt-in --feature-roi: SAME full
  polygon mask/80 corners/.01 quality/3px min distance/block3, compute only seed
  bounds+8px context, convert points back globally. No favourable-region gating,
  refresh relaxation, corner reduction, model/threshold/calibration changes.
- Camera-free board31-frame hash-checked audit returns EXACT ordered feature
  coordinates; seeding81.05→2.46ms interleaved. New live pose metadata separates
  full localization/feature times. Live seeding93.03→4.31ms; full localization
  remains~42ms. Seven tests added for parity/rotations/edges/featureless/removal/CLI.
- Both live arms use same prior C++8 kernel: baseline13.22 model calls/s; ROI40s
  15.65 processed/15.31 actual model calls/s,14 unknown. Final60s974 model calls/
  61.05s=15.95/s,0 unknown,110 full/864 tracked, one confidence fallback, no errors.
  Host decode30.03FPS,974/974 metadata hashes matched,0 final RTP sequence gaps;
  p99 gap63ms/max610ms, not zero jitter/optical latency/soak acceptance.
- Final39 raw host audits: hash and ordered features exact; fresh-selector seed
  coverage>=.999999956, geometry diagnostics NOT human site accuracy. Sequential
  live scenes not identical frames/independent parts. Faster sampling can alter
  which poses are inspected; no fabricated precision/recall equivalence.
- Suite353 passed/two existing warnings; latest40/prior24-file fallbacks verified
  unchanged. ROI/native candidate not packaged, explicit flags required; builder
  refuses unsupported ROI proof. All runs stopped;11:39 UTC process check clear.
  No r33/vendor/SDK/service/fixed-station edits. Updated runtime/status/reference
  and full experiment report; raw audit/benchmark/source evidence preserved.

## 2026-11-16 — Feature-ROI owner streaming review opened

- Owner requested quality/latency viewing. Started300s feature-ROI+C++8/pose browser
  trial at11:44 UTC, `feature-roi-owner-view-20261116-114404`, bound~11:49 UTC.
  Verified running and exact-matched frame10/located,59ms processing (not optical
  latency). Loopback panel54588, live left/inspected right; approved5010/5012.
  No fallback/r33 changes; await owner feedback, no other camera graph started.

## 2026-11-16 — Owner-approved feature-ROI/C++ pipeline promoted to operations

- Owner accepted viewer and requested packaging/production default before frontend.
  Frozen selected release `releases/portable-vision-production-20261116-r2`,119.2MB/
  42 files, native binary/source, all flags, offline wheels/evidence; no credentials.
  r1 staging retained (r2 fixes README duration wording only, no inference changes).
- Extended packager to validate/freeze ROI and native hash/kernel. Dedicated r2
  production venv installed OFFLINE, pip check clean. Actual packaged r1 smoke
  639 calls/15.56 per sec/30.04 video,639 matches; selected r2 smoke468/15.09 per
  sec/30.06 video,468 matches, no unknown/RTP gaps/errors. Startup overhead included;
  no independent accuracy, optical latency or soak acceptance claim.
- New run-production.ps1 and pinned config/production-pipeline.json select frozen
  code, not mutable experiments. Promotion requires completed packaged smoke,
  model/native/mode/source/hash matching; atomic selection/history keeps rollback.
  Eleven guard/launcher tests; suite364 passed/two existing warnings.42-file release
  and prior40-file fallback verifiers pass. Earlier releases/fallbacks untouched.
- Operational production selection is owner-approved baseline, not a new autostart
  service or24/7 supervisor. Default/max300s retained until frontend lifecycle work.
  All smoke jobs completed; no r33/SDK/vendor/services/fixed-station modifications.
  Updated quickstart/README/runtime/deployment/status and production/frontend guide.

## 2026-11-16 — Owner production stream test

- Owner requested stream viewing. Launched DEFAULT run-production.ps1, selected
  immutable r2,300s at12:18 UTC (~12:23 bound). Session
  `production-runs/coordinate-pose-20261116-121859-188`; verified running and
  matched frame15/located, loopback panel58078. No release changes or concurrent
  camera graph; await owner feedback.

## 2026-11-16 — Mac/private-link portability preparation (not promoted)

- Owner confirmed switch and M3 Air/macOS27.0. Read-only board network inventory
  found single end0 with management/camera IPv4 addresses; RTSP OPTIONS200.
  No interface/service settings changed. Added experimental route-selected private
  receiver, source-bound SSH, pinned device identity, configurable board/camera,
  Apple Silicon setup/launcher helpers and nonsecret example profile.
- Downloaded nine native Mac-compatible CPython3.11 dependency wheels, not executed
  on Mac. Windows route test747/747 matched video/results at~30 decodedFPS; all
  UNKNOWN/no PCB, NOT inference accuracy/speed evidence. Portable package and
  actual Mac test remain pending; production r2/fallbacks unchanged.
- Added owned-worker stop prototype; startup-before-ready test exposed interruption
  during SDK preflight and empty local downloads. Fixed missing-file handling;
  ready-state stop/restart qualification still pending.28 focused tests pass.
- Ignored local secrets/artifacts and removed plaintext passwords from credential
  wiki before planned Git import. No Git initialization/history/push yet.














## 2026-11-16 — Owner clarified clone-only deployment requirement

- Copied complete portable55-file payload/model/native library/offline Mac and
  Windows wheels INTO deployment/portable-v1; integrity passes, every file below
  GitHub100MiB, ordinary Git inclusion exceptions added (no LFS/external artifacts).
- Root setup-host.py and run-portable.py default exclusively to clone-local payload,
  local/host-env and local/runs. Board supplied at launch, secret prompted, no
  sibling folders needed. Historical Windows production selectors not Mac path.
- No board changes. Actual Mac/fresh-clone execution and Git push remain pending;
  Python interpreter/OS/existing board SDK are platform prerequisites.

## 2026-11-16 — GitHub initialization and clone-only Mac handoff

- Initialized main, meaningful component-based commits under configured owner
  identity, no co-author trailers/backdated history. Pushed to Gitter09 portable-
  machine-vision. Local credentials/selectors ignored, known-secret scan clean.
- Added MAC_TESTING.md, byte-preserving attributes for frozen payload/SDK; ordinary
  Git stores all18 wheels/model/native files below100MiB, no LFS/artifact fetch.
- Fresh independent clone verifies55 payload files,18 wheels, root launcher help
  and credential exclusion. Actual Mac run remains owner task; no board changes.

## 2026-11-16 — Owner-requested Git identity and history replacement

- Owner requested replacing Parth-authored Git history with one fresh root commit
  attributed to Gitter09 using account-ID GitHub noreply address. Current pipeline
  files retained, no co-author trailers. Earlier changelog records are preserved
  as operational documentation, not assertions of retained Git ancestry.

## 2026-11-16 — Offline summit detection dashboard

- Added standalone `frontend/` white/navy dashboard with owner-supplied BITS WILP logo, locally vendored Lucide icons, schematic PCB simulation, selectable missing/present/unknown/stale/disconnected scenarios, fullscreen and presenter controls.
- Added read-only loopback bridge to existing coordinate receiver endpoints; exact-frame overlays only, stale/disconnected findings cleared, conflicts explicit. No overall verdict, product counter, accuracy or yield claims. Live result rate is dashboard-observed, not model throughput.
- Added dashboard guide and product/design context. Seven Python server tests and three Node state tests pass; JavaScript syntax and local HTTP smoke pass. Rendered browser/4K/accessibility and physical-board integration validation remain outstanding. No SDK, frozen payload, model, runtime lifecycle, board or fixed-station changes.

## 2026-11-16 — Summit first-screen layout refinement

- Removed owner-rejected introductory subtitle and side simulation/connection copy, retaining the header mode badge and inspection failure messages.
- Desktop presentation now budgets viewport height for inspection plus performance strip together. Canvas uses contained scaling without cropping or distortion; supporting preview/history remain below. Very short windows retain a readable minimum height. Narrow layout remains stacked.
- Updated dashboard guide and design record. All seven server and three state tests plus JavaScript syntax checks pass. Rendered fit still needs owner confirmation; no board, SDK or frozen runtime changes.

## 2026-11-16 — Presenter dropdown and interaction refinement

- Removed confidence disclaimer and frame-pairing caption from the dashboard, and removed detections-panel scrolling. Moved fullscreen into the inspection header in place of Exact-frame overlay.
- Replaced expanding presenter controls with an anchored, stacked source/scenario dropdown adapted from the owner's React reference for the existing plain-JavaScript UI. Added outside-click dismissal, keyboard traversal/back/escape, selection indicators, reduced-motion support, click feedback and action confirmations.
- Updated dashboard/design documentation. Existing seven Python and three Node checks pass, and menu syntax passes; rendered menu and hardware checks remain outstanding. No board or frozen runtime changes.

## 2026-11-16 — Video-only fullscreen and compact presentation controls

- Moved simulation Pause/Resume beside the headline with dynamic play/pause icon and pressed state. Removed it from presenter dropdown; remains disabled in live mode rather than implying board control.
- Fullscreen now targets only the inspection viewport, retaining stale/error safety messages and hiding auxiliary tags. Fullscreen/settings buttons share 44px dimensions.
- Shortened topbar, reduced logo scale, enlarged station title to Portable Machine Vision Inspection Setup, and removed station subtitle, footer sales line, host-receipt qualifier and simulation-session caption. Centered and enlarged the inference note below findings.
- Added Node regressions for pause icon changes and inspection-only fullscreen; updated documentation. Browser visual and hardware validation still required. No board/runtime changes.

## 2026-11-16 — Floating inspection chrome and source-aware pause visibility

- Removed topbar source badge and live preview description. Pause/Resume now disappears in Live mode and returns in simulation; schematic simulation remains labelled within the image.
- Reduced headline size and replaced inspection header/legend bezels with user-confirmed translucent, backdrop-blurred floating overlays, freeing image height within the first-screen budget. Repositioned source/frame labels around the overlays without stretching video.
- Added pause visibility regression and updated design/dashboard documentation. Browser-rendered acceptance and board checks remain outstanding; no board/runtime changes.

## 2026-11-16 — Summit dashboard publication and label cleanup

- Removed PCB-location overlay text and its update code; frame identifier remains right-aligned. Updated dashboard guide.
- Prepared the offline summit dashboard for owner-requested publication on main, using Gitter09 author/committer identity without co-author trailers. GitHub contributors API and current commit history both report only Gitter09; no contributor-removal or history rewrite was needed. A stale website contributor display is not authoritative evidence of current history.
- No board, SDK, frozen runtime, repository visibility or access-setting changes.

## 2026-11-16 — Focused summit reliability and recovery candidate

- Added bounded HTTP/image requests, independent freshness watchdog, metadata/geometry validation, frame/session/receipt identity, rolling observed-result rate, independent stats polling, source-switch/tab-resume cancellation, persisted Live selection and automatic MJPEG reconnect/stall hiding. Live errors never substitute mock data.
- Hardened the loopback bridge with trusted local session/port discovery, cross-session frame rejection, payload/connection/stream limits, UI-asset serving allowlist, no-cache/security headers and sanitized supervisor recovery status.
- Added opt-in `run-summit.py` sequential bounded-session renewal through the unchanged verified launcher. Includes a per-checkout lock, finite failure retries, host watchdog, explicit cleanup-evidence safety stops and low-host-disk guard. Prompt-once authentication uses a private temporary profile removed on ordinary exit; no credentials are committed. Worker300s bounds, frozen payload, model, SDK, board services and production selection are unchanged.
- Added HTTP fault-injection, client recovery and supervisor-policy tests;49 targeted tests pass (29 Python,20 Node), and the frozen55-file verifier passes. Added summit runbook and updated Mac/runtime/status/test guidance. Board access, actual Mac/video/renewal cycles, real browser rendering and extended soak remain unverified; this is a host recovery candidate, not production qualification.

