# Agent guidance

This workspace is the **portable** machine-vision station: one SiMa Modalix
board inspecting ICs and screw terminals on a rotating plate. It is separate
from the fixed AOI screw station in `../demeter-systems-mv-full/`; do not assume
files, releases, ports or IP addresses carry across.

Read `openwiki/quickstart.md` first, then `openwiki/operations/status.md` — the
second records live board state (what is staged, what is running, known drift)
that cannot be derived from the code. Then route to the page for the subsystem you
are changing.

- Board runtime, release staging, worker protocol: `openwiki/architecture/runtime.md`
- Six-head graph, decode paths, quantisation: `openwiki/architecture/inference-backends.md`
- Dataset → train → export → configs → compile: `openwiki/architecture/model-pipeline.md`
- Why the system is built this way: `openwiki/architecture/decisions.md`
- Hosts, logins, endpoints: `openwiki/reference/credentials.md`
- Measurement and diagnostic scripts: `openwiki/reference/analysis-scripts.md`
- Tool flags: `openwiki/reference/cli.md`
- Paths, data roots, board directories: `openwiki/reference/files-and-paths.md`
- Tests: `openwiki/reference/tests.md`
- Stage, run and measure on the board: `openwiki/operations/deployment.md`
- Known defects and recovery: `openwiki/operations/troubleshooting.md`
- The four-class label contract: `openwiki/guides/ic-terminal-model.md`

Every change to behaviour, configuration or deployment must update the affected
pages and append an entry to `openwiki/CHANGELOG.md`. Existing changelog entries
are never edited, removed, reordered or squashed.

## Boundaries

- `vision_sdk/` is a **copy of the fixed station's SDK**, byte-identical to
  `../demeter-systems-mv-full/vision_sdk/`. Changing it here changes the screw
  station's runtime; if the change belongs to both, apply it there too and say so
  in the changelog. `vision_sdk/board_runtime/` is the screw rig, not this station.
- `runtime/launcher/` is **this station's** runtime and worker. It is the file that
  builds the RTSP graph and decides whether a run qualifies. The candidate tree
  outside the repository holds staged copies; treat this directory as the source.
- The board is shared hardware: the capture feed and any inference pipeline both
  need the hardware decoder, so they cannot run at the same time.
- Only the portable board is yours to stage and restart. The fixed station's board
  is a different unit.
