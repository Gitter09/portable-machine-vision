---
type: index
title: Portable Vision Wiki
description: Entry point for the portable SiMa Modalix inspection station.
status: active
tags: [index, orientation]
generated:
  by: opencode/deepseek-v4.1-flash
  at: 2026-11-14T09:00:00Z
sources:
  - repo://portable-vision/README.md
---

# Portable Vision Wiki

Documentation for the portable Demeter machine-vision station: one SiMa Modalix
board that inspects ICs and screw terminals on a rotating plate.

**New here?** Read [quickstart](quickstart.md), then the architecture page for
the part you intend to change.

## Architecture

| Page | Covers |
|---|---|
| [runtime](architecture/runtime.md) | Board, GStreamer pipeline, worker protocol, release staging |
| [inference-backends](architecture/inference-backends.md) | Six-head graph, decode paths, quantisation, measured throughput |
| [model-pipeline](architecture/model-pipeline.md) | Dataset → train → export → configs → compile → stage |
| [decisions](architecture/decisions.md) | Why it is built this way, and the options rejected |

## Reference

| Page | Covers |
|---|---|
| [cli](reference/cli.md) | Flags for the four pipeline tools |
| [credentials](reference/credentials.md) | Hosts, logins and endpoints, explicitly |
| [analysis-scripts](reference/analysis-scripts.md) | The measurement and diagnostics scripts |
| [files-and-paths](reference/files-and-paths.md) | Workspace, board and training-data locations |
| [tests](reference/tests.md) | What is covered and what the tests do not prove |

## Operations

| Page | Covers |
|---|---|
| [status](operations/status.md) | **Live state**: what is staged, what is running, known drift, next actions |
| [experiments](operations/experiments.md) | What was tried in order, with numbers and verdicts |
| [deployment](operations/deployment.md) | Stage a release, run it, measure it, hand the camera back |
| [troubleshooting](operations/troubleshooting.md) | Camera, decoder contention, known decode defect |

## Guides

| Page | Covers |
|---|---|
| [ic-terminal-model](guides/ic-terminal-model.md) | The four-class label contract and dataset rules |
| [sima-modalix-test-pipeline](guides/sima-modalix-test-pipeline.md) | What the board path can and cannot be trusted to do |

## Relationship to the fixed station

The fixed AOI screw station lives in `../demeter-systems-mv-full/` and has its own
wiki. The two share the `vision_sdk` package **by copy**. Anything that says
"shared" in this wiki refers to that duplication, not to a runtime import.
