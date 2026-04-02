# Architecture Overview

EngineKonzept is converging toward a latent-planning chess engine, not a classical search engine with neural scoring attached.

## Target Runtime Path

`position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`

The current canonical roadmap for the learned stack is documented in [model-roadmap.md](/home/torsten/EngineKonzept/docs/architecture/model-roadmap.md).

## Repository Boundaries

- `rust/` will hold runtime, exact rules, UCI, inference integration, planner runtime, and evaluation harnesses.
- `python/` will hold datasets, training, model export, and experiment code.
- `tests/` will hold shared fixtures and reference suites.
- `models/` and `artifacts/` will store generated outputs, not source code.
- local IPC and future distribution notes live in [distribution.md](/home/torsten/EngineKonzept/docs/architecture/distribution.md)

## Phase 0 Constraints

This phase intentionally stops at scaffolding:

- no chess rules
- no UCI handling
- no planner code
- no model training
- no classical search fallback

The purpose of the current tree is to lock the module boundaries and validation workflow before any engine logic starts.

## Phase 1 Status

The repository now also includes the exact symbolic rules core under `core-types`, `position`, and `rules`.
That core is the correctness floor for later planner-oriented phases, not a conventional engine runtime.

## Phase 2 Status

The repository now additionally exposes a minimal UCI shell under `uci-protocol` and `engine-app`.
That shell can reconstruct exact positions and emit deterministic legal stub moves, but it still contains no search, evaluation, or planner logic.

## Phase 3 Status

The repository now additionally defines a factorized move vocabulary in `action-space` and a deterministic object-centric position encoder in `encoder`.
These layers make symbolic state consumable by later learned components without introducing inference, policy models, or planner logic yet.

## Phase 4 Status

The repository now additionally includes a Python dataset pipeline and a Rust dataset oracle backed by the exact rules kernel.
This adds reproducible example schemas, labels, split generation, and summary reporting without introducing training, inference, or selfplay yet.

## Phase 5 Status

The repository now additionally includes the first learned legality/policy proposer in Python, held-out proposer metrics, a `torch.export` bundle, a Rust-side bundle loader, and an offline PGN-to-Stockfish labeling path for larger policy datasets.
The runtime still does not execute learned inference yet, and no dynamics, opponent, planner, or classical search logic has been introduced.

Phase 5 now has two architecture families behind the same export contract:

- `mlp_v1`
- `multistream_v2`

The measured result so far is that structured inputs are promising but not yet enough by themselves. The current default remains the simpler MLP path, while the next preferred model step is a factorized proposer decoder rather than immediately moving to heavier expert-routing designs.
