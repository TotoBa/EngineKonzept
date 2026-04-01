# Architecture Overview

EngineKonzept is converging toward a latent-planning chess engine, not a classical search engine with neural scoring attached.

## Target Runtime Path

`position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`

## Repository Boundaries

- `rust/` will hold runtime, exact rules, UCI, inference integration, planner runtime, and evaluation harnesses.
- `python/` will hold datasets, training, model export, and experiment code.
- `tests/` will hold shared fixtures and reference suites.
- `models/` and `artifacts/` will store generated outputs, not source code.

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
