# GPT-Pro Review Brief

This document prepares an external review of `TotoBa/EngineKonzept`.

## Repository Identity

- GitHub repository: `TotoBa/EngineKonzept`
- Mission: build a chess engine around latent adversarial planning
- Runtime target:
  `position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`

## Source Of Truth

Reviewers should read these first:

- [AGENTS.md](/home/torsten/EngineKonzept/AGENTS.md)
- [PLANS.md](/home/torsten/EngineKonzept/PLANS.md)
- [README.md](/home/torsten/EngineKonzept/README.md)
- [model-roadmap.md](/home/torsten/EngineKonzept/docs/architecture/model-roadmap.md)
- [search-workflows.md](/home/torsten/EngineKonzept/docs/architecture/search-workflows.md)

## Current Best Repo State

### Phase 5

Current preferred proposer path:

- config: [phase5_stockfish_pgn_symbolic_v1_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_symbolic_v1_v1.json)
- bundle: [stockfish_pgn_symbolic_v1_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_symbolic_v1_v1)

Meaning:

- legality is exact and symbolic
- the model scores only legal candidates
- candidate features include symbolic move-side context
- Rust runtime already uses the symbolic proposer path natively

### Phase 6

Current preferred dynamics path:

- config: [phase6_dynamics_merged_unique_structured_v5_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_merged_unique_structured_v5_v1.json)
- bundle: [dynamics_merged_unique_structured_v5_v1](/home/torsten/EngineKonzept/models/dynamics/dynamics_merged_unique_structured_v5_v1)
- comparison: [dynamics_merged_unique_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_compare_v1.json)

Current large-corpus verify metrics:

- `structured_v2_latent`: `feature_l1_error=1.067843`, `drift_feature_l1_error=6.305117`
- `structured_v3`: `feature_l1_error=1.02784`, `drift_feature_l1_error=6.18409`
- `structured_v5`: `feature_l1_error=0.924808`, `drift_feature_l1_error=1.548861`

Interpretation:

- symbolic move-side features now clearly help Phase 6 at larger scale
- `structured_v5` is the best current Phase-6 arm
- exact next-state accuracy remains `0.0`

## Current Pressures

The main open problems are now:

1. improve dynamics further on top of `structured_v5`
2. define the first opponent-model contract
3. design planner-facing workflows without drifting into hidden classical search
4. use alpha-beta and MCTS as workflow tools in ways that strengthen the learned stack

## What An External Review Should Deliver

The external review should produce:

1. a codebase audit focused on architecture consistency, training/inference contracts, and phase discipline
2. a prioritized plan for improving the `structured_v5` dynamics line
3. a concrete proposal for how alpha-beta and MCTS should be used as offline workflows in this repo
4. a separation between:
   - immediate code changes
   - medium-term experiments
   - later planner work
5. explicit warnings wherever a proposal would risk violating the runtime architecture

## What To Avoid

Do **not** recommend:

- replacing runtime move selection with alpha-beta
- replacing runtime move selection with MCTS
- adding a fallback classical engine
- hiding search inside the planner path without clearly labeling it as non-runtime research tooling

## Best Review Angle

The most useful external contribution would be:

- strengthen the symbolic proposer and symbolic-action dynamics contract
- define how search teachers can supervise later opponent/planner models
- propose a workflow that uses alpha-beta and MCTS to improve training, benchmarking, and curriculum
- keep the shipped engine on the learned runtime path
