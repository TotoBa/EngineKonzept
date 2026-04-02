# External Review Synthesis 2026-04

This note consolidates the actionable overlap between the two unversioned external review inputs:

- `docs/next.followup.md`
- `docs/next.followup.deep-research-report.md`

Those source files are intentionally left unversioned. This document is the versioned synthesis.

## Accepted Conclusions

### 1. The current direction is correct

The repo should keep:

- exact symbolic legality
- symbolic proposer candidate scoring
- symbolic-action dynamics built on top of the proposer contract
- Rust runtime as the authoritative execution path

### 2. The main bottleneck is now a missing Phase-6 to Phase-7 contract

The biggest gap is not “find another broad model family.”

It is to define:

- richer candidate context
- a transition-specific selected-action context
- a planner-facing dual-channel state
- an explicit opponent-head contract

### 3. Alpha-beta and MCTS should be raised as workflow tools, not runtime logic

They should be used for:

- teacher labels over exact legal candidates
- search traces
- disagreement mining
- curriculum generation
- evaluation harnesses

They should not be used for:

- `engine-app` move selection
- hidden fallback planning
- runtime-bestmove rescue logic

## Concrete Next Contracts

### CandidateContextV2

Versioned symbolic candidate features for proposer roots and later planner roots.

Expected additions:

- promotion identity
- castle side
- full captured-piece type
- normalized move geometry
- clearer naming of pre-move attack-map slots

### TransitionContextV1

Selected-action features for dynamics and later opponent modeling.

Expected additions:

- `CandidateContextV2`
- plus post-move exact tags from symbolic apply

Examples:

- `opponent_in_check_after_move`
- destination attacked/defended after move
- halfmove reset
- castling-rights delta
- en-passant created or cleared

### LatentStateV1

Planner-facing node state should be dual-channel:

- exact symbolic shadow state
- exact global-summary features
- learned latent state
- uncertainty summary

### OpponentHeadV1

The first explicit opponent module should predict:

- reply distribution over exact legal replies
- threat or pressure signal
- uncertainty

The first comparison baseline should be:

1. exact apply our move
2. exact-generate opponent legal candidates
3. reuse the current symbolic proposer as the opponent-reply scorer

## Concrete Workflow Layer

### Alpha-Beta First

Build first:

- soft teacher policy over exact legal candidates
- root value or WDL labels
- per-candidate reply value
- top-k teacher sets
- short PV traces

Recommended first dataset family:

- `search_teacher_<split>.jsonl`

### Search Traces

Recommended new dataset family:

- exact candidate set
- candidate symbolic features
- teacher ranking
- reply set
- short PV line
- depth/nodes metadata

Recommended first dataset family:

- `search_traces_<split>.jsonl`

Status now:

- `search_teacher_<split>.jsonl` is implemented
- `search_traces_<split>.jsonl` is implemented as the next offline workflow layer
- `search_disagreements_<split>.jsonl` is implemented as the first proposer-vs-teacher mining layer

### Disagreement Mining

Mine positions where:

- proposer and teacher disagree strongly
- best reply is highly forced
- tactical punishment is missed
- symbolic move categories expose systematic weakness

### MCTS Later

Use later for:

- offline policy-improvement distillation
- planner compute-allocation supervision

## Immediate Repo Actions

1. version the next contracts before Phase 7 code grows
2. unify symbolic feature authority or add golden Python/Rust identity tests
3. keep improving Phase 6 on top of large-corpus `structured_v5`
4. add granular exactness metrics beyond the current all-or-nothing exact next-state metric
5. use the now-implemented offline alpha-beta teacher/trace/disagreement layer for curriculum and later opponent/planner supervision
6. only then expand opponent/planner code
