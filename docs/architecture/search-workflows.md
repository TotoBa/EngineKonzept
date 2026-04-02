# Search Workflows

This page defines how classical search methods may be used in EngineKonzept without violating the project's runtime architecture.

## Purpose

EngineKonzept is not allowed to turn into a conventional search engine with a neural add-on.

That does **not** mean alpha-beta or MCTS are useless here. It means they must be used as:

- offline teachers
- benchmark harnesses
- analysis tools
- curriculum generators
- counterexample miners

and **not** as the runtime decision mechanism inside the shipped engine.

## Allowed Uses

Alpha-beta and MCTS are allowed in repository workflows when they are used for one or more of the following:

- offline move-label generation over exact legal candidates
- search-trace generation for proposer, dynamics, opponent, or planner supervision
- benchmark baselines to measure how far the learned stack is from a classical reference
- disagreement mining, for example positions where the learned proposer or dynamics model fails badly
- curriculum shaping for later selfplay or planner training
- evaluation harnesses that compare learned planning against stronger symbolic teachers
- stress tests for candidate quality, drift, or tactical blind spots

## Forbidden Uses

Alpha-beta and MCTS are still forbidden as the runtime move-selection path.

That means:

- no alpha-beta in `engine-app` as the real move selector
- no MCTS as a hidden backup planner for `bestmove`
- no handcrafted static evaluation as a fallback engine
- no “temporary” classical search path that silently remains after experiments
- no search-generated move output unless it is clearly isolated as a benchmark tool and not the main runtime

## Repo-Compatible Workflow Direction

The current repo state suggests a specific direction for classical workflows:

1. exact legal candidates are generated symbolically
2. the symbolic proposer scores only legal candidates
3. the current preferred dynamics model consumes symbolic move-side features
4. later opponent and planner modules will work over the same candidate contract

That makes alpha-beta and MCTS most useful as **offline structured teachers** over the exact legal candidate set, not as direct runtime substitutes.

Concretely, the highest-leverage workflow ideas are:

- candidate-ranking teacher labels for the symbolic proposer
- short search traces attached to selected candidate moves for planner supervision
- search-derived targets for opponent reply prediction
- disagreement sets between symbolic proposer ranking and search ranking
- search-guided curriculum buckets, for example tactical, quiet, defensive, or forced-line positions
- dynamics stress suites built from lines where the search teacher is sensitive to exact tactical consequences

## Immediate Workflow Plan

The next workflow layer should be built in this order:

1. alpha-beta teacher labels over exact legal candidates
2. search-trace datasets for future opponent and planner supervision
3. disagreement mining between learned ranking and search ranking
4. curriculum buckets driven by those disagreements and traces
5. only later, offline MCTS distillation over the learned stack

### Alpha-Beta Teacher Labels

For each root position, the most useful initial outputs are:

- soft teacher policy over exact legal candidates
- root value or WDL target
- per-candidate reply value after best reply
- top-k candidate set
- optional short PV prefix

These targets fit the current symbolic proposer contract directly.

Suggested first dataset family:

- `search_teacher_train.jsonl`
- `search_teacher_validation.jsonl`
- `search_teacher_test.jsonl`

### Search Traces

The next dataset family should contain short offline traces, not runtime search code.

Recommended fields:

- root exact candidate set
- candidate symbolic features
- teacher top-k ranking
- best reply or top-m replies
- short PV line
- depth/nodes/instability metadata

That makes the trace useful later for opponent-head supervision and bounded planner supervision.

Suggested first dataset family:

- `search_traces_train.jsonl`
- `search_traces_validation.jsonl`
- `search_traces_test.jsonl`

### Disagreement Mining

The repo should explicitly mine positions where:

- proposer top-1 and teacher top-1 disagree sharply
- best reply is highly forced
- tactical punishment is under-modeled by the learned stack
- special-move or mobility edge cases cause large ranking errors

These positions should become regression fixtures and curriculum buckets, not one-off analysis files.

## Baseline Rule For Phase 7

Before a learned opponent head is treated as progress, compare it against one exact symbolic baseline:

1. exact apply our move
2. exact-generate opponent legal candidates
3. reuse the current symbolic proposer as the opponent reply scorer

That baseline is cheap, repo-compatible, and strong enough to prevent low-value opponent-model churn.

### MCTS Later

MCTS is more useful later than now.

The best future use here is:

- offline policy-improvement distillation over the learned candidate/value stack
- difficulty and instability targets for later planner compute allocation

That is acceptable only as an offline workflow, not as the shipped runtime move selector.

## Current Preferred Boundary

If a future experiment uses alpha-beta or MCTS in this repo, it should satisfy all of these:

- runtime UCI output remains driven by the learned stack
- exact rules core remains the authority for legality
- the search system is clearly labeled as a benchmark, teacher, or workflow tool
- exported runtime bundles do not depend on search code to choose moves
- docs state exactly what part of the pipeline is using the classical method

## Review Guidance

External reviewers should treat alpha-beta and MCTS as tools to improve:

- data quality
- training targets
- evaluation quality
- curriculum design
- planner supervision

They should **not** recommend quietly replacing the learned planner path with a classical search engine.
