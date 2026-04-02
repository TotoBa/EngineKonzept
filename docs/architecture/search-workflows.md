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
