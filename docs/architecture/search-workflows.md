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

Status now:

- the first `search_teacher_<split>.jsonl` workflow is implemented
- the second `search_traces_<split>.jsonl` workflow is implemented
- the third `search_disagreements_<split>.jsonl` workflow is implemented
- the fourth `search_curriculum_<split>.jsonl` workflow is implemented
- it is explicitly an offline UCI-teacher workflow
- it uses the exact legal candidate set plus `CandidateContextV2`
- it does not modify the runtime move-selection path

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

The current repo implementation builds these with:

- [build_search_teacher_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_search_teacher_dataset.py)
- [search_teacher.py](/home/torsten/EngineKonzept/python/train/datasets/search_teacher.py)

The first implementation deliberately uses an external UCI alpha-beta teacher, for example Stockfish, instead of embedding search inside the shipped runtime path. That keeps the architectural boundary clear while still producing structured offline labels.

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

The current repo implementation builds these with:

- [build_search_trace_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_search_trace_dataset.py)
- [search_traces.py](/home/torsten/EngineKonzept/python/train/datasets/search_traces.py)

The first implementation deliberately keeps traces root-centric:

- exact root candidate set
- root teacher ranking and scores
- principal variation as UCI plus flat action indices
- best reply from the first PV
- simple instability metadata via top-1 minus top-2 score gap

### Disagreement Mining

The repo should explicitly mine positions where:

- proposer top-1 and teacher top-1 disagree sharply
- best reply is highly forced
- tactical punishment is under-modeled by the learned stack
- special-move or mobility edge cases cause large ranking errors

These positions should become regression fixtures and curriculum buckets, not one-off analysis files.

Suggested dataset family:

- `search_disagreements_train.jsonl`
- `search_disagreements_validation.jsonl`
- `search_disagreements_test.jsonl`

The current repo implementation builds these with:

- [build_search_disagreement_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_search_disagreement_dataset.py)
- [search_disagreements.py](/home/torsten/EngineKonzept/python/train/datasets/search_disagreements.py)

The first implementation is intentionally proposer-centric:

- it consumes a `search_teacher_<split>.jsonl` artifact plus a symbolic proposer checkpoint
- it rebuilds current proposer-side `CandidateContextV1` inputs from the exact dataset split
- it aligns teacher and proposer rankings by exact flat action index, not by candidate-row position
- it records full proposer candidate scores and proposer policy over the exact legal set
- it records disagreement severity signals such as rank mismatch, top-1 mismatch, teacher top-1 advantage, and policy L1 distance

That keeps the workflow useful for:

- targeted proposer retraining
- later curriculum bucketing
- future planner and opponent regression suites

### Curriculum Buckets

The next useful workflow layer is to turn raw disagreement and trace artifacts into stable bucketed curriculum records.

Suggested dataset family:

- `search_curriculum_train.jsonl`
- `search_curriculum_validation.jsonl`
- `search_curriculum_test.jsonl`

The current repo implementation builds these with:

- [build_search_curriculum_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_search_curriculum_dataset.py)
- [search_curriculum.py](/home/torsten/EngineKonzept/python/train/datasets/search_curriculum.py)

The first implementation is intentionally simple and root-centric:

- it joins `search_traces_<split>.jsonl` and `search_disagreements_<split>.jsonl` by `sample_id`
- it assigns stable bucket labels such as:
  - `forced_teacher`
  - `unstable_teacher`
  - `reply_supervised`
  - `top1_disagreement`
  - `large_rank_mismatch`
  - `teacher_punishes_proposer`
  - `policy_shape_mismatch`
  - `capture_line`
  - `promotion_line`
  - `checking_line`
  - fallback `stable_agreement`
- it writes a single numeric `curriculum_priority` for easy later sampling and bucketing

That keeps the output directly usable for:

- proposer hard-example replay
- opponent-dataset filtering
- planner regression suites
- later curriculum schedulers without re-running search

The first concrete downstream consumer is now the `OpponentHeadV1` dataset workflow:

- it joins root traces with curriculum buckets
- it exact-applies the teacher root move
- it exact-generates opponent replies from the successor state
- it exposes the teacher best reply plus simple pressure and uncertainty targets

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
