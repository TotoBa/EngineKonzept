# Phase 8

## Goal

Introduce the first bounded planner over:

- exact proposer candidates
- exact successor states
- explicit opponent signals

without crossing the project boundary into classical search.

## Current repository state

The repository does not yet have a trained planner model.

It now does have the first planner-facing offline baseline:

- [eval_planner_baseline.py](/home/torsten/EngineKonzept/python/scripts/eval_planner_baseline.py)
- [planner.py](/home/torsten/EngineKonzept/python/train/eval/planner.py)
- [planner_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/planner_compare_v1.json)

That baseline is explicitly bounded and symbolic-contract-aware:

1. score exact legal root candidates with the symbolic proposer
2. exact-apply a bounded top-k root slice
3. exact-generate successor reply candidates
4. score replies with either:
   - no opponent head
   - the symbolic reply scorer
   - the first learned opponent head
5. aggregate a bounded pessimistic root score

## What it is not

- no alpha-beta
- no tree search
- no transposition table
- no runtime planner module

It is an offline evaluation baseline that proves the proposer/opponent contracts can already be composed into a first planner-like decision layer.

## Current baseline result

On the current `128`-example merged verify slice:

- root-only symbolic proposer:
  - `root_top1_accuracy=0.148438`
  - `teacher_root_mean_reciprocal_rank=0.213542`
- symbolic-reply aggregation:
  - `root_top1_accuracy=0.15625`
  - `teacher_root_mean_reciprocal_rank=0.216797`
- learned-reply aggregation:
  - `root_top1_accuracy=0.15625`
  - `teacher_root_mean_reciprocal_rank=0.21875`

That means the first planner-facing composition is real and slightly better than root-only ranking on this slice, but it is still far from a mature planner.

## Next pressure

The next useful Phase-8 steps are now:

1. switch the bounded planner baseline to the new learned `set_v2` opponent head by default
2. tighten the aggregation contract between proposer, dynamics, and opponent signals
3. move from offline bounded planner evaluation to a trained planner module
