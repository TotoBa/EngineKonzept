# Phase 8

## Goal

Introduce the first bounded planner over:

- exact proposer candidates
- exact successor states
- explicit opponent signals

without crossing the project boundary into classical search.

## Current repository state

The repository now has the first trainable planner contract and code path, but no materialized trained planner run yet.

It first gained the planner-facing offline baseline:

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

The repo now also has the first trainable planner-arm plumbing:

- [build_planner_head_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_planner_head_dataset.py)
- [build_phase8_workflow_suite.py](/home/torsten/EngineKonzept/python/scripts/build_phase8_workflow_suite.py)
- [planner_head.py](/home/torsten/EngineKonzept/python/train/datasets/planner_head.py)
- [planner.py](/home/torsten/EngineKonzept/python/train/models/planner.py)
- [planner.py](/home/torsten/EngineKonzept/python/train/trainers/planner.py)
- [train_planner.py](/home/torsten/EngineKonzept/python/scripts/train_planner.py)
- [eval_planner.py](/home/torsten/EngineKonzept/python/scripts/eval_planner.py)

That trainable arm keeps the project boundary:

1. exact root candidates still come from the symbolic proposer contract
2. successor states still come from exact move application plus oracle relabeling
3. opponent-side signals still come from a bounded reply module
4. the learned planner only scores a bounded candidate set; it does not become hidden tree search

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

1. materialize the first trained planner run on the current multi-corpus workflow suite
2. compare it directly against the current offline bounded baselines
3. only then decide whether the next planner gain should come from stronger root scoring or richer dynamics/opponent context
