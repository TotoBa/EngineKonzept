# Phase 8

## Goal

Introduce the first bounded planner over:

- exact proposer candidates
- exact successor states
- explicit opponent signals

without crossing the project boundary into classical search.

## Current repository state

The repository now has the first materialized trained bounded planner run over the current multi-corpus workflow suite.

It first gained the planner-facing offline baselines:

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

The repo now also has the first trainable planner-arm stack:

- [build_planner_head_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_planner_head_dataset.py)
- [build_phase8_workflow_suite.py](/home/torsten/EngineKonzept/python/scripts/build_phase8_workflow_suite.py)
- [eval_planner_suite_baseline.py](/home/torsten/EngineKonzept/python/scripts/eval_planner_suite_baseline.py)
- [compare_planner_suite_runs.py](/home/torsten/EngineKonzept/python/scripts/compare_planner_suite_runs.py)
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

The first materialized planner reference is now:

- config: [phase8_planner_corpus_suite_set_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v1.json)
- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_v1/summary.json)
- bundle: [corpus_suite_set_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1/summary.json)
- verify: [planner_corpus_suite_set_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1_verify.json)
- comparison: [planner_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v1.json)

## Current result

Planner workflow suite:

- `10k` tier
- merged unique `122k` tier
- imported unique `400k` tier

Training summary:

- `best_epoch=5`
- validation `root_top1_accuracy=0.799107`
- validation `root_top3_accuracy=0.967634`
- validation `teacher_root_mean_reciprocal_rank=0.880952`

Aggregate held-out verify result over `1,410` planner examples:

- root-only bounded baseline:
  - `root_top1_accuracy=0.153901`
  - `teacher_root_mean_reciprocal_rank=0.230615`
- symbolic-reply bounded baseline:
  - `root_top1_accuracy=0.159574`
  - `teacher_root_mean_reciprocal_rank=0.232861`
- learned-reply bounded baseline:
  - `root_top1_accuracy=0.142553`
  - `teacher_root_mean_reciprocal_rank=0.224232`
- trained planner `set_v1`:
  - `root_top1_accuracy=0.788652`
  - `root_top3_accuracy=0.958156`
  - `teacher_root_mean_reciprocal_rank=0.872636`
  - `teacher_root_mean_probability=0.616233`

That means the repository is now past pure planner baselines. The first trained bounded planner is materially stronger than the current bounded hand-aggregation baselines on the same multi-corpus verify suite.

## Next pressure

The next useful Phase-8 steps are now:

1. add richer planner targets than teacher top-1 plus restricted teacher policy
2. bring stronger Phase-6 latent-state information into the planner-facing dataset contract
3. test whether better opponent uncertainty signals improve planner calibration more than raw reply accuracy alone
