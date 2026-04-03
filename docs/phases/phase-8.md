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

The first materialized planner reference was:

- config: [phase8_planner_corpus_suite_set_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v1.json)
- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_v1/summary.json)
- bundle: [corpus_suite_set_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1/summary.json)
- verify: [planner_corpus_suite_set_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v1_verify.json)
- comparison: [planner_corpus_suite_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v1.json)

The current preferred planner reference is now:

- config: [phase8_planner_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_v1.json)
- bundle: [corpus_suite_set_v2_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v2_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1/summary.json)
- verify: [planner_corpus_suite_set_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v2_v1_verify.json)
- comparison: [planner_corpus_suite_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_compare_v2.json)

The repo now also has a filtered latent-state validation slice over just the `10k` and `122k` tiers:

- workflow suite: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_workflow_corpus_suite_latent_two_tier_v1/summary.json)
- experimental config: [phase8_planner_corpus_suite_set_v3_two_tier_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v3_two_tier_v1.json)
- experimental bundle: [corpus_suite_set_v3_two_tier_v1](/home/torsten/EngineKonzept/models/planner/corpus_suite_set_v3_two_tier_v1)
- experimental summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_set_v3_two_tier_v1/summary.json)
- filtered comparison: [planner_corpus_suite_two_tier_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase8/planner_corpus_suite_two_tier_compare_v1.json)

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
- first trained planner `set_v1`:
  - `root_top1_accuracy=0.788652`
  - `root_top3_accuracy=0.958156`
  - `teacher_root_mean_reciprocal_rank=0.872636`
  - `teacher_root_mean_probability=0.616233`
- current planner `set_v2`:
  - `root_top1_accuracy=0.795035`
  - `root_top3_accuracy=0.968085`
  - `teacher_root_mean_reciprocal_rank=0.875355`
  - `teacher_root_mean_probability=0.685788`
  - `root_value_mae_cp=90.521303`
  - `root_gap_mae_cp=264.01746`

That means the repository is now past pure planner baselines. The richer-target `set_v2` arm stays comfortably above all bounded hand-aggregation baselines and improves modestly over the first `set_v1` planner on the same multi-corpus verify suite.

## Latent-state validation slice

To validate planner-state changes without waiting on the `400k` tier, the repo now also has a filtered `10k + 122k` planner workflow slice with explicit Phase-6 latent successor features.

Filtered verify result over `1,024` held-out planner examples:

- root-only bounded baseline:
  - `root_top1_accuracy=0.151367`
  - `teacher_root_mean_reciprocal_rank=0.219482`
- symbolic-reply bounded baseline:
  - `root_top1_accuracy=0.158203`
  - `teacher_root_mean_reciprocal_rank=0.222819`
- learned-reply bounded baseline:
  - `root_top1_accuracy=0.135742`
  - `teacher_root_mean_reciprocal_rank=0.2111`
- reference planner `set_v2` on the same filtered slice:
  - `root_top1_accuracy=0.80957`
  - `root_top3_accuracy=0.975586`
  - `teacher_root_mean_reciprocal_rank=0.883382`
- latent-state planner `set_v3`:
  - `root_top1_accuracy=0.708008`
  - `root_top3_accuracy=0.933594`
  - `teacher_root_mean_reciprocal_rank=0.825521`

That is a clear negative result for the first latent-state planner arm: `PlannerHeadV1` can now carry Phase-6 latent successor vectors, but the first direct `set_v3` integration loses clearly to `set_v2`.

## Next pressure

The next useful Phase-8 steps are now:

1. test whether better opponent uncertainty signals improve planner calibration more than raw reply accuracy alone
2. decide whether latent planner state needs a richer contract or a different integration path before it can help
3. then try richer bounded recurrence over the same exact candidate slice
