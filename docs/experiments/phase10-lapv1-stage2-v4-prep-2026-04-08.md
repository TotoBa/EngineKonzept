# Phase 10 LAPv1 Stage-T2 v4 Prep

This note captures the prepared follow-up after the completed
[phase10-lapv1-stage2-fast-arena-v3-summary-2026-04-08.md](/home/torsten/EngineKonzept/docs/experiments/phase10-lapv1-stage2-fast-arena-v3-summary-2026-04-08.md)
run.

`v3` established three important points:

- the hard subset can activate the inner loop
- `inner0` and `auto4` are the only LAPv1 runtime variants still worth carrying
- the main remaining problem is no longer instability, but the transition from
  hard-subset gains into the later full-data joint phase

## Structural v4 Changes

The prepared `v4` path is:

- [phase10_lapv1_stage2_fast_all_unique_v4.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage2_fast_all_unique_v4.json)
- [phase10_agent_lapv1_stage2_fast_all_unique_v4.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage2_fast_all_unique_v4.json)
- [phase10_lapv1_stage2_fast_arena_all_unique_v4.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage2_fast_arena_all_unique_v4.json)
- [run_phase10_lapv1_stage2_fast_arena_v4_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage2_fast_arena_v4_longrun.sh)

The actual trainer/runtime changes behind `v4` are:

1. fixed common-holdout checkpoint selection
2. mixed `hard + full` training in the joint phase
3. differential unfreezing and differential learning rates
4. explicit collapse alarms in Stage-T2 logs

## Common-Holdout Selection

`v3` picked `best_epoch=2` on the hard validation subset. That made the
checkpoint-selection signal incomparable across phases.

`v4` fixes this by selecting checkpoints on a fixed common holdout:

- [all_unique_validation_v1/lapv1_validation.jsonl](/srv/schach/engine_training/phase10/lapv1_workflow_all_unique_v1/all_unique_validation_v1/lapv1_validation.jsonl)

This selection pass runs after every epoch with a fixed deliberation budget:

- `selection_min_inner_steps = 2`
- `selection_max_inner_steps = 4`

The phase-local validation sets are still logged, but they no longer decide the
exported checkpoint on their own.

## Mixed Joint Training

The joint phase no longer jumps straight from `hard` to `full`.

Instead, `v4` keeps a persistent hard fraction in the joint epochs:

- full train path:
  [all_unique_train_v1/lapv1_train.jsonl](/srv/schach/engine_training/phase10/lapv1_workflow_all_unique_v1/all_unique_train_v1/lapv1_train.jsonl)
- hard train path:
  [all_unique_train_hard_v1/lapv1_train_hard.jsonl](/srv/schach/engine_training/phase10/lapv1_workflow_all_unique_v1/all_unique_train_hard_v1/lapv1_train_hard.jsonl)
- mix weights:
  - full `0.75`
  - hard `0.25`

The trainer now supports weighted phase-local mixes directly, instead of
requiring separate pre-merged artifacts.

## Differential Opening

The later joint phase now reopens the model in two steps:

1. `joint_heads_mix`
   - `inner_delta_network`
   - `inner_loop_core`
   - `root_heads`
2. `joint_backbone_mix`
   - same as above
   - plus `root_backbone` at a much smaller LR scale

Current LR scales:

- `inner_delta_network = 1.0`
- `inner_loop_core = 0.75` then `0.5`
- `root_heads = 0.35` then `0.25`
- `root_backbone = 0.1`

This is the direct follow-up to the `v3` finding that the full reopen was
washing out too much of the hard-subset gain.

## Narrowed Arena Family

The next arena intentionally drops `inner1`.

`v4` LAPv1 runtime variants:

- `lapv1_stage2_all_unique_v4_inner0`
- `lapv1_stage2_all_unique_v4_auto4`

Reference field:

- `planner_set_v6_expanded_v1`
- `planner_set_v6_rank_expanded_v1`
- `planner_recurrent_expanded_v1`
- `planner_set_v6_replay_expanded_v2`
- `vice_v2`

This keeps the comparison focused on the only two LAPv1 runtime budgets that are
still plausible follow-up candidates.

## New Trainer Contracts

Stage-T2 phases now support:

- `train_path_weights`
- `train_epoch_examples`
- `learning_rate_scale_by_group`
- stage-wide `selection_validation_paths`
- stage-wide `selection_min_inner_steps`
- stage-wide `selection_max_inner_steps`

The trainer also now emits collapse warnings when a validation pass effectively
falls back to `root ~= final`.

## Goal of v4

`v4` is not trying to make `inner1` work again.

The narrower goal is:

- preserve the hard-subset correction gain
- stop the joint phase from erasing it
- see whether `auto4` can become strictly better than `inner0`
  under a cleaner checkpoint-selection and finetune regime

If `v4` still leaves `auto4` only tied with or below `inner0`, the next step
should likely move away from more curriculum tuning and toward a stronger
deliberation update or reply-signal formulation.
