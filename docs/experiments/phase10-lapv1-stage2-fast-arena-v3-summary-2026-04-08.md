# Phase-10 LAPv1 Stage-T2 Fast Arena v3 Summary (2026-04-08)

## Scope

This note summarizes the completed `phase10_lapv1_stage2_fast_arena_all_unique_v3` run:

- campaign root: [/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v3](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v3)
- training summary: [summary.json](/home/torsten/EngineKonzept/models/lapv1/stage2_fast_all_unique_v3/summary.json)
- verify report: [lapv1_verify.json](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v3/lapv1_verify.json)
- arena summary: [summary.json](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v3/arena/summary.json)
- arena matrix: [arena_matrix.json](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v3/arena_matrix.json)

The run tested the `v3` hard-curriculum Stage-T2 path:

- early `freeze_inner_hard` phases on the hard subset
- later `joint_finetune_full` phases on the full all-unique set
- arena variants:
  - `lapv1_stage2_all_unique_v3_inner0`
  - `lapv1_stage2_all_unique_v3_inner1`
  - `lapv1_stage2_all_unique_v3_auto4`
- references:
  - `planner_set_v6_replay_expanded_v2`
  - `planner_recurrent_expanded_v1`
  - `planner_set_v6_rank_expanded_v1`
  - `planner_set_v6_expanded_v1`
  - `vice_v2`

## Headline

`v3` proved that the inner loop can be made active on the hard subset, but it still does not survive the later full-data joint finetune cleanly.

The result is mixed:

- better inner-loop activity than `v2`
- slightly stronger LAPv1 arena behavior for `inner0` and `auto4`
- but worse common-holdout verify quality than `v2`
- and still no convincing case that `inner1` is the best runtime path

The most important structural problem is now clear:

- checkpoint selection is currently comparing epochs that were validated on different datasets
- `best_epoch=2` came from the hard-validation phase, not from the common final verify holdout
- that makes model selection across T2 phases unreliable

## Training History

### Hard subset phases

Epoch 1 `freeze_inner_hard`, `min=1`, `max=2`:

- validation `top1=0.906467`
- validation `MRR=0.946611`
- `top1_changed_rate=0.007333`
- `root_incorrect_improvement_rate=0.071031`
- `mean_inner_steps_executed=1.007333`

Epoch 2 `freeze_inner_hard`, `min=2`, `max=4`:

- validation `top1=0.912733`
- validation `MRR=0.950272`
- `top1_changed_rate=0.031`
- `root_incorrect_improvement_rate=0.25`
- `mean_inner_steps_executed=2.014467`

Interpretation:

- the hard subset did what it was supposed to do
- the inner loop changed decisions meaningfully
- deeper-step exposure was real in the hard phase

### Full-data joint finetune

Epoch 3 `joint_finetune_full`, `min=2`, `max=2`:

- validation `top1=0.807097`
- validation `MRR=0.888712`
- `top1_changed_rate=0.001652`
- `root_incorrect_improvement_rate=0.004835`
- `mean_inner_steps_executed=1.990859`

Epoch 4 `joint_finetune_full`, `min=4`, `max=4`:

- validation `top1=0.80747`
- validation `MRR=0.888965`
- `top1_changed_rate=0.005397`
- `root_incorrect_improvement_rate=0.017106`
- `mean_inner_steps_executed=3.981717`

Interpretation:

- the joint phase largely washed out the hard-phase gain
- even when forced to depth 4, the later full-data phase did not preserve the stronger correction behavior from epoch 2
- the inner loop stayed alive structurally, but its gain on the common full-data target almost vanished

## Final Verify Result

Final verify from [lapv1_verify.json](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v3/lapv1_verify.json):

- `top1=0.781467`
- `MRR=0.869391`
- `top3=0.958409`
- `top1_changed_rate=0.032835`
- `teacher_rank_improved_rate=0.020795`
- `teacher_rank_degraded_rate=0.023349`
- `root_incorrect_improvement_rate=0.094059`
- `root_correct_degraded_rate=0.014052`
- `mean_teacher_rank_delta=-0.004378`
- `mean_inner_steps_executed=1.026997`
- step histogram:
  - `0: 17`
  - `1: 2634`
  - `2: 89`
  - `3: 1`

Comparison:

- Stage-T1 `v1`: `top1=0.778913`, `MRR=0.869087`
- Stage-T2 `v2`: `top1=0.793506`, `MRR=0.876931`
- Stage-T2 `v3`: `top1=0.781467`, `MRR=0.869391`

Interpretation:

- `v3` is only marginally above Stage-T1 on the common verify holdout
- `v3` clearly regressed against `v2` on the same final measure
- the step histogram shows that the deployed checkpoint still behaves almost entirely like a one-step model at verify time
- the average teacher-rank delta is slightly negative, so the extra decision changes are not yet net-positive

## Arena Result

Arena totals from [summary.json](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v3/arena/summary.json):

- total games: `112`
- total matchups: `56`

Standings:

- `vice_v2`: `27.5 / 28`, rate `0.982143`
- `planner_set_v6_expanded_v1`: `15.5 / 28`, rate `0.553571`
- `planner_set_v6_rank_expanded_v1`: `14.5 / 28`, rate `0.517857`
- `planner_recurrent_expanded_v1`: `14.5 / 28`, rate `0.517857`
- `planner_set_v6_replay_expanded_v2`: `14.0 / 28`, rate `0.5`
- `lapv1_stage2_all_unique_v3_inner0`: `10.0 / 28`, rate `0.357143`
- `lapv1_stage2_all_unique_v3_auto4`: `10.0 / 28`, rate `0.357143`
- `lapv1_stage2_all_unique_v3_inner1`: `6.0 / 28`, rate `0.214286`

Termination mix:

- `threefold_repetition: 43`
- `checkmate: 41`
- `engine_adjudication_black_advantage: 14`
- `engine_adjudication_white_advantage: 13`
- `stalemate: 1`

Comparison to `v2` LAPv1 arena variants:

- `v2 inner0`: `12.5 / 40`, rate `0.3125`
- `v2 auto4`: `11.5 / 40`, rate `0.2875`
- `v2 inner1`: `11.5 / 40`, rate `0.2875`

Interpretation:

- `inner0` improved in arena relative to `v2`
- `auto4` also improved and matched `inner0`
- `inner1` got worse, not better
- `auto4` no longer looks obviously harmful, but it also does not yet beat `inner0`

## What Worked

1. The hard subset and forced-depth schedule can activate the inner loop.
2. The per-example rollback and residual reranker path stayed stable through training.
3. `auto4` now behaves about as safely as `inner0` in the arena.

## What Failed

1. The joint full-data finetune still collapses most of the inner-loop gain.
2. Checkpoint selection is not trustworthy across phase-specific validation sets.
3. `inner1` is still not a competitive runtime budget.
4. The deployed checkpoint still behaves mostly like a one-step policy on the common verify holdout.

## Main Diagnosis

The next problem is no longer basic stability. The run completed. The hard curriculum worked locally. The remaining issue is training/control discipline:

1. The hard subset creates a useful correction signal.
2. The later joint phase overwrites too much of it.
3. The exported checkpoint is selected using a validation score that is not comparable across phases.

That means the current pipeline can produce a promising inner-loop behavior temporarily, but it does not yet preserve or select it correctly.

## Recommended Next Actions

### 1. Fix checkpoint selection first

Do not select `best_epoch` across phase-local validation sets.

Instead:

- either evaluate every epoch on one fixed common holdout and select on that
- or keep per-phase best checkpoints and run a final common-verify selection step before arena

This is the highest-priority fix because `v3` likely exported the hard-phase winner instead of the true best full-run checkpoint on the final target.

### 2. Replace hard-then-full with mixed joint training

Do not switch from `hard` to `full` abruptly.

Instead:

- keep a persistent hard-fraction in every joint epoch
- for example `25-50%` hard subset plus `50-75%` full-data batches

The goal is to stop the joint phase from erasing the correction behavior learned in epoch 2.

### 3. Use differential unfreezing and differential learning rates

The current joint phase still opens too much at once.

Next run should prefer:

- `inner_loop + candidate_delta_network + policy heads` first
- backbone/trunk later or with a much smaller LR

The hard-phase evidence suggests the inner loop can learn. The issue is preserving it while reopening the rest of the model.

### 4. Narrow the runtime family

For the next arena cycle:

- keep `inner0`
- keep `auto4`
- drop `inner1` from the main comparison set

`inner1` should remain a diagnostic budget, not a primary runtime candidate, until it clearly beats `inner0` on both verify and arena.

### 5. Build the hard curriculum from real Stage-T1 errors

The current hard subset is still heuristic.

The next upgrade should mine:

- real Stage-T1 root mistakes
- positions where deeper correction improves teacher rank
- positions where root is overconfident and wrong

That will likely be a better T2 curriculum than the current generic hard subset.

### 6. Add collapse alarms to future runs

Future T2 logs should explicitly flag:

- `final_top1 ~= root_top1`
- `final_mrr ~= root_mrr`
- `top1_changed_rate` near zero
- mean inner steps collapsing back toward `1`

That should be treated as a runtime warning, not only as a post-run interpretation.

## Recommendation

Do not promote `LAPv1` into the active planner family yet.

The next sensible step is a focused `v4` run with:

- fixed common-holdout checkpoint selection
- mixed hard/full joint training
- differential unfreezing or differential LRs
- only `inner0` and `auto4` in the main arena family

That is the shortest path to learning whether the inner loop can become a real runtime advantage rather than a temporary hard-subset effect.
