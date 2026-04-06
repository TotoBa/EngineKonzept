# Phase 10 LAPv1 Inner-Loop Follow-Up

This note records the architectural conclusion from the first completed fast
Stage-T1 LAPv1 arena run and the concrete repair path for the inner loop.

## Core diagnosis

The important result from
[phase10-lapv1-stage1-fast-arena-summary-2026-04-06.md](/home/torsten/EngineKonzept/docs/experiments/phase10-lapv1-stage1-fast-arena-summary-2026-04-06.md)
is not simply that `inner1` lost to `inner0`.

The narrower and more actionable result is:

- `inner1` was only a runtime override on a checkpoint trained with
  `max_inner_steps = 0`
- so `inner1` was out-of-distribution for the trained model
- the correct fix is therefore not to discard deliberation, but to train it

## What needed to change

Two concrete gaps were addressed.

1. Stage-T2 must warm-start from Stage-T1.

Without warm-start, the first deliberation-on run would spend too much of its
budget relearning the shared encoder/state/policy/value stack instead of the
new inner-loop behavior.

2. The inner loop needs direct supervision.

Before this follow-up, the trainer optimized only:

- final policy
- final value
- sharpness
- rollback/monotonicity auxiliaries

That leaves the first refined steps too weakly guided.

The trainer now also supervises the intermediate candidate-score tensors emitted
after each actual deliberation step. This makes the first extra step trainable
as a first-class target instead of a side effect.

3. Deliberation control must be per-example, not batch-global.

The original T2 prototype still let halting and rollback fire at batch level.
That meant one bad example in a large batch could force rollback semantics onto
every other example, and one sharp/easy example could stop the whole batch
early. The follow-up path now treats:

- halting
- rollback
- rollback-rate accounting
- step-loss masking

as per-example decisions. This is a structural fix, not just a hyperparameter
change.

4. Stage-T2 needs an explicit two-phase train path.

The original T2 run fine-tuned the whole wrapper end-to-end from the first
batch. That made it hard to tell whether extra inner steps improved the planner
or whether the root trunk simply adapted again. The follow-up path now supports:

- a freeze phase that trains only the inner-loop/opponent path
- a short joint phase that reopens the full wrapper

This also fixes the schedule problem where a short T2 run could configure a
`max_inner_steps=4` budget without ever actually training the final budgeted
step.

## Budget semantics

The runtime budget contract is now treated explicitly as:

- `deliberation_max_inner_steps` = hard runtime cap
- the loop may stop earlier via learned sharpness/stability gates

So deeper variants do not require a new runtime architecture.

They require:

- a checkpoint trained with nonzero inner steps
- runtime variants evaluated at multiple caps

The prepared comparison family is:

- `inner0`
- `inner1`
- `inner2`
- `auto4`

`auto4` means:

- budget cap `4`
- but stop early if the loop decides no further refinement is worth it

## Prepared next run

The prepared next run is:

- [phase10_lapv1_stage2_fast_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage2_fast_all_unique_v1.json)
- [phase10_lapv1_stage2_fast_arena_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage2_fast_arena_all_unique_v1.json)
- [run_phase10_lapv1_stage2_fast_arena_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage2_fast_arena_longrun.sh)

That run:

- warm-starts from the finished fast Stage-T1 checkpoint
- uses the same all-unique `lapv1_*` workflow artifacts
- trains a real Stage-T2 checkpoint with schedule `1 -> 2 -> 4`
- compares four LAPv1 runtime caps against the same six reference arms plus
  `vice_v2`

## Deferred but noted

The inner loop should later be able to emit runtime status for UCI `info`, but
that is still deliberately deferred.

Useful future fields:

- current inner step
- current top-1 move
- sharpness / uncertainty
- rollback fired
- early-stop reason
- remaining budget
