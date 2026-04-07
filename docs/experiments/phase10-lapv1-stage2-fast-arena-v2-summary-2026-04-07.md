# Phase 10 LAPv1 Stage-T2 Fast Arena v2 Summary

This note records the completed `phase10_lapv1_stage2_fast_arena_all_unique_v2`
run and the architectural conclusion from the first real trained inner-loop
comparison (`inner0`, `inner1`, `inner2`, `auto4`) against the current six best
planner-family reference arms plus `vice_v2`.

Primary artifacts:

- [campaign summary.json](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v2/summary.json)
- [LAPv1 verify](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v2/lapv1_verify.json)
- [arena summary](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v2/arena/summary.json)
- [arena matrix](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v2/arena_matrix.json)
- [Stage-T2 training summary](/home/torsten/EngineKonzept/models/lapv1/stage2_fast_all_unique_v2/summary.json)

Reference baselines for comparison:

- [Stage-T1 fast arena summary](/home/torsten/EngineKonzept/docs/experiments/phase10-lapv1-stage1-fast-arena-summary-2026-04-06.md)
- [Phase-9 vice evolution summary](/home/torsten/EngineKonzept/docs/experiments/phase9-evolution-round03-vice-v1-summary-2026-04-05.md)

## Run shape

The completed run used:

- the merged all-unique Phase-5 corpus family
- the precomputed `lapv1_*` workflow artifacts
- `Stage-T2` warm-start from the finished fast `Stage-T1` checkpoint
- phased T2 training:
  - `freeze_inner` for epochs `1-2`
  - `joint_finetune` for epochs `3-4`
- four runtime LAPv1 variants from one shared Stage-T2 checkpoint:
  - `inner0`
  - `inner1`
  - `inner2`
  - `auto4`
- six reference planner-family arms:
  - `planner_recurrent_expanded_v1`
  - `planner_set_v2_expanded_v1`
  - `planner_moe_v2_expanded_v1`
  - `planner_set_v6_replay_expanded_v2`
  - `planner_set_v6_expanded_v1`
  - `planner_set_v6_rank_expanded_v1`
- external benchmark:
  - `vice_v2`

Arena size:

- `11` agents
- `110` matchups
- `220` games

## Stage-T2 training result

The Stage-T2 training itself was stable and did improve the model over
`Stage-T1`.

Best validation epoch (`75,043` examples):

- `Stage-T1`: `top1=0.796303`, `MRR=0.882240`
- `Stage-T2 v2`: `top1=0.804965`, `MRR=0.887493`

Held-out verify (`2,741` examples):

- `Stage-T1`: `top1=0.778913`, `MRR=0.869087`
- `Stage-T2 v2`: `top1=0.793506`, `MRR=0.876931`

So the total wrapper did get stronger:

- `top1`: `+0.014593` over the Stage-T1 verify run
- `MRR`: `+0.007844` over the Stage-T1 verify run

## Inner-loop diagnosis

The critical result is that the extra inner-step budgets still barely change the
root choice.

Held-out verify diagnostics:

- initial root `top1=0.792776`
- final `top1=0.793506`
- initial root `MRR=0.876566`
- final `MRR=0.876931`
- `top1_changed_rate=0.000730`
- `teacher_rank_improved_rate=0.000730`
- `teacher_rank_degraded_rate=0.0`
- `mean_teacher_rank_delta=0.000730`

That means:

- the trained Stage-T2 checkpoint is better than Stage-T1
- but almost all of that gain comes from the stronger shared wrapper/root stack
- the actual deliberation pass is still making very few useful ranking changes

The execution histogram on verify reinforces that reading:

- step histogram: `{'0': 17, '1': 2722, '2': 2}`
- mean executed inner steps: `0.994528`

In practice the runtime remains almost entirely a one-step system. The budgeted
deeper path exists, but the trained checkpoint almost never uses it.

The Stage-T2 validation history shows the same pattern:

- epoch `1` (`freeze_inner`, budget `1`):
  - root-vs-final slightly worse than the initial root
- epoch `2` (`freeze_inner`, budget `2`):
  - still slightly worse than the initial root
- epoch `3` (`joint_finetune`, budget `2`):
  - still slightly worse than the initial root
- epoch `4` (`joint_finetune`, budget `4`):
  - finally slightly positive, but only by a tiny margin

So the new T2 design fixed the earlier structural issues, but it still does not
yet produce a strong multi-step deliberation effect.

## Arena result

Final arena ranking by score rate:

1. `vice_v2`: `1.000000`
2. `planner_set_v6_replay_expanded_v2`: `0.675000`
3. `planner_recurrent_expanded_v1`: `0.575000`
4. `planner_set_v6_rank_expanded_v1`: `0.575000`
5. `planner_set_v6_expanded_v1`: `0.537500`
6. `planner_moe_v2_expanded_v1`: `0.500000`
7. `planner_set_v2_expanded_v1`: `0.487500`
8. `lapv1_stage2_all_unique_v2_inner0`: `0.312500`
9. `lapv1_stage2_all_unique_v2_auto4`: `0.287500`
10. `lapv1_stage2_all_unique_v2_inner1`: `0.287500`
11. `lapv1_stage2_all_unique_v2_inner2`: `0.262500`

So the current internal ordering is:

- best LAPv1 runtime variant: `inner0`
- then `auto4 ~= inner1`
- then `inner2`

This is still not the target architecture outcome. The extra budget variants do
not beat the zero-extra-step runtime.

## Comparison against the Stage-T1 arena

Compared with the first fast Stage-T1 arena:

- `inner0` stayed flat:
  - Stage-T1: `0.312500`
  - Stage-T2 v2: `0.312500`
- `inner1` improved materially:
  - Stage-T1: `0.210938`
  - Stage-T2 v2: `0.287500`

That is an important nuance:

- training the inner loop did help `inner1`
- but not enough to surpass `inner0`
- and deeper budgets (`inner2`, `auto4`) still do not deliver a net gain

## Inner-family matchups

The intra-LAPv1 family gives two additional clues.

1. `inner0` vs `inner1` ended tied overall (`2.0 - 2.0`), but all four games
   were black wins.

This is a genuine signal to inspect:

- color symmetry
- opening sensitivity
- side-to-move calibration
- adjudication sign handling

2. All other inner-family pairings were full draws by repetition.

That strongly suggests the deeper variants are not yet learning targeted
corrective updates. They mostly collapse back to the same practical trajectory.

## Termination profile

Across the full arena:

- `checkmate`: `87`
- `threefold_repetition`: `80`
- `engine_adjudication_black_advantage`: `28`
- `engine_adjudication_white_advantage`: `23`
- `stalemate`: `2`

This is better than a pure draw swamp, but still repetition-heavy. The LAPv1
family in particular remains draw-prone against itself and weak against the
stronger planner-family arms.

## Architectural conclusion

The conclusion is now narrower and clearer than after the Stage-T1 run:

1. The Stage-T2 wrapper path is worth keeping.
   It improved the overall model over Stage-T1 on both validation and verify.

2. The current bottleneck is no longer “can T2 train at all?”
   It can.

3. The bottleneck is now specifically:
   the inner-loop budget does not yet create enough useful deviation from the
   root ranking.

4. `inner1` should not be discarded.
   It improved significantly over the untrained Stage-T1 `inner1`.
   But it is still not the best runtime variant.

5. `inner2` and `auto4` are currently premature as headline evaluation targets.
   They are useful diagnostics, but they are not yet runtime winners.

## Recommended follow-up work

The next useful work is not another unchanged T2 rerun.

### 1. Train explicit improvement-over-root behavior

Add a new auxiliary objective for step `>0`:

- reward only when final teacher rank improves over the root rank
- penalize unnecessary top-1 changes that do not improve teacher rank

Right now the inner loop is mostly harmless rather than helpful. That needs a
loss that explicitly prefers beneficial divergence from the root.

### 2. Build a hard-position T2 curriculum

The current all-unique T2 set is too easy for the inner loop.

Prioritize positions where:

- root top-1 is wrong but top-3 contains the teacher move
- top1/top2 teacher gap is small
- reply pressure / uncertainty is high
- planner-family arms disagree strongly

This should be a T2-specific subset, not just “more of the same” all-unique
corpus.

### 3. Force real depth exposure during training

The current budgeted checkpoint still executes almost only one step.

Add an explicit late-phase curriculum such as:

- phase A: `min=max=1`
- phase B: `min=1, max=2`
- phase C: `min=2, max=4` on the hard subset

Without a phase that truly requires deeper steps, `inner2` and `auto4` will
keep behaving like weak extensions of `inner1`.

### 4. Audit the black-win asymmetry in `inner0` vs `inner1`

The tied but all-black-win four-game micro-match is too suspicious to ignore.

Check:

- sign conventions for value / reply penalties
- side-to-move normalization in the inner update path
- any opening-color asymmetry in the selected Thor suite
- adjudication-color conversion

### 5. Narrow the next arena family

For the next LAPv1 cycle, keep:

- `inner0`
- `inner1`
- optionally `auto4` as a diagnostic shadow arm

Drop `inner2` from the next mainline arena unless the new training curriculum
actually exposes and rewards depth `>=2`.

### 6. Compare against fewer but stronger references

The next LAPv1-focused arena does not need the whole current family.

Use:

- `planner_set_v6_replay_expanded_v2`
- `planner_recurrent_expanded_v1`
- `planner_set_v6_rank_expanded_v1`
- `vice_v2`

That keeps the benchmark sharp and iteration time lower.

## Bottom line

This run was a success in one important sense:

- `Stage-T2` is now trainable, resumable, and measurably stronger than
  `Stage-T1`

But it was not yet a success in the architectural end goal:

- the trained inner-step variants still do not outperform the zero-extra-step
  runtime

So the next phase should focus specifically on making deliberation useful, not
just making the whole wrapper slightly better.
