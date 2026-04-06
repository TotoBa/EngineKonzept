# Phase 10 LAPv1 Stage1 Fast Arena Summary

This note records the first completed all-unique fast LAPv1 Stage1 arena run
with two runtime LAPv1 variants against the six strongest internal planner arms
from the last `vice` evolution run plus `vice_v2`.

## Artifact roots

- campaign root:
  [/srv/schach/engine_training/phase10/lapv1_stage1_fast_arena_all_unique_v1](/srv/schach/engine_training/phase10/lapv1_stage1_fast_arena_all_unique_v1)
- trained Stage1 bundle:
  [stage1_fast_all_unique_v1](/home/torsten/EngineKonzept/models/lapv1/stage1_fast_all_unique_v1)
- arena matrix:
  [arena_matrix.json](/srv/schach/engine_training/phase10/lapv1_stage1_fast_arena_all_unique_v1/arena_matrix.json)

## Verify result

The shared Stage1 checkpoint reached:

- `root_top1_accuracy = 0.778913`
- `root_top3_accuracy = 0.960598`
- `teacher_root_mean_reciprocal_rank = 0.869087`

That is still below the six reused planner-family references from the last
completed Phase-9 run. The strongest internal comparison remains
`planner_recurrent_expanded_v1`.

## Arena setup

The arena used:

- `9` agents total
- `72` ordered matchups
- `288` games
- `150` Thor-derived openings
- `144` unique non-swapped openings
- color-swapped rematches on the same opening

The two LAPv1 entrants were:

- `lapv1_stage1_all_unique_v1_inner0`
  `deliberation_max_inner_steps = 0`
- `lapv1_stage1_all_unique_v1_inner1`
  `deliberation_max_inner_steps = 1`

They shared the exact same trained Stage1 checkpoint.

## Main outcome

`inner0` clearly beat `inner1` overall:

- `inner0`: `score_rate = 0.3125`
- `inner1`: `score_rate = 0.210938`

Direct H2H between them was not decisive:

- `8 / 8` draws
- all by `threefold_repetition`

The difference came from the rest of the field.

`inner0` final record:

- `0` wins
- `40` draws
- `24` losses

`inner1` final record:

- `0` wins
- `27` draws
- `37` losses

Against `vice_v2`, both lost `8 / 8`.

## Important interpretation

This run does **not** show that deliberation is a bad idea.

It shows something narrower and more important:

- the current Stage1 checkpoint was trained only with `max_inner_steps = 0`
- the Stage1 training history confirms that all deliberation-specific metrics were `0`
- `inner1` was therefore only a **runtime override**, not a trained deliberation arm

In other words:

- `inner0` is in-distribution for the trained checkpoint
- `inner1` is out-of-distribution use of the same checkpoint

So the current result should be read as:

- "runtime-only activation of one inner step hurts"
- not "trained deliberation hurts"

## Why `inner1` is weaker right now

Three concrete reasons are visible in the current code and metrics.

1. Stage-T1 never trains the inner loop.

The saved Stage1 history in
[summary.json](/home/torsten/EngineKonzept/models/lapv1/stage1_fast_all_unique_v1/summary.json)
shows:

- `max_inner_steps = 0` for both epochs
- `rollbacks = 0`
- `rollback_hit_rate = 0.0`
- `deliberation_monotonicity_loss = 0.0`

So the recurrent update, selector, transition, rollback logic, and stop rule are
not optimized toward the Stage1 objective.

2. One extra step changes behavior, but not in a trained direction.

`inner1` does not simply reproduce `inner0`. It loses materially more often,
especially against:

- `planner_set_v2_expanded_v1`
- `planner_set_v6_replay_expanded_v2`
- `planner_set_v6_rank_expanded_v1`

That means the added step is strong enough to perturb decisions, but not yet
well enough trained to improve them.

3. The current runtime stop rule already behaves like a bounded budget, but it
is not backed by a matching trained checkpoint.

The current loop already has:

- `max_inner_steps`
- `q_threshold`
- `top1_stable_steps`
- rollback support

So the architecture can already stop early when confidence/stability says it
should. The missing piece is not the existence of a budget mechanism; it is a
checkpoint trained under nonzero inner-step budgets.

## Concrete path to make `inner1 > inner0`

The next LAPv1 step should be a real Stage-T2 run, not another Stage-T1 arena.

Recommended order:

1. Prepare a dedicated `phase10_lapv1_stage2_fast_all_unique_v1.json`.
   Use the same all-unique `lapv1_*` artifacts and warm-start from the current
   Stage1 checkpoint.

2. Train with nonzero inner-step curriculum instead of runtime override only.
   Recommended first schedule:
   - `max_inner_steps_schedule = (1, 2, 4)`
   - keep trained hard cap small at first
   - do not jump straight to `8`

3. Give the loop a real runtime budget cap, then compare:
   - `inner0`
   - `inner1`
   - `inner2`
   - `auto4`

`auto4` here means:

- checkpoint trained with deliberation on
- runtime `deliberation_max_inner_steps = 4`
- the loop may stop earlier through `q_threshold` / top1 stability

4. Use the trained Stage-T2 checkpoint for the arena variants.
   Only then is `inner1 vs inner0` a meaningful architectural comparison.

## Concrete path to reach deeper than `inner1`

The current code already supports deeper bounded loops through
`deliberation_max_inner_steps`. What is missing is disciplined training and
evaluation around that cap.

The clean near-term contract is:

- training checkpoint defines the maximum supported inner depth
- runtime spec chooses a cap `<= trained_max_inner_steps`
- stop rule still exits early if no more refinement is needed

That means deeper variants like `inner2` or `inner4` do not require a new
runtime architecture. They require:

- a real Stage-T2 checkpoint
- comparison specs for several runtime caps
- per-cap arena evaluation

## Recommended implementation priorities

1. Fix the missing arena summary artifact write path.
   This is now repaired in code and should remain part of every future arena run.

2. Add the first real Stage-T2 config and train it.

3. Re-run the same Phase-10 arena with:
   - `inner0`
   - `inner1`
   - `inner2`
   - one `auto` budgeted variant

4. Only after that decide whether the inner loop itself is helping.

## Deferred but important

Future UCI/runtime integration will need per-step status reporting from the
inner loop. That is **not** a current blocker, but it should be kept on the
deliberation TODO list.

Useful future runtime status items would be:

- current inner step
- current top1 move
- sharpness / uncertainty
- rollback fired or not
- early-stop reason

That is a later interface task, not part of this run fix.
