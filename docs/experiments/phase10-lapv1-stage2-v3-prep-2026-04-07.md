# Phase 10 LAPv1 Stage-T2 v3 Prep

This note captures the concrete follow-up work after the completed
[phase10-lapv1-stage2-fast-arena-v2-summary-2026-04-07.md](/home/torsten/EngineKonzept/docs/experiments/phase10-lapv1-stage2-fast-arena-v2-summary-2026-04-07.md)
run.

The key `v2` conclusion remains unchanged:

- `Stage-T2` improved the wrapper over `Stage-T1`
- but the inner loop still changed the root decision far too rarely
- `inner0` remained the strongest LAPv1 runtime arm in arena play

## Added Follow-Up Instrumentation

The follow-up now has two new infrastructure pieces:

- hard-position subset builder:
  [build_lapv1_hard_positions_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_lapv1_hard_positions_dataset.py)
- arena color-asymmetry audit:
  [analyze_selfplay_arena_asymmetry.py](/home/torsten/EngineKonzept/python/scripts/analyze_selfplay_arena_asymmetry.py)

The completed `v2` arena was re-analyzed into:

- [arena_asymmetry.json](/srv/schach/engine_training/phase10/lapv1_stage2_fast_arena_all_unique_v2/arena_asymmetry.json)

## Asymmetry Findings

Global arena color balance is only mildly skewed:

- white score rate: `0.477273`
- black score rate: `0.522727`
- white wins: `64`
- black wins: `74`
- draws: `82`

So there is some black tilt, but not enough to explain the full LAPv1 ranking.

The strongest pairwise anomaly is still the direct `inner0` vs `inner1` matchup:

- all `4` games were black wins
- `inner0` score rate as white vs `inner1`: `0.0`
- `inner0` score rate as black vs `inner1`: `1.0`

That is real and should not be dismissed as noise, but the broader per-agent
color profiles are less extreme:

- `inner0`: white `0.25`, black `0.375`, delta `-0.125`
- `inner1`: white `0.25`, black `0.325`, delta `-0.075`
- `inner2`: white `0.275`, black `0.25`, delta `+0.025`
- `auto4`: white `0.275`, black `0.3`, delta `-0.025`

Interpretation:

- the pairwise `inner0` vs `inner1` anomaly is important to audit
- but the wider arena still says `inner1` is structurally weaker than `inner0`
- this is not just one global color-bias artifact

## Structural v3 Changes

The next run is prepared as a harder and narrower `v3` path:

- [phase10_lapv1_stage2_fast_all_unique_v3.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage2_fast_all_unique_v3.json)
- [phase10_agent_lapv1_stage2_fast_all_unique_v3.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage2_fast_all_unique_v3.json)
- [phase10_lapv1_stage2_fast_arena_all_unique_v3.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv1_stage2_fast_arena_all_unique_v3.json)
- [run_phase10_lapv1_stage2_fast_arena_v3_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv1_stage2_fast_arena_v3_longrun.sh)

The important changes relative to `v2` are:

- explicit improvement-over-root supervision
- early `freeze_inner_hard` training on a dedicated hard subset
- explicit `min_inner_steps` schedules, not only `max_inner_steps`
- forced deeper exposure in the later full-data phase
- narrowed LAPv1 runtime family:
  - `inner0`
  - `inner1`
  - `auto4`
- narrowed reference family:
  - `planner_set_v6_replay_expanded_v2`
  - `planner_recurrent_expanded_v1`
  - `planner_set_v6_rank_expanded_v1`
  - `planner_set_v6_expanded_v1`
  - plus `vice_v2`

## Why This v3 Shape

The `v2` run told us two things at once:

1. the wrapper still benefits from T2 training
2. the inner loop still is not learning strong enough corrections

So `v3` does not widen the arena or add more runtime variants. It instead tries
to make the correction task itself easier to learn:

- focus early epochs on rows that are sharp, close, and already high-priority
- explicitly require the residual loop to beat the detached root on rows where
  the root is still wrong
- force real multi-step exposure late in training so `auto4` is not only a
  runtime extrapolation

## Next Checkpoint

The next meaningful decision after `v3` is:

- does `inner1` finally overtake `inner0` on arena score rate?
- does `auto4` at least become non-inferior to `inner1`?
- does the verify report show materially larger `top1_changed_rate` and
  `root_incorrect_improvement_rate` than `v2`?

If not, the next change should move away from more curriculum tuning and toward a
stronger inner-loop update path or a stronger reply-signal formulation.
