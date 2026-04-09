# Phase-10 LAPv2 Stage-T2 v1 Preparation

This note captures the first real LAPv2 training and arena campaign
prepared on top of the completed `stage2_fast_all_unique_v4` LAPv1 run.

## Goal

Run the first end-to-end LAPv2 campaign that:

- warm-starts from the best available completed LAPv1 Stage-T2 checkpoint
- trains one shared LAPv2 Stage-T2 model with all currently implemented
  LAPv2 flags enabled
- benchmarks four LAPv2 runtime budgets against:
  - four LAPv1-v4 runtime budgets
  - four strong non-LAPv1 reference arms
  - `vice_v2`

## Important deviation from Rev-3.1 plan

The imported Rev-3.1 plan prefers a warm-start from the
`freeze_inner_hard_best` checkpoint. That exact phase-local checkpoint is
not materialized in the current repo or artifact roots. The available
completed artifact is:

- [checkpoint.pt](/home/torsten/EngineKonzept/models/lapv1/stage2_fast_all_unique_v4/bundle/checkpoint.pt)

This run therefore uses the completed `v4` best checkpoint as the first
pragmatic LAPv2 warm-start source and records that deviation explicitly.

The warm-start export has been verified against that real `v4`
checkpoint. As expected, the first LAPv2 init still fresh-initializes
the new FT/NNUE, shared-opponent-readout, and sharpness-phase-MoE alias
paths, while the remaining LAPv1-v4 weights load directly.

## Prepared artifacts

- Train config:
  [phase10_lapv2_stage2_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv2_stage2_all_unique_v1.json)
- Base runtime spec:
  [phase10_agent_lapv2_stage2_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv2_stage2_all_unique_v1.json)
- Arena campaign:
  [phase10_lapv2_stage2_arena_all_unique_v1.json](/home/torsten/EngineKonzept/python/configs/phase10_lapv2_stage2_arena_all_unique_v1.json)
- Launcher:
  [run_phase10_lapv2_stage2_arena_v1_longrun.sh](/home/torsten/EngineKonzept/python/scripts/run_phase10_lapv2_stage2_arena_v1_longrun.sh)

Comparison baselines:

- [phase10_agent_lapv1_stage2_fast_all_unique_v4_inner0.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage2_fast_all_unique_v4_inner0.json)
- [phase10_agent_lapv1_stage2_fast_all_unique_v4_inner1.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage2_fast_all_unique_v4_inner1.json)
- [phase10_agent_lapv1_stage2_fast_all_unique_v4_inner2.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage2_fast_all_unique_v4_inner2.json)
- [phase10_agent_lapv1_stage2_fast_all_unique_v4_auto4.json](/home/torsten/EngineKonzept/python/configs/phase10_agent_lapv1_stage2_fast_all_unique_v4_auto4.json)

## Training shape

The first LAPv2 run is intentionally conservative:

- `batch_size=256`
- `4` T2 epochs
- common full-selection holdout remains active
- explicit `phase_load_balance`
- explicit `stage_a` / `stage_b` gate
- mixed hard/full phases from the start

The point is not to maximize Elo in one shot, but to get the first fully
wired LAPv2 run stable enough to inspect:

- whether the warm-start holds
- whether the new heads train without collapse
- whether `inner1/2/auto4` benefit from the fully trained LAPv2 path

## Arena shape

The prepared arena contains:

- `4` LAPv2 budgets: `inner0`, `inner1`, `inner2`, `auto4`
- `4` LAPv1-v4 budgets: `inner0`, `inner1`, `inner2`, `auto4`
- `4` strong reference planners
- `vice_v2`

This yields a first directly comparable LAPv1-vs-LAPv2 matrix without
requiring another bespoke harness.
