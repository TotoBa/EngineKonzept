# LAPv2 Changelog

## Schritt 0

- Normalized the repo-local LAPv2 plan setup against the actual completed
  `stage2_fast_arena_all_unique_v4` artifacts.
- Added the initial LAPv2 path map in
  [path_map.md](/home/torsten/EngineKonzept/docs/lapv2/path_map.md).
- Added a local bridge for the imported Rev-3 detail plan.

## Schritt 1

- Mapped the logical LAPv2 module names to the current Python implementation
  paths used by Phase-10 LAPv1 training and arena runs.

## Schritt 2

- Added deterministic phase heuristics in
  [phase_features.py](/home/torsten/EngineKonzept/python/train/datasets/phase_features.py).
- Added HalfKA-style sparse feature helpers in
  [nnue_features.py](/home/torsten/EngineKonzept/python/train/datasets/nnue_features.py).
- Added candidate move-delta helpers in
  [move_delta.py](/home/torsten/EngineKonzept/python/train/datasets/move_delta.py).
- Extended the LAPv1 Phase-10 artifact loader and builder in
  [lapv1_training.py](/home/torsten/EngineKonzept/python/train/datasets/lapv1_training.py)
  with backward-compatible LAPv2 enrichment fields.
- Added regression coverage in
  [test_lapv2_artifact_features.py](/home/torsten/EngineKonzept/python/tests/test_lapv2_artifact_features.py).

## Schritt 3

- Added a deterministic hard phase router in
  [phase_router.py](/home/torsten/EngineKonzept/python/train/models/phase_router.py).
- Added the generic hard-routed expert wrapper in
  [phase_moe.py](/home/torsten/EngineKonzept/python/train/models/phase_moe.py).
- Added isolated routing and gradient tests in
  [test_phase_moe.py](/home/torsten/EngineKonzept/python/tests/test_phase_moe.py).

## Schritt 4

- Added the shared sparse FT backbone in
  [feature_transformer.py](/home/torsten/EngineKonzept/python/train/models/feature_transformer.py).
- Added batch and incremental dual-accumulator helpers in
  [dual_accumulator.py](/home/torsten/EngineKonzept/python/train/models/dual_accumulator.py).
- Extended the LAPv1 collator in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/trainers/lapv1.py)
  to emit `EmbeddingBag`-ready sparse NNUE inputs.
- Added FT/incremental consistency coverage in
  [test_dual_accumulator.py](/home/torsten/EngineKonzept/python/tests/test_dual_accumulator.py).

## Schritt 5

- Added feature-flagged `lapv2` settings to the LAPv1 wrapper config in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py),
  including `enabled` plus `phase_moe`.
- Wrapped `intention_encoder` and `state_embedder` with
  [PhaseMoE](/home/torsten/EngineKonzept/python/train/models/phase_moe.py)
  when the new `lapv2.phase_moe` flag is active, and routed batches via
  [PhaseRouter](/home/torsten/EngineKonzept/python/train/models/phase_router.py).
- Extended the trainer and runtime paths to propagate `phase_index`
  through the batch/model interface.
- Added legacy checkpoint warm-start expansion so old single-expert LAPv1
  checkpoints can load into the new phase-expert wrapper without changing
  outputs.
- Added regression coverage for flag-off bit identity, legacy warm-start
  equivalence, and one no-NaN training step with `phase_moe` enabled.

## Schritt 6

- Added the single-phase shared FT-backed NNUE value head in
  [value_head_nnue.py](/home/torsten/EngineKonzept/python/train/models/value_head_nnue.py).
- Extended the LAPv1 artifact schema in
  [lapv1_training.py](/home/torsten/EngineKonzept/python/train/datasets/lapv1_training.py)
  with `side_to_move` so NNUE value evaluation can order `a_stm` and
  `a_other` correctly.
- Integrated `FeatureTransformer` + `DualAccumulatorBuilder` into the
  top-level wrapper in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py)
  behind the new `lapv2.nnue_value` flag while keeping the legacy dense
  value head active for flag-off and inner-loop compatibility.
- Extended trainer/runtime batch plumbing so the NNUE value path receives
  sparse HalfKA inputs plus `side_to_move`.
- Added standalone NNUE value-head tests, legacy flag-off checks, and a
  no-NaN Stage-T1 smoke step with `lapv2.nnue_value` enabled.

## Schritt 7

- Upgraded the shared FT and NNUE value head in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py)
  to optional phase-routed variants via `lapv2.nnue_value_phase_moe`.
- Taught the dual-accumulator builder in
  [dual_accumulator.py](/home/torsten/EngineKonzept/python/train/models/dual_accumulator.py)
  how to repack sparse `EmbeddingBag` rows per phase, instead of relying
  on the generic MoE slicer for flat index buffers.
- Added the phase-gate mean-pull hook in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/trainers/lapv1.py),
  keyed by `lapv2.nnue_phase_gate_steps`, so early training can keep FT
  and NNUE-value experts synchronized.
- Extended legacy warm-start handling so single-phase step-6 checkpoints
  can expand into the new phase-expert FT and NNUE value modules.
- Added regression coverage for single->phase warm starts, gate mean-pull,
  all-phase forwards, and one no-NaN training step with the phase-routed
  NNUE value path enabled.

## Schritt 8

- Added the shared-FT NNUE policy head in
  [policy_head_nnue.py](/home/torsten/EngineKonzept/python/train/models/policy_head_nnue.py).
- Extended the LAPv1 artifact contract in
  [lapv1_training.py](/home/torsten/EngineKonzept/python/train/datasets/lapv1_training.py)
  with optional successor rebuild features for king-move candidates.
- Extended the trainer batch path in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/trainers/lapv1.py)
  to materialize candidate move-type ids, sparse move deltas, king-move
  successor rebuilds, and shared-FT loss balancing between value and policy.
- Integrated `lapv2.nnue_policy` into the top-level wrapper and runtime
  paths in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py)
  and
  [lapv1_runtime.py](/home/torsten/EngineKonzept/python/train/eval/lapv1_runtime.py),
  reusing the same FT as the NNUE value head.
- Added module, model, artifact, trainer, runtime, and FT-gradient
  regression coverage for the new policy path.

## Schritt 9

- Wrapped the sharpness head behind `lapv2.sharpness_phase_moe` in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py)
  via the existing hard-routed
  [PhaseMoE](/home/torsten/EngineKonzept/python/train/models/phase_moe.py).
- Extended the inner-loop sharpness projector and root forward path so
  phase-routed sharpness works both at the root and inside deliberation
  without changing the flag-off behavior.
- Added legacy checkpoint warm-start expansion for phase-routed
  `sharpness_head` weights in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/trainers/lapv1.py).
- Added regression coverage for sharpness-phase forwards, flag-off
  identity, and one no-NaN training step with `lapv2.sharpness_phase_moe`
  enabled.

## Schritt 10

- Added the shared-backbone opponent readout modules in
  [opponent_readout.py](/home/torsten/EngineKonzept/python/train/models/opponent_readout.py),
  including `DeltaOperator` plus the lightweight
  `OpponentReadout` reply/pressure/uncertainty heads.
- Integrated the new `lapv2.shared_opponent_readout` flag into
  [lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py)
  without changing the legacy reply-signal aggregation formula
  `best_reply - 10 * pressure - 10 * uncertainty`.
- Kept the legacy opponent-head path intact for flag-off runs and added
  warm-start compatibility so older checkpoints can switch into the new
  readout path while freshly initializing only the step-10 weights.
- Extended the trainer and runtime coverage to exercise the new readout
  path, the flag-off identity path, and legacy checkpoint upgrades in
  [test_opponent_readout.py](/home/torsten/EngineKonzept/python/tests/test_opponent_readout.py),
  [test_lapv1_model.py](/home/torsten/EngineKonzept/python/tests/test_lapv1_model.py),
  [test_lapv1_trainer.py](/home/torsten/EngineKonzept/python/tests/test_lapv1_trainer.py),
  and [test_lapv1_runtime.py](/home/torsten/EngineKonzept/python/tests/test_lapv1_runtime.py).

## Schritt 11

- Added the optional `lapv2.distill_opponent` training hook in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/trainers/lapv1.py),
  including configurable `distill_fraction`, `distill_reply_weight`,
  `distill_pressure_weight`, and `distill_uncertainty_weight`.
- Extended the deliberation path in
  [deliberation.py](/home/torsten/EngineKonzept/python/train/models/deliberation.py)
  and [lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py)
  to emit per-step student and teacher opponent targets only when the
  trainer explicitly requests distillation diagnostics.
- Kept runtime unchanged: normal model forwards and the LAPv1 runtime do
  not request teacher targets, so `lapv2.distill_opponent` adds no extra
  eval/runtime compute by default.
- Added regression coverage for positive/nonzero distill loss,
  zero-loss teacher identity, one no-NaN T2 training step with
  distillation enabled, and runtime identity with the distill flag set.

## Schritt 12

- Added eval-only successor caching in
  [dual_accumulator.py](/home/torsten/EngineKonzept/python/train/models/dual_accumulator.py)
  via `AccumulatorCache`, plus fixed-phase support for phase-routed FT
  experts.
- Extended the deliberation loop in
  [deliberation.py](/home/torsten/EngineKonzept/python/train/models/deliberation.py)
  to emit per-step selected candidate tensors and fixed phase indices so
  cache reuse can be audited explicitly.
- Integrated `lapv2.accumulator_cache` into the NNUE policy path in
  [lapv1.py](/home/torsten/EngineKonzept/python/train/models/lapv1.py).
  Training keeps the previous vectorized path; eval/runtime now optionally
  score successors through the incremental cache without changing outputs.
- Added regression coverage for phase fixation over the loop, cache-vs-no-cache
  eval identity, and top-k cache reuse hits in
  [test_dual_accumulator.py](/home/torsten/EngineKonzept/python/tests/test_dual_accumulator.py)
  and [test_lapv1_model.py](/home/torsten/EngineKonzept/python/tests/test_lapv1_model.py).
