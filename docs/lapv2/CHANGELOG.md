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
