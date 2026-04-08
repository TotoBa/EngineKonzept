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
