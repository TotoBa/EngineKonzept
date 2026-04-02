# Dynamics Architecture

This page documents the first Phase-6 latent-dynamics baselines.

## Current Baseline

The current implementations are local, action-conditioned transition models:

- state input: the same packed `230`-feature encoder vector used by the proposer
- action input: one flattened `20480`-space action index
- latent state: `z = E(s)`
- transition: `z' = z + G(z, a)`
- decoder: `D(z') -> next packed state features`

The exported bundle stays intentionally narrow:

- checkpoint
- `torch.export` program
- Rust-loadable metadata

Materialized bundles:

- [v1](/home/torsten/EngineKonzept/models/dynamics/v1)
  Flat decoder baseline.
- [structured_v2_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_v1)
  Piece-/square-/rule-decoder follow-up with section-wise losses.
- [structured_v2_drift_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_drift_v1)
  Same structured decoder, but with checkpoint selection against an explicit held-out drift slice.

## Training Contract

The first dynamics trainer uses exact one-step supervision derived from existing Phase-5 datasets:

- current packed state features from `position_encoding`
- selected action index from `selected_action_encoding`
- exact next packed state features by re-encoding `next_fen` through the Rust oracle

Optional lean artifacts are now supported:

- `dynamics_train.jsonl`
- `dynamics_validation.jsonl`
- `dynamics_test.jsonl`

These files are backfilled with [materialize_dynamics_artifacts.py](/home/torsten/EngineKonzept/python/scripts/materialize_dynamics_artifacts.py).

## Current Metrics

The current baseline tracks:

- one-step reconstruction loss
- mean absolute feature error
- exact next-feature-vector accuracy after integer rounding
- separate exact-accuracy slices for capture, promotion, castle, en-passant, and check-giving moves
- multi-step drift over contiguous in-split action chains

The current materialized runs are:

- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1/summary.json)
- verify: [dynamics_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1_verify.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1/summary.json)
- verify: [dynamics_structured_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1_verify.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_drift_v1/summary.json)
- verify: [dynamics_structured_v2_drift_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_drift_v1_verify.json)
- comparison: [dynamics_structured_v2_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_compare_v1.json)

## Current Reading Of The Results

The initial `v1` dynamics arm learns a smoother next-state reconstruction signal, but it does **not** yet recover exact packed next states.

The first structured follow-up `structured_v2_v1` materially improves the soft metrics:

- validation `feature_l1_error`: `1.699881 -> 1.448869`
- verify `feature_l1_error`: `1.686233 -> 1.433716`
- verify `drift_feature_l1_error`: `1.902871 -> 1.595053`
- rule-section validation `feature_l1_error`: `1.161234`

The drift-aware `structured_v2_drift_v1` run improves the main held-out soft metrics again:

- verify `feature_l1_error`: `1.433716 -> 1.425823`
- verify `drift_feature_l1_error`: `1.595053 -> 1.557198`
- verify `square_feature_l1_error`: `1.382824 -> 1.28254`

But the hard limit is unchanged:

- exact next-state accuracy is still `0.0`
- multi-step drift remains measurable but weak

This is enough to establish the Phase-6 dataset, training, export, and Rust-boundary plumbing, but not enough to call the dynamics model good.

## Next Model Pressure

The obvious next pressures are now:

- a better special-move treatment that does not destabilize drift
- potentially partial-state or tokenized reconstruction instead of one flat feature regression target
- stronger multi-step drift supervision beyond the current short held-out slice

Those changes should stay action-conditioned and must not drift toward any hidden symbolic search or fallback evaluator.
