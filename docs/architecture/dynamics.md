# Dynamics Architecture

This page documents the first Phase-6 latent-dynamics baseline.

## Current Baseline

The current implementation is a local, action-conditioned transition model:

- state input: the same packed `230`-feature encoder vector used by the proposer
- action input: one flattened `20480`-space action index
- latent state: `z = E(s)`
- transition: `z' = z + G(z, a)`
- decoder: `D(z') -> next packed state features`

The exported bundle stays intentionally narrow:

- checkpoint
- `torch.export` program
- Rust-loadable metadata

The current bundle lives under [v1](/home/torsten/EngineKonzept/models/dynamics/v1).

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

The first materialized run is:

- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1/summary.json)
- verify: [dynamics_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1_verify.json)

## Current Reading Of The Results

The initial `v1` dynamics arm learns a smoother next-state reconstruction signal, but it does **not** yet recover exact packed next states:

- validation `feature_l1_error` falls materially during training
- verify `feature_l1_error` is stable and externally checkable
- exact next-state accuracy is still `0.0`
- multi-step drift remains measurable but weak

This is enough to establish the Phase-6 dataset, training, export, and Rust-boundary plumbing, but not enough to call the dynamics model good.

## Next Model Pressure

The obvious next pressures are:

- stronger local structure than a flat MLP encoder/decoder
- more explicit handling of special-move transitions
- potentially partial-state or tokenized reconstruction instead of one flat feature regression target

Those changes should stay action-conditioned and must not drift toward any hidden symbolic search or fallback evaluator.
