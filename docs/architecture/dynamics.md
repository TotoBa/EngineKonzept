# Dynamics Architecture

This page documents the first Phase-6 latent-dynamics baselines.

## Current Baseline

The current implementations are local, action-conditioned transition models:

- state input: the same packed `230`-feature encoder vector used by the proposer
- action input: one flattened `20480`-space action index
- optional symbolic action side input: the selected move's exact candidate-feature row from the symbolic proposer contract
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
- [structured_v2_latent_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_latent_v1)
  Same drift-aware structured decoder, but with an auxiliary latent-consistency loss.
- [structured_v3_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v3_v1)
  Same latent-stable main path, plus auxiliary delta decoders trained only as a side target.
- [structured_v4_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v4_v1)
  Drift-supervised follow-up that keeps the latent-stable path but adds explicit short-horizon rollout supervision during training.
- [structured_v5_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v5_v1)
  Symbolic-action follow-up that keeps the latent-consistency baseline but augments the action pathway with the selected move's exact symbolic candidate features.
- [edit_v1](/home/torsten/EngineKonzept/models/dynamics/edit_v1)
  Experimental local edit-target arm that reconstructs delta sections relative to the current state.

## Training Contract

The first dynamics trainer uses exact one-step supervision derived from existing Phase-5 datasets:

- current packed state features from `position_encoding`
- selected action index from `selected_action_encoding`
- optional selected-move symbolic feature row aligned with the symbolic proposer contract
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
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_latent_v1/summary.json)
- verify: [dynamics_structured_v2_latent_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_latent_v1_verify.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1/summary.json)
- verify: [dynamics_structured_v3_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1_verify.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1/summary.json)
- verify: [dynamics_structured_v4_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1_verify.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v5_v1/summary.json)
- verify: [dynamics_structured_v5_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v5_v1_verify.json)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1/summary.json)
- verify: [dynamics_edit_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1_verify.json)
- comparison: [dynamics_phase6_parallel_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_phase6_parallel_compare_v1.json)
- comparison: [dynamics_phase6_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_phase6_compare_v2.json)
- comparison: [dynamics_phase6_compare_v3.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_phase6_compare_v3.json)

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

The parallel latent-consistency follow-up `structured_v2_latent_v1` improves them again while preserving the same basic architecture:

- verify `feature_l1_error`: `1.425823 -> 1.425074`
- verify `drift_feature_l1_error`: `1.557198 -> 1.429654`
- verify `piece_feature_l1_error`: `1.629223 -> 1.453196`
- verify `rule_feature_l1_error`: `1.228122 -> 1.210871`

The `structured_v3_v1` follow-up then tests whether delta supervision can be added without moving reconstruction onto the delta path itself:

- verify `feature_l1_error`: `1.425074 -> 1.353977`
- verify `piece_feature_l1_error`: `1.453196 -> 1.408795`
- verify `rule_feature_l1_error`: `1.210871 -> 1.086872`
- verify `drift_feature_l1_error`: `1.429654 -> 1.47778`

So `structured_v3_v1` is promising as a one-step improvement, but it is not the new default because drift gets slightly worse than `structured_v2_latent_v1`.

The explicit drift-supervision `structured_v4_v1` follow-up does not rescue that tradeoff:

- verify `feature_l1_error`: `1.425074 -> 1.611914`
- verify `drift_feature_l1_error`: `1.429654 -> 1.49735`

That makes it useful as a checked negative result, but not as a new baseline.

The `structured_v5_v1` follow-up then aligns Phase 6 with the symbolic proposer contract by feeding the exact selected-move candidate features into the transition path:

- verify `feature_l1_error`: `1.425074 -> 1.404499`
- verify `piece_feature_l1_error`: `1.453196 -> 1.38897`
- verify `rule_feature_l1_error`: `1.210871 -> 1.085876`
- verify `drift_feature_l1_error`: `1.429654 -> 1.556962`

So the symbolic action side input helps one-step local reconstruction, but it still gives back too much drift quality to replace `structured_v2_latent_v1` as the default.

The parallel local edit-target arm `edit_v1` shows the opposite tradeoff:

- verify `feature_l1_error`: `1.425823 -> 0.349443`
- verify `drift_feature_l1_error`: `1.557198 -> 13.251525`

That is informative, but not acceptable for the current Phase-6 default. `edit_v1` is therefore kept as an experimental reference only.

But the hard limit is unchanged:

- exact next-state accuracy is still `0.0`
- multi-step drift remains measurable but weak

This is enough to establish the Phase-6 dataset, training, export, and Rust-boundary plumbing, but not enough to call the dynamics model good.

## Next Model Pressure

The obvious next pressures are now:

- a better local-transition target that keeps `structured_v3_v1`'s one-step gains without giving back drift
- a better way to use multi-step supervision than the current `structured_v4_v1` rollout-loss formulation
- potentially partial-state or tokenized reconstruction instead of one flat feature regression target
- stronger multi-step drift supervision beyond the current short held-out slice

Those changes should stay action-conditioned and must not drift toward any hidden symbolic search or fallback evaluator.
