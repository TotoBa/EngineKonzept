# Phase 6

## Goal

Add the first action-conditioned latent dynamics model.

## Current repository state

The repository now includes the first Phase-6 baseline:

- a dedicated dynamics config schema in [config.py](/home/torsten/EngineKonzept/python/train/config.py)
- lean dynamics split artifacts derived from Phase-5 datasets
- a PyTorch dynamics model in [dynamics.py](/home/torsten/EngineKonzept/python/train/models/dynamics.py)
- config-driven training in [train_dynamics.py](/home/torsten/EngineKonzept/python/scripts/train_dynamics.py)
- held-out one-step and drift metrics in [dynamics.py](/home/torsten/EngineKonzept/python/train/trainers/dynamics.py)
- a `torch.export` dynamics bundle plus Rust-side metadata validation in [lib.rs](/home/torsten/EngineKonzept/rust/crates/inference/src/lib.rs)

The first materialized baseline run is:

- config: [phase6_dynamics_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_v1.json)
- bundle: [v1](/home/torsten/EngineKonzept/models/dynamics/v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1/summary.json)
- verify: [dynamics_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1_verify.json)

The first structured follow-up run is:

- config: [phase6_dynamics_structured_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_v1.json)
- bundle: [structured_v2_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1/summary.json)
- verify: [dynamics_structured_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1_verify.json)

The drift-aware follow-up established the first useful Phase-6 reference:

- config: [phase6_dynamics_structured_v2_drift_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_drift_v1.json)
- bundle: [structured_v2_drift_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_drift_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_drift_v1/summary.json)
- verify: [dynamics_structured_v2_drift_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_drift_v1_verify.json)

The current preferred latent-consistency follow-up run is:

- config: [phase6_dynamics_structured_v2_latent_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_latent_v1.json)
- bundle: [structured_v2_latent_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_latent_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_latent_v1/summary.json)
- verify: [dynamics_structured_v2_latent_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_latent_v1_verify.json)

The parallel local edit-target experimental run is:

- config: [phase6_dynamics_edit_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_edit_v1.json)
- bundle: [edit_v1](/home/torsten/EngineKonzept/models/dynamics/edit_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1/summary.json)
- verify: [dynamics_edit_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1_verify.json)

## What this baseline does

- encodes packed current-state features into a latent vector
- applies a residual action-conditioned transition in latent space
- decodes the resulting latent to packed next-state features
- measures exactness separately for special-move subsets and short multi-step drift chains

## What it does not do yet

- no opponent reasoning
- no planner integration
- no Rust-side learned execution beyond bundle loading and schema validation
- no exact next-state recovery yet

## Current findings

The `v1` baseline is good enough to establish the Phase-6 plumbing, but not yet good enough to count as a strong dynamics model:

- reconstruction loss decreases materially across training
- held-out feature-L1 error is stable and measurable
- exact next-state accuracy remains `0.0`
- drift metrics are now externally checkable, but still weak

The structured `v2` follow-up is the first real modeling improvement:

- lower validation and verify feature-L1 error
- lower verify drift error
- separate piece/square/rule reconstruction losses now visible

The drift-aware `structured_v2_drift_v1` follow-up improved the main verify soft metrics over `structured_v2_v1`:

- verify `feature_l1_error`: `1.433716 -> 1.425823`
- verify `drift_feature_l1_error`: `1.595053 -> 1.557198`

The parallel latent-consistency `structured_v2_latent_v1` follow-up is now preferred because it improves them again:

- verify `feature_l1_error`: `1.425823 -> 1.425074`
- verify `drift_feature_l1_error`: `1.557198 -> 1.429654`

The parallel `edit_v1` arm is informative but remains experimental:

- verify `feature_l1_error`: `1.425823 -> 0.349443`
- verify `drift_feature_l1_error`: `1.557198 -> 13.251525`

That means `edit_v1` is currently useful as a diagnostic counterexample, not as a planner-facing dynamics default.

But even the preferred arm still does **not** recover exact packed next states.

That means the repo has crossed from “Phase 6 placeholder” into “first checkable dynamics baseline”, but the modeling work is still ahead.
