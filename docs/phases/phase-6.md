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
- optional symbolic selected-move action features aligned with the current symbolic proposer contract

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

The next structured-delta-auxiliary experimental run is:

- config: [phase6_dynamics_structured_v3_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v3_v1.json)
- bundle: [structured_v3_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v3_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1/summary.json)
- verify: [dynamics_structured_v3_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1_verify.json)

The next drift-supervised experimental run is:

- config: [phase6_dynamics_structured_v4_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v4_v1.json)
- bundle: [structured_v4_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v4_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1/summary.json)
- verify: [dynamics_structured_v4_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1_verify.json)

The next symbolic-action experimental run is:

- config: [phase6_dynamics_structured_v5_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v5_v1.json)
- bundle: [structured_v5_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v5_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v5_v1/summary.json)
- verify: [dynamics_structured_v5_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v5_v1_verify.json)

The next transition-context experimental run is:

- config: [phase6_dynamics_structured_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v6_v1.json)
- bundle: [structured_v6_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v6_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v6_v1/summary.json)
- verify: [dynamics_structured_v6_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v6_v1_verify.json)
- direct comparison: [dynamics_phase6_compare_v5.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_phase6_compare_v5.json)

The larger-corpus reruns now provide the preferred Phase-6 reference:

- config: [phase6_dynamics_merged_unique_structured_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_merged_unique_structured_v6_v1.json)
- bundle: [dynamics_merged_unique_structured_v6_v1](/home/torsten/EngineKonzept/models/dynamics/dynamics_merged_unique_structured_v6_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_structured_v6_v1/summary.json)
- verify: [dynamics_merged_unique_structured_v6_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_structured_v6_v1_verify.json)
- direct comparison: [dynamics_merged_unique_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_compare_v2.json)

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

The next `structured_v3_v1` arm improves one-step reconstruction again, but not enough on drift to replace the default:

- verify `feature_l1_error`: `1.425074 -> 1.353977`
- verify `drift_feature_l1_error`: `1.429654 -> 1.47778`

The explicit drift-supervision `structured_v4_v1` arm also fails to replace the default:

- verify `feature_l1_error`: `1.425074 -> 1.611914`
- verify `drift_feature_l1_error`: `1.429654 -> 1.49735`

The `structured_v5_v1` arm binds Phase 6 to the symbolic proposer-side move contract:

- verify `feature_l1_error`: `1.425074 -> 1.404499`
- verify `piece_feature_l1_error`: `1.453196 -> 1.38897`
- verify `rule_feature_l1_error`: `1.210871 -> 1.085876`
- verify `drift_feature_l1_error`: `1.429654 -> 1.556962`

That makes it a useful new experimental arm, but not the new default: the symbolic move-side features clearly help local one-step reconstruction, yet drift remains worse than `structured_v2_latent_v1`.

The `structured_v6_v1` follow-up is the first Phase-6 arm to consume `TransitionContextV1` directly:

- verify `feature_l1_error`: `1.404499 -> 1.398971` versus `structured_v5_v1`
- verify `drift_feature_l1_error`: `1.556962 -> 1.487676` versus `structured_v5_v1`

That is enough to validate the richer transition contract as a real modeling path, but not enough to flip the overall Phase-6 default:

- it still trails `structured_v2_latent_v1` on `10k`-corpus drift (`1.429654`)
- it has not yet been rerun on the larger merged unique corpus where the current default `structured_v5` was selected

The larger `merged_unique` reruns overturn that smaller-corpus conclusion:

- `structured_v2_latent_v1` on `110,570 / 12,286 / 2,169`: verify `feature_l1_error=1.067843`, `drift_feature_l1_error=6.305117`
- `structured_v3_v1` on the same corpus: verify `feature_l1_error=1.02784`, `drift_feature_l1_error=6.18409`
- `structured_v5_v1` on the same corpus: verify `feature_l1_error=0.924808`, `drift_feature_l1_error=1.548861`
- `structured_v6_v1` on the same corpus: verify `feature_l1_error=0.923791`, `drift_feature_l1_error=1.464848`

So the current preferred Phase-6 reference is now the large-corpus transition-context run [dynamics_merged_unique_structured_v6_v1](/home/torsten/EngineKonzept/models/dynamics/dynamics_merged_unique_structured_v6_v1), not the earlier `10k`-corpus `structured_v2_latent_v1` and not the earlier large-corpus `structured_v5` symbolic-action default.

The parallel `edit_v1` arm is informative but remains experimental:

- verify `feature_l1_error`: `1.425823 -> 0.349443`
- verify `drift_feature_l1_error`: `1.557198 -> 13.251525`

That means `edit_v1` is currently useful as a diagnostic counterexample, not as a planner-facing dynamics default.

But even the preferred arm still does **not** recover exact packed next states.

That means the repo has crossed from “Phase 6 placeholder” into “first checkable dynamics baseline”, but the modeling work is still ahead.
