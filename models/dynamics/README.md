# Phase 6 Dynamics Bundles

This directory contains exported latent-dynamics bundles.

## Current Default

- [structured_v2_latent_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_latent_v1)
  Current preferred Phase-6 bundle exported from [phase6_dynamics_structured_v2_latent_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_latent_v1.json). It keeps the structured decoder and drift-aware selection, then adds auxiliary latent-consistency supervision.
- [structured_v2_drift_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_drift_v1)
  Previous preferred Phase-6 bundle exported from [phase6_dynamics_structured_v2_drift_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_drift_v1.json). It keeps the structured decoder and selects checkpoints against an explicit held-out drift slice.

## Experimental Variants

- [structured_v3_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v3_v1)
  Latent-stable structured follow-up with auxiliary delta heads. Better one-step soft metrics than the current default, but slightly worse drift.
- [edit_v1](/home/torsten/EngineKonzept/models/dynamics/edit_v1)
  Experimental local edit-target dynamics bundle. It is useful as a one-step-strong counterexample, but not acceptable as the current default because drift is much worse.
- [structured_v2_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_v1)
  First structured dynamics follow-up with separate piece/square/rule decoder heads.
- [v1](/home/torsten/EngineKonzept/models/dynamics/v1)
  Original flat action-conditioned latent-dynamics baseline.

The current externally checkable Phase-6 reference bundle is still a soft-reconstruction baseline, not a strong exact next-state model.
