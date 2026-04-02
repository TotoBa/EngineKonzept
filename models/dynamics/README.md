# Phase 6 Dynamics Bundles

This directory contains exported latent-dynamics bundles.

## Current Default

- [dynamics_merged_unique_structured_v5_v1](/home/torsten/EngineKonzept/models/dynamics/dynamics_merged_unique_structured_v5_v1)
  Current preferred Phase-6 bundle exported from [phase6_dynamics_merged_unique_structured_v5_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_merged_unique_structured_v5_v1.json). It runs the symbolic-action dynamics arm on the merged unique `110,570 / 12,286 / 2,169` corpus and is the best measured dynamics bundle so far.
- [structured_v2_latent_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_latent_v1)
  Previous smaller-corpus preferred Phase-6 bundle exported from [phase6_dynamics_structured_v2_latent_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_latent_v1.json). It keeps the structured decoder and drift-aware selection, then adds auxiliary latent-consistency supervision.
- [structured_v2_drift_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_drift_v1)
  Previous preferred Phase-6 bundle exported from [phase6_dynamics_structured_v2_drift_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_drift_v1.json). It keeps the structured decoder and selects checkpoints against an explicit held-out drift slice.

## Experimental Variants

- [dynamics_merged_unique_structured_v3_v1](/home/torsten/EngineKonzept/models/dynamics/dynamics_merged_unique_structured_v3_v1)
  Large-corpus rerun of the delta-auxiliary structured arm. It beats the old large `structured_v2_latent` baseline on both verify feature error and verify drift, but remains behind the large symbolic-action default.
- [structured_v3_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v3_v1)
  Latent-stable structured follow-up with auxiliary delta heads. Better one-step soft metrics than the current default, but slightly worse drift.
- [structured_v4_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v4_v1)
  Explicit short-horizon drift-supervision follow-up. It does not beat the current default on held-out drift.
- [structured_v5_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v5_v1)
  Symbolic-action follow-up. It consumes the selected move's exact symbolic candidate features from the proposer contract, improves one-step reconstruction, and still remains experimental because drift is worse than the current default.
- [structured_v6_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v6_v1)
  First `TransitionContextV1` dynamics bundle. It validates the richer selected-action transition contract on the `10k` corpus and improves both feature-L1 and drift over `structured_v5_v1`, but it has not yet displaced the large-corpus `structured_v5` default.
- [edit_v1](/home/torsten/EngineKonzept/models/dynamics/edit_v1)
  Experimental local edit-target dynamics bundle. It is useful as a one-step-strong counterexample, but not acceptable as the current default because drift is much worse.
- [structured_v2_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_v1)
  First structured dynamics follow-up with separate piece/square/rule decoder heads.
- [v1](/home/torsten/EngineKonzept/models/dynamics/v1)
  Original flat action-conditioned latent-dynamics baseline.

The current externally checkable Phase-6 reference bundle is still a soft-reconstruction baseline, not a strong exact next-state model.
