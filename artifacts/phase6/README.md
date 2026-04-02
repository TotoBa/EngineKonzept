# Phase 6 Artifacts

This directory stores materialized latent-dynamics runs and held-out evaluation outputs.

## Current Default

- [dynamics_structured_v2_drift_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_drift_v1)
  Training summary for the current preferred structured Phase-6 run with explicit drift-slice checkpoint selection.
- [dynamics_structured_v2_drift_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_drift_v1_verify.json)
  Held-out verify evaluation for the same bundle on the current standard Phase-5 verify corpus.
- [dynamics_structured_v2_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_compare_v1.json)
  Direct comparison between the flat `v1`, structured `v2`, and drift-aware `structured_v2_drift` arms.

## Experimental Variants

- [dynamics_structured_v2_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1)
  Training summary for the first structured piece/square/rule dynamics follow-up.
- [dynamics_structured_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1_verify.json)
  Held-out verify evaluation for the same follow-up run.
- [dynamics_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1)
  Original flat decoder baseline summary.
- [dynamics_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1_verify.json)
  Held-out verify evaluation for the original baseline.

The current baseline is useful because it makes one-step and drift metrics externally checkable. It should still be read as a plumbing baseline, not as a final dynamics design.
