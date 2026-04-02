# Phase 6 Artifacts

This directory stores materialized latent-dynamics runs and held-out evaluation outputs.

## Current Default

- [dynamics_structured_v2_latent_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_latent_v1)
  Training summary for the current preferred structured Phase-6 run with auxiliary latent-consistency supervision.
- [dynamics_structured_v2_latent_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_latent_v1_verify.json)
  Held-out verify evaluation for the current preferred bundle on the current standard Phase-5 verify corpus.
- [dynamics_structured_v2_drift_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_drift_v1)
  Training summary for the previous preferred structured Phase-6 run with explicit drift-slice checkpoint selection.
- [dynamics_structured_v2_drift_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_drift_v1_verify.json)
  Held-out verify evaluation for the previous preferred bundle on the current standard Phase-5 verify corpus.
- [dynamics_phase6_parallel_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_phase6_parallel_compare_v1.json)
  Direct comparison between the previous drift-aware default, the new latent-consistency default, and the experimental `edit_v1` arm.
- [dynamics_phase6_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_phase6_compare_v2.json)
  Direct comparison that adds `structured_v3_v1` on top of the previous Phase-6 arms.
- [dynamics_phase6_compare_v3.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_phase6_compare_v3.json)
  Direct comparison that adds `structured_v4_v1` on top of the earlier Phase-6 arms.

## Experimental Variants

- [dynamics_structured_v3_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1)
  Training summary for the latent-stable plus delta-auxiliary follow-up.
- [dynamics_structured_v3_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1_verify.json)
  Held-out verify evaluation for the same arm.
- [dynamics_structured_v4_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1)
  Training summary for the explicit drift-supervision follow-up.
- [dynamics_structured_v4_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1_verify.json)
  Held-out verify evaluation for the same arm.
- [dynamics_edit_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1)
  Training summary for the experimental local edit-target arm.
- [dynamics_edit_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1_verify.json)
  Held-out verify evaluation for the same arm. Useful because it shows the one-step-vs-drift tradeoff explicitly.
- [dynamics_structured_v2_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1)
  Training summary for the first structured piece/square/rule dynamics follow-up.
- [dynamics_structured_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1_verify.json)
  Held-out verify evaluation for the same follow-up run.
- [dynamics_v1](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1)
  Original flat decoder baseline summary.
- [dynamics_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_v1_verify.json)
  Held-out verify evaluation for the original baseline.

The current baseline is useful because it makes one-step and drift metrics externally checkable. It should still be read as a plumbing baseline, not as a final dynamics design.
