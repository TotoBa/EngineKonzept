# Phase 6 Dynamics Bundles

This directory contains exported latent-dynamics bundles.

## Current Default

- [v1](/home/torsten/EngineKonzept/models/dynamics/v1)
  First action-conditioned latent-dynamics baseline exported from [phase6_dynamics_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_v1.json).

## Experimental Variants

- [structured_v2_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_v1)
  First structured dynamics follow-up with separate piece/square/rule decoder heads. It improves soft reconstruction metrics over `v1`, but exact next-state accuracy is still `0.0`.

This is the current externally checkable Phase-6 reference bundle. It establishes the bundle contract and the first measured baseline; it is not yet a strong exact next-state model.
