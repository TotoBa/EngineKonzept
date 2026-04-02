# Phase 5/6 Config Guide

This directory contains the externally checkable proposer-training and dynamics-training configs.

## Current Default

Use this first for the current standard Phase-5 proposer path:

- [phase5_stockfish_pgn_symbolic_v1_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_symbolic_v1_v1.json)

It points at the current standard Pi-labeled `10,240 / 2,048` corpus and is the official symbolic-candidate proposer path.

The previous learned-legality default remains available as a legacy baseline:

- [phase5_stockfish_pgn_current_default_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_current_default_v1.json)

## Experimental Variants

Use these when comparing architecture or optimization changes against the current default corpus:

- [phase5_stockfish_pgn_pi_10k_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_pi_10k_v1.json)
  Early `10k` throughput-oriented point with `batch_size=256`.
- [phase5_stockfish_pgn_policy_focus_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_policy_focus_v1.json)
  Isolated policy-focus run on the same `10k` corpus with higher `policy_loss_weight` and lower learning rate.
- [phase5_stockfish_pgn_multistream_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_multistream_v2_v1.json)
  Structured multi-stream proposer that restores piece/square/rule streams inside the model.
- [phase5_stockfish_pgn_factorized_v3_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_factorized_v3_v1.json)
  First additive factorized-decoder baseline; useful as a negative reference for the next conditional decoder arm.
- [phase5_stockfish_pgn_factorized_v4_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_factorized_v4_v1.json)
  Conditional factorized-decoder arm; current best legal-set-F1 result on the `10k` corpus.
- [phase5_stockfish_pgn_factorized_v5_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_factorized_v5_v1.json)
  Policy-stronger conditional factorized-decoder arm; current best balance among the factorized variants.
- [phase5_stockfish_pgn_factorized_v5_balanced_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_factorized_v5_balanced_v1.json)
  Same `factorized_v5` architecture, but with `balanced` checkpoint selection and higher policy weight in epoch choice.
- [phase5_stockfish_pgn_factorized_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_factorized_v6_v1.json)
  Pairwise-coupled conditional factorized decoder; current best legality arm on the `10k` corpus.
- [phase5_stockfish_pgn_relational_v1_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_relational_v1_v1.json)
  Typed multi-stream backbone plus stronger factorized heads; current relational policy-path baseline.
- [phase5_stockfish_pgn_symbolic_v1_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_symbolic_v1_v1.json)
  Official symbolic-candidate proposer arm over the current `10k` corpus. Exact legality via legal-move generation, learned policy scoring only, and exported Rust-loadable bundle support.
- [phase5_stockfish_merged_unique_symbolic_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_merged_unique_symbolic_v1.json)
  Prepared larger-corpus follow-up for the same symbolic arm. Kept as the next scale-up config after the `10k` validation pass.
- [phase5_stockfish_pgn_pi_10k_h192_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_pi_10k_h192_v1.json)
  Wider hidden layer variant.
- [phase5_stockfish_pgn_pi_10k_h256_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_pi_10k_h256_v1.json)
  Earlier wide-MLP legal-F1 reference point.

## Legacy Baselines

These remain useful as small-corpus regression baselines, but they are no longer the preferred starting point:

- [phase5_stockfish_pgn_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_v1.json)
  Small local Stockfish-labeled corpus.
- [phase5_stockfish_pgn_pi_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_pi_v1.json)
  Small Raspberry-Pi-labeled Stockfish corpus.
- [phase5_proposer_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_proposer_v1.json)
  Original Phase-5 proposer smoke path over the Phase-4 dataset.
- [phase5_policy_seed_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_policy_seed_v1.json)
  Tiny manually curated seed set for policy supervision plumbing.

## Phase 6 Baseline

- [phase6_dynamics_structured_v2_latent_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_latent_v1.json)
  Current preferred Phase-6 config: drift-aware structured decoder plus auxiliary latent-consistency supervision.
- [phase6_dynamics_structured_v2_drift_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_drift_v1.json)
  Earlier preferred Phase-6 config: structured piece/square/rule decoder plus explicit held-out drift-slice checkpoint selection.
- [phase6_dynamics_structured_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_v1.json)
  First structured dynamics follow-up with separate piece/square/rule decoder heads and section-wise reconstruction weights.
- [phase6_dynamics_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_v1.json)
  Original flat decoder baseline for the first action-conditioned latent-dynamics plumbing.

## Phase 6 Experimental Variants

- [phase6_dynamics_structured_v3_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v3_v1.json)
  Latent-stable structured follow-up with auxiliary delta supervision. Better one-step reconstruction than the current default, but slightly worse drift.
- [phase6_dynamics_structured_v4_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v4_v1.json)
  Explicit short-horizon drift-supervision follow-up. Useful as a checked negative result; it does not beat the current default or `structured_v3_v1`.
- [phase6_dynamics_edit_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_edit_v1.json)
  Local edit-target dynamics arm. Very strong one-step reconstruction, but currently unacceptable drift.
