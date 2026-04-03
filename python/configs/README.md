# Phase 5/6/7/8 Config Guide

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

- [phase6_dynamics_merged_unique_structured_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_merged_unique_structured_v6_v1.json)
  Current preferred Phase-6 config. Runs the `TransitionContextV1` dynamics arm on the merged unique `110,570 / 12,286 / 2,169` corpus and is the best measured large-corpus dynamics path so far.
- [phase6_dynamics_structured_v2_latent_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_latent_v1.json)
  Previous smaller-corpus preferred Phase-6 config: drift-aware structured decoder plus auxiliary latent-consistency supervision.
- [phase6_dynamics_structured_v2_drift_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_drift_v1.json)
  Earlier preferred Phase-6 config: structured piece/square/rule decoder plus explicit held-out drift-slice checkpoint selection.
- [phase6_dynamics_structured_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_v1.json)
  First structured dynamics follow-up with separate piece/square/rule decoder heads and section-wise reconstruction weights.
- [phase6_dynamics_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_v1.json)
  Original flat decoder baseline for the first action-conditioned latent-dynamics plumbing.

## Phase 6 Experimental Variants

- [phase6_dynamics_merged_unique_structured_v3_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_merged_unique_structured_v3_v1.json)
  Large-corpus rerun of the delta-auxiliary structured arm. On the merged unique corpus it beats the old large `structured_v2_latent` baseline on both one-step and drift, but still trails the new symbolic-action large-corpus default.
- [phase6_dynamics_merged_unique_structured_v5_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_merged_unique_structured_v5_v1.json)
  Previous large-corpus symbolic-action default. Still a strong reference point, but now slightly behind the large `structured_v6` transition-context run on both one-step error and drift.
- [phase6_dynamics_structured_v3_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v3_v1.json)
  Latent-stable structured follow-up with auxiliary delta supervision. Better one-step reconstruction than the current default, but slightly worse drift.
- [phase6_dynamics_structured_v4_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v4_v1.json)
  Explicit short-horizon drift-supervision follow-up. Useful as a checked negative result; it does not beat the current default or `structured_v3_v1`.
- [phase6_dynamics_structured_v5_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v5_v1.json)
  Symbolic-action follow-up. Keeps the latent-consistency baseline and adds exact selected-move symbolic features aligned with the current symbolic proposer contract.
- [phase6_dynamics_structured_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v6_v1.json)
  First `TransitionContextV1` follow-up on the smaller `10k` corpus. It improves both feature-L1 and drift over `structured_v5_v1` there and is now backed by the promoted large-corpus `structured_v6` default.
- [phase6_dynamics_edit_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_edit_v1.json)
  Local edit-target dynamics arm. Very strong one-step reconstruction, but currently unacceptable drift.

## Phase 7 Baseline

- [phase7_opponent_merged_unique_mlp_v1.json](/home/torsten/EngineKonzept/python/configs/phase7_opponent_merged_unique_mlp_v1.json)
  First explicit learned `OpponentHeadV1` run over the merged-unique workflow slices. It is the current trained Phase-7 reference, but it remains below the symbolic reply-scorer baseline on held-out reply ranking.
- [phase7_opponent_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase7_opponent_corpus_suite_set_v2_v1.json)
  Current preferred Phase-7 config. Trains the stronger `set_v2` opponent head over the `10k`, `122k`, and `400k` workflow tiers together and is the first learned opponent arm that beats the symbolic reply baseline on the current multi-corpus holdout.

## Phase 8 Baseline

- [phase8_planner_corpus_suite_set_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v1.json)
  First materialized Phase-8 planner config. Trains the initial bounded planner head over the `10k`, `122k`, and `400k` planner-workflow tiers.
- [phase8_planner_corpus_suite_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_v1.json)
  Current preferred Phase-8 config. Keeps the same bounded multi-corpus workflow suite but adds richer teacher-value and teacher-gap auxiliary targets on top of the planner ranking objective.
- [phase8_planner_corpus_suite_set_v3_two_tier_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v3_two_tier_v1.json)
  First latent-state planner follow-up on the filtered `10k + 122k` workflow slice. Useful as the current negative reference for planner-facing Phase-6 latent integration; it does not beat `set_v2` on that slice.
