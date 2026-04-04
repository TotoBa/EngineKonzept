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
- [phase8_planner_corpus_suite_set_v2_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_expanded_v1.json)
  Expanded-data follow-up for the same planner line. It improves the full mixed-suite training and validation picture over the earlier `set_v2` run, but it does not beat the older two-tier `set_v2` reference on the preferred `10k + 122k` slice.
- [phase8_planner_corpus_suite_set_v2_wide_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_wide_expanded_v1.json)
  Wider expanded-data follow-up. Useful as the current negative width reference; it does not improve over the narrower expanded `set_v2`.
- [phase8_planner_corpus_suite_set_v5_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v5_expanded_v1.json)
  Expanded-data multi-head self-attention planner arm. It becomes competitive again on the filtered `10k + 122k` slice, but still does not clearly beat the older two-tier `set_v2` reference.
- [phase8_planner_corpus_suite_set_v2_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v2_10k_122k_expanded_v1.json)
  Current preferred filtered Phase-8 follow-up. Reuses the stronger expanded workflow material on just the `10k + 122k` tiers and is the first rerun that improves the preferred filtered planner metrics over the older two-tier `set_v2` reference.
- [phase8_planner_corpus_suite_set_v3_two_tier_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v3_two_tier_v1.json)
  First latent-state planner follow-up on the filtered `10k + 122k` workflow slice. Useful as the current negative reference for planner-facing Phase-6 latent integration; it does not beat `set_v2` on that slice.
- [phase8_planner_corpus_suite_set_v3_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v3_10k_122k_expanded_v1.json)
  Stronger latent-state follow-up on the filtered expanded `10k + 122k` workflow material. It improves over the older latent `set_v3` baseline, but still does not beat the current filtered `set_v2_10k_122k_expanded` reference on `top1` or `MRR`.
- [phase8_planner_corpus_suite_set_v6_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_10k_122k_expanded_v1.json)
  Richer-target score-aux follow-up on the same filtered expanded `10k + 122k` slice. It keeps the stronger `set_v2` backbone and adds an auxiliary bounded candidate-score regression head fed by restricted `teacher_candidate_scores_cp`. On held-out verify it becomes a competitive experimental arm with slightly better `MRR` than `set_v2`, but it does not take the `top1` lead.
- [phase8_planner_corpus_suite_set_v6_margin_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_margin_10k_122k_expanded_v1.json)
  Next filtered follow-up over the same `10k + 122k` slice. It keeps `set_v6`, lowers the raw score-regression weight, and adds explicit `top1-vs-top2/top3` margin supervision over the bounded candidate slice. It improves score-target stability sharply, but still does not beat the current filtered `set_v2` planner on held-out `top1`.
- [phase8_planner_corpus_suite_set_v6_rank_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_rank_10k_122k_expanded_v1.json)
  Next filtered same-backbone follow-up over the same `10k + 122k` slice. It keeps `set_v6`, lowers the raw score-regression weight further, and replaces the most aggressive score shaping with a discrete bounded candidate rank-bucket auxiliary target over `top1`, `top2/top3`, and `tail`. It is a useful contract experiment, but it underperforms `set_v2` and the earlier score-aux `set_v6` on held-out planner ranking.
- [phase8_planner_corpus_suite_recurrent_v1_10k_122k_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_recurrent_v1_10k_122k_expanded_v1.json)
  First bounded recurrent planner follow-up on the same filtered `10k + 122k` slice. It keeps the existing planner-head contract intact and adds small configurable `memory_slots` plus `deliberation_steps`, so recurrence can be tested without another workflow-schema break. The first held-out rerun makes recurrence a real reusable capability, but it does not beat the filtered `set_v2` reference on `top1` or `MRR`.
- [phase8_planner_corpus_suite_set_v6_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_expanded_v1.json)
  First 400k-ready score-aux expanded rerun config. It now points at the single current `planner_workflow_fulltargets_expanded_v2` contract and is the current best `top1` arm on the materialized full expanded `10k + 122k + 400k` verify suite.
- [phase8_planner_corpus_suite_set_v6_margin_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_margin_expanded_v1.json)
  400k-ready margin-aux expanded rerun config over the same full-target workflow root. On the materialized full expanded suite it is effectively tied for best `MRR` and is the strongest `top3` rerun.
- [phase8_planner_corpus_suite_set_v6_rank_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_rank_expanded_v1.json)
  400k-ready rank-aux expanded rerun config over the same full-target workflow root. On the materialized full expanded suite it is the current best `MRR` rerun.
- [phase8_planner_corpus_suite_set_v6_rank_replay_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_rank_replay_expanded_v1.json)
  First replay-driven planner fine-tune. Warm-starts from the full expanded `set_v6_rank` checkpoint, mixes in the replay-derived planner-head artifact from Phase 9, and keeps the same expanded held-out validation contract.
- [phase8_planner_corpus_suite_set_v6_replay_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_replay_expanded_v2.json)
  Replay-only warm-start mirror over the stronger `Phase 9` expanded-v2 replay source. Starts from the full expanded `set_v6` checkpoint and fine-tunes only on the replay planner-head artifact while keeping the same expanded held-out validation contract.
- [phase8_planner_corpus_suite_set_v6_margin_replay_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_set_v6_margin_replay_expanded_v2.json)
  Replay-only warm-start mirror over the same stronger `Phase 9` replay source, but starting from the full expanded `set_v6_margin` checkpoint.
- [phase8_planner_corpus_suite_recurrent_v1_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase8_planner_corpus_suite_recurrent_v1_expanded_v1.json)
  400k-ready recurrent expanded rerun config over the same full-target workflow root. It now materializes a real expanded recurrent baseline, but still trails the stronger non-recurrent `set_v6` family on the same full expanded holdout.

## Phase 9 Agent Specs

- [phase9_agent_symbolic_root_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_symbolic_root_v1.json)
  Minimal symbolic root-only selfplay baseline. Useful as the simplest exact legal-move selector in later arena or regression runs.
- [phase9_agent_planner_set_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v2_v1.json)
  Current preferred small selfplay agent. Uses the official symbolic proposer, the learned `OpponentHeadV1` default, and the bounded `set_v2_10k_122k_expanded` planner.
- [phase9_agent_planner_set_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_v1.json)
  Experimental score-aux selfplay agent over the same symbolic proposer and learned opponent contract.
- [phase9_agent_planner_set_v6_margin_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_margin_v1.json)
  Experimental margin-aux selfplay agent.
- [phase9_agent_planner_set_v6_rank_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_rank_v1.json)
  Experimental rank-bucket selfplay agent.
- [phase9_agent_planner_recurrent_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_recurrent_v1.json)
  Experimental recurrent selfplay agent over the same bounded planner contract.
- [phase9_agent_planner_set_v2_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v2_expanded_v1.json)
  Post-400k baseline selfplay agent spec. It still points at the older expanded `set_v2` planner checkpoint and now serves as the launch baseline, not the strongest full expanded rerun.
- [phase9_agent_planner_set_v6_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_expanded_v1.json)
  Post-400k experimental score-aux selfplay agent spec.
- [phase9_agent_planner_set_v6_margin_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_margin_expanded_v1.json)
  Post-400k experimental margin-aux selfplay agent spec.
- [phase9_agent_planner_set_v6_rank_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_rank_expanded_v1.json)
  Post-400k experimental rank-aux selfplay agent spec.
- [phase9_agent_planner_recurrent_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_recurrent_expanded_v1.json)
  Post-400k experimental recurrent selfplay agent spec.
- [phase9_agent_planner_set_v6_replay_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_set_v6_replay_expanded_v2.json)
  Replay-mirror challenger over the stronger `Phase 9` replay source. Starts from the replay-only `set_v6` mirror and stays available as a direct experimental selfplay arm.
- [phase9_agent_planner_active_expanded_v2.json](/home/torsten/EngineKonzept/python/configs/phase9_agent_planner_active_expanded_v2.json)
  Current promoted expanded active selfplay agent spec. Points at the replay-promoted `set_v6_margin_replay_expanded_v2` planner checkpoint and replaces the older `set_v2_expanded` launch reference in the active slot.

## Phase 9 Arena Specs

- [phase9_arena_active_probe_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_probe_v1.json)
  Small reproducible active-only probe. Runs a color-swapped round-robin between the bounded `set_v2` agent and the symbolic root baseline.
- [phase9_arena_active_experimental_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_experimental_v1.json)
  Main future-facing Phase-9 arena suite. It keeps the contract versioned and lists the active plus experimental planner arms so later selfplay reruns can start without another orchestration rewrite.
- [phase9_arena_active_experimental_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_experimental_expanded_v1.json)
  Post-400k arena suite. Uses the future expanded planner checkpoints for the active arm and the currently tracked experimental follow-ups.
- [phase9_arena_active_experimental_replay_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_arena_active_experimental_replay_expanded_v1.json)
  Replay-aware post-promotion arena suite. Keeps the promoted expanded active arm plus the older expanded family and the new replay challenger in the same versioned round-robin contract.

Arena specs now also support an optional `max_plies_adjudication` block:

- `engine_path`
- one of `nodes`, `depth`, or `movetime_ms`
- `score_threshold_pawns`
- `extension_step_plies`
- `max_extensions`

This is intended for bounded offline selfplay only. It adjudicates only after a game reaches `max_plies`; inside the neutral band the game is extended, outside the band it is resolved by the engine judge.

Arena specs also support `parallel_workers` for offline process-level concurrency.
The intended use is one arena session per worker while the adjudicator engine stays at `Threads=1`.

## Phase 9 Campaigns

- [phase9_replay_campaign_active_expanded_v1.json](/home/torsten/EngineKonzept/python/configs/phase9_replay_campaign_active_expanded_v1.json)
  Current preferred long-run replay campaign. Starts from the promoted expanded active arm plus the replay challenger, runs the replay-aware expanded arena stage, rebuilds replay supervision, retrains the configured replay mirrors, and writes a held-out planner verify matrix.
