# Phase 5

## Goal

Train and export the first legality/policy proposer without introducing dynamics, opponent modeling, planner logic, or classical search.

## Deliverables in this repository state

- a PyTorch proposer under `python/train/models/proposer.py`
- config-driven training under `python/scripts/train_legality.py`
- deterministic dataset-artifact loading and feature packing for proposer supervision
- held-out legal-set precision/recall/F1 and policy top-1 metrics
- measured training throughput in examples/second
- offline PGN sampling plus bounded Stockfish 18 labeling for larger policy datasets
- lean proposer split artifacts for larger policy datasets, with loader fallback to the canonical full dataset JSONL
- a `torch.export` + metadata export bundle
- a Rust loader for the exported proposer bundle under `rust/crates/inference`
- an optional local IPC dataset-oracle daemon for reproducible batch builds without repeated process spawn overhead

## Current externally checkable artifacts

- small seed policy dataset: [policy_seed.jsonl](/home/torsten/EngineKonzept/tests/positions/policy_seed.jsonl)
- current standard training corpus: [phase5_stockfish_pgn_train_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1)
- current standard verify corpus: [phase5_stockfish_pgn_verify_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1)
- current default training config: [phase5_stockfish_pgn_current_default_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_current_default_v1.json)
- current default bundle: [stockfish_pgn_current_default_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_current_default_v1)
- current default training summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_current_default_v1/summary.json)
- current three-way comparison: [stockfish_pgn_10k_three_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_three_way_compare_v1.json)
- current four-way comparison with the structured multi-stream arm: [stockfish_pgn_10k_four_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_four_way_compare_v1.json)
- current five-way comparison with the first factorized decoder arm: [stockfish_pgn_10k_five_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_five_way_compare_v1.json)
- current six-way comparison with the conditional factorized decoder arm: [stockfish_pgn_10k_six_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_six_way_compare_v1.json)
- current seven-way comparison with the policy-stronger conditional decoder arm: [stockfish_pgn_10k_seven_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_seven_way_compare_v1.json)
- earlier small local baseline corpus: [phase5_stockfish_pgn_train_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_v1)
- earlier small Pi baseline corpus: [phase5_stockfish_pgn_train_pi_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_pi_v1)
- 10k comparison summary: [stockfish_pgn_pi_10k_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_compare_v1.json)

The `*_pi_10k_v1` corpora are the current default Phase-5 reference datasets for proposer work. The smaller `*_v1` and `*_pi_v1` Stockfish corpora remain in the repository as early baselines and regression fixtures, not as the preferred training starting point.

The same `current default` / `experimental variant` / `legacy baseline` split is now documented for:

- [python/configs/README.md](/home/torsten/EngineKonzept/python/configs/README.md)
- [models/proposer/README.md](/home/torsten/EngineKonzept/models/proposer/README.md)
- [artifacts/phase5/README.md](/home/torsten/EngineKonzept/artifacts/phase5/README.md)
- [model-roadmap.md](/home/torsten/EngineKonzept/docs/architecture/model-roadmap.md)

## Current findings

- legality is materially easier for the current MLP than policy
- reducing batch size from `256` to `128` improved validation and verify quality on the same 10k corpus
- increasing hidden width to `256` improved legal-set F1 further, but did not materially improve policy top-1 accuracy
- increasing `policy_loss_weight` to `2.0` on the same 128-wide default backbone did not beat the current default on verify policy accuracy and regressed legal-set F1
- the first structured `multistream_v2` arm slightly improved validation legal-set F1 over the `current_default` MLP, but it did not beat `h256` on legality, regressed policy accuracy, and was materially slower
- the first additive `factorized_v3` decoder drastically reduced parameter count, but collapsed held-out legality and policy quality, so the next decoder step must keep more coupling between move components
- the first conditional `factorized_v4` decoder became the best legal-set-F1 arm so far on the `10k` corpus, but it still trails the current default on policy accuracy
- the policy-stronger `factorized_v5` arm recovered a large part of the policy gap while still outperforming the old MLP baselines on legality, but it no longer beats `factorized_v4` on legal-set F1

The current Phase-5 architecture decision is therefore:

- keep `current_default` as the standard path
- keep `h256` as the best legal-F1 reference
- keep `multistream_v2` as the first structured baseline
- keep `factorized_v3` as an explicit negative baseline
- keep `factorized_v4` as the best current legality arm
- keep `factorized_v5` as the best current factorized balance arm
- prioritize checkpoint-selection and policy-coupling follow-up on top of the conditional factorized decoder line

These findings suggest that raw capacity helps, but the current flat MLP is likely not sufficient by itself for strong policy learning.

## Non-goals still preserved

- no latent dynamics model
- no opponent module
- no recurrent planner
- no UCI runtime integration of the learned proposer yet
- no classical search or evaluation fallback
