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
- Pi-labeled 10k training corpus: [phase5_stockfish_pgn_train_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1)
- Pi-labeled 2k verify corpus: [phase5_stockfish_pgn_verify_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1)
- 10k comparison summary: [stockfish_pgn_pi_10k_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_compare_v1.json)

## Current findings

- legality is materially easier for the current MLP than policy
- reducing batch size from `256` to `128` improved validation and verify quality on the same 10k corpus
- increasing hidden width to `256` improved legal-set F1 further, but did not materially improve policy top-1 accuracy

These findings suggest that raw capacity helps, but the current flat MLP is likely not sufficient by itself for strong policy learning.

## Non-goals still preserved

- no latent dynamics model
- no opponent module
- no recurrent planner
- no UCI runtime integration of the learned proposer yet
- no classical search or evaluation fallback
