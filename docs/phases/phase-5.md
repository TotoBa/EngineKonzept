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
- a native Rust symbolic proposer scorer over exact legal candidates for the current `symbolic_v1` path
- an optional local IPC dataset-oracle daemon for reproducible batch builds without repeated process spawn overhead

## Current externally checkable artifacts

- small seed policy dataset: [policy_seed.jsonl](/home/torsten/EngineKonzept/tests/positions/policy_seed.jsonl)
- current standard training corpus: [phase5_stockfish_pgn_train_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1)
- current standard verify corpus: [phase5_stockfish_pgn_verify_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1)
- current default training config: [phase5_stockfish_pgn_symbolic_v1_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_symbolic_v1_v1.json)
- current default bundle: [stockfish_pgn_symbolic_v1_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_symbolic_v1_v1)
- current default training summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_symbolic_v1_v1/summary.json)
- current three-way comparison: [stockfish_pgn_10k_three_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_three_way_compare_v1.json)
- current four-way comparison with the structured multi-stream arm: [stockfish_pgn_10k_four_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_four_way_compare_v1.json)
- current five-way comparison with the first factorized decoder arm: [stockfish_pgn_10k_five_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_five_way_compare_v1.json)
- current six-way comparison with the conditional factorized decoder arm: [stockfish_pgn_10k_six_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_six_way_compare_v1.json)
- current seven-way comparison with the policy-stronger conditional decoder arm: [stockfish_pgn_10k_seven_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_seven_way_compare_v1.json)
- direct checkpoint-selection comparison for `factorized_v5`: [stockfish_pgn_factorized_v5_selection_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_factorized_v5_selection_compare_v1.json)
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
- explicit balanced checkpoint selection for `factorized_v5` further improved policy on the held-out verify split (`0.015137` vs `0.014648`), but it reduced legal-set F1 sharply (`0.029989` vs `0.06438`)
- `factorized_v6` is now the strongest legality arm on the `10k` corpus with verify `legal_set_f1=0.123078`, but it still trails the current default on policy
- `relational_v1` keeps the stronger typed backbone and reaches verify `policy_top1_accuracy=0.01416`, which is better than the earlier factorized arms except `factorized_v5`, but still below `current_default`
- `symbolic_v1` replaces learned legality with exact legal-candidate generation plus a learned scorer and reaches verify `policy_top1_accuracy=0.127441` with exact legality on the same `10k` corpus

The current Phase-5 architecture decision is therefore:

- keep `symbolic_v1` as the standard path
- keep the old learned `current_default` MLP as a legacy baseline
- keep `h256` as the best old learned legal-F1 reference
- keep `multistream_v2` as the first structured baseline
- keep `factorized_v3` as an explicit negative baseline
- keep `factorized_v6` as the best current legality arm
- keep `factorized_v5` as the clearest checkpoint-selection tradeoff example inside the factorized line
- keep `relational_v1` as the current typed-backbone policy-path reference
- treat the learned-legality proposer family as legacy experimental baselines
- keep checkpoint selection explicit rather than implicit
- accept the symbolic-candidate proposer line as the official proposer direction until a stronger replacement appears

These findings suggest that raw capacity helps, but the current flat MLP is likely not sufficient by itself for strong policy learning.

## Non-goals still preserved

- no latent dynamics model
- no opponent module
- no recurrent planner
- no planner-driven runtime integration beyond single-step symbolic proposer scoring
- no classical search or evaluation fallback
