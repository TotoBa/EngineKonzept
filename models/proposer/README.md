# Proposer Model Outputs

This directory stores exported Phase-5 proposer bundles.

Each model bundle contains:

- `checkpoint.pt`
- `proposer.pt2`
- `metadata.json`

## Current Default

Use this first for the current standard Phase-5 proposer path:

- [phase5_stockfish_pgn_current_default_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_current_default_v1.json)

Materialized current-default bundle:

- [stockfish_pgn_current_default_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_current_default_v1)

## Experimental Variants

These bundles correspond to the main `10k` comparison grid:

- [stockfish_pgn_pi_10k_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_pi_10k_v1)
  Early `10k` batch-size reference point.
- [stockfish_pgn_pi_10k_bs128_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_pi_10k_bs128_v1)
  Best current speed/quality trade-off on the `10k` corpus.
- [stockfish_pgn_policy_focus_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_policy_focus_v1)
  Isolated policy-focus run on the same `10k` corpus.
- [stockfish_pgn_multistream_v2_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_multistream_v2_v1)
  First structured multi-stream proposer over the same export contract.
- [stockfish_pgn_factorized_v3_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_factorized_v3_v1)
  First additive factorized-decoder proposer; kept as an explicit failed-but-informative baseline.
- [stockfish_pgn_factorized_v4_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_factorized_v4_v1)
  Conditional factorized-decoder proposer; current best legality-focused Phase-5 arm.
- [stockfish_pgn_factorized_v5_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_factorized_v5_v1)
  Policy-stronger conditional factorized-decoder proposer; current best balance within the factorized line.
- [stockfish_pgn_factorized_v6_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_factorized_v6_v1)
  Pairwise-coupled conditional factorized proposer; current best legality-focused Phase-5 arm.
- [stockfish_pgn_relational_v1_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_relational_v1_v1)
  Typed relational backbone plus stronger factorized heads; current relational policy-path baseline.
- [stockfish_pgn_pi_10k_h192_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_pi_10k_h192_v1)
  Wider hidden-layer experimental variant.
- [stockfish_pgn_pi_10k_h256_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_pi_10k_h256_v1)
  Earlier wide-MLP legal-F1 reference point.

## Legacy Baselines

These remain useful as historical reference points and small-regression baselines:

- [v1](/home/torsten/EngineKonzept/models/proposer/v1)
  Original Phase-5 proposer smoke bundle over the Phase-4 dataset.
- [policy_seed_v1](/home/torsten/EngineKonzept/models/proposer/policy_seed_v1)
  Tiny seed-policy supervision run.
- [stockfish_pgn_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_v1)
  Early small local Stockfish corpus run.
- [stockfish_pgn_pi_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_pi_v1)
  Early small Raspberry-Pi-labeled Stockfish corpus run.

## Benchmark Artifacts

The `bench_*` bundles are throughput-only training benchmarks and should not be treated as preferred model checkpoints:

- [bench_bs64_t7](/home/torsten/EngineKonzept/models/proposer/bench_bs64_t7)
- [bench_bs128_t7](/home/torsten/EngineKonzept/models/proposer/bench_bs128_t7)
- [bench_bs256_t7](/home/torsten/EngineKonzept/models/proposer/bench_bs256_t7)
- [bench_bs128_t4](/home/torsten/EngineKonzept/models/proposer/bench_bs128_t4)
- [bench_bs256_t4](/home/torsten/EngineKonzept/models/proposer/bench_bs256_t4)
- [bench_bs512_t4](/home/torsten/EngineKonzept/models/proposer/bench_bs512_t4)
