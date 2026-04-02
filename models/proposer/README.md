# Proposer Model Outputs

This directory stores exported Phase-5 proposer bundles.

Each model bundle contains:

- `checkpoint.pt`
- `proposer.pt2`
- `metadata.json`

## Current Default

Use this first for the current standard Phase-5 proposer path:

- [phase5_stockfish_pgn_current_default_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_current_default_v1.json)

That config writes its bundle to `models/proposer/stockfish_pgn_current_default_v1`.

## Experimental Variants

These bundles correspond to the main `10k` comparison grid:

- [stockfish_pgn_pi_10k_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_pi_10k_v1)
  Early `10k` batch-size reference point.
- [stockfish_pgn_pi_10k_bs128_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_pi_10k_bs128_v1)
  Best current speed/quality trade-off on the `10k` corpus.
- [stockfish_pgn_pi_10k_h192_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_pi_10k_h192_v1)
  Wider hidden-layer experimental variant.
- [stockfish_pgn_pi_10k_h256_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_pi_10k_h256_v1)
  Best verify legal-set F1 so far on the `10k` corpus.

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
