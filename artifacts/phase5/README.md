# Phase 5 Artifact Guide

This directory contains reproducible Phase-5 training summaries, benchmark outputs, and oracle-performance artifacts.

## Current Default

Use these first when reviewing the current standard proposer path:

- [stockfish_pgn_pi_10k_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_compare_v1.json)
  Main `10k` comparison summary for the Pi-labeled reference corpus.
- [proposer_artifact_bench_10k_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/proposer_artifact_bench_10k_v2.json)
  Full-vs-lean proposer artifact benchmark on the current `10k` corpus.

The current default config writes its training summary to:

- [stockfish_pgn_current_default_v1/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_current_default_v1/summary.json)

## Experimental Variants

These summaries correspond to the main `10k` proposer comparison runs:

- [stockfish_pgn_pi_10k_v1/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_v1/summary.json)
- [stockfish_pgn_pi_10k_bs128_v1/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_bs128_v1/summary.json)
- [stockfish_pgn_pi_10k_h192_v1/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_h192_v1/summary.json)
- [stockfish_pgn_pi_10k_h256_v1/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_h256_v1/summary.json)

## Legacy Baselines

These are still useful as early small-corpus or smoke-run references:

- [proposer_v1/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/proposer_v1/summary.json)
- [policy_seed_v1/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/policy_seed_v1/summary.json)
- [stockfish_pgn_v1/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_v1/summary.json)
- [stockfish_pgn_pi_v1/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_v1/summary.json)

## Throughput Benchmarks

These are training-throughput or schedule sweeps, not preferred model-quality endpoints:

- [bench_bs64_t7/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/bench_bs64_t7/summary.json)
- [bench_bs128_t7/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/bench_bs128_t7/summary.json)
- [bench_bs256_t7/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/bench_bs256_t7/summary.json)
- [bench_bs128_t4/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/bench_bs128_t4/summary.json)
- [bench_bs256_t4/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/bench_bs256_t4/summary.json)
- [bench_bs512_t4/summary.json](/home/torsten/EngineKonzept/artifacts/phase5/bench_bs512_t4/summary.json)
- [oracle_build_sweep_10k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_build_sweep_10k_v1.json)
- [oracle_pair_10k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_10k_v1.json)
- [oracle_pair_20k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_20k_v1.json)
- [oracle_pair_50k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_v1.json)

## Oracle Profiling And Transport Work

The `oracle_*` artifacts capture local-vs-daemon transport, host comparison, profiling, and intentionally retained benchmark history for the Phase-5 dataset-oracle optimization work.

Useful entry points:

- [oracle_transport_bench_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_transport_bench_v1.json)
- [oracle_host_compare_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_host_compare_v4.json)
- [oracle_profile_50k_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v4.json)
- [oracle_profile_50k_pi_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_pi_v4.json)
