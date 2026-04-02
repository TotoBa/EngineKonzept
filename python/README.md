# EngineKonzept Python Project

This directory hosts the dataset, training, export, and experiment code for EngineKonzept.

The repository now includes the Phase-5 proposer stack:

- raw position ingestion from edge-case files, FEN line files, EPD suites, and JSONL
- exact-rule labeling via the Rust dataset oracle
- deterministic train/validation/test splitting and JSONL dataset artifacts
- fixed-width feature packing for proposer training
- a first PyTorch legality/policy proposer
- bounded PGN sampling with Stockfish 18 move labels for policy-supervised runs
- config-driven training with held-out legal-set precision/recall reporting and examples/second
- `torch.export` bundles plus Rust-loadable metadata

The PGN utility entry point is `python/scripts/build_stockfish_pgn_dataset.py`. It streams selected PGNs, queries `/usr/games/stockfish18` for bounded move labels, then routes legality and next-state generation back through the Rust oracle.

The Rust oracle can now run either as a subprocess or as a local Unix-domain-socket daemon. Set `ENGINEKONZEPT_DATASET_ORACLE=unix:///path/to/socket` to use the daemon path during dataset builds.

To compare both transports directly, use [benchmark_dataset_oracle.py](/home/torsten/EngineKonzept/python/scripts/benchmark_dataset_oracle.py).

The dataset builders also support offline oracle parallelism:

- `--oracle-workers`
- `--oracle-batch-size`

These knobs are intended for throughput tuning during dataset generation only. Current reference measurements are recorded in [oracle_e2e_parallel_bench_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_parallel_bench_v1.json).

If `rust/target/debug/dataset-oracle` already exists, the Python wrapper now uses that binary directly for one-shot oracle calls instead of spawning `cargo run` each time. The warmed reference measurement for that path is stored in [oracle_one_shot_binary_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_one_shot_binary_v1.json).

The proposer config also accepts a `runtime` object for CPU tuning:

- `torch_threads` to cap PyTorch CPU threads
- `dataloader_workers` to control `DataLoader` workers

Reference configs in the repository currently cover:

- small smoke and seed runs
- Pi-labeled `10,240 / 2,048` PGN policy runs
- a small comparison grid across batch size and hidden width

The current larger comparison summary lives at [stockfish_pgn_pi_10k_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_compare_v1.json).

Install the training dependency set with `pip install -e python[train,dev]` or equivalent before running the proposer training script.
