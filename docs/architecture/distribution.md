# Distribution And IPC

EngineKonzept currently remains local-first.

The only required external runtime protocol is still UCI over stdio. Dataset generation and training stay outside the runtime and must preserve the Rust/Python boundary from [AGENTS.md](/home/torsten/EngineKonzept/AGENTS.md).

## Current State

- `engine-app` speaks UCI over stdin/stdout only
- the Rust dataset oracle remains the authority for legality, action encoding, and next-state generation
- Python can now reach the oracle in two compatible ways:
  - subprocess mode via `cargo run -p tools --bin dataset-oracle`
  - local IPC mode via Unix domain socket and `dataset-oracle-daemon`

The IPC mode is intentionally narrow:

- one batch per socket connection
- newline-delimited JSON in and newline-delimited JSON out
- same `tools::label_json_line` logic as the subprocess oracle
- no new runtime networking stack

This keeps Phase 5 deterministic and externally checkable while removing the need to spawn a fresh Rust process for every dataset batch.

## Why IPC First

The current repository constraints favor:

- minimal dependencies
- testable interfaces
- deterministic contracts
- no premature multi-host complexity

A Unix domain socket daemon is the lowest-risk improvement that fits those constraints. It reduces orchestration overhead in dataset builds without changing label semantics.

## Current Python Interface

`ENGINEKONZEPT_DATASET_ORACLE` now supports two forms:

- command mode, for example:
  - `cargo run --quiet -p tools --bin dataset-oracle`
- Unix socket mode:
  - `unix:///tmp/enginekonzept-oracle.sock`

The default remains the subprocess command. Socket mode is optional and intended for local developer and batch workflows.

## Benchmarking

The repository includes a reproducible transport benchmark at [benchmark_dataset_oracle.py](/home/torsten/EngineKonzept/python/scripts/benchmark_dataset_oracle.py).

It compares:

- the current subprocess oracle path
- the local Unix-domain-socket daemon path

on the same expanded raw-record input and verifies output equality via a stable digest.

Representative command:

```bash
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/benchmark_dataset_oracle.py --input tests/positions/edge_cases.txt --source-format edge-cases --records 10000
```

Current reference measurements are stored in [oracle_transport_bench_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_transport_bench_v1.json).
An end-to-end dataset-build comparison is stored in [oracle_e2e_bench_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_bench_v1.json).

Headline result:

- against the current developer-facing `cargo run` subprocess path on a 500-record run, the daemon was about `1.36x` faster
- against a direct one-shot oracle binary on a 2000-record run, the daemon was still about `1.11x` faster

That pattern matches the intended benefit: the daemon mainly removes process-lifecycle overhead, while the exact Rust labeling work still dominates larger batches.

On a real 2000-record dataset build from the Pi-labeled PGN corpus, the daemon still produced byte-identical artifacts but improved wall-clock time only modestly, from about `2.52s` to `2.45s` (`1.03x`).

That is the important operational result:

- the daemon is a valid local optimization
- but the next meaningful throughput gains are more likely to come from the Rust oracle work itself or from better batching/parallelization higher up the pipeline

That expectation was confirmed by the next optimization step: once the Rust oracle hot path stopped recomputing legal moves twice per record, the same 2000-record build dropped further to about `2.03s`. In other words, the larger gain came from reducing core oracle work, not from transport alone.

The next step confirmed the same pattern again: once the oracle also stopped re-validating a move that had already been proven legal in the current request, the same build dropped further to about `1.62s`. At that point the cumulative gain over the original subprocess baseline reached about `1.55x`.

## Deferred Options

The deep research report identifies two plausible later upgrades, but they are not implemented yet:

- gRPC for explicit multi-host RPC with stronger transport contracts and TLS/mTLS
- a brokered job system for asynchronous selfplay, labeling, or evaluation farms

Those remain future options because the repository does not yet have firm requirements for:

- multi-host deployment
- remote inference latency budgets
- trust boundaries
- cluster operations

## Design Rule

Any future distribution layer must preserve:

- UCI as the only required external runtime protocol
- Rust as the runtime and legality authority
- Python as the training and experiment layer
- reproducible dataset and model contracts
