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

The repository now also includes a profiling-only binary:

- `cargo run --quiet -p tools --bin dataset-oracle-profile`

It consumes the same newline-delimited oracle input as `dataset-oracle`, but instead of emitting labels it reports aggregated phase timings. The baseline profile is stored in [oracle_profile_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v1.json), the post-serialization-optimization profile is stored in [oracle_profile_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v2.json), the post-check-path profile is stored in [oracle_profile_v3.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v3.json), the first fine-grained split is stored in [oracle_profile_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v4.json), the board-snapshot profile is stored in [oracle_profile_v5.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v5.json), the king-square profile is stored in [oracle_profile_v6.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v6.json), and the current profile is stored in [oracle_profile_v7.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v7.json).

The first profile showed:

- `legal_generation`: about `48.7%`
- `json_serialize`: about `32.3%`
- `legal_action_encoding`: about `4.5%`

That matters because it narrows the remaining throughput pressure:

- transport overhead is no longer the main cost center
- move application is now cheap after the known-legal shortcut
- the next meaningful wins are most likely inside exact legal move generation or in the volume and shape of JSON serialization

That led to the next measured improvement: the oracle now writes JSON directly into the stream instead of allocating a temporary `String` for each response line. On the same 2000-record daemon build, wall-clock time dropped from about `1.624s` to about `1.555s`, with byte-identical artifacts. The corresponding measurement is stored in [oracle_e2e_jsonopt_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_jsonopt_v1.json).

After that change, the updated profile shifted to:

- `legal_generation`: about `57.0%`
- `json_serialize`: about `20.7%`
- `legal_action_encoding`: about `5.2%`

At this point the optimization target is much narrower: exact legal move generation is now the clear dominant cost block.

The next small but externally repeatable step was to trim the self-check filter inside `legal_moves`: for that specific test, the rules kernel now applies moves through a lighter path that updates only the board state needed by `is_in_check`, not the full metadata path used for committed next states. On the same 2000-record daemon benchmark, two back-to-back runs landed at about `1.547s` and `1.538s`, both below the previous `1.555s` jsonopt baseline. The measurement is stored in [oracle_e2e_checkpath_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_checkpath_v1.json).

The corresponding profile tightened again:

- `legal_generation`: about `56.0%`
- `json_serialize`: about `21.2%`
- `legal_action_encoding`: about `5.3%`

This is not a dramatic jump, but it confirms the direction: remaining wins are now incremental and concentrated almost entirely inside exact legality work.

The next profiling refinement made that even narrower. In [oracle_profile_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v4.json), `legal_generation` was split into:

- `pseudo_legal_generation`: about `6.0%`
- `self_check_filter`: about `49.8%`

That pointed directly at the next useful optimization: the self-check filter now evaluates king safety on a board snapshot instead of cloning and mutating a full `Position` for every pseudo-legal candidate. On the same 2000-record daemon benchmark, wall-clock time dropped from about `1.542s` to about `1.443s` on average across two runs. That measurement is stored in [oracle_e2e_boardcheck_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_boardcheck_v1.json).

After that change, [oracle_profile_v5.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v5.json) shifted to:

- `legal_generation`: about `47.8%`
- `self_check_filter`: about `40.9%`
- `json_serialize`: about `25.5%`

That is the current key result: the self-check filter is still the largest single rules block, but it is now materially cheaper than before, and the remaining runtime is split more evenly between legality work and output serialization.

The next refinement kept the same board-snapshot path but stopped re-scanning the board for the moving side's king on every candidate. For non-king moves, the king square is stable; for king moves, the destination square is already known. On the same 2000-record daemon benchmark, wall-clock time dropped again from about `1.443s` to about `1.375s` on average across two runs. That measurement is stored in [oracle_e2e_kingsquare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_kingsquare_v1.json).

After that change, [oracle_profile_v6.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v6.json) shifted to:

- `legal_generation`: about `42.7%`
- `self_check_filter`: about `34.1%`
- `json_serialize`: about `28.4%`

At this point the Oracle fast path is substantially leaner than the start of this optimization series, and the remaining cost centers are much closer together.

The next and likely last low-risk improvement in this series was to specialize the JSON output path for `DatasetOracleOutput`. The writer now emits the same schema and the same byte layout as `serde_json` for this struct, but avoids the generic per-field serialization overhead. A dedicated unit test checks byte equality on a representative labeled record. On the same 2000-record daemon benchmark, wall-clock time dropped from about `1.375s` to about `1.358s` on average across two runs. The measurement is stored in [oracle_e2e_customjson_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_customjson_v1.json).

After that change, [oracle_profile_v7.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v7.json) shifted to:

- `legal_generation`: about `47.9%`
- `self_check_filter`: about `38.4%`
- `json_serialize`: about `19.8%`

That is the current best end-to-end balance in this series: the legality path is clearly dominant again, and the generic output overhead has been pushed down substantially.

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
