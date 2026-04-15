# Distribution And IPC

EngineKonzept runtime remains local-first, but the training and evaluation stack now has
an explicit distributed control plane.

The only required external runtime protocol is still UCI over stdio. Dataset generation and training stay outside the runtime and must preserve the Rust/Python boundary from [AGENTS.md](/home/torsten/EngineKonzept/AGENTS.md).

## Current Control Plane

Distributed training is now split into three layers:

- MySQL as the control plane for campaigns, tasks, leases, heartbeats, and artifact metadata
- filesystem artifacts for datasets, workflow chunks, checkpoints, arena sessions, and summaries
- worker-local scratch plus worker-local logs for hot state

The new operator entry points are:

- [ek_ctl.py](/home/persk/repos/EngineKonzept/python/scripts/ek_ctl.py)
- [ek_worker.py](/home/persk/repos/EngineKonzept/python/scripts/ek_worker.py)
- [ek_master.py](/home/persk/repos/EngineKonzept/python/scripts/ek_master.py)
- [master-control-api.md](/home/persk/repos/EngineKonzept/docs/architecture/master-control-api.md)
- [mysql-label-ledger.md](/home/persk/repos/EngineKonzept/docs/architecture/mysql-label-ledger.md)

The corresponding Python modules live under:

- [train/orchestrator](/home/persk/repos/EngineKonzept/python/train/orchestrator)

The control plane intentionally stores mostly metadata plus one resumable label ledger:

- task payloads
- task states
- leases
- worker heartbeats
- compact result summaries
- artifact paths and checksums
- unique-corpus label rows for `label_pgn_corpus`

Large payloads remain file-based:

- no checkpoints in MySQL
- no dataset JSONL in MySQL
- no arena sessions in MySQL
- no workflow blobs in MySQL

Credentials are intentionally externalized. Use CLI flags or environment variables such as
`EK_MYSQL_HOST`, `EK_MYSQL_DATABASE`, `EK_MYSQL_USER`, and `EK_MYSQL_PASSWORD`; do not
write them into tracked configs.

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

For multi-step Phase-10 style runs, the new control plane replaces the old assumption that one
long-lived process must materialize datasets, build workflow chunks, train, verify, and run arena
serially. The current DAG is now expressed as MySQL-tracked tasks, with the existing
artifact-producing scripts still acting as the execution units.

The current Phase-10 DAG is:

- `materialize -> workflow_prepare -> workflow_chunk* -> workflow_finalize -> train`
- optional `selfplay_prepare -> selfplay_shard* -> selfplay_finalize`
- `verify -> arena_prepare -> arena_match* -> arena_finalize -> phase10_finalize`
- `label_pgn_corpus` exists as a separate resumable campaign type for PGN/Stockfish corpus jobs
  - its exported raw corpora live on NAS, while uniqueness and resume state now live in MySQL

The new master layer sits above that DAG:

- submit a label campaign when fresh raw data is needed
- wait for the exported `train_raw.jsonl` / `verify_raw.jsonl`
- materialize one generation-specific Phase-10 config
- evaluate the finished generation from `verify` and `arena` summaries
- either stop, reject, or spawn the next warm-started generation
- while one host trains, other hosts may keep feeding the lineage through low-priority idle shard exports
- the master now tracks per-lineage FEN reuse in MySQL so future generations prefer unseen positions and only recycle the least-used positions when they need to refill the corpus
- the usage-ledger write path is now hash-and-counter oriented:
  - one-time historical backfills can still be large
  - steady-state generations update only per-FEN counters and last-generation markers, not full sample metadata on every row
- a fresh lineage can now be seeded from an already accepted checkpoint and extra raw snapshot dirs:
  - `seed_warm_start_checkpoint` bootstraps generation 1 from the current network
  - `bootstrap_generation1_skip_training=true` lets that first generation skip the initial train step and begin directly at `selfplay -> verify -> arena`
  - `seed_raw_dirs` plus `use_all_available_labeled_positions=true` lets a new lineage consume the full currently labeled raw corpus immediately

For future runs, the same master can now also run behind a CherryPy HTTP server:

- the runtime loop remains the same `OrchestratorMaster`
- the API reads and writes the same master spec file
- the status page at `/` is only a thin UI over the JSON API
- direct DB/task status still comes from MySQL, not from browser-local state

The first HTTP control surface exposes:

- runtime lifecycle controls: start, stop, pause, resume, reconcile now, requeue expired leases
- read-only snapshots: latest master summary, active spec, DB status snapshot
- write controls: spec patching, per-lineage toggles, per-job toggles
- ad-hoc submit endpoints mirroring the existing `ek_ctl.py` submission commands

The pre-verify selfplay stage is intentionally limited to the tracked LAP runtime and reuses the
opening FEN suite from the campaign spec. It is not a replacement for the later arena stage.

On multi-core hosts, the preferred topology is also asymmetric:

- one strong `train` worker for the exclusive training job
- multiple narrow workers for `selfplay`, `verify`, `workflow`, and `arena`

That matches the current worker contract, where one worker executes one leased task at a time.
Throughput for selfplay therefore comes from more shard tasks and more workers, not from one giant
worker process. `ek_worker.py` now exposes `--distributed-task-threads` so those distributed tasks
can be clamped to `1` CPU thread while leaving the dedicated training worker free to use a wider
Torch thread budget.

The same topology also worked in the first real master smoke run:

- one `aggregate,train` worker
- two narrow distributed workers
- one `label,selfplay,arena` worker

That run completed a real `label -> g0001 -> evaluate -> g0002` chain against MySQL.

The new HTTP layer was intentionally kept outside the active production run. It is implemented for later runs and does not need to be rolled onto a live generation mid-flight.

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

It consumes the same newline-delimited oracle input as `dataset-oracle`, but instead of emitting labels it reports aggregated phase timings. The baseline profile is stored in [oracle_profile_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v1.json), the post-serialization-optimization profile is stored in [oracle_profile_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v2.json), the post-check-path profile is stored in [oracle_profile_v3.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v3.json), the first fine-grained split is stored in [oracle_profile_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v4.json), the board-snapshot profile is stored in [oracle_profile_v5.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v5.json), the king-square profile is stored in [oracle_profile_v6.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v6.json), the custom-json profile is stored in [oracle_profile_v7.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v7.json), the attack-split profile is stored in [oracle_profile_v8.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v8.json), and the current local-attack profile is stored in [oracle_profile_v9.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v9.json).

Separately from the daemon path, the Python oracle wrapper now also prefers a prebuilt local `rust/target/debug/dataset-oracle` binary for one-shot subprocess calls before falling back to `cargo run`. On a warmed 250-record one-shot benchmark, that reduced wall-clock time from about `0.117s` to about `0.076s`, or about `1.53x` faster, with identical output digests. That measurement is stored in [oracle_one_shot_binary_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_one_shot_binary_v1.json).

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

The next profiling refinement split the remaining attack validation cost inside the self-check filter into local attackers and slider attackers. The current result in [oracle_profile_v8.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v8.json) is:

- `attack_check_local`: about `23.0%`
- `attack_check_slider`: about `12.5%`

That changes the next likely optimization target again: the larger remaining attack-check cost is now in pawn/knight/king detection, not in ray scans.

That directly led to the next rules-kernel change: the local pawn, knight, and king attack checks now use direct board-index arithmetic instead of `Square::offset`-driven coordinate reconstruction. On the same 2000-record daemon benchmark, wall-clock time dropped from about `1.358s` to about `1.343s` on average across two runs. That measurement is stored in [oracle_e2e_localattack_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_localattack_v1.json).

After that change, [oracle_profile_v9.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_v9.json) shifted to roughly:

- `legal_generation`: about `44.8%`
- `self_check_filter`: about `34.7%`
- `attack_check_local`: about `16.3%`
- `attack_check_slider`: about `13.8%`
- `json_serialize`: about `21.3%`

That is a more balanced end state for this low-risk series: the local attacker path is cheaper, but legality generation as a whole is still the clear primary target.

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
