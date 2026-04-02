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

Both benchmark helpers now embed basic runtime metadata in their JSON output, including host name, platform string, Python version, and visible CPU count. That makes cross-host comparisons easier to review without hand-written side notes.

For reproducible hotspot analysis on a raw dataset slice, use [profile_dataset_oracle.py](/home/torsten/EngineKonzept/python/scripts/profile_dataset_oracle.py). The current large reference run is [oracle_profile_50k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v1.json).

The latest JSON-writer follow-up is captured in [oracle_pair_50k_json_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_json_v2.json) and [oracle_profile_50k_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v2.json). It is a small optimization, not a step change: the large-build result improves modestly, while the profile still points primarily at `legal_generation` and secondarily at JSON serialization.

The finer attack-check breakdown now lives in [oracle_profile_50k_v3.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v3.json). On the current large run, the remaining self-check pressure is led by rook-ray scans, king-local checks, and knight-local checks, not by pawn checks.

The next move-label-path follow-up is captured in [oracle_profile_50k_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v4.json) and [oracle_pair_50k_encode_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_encode_v4.json). It reuses preformatted legal UCI strings for `selected_move_resolution` and splits `legal_action_encoding` into `encode` and `sort`. On the current 50k run, that is a real step forward, not just a profiling refinement.

That `v4` 50k result was also rerun on `raspberrypi` and fetched back as [oracle_pair_50k_pi_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_pi_v4.json). The regenerated cross-host roll-up is [oracle_host_compare_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_host_compare_v4.json). On both hosts, `auto_w4` remains the fastest label for the same effective `500`-record schedule:

- local `50k`: `auto_w4` about `23.149s`, `w4_b500` about `23.861s`
- `raspberrypi` `50k`: `auto_w4` about `74.679s`, `w4_b500` about `76.729s`

The matching Pi hotspot profile is [oracle_profile_50k_pi_v4.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_pi_v4.json). It keeps the same ordering as the local profile: `legal_generation` first, `self_check_filter` inside it second, and `json_serialize` still large enough to remain a real secondary optimization target.

The newer local profile [oracle_profile_50k_v10.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v10.json) also records JSON subsection byte shares. On the current `50k` run, `position_encoding` is the largest JSON payload block, ahead of `annotations`, `legal_action_encodings`, and `legal_moves`.

The follow-up profile [oracle_profile_50k_v12.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v12.json) splits `position_encoding` further and shows that `square_tokens` is the dominant payload inside that block. A first focused `square_tokens` writer experiment is recorded in [oracle_pair_50k_squaretokens_v13.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_squaretokens_v13.json); it did not improve the real `50k` build and was therefore not kept.

The next profile [oracle_profile_50k_v15.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_profile_50k_v15.json) does the same for `annotations`. There, `core_flags` and `selected_move_fields` dominate the section, while the numeric count fields are small. A first section-buffered experiment is recorded in [oracle_pair_50k_annotations_v16.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_annotations_v16.json); it also failed to beat the real `50k` baseline and was not kept.

The dataset builders also support offline oracle parallelism:

- `--oracle-workers`
- `--oracle-batch-size`

These knobs are intended for throughput tuning during dataset generation only. Current reference measurements are recorded in [oracle_e2e_parallel_bench_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_e2e_parallel_bench_v1.json).

For Phase-5 proposer training, the dataset builders now also support lean proposer artifacts. The generic `make_dataset.py` path exposes that via `--write-proposer-artifacts`. The larger PGN/Stockfish builder [build_stockfish_pgn_dataset.py](/home/torsten/EngineKonzept/python/scripts/build_stockfish_pgn_dataset.py) now emits them by default and only needs `--no-proposer-artifacts` if you explicitly want to suppress them. The emitted `proposer_train.jsonl`, `proposer_validation.jsonl`, and `proposer_test.jsonl` files contain packed fixed-width features plus flattened legality/policy supervision, so the proposer trainer can skip reparsing the larger `DatasetExample` payloads. If those files are absent, training still falls back to the existing `train.jsonl` / `validation.jsonl` split artifacts.

For older datasets, use [materialize_proposer_artifacts.py](/home/torsten/EngineKonzept/python/scripts/materialize_proposer_artifacts.py) to backfill those lean split files in place. To benchmark the effect on proposer loading and a short real training run, use [benchmark_proposer_artifacts.py](/home/torsten/EngineKonzept/python/scripts/benchmark_proposer_artifacts.py).

The current `10k` reference benchmark is [proposer_artifact_bench_10k_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/proposer_artifact_bench_10k_v2.json). On the Pi-labeled `10,240` dataset, that run shows:

- `train` proposer loading about `4.86x` faster
- `validation` proposer loading about `3.74x` faster
- lean split files materially smaller than the full split JSONL
- a matched `1`-epoch proposer run about `1.37x` faster end-to-end, with identical validation metrics

If `rust/target/debug/dataset-oracle` already exists, the Python wrapper now uses that binary directly for one-shot oracle calls instead of spawning `cargo run` each time. The warmed reference measurement for that path is stored in [oracle_one_shot_binary_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_one_shot_binary_v1.json).

If `oracle_workers > 1` and no explicit `oracle_batch_size` is provided, the dataset builder now auto-splits the workload into roughly one batch per worker. The current reference measurement for that path is stored in [oracle_auto_batch_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_auto_batch_v1.json).

The emitted `summary.json` now also records the effective offline oracle schedule under `oracle_schedule`, so external reviewers can see the resolved worker count, requested batch size, effective batch size, and resulting batch count directly from the artifact.

For larger real builds, use [benchmark_dataset_build.py](/home/torsten/EngineKonzept/python/scripts/benchmark_dataset_build.py) to compare multiple worker/batch schedules on the same raw input. The first 10k reference sweep is stored in [oracle_build_sweep_10k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_build_sweep_10k_v1.json), and repeated pairwise confirmations are stored in [oracle_pair_10k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_10k_v1.json) and [oracle_pair_20k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_20k_v1.json). Based on those runs, the default auto heuristic now caps the effective batch size at `500` when `oracle_workers > 1`.

The same pair runner was also executed on `raspberrypi`; the fetched artifacts are [oracle_pair_10k_pi_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_10k_pi_v1.json), [oracle_pair_20k_pi_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_20k_pi_v1.json), and the local roll-up is [oracle_host_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_host_compare_v1.json).

To regenerate that host roll-up from raw pair artifacts, use [compare_dataset_build_benchmarks.py](/home/torsten/EngineKonzept/python/scripts/compare_dataset_build_benchmarks.py).

The larger 50k follow-up artifacts are [oracle_pair_50k_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_v1.json), [oracle_pair_50k_pi_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_pair_50k_pi_v1.json), and [oracle_host_compare_v2.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_host_compare_v2.json). Those runs keep the same conclusion: `4` workers remain clearly better than `2`, and once the auto heuristic resolves to `effective_batch_size = 500`, `auto_w4` and explicit `w4_b500` should be treated as the same schedule.

The proposer config also accepts a `runtime` object for CPU tuning:

- `torch_threads` to cap PyTorch CPU threads
- `dataloader_workers` to control `DataLoader` workers

Reference configs in the repository currently cover:

- a current default entry point:
  [phase5_stockfish_pgn_current_default_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_current_default_v1.json)
- small smoke and seed runs
- legacy small-baseline Stockfish runs
- Pi-labeled `10,240 / 2,048` PGN policy runs
- experimental variants across batch size and hidden width

The current larger comparison summary lives at [stockfish_pgn_pi_10k_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_compare_v1.json).

Install the training dependency set with `pip install -e python[train,dev]` or equivalent before running the proposer training script.
