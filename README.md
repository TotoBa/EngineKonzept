# EngineKonzept

EngineKonzept is a new chess-engine repository built around latent adversarial planning.
It is intentionally **not** a conventional search engine with a neural add-on.

The target runtime path is:

`position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`

## Current Scope

The repository now covers Phase 6 foundations:

- root project rules and execution plans
- Rust workspace boundaries and placeholder future crates
- Python training-project boundaries and placeholder modules
- exact symbolic chess primitives, position state, FEN support, legal move generation, move application, and perft coverage
- a minimal UCI shell with exact legal move generation and symbolic proposer scoring
- a factorized action space for model-facing move IO
- a deterministic object-centric position encoder
- a Python dataset pipeline with exact-rule labels, deterministic splits, and summary reporting
- a first PyTorch legality/policy proposer with config-driven training, held-out metrics, and measured CPU throughput
- bounded PGN ingestion with offline Stockfish 18 labeling for policy-supervised runs
- a `torch.export` proposer bundle plus Rust-side metadata loading/validation
- a local-first Unix-domain-socket oracle daemon for faster reproducible dataset builds
- a first action-conditioned latent dynamics baseline with held-out reconstruction and drift metrics
- the first explicit Phase-7 dataset contract for opponent-reply supervision
- the first trained Phase-7 opponent-head baseline
- the first bounded opponent-aware planner baseline
- CI, lint, and test wiring
- architecture and phase documentation

It still does **not** implement:

- full planner-driven runtime inference
- any search or evaluation runtime
- any planner model
- any classical engine/search machinery

## Phase 5 Snapshot

The current Phase-5 stack is intentionally narrow but externally checkable:

- training data can come from exact-rule fixtures, labeled JSONL seeds, or streamed PGN files
- legality, action encoding, and next-state generation still go through the Rust rules oracle
- policy labels for larger runs are generated offline with `/usr/games/stockfish18`
- the current 10k reference corpus was labeled on a Raspberry Pi host and evaluated locally
- the current default symbolic bundle now also carries a native Rust runtime weight file for candidate scoring

Current proposer shape:

- input: `230` packed encoder features
- backbone: `230 -> hidden -> hidden` MLP with ReLU and optional dropout
- heads: two flat `hidden -> 20480` heads for legality and policy

There are now two proposer families:

- `multistream_v2`: unpacks the same `230` features back into typed piece/square/rule streams, applies light cross-attention, then returns to the same flat legality/policy heads
- `factorized_v6`: keeps the stronger conditional factorized legality path and adds explicit policy-side `from-to` coupling plus a low-rank residual
- `relational_v1`: combines the typed multi-stream backbone with the stronger factorized legality/policy heads
- `symbolic_v1`: replaces learned legality with exact legal-move generation and learns only a scorer over legal candidates plus symbolic move features

Reference model sizes:

- `hidden_dim=128`: `5,329,920` parameters
- `hidden_dim=192`: `7,986,688` parameters
- `hidden_dim=256`: `10,651,648` parameters

Reference Phase-5 experiments on the `10,240` train / `2,048` verify Pi-labeled corpus:

- current default entry point: [phase5_stockfish_pgn_symbolic_v1_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_symbolic_v1_v1.json)
- materialized current-default bundle: [stockfish_pgn_symbolic_v1_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_symbolic_v1_v1)
- materialized current-default summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_symbolic_v1_v1/summary.json)
- best speed/quality trade-off so far: [phase5_stockfish_pgn_pi_10k_bs128_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_pi_10k_bs128_v1.json)
- best verify legal-set F1 so far: [phase5_stockfish_pgn_factorized_v6_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_factorized_v6_v1.json)
- current three-way comparison: [stockfish_pgn_10k_three_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_three_way_compare_v1.json)
- current four-way comparison including the structured multi-stream arm: [stockfish_pgn_10k_four_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_four_way_compare_v1.json)
- current five-way comparison including the first factorized decoder arm: [stockfish_pgn_10k_five_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_five_way_compare_v1.json)
- current six-way comparison including the conditional factorized decoder arm: [stockfish_pgn_10k_six_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_six_way_compare_v1.json)
- current seven-way comparison including the policy-stronger conditional decoder arm: [stockfish_pgn_10k_seven_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_seven_way_compare_v1.json)
- current nine-way comparison including `factorized_v6` and `relational_v1`: [stockfish_pgn_10k_nine_way_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_10k_nine_way_compare_v1.json)
- current symbolic comparison against the previous learned-legality default and `factorized_v6`: [stockfish_pgn_symbolic_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_symbolic_compare_v1.json)
- direct checkpoint-selection comparison for `factorized_v5`: [stockfish_pgn_factorized_v5_selection_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_factorized_v5_selection_compare_v1.json)
- comparison summary: [stockfish_pgn_pi_10k_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_compare_v1.json)
- oracle transport benchmark: [oracle_transport_bench_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/oracle_transport_bench_v1.json)

For dataset naming, treat [phase5_stockfish_pgn_train_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1) and [phase5_stockfish_pgn_verify_pi_10k_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1) as the current standard Phase-5 corpora. The smaller [phase5_stockfish_pgn_train_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_v1), [phase5_stockfish_pgn_verify_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_v1), [phase5_stockfish_pgn_train_pi_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_train_pi_v1), and [phase5_stockfish_pgn_verify_pi_v1](/home/torsten/EngineKonzept/artifacts/datasets/phase5_stockfish_pgn_verify_pi_v1) remain available as early small-baseline artifacts and regression fixtures.

For the same categorization on configs, model bundles, and Phase-5 summaries, see:

- [python/configs/README.md](/home/torsten/EngineKonzept/python/configs/README.md)
- [models/proposer/README.md](/home/torsten/EngineKonzept/models/proposer/README.md)
- [artifacts/phase5/README.md](/home/torsten/EngineKonzept/artifacts/phase5/README.md)
- [model-roadmap.md](/home/torsten/EngineKonzept/docs/architecture/model-roadmap.md)
- [contracts.md](/home/torsten/EngineKonzept/docs/architecture/contracts.md)
- [search-workflows.md](/home/torsten/EngineKonzept/docs/architecture/search-workflows.md)
- [opponent.md](/home/torsten/EngineKonzept/docs/architecture/opponent.md)
- [gpt-pro-review-brief.md](/home/torsten/EngineKonzept/docs/experiments/gpt-pro-review-brief.md)

The first architecture-extension notes beyond the flat MLP live in [docs/arch.ideas.md](/home/torsten/EngineKonzept/docs/arch.ideas.md). The current implementation applies only the low-risk part of that direction so far: typed multi-stream fusion before considering any heavier routing or expert machinery.

The newer proposer comparison now extends beyond the first three factorized decoder baselines. `factorized_v6` is the current best learned-legality arm on the `10k` corpus by a clear margin, while `relational_v1` improves policy over the earlier factorized runs without taking the policy lead from the old learned-legality default. There is also an explicit checkpoint-selection comparison for `factorized_v5`, showing the expected tradeoff between legality-first and balanced selection. The `symbolic_v1` arm goes one step further and removes learned legality entirely in favor of exact legal-candidate generation; on the current `10k` corpus it is the strongest proposer result so far, and it now carries the official proposer export/runtime contract.

The current `engine-app` binary will use that symbolic proposer bundle automatically when it can find [stockfish_pgn_symbolic_v1_v1](/home/torsten/EngineKonzept/models/proposer/stockfish_pgn_symbolic_v1_v1). Override the bundle location with `ENGINEKONZEPT_PROPOSER_BUNDLE=/path/to/bundle` when needed.

## Phase 6 Snapshot

The proposer is now accepted as a temporary frontier, and the repository includes a checkable latent-dynamics baseline plus larger-corpus Phase-6 follow-ups:

- current Phase-6 config: [phase6_dynamics_merged_unique_structured_v5_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_merged_unique_structured_v5_v1.json)
- current Phase-6 bundle: [dynamics_merged_unique_structured_v5_v1](/home/torsten/EngineKonzept/models/dynamics/dynamics_merged_unique_structured_v5_v1)
- current Phase-6 summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_structured_v5_v1/summary.json)
- current Phase-6 verify eval: [dynamics_merged_unique_structured_v5_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_structured_v5_v1_verify.json)
- direct large-corpus comparison: [dynamics_merged_unique_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_merged_unique_compare_v1.json)
- architecture note: [dynamics.md](/home/torsten/EngineKonzept/docs/architecture/dynamics.md)
- phase note: [phase-6.md](/home/torsten/EngineKonzept/docs/phases/phase-6.md)

The first `v1` run establishes the exact Phase-6 plumbing:

- lean `dynamics_<split>.jsonl` artifacts
- action-conditioned latent transition training
- `torch.export` + Rust metadata validation
- one-step reconstruction and multi-step drift metrics

The current model family is still weak in the exact sense:

- validation and verify feature-reconstruction errors decrease into a stable range
- exact packed next-state accuracy remains `0.0`
- multi-step drift is measurable, but not yet good

The first structured follow-up is now also materialized:

- config: [phase6_dynamics_structured_v2_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v2_v1.json)
- bundle: [structured_v2_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v2_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1/summary.json)
- verify: [dynamics_structured_v2_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v2_v1_verify.json)

The explicit drift-selection follow-up established the first useful Phase-6 reference:

- verify `feature_l1_error`: `1.433716 -> 1.425823`
- verify `drift_feature_l1_error`: `1.595053 -> 1.557198`

The smaller-corpus latent-consistency follow-up kept that drift-aware structure and improved both main verify soft metrics again:

- verify `feature_l1_error`: `1.425823 -> 1.425074`
- verify `drift_feature_l1_error`: `1.557198 -> 1.429654`

The next `structured_v3_v1` follow-up adds a delta auxiliary head on top of the same latent-stable path:

- config: [phase6_dynamics_structured_v3_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v3_v1.json)
- bundle: [structured_v3_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v3_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1/summary.json)
- verify: [dynamics_structured_v3_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v3_v1_verify.json)

On the smaller `10k` corpus it improves one-step verify reconstruction again, but gives back a little drift quality:

- verify `feature_l1_error`: `1.425074 -> 1.353977`
- verify `drift_feature_l1_error`: `1.429654 -> 1.47778`

The next `structured_v4_v1` follow-up keeps the same `structured_v4` decoder family but adds explicit short-horizon drift supervision during training:

- config: [phase6_dynamics_structured_v4_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v4_v1.json)
- bundle: [structured_v4_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v4_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1/summary.json)
- verify: [dynamics_structured_v4_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v4_v1_verify.json)

It does not beat the current default or `structured_v3_v1`:

- verify `feature_l1_error`: `1.425074 -> 1.611914`
- verify `drift_feature_l1_error`: `1.429654 -> 1.49735`

The parallel `edit_v1` arm is also materialized as an experimental counterexample:

- config: [phase6_dynamics_edit_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_edit_v1.json)
- bundle: [edit_v1](/home/torsten/EngineKonzept/models/dynamics/edit_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1/summary.json)
- verify: [dynamics_edit_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_edit_v1_verify.json)

It wins one-step reconstruction strongly, but collapses multi-step drift and therefore remains experimental only.

The next `structured_v5_v1` arm connects Phase 6 directly to the symbolic proposer contract by feeding the selected move's exact symbolic candidate features into the transition path:

- config: [phase6_dynamics_structured_v5_v1.json](/home/torsten/EngineKonzept/python/configs/phase6_dynamics_structured_v5_v1.json)
- bundle: [structured_v5_v1](/home/torsten/EngineKonzept/models/dynamics/structured_v5_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v5_v1/summary.json)
- verify: [dynamics_structured_v5_v1_verify.json](/home/torsten/EngineKonzept/artifacts/phase6/dynamics_structured_v5_v1_verify.json)

On the smaller `10k` corpus it improves one-step held-out reconstruction over the current default, but not drift:

- verify `feature_l1_error`: `1.425074 -> 1.404499`
- verify `drift_feature_l1_error`: `1.429654 -> 1.556962`

The large `merged_unique` reruns change that decision on the `110,570 / 12,286 / 2,169` corpus:

- large `structured_v2_latent_v1`: verify `feature_l1_error=1.067843`, `drift_feature_l1_error=6.305117`
- large `structured_v3_v1`: verify `feature_l1_error=1.02784`, `drift_feature_l1_error=6.18409`
- large `structured_v5_v1`: verify `feature_l1_error=0.924808`, `drift_feature_l1_error=1.548861`

That makes `dynamics_merged_unique_structured_v5_v1` the new preferred Phase-6 reference. The symbolic selected-move features do not just help one-step reconstruction at this scale; on the larger corpus they also become the best measured drift path so far.

Exact next-state accuracy still remains `0.0`.

## Phase 7 Preparation

The repository now has the first explicit trained Phase-7 opponent-head baseline, but the symbolic reply-scorer baseline is still stronger:

- architecture note: [opponent.md](/home/torsten/EngineKonzept/docs/architecture/opponent.md)
- phase note: [phase-7.md](/home/torsten/EngineKonzept/docs/phases/phase-7.md)
- workflow artifacts: [README.md](/home/torsten/EngineKonzept/artifacts/phase7/README.md)
- opponent config: [phase7_opponent_merged_unique_mlp_v1.json](/home/torsten/EngineKonzept/python/configs/phase7_opponent_merged_unique_mlp_v1.json)
- trained bundle: [merged_unique_mlp_v1](/home/torsten/EngineKonzept/models/opponent/merged_unique_mlp_v1)
- summary: [summary.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_mlp_v1/summary.json)
- verify comparison: [opponent_merged_unique_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/opponent_merged_unique_compare_v1.json)

The `OpponentHeadV1` workflow derives:

- the teacher-chosen root move
- the exact successor state after that move
- the exact legal opponent replies
- symbolic reply features
- teacher best-reply supervision
- first pressure and uncertainty targets

That is the intended bridge between the offline search-workflow layer and an explicit Phase-7 opponent model.

Current larger merged-unique verify result:

- symbolic baseline:
  - `reply_top1_accuracy=0.3`
  - `reply_top3_accuracy=0.4`
  - `teacher_reply_mean_reciprocal_rank=0.419262`
- first trained `mlp_v1` opponent head:
  - `reply_top1_accuracy=0.066667`
  - `reply_top3_accuracy=0.333333`
  - `teacher_reply_mean_reciprocal_rank=0.272664`

So the first trained head is now a real repo baseline, but it does not yet beat the symbolic reply scorer.

The repo now also has the first bounded planner-style comparison on the same verify slice:

- root-only symbolic proposer:
  - `root_top1_accuracy=0.148438`
  - `teacher_root_mean_reciprocal_rank=0.213542`
- symbolic-reply aggregation:
  - `root_top1_accuracy=0.15625`
  - `teacher_root_mean_reciprocal_rank=0.216797`
- learned-reply aggregation:
  - `root_top1_accuracy=0.15625`
  - `teacher_root_mean_reciprocal_rank=0.21875`

Those runs are bounded offline planner baselines, not runtime search:

- [planner_symbolic_root_only_verify_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/planner_symbolic_root_only_verify_v1.json)
- [planner_symbolic_reply_verify_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/planner_symbolic_reply_verify_v1.json)
- [planner_learned_reply_verify_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/planner_learned_reply_verify_v1.json)
- [planner_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase7/planner_compare_v1.json)

## Repository Layout

```text
.
|-- AGENTS.md
|-- PLANS.md
|-- CODEX_RUNBOOK.md
|-- README.md
|-- docs/
|   |-- architecture/
|   |-- experiments/
|   `-- phases/
|-- rust/
|   |-- Cargo.toml
|   `-- crates/
|-- python/
|   |-- pyproject.toml
|   |-- train/
|   |-- scripts/
|   `-- tests/
|-- tests/
|   |-- perft/
|   |-- planner/
|   `-- positions/
|-- models/
`-- artifacts/
```

## Validation Commands

From the repository root:

```bash
make check
```

Or run the individual commands directly:

```bash
cd rust && cargo fmt --all --check
cd rust && cargo clippy --workspace --all-targets --all-features -- -D warnings
cd rust && cargo test --workspace
python3 -m ruff check python
python3 -m pytest python/tests
```

Representative Phase-5 commands:

```bash
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/build_stockfish_pgn_dataset.py --help
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/train_legality.py --config python/configs/phase5_stockfish_pgn_symbolic_v1_v1.json
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/eval_suite.py --checkpoint models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt --dataset-path artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1 --split test
cargo run --quiet -p tools --bin dataset-oracle-daemon --socket /tmp/enginekonzept-oracle.sock
ENGINEKONZEPT_DATASET_ORACLE=unix:///tmp/enginekonzept-oracle.sock TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/build_stockfish_pgn_dataset.py --help
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/benchmark_dataset_oracle.py --input tests/positions/edge_cases.txt --source-format edge-cases --records 10000
TMPDIR=/srv/schach/tmp cargo run --quiet -p tools --bin dataset-oracle-profile < /srv/schach/tmp/oracle-e2e/train_raw_2k.jsonl
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/materialize_dynamics_artifacts.py --dataset-dir artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/train_dynamics.py --config python/configs/phase6_dynamics_v1.json
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/eval_dynamics.py --checkpoint models/dynamics/v1/checkpoint.pt --dataset-path artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1 --split test --drift-horizon 2
```

## Guardrails

- No alpha-beta, negamax, PVS, quiescence, TT-search, null-move pruning, LMR, or heuristic fallback engine.
- The symbolic chess core is allowed later only for exact rules, labels, tests, and final move safety checks.

If classical search methods are introduced for experiments, use them only under the boundary defined in [search-workflows.md](/home/torsten/EngineKonzept/docs/architecture/search-workflows.md): benchmark, teacher, analysis, and curriculum workflows are acceptable; runtime move selection is not.
- The current exact rules core is intentionally isolated from any runtime search logic.
- The current UCI shell is protocol-only and must not accrete classical engine behavior.
- The current action space and encoder are deterministic schema layers, not learned components.
- Runtime and protocol code belong in Rust.
- Training, datasets, and experiment code belong in Python.
- Every phase must leave the tree buildable, testable, and documented.

See [AGENTS.md](/home/torsten/EngineKonzept/AGENTS.md), [PLANS.md](/home/torsten/EngineKonzept/PLANS.md), and [CODEX_RUNBOOK.md](/home/torsten/EngineKonzept/CODEX_RUNBOOK.md) for the binding project rules.
