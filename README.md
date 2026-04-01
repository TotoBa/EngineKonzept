# EngineKonzept

EngineKonzept is a new chess-engine repository built around latent adversarial planning.
It is intentionally **not** a conventional search engine with a neural add-on.

The target runtime path is:

`position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`

## Current Scope

The repository now covers Phase 5:

- root project rules and execution plans
- Rust workspace boundaries and placeholder future crates
- Python training-project boundaries and placeholder modules
- exact symbolic chess primitives, position state, FEN support, legal move generation, move application, and perft coverage
- a minimal UCI shell with deterministic legal stub move output
- a factorized action space for model-facing move IO
- a deterministic object-centric position encoder
- a Python dataset pipeline with exact-rule labels, deterministic splits, and summary reporting
- a first PyTorch legality/policy proposer with config-driven training, held-out metrics, and measured CPU throughput
- bounded PGN ingestion with offline Stockfish 18 labeling for policy-supervised runs
- a `torch.export` proposer bundle plus Rust-side metadata loading/validation
- CI, lint, and test wiring
- architecture and phase documentation

It still does **not** implement:

- learned runtime inference
- any search or evaluation runtime
- any dynamics, opponent, or planner model
- any classical engine/search machinery

## Phase 5 Snapshot

The current Phase-5 stack is intentionally narrow but externally checkable:

- training data can come from exact-rule fixtures, labeled JSONL seeds, or streamed PGN files
- legality, action encoding, and next-state generation still go through the Rust rules oracle
- policy labels for larger runs are generated offline with `/usr/games/stockfish18`
- the current 10k reference corpus was labeled on a Raspberry Pi host and evaluated locally

Current proposer shape:

- input: `230` packed encoder features
- backbone: `230 -> hidden -> hidden` MLP with ReLU and optional dropout
- heads: two flat `hidden -> 20480` heads for legality and policy

Reference model sizes:

- `hidden_dim=128`: `5,329,920` parameters
- `hidden_dim=192`: `7,986,688` parameters
- `hidden_dim=256`: `10,651,648` parameters

Reference Phase-5 experiments on the `10,240` train / `2,048` verify Pi-labeled corpus:

- best speed/quality trade-off so far: [phase5_stockfish_pgn_pi_10k_bs128_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_pi_10k_bs128_v1.json)
- best verify legal-set F1 so far: [phase5_stockfish_pgn_pi_10k_h256_v1.json](/home/torsten/EngineKonzept/python/configs/phase5_stockfish_pgn_pi_10k_h256_v1.json)
- comparison summary: [stockfish_pgn_pi_10k_compare_v1.json](/home/torsten/EngineKonzept/artifacts/phase5/stockfish_pgn_pi_10k_compare_v1.json)

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
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/train_legality.py --config python/configs/phase5_stockfish_pgn_pi_10k_bs128_v1.json
TMPDIR=/srv/schach/tmp .venv/bin/python python/scripts/eval_suite.py --checkpoint models/proposer/stockfish_pgn_pi_10k_bs128_v1/checkpoint.pt --dataset-path artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1 --split test
```

## Guardrails

- No alpha-beta, negamax, PVS, quiescence, TT-search, null-move pruning, LMR, or heuristic fallback engine.
- The symbolic chess core is allowed later only for exact rules, labels, tests, and final move safety checks.
- The current exact rules core is intentionally isolated from any runtime search logic.
- The current UCI shell is protocol-only and must not accrete classical engine behavior.
- The current action space and encoder are deterministic schema layers, not learned components.
- Runtime and protocol code belong in Rust.
- Training, datasets, and experiment code belong in Python.
- Every phase must leave the tree buildable, testable, and documented.

See [AGENTS.md](/home/torsten/EngineKonzept/AGENTS.md), [PLANS.md](/home/torsten/EngineKonzept/PLANS.md), and [CODEX_RUNBOOK.md](/home/torsten/EngineKonzept/CODEX_RUNBOOK.md) for the binding project rules.
