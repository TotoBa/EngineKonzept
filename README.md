# EngineKonzept

EngineKonzept is a new chess-engine repository built around latent adversarial planning.
It is intentionally **not** a conventional search engine with a neural add-on.

The target runtime path is:

`position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`

## Phase 0 Scope

This repository currently contains only the project scaffold:

- root project rules and execution plans
- Rust workspace boundaries and placeholder crates
- Python training-project boundaries and placeholder modules
- CI, lint, and test wiring
- documentation scaffolding for architecture, phases, and experiments

It does **not** yet implement:

- chess rules
- UCI protocol handling
- model training or inference
- any classical engine/search machinery

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
PYTHONPATH=python python3 -m pytest python/tests
```

## Guardrails

- No alpha-beta, negamax, PVS, quiescence, TT-search, null-move pruning, LMR, or heuristic fallback engine.
- The symbolic chess core is allowed later only for exact rules, labels, tests, and final move safety checks.
- Runtime and protocol code belong in Rust.
- Training, datasets, and experiment code belong in Python.
- Every phase must leave the tree buildable, testable, and documented.

See [AGENTS.md](/home/persk/repos/EngineKonzept/AGENTS.md), [PLANS.md](/home/persk/repos/EngineKonzept/PLANS.md), and [CODEX_RUNBOOK.md](/home/persk/repos/EngineKonzept/CODEX_RUNBOOK.md) for the binding project rules.
