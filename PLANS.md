# Codex Execution Plans for EngineKonzept

This file defines the execution plans that Codex should follow when implementing the engine.
Treat each phase as a bounded milestone with its own goal, non-goals, deliverables, tests, and exit criteria.

The project goal is to build a **new chess engine from scratch** around latent adversarial planning, with UCI as the only required external protocol surface.

## How to use this file
- Read `AGENTS.md` first.
- Select the current phase.
- Implement only that phase unless explicitly instructed to advance.
- Keep the tree buildable at every stopping point.
- Do not add conventional search machinery to “help temporarily”.
- When a phase ends, record what changed and what the next pressure points are.

## Global non-goals
Across all phases, do not implement these as runtime decision mechanisms:
- alpha-beta / negamax / PVS
- quiescence search
- TT-driven search
- null-move pruning
- LMR
- killer/history heuristics
- handcrafted static evaluation as a fallback engine

## Global deliverables expected by the end of the program
- a stable Rust UCI engine
- a symbolic rules kernel with exact legality
- a documented action space and encoder
- a Python training stack for legality, dynamics, and planning models
- a Rust inference boundary
- a recurrent latent planner that selects moves without classical search
- evaluation harnesses, benchmarks, and reproducible experiments

---

# Phase 0 — Repository bootstrap

## Goal
Create the repository scaffold, workspace boundaries, basic documentation, and validation commands so the project can evolve in small, safe increments.

## Non-goals
- no chess rules yet
- no UCI yet
- no ML model yet
- no benchmark harness yet

## Deliverables
- Rust workspace under `rust/`
- Python project under `python/`
- root `README.md`
- root `AGENTS.md`
- root `PLANS.md`
- initial docs directories
- placeholder crates and modules with short crate-level docs
- basic formatting and lint configuration
- a minimal `engine-app` binary that starts and exits cleanly

## Files expected to change
- `rust/Cargo.toml`
- `rust/crates/*`
- `python/pyproject.toml`
- `.gitignore`
- `README.md`
- `docs/**`

## Suggested implementation steps
1. Create the workspace layout.
2. Add empty but documented crates.
3. Add a tiny binary target in `engine-app`.
4. Add lint and test commands.
5. Add a short architecture overview in `README.md`.

## Tests to add or run
- `cargo test --workspace`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `python -m pytest` with at least a trivial smoke test
- `ruff check python`

## Exit criteria
- the repository builds
- all placeholder crates compile
- test commands are wired and pass
- docs describe the planned architecture and constraints

## Risks / open questions
- exact crate boundaries may move later
- Python packaging choice may evolve

---

# Phase 1 — Exact chess state and rules kernel

## Goal
Build the symbolic chess core from scratch, with exact board state, legal move generation, and exact move application.

## Non-goals
- no UCI integration yet
- no evaluation
- no search
- no ML hooks beyond clean interfaces

## Deliverables
- `core-types` for colors, pieces, squares, files, ranks, move enums/structs
- `position` crate with board state, side to move, clocks, castling rights, en passant, repetition support
- `rules` crate with:
  - pseudo-legal move generation as needed internally
  - legal move generation
  - attack detection
  - move application
  - optional undo or immutable next-state creation
- FEN parse and serialize
- perft utility or test harness

## Files expected to change
- `rust/crates/core-types/**`
- `rust/crates/position/**`
- `rust/crates/rules/**`
- `tests/perft/**`
- `tests/positions/**`

## Suggested implementation steps
1. Define core enums and compact move representation.
2. Implement board state and FEN.
3. Implement attack maps / attack queries.
4. Implement legal move generation.
5. Implement move application.
6. Add perft and edge-case tests.

## Tests to add or run
- unit tests for each piece’s movement rules
- castling legality tests
- en passant legality tests, including pinned/discovered cases
- promotion tests
- in-check and double-check tests
- repetition bookkeeping tests
- perft on standard reference positions

## Exit criteria
- perft matches known values on the chosen suite
- FEN roundtrips cleanly
- no illegal move is produced in tested positions
- all rule edge cases above have explicit tests

## Risks / open questions
- exact board representation may affect future performance
- undo vs immutable next-state design should remain flexible until planner needs are clearer

---

# Phase 2 — UCI shell with legal stub move output

## Goal
Add the UCI protocol shell so the engine can connect to GUIs and emit legal moves, while still avoiding any classical search path.

## Non-goals
- no search
- no evaluation engine
- no neural inference yet

## Deliverables
- `uci-protocol` crate with parser and command handling
- `engine-app` main loop handling:
  - `uci`
  - `isready`
  - `ucinewgame`
  - `position`
  - `go`
  - `stop`
  - `quit`
- internal game state synchronization from UCI commands
- a legal stub move policy for `go`, such as “first legal move” or deterministic policy from legal list
- structured logging or debug mode

## Files expected to change
- `rust/crates/uci-protocol/**`
- `rust/crates/engine-app/**`

## Suggested implementation steps
1. Model UCI commands and responses.
2. Maintain current position from `startpos` and FEN.
3. Implement a minimal `go` path that asks the rules kernel for legal moves.
4. Emit `bestmove` deterministically.
5. Add protocol smoke tests.

## Tests to add or run
- parser tests for the core UCI commands
- position reconstruction tests from move lists
- smoke test: send a short UCI session and verify legal `bestmove`

## Exit criteria
- a GUI can connect and talk to the engine
- `bestmove` is always legal in tested scenarios
- the binary remains free of search/eval logic

## Risks / open questions
- future async handling for `go`/`stop`
- UCI option surface should stay minimal until runtime stabilizes

---

# Phase 3 — Action space and object-centric encoder

## Goal
Define the action vocabulary and the first deterministic encoder that converts an exact symbolic position into model-ready features or tokens.

## Non-goals
- no training yet
- no inference yet
- no planner yet

## Deliverables
- `action-space` crate documenting the move vocabulary
- factorized move representation for model IO, for example:
  - from-square head
  - to-square or move-type head
  - promotion head
- encode/decode utilities between symbolic moves and model action representation
- `encoder` crate producing deterministic inputs from a position:
  - piece tokens
  - rule token(s)
  - optional square tokens
  - optional relational features
- schema documentation for tensor shapes and token semantics

## Files expected to change
- `rust/crates/action-space/**`
- `rust/crates/encoder/**`
- `docs/architecture/encoding.md`

## Suggested implementation steps
1. Choose the canonical move vocabulary.
2. Implement move tokenization and inverse mapping.
3. Implement object-centric piece/rule encoding.
4. Add deterministic tests.
5. Document feature ordering and invariants.

## Tests to add or run
- move encode/decode roundtrips
- deterministic encoding tests on fixed FENs
- promotion and castling mapping tests
- invariance tests where appropriate and intentionally non-invariant cases documented

## Exit criteria
- every legal move in a position can be represented in the action space
- encoder output is deterministic and documented
- test fixtures cover edge cases

## Risks / open questions
- whether square tokens are needed in v1
- whether to include attack/pin helper features now or later

---

# Phase 4 — Dataset and label pipeline

## Goal
Build the dataset pipeline that turns exact symbolic positions into supervised training examples for legality, policy, dynamics, and later planning.

## Non-goals
- no selfplay yet
- no learned planner yet
- no engine-strength optimization yet

## Deliverables
- dataset schema definitions
- scripts to generate positions from games, sampled positions, or synthetic suites
- labels for:
  - legal move set
  - selected action encoding
  - next-state targets
  - WDL or surrogate outcome labels
  - tactical annotations where practical
- train/validation/test split generation
- reporting on dataset composition and edge-case coverage

## Files expected to change
- `python/train/datasets/**`
- `python/scripts/make_dataset.py`
- `docs/architecture/datasets.md`

## Suggested implementation steps
1. Define example schema.
2. Add raw position ingestion.
3. Add label generation via the exact rules kernel.
4. Create split logic.
5. Add sanity checks and summary reports.

## Tests to add or run
- dataset schema validation
- label consistency checks
- reconstruction checks from serialized examples
- coverage report for checks, promotions, castling, en passant, endgames

## Exit criteria
- a reproducible dataset build command exists
- labels are internally consistent
- reports expose class balance and edge-case counts

## Risks / open questions
- training targets for policy may evolve
- dataset source mix can bias the planner later

---

# Phase 5 — Legality and policy proposer v1

## Goal
Train and integrate the first model that predicts legal moves and a prior over candidate moves from the encoded position.

## Non-goals
- no latent dynamics yet
- no opponent reasoning yet
- no recurrent planner yet

## Deliverables
- PyTorch model for legality + policy
- training loop and config
- model export path
- Rust-side inference loading for proposer outputs
- metrics for legal-set precision/recall
- at least one structured proposer comparison beyond the flat baseline

## Exit criteria
- proposer metrics are reported on held-out data
- model export can be consumed from Rust
- runtime still verifies legality symbolically
- the repository records which proposer family is current default vs. experimental

---

# Phase 6 — Dynamics model v1

## Goal
Add the first action-conditioned latent dynamics model.

## Non-goals
- no opponent reasoning yet
- no recurrent planner yet

## Deliverables
- encoder state `z = E(s)`
- dynamics step `z' = G(z, a)`
- one-step reconstruction targets
- multi-step drift measurements
- Rust-side inference boundary for `E` + `G`
- explicit treatment of special-move transitions

## Exit criteria
- exact next-state accuracy and drift metrics are reported
- special-move cases are tracked separately
- Rust can execute the exported encoder/dynamics stack
- the dynamics design is local and action-conditioned rather than a hidden classical fallback

---

# Phase 7 — Opponent model and 2-ply latent planner

## Goal
Add the opponent model and a first adversarial latent planner.

## Non-goals
- no recurrent memory planner yet

## Deliverables
- opponent reply model `O(z)`
- proposer -> dynamics -> opponent -> aggregation runtime path
- root WDL head
- planner-vs-proposer comparisons
- explicit opponent-facing diagnostics such as reply quality or threat signals

## Exit criteria
- better than proposer-only baseline
- tactical smoke tests improve on trivial forcing cases
- runtime still avoids classical search

---

# Phase 8 — Recurrent planner with memory

## Goal
Extend the planner into a bounded recurrent latent planner with memory slots and uncertainty-guided compute allocation.

## Deliverables
- planner memory slots
- multiple internal deliberation steps
- uncertainty-aware prioritization
- planner diagnostics and regression tests
- a bounded compute budget that stays visibly distinct from classical search

## Exit criteria
- measurable gain over the 2-ply planner
- better tactical robustness
- uncertainty outputs are calibrated enough to inspect

---

# Phase 9 — Selfplay and curriculum

## Goal
Build selfplay, replay buffering, checkpoint comparison, and curriculum training around the planner stack.

## Deliverables
- selfplay loop
- replay buffer
- checkpoint evaluation harness
- curriculum schedule
- reproducible reports

## Exit criteria
- selfplay games complete legally
- checkpoint comparisons are reproducible
- training outputs are versioned and documented

---

# Phase 10 — Planner-driven UCI runtime

## Goal
Make the latent planner the primary UCI move-selection path.

## Deliverables
- planner-driven `go`
- time budgeting
- runtime diagnostics
- final symbolic legality verification before `bestmove`

## Exit criteria
- full games run through UCI
- no illegal UCI moves are emitted
- no classical search fallback exists

---

# Phase 11 — Hardening

## Goal
Harden the system with benchmarks, regression tracking, optimization, and failure-mode documentation.

## Deliverables
- benchmark harnesses
- regression tracking
- model metadata
- profiling notes
- runtime and training failure-mode documentation

## Exit criteria
- new checkpoints are comparable
- regressions are caught automatically
- hot-path optimizations preserve architectural boundaries

---

# Phase 10 — LAPv1 unified planner (supplemental architecture track)

This supplemental track does not renumber the canonical milestone sequence above.
It records the current unification target for the planner family before any future
planner-driven UCI promotion is attempted.

## Goal
Unify the current planner-arm zoo behind one bounded latent-adversarial planner
stack that keeps exact legality symbolic, keeps runtime compute bounded, and
stays observably distinct from classical search.

## Non-goals
- no alpha-beta, negamax, PVS, TT-search, or quiescence runtime
- no handcrafted static evaluation fallback
- no removal of existing planner checkpoints until LAPv1 beats them empirically
- no unbounded inner-loop recursion or hidden search-tree expansion

## Deliverables
- versioned symbolic `StateContextV1` alongside `CandidateContextV2`
- piece-intention encoder
- relational state embedder producing `z_root` and uncertainty
- large value and candidate-policy heads
- bounded recurrent deliberation loop with rollback-safe trace emission
- wrapper model and staged trainer support (`T1` static heads, `T2` deliberation-on)
- runtime-facing LAPv1 agent contract and benchmark-ready arena template
- migration plan for which older planner arms stay as references vs. deprecate
- LAPv2 depth-alignment refinements documented in
  `docs/architecture/lapv2-depth-alignment.md`, including step-rank progress,
  learned step utility, depth-conditioned recurrent updates, guarded frontier
  diversity, and external-hard validation slices.

## Exit criteria
- LAPv1 reaches a prepared-and-testable state without disturbing existing runtime paths
- LAPv1 stays inside the architectural boundaries from `AGENTS.md`
- the repository documents which current planner arms remain benchmarks during migration
- promotion pressure is defined empirically rather than assumed architecturally
