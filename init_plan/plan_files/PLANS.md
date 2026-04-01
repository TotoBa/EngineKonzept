# Codex Execution Plans for latent-chess

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
- Rust inference boundary for proposer inference
- UCI runtime path that can score or rank legal candidates using the model while still verifying legality symbolically

## Files expected to change
- `python/train/models/proposer.py`
- `python/train/losses/**`
- `python/train/trainers/**`
- `python/scripts/train_legality.py`
- `rust/crates/inference/**`
- `rust/crates/engine-app/**`

## Suggested implementation steps
1. Define proposer model API.
2. Train legality head.
3. Add policy head.
4. Add metrics and calibration reports.
5. Export model and integrate inference in Rust.
6. Add runtime candidate-ranking path.

## Tests to add or run
- training smoke test on a tiny dataset
- export/import test
- Rust-side inference contract tests
- metrics by edge-case bucket

## Exit criteria
- legality recall is high enough that the runtime can rely on the proposer for ranking candidates
- Rust can load and run the model
- UCI runtime can use the proposer without emitting illegal moves

## Metrics to track
- legal-set precision / recall
- recall in check positions
- promotion recall
- en passant recall
- castling recall
- inference latency

## Risks / open questions
- action-factorization quality
- whether auxiliary tactical heads are needed immediately

---

# Phase 6 — Latent dynamics model v1

## Goal
Learn a one-step latent transition model that predicts how the internal state changes when an action is applied.

## Non-goals
- no recurrent planner yet
- no full selfplay yet

## Deliverables
- position encoder `E`
- action-conditioned dynamics model `G(z, a) -> z'`
- optional reconstruction heads for board/rule state
- one-step and multi-step training losses
- export path and Rust inference support for dynamics

## Files expected to change
- `python/train/models/encoder.py`
- `python/train/models/dynamics.py`
- `python/scripts/train_dynamics.py`
- `rust/crates/inference/**`
- `docs/architecture/dynamics.md`

## Suggested implementation steps
1. Define latent-state contract.
2. Train one-step transition.
3. Add reconstruction heads.
4. Measure multi-step drift.
5. Export and load from Rust.

## Tests to add or run
- one-step consistency checks
- multi-step rollout drift tests on held-out examples
- special-move transition tests
- Rust-side contract tests for latent tensor shapes

## Exit criteria
- exact next-state or high-fidelity reconstruction is reliable enough to support latent planning experiments
- multi-step drift is measured and reported
- Rust can execute encoder + dynamics inference

## Metrics to track
- exact next-state accuracy
- board occupancy accuracy
- rule-state accuracy
- special-move accuracy
- 2/4/8-step drift

## Risks / open questions
- latent bottleneck size
- whether to reconstruct full board state or targeted auxiliary state only

---

# Phase 7 — Opponent model and 2-ply latent planner

## Goal
Create the first genuinely planner-like runtime: propose candidate moves, imagine our move, imagine opponent replies, aggregate adversarially, and choose a move.

## Non-goals
- no conventional search fallback
- no multi-step recurrent memory yet

## Deliverables
- opponent reply model or symmetric reply usage
- root candidate selection logic
- latent 2-ply aggregation, such as soft-min over opponent replies
- root WDL head
- runtime integration into UCI `go`

## Files expected to change
- `python/train/models/opponent.py`
- `python/train/models/planner_v1.py`
- `python/scripts/train_planner.py`
- `rust/crates/planner/**`
- `rust/crates/engine-app/**`

## Suggested implementation steps
1. Define planner state and APIs.
2. Generate candidate root moves from proposer.
3. Use dynamics for imagined root transitions.
4. Use opponent module for reply candidates.
5. Aggregate reply values.
6. Emit root policy and WDL.

## Tests to add or run
- planner smoke tests on fixed tactical positions
- legality checks on final chosen move
- comparison against proposer-only baseline
- mate-in-1 and simple tactic benchmark slices

## Exit criteria
- planner beats proposer-only on the chosen evaluation slices
- runtime remains legal and stable
- WDL output is exposed and logged

## Metrics to track
- top-1 move accuracy on held-out labels
- puzzle-slice solve rate
- invalid proposal rate
- final illegal output rate (target: zero)
- latency by candidate budget

## Risks / open questions
- top-k and top-m budgets
- soft-min vs alternative adversarial aggregation

---

# Phase 8 — Recurrent latent planner with memory

## Goal
Extend the planner from a fixed 2-ply latent computation to a bounded recurrent internal deliberation loop with memory slots and uncertainty-aware budget allocation.

## Non-goals
- no external tree search
- no TT as primary planner

## Deliverables
- planner memory state
- recurrent deliberation steps
- branch prioritization based on uncertainty or instability
- improved root policy and WDL after multiple inner steps
- diagnostics for internal planning trajectories

## Files expected to change
- `python/train/models/planner_recurrent.py`
- `rust/crates/planner/**`
- `docs/architecture/planner.md`

## Suggested implementation steps
1. Define memory-slot structure.
2. Add recurrent update block.
3. Add branch-selection policy.
4. Add confidence or uncertainty head.
5. Integrate bounded inner-loop compute.

## Tests to add or run
- deterministic planner-loop tests
- regression tests for memory updates
- comparison against Phase 7 on tactical and strategic suites

## Exit criteria
- measurable improvement over Phase 7
- planning loop remains bounded and observable
- logs explain which candidate lines received extra inner compute

## Metrics to track
- quality vs number of inner steps
- uncertainty calibration
- latency vs strength trade-off
- failure buckets by motif

## Risks / open questions
- recurrent instability
- latent collapse or repeated-loop degeneration

---

# Phase 9 — Selfplay and curriculum training

## Goal
Establish selfplay and curriculum mechanisms so the planner can improve from its own trajectories instead of only offline supervised labels.

## Non-goals
- no cloud-scale distributed system unless explicitly requested

## Deliverables
- selfplay harness
- replay buffer format
- curriculum schedule
- checkpoint evaluation against prior versions
- experiment configuration and artifact tracking

## Files expected to change
- `rust/crates/selfplay/**`
- `python/train/trainers/selfplay.py`
- `python/scripts/run_selfplay.py`
- `docs/experiments/selfplay.md`

## Suggested implementation steps
1. Define selfplay game loop.
2. Capture trajectories and planner outputs.
3. Build replay-buffer ingestion.
4. Add curriculum stages.
5. Add checkpoint comparison reports.

## Tests to add or run
- selfplay smoke tests for complete legal games
- replay-buffer schema tests
- checkpoint load/eval tests

## Exit criteria
- reproducible selfplay training cycle exists
- new checkpoints can be compared against old ones on fixed suites
- selfplay games complete legally

## Risks / open questions
- exploration schedule
- collapse toward shallow move priors

---

# Phase 10 — Full UCI runtime driven by the planner

## Goal
Make the planner the primary runtime decision system for `go`, with the symbolic layer limited to legality and safety verification.

## Non-goals
- no hidden minimax fallback
- no handcrafted eval rescue path

## Deliverables
- UCI move selection through proposer + dynamics + opponent + recurrent planner
- time-budgeted inner-loop controls
- final legality verification before `bestmove`
- runtime diagnostics for planner behavior

## Files expected to change
- `rust/crates/engine-app/**`
- `rust/crates/planner/**`
- `rust/crates/inference/**`

## Suggested implementation steps
1. Define planner runtime configuration.
2. Integrate time budgeting.
3. Add final legality/safety gate.
4. Log planner statistics.
5. Run full-game validations.

## Tests to add or run
- complete-game smoke tests
- GUI interoperability tests
- stress tests on time budgets
- repeated-run determinism tests where expected

## Exit criteria
- engine plays complete legal UCI games using planner-driven move choice
- no classical-search fallback exists in runtime
- planner metrics are observable in logs

## Metrics to track
- move latency
- planner inner-step usage
- final safety-gate rejection rate
- game completion rate

## Risks / open questions
- time management under very low budgets
- planner instability in tactical chaos

---

# Phase 11 — Hardening, evaluation, and optimization

## Goal
Stabilize the system, improve observability, and optimize for practical use without violating the architectural boundaries.

## Non-goals
- no abandonment of the planner architecture for easier Elo gains

## Deliverables
- benchmark harnesses
- tactical and strategic evaluation suites
- regression tracking
- model registry / checkpoint metadata
- runtime profiling and optimization reports
- documentation of known failure modes

## Files expected to change
- `rust/crates/eval-metrics/**`
- `docs/experiments/**`
- CI scripts and benchmark tooling

## Suggested implementation steps
1. Add benchmark suites.
2. Add profiling hooks.
3. Track regressions per model version.
4. Optimize hot paths without changing semantics.
5. Document known limitations.

## Tests to add or run
- benchmark regression runs
- profile-guided sanity checks
- compatibility tests for model loading

## Exit criteria
- performance and behavior regressions are visible
- checkpoint lineage is documented
- runtime optimizations preserve legality and planner semantics

## Risks / open questions
- compression vs accuracy trade-offs
- deployment target constraints

---

# Optional research tracks after Phase 11

These are explicitly optional and must not contaminate the core implementation plan prematurely.

## Track A — Improved latent representations
Explore:
- relational attention variants
- square-token hybrids
- sparse piece-centric updates
- auxiliary tactical geometry heads

## Track B — Better adversarial aggregation
Explore:
- soft-min variants
- learned reply selectors
- uncertainty-aware candidate expansion
- calibrated draw modeling

## Track C — Runtime optimization
Explore:
- quantization
- distilled proposer/planner pairs
- cached latent features with exact invalidation
- CPU-specific inference optimizations

## Track D — Tooling and orchestration
Explore:
- subagent-assisted research prompts
- structured `codex exec --json` workflows
- model artifact metadata schemas
