# EngineKonzept/AGENTS.md

## Mission
Build a chess engine from scratch around **latent adversarial planning**.

The runtime decision path must converge toward this architecture:

`position -> encoder -> legality/policy proposer -> latent dynamics -> opponent module -> recurrent planner -> WDL + move selection -> UCI output`

This repository is **not** a conventional chess engine. Do not drift toward alpha-beta based designs. The project may use a symbolic rules core, but that core exists only to guarantee correctness, provide labels, and validate outputs.

## Product intent
The engine must:
- speak UCI cleanly
- maintain exact chess rules and exact legal move semantics
- use learned latent planning as the primary move-selection mechanism
- remain understandable, testable, and reproducible at every phase

The engine must **not** become a classical engine with a neural add-on.

## Hard architectural boundaries
Allowed symbolic components:
- board state representation
- FEN parsing and serialization
- exact legal move generation
- exact move application and undo if needed
- repetition / fifty-move bookkeeping
- test oracles
- dataset generation
- final legality verification before emitting `bestmove`

Forbidden as runtime decision mechanisms:
- alpha-beta
- negamax
- PVS / NegaScout
- quiescence search
- transposition-table search as the primary planner
- null-move pruning
- late move reductions
- killer/history/countermove heuristics
- handcrafted static evaluation as a fallback engine
- conventional opening book as a substitute for planning

If a task seems to require one of those, stop and report the pressure instead of quietly adding it.

## Design doctrine
1. **Correctness before strength.**
2. **Latent planning before conventional search.**
3. **Small, testable modules before clever monoliths.**
4. **No hidden fallback engine.**
5. **Every phase leaves the tree buildable and documented.**

## Language split
Use:
- **Rust** for runtime, UCI, rules, data structures, inference integration, planner loop, evaluation harnesses
- **Python** for datasets, training, experiments, export, model analysis

Do not move training logic into Rust prematurely.
Do not move UCI/runtime logic into Python.

## Repository layout
Target layout:

```text
EngineKonzept/
  AGENTS.md
  PLANS.md
  README.md
  docs/
    architecture/
    experiments/
    phases/
  rust/
    Cargo.toml
    crates/
      engine-app/
      uci-protocol/
      core-types/
      position/
      rules/
      action-space/
      encoder/
      inference/
      planner/
      eval-metrics/
      selfplay/
      tools/
  python/
    pyproject.toml
    train/
      datasets/
      models/
      losses/
      trainers/
      export/
    scripts/
  tests/
    perft/
    positions/
    planner/
  models/
  artifacts/
```

Small deviations are acceptable if they simplify the codebase, but preserve the split between runtime, rules, ML, and tooling.

## ExecPlans and PLANS.md
For any feature that spans more than a small bugfix or a single module, use an **ExecPlan** from `PLANS.md`.

Rules:
- Read `AGENTS.md` and the relevant section of `PLANS.md` before editing.
- Work only on the current phase unless explicitly told to advance.
- Do not pre-implement future phases “just because it is convenient”.
- When a phase is complete, update progress notes in `PLANS.md` or in the task summary.
- If an interface change affects later phases, note it explicitly.

If the user says “implement Phase X”, treat the matching `PLANS.md` section as the source of truth.
If the user says “use an ExecPlan”, produce or follow an implementation plan that matches the `PLANS.md` structure.

## Phase discipline
You must preserve the intended ordering:
1. workspace bootstrap
2. exact rules kernel
3. UCI shell
4. action space and encoder
5. data pipeline
6. legality/policy proposer
7. latent dynamics
8. opponent model and 2-ply latent planner
9. recurrent planner with memory
10. selfplay and curriculum
11. UCI runtime fully driven by the planner
12. hardening, benchmarks, optimization

Do not skip ahead to planner code before rules, tests, encoding, and data contracts exist.

## Chess-domain invariants
Always preserve these invariants:
- no illegal move may be emitted under UCI
- castling legality must be exact
- en passant legality must be exact, including discovered-line edge cases
- promotion semantics must be exact
- side-to-move and clocks must remain consistent
- repetition and fifty-move state must be reproducible from move history or tracked state
- the symbolic rules layer remains the authority for legality
- the neural planner may propose moves, but legality is verified before output

## Runtime model philosophy
The neural runtime should evolve through these stages:
- legality/policy proposal
- one-step latent dynamics
- opponent reply modeling
- soft-min or equivalent adversarial aggregation
- recurrent internal deliberation with a bounded compute budget
- calibrated WDL + move choice

This repository is trying to learn:
- which moves are legal
- which legal moves are promising
- what latent state results from a move
- what the opponent’s best reply is likely to be
- how to iterate internal planning without a classical search tree

## Coding standards
General:
- prefer explicit types and explicit interfaces
- avoid premature abstraction
- avoid macros unless they materially simplify repetitive correctness code
- keep functions short enough to test in isolation
- document invariants near the type or function they protect
- avoid hidden global state
- make determinism the default unless stochasticity is intentional

Rust:
- stable Rust only unless explicitly approved
- prefer small crates with clear ownership boundaries
- derive traits thoughtfully; avoid unnecessary cloning
- keep unsafe code out unless there is a very strong reason
- if unsafe is introduced, document the invariant in detail and add focused tests

Python:
- use type hints for nontrivial functions
- keep notebooks optional; core logic belongs in importable modules and scripts
- keep training scripts reproducible from config files or CLI arguments

## Testing and validation
Minimum validation priorities, in order:
1. rule correctness
2. illegal-move safety
3. reproducibility
4. latent-planner quality
5. speed

Expected commands when relevant:
- `cargo fmt --all`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace`
- `ruff check python`
- `python -m pytest`

If a phase adds new tooling, update this section or the phase notes accordingly.

## Definition of done
A task is done only when:
- code builds
- tests pass
- docs reflect the new interfaces and constraints
- no forbidden classical-search logic slipped in
- the summary clearly states what changed, what remains, and what assumptions were made

## Documentation obligations
When changing architecture or interfaces, update one or more of:
- `README.md`
- `docs/architecture/*`
- `docs/phases/*`
- inline module docs

New crates or training modules should include a short purpose description.

## Review protocol
After each substantial phase:
- run the relevant tests
- summarize changed files
- list known limitations
- list risks for the next phase

If running interactively in Codex, request a `/review` before declaring the phase complete.

## Subagent policy
Use subagents only when explicitly requested or when the task clearly benefits from parallel exploration and the user has asked for broad planning or investigation.

Good subagent use cases:
- parallel exploration of model architecture options
- separate audits of rules correctness, data schemas, and inference boundaries
- benchmarking multiple encoding designs

Do not use subagents for simple local edits.

## Anti-patterns to avoid
Do not:
- sneak in a minimax baseline “just temporarily” without clearly labeling it as a non-runtime benchmark tool
- introduce a handcrafted eval because the model is weak
- overfit interfaces to one early model checkpoint
- bury rule assumptions in the ML code
- let dataset scripts silently mutate label semantics
- mix research prototypes with production runtime paths without a clear boundary

## Preferred decision rule when stuck
If there is tension between:
- elegance and testability -> choose testability
- speed and correctness -> choose correctness
- novelty and observability -> choose observability
- short-term Elo and architectural purity -> keep the architectural boundary unless the user explicitly decides otherwise

## Expected task summaries
When finishing a task, summarize under these headings:
- What changed
- Tests run
- Open risks
- Next recommended step

