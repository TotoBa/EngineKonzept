# Codex CLI Runbook for latent-chess

This runbook is the operator guide for driving Codex through the repository phase by phase.

It assumes:
- `AGENTS.md` and `PLANS.md` are committed at the repo root
- work happens inside a Git repository
- Codex is used in both interactive and non-interactive modes

## Why this setup works well with Codex
- `AGENTS.md` is read by Codex before work starts, so persistent repo rules belong there.
- `PLANS.md` is a good fit for long, multi-step work and is explicitly recommended by OpenAI’s Codex cookbook as a living design/implementation document.
- the interactive TUI is best for architecture, review, and guided refactors
- `codex exec` is best for bounded phase work, CI, and repeatable implementation prompts

## Day-0 setup
1. Initialize the repository with Git.
2. Copy `AGENTS.md` and `PLANS.md` into the repo root.
3. Optionally run `/init` in Codex to compare the scaffold with the curated `AGENTS.md`, but keep the curated version as source of truth.
4. Start Codex in the repo root.
5. Ask Codex to read `AGENTS.md` and `PLANS.md` before any implementation.

## Session discipline
Use this rhythm for every phase:
1. create or update the phase branch
2. start Codex in the repo root
3. ask for a short phase plan referencing `PLANS.md`
4. implement only that phase
5. run tests and linters
6. run `/review` in the interactive TUI or ask for a working-tree review
7. commit
8. summarize open risks for the next phase

## Recommended split between TUI and `codex exec`
Use the interactive TUI for:
- architecture questions
- reviewing interfaces before implementation
- code review and cleanup
- comparing alternative designs

Use `codex exec` for:
- well-scoped phase implementation
- CI or automation jobs
- repeatable prompt templates
- structured output flows

## Suggested global commands
Interactive:
```bash
codex
```

Resume the last session in this repo:
```bash
codex resume --last
```

Bounded non-interactive work:
```bash
codex exec "Read AGENTS.md and PLANS.md, implement Phase X only, run tests, and stop at the exit criteria."
```

Machine-readable logs for automation:
```bash
codex exec --json "Read AGENTS.md and PLANS.md and summarize the repository status" > codex-events.jsonl
```

## Phase-by-phase prompts

### Phase 0
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 0 only. Create the Rust workspace, Python project, docs tree, placeholder crates, basic lint/test wiring, and a minimal engine-app binary. Do not implement chess rules, UCI, or any model code yet. Run the required checks and stop exactly at the Phase 0 exit criteria. Then summarize changed files, tests run, open risks, and the next recommended step."
```

### Phase 1
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 1 only. Build the exact symbolic chess state and rules kernel from scratch: core types, board state, FEN parse/serialize, legal move generation, move application, castling, en passant, promotions, clocks, and repetition bookkeeping. Add thorough edge-case tests and perft coverage. Do not add UCI, search, evaluation, or model code. Run the Rust checks and stop at the Phase 1 exit criteria."
```

### Phase 2
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 2 only. Add a clean UCI shell with uci, isready, ucinewgame, position, go, stop, and quit. Maintain exact position state and emit a deterministic legal stub move on go. Do not add search or evaluation. Add protocol smoke tests, run checks, and stop at the Phase 2 exit criteria."
```

### Phase 3
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 3 only. Design and implement the action-space crate and the object-centric encoder. Support deterministic move encode/decode, piece tokens, rule tokens, and documented tensor or token schemas. Do not begin training or inference. Add roundtrip and determinism tests, run checks, and stop at the Phase 3 exit criteria."
```

### Phase 4
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 4 only. Build the dataset and label pipeline for legality, policy, dynamics, and WDL-related supervision. Use the exact symbolic rules kernel as the label oracle. Add dataset schema docs, reproducible build scripts, sanity reports, and validation tests. Do not add selfplay yet. Stop at the Phase 4 exit criteria."
```

### Phase 5
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 5 only. Create the first legality and policy proposer model in Python, add training and export, then integrate Rust-side inference for proposal scoring. Keep symbolic legality verification in the runtime. Add metrics for legal-set precision and recall, especially in check, castling, promotion, and en passant cases. Stop at the Phase 5 exit criteria."
```

### Phase 6
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 6 only. Add the latent encoder and one-step action-conditioned dynamics model, including training, export, and Rust-side inference integration. Measure one-step accuracy and multi-step drift. Keep the implementation aligned with the architectural constraint that this is building latent planning, not conventional search. Stop at the Phase 6 exit criteria."
```

### Phase 7
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 7 only. Add the opponent model and a 2-ply latent planner: proposer -> imagine our move -> imagine opponent replies -> aggregate adversarially -> choose move and WDL. Integrate it into UCI go without introducing alpha-beta or fallback search. Add tactical smoke tests and planner-vs-proposer baseline comparisons. Stop at the Phase 7 exit criteria."
```

### Phase 8
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 8 only. Extend the planner to a bounded recurrent latent planner with memory slots, uncertainty-aware candidate prioritization, and multiple inner deliberation steps. Keep the planner observable and bounded. Add diagnostics and regression tests. Stop at the Phase 8 exit criteria."
```

### Phase 9
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 9 only. Build selfplay, replay buffering, checkpoint comparison, and a curriculum training loop for the planner stack. Keep outputs reproducible and fully documented. Add smoke tests for complete legal selfplay games and checkpoint evaluation. Stop at the Phase 9 exit criteria."
```

### Phase 10
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 10 only. Make the planner the primary UCI runtime decision mechanism, with only symbolic legality and safety verification before bestmove. Add time budgeting, planner diagnostics, and full-game validations. Do not add a conventional-search fallback. Stop at the Phase 10 exit criteria."
```

### Phase 11
```bash
codex exec "Read AGENTS.md and PLANS.md. Implement Phase 11 only. Harden the system with benchmarks, regression tracking, model metadata, profiling, and documentation of failure modes. Optimize hot paths without violating architectural boundaries. Stop at the Phase 11 exit criteria."
```

## Interactive prompts worth using in the TUI
Before a phase:
```text
Read AGENTS.md and the relevant phase in PLANS.md. Give me a concise implementation plan, the files you expect to touch, the tests you will run, and the top 3 architectural risks before you start editing.
```

After a phase:
```text
Review the working tree against AGENTS.md and the relevant PLANS.md phase. Check for drift toward conventional engine techniques, missing tests, poor module boundaries, or undocumented interfaces.
```

For architecture checkpoints:
```text
Compare the current codebase against the target architecture in AGENTS.md. Tell me where the implementation is converging well and where shortcuts are creating future debt.
```

## Recommended Git rhythm
Per phase:
```bash
git checkout -b phase-X-name
# run Codex
# run tests
git add -A
git commit -m "phase X: short description"
```

Checkpoint after every significant sub-milestone. Avoid mixing multiple phases in one commit.

## Suggested review gates
After each phase, ask Codex to answer these four questions:
1. What changed?
2. What tests ran?
3. What open risks remain?
4. What is the next recommended step?

## When to use subagents
Use subagents only when you explicitly want parallel work, for example:
- one agent auditing rule correctness
- one agent reviewing encoder contracts
- one agent proposing planner APIs

Example interactive prompt:
```text
Spawn three subagents. One audits the rules kernel for missing edge cases, one audits the Rust/Python inference boundary, and one audits PLANS.md alignment. Then merge the findings into one actionable review.
```

## Guardrails for later phases
When the model is weak, Codex may be tempted to add classical engine shortcuts. Prevent that with direct prompts like:
```text
Do not add alpha-beta, negamax, quiescence, TT search, handcrafted eval, or any hidden fallback engine. If the planner is weak, improve observability, labels, datasets, or planner structure instead.
```

## Useful automation patterns
Structured output with a schema can be useful for nightly checks, for example:
- repo health summary
- planner regression report
- model metadata extraction

Example:
```bash
codex exec --json "Read AGENTS.md and PLANS.md and summarize the current phase status" > status.jsonl
```

## Final operator rule
If Codex starts trying to solve a planner weakness by adding conventional search logic, stop the run, point it back to `AGENTS.md`, and restate the current phase goal.
