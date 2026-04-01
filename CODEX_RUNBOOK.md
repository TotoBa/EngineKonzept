# Codex CLI Runbook for EngineKonzept

This runbook is the operator guide for driving Codex through the repository phase by phase.

It assumes:
- `AGENTS.md` and `PLANS.md` are committed at the repo root
- work happens inside a Git repository
- Codex is used in both interactive and non-interactive modes

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

## Review gate
After each phase, answer these four questions:
1. What changed?
2. What tests ran?
3. What open risks remain?
4. What is the next recommended step?

## Guardrail reminder
If Codex starts trying to solve a planner weakness by adding conventional search logic, stop the run, point it back to `AGENTS.md`, and restate the current phase goal.

