# UCI Runtime

The repository started with a thin Phase-2 UCI shell around the exact rules
kernel. It now also has the first planner-driven Rust `go` path over the
current shipped runtime bundle contract.

## Supported Commands

- `uci`
- `debug on|off`
- `isready`
- `ucinewgame`
- `position startpos [moves ...]`
- `position fen <fen> [moves ...]`
- `go ...`
- `stop`
- `quit`

## Runtime Behavior

- `position` reconstructs the exact current state by replaying legal moves over either `startpos` or a supplied FEN.
- `go` uses exact legal candidates, respects `searchmoves`, and, when a Rust-loadable proposer bundle is available, runs a bounded frontier-deliberation selector over the legal root moves before emitting `bestmove`.
- the current shipped Rust runtime contract is the symbolic proposer bundle:
  - `metadata.json`
  - `proposer.pt2`
  - `symbolic_runtime.bin`
- without a loadable bundle the engine still falls back to a deterministic legal move so the shell remains usable, but that fallback is explicitly not a hidden evaluation/search engine.
- `stop` is currently a no-op because there is no asynchronous search yet.
- parse and position-reconstruction errors are surfaced as `info string ...` only when debug mode is enabled.
- the planner path emits a summary `info string` for the chosen move, and `debug on` additionally emits per-step frontier diagnostics.

## Runtime Controls

The default bundle lookup still uses `ENGINEKONZEPT_PROPOSER_BUNDLE`.

The frontier planner can be tuned through environment variables:

- `ENGINEKONZEPT_FRONTIER_ROOT_TOP_K`
- `ENGINEKONZEPT_FRONTIER_BEAM_WIDTH`
- `ENGINEKONZEPT_FRONTIER_MIN_INNER_STEPS`
- `ENGINEKONZEPT_FRONTIER_MAX_INNER_STEPS`
- `ENGINEKONZEPT_FRONTIER_STABLE_MARGIN`
- `ENGINEKONZEPT_FRONTIER_TACTICAL_PRESSURE_SCALE`
- `ENGINEKONZEPT_FRONTIER_EXPLORATION_SCALE`
- `ENGINEKONZEPT_FRONTIER_REVISIT_PENALTY_SCALE`
- `ENGINEKONZEPT_FRONTIER_STABILITY_HYSTERESIS`

`go depth ...` and `go movetime ...` also clamp the maximum inner-step budget,
but they still do so as a bounded deliberation budget, not as classical search
depth.

## Boundary

This runtime remains intentionally bounded:

- no alpha-beta
- no MCTS / PUCT
- no board-tree expansion
- no hidden classical evaluation engine
- no hidden fallback engine

Its current job is:

- protocol correctness
- exact legal move reconstruction
- a planner-driven, traceable, non-classical `bestmove` path over the shipped
  Rust-loadable bundle contract
