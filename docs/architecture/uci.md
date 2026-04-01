# UCI Shell

Phase 2 adds the first runtime-facing shell around the exact rules kernel.

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
- `go` does not search. It asks the exact rules kernel for legal moves, honors `searchmoves` restrictions, and emits a deterministic stub `bestmove`.
- `stop` is currently a no-op because there is no asynchronous search yet.
- parse and position-reconstruction errors are surfaced as `info string ...` only when debug mode is enabled.

## Boundary

This shell is intentionally thin:

- no alpha-beta
- no evaluation
- no timing logic beyond accepting `go` arguments syntactically
- no hidden fallback engine

Its job in Phase 2 is only protocol correctness and legal move emission.
