# Dataset Architecture

Phase 4 adds a reproducible dataset pipeline that keeps label generation anchored to the exact symbolic rules kernel.

## Raw Source Formats

The Python builder currently accepts four source formats:

- `edge-cases`: `name|fen` lines such as [edge_cases.txt](/home/torsten/EngineKonzept/tests/positions/edge_cases.txt)
- `fen-lines`: one FEN per line, optionally `name|fen`
- `epd`: synthetic suites using the first four EPD fields, normalized to full FEN with `0 1`
- `jsonl`: structured records with optional `selected_move_uci`, `result`, `source`, and `metadata`

For small reproducible Phase-5 policy runs, the repository also carries a labeled JSONL seed set at [policy_seed.jsonl](/home/torsten/EngineKonzept/tests/positions/policy_seed.jsonl).

## Exact Oracle Boundary

Labels are not reimplemented in Python.

Instead, Python sends raw records to the Rust `dataset-oracle` tool, which reuses:

- `position` for exact FEN parsing and state tracking
- `rules` for legal move generation, checks, and next-state application
- `action-space` for factorized move labels
- `encoder` for deterministic model-facing position tokens

This keeps dataset labels aligned with the runtime legality authority.

For policy-supervised experiments, Python may also derive `selected_move_uci` labels from external analysis engines during offline dataset generation. The current Phase-5 utility supports bounded PGN sampling with Stockfish 18 while still routing legality, action encoding, and next-state generation back through the exact Rust oracle.

The current larger reference corpus was produced by streaming PGNs on a separate Raspberry Pi host and labeling candidate positions there with `/usr/games/stockfish18` at a fixed `1500`-node budget per position. This keeps the label semantics reproducible across machines while moving the slowest offline work off the main development host.

## Example Schema

Each emitted example includes:

- raw position identity: `sample_id`, `source`, `fen`, `side_to_move`
- optional selected action supervision:
  - `selected_move_uci`
  - `selected_action_encoding`
  - `next_fen`
- legality labels:
  - `legal_moves`
  - `legal_action_encodings`
- deterministic position encoding:
  - `position_encoding.piece_tokens`
  - `position_encoding.square_tokens`
  - `position_encoding.rule_token`
- WDL target when available:
  - from supplied game `result`, mapped relative to side to move
  - or from immediate terminal surrogate on checkmate/stalemate positions
- tactical annotations:
  - `in_check`
  - `is_checkmate`
  - `is_stalemate`
  - `has_legal_en_passant`
  - `has_legal_castle`
  - `has_legal_promotion`
  - `is_low_material_endgame`
  - selected-move tactical flags when a selected move is present

## Splits and Reporting

Splits are assigned deterministically from `sample_id` and a user-provided seed.

The builder writes:

- `dataset.jsonl`
- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `summary.json`

When using PGN/Stockfish labeling with `--raw-output-dir`, the builder also writes:

- `train_raw.jsonl`
- `verify_raw.jsonl`
- `selection_summary.json`

The summary reports:

- split counts
- source counts
- available WDL counts
- annotation coverage for checks, mate, stalemate, castling, en passant, promotion, and low-material endgames
- legal-move-count and piece-count statistics

## Current Limits

- PGN ingestion is bounded, offline-only, and intended for dataset generation rather than runtime support
- no selfplay yet
- no policy probabilities yet
- no learned planner targets yet
- `is_low_material_endgame` is intentionally a conservative proxy based on total piece count, used only for Phase-4 reporting
