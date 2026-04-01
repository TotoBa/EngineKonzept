# Phase 1

## Goal

Build the exact symbolic chess state and rules kernel from scratch.

## Deliverables in this repository state

- canonical chess primitives in `core-types`
- exact board state and FEN handling in `position`
- attack detection, legal move generation, move application, and perft in `rules`
- explicit regression tests for castling, en passant, promotions, checks, and repetition bookkeeping

## Non-goals still preserved

- no UCI shell
- no evaluation
- no search
- no model integration

