# Rules Kernel

Phase 1 establishes the exact symbolic chess core that later phases will treat as the authority for legality.

## Scope

- board state and side-to-move tracking
- castling rights, en-passant target, halfmove clock, fullmove number
- FEN parse and serialize
- pseudo-legal and legal move generation
- attack detection and in-check queries
- exact move application
- repetition-history bookkeeping
- perft regression support

## Boundaries

This layer exists for correctness, labels, regression tests, and final legality verification.
It is not allowed to grow into a classical search engine.

## Current crate split

- `core-types`: colors, pieces, squares, files, ranks, and exact move types
- `position`: immutable-ish board state with explicit rule state and FEN handling
- `rules`: attack queries, legal move generation, move application, and perft

