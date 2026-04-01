# Encoding Architecture

Phase 3 introduces the first model-facing schema layers, while keeping the runtime fully symbolic.

## Action Space

`action-space` uses a factorized move representation:

- `from` head: 64 classes, one per origin square
- `to` head: 64 classes, one per destination square
- `promotion` head: 5 classes
  - `0 = none`
  - `1 = knight`
  - `2 = bishop`
  - `3 = rook`
  - `4 = queen`

This representation intentionally does not encode semantic move kinds such as capture, castling, en passant, or double-pawn-push directly.
Those exact semantics are recovered against the current symbolic position by matching the factorized tuple against the legal move list.

## Position Encoder

`encoder` emits three deterministic views of a position.

### Piece Tokens

Shape: `N x 3`, where `N` is the number of pieces on the board and `N <= 32`.

Column semantics:

1. `square_index` in `0..63`
2. `color_index` with `0 = white`, `1 = black`
3. `piece_kind_index` with `0 = pawn`, `1 = knight`, `2 = bishop`, `3 = rook`, `4 = queen`, `5 = king`

Ordering:

- tokens are sorted by ascending `square_index`

### Square Tokens

Shape: `64 x 2`.

Column semantics:

1. `square_index` in `0..63`
2. `occupant_code`
   - `0 = empty`
   - `1..6 = white pawn..king`
   - `7..12 = black pawn..king`

Ordering:

- row `i` always corresponds to square index `i`

### Rule Token

Shape: `6`.

Field semantics:

1. `side_to_move` with `0 = white`, `1 = black`
2. `castling_bits` with bit layout `KQkq`
3. `en_passant_square` in `0..63`, or `64` when absent
4. `halfmove_clock`
5. `fullmove_number`
6. `repetition_count`

## Determinism and Non-Invariances

The encoding is deterministic for identical symbolic positions.

The encoding intentionally changes when any of the following change:

- side to move
- castling rights
- en-passant square
- halfmove clock
- fullmove number
- exact piece squares
- repetition history count for the current key

No symmetry normalization, attack maps, or planner-side helper features are introduced in this phase.
