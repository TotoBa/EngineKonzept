"""Deterministic chess phase features used by LAPv2 artifact enrichment."""

from __future__ import annotations

from typing import Final

try:
    import chess
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None


_MINOR_PHASE_WEIGHT: Final[int] = 1
_ROOK_PHASE_WEIGHT: Final[int] = 2
_QUEEN_PHASE_WEIGHT: Final[int] = 4
_WHITE_MINOR_HOME_SQUARES: Final[tuple[int, ...]] = (
    1,  # b1
    2,  # c1
    5,  # f1
    6,  # g1
)
_BLACK_MINOR_HOME_SQUARES: Final[tuple[int, ...]] = (
    57,  # b8
    58,  # c8
    61,  # f8
    62,  # g8
)
_CENTRAL_PAWN_FILES: Final[frozenset[int]] = frozenset({3, 4})  # d/e files


def phase_score(board: "chess.Board") -> int:
    """Return the Stockfish-like phase score on the current board."""
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for phase feature evaluation.")
    counts = board.piece_map().values()
    score = 0
    for piece in counts:
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            score += _MINOR_PHASE_WEIGHT
        elif piece.piece_type == chess.ROOK:
            score += _ROOK_PHASE_WEIGHT
        elif piece.piece_type == chess.QUEEN:
            score += _QUEEN_PHASE_WEIGHT
    return score


def phase_index(board: "chess.Board") -> int:
    """Return one of four hard-coded phase buckets.

    Buckets:
    - `0`: opening
    - `1`: early middlegame
    - `2`: late middlegame
    - `3`: endgame
    """
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for phase feature evaluation.")
    score = phase_score(board)
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    minors_home = _minors_on_home_rank(board)
    blocked_pawns = _blocked_central_pawns(board)
    if score >= 22 and minors_home >= 5:
        return 0
    if score <= 8 or (queens == 0 and score <= 12):
        return 3
    if score <= 14 or blocked_pawns >= 3:
        return 2
    return 1


def _minors_on_home_rank(board: "chess.Board") -> int:
    count = 0
    for square in _WHITE_MINOR_HOME_SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color == chess.WHITE and piece.piece_type in (
            chess.KNIGHT,
            chess.BISHOP,
        ):
            count += 1
    for square in _BLACK_MINOR_HOME_SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color == chess.BLACK and piece.piece_type in (
            chess.KNIGHT,
            chess.BISHOP,
        ):
            count += 1
    return count


def _blocked_central_pawns(board: "chess.Board") -> int:
    blocked = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.PAWN or chess.square_file(square) not in _CENTRAL_PAWN_FILES:
            continue
        direction = 8 if piece.color == chess.WHITE else -8
        target_square = square + direction
        if 0 <= target_square < 64 and board.piece_at(target_square) is not None:
            blocked += 1
    return blocked
