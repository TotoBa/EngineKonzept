"""HalfKA-style sparse feature helpers for LAPv2 artifacts."""

from __future__ import annotations

from typing import Final

try:
    import chess
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None


FEATURES_PER_KING: Final[int] = 64 * 12
TOTAL_FEATURES: Final[int] = 64 * FEATURES_PER_KING


def halfka_index(king_sq: int, piece_sq: int, piece_type_color: int) -> int:
    """Encode one HalfKA feature index."""
    if not 0 <= king_sq < 64:
        raise ValueError("king_sq must be in 0..63")
    if not 0 <= piece_sq < 64:
        raise ValueError("piece_sq must be in 0..63")
    if not 0 <= piece_type_color < 12:
        raise ValueError("piece_type_color must be in 0..11")
    return king_sq * FEATURES_PER_KING + piece_sq * 12 + piece_type_color


def piece_type_color_index(piece: "chess.Piece", *, perspective: "chess.Color") -> int:
    """Return the 0..11 piece-type-with-relative-color encoding."""
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for HalfKA feature encoding.")
    if piece.piece_type not in (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ):
        raise ValueError(f"unsupported piece type: {piece.piece_type}")
    base = piece.piece_type - 1
    if piece.color != perspective:
        base += 6
    return base


def halfka_active_indices(
    board: "chess.Board",
    perspective: str | bool,
) -> list[int]:
    """Return active HalfKA indices from one king perspective.

    Both kings are excluded as sparse features so the start position yields
    30 active features per perspective.
    """
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for HalfKA feature encoding.")
    color = _normalize_perspective(perspective)
    king_sq = board.king(color)
    if king_sq is None:
        raise ValueError("board is missing the king for the requested perspective")
    indices: list[int] = []
    for square, piece in sorted(board.piece_map().items()):
        if piece.piece_type == chess.KING:
            continue
        indices.append(
            halfka_index(
                king_sq,
                square,
                piece_type_color_index(piece, perspective=color),
            )
        )
    return indices


def _normalize_perspective(perspective: str | bool) -> "chess.Color":
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for HalfKA feature encoding.")
    if isinstance(perspective, bool):
        return perspective
    normalized = perspective.lower()
    if normalized == "w":
        return chess.WHITE
    if normalized == "b":
        return chess.BLACK
    raise ValueError("perspective must be 'w', 'b', chess.WHITE, or chess.BLACK")
