"""Sparse HalfKA move-delta helpers for LAPv2 artifact enrichment."""

from __future__ import annotations

from typing import Final

from train.datasets.nnue_features import halfka_index, piece_type_color_index

try:
    import chess
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None


_PROMOTION_CLASS: Final[dict[int | None, int]] = {
    None: 0,
    2: 1,  # knight
    3: 2,  # bishop
    4: 3,  # rook
    5: 4,  # queen
}
_SPECIAL_CLASS: Final[dict[str, int]] = {
    "quiet": 0,
    "capture": 1,
    "castle": 2,
    "en_passant": 3,
    "double_pawn": 4,
}


def halfka_delta(
    board: "chess.Board",
    move: "chess.Move",
    perspective: str | bool,
) -> tuple[list[int], list[int]]:
    """Return `(leaving, entering)` HalfKA feature rows for one move.

    For own-king moves the runtime should rebuild from scratch, but deltas for
    opponent-side rook motion during castling are still emitted when valid.
    """
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for HalfKA delta evaluation.")
    color = _normalize_perspective(perspective)
    king_sq = board.king(color)
    if king_sq is None:
        raise ValueError("board is missing the king for the requested perspective")
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError("move source square is empty")

    leaving: list[int] = []
    entering: list[int] = []

    def add_leave(square: int, existing_piece: "chess.Piece") -> None:
        if existing_piece.piece_type == chess.KING:
            return
        leaving.append(
            halfka_index(
                king_sq,
                square,
                piece_type_color_index(existing_piece, perspective=color),
            )
        )

    def add_enter(square: int, *, piece_type: int, piece_color: "chess.Color") -> None:
        if piece_type == chess.KING:
            return
        entering.append(
            halfka_index(
                king_sq,
                square,
                piece_type_color_index(
                    chess.Piece(piece_type, piece_color),
                    perspective=color,
                ),
            )
        )

    if piece.piece_type == chess.KING:
        if board.is_castling(move):
            rook_from, rook_to = _castle_rook_squares(move)
            rook = board.piece_at(rook_from)
            if rook is None or rook.piece_type != chess.ROOK:
                raise ValueError("castling move is missing the rook on the expected square")
            add_leave(rook_from, rook)
            add_enter(rook_to, piece_type=chess.ROOK, piece_color=rook.color)
    else:
        add_leave(move.from_square, piece)
        add_enter(
            move.to_square,
            piece_type=move.promotion or piece.piece_type,
            piece_color=piece.color,
        )

    capture_square = _capture_square(board, move)
    if capture_square is not None:
        captured_piece = board.piece_at(capture_square)
        if captured_piece is None:
            raise ValueError("capture square does not contain a captured piece")
        add_leave(capture_square, captured_piece)

    return leaving, entering


def is_king_move(
    board: "chess.Board",
    move: "chess.Move",
    perspective: str | bool,
) -> bool:
    """Return whether the move changes the king square of the given side."""
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for HalfKA delta evaluation.")
    color = _normalize_perspective(perspective)
    piece = board.piece_at(move.from_square)
    return bool(
        piece is not None
        and piece.color == color
        and piece.piece_type == chess.KING
    )


def move_type_hash(board: "chess.Board", move: "chess.Move") -> int:
    """Return a deterministic 0..127 bucket for move-shape metadata."""
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for move-shape hashing.")
    piece = board.piece_at(move.from_square)
    if piece is None:
        raise ValueError("move source square is empty")
    if board.is_en_passant(move):
        special = _SPECIAL_CLASS["en_passant"]
    elif board.is_castling(move):
        special = _SPECIAL_CLASS["castle"]
    elif piece.piece_type == chess.PAWN and abs(move.to_square - move.from_square) == 16:
        special = _SPECIAL_CLASS["double_pawn"]
    elif board.is_capture(move):
        special = _SPECIAL_CLASS["capture"]
    else:
        special = _SPECIAL_CLASS["quiet"]
    capture = 1 if board.is_capture(move) else 0
    promotion = _PROMOTION_CLASS.get(move.promotion, 0)
    raw = (
        (piece.piece_type - 1) * 32
        + capture * 16
        + promotion * 3
        + special
    )
    return raw % 128


def _capture_square(board: "chess.Board", move: "chess.Move") -> int | None:
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for HalfKA delta evaluation.")
    if board.is_en_passant(move):
        return move.to_square - 8 if board.turn == chess.WHITE else move.to_square + 8
    if board.is_capture(move):
        return move.to_square
    return None


def _castle_rook_squares(move: "chess.Move") -> tuple[int, int]:
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for HalfKA delta evaluation.")
    if move.to_square == chess.G1:
        return chess.H1, chess.F1
    if move.to_square == chess.C1:
        return chess.A1, chess.D1
    if move.to_square == chess.G8:
        return chess.H8, chess.F8
    if move.to_square == chess.C8:
        return chess.A8, chess.D8
    raise ValueError("unsupported castling destination square")


def _normalize_perspective(perspective: str | bool) -> "chess.Color":
    if chess is None:  # pragma: no cover
        raise RuntimeError("python-chess is required for HalfKA delta evaluation.")
    if isinstance(perspective, bool):
        return perspective
    normalized = perspective.lower()
    if normalized == "w":
        return chess.WHITE
    if normalized == "b":
        return chess.BLACK
    raise ValueError("perspective must be 'w', 'b', chess.WHITE, or chess.BLACK")
