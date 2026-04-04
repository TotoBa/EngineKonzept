"""Versioned symbolic feature contracts shared across proposer, dynamics, and workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from train.datasets.schema import DatasetExample

try:
    import chess
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import chess as chess_module

SYMBOLIC_MAX_LEGAL_CANDIDATES = 256
DEFAULT_CANDIDATE_CONTEXT_VERSION = 1
DEFAULT_GLOBAL_CONTEXT_VERSION = 1
DEFAULT_STATE_CONTEXT_VERSION = 1

_STATE_CONTEXT_V1_SQUARE_FEATURE_ORDER = (
    "own_attackers_count",
    "opp_attackers_count",
    "reaches_own_king",
    "reaches_opp_king",
    "pin_axis_orthogonal",
    "pin_axis_diagonal",
    "xray_attackers_count",
)
_STATE_CONTEXT_V1_GLOBAL_FEATURE_ORDER = (
    "in_check",
    "own_king_attackers_count",
    "opp_king_attackers_count",
    "own_king_escape_squares",
    "opp_king_escape_squares",
    "material_phase",
    "single_legal_move",
    "legal_move_count_normalized",
    "has_legal_castle",
    "has_legal_en_passant",
    "has_legal_promotion",
)
STATE_CONTEXT_V1_FEATURE_ORDER = tuple(
    f"square_{square_index}_{feature_name}"
    for square_index in range(64)
    for feature_name in _STATE_CONTEXT_V1_SQUARE_FEATURE_ORDER
) + _STATE_CONTEXT_V1_GLOBAL_FEATURE_ORDER
STATE_CONTEXT_V1_EDGE_PIECE_TYPE_ENCODING = {
    "pawn": 1,
    "knight": 2,
    "bishop": 3,
    "rook": 4,
    "queen": 5,
    "king": 6,
}

_MATERIAL_PHASE_WEIGHTS = {
    2: 1,
    3: 1,
    4: 2,
    5: 4,
}
_MAX_MATERIAL_PHASE = 24.0
_RAY_DIRECTIONS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

_CANDIDATE_CONTEXT_V1_FEATURE_ORDER = (
    "is_capture",
    "is_promotion",
    "is_castle",
    "is_en_passant",
    "gives_check",
    "from_attacked_by_opponent",
    "to_attacked_by_opponent",
    "from_defended_by_self",
    "to_attacked_by_self",
    "moving_piece_pawn",
    "moving_piece_knight",
    "moving_piece_bishop",
    "moving_piece_rook",
    "moving_piece_queen",
    "moving_piece_king",
    "captured_piece_present",
    "captured_piece_pawn",
    "captured_piece_minor_or_major",
)

_CANDIDATE_CONTEXT_V2_FEATURE_ORDER = (
    "is_capture",
    "is_promotion",
    "is_castle",
    "is_en_passant",
    "gives_check",
    "pre_from_square_attacked_by_opponent",
    "pre_to_square_attacked_by_opponent",
    "pre_from_square_defended_by_self",
    "pre_to_square_attacked_by_self",
    "moving_piece_pawn",
    "moving_piece_knight",
    "moving_piece_bishop",
    "moving_piece_rook",
    "moving_piece_queen",
    "moving_piece_king",
    "captured_piece_present",
    "captured_piece_pawn",
    "captured_piece_knight",
    "captured_piece_bishop",
    "captured_piece_rook",
    "captured_piece_queen",
    "promotion_to_knight",
    "promotion_to_bishop",
    "promotion_to_rook",
    "promotion_to_queen",
    "castle_kingside",
    "castle_queenside",
    "from_file_normalized",
    "from_rank_normalized",
    "to_file_normalized",
    "to_rank_normalized",
    "delta_file_normalized",
    "delta_rank_normalized",
    "abs_delta_file_normalized",
    "abs_delta_rank_normalized",
)

_GLOBAL_CONTEXT_V1_FEATURE_ORDER = (
    "in_check",
    "has_legal_castle",
    "has_legal_en_passant",
    "has_legal_promotion",
    "is_low_material_endgame",
    "legal_move_count_normalized",
    "piece_count_normalized",
    "self_attack_square_ratio",
    "opponent_attack_square_ratio",
)
_TRANSITION_CONTEXT_V1_POST_MOVE_FEATURE_ORDER = (
    "opponent_in_check_after_move",
    "destination_attacked_after_move",
    "destination_defended_after_move",
    "halfmove_reset",
    "white_kingside_castling_cleared",
    "white_queenside_castling_cleared",
    "black_kingside_castling_cleared",
    "black_queenside_castling_cleared",
    "en_passant_created",
    "en_passant_cleared",
)
_SYMBOLIC_MOVE_DELTA_V1_FEATURE_ORDER = (
    "moving_piece_pawn",
    "moving_piece_knight",
    "moving_piece_bishop",
    "moving_piece_rook",
    "moving_piece_queen",
    "moving_piece_king",
    "captured_piece_present",
    "captured_piece_pawn",
    "captured_piece_knight",
    "captured_piece_bishop",
    "captured_piece_rook",
    "captured_piece_queen",
    "is_capture",
    "is_promotion",
    "is_castle",
    "is_en_passant",
    "gives_check",
    "promotion_to_knight",
    "promotion_to_bishop",
    "promotion_to_rook",
    "promotion_to_queen",
    "castle_kingside",
    "castle_queenside",
    "white_kingside_castling_cleared",
    "white_queenside_castling_cleared",
    "black_kingside_castling_cleared",
    "black_queenside_castling_cleared",
    "en_passant_created",
    "en_passant_cleared",
    "halfmove_reset",
)


@dataclass(frozen=True)
class FeatureContractSpec:
    """Structured description of one symbolic feature contract."""

    contract_name: str
    version: int
    feature_order: tuple[str, ...]

    @property
    def feature_dim(self) -> int:
        return len(self.feature_order)

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_name": self.contract_name,
            "version": self.version,
            "feature_dim": self.feature_dim,
            "feature_order": list(self.feature_order),
        }


@dataclass(frozen=True)
class StateContextV1:
    """Structured symbolic state context with a flat feature vector and sparse edges."""

    feature_values: list[float]
    edge_src_square: list[int]
    edge_dst_square: list[int]
    edge_piece_type: list[int]

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "contract_name": "StateContext",
            "version": 1,
            "feature_values": self.feature_values,
            "edge_src_square": self.edge_src_square,
            "edge_dst_square": self.edge_dst_square,
            "edge_piece_type": self.edge_piece_type,
        }


def candidate_context_spec(version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION) -> FeatureContractSpec:
    """Return the named candidate-context contract."""
    feature_order = _candidate_context_feature_order(version)
    return FeatureContractSpec(
        contract_name="CandidateContext",
        version=version,
        feature_order=feature_order,
    )


def global_context_spec(version: int = DEFAULT_GLOBAL_CONTEXT_VERSION) -> FeatureContractSpec:
    """Return the named global-context contract."""
    feature_order = _global_context_feature_order(version)
    return FeatureContractSpec(
        contract_name="GlobalContext",
        version=version,
        feature_order=feature_order,
    )


def state_context_feature_order(
    version: int = DEFAULT_STATE_CONTEXT_VERSION,
) -> tuple[str, ...]:
    """Return the flat feature order for one state-context version."""
    if version != 1:
        raise ValueError(f"unsupported StateContext version: {version}")
    return STATE_CONTEXT_V1_FEATURE_ORDER


def state_context_feature_dim(version: int = DEFAULT_STATE_CONTEXT_VERSION) -> int:
    """Return the flat feature width for one state-context version."""
    return len(state_context_feature_order(version))


def state_context_v1_feature_spec() -> dict[str, object]:
    """Describe the first symbolic state-context contract."""
    return {
        "contract_name": "StateContext",
        "version": 1,
        "feature_dim": state_context_feature_dim(1),
        "feature_order": list(STATE_CONTEXT_V1_FEATURE_ORDER),
        "square_count": 64,
        "square_feature_dim": len(_STATE_CONTEXT_V1_SQUARE_FEATURE_ORDER),
        "square_feature_order": list(_STATE_CONTEXT_V1_SQUARE_FEATURE_ORDER),
        "global_feature_dim": len(_STATE_CONTEXT_V1_GLOBAL_FEATURE_ORDER),
        "global_feature_order": list(_STATE_CONTEXT_V1_GLOBAL_FEATURE_ORDER),
        "edge_fields": [
            "edge_src_square",
            "edge_dst_square",
            "edge_piece_type",
        ],
        "edge_piece_type_encoding": dict(STATE_CONTEXT_V1_EDGE_PIECE_TYPE_ENCODING),
    }


def candidate_context_feature_order(
    version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION,
) -> tuple[str, ...]:
    """Return the feature order for one candidate-context version."""
    return _candidate_context_feature_order(version)


def candidate_context_feature_dim(version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION) -> int:
    """Return the feature width for one candidate-context version."""
    return len(_candidate_context_feature_order(version))


def project_candidate_context_to_v1(
    feature_values: list[float] | tuple[float, ...],
    *,
    version: int,
) -> list[float]:
    """Project a versioned candidate-context row onto the V1 dynamics contract."""
    values = [float(value) for value in feature_values]
    if version == 1:
        expected_width = candidate_context_feature_dim(1)
        if len(values) != expected_width:
            raise ValueError(
                f"CandidateContextV1 row must have width {expected_width}, got {len(values)}"
            )
        return values
    if version != 2:
        raise ValueError(f"unsupported CandidateContext version: {version}")
    expected_width = candidate_context_feature_dim(2)
    if len(values) != expected_width:
        raise ValueError(
            f"CandidateContextV2 row must have width {expected_width}, got {len(values)}"
        )
    captured_piece_minor_or_major = max(values[17:21]) if values[15] > 0.0 else 0.0
    return [
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
        values[7],
        values[8],
        values[9],
        values[10],
        values[11],
        values[12],
        values[13],
        values[14],
        values[15],
        values[16],
        float(captured_piece_minor_or_major),
    ]


def global_context_feature_order(
    version: int = DEFAULT_GLOBAL_CONTEXT_VERSION,
) -> tuple[str, ...]:
    """Return the feature order for one global-context version."""
    return _global_context_feature_order(version)


def global_context_feature_dim(version: int = DEFAULT_GLOBAL_CONTEXT_VERSION) -> int:
    """Return the feature width for one global-context version."""
    return len(_global_context_feature_order(version))


def transition_context_spec(version: int = 1) -> dict[str, object]:
    """Return the selected-action transition contract."""
    if version != 1:
        raise ValueError(f"unsupported TransitionContext version: {version}")
    candidate = candidate_context_spec(2)
    feature_order = transition_context_feature_order(version)
    return {
        "contract_name": "TransitionContext",
        "version": version,
        "candidate_context_version": candidate.version,
        "feature_dim": len(feature_order),
        "feature_order": list(feature_order),
        "post_move_feature_order": list(_TRANSITION_CONTEXT_V1_POST_MOVE_FEATURE_ORDER),
    }


def transition_context_feature_order(version: int = 1) -> tuple[str, ...]:
    """Return the feature order for one transition-context version."""
    if version != 1:
        raise ValueError(f"unsupported TransitionContext version: {version}")
    return _candidate_context_feature_order(2) + _TRANSITION_CONTEXT_V1_POST_MOVE_FEATURE_ORDER


def transition_context_feature_dim(version: int = 1) -> int:
    """Return the feature width for one transition-context version."""
    return len(transition_context_feature_order(version))


def symbolic_candidate_context_spec(
    *,
    candidate_context_version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION,
    global_context_version: int = DEFAULT_GLOBAL_CONTEXT_VERSION,
    max_legal_candidates: int = SYMBOLIC_MAX_LEGAL_CANDIDATES,
) -> dict[str, object]:
    """Return the symbolic proposer side-input contract."""
    candidate = candidate_context_spec(candidate_context_version)
    global_context = global_context_spec(global_context_version)
    return {
        "max_legal_candidates": max_legal_candidates,
        "candidate_context_version": candidate.version,
        "candidate_feature_dim": candidate.feature_dim,
        "candidate_feature_order": list(candidate.feature_order),
        "global_context_version": global_context.version,
        "global_feature_dim": global_context.feature_dim,
        "global_feature_order": list(global_context.feature_order),
    }


def symbolic_move_delta_spec(version: int = 1) -> FeatureContractSpec:
    """Return the symbolic move-delta contract used by hybrid dynamics arms."""
    feature_order = symbolic_move_delta_feature_order(version)
    return FeatureContractSpec(
        contract_name="SymbolicMoveDelta",
        version=version,
        feature_order=feature_order,
    )


def symbolic_move_delta_feature_order(version: int = 1) -> tuple[str, ...]:
    """Return the feature order for one symbolic move-delta version."""
    if version != 1:
        raise ValueError(f"unsupported SymbolicMoveDelta version: {version}")
    return _SYMBOLIC_MOVE_DELTA_V1_FEATURE_ORDER


def symbolic_move_delta_feature_dim(version: int = 1) -> int:
    """Return the feature width for one symbolic move-delta version."""
    return len(symbolic_move_delta_feature_order(version))


def build_state_context_v1(example: DatasetExample) -> StateContextV1:
    """Build the Python reference StateContextV1 from one dataset example."""
    if chess is None:  # pragma: no cover - exercised when chess is absent
        raise RuntimeError(
            "python-chess is required for StateContextV1. Install the 'train' extra."
        )

    board = chess.Board(example.fen)
    own_color = board.turn
    opp_color = not board.turn
    own_king_square = _required_king_square(board, own_color)
    opp_king_square = _required_king_square(board, opp_color)
    legal_moves = list(board.legal_moves)

    feature_values: list[float] = []
    for square in chess.SQUARES:
        feature_values.extend(
            _state_context_square_features(
                board,
                square,
                own_color=own_color,
                opp_color=opp_color,
                own_king_square=own_king_square,
                opp_king_square=opp_king_square,
            )
        )

    feature_values.extend(
        [
            float(board.is_check()),
            float(len(board.attackers(opp_color, own_king_square))),
            float(len(board.attackers(own_color, opp_king_square))),
            float(_king_escape_square_count(board, own_color)),
            float(_king_escape_square_count(board, opp_color)),
            _material_phase(board),
            float(len(legal_moves) == 1),
            float(len(legal_moves)) / 256.0,
            float(any(board.is_castling(move) for move in legal_moves)),
            float(any(board.is_en_passant(move) for move in legal_moves)),
            float(any(move.promotion is not None for move in legal_moves)),
        ]
    )
    edge_src_square, edge_dst_square, edge_piece_type = _build_reachability_graph(board)
    return StateContextV1(
        feature_values=feature_values,
        edge_src_square=edge_src_square,
        edge_dst_square=edge_dst_square,
        edge_piece_type=edge_piece_type,
    )


def _candidate_context_feature_order(version: int) -> tuple[str, ...]:
    if version == 1:
        return _CANDIDATE_CONTEXT_V1_FEATURE_ORDER
    if version == 2:
        return _CANDIDATE_CONTEXT_V2_FEATURE_ORDER
    raise ValueError(f"unsupported CandidateContext version: {version}")


def _global_context_feature_order(version: int) -> tuple[str, ...]:
    if version == 1:
        return _GLOBAL_CONTEXT_V1_FEATURE_ORDER
    raise ValueError(f"unsupported GlobalContext version: {version}")


def _state_context_square_features(
    board: "chess_module.Board",
    square: int,
    *,
    own_color: bool,
    opp_color: bool,
    own_king_square: int,
    opp_king_square: int,
) -> list[float]:
    pin_axis_orthogonal, pin_axis_diagonal = _pin_axis_flags(board, square)
    return [
        float(len(board.attackers(own_color, square))),
        float(len(board.attackers(opp_color, square))),
        float(_square_reaches_target_king(board, square, own_king_square)),
        float(_square_reaches_target_king(board, square, opp_king_square)),
        pin_axis_orthogonal,
        pin_axis_diagonal,
        float(_xray_attackers_count(board, square)),
    ]


def _square_reaches_target_king(
    board: "chess_module.Board",
    square: int,
    target_king_square: int,
) -> bool:
    piece = board.piece_at(square)
    if piece is None:
        return False
    return bool(square in board.attackers(piece.color, target_king_square))


def _pin_axis_flags(board: "chess_module.Board", square: int) -> tuple[float, float]:
    piece = board.piece_at(square)
    if piece is None or piece.piece_type == chess.KING:
        return (0.0, 0.0)
    if not board.is_pinned(piece.color, square):
        return (0.0, 0.0)
    king_square = _required_king_square(board, piece.color)
    file_delta = chess.square_file(square) - chess.square_file(king_square)
    rank_delta = chess.square_rank(square) - chess.square_rank(king_square)
    if file_delta == 0 or rank_delta == 0:
        return (1.0, 0.0)
    if abs(file_delta) == abs(rank_delta):
        return (0.0, 1.0)
    return (0.0, 0.0)


def _xray_attackers_count(board: "chess_module.Board", square: int) -> int:
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    count = 0
    for file_step, rank_step in _RAY_DIRECTIONS:
        current_file = file_index + file_step
        current_rank = rank_index + rank_step
        blocker_seen = False
        while 0 <= current_file < 8 and 0 <= current_rank < 8:
            current_square = chess.square(current_file, current_rank)
            piece = board.piece_at(current_square)
            if piece is not None:
                if not blocker_seen:
                    blocker_seen = True
                else:
                    if _piece_matches_ray(piece.piece_type, file_step, rank_step):
                        count += 1
                    break
            current_file += file_step
            current_rank += rank_step
    return count


def _piece_matches_ray(piece_type: int, file_step: int, rank_step: int) -> bool:
    orthogonal = file_step == 0 or rank_step == 0
    if orthogonal:
        return piece_type in {chess.ROOK, chess.QUEEN}
    return piece_type in {chess.BISHOP, chess.QUEEN}


def _material_phase(board: "chess_module.Board") -> float:
    phase_units = 0.0
    for piece_type, weight in _MATERIAL_PHASE_WEIGHTS.items():
        phase_units += weight * len(board.pieces(piece_type, chess.WHITE))
        phase_units += weight * len(board.pieces(piece_type, chess.BLACK))
    return phase_units / _MAX_MATERIAL_PHASE


def _king_escape_square_count(board: "chess_module.Board", color: bool) -> int:
    king_square = _required_king_square(board, color)
    board_for_color = _board_for_color(board, color)
    return sum(1 for move in board_for_color.legal_moves if move.from_square == king_square)


def _build_reachability_graph(board: "chess_module.Board") -> tuple[list[int], list[int], list[int]]:
    boards_by_color = {
        chess.WHITE: _board_for_color(board, chess.WHITE),
        chess.BLACK: _board_for_color(board, chess.BLACK),
    }
    edge_src_square: list[int] = []
    edge_dst_square: list[int] = []
    edge_piece_type: list[int] = []
    for square, piece in sorted(board.piece_map().items()):
        reachable_squares = _piece_reachability_squares(boards_by_color[piece.color], square)
        for destination_square in reachable_squares:
            edge_src_square.append(square)
            edge_dst_square.append(destination_square)
            edge_piece_type.append(piece.piece_type)
    return edge_src_square, edge_dst_square, edge_piece_type


def _piece_reachability_squares(board: "chess_module.Board", square: int) -> list[int]:
    seen_destinations: set[int] = set()
    destinations: list[int] = []
    for move in board.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[square]):
        if move.from_square != square or move.to_square in seen_destinations:
            continue
        seen_destinations.add(move.to_square)
        destinations.append(move.to_square)
    destinations.sort()
    return destinations


def _board_for_color(board: "chess_module.Board", color: bool) -> "chess_module.Board":
    if board.turn == color:
        return board
    copied = board.copy(stack=False)
    copied.turn = color
    return copied


def _required_king_square(board: "chess_module.Board", color: bool) -> int:
    king_square = board.king(color)
    if king_square is None:
        color_name = "white" if color == chess.WHITE else "black"
        raise ValueError(f"board is missing the {color_name} king")
    return king_square
