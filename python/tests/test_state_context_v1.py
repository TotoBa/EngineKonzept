from __future__ import annotations

import chess

from train.datasets.contracts import (
    STATE_CONTEXT_V1_FEATURE_ORDER,
    build_state_context_v1,
    state_context_feature_dim,
    state_context_v1_feature_spec,
)
from train.datasets.schema import DatasetExample, PositionEncoding, TacticalAnnotations


def _make_example(fen: str) -> DatasetExample:
    board = chess.Board(fen)
    legal_moves = [move.uci() for move in board.legal_moves]
    return DatasetExample(
        sample_id="state_context_test",
        split="test",
        source="state_context_test",
        fen=fen,
        side_to_move="w" if board.turn else "b",
        selected_move_uci=None,
        selected_action_encoding=None,
        next_fen=None,
        legal_moves=legal_moves,
        legal_action_encodings=[[0, 0, index] for index, _move in enumerate(legal_moves)],
        position_encoding=PositionEncoding(
            piece_tokens=[],
            square_tokens=[[square_index, 0] for square_index in range(64)],
            rule_token=[0, 0, -1, 0, 1, 0],
        ),
        wdl_target=None,
        annotations=TacticalAnnotations(
            in_check=board.is_check(),
            is_checkmate=board.is_checkmate(),
            is_stalemate=board.is_stalemate(),
            has_legal_en_passant=any(board.is_en_passant(move) for move in board.legal_moves),
            has_legal_castle=any(board.is_castling(move) for move in board.legal_moves),
            has_legal_promotion=any(move.promotion is not None for move in board.legal_moves),
            is_low_material_endgame=len(board.piece_map()) <= 6,
            legal_move_count=len(legal_moves),
            piece_count=len(board.piece_map()),
            selected_move_is_capture=None,
            selected_move_is_promotion=None,
            selected_move_is_castle=None,
            selected_move_is_en_passant=None,
            selected_move_gives_check=None,
        ),
        result=None,
        metadata={},
    )


def _global_feature_map(feature_values: list[float]) -> dict[str, float]:
    spec = state_context_v1_feature_spec()
    global_feature_order = [str(name) for name in spec["global_feature_order"]]
    global_values = feature_values[-len(global_feature_order) :]
    return dict(zip(global_feature_order, global_values, strict=True))


def test_state_context_v1_feature_order_has_no_duplicates() -> None:
    spec = state_context_v1_feature_spec()
    feature_order = [str(name) for name in spec["feature_order"]]
    assert len(feature_order) == len(set(feature_order))
    assert tuple(feature_order) == STATE_CONTEXT_V1_FEATURE_ORDER
    assert state_context_feature_dim(1) == 64 * 7 + 11


def test_state_context_v1_startpos_is_deterministic() -> None:
    example = _make_example(chess.STARTING_FEN)
    first = build_state_context_v1(example)
    second = build_state_context_v1(example)

    assert first == second
    assert len(first.feature_values) == state_context_feature_dim(1)
    assert len(first.edge_src_square) == len(first.edge_dst_square) == len(first.edge_piece_type)

    global_features = _global_feature_map(first.feature_values)
    assert global_features == {
        "in_check": 0.0,
        "own_king_attackers_count": 0.0,
        "opp_king_attackers_count": 0.0,
        "own_king_escape_squares": 0.0,
        "opp_king_escape_squares": 0.0,
        "material_phase": 1.0,
        "single_legal_move": 0.0,
        "legal_move_count_normalized": 20.0 / 256.0,
        "has_legal_castle": 0.0,
        "has_legal_en_passant": 0.0,
        "has_legal_promotion": 0.0,
    }


def test_state_context_v1_single_legal_move_triggers_on_curated_forcing_position() -> None:
    example = _make_example("r5r1/3bk3/1n2p3/p1pP3P/P7/2P2N2/nP1N1q1P/4RK2 w - - 0 43")
    context = build_state_context_v1(example)

    global_features = _global_feature_map(context.feature_values)
    assert example.legal_moves == ["f1f2"]
    assert global_features["single_legal_move"] == 1.0
    assert global_features["legal_move_count_normalized"] == 1.0 / 256.0


def test_state_context_v1_reachability_graph_contains_expected_startpos_edges() -> None:
    example = _make_example(chess.STARTING_FEN)
    context = build_state_context_v1(example)
    edge_set = set(
        zip(
            context.edge_src_square,
            context.edge_dst_square,
            context.edge_piece_type,
            strict=True,
        )
    )

    assert (chess.B1, chess.A3, chess.KNIGHT) in edge_set
    assert (chess.B1, chess.C3, chess.KNIGHT) in edge_set
    assert (chess.E2, chess.E3, chess.PAWN) in edge_set
    assert (chess.E2, chess.E4, chess.PAWN) in edge_set
    assert (chess.G8, chess.F6, chess.KNIGHT) in edge_set
    assert (chess.G8, chess.H6, chess.KNIGHT) in edge_set
    assert not any(src_square == chess.C1 for src_square in context.edge_src_square)
