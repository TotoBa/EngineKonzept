"""Tests for the offline alpha-beta teacher workflow."""

from __future__ import annotations

import json

import pytest

from train.action_space import flatten_action
from train.datasets import (
    DatasetExample,
    build_search_teacher_example_from_analysis,
    build_symbolic_proposer_example,
)
from train.datasets.search_teacher import SearchTeacherExample


def test_build_search_teacher_example_from_analysis_aligns_scores_to_candidates() -> None:
    chess = pytest.importorskip("chess")

    example = DatasetExample.from_dict(_dataset_example_dict())
    symbolic_example = build_symbolic_proposer_example(
        example,
        candidate_context_version=2,
        global_context_version=1,
    )
    analysis_list = [
        {
            "score": chess.engine.PovScore(chess.engine.Cp(80), chess.WHITE),
            "pv": [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")],
        },
        {
            "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE),
            "pv": [chess.Move.from_uci("d2d4"), chess.Move.from_uci("d7d5")],
        },
    ]

    built = build_search_teacher_example_from_analysis(
        example,
        symbolic_example=symbolic_example,
        analysis_list=analysis_list,
        teacher_engine="/usr/games/stockfish18",
        nodes=256,
        depth=None,
        movetime_ms=None,
        effective_multipv=2,
        policy_temperature_cp=100.0,
    )

    e2e4 = flatten_action([12, 28, 0])
    d2d4 = flatten_action([11, 27, 0])
    e2e4_index = built.candidate_action_indices.index(e2e4)
    d2d4_index = built.candidate_action_indices.index(d2d4)

    assert built.candidate_context_version == 2
    assert built.global_context_version == 1
    assert built.teacher_top_k_action_indices == [e2e4, d2d4]
    assert built.teacher_candidate_scores_cp[e2e4_index] == 80.0
    assert built.teacher_candidate_scores_cp[d2d4_index] == 20.0
    assert built.teacher_root_value_cp == 80.0
    assert built.teacher_root_value_mate is None
    assert built.teacher_pv_uci == ["e2e4", "e7e5"]
    assert built.teacher_coverage_ratio == 2 / len(built.candidate_action_indices)
    assert abs(sum(built.teacher_policy) - 1.0) < 1e-6
    assert built.teacher_policy[e2e4_index] > built.teacher_policy[d2d4_index]


def test_search_teacher_example_roundtrips_json() -> None:
    example = SearchTeacherExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "train",
            "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "feature_vector": [0.0, 1.0],
            "candidate_context_version": 2,
            "global_context_version": 1,
            "global_features": [0.0],
            "candidate_action_indices": [1, 2],
            "candidate_features": [[0.0], [1.0]],
            "teacher_engine": "/usr/games/stockfish18",
            "teacher_nodes": 128,
            "teacher_depth": None,
            "teacher_movetime_ms": None,
            "teacher_multipv": 2,
            "teacher_coverage_ratio": 1.0,
            "teacher_root_value_cp": 15.0,
            "teacher_root_value_mate": None,
            "teacher_candidate_scores_cp": [15.0, 5.0],
            "teacher_policy": [0.6, 0.4],
            "teacher_top_k_action_indices": [1, 2],
            "teacher_pv_uci": ["e1e2"],
        }
    )

    roundtrip = SearchTeacherExample.from_json(json.dumps(example.to_dict()))

    assert roundtrip == example


def _dataset_example_dict() -> dict[str, object]:
    return {
        "sample_id": "sample-1",
        "split": "train",
        "source": "fixture",
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "side_to_move": "w",
        "selected_move_uci": "e2e4",
        "selected_action_encoding": [12, 28, 0],
        "next_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "legal_moves": [
            "b1a3",
            "b1c3",
            "g1f3",
            "g1h3",
            "a2a3",
            "a2a4",
            "b2b3",
            "b2b4",
            "c2c3",
            "c2c4",
            "d2d3",
            "d2d4",
            "e2e3",
            "e2e4",
            "f2f3",
            "f2f4",
            "g2g3",
            "g2g4",
            "h2h3",
            "h2h4",
        ],
        "legal_action_encodings": [
            [1, 16, 0],
            [1, 18, 0],
            [6, 21, 0],
            [6, 23, 0],
            [8, 16, 0],
            [8, 24, 0],
            [9, 17, 0],
            [9, 25, 0],
            [10, 18, 0],
            [10, 26, 0],
            [11, 19, 0],
            [11, 27, 0],
            [12, 20, 0],
            [12, 28, 0],
            [13, 21, 0],
            [13, 29, 0],
            [14, 22, 0],
            [14, 30, 0],
            [15, 23, 0],
            [15, 31, 0],
        ],
        "position_encoding": {
            "piece_tokens": [[0, 0, 3]],
            "square_tokens": [[square, 0] for square in range(64)],
            "rule_token": [0, 15, -1, 0, 1, 0],
        },
        "wdl_target": {"win": 1, "draw": 0, "loss": 0},
        "annotations": {
            "in_check": False,
            "is_checkmate": False,
            "is_stalemate": False,
            "has_legal_en_passant": False,
            "has_legal_castle": False,
            "has_legal_promotion": False,
            "is_low_material_endgame": False,
            "legal_move_count": 20,
            "piece_count": 32,
            "selected_move_is_capture": False,
            "selected_move_is_promotion": False,
            "selected_move_is_castle": False,
            "selected_move_is_en_passant": False,
            "selected_move_gives_check": False,
        },
        "result": "1-0",
        "metadata": {},
    }
