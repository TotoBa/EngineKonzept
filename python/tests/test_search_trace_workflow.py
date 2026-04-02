"""Tests for the offline search-trace workflow."""

from __future__ import annotations

import json

import pytest

from train.action_space import flatten_action
from train.datasets import (
    DatasetExample,
    build_search_trace_example_from_analysis,
    build_symbolic_proposer_example,
)
from train.datasets.search_traces import SearchTraceExample


def test_build_search_trace_example_from_analysis_captures_pv_and_reply() -> None:
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
            "pv": [
                chess.Move.from_uci("e2e4"),
                chess.Move.from_uci("e7e5"),
                chess.Move.from_uci("g1f3"),
            ],
        },
        {
            "score": chess.engine.PovScore(chess.engine.Cp(20), chess.WHITE),
            "pv": [chess.Move.from_uci("d2d4"), chess.Move.from_uci("d7d5")],
        },
    ]

    built = build_search_trace_example_from_analysis(
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
    e7e5 = flatten_action([52, 36, 0])
    g1f3 = flatten_action([6, 21, 0])

    assert built.candidate_context_version == 2
    assert built.teacher_top_k_action_indices == [e2e4, d2d4]
    assert built.principal_variation_uci == ["e2e4", "e7e5", "g1f3"]
    assert built.principal_variation_action_indices == [e2e4, e7e5, g1f3]
    assert built.best_reply_uci == "e7e5"
    assert built.best_reply_action_index == e7e5
    assert built.pv_length == 3
    assert built.top1_minus_top2_cp == 60.0


def test_search_trace_example_roundtrips_json() -> None:
    example = SearchTraceExample.from_dict(
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
            "teacher_top_k_action_indices": [1, 2],
            "principal_variation_uci": ["e1e2", "e8e7"],
            "principal_variation_action_indices": [1, 2],
            "best_reply_uci": "e8e7",
            "best_reply_action_index": 2,
            "pv_length": 2,
            "top1_minus_top2_cp": 10.0,
        }
    )

    roundtrip = SearchTraceExample.from_json(json.dumps(example.to_dict()))

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
