"""Tests for the first OpponentHeadV1 dataset workflow."""

from __future__ import annotations

import json

import pytest

from train.action_space import flatten_action
from train.datasets import DatasetExample
from train.datasets.opponent_head import (
    OpponentHeadExample,
    build_opponent_head_examples,
)
from train.datasets.search_curriculum import SearchCurriculumExample
from train.datasets.search_traces import SearchTraceExample


def test_build_opponent_head_examples_derives_reply_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("chess")

    example = DatasetExample.from_dict(_root_dataset_example_dict())
    trace_example = SearchTraceExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "train",
            "fen": example.fen,
            "feature_vector": [0.0, 1.0],
            "candidate_context_version": 2,
            "global_context_version": 1,
            "global_features": [0.1],
            "candidate_action_indices": [flatten_action([12, 28, 0]), flatten_action([11, 27, 0])],
            "candidate_features": [[0.0] * 35, [0.0] * 35],
            "teacher_engine": "/usr/games/stockfish18",
            "teacher_nodes": 64,
            "teacher_depth": None,
            "teacher_movetime_ms": None,
            "teacher_multipv": 2,
            "teacher_coverage_ratio": 1.0,
            "teacher_root_value_cp": 80.0,
            "teacher_root_value_mate": None,
            "teacher_candidate_scores_cp": [80.0, 20.0],
            "teacher_top_k_action_indices": [flatten_action([12, 28, 0]), flatten_action([11, 27, 0])],
            "principal_variation_uci": ["e2e4", "e7e5"],
            "principal_variation_action_indices": [
                flatten_action([12, 28, 0]),
                flatten_action([52, 36, 0]),
            ],
            "best_reply_uci": "e7e5",
            "best_reply_action_index": flatten_action([52, 36, 0]),
            "pv_length": 2,
            "top1_minus_top2_cp": 80.0,
        }
    )
    curriculum_example = SearchCurriculumExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "train",
            "fen": example.fen,
            "teacher_top1_action_index": flatten_action([12, 28, 0]),
            "best_reply_action_index": flatten_action([52, 36, 0]),
            "pv_length": 2,
            "bucket_labels": ["forced_teacher", "reply_supervised"],
            "curriculum_priority": 1.5,
            "teacher_top1_minus_top2_cp": 80.0,
            "proposer_rank_of_teacher_top1": 1,
            "teacher_rank_of_proposer_top1": 1,
            "teacher_top1_advantage_cp": 0.0,
            "policy_l1_distance": 0.0,
            "top1_disagrees": False,
        }
    )

    root_payload = {
        "side_to_move": "w",
        "selected_move_uci": "e2e4",
        "selected_action_encoding": [12, 28, 0],
        "next_fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        "legal_moves": example.legal_moves,
        "legal_action_encodings": example.legal_action_encodings,
        "position_encoding": example.position_encoding.to_dict(),
        "annotations": example.annotations.to_dict(),
    }
    successor_payload = {
        "side_to_move": "b",
        "selected_move_uci": "e7e5",
        "selected_action_encoding": [52, 36, 0],
        "next_fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "legal_moves": ["e7e5", "c7c5"],
        "legal_action_encodings": [[52, 36, 0], [50, 34, 0]],
        "position_encoding": {
            "piece_tokens": [[0, 0, 3]],
            "square_tokens": [[square, 0] for square in range(64)],
            "rule_token": [1, 15, -1, 0, 1, 0],
        },
        "annotations": {
            "in_check": False,
            "is_checkmate": False,
            "is_stalemate": False,
            "has_legal_en_passant": False,
            "has_legal_castle": False,
            "has_legal_promotion": False,
            "is_low_material_endgame": False,
            "legal_move_count": 2,
            "piece_count": 32,
            "selected_move_is_capture": False,
            "selected_move_is_promotion": False,
            "selected_move_is_castle": False,
            "selected_move_is_en_passant": False,
            "selected_move_gives_check": False,
        },
    }

    def fake_label_records_with_oracle(records, **_kwargs):
        if records and records[0].sample_id.endswith(":opponent_root"):
            return [root_payload]
        if records and records[0].sample_id.endswith(":opponent_successor"):
            return [successor_payload]
        raise AssertionError("unexpected oracle request")

    monkeypatch.setattr(
        "train.datasets.opponent_head.label_records_with_oracle",
        fake_label_records_with_oracle,
    )

    built = build_opponent_head_examples(
        [example],
        [trace_example],
        [curriculum_example],
    )

    assert len(built) == 1
    opponent_example = built[0]
    assert opponent_example.chosen_move_uci == "e2e4"
    assert opponent_example.chosen_action_index == flatten_action([12, 28, 0])
    assert opponent_example.teacher_reply_uci == "e7e5"
    assert opponent_example.teacher_reply_action_index == flatten_action([52, 36, 0])
    assert opponent_example.reply_candidate_action_indices == sorted(
        [
            flatten_action([52, 36, 0]),
            flatten_action([50, 34, 0]),
        ]
    )
    reply_index = opponent_example.reply_candidate_action_indices.index(
        flatten_action([52, 36, 0])
    )
    assert opponent_example.teacher_reply_policy[reply_index] == 1.0
    assert sum(opponent_example.teacher_reply_policy) == 1.0
    assert opponent_example.curriculum_bucket_labels == ["forced_teacher", "reply_supervised"]
    assert opponent_example.curriculum_priority == 1.5
    assert opponent_example.uncertainty_target == 0.0
    assert opponent_example.pressure_target == 0.0


def test_opponent_head_example_roundtrips_json() -> None:
    example = OpponentHeadExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "validation",
            "root_fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "root_feature_vector": [0.0] * 230,
            "curriculum_bucket_labels": ["forced_teacher"],
            "curriculum_priority": 1.0,
            "chosen_move_uci": "e2e4",
            "chosen_action_index": 1,
            "transition_context_version": 1,
            "transition_features": [0.0] * 45,
            "next_fen": "4k3/8/8/8/4P3/8/8/4K3 b - - 0 1",
            "next_feature_vector": [0.0] * 230,
            "reply_candidate_context_version": 2,
            "reply_global_context_version": 1,
            "reply_global_features": [0.0] * 9,
            "reply_candidate_action_indices": [2, 3],
            "reply_candidate_features": [[0.0] * 35, [1.0] * 35],
            "teacher_reply_uci": "e7e5",
            "teacher_reply_action_index": 2,
            "teacher_reply_policy": [1.0, 0.0],
            "teacher_root_value_cp": 25.0,
            "teacher_root_value_mate": None,
            "teacher_top1_minus_top2_cp": 40.0,
            "pressure_target": 0.5,
            "uncertainty_target": 0.25,
            "reply_is_capture": False,
            "reply_is_promotion": False,
            "reply_is_castle": False,
            "reply_is_en_passant": False,
            "reply_gives_check": False,
        }
    )

    roundtrip = OpponentHeadExample.from_json(json.dumps(example.to_dict()))

    assert roundtrip == example


def _root_dataset_example_dict() -> dict[str, object]:
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
