from __future__ import annotations

import json

import pytest

from train.action_space import flatten_action
from train.datasets import (
    SearchTeacherExample,
    SelfplayTeacherReviewExample,
    build_selfplay_mistake_priority,
    build_selfplay_teacher_review_example_from_teacher,
)
from train.datasets.planner_head import _select_root_candidate_indices


def test_build_selfplay_teacher_review_example_from_teacher_scales_mistake_priority() -> None:
    e2e4 = flatten_action([12, 28, 0])
    d2d4 = flatten_action([11, 27, 0])
    teacher = SearchTeacherExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "train",
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "feature_vector": [0.0, 1.0],
            "candidate_context_version": 2,
            "global_context_version": 1,
            "global_features": [0.0],
            "candidate_action_indices": [e2e4, d2d4],
            "candidate_features": [[0.0], [1.0]],
            "teacher_engine": "/usr/games/stockfish18",
            "teacher_nodes": None,
            "teacher_depth": 5,
            "teacher_movetime_ms": None,
            "teacher_multipv": 2,
            "teacher_coverage_ratio": 1.0,
            "teacher_root_value_cp": 80.0,
            "teacher_root_value_mate": None,
            "teacher_candidate_scores_cp": [80.0, 20.0],
            "teacher_policy": [0.8, 0.2],
            "teacher_top_k_action_indices": [e2e4, d2d4],
            "teacher_pv_uci": ["e2e4", "e7e5"],
        }
    )

    review = build_selfplay_teacher_review_example_from_teacher(
        teacher,
        agent_name="planner_a",
        game_id="game-1",
        ply_index=0,
        side_to_move="w",
        selected_action_index=d2d4,
        selected_move_uci="d2d4",
        game_result="0-1",
        outcome_pov="loss",
        termination_reason="checkmate",
        mistake_deadzone_cp=8.0,
        mistake_priority_scale_cp=64.0,
        max_mistake_priority=4.0,
    )

    assert review.selected_action_index == d2d4
    assert review.selected_score_cp == 20.0
    assert review.mistake_raw_cp == 60.0
    assert review.mistake_cp == 52.0
    assert review.mistake_priority == pytest.approx(52.0 / 64.0)
    assert not review.selected_is_teacher_top1


def test_selfplay_teacher_review_roundtrips_json() -> None:
    example = SelfplayTeacherReviewExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "train",
            "agent_name": "planner_a",
            "game_id": "game-1",
            "ply_index": 3,
            "side_to_move": "w",
            "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "feature_vector": [0.0, 1.0],
            "candidate_context_version": 2,
            "global_context_version": 1,
            "global_features": [0.0],
            "candidate_action_indices": [1, 2],
            "candidate_features": [[0.0], [1.0]],
            "teacher_engine": "/usr/games/stockfish18",
            "teacher_nodes": None,
            "teacher_depth": 5,
            "teacher_movetime_ms": None,
            "teacher_multipv": 2,
            "teacher_coverage_ratio": 1.0,
            "teacher_root_value_cp": 12.0,
            "teacher_root_value_mate": None,
            "teacher_candidate_scores_cp": [12.0, -5.0],
            "teacher_policy": [0.7, 0.3],
            "teacher_top_k_action_indices": [1, 2],
            "teacher_pv_uci": ["e1e2"],
            "selected_action_index": 2,
            "selected_move_uci": "e1e2",
            "selected_candidate_index": 1,
            "selected_score_cp": -5.0,
            "selected_is_teacher_top1": False,
            "game_result": "1-0",
            "outcome_pov": "win",
            "termination_reason": "checkmate",
            "mistake_deadzone_cp": 8.0,
            "mistake_raw_cp": 17.0,
            "mistake_cp": 9.0,
            "mistake_priority": 0.140625,
        }
    )

    roundtrip = SelfplayTeacherReviewExample.from_json(json.dumps(example.to_dict()))

    assert roundtrip == example


def test_select_root_candidate_indices_keeps_required_teacher_and_played_moves() -> None:
    selected = _select_root_candidate_indices(
        [0.9, 0.8, 0.7, 0.1],
        required_root_indices=(0, 3),
        root_top_k=1,
    )

    assert selected == [0, 3]


def test_build_selfplay_mistake_priority_clips_large_errors() -> None:
    assert build_selfplay_mistake_priority(32.0, scale_cp=64.0, max_priority=4.0) == 0.5
    assert build_selfplay_mistake_priority(999.0, scale_cp=64.0, max_priority=4.0) == 4.0
