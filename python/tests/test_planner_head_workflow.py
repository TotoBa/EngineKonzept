"""Tests for planner-head curriculum shaping and external benchmark pressure."""

from __future__ import annotations

from train.datasets.planner_head import (
    _planner_curriculum_focus_from_dataset_example,
)
from train.datasets.schema import DatasetExample
from train.datasets.search_curriculum import SearchCurriculumExample


def test_external_benchmark_curriculum_focus_boosts_stockfish_loss_feedback() -> None:
    dataset_example = DatasetExample.from_dict(
        {
            "sample_id": "arena-stockfish-loss",
            "split": "train",
            "source": "stockfish-unique-pgn",
            "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "side_to_move": "w",
            "selected_move_uci": "e1d1",
            "selected_action_encoding": [4, 3, 0],
            "next_fen": "4k3/8/8/8/8/8/8/3K4 b - - 1 1",
            "legal_moves": ["e1d1", "e1f1"],
            "legal_action_encodings": [[4, 3, 0], [4, 5, 0]],
            "position_encoding": {
                "piece_tokens": [[4, 0, 5], [60, 1, 5]],
                "square_tokens": [[square_index, 0] for square_index in range(64)],
                "rule_token": [0, 0, -1, 0, 1, 0],
            },
            "wdl_target": {"win": 0, "draw": 0, "loss": 1},
            "annotations": {
                "in_check": False,
                "is_checkmate": False,
                "is_stalemate": False,
                "has_legal_en_passant": False,
                "has_legal_castle": False,
                "has_legal_promotion": False,
                "is_low_material_endgame": True,
                "legal_move_count": 2,
                "piece_count": 2,
                "selected_move_is_capture": False,
                "selected_move_is_promotion": False,
                "selected_move_is_castle": False,
                "selected_move_is_en_passant": False,
                "selected_move_gives_check": False,
            },
            "result": "0-1",
            "metadata": {
                "event": "lapv2_frontier_interaction_random_full_labeled_arena_feedback",
                "white": "lapv2_stage2_native_all_sources_v1_inner0",
                "black": "stockfish18_skill_00",
            },
        }
    )
    curriculum_example = SearchCurriculumExample.from_dict(
        {
            "sample_id": dataset_example.sample_id,
            "split": dataset_example.split,
            "fen": dataset_example.fen,
            "teacher_top1_action_index": 1,
            "best_reply_action_index": 2,
            "pv_length": 2,
            "bucket_labels": ["forced_teacher"],
            "curriculum_priority": 1.25,
            "teacher_top1_minus_top2_cp": 80.0,
            "proposer_rank_of_teacher_top1": 2,
            "teacher_rank_of_proposer_top1": 3,
            "teacher_top1_advantage_cp": 50.0,
            "policy_l1_distance": 0.5,
            "top1_disagrees": True,
        }
    )

    labels, priority = _planner_curriculum_focus_from_dataset_example(
        dataset_example=dataset_example,
        curriculum_example=curriculum_example,
    )

    assert labels[0] == "forced_teacher"
    assert "external_arena_feedback" in labels
    assert "external_benchmark:stockfish18_skill_00" in labels
    assert "external_benchmark_nonwin" in labels
    assert "external_benchmark_loss_recovery" in labels
    assert priority > curriculum_example.curriculum_priority


def test_external_benchmark_curriculum_focus_leaves_non_external_rows_unchanged() -> None:
    dataset_example = DatasetExample.from_dict(
        {
            "sample_id": "selfplay-row",
            "split": "train",
            "source": "stockfish-unique-pgn",
            "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "side_to_move": "w",
            "selected_move_uci": "e1d1",
            "selected_action_encoding": [4, 3, 0],
            "next_fen": "4k3/8/8/8/8/8/8/3K4 b - - 1 1",
            "legal_moves": ["e1d1", "e1f1"],
            "legal_action_encodings": [[4, 3, 0], [4, 5, 0]],
            "position_encoding": {
                "piece_tokens": [[4, 0, 5], [60, 1, 5]],
                "square_tokens": [[square_index, 0] for square_index in range(64)],
                "rule_token": [0, 0, -1, 0, 1, 0],
            },
            "wdl_target": {"win": 1, "draw": 0, "loss": 0},
            "annotations": {
                "in_check": False,
                "is_checkmate": False,
                "is_stalemate": False,
                "has_legal_en_passant": False,
                "has_legal_castle": False,
                "has_legal_promotion": False,
                "is_low_material_endgame": True,
                "legal_move_count": 2,
                "piece_count": 2,
                "selected_move_is_capture": False,
                "selected_move_is_promotion": False,
                "selected_move_is_castle": False,
                "selected_move_is_en_passant": False,
                "selected_move_gives_check": False,
            },
            "result": "1-0",
            "metadata": {
                "event": "lapv2_frontier_interaction_random_full_labeled_selfplay_feedback",
                "white": "lapv2_stage2_native_all_sources_v1_inner0",
                "black": "lapv2_stage2_native_all_sources_v1_inner1",
            },
        }
    )

    labels, priority = _planner_curriculum_focus_from_dataset_example(
        dataset_example=dataset_example,
        curriculum_example=None,
    )

    assert labels == []
    assert priority == 0.0
