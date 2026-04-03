"""Tests for offline search curriculum workflows."""

from __future__ import annotations

import json

from train.datasets.search_curriculum import (
    SearchCurriculumExample,
    build_search_curriculum_examples,
)
from train.datasets.search_disagreements import SearchDisagreementExample
from train.datasets.search_traces import SearchTraceExample


def test_build_search_curriculum_examples_assigns_expected_buckets() -> None:
    trace_example = SearchTraceExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "validation",
            "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "feature_vector": [0.0, 1.0],
            "candidate_context_version": 2,
            "global_context_version": 1,
            "global_features": [0.1],
            "candidate_action_indices": [11, 22],
            "candidate_features": [
                [1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 30,
                [0.0] * 35,
            ],
            "teacher_engine": "/usr/games/stockfish18",
            "teacher_nodes": 128,
            "teacher_depth": None,
            "teacher_movetime_ms": None,
            "teacher_multipv": 2,
            "teacher_coverage_ratio": 1.0,
            "teacher_root_value_cp": 90.0,
            "teacher_root_value_mate": None,
            "teacher_candidate_scores_cp": [90.0, 0.0],
            "teacher_top_k_action_indices": [11, 22],
            "principal_variation_uci": ["e2e4", "e7e5"],
            "principal_variation_action_indices": [11, 101],
            "best_reply_uci": "e7e5",
            "best_reply_action_index": 101,
            "pv_length": 2,
            "top1_minus_top2_cp": 90.0,
        }
    )
    disagreement_example = SearchDisagreementExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "validation",
            "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "feature_vector": [0.0, 1.0],
            "candidate_context_version": 1,
            "global_context_version": 1,
            "global_features": [0.0],
            "candidate_action_indices": [11, 22],
            "candidate_features": [[0.0], [1.0]],
            "teacher_engine": "/usr/games/stockfish18",
            "teacher_nodes": 128,
            "teacher_depth": None,
            "teacher_movetime_ms": None,
            "teacher_multipv": 2,
            "teacher_coverage_ratio": 1.0,
            "teacher_root_value_cp": 90.0,
            "teacher_root_value_mate": None,
            "teacher_candidate_scores_cp": [90.0, 0.0],
            "teacher_policy": [0.8, 0.2],
            "teacher_top_k_action_indices": [11, 22],
            "proposer_checkpoint": "models/proposer/symbolic/checkpoint.pt",
            "proposer_candidate_scores": [0.0, 1.0],
            "proposer_policy": [0.2, 0.8],
            "proposer_top_k_action_indices": [22, 11],
            "teacher_top1_action_index": 11,
            "proposer_top1_action_index": 22,
            "teacher_rank_of_proposer_top1": 2,
            "proposer_rank_of_teacher_top1": 2,
            "top1_disagrees": True,
            "teacher_top1_minus_top2_cp": 90.0,
            "proposer_top1_minus_top2_logit": 1.0,
            "teacher_top1_advantage_cp": 90.0,
            "policy_l1_distance": 1.2,
        }
    )

    built = build_search_curriculum_examples([trace_example], [disagreement_example])

    assert len(built) == 1
    curriculum = built[0]
    assert curriculum.teacher_top1_action_index == 11
    assert curriculum.best_reply_action_index == 101
    assert curriculum.pv_length == 2
    assert "forced_teacher" in curriculum.bucket_labels
    assert "reply_supervised" in curriculum.bucket_labels
    assert "top1_disagreement" in curriculum.bucket_labels
    assert "teacher_punishes_proposer" in curriculum.bucket_labels
    assert "policy_shape_mismatch" in curriculum.bucket_labels
    assert "capture_line" in curriculum.bucket_labels
    assert "checking_line" in curriculum.bucket_labels
    assert curriculum.curriculum_priority > 0.0


def test_search_curriculum_example_roundtrips_json() -> None:
    example = SearchCurriculumExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "test",
            "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "teacher_top1_action_index": 11,
            "best_reply_action_index": 101,
            "pv_length": 2,
            "bucket_labels": ["forced_teacher", "reply_supervised"],
            "curriculum_priority": 2.75,
            "teacher_top1_minus_top2_cp": 90.0,
            "proposer_rank_of_teacher_top1": 2,
            "teacher_rank_of_proposer_top1": 3,
            "teacher_top1_advantage_cp": 40.0,
            "policy_l1_distance": 0.5,
            "top1_disagrees": True,
        }
    )

    roundtrip = SearchCurriculumExample.from_json(json.dumps(example.to_dict()))

    assert roundtrip == example
