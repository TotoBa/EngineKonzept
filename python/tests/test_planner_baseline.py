"""Tests for the first bounded opponent-aware planner baseline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from train.datasets.schema import DatasetExample
from train.datasets.search_teacher import SearchTeacherExample
from train.eval.planner import evaluate_two_ply_planner_baseline


def test_evaluate_two_ply_planner_baseline_with_learned_opponent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    pytest.importorskip("torch")

    root_example = DatasetExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "test",
            "source": "fixture",
            "fen": "4k3/8/8/8/8/8/PP6/4K3 w - - 0 1",
            "side_to_move": "w",
            "selected_move_uci": None,
            "selected_action_encoding": None,
            "next_fen": None,
            "legal_moves": ["a2a3", "a2a4"],
            "legal_action_encodings": [[8, 16, 0], [8, 24, 0]],
            "position_encoding": {
                "piece_tokens": [[0, 0, 3]],
                "square_tokens": [[square, 0] for square in range(64)],
                "rule_token": [0, 2, -1, 0, 1, 0],
            },
            "wdl_target": None,
            "annotations": {
                "in_check": False,
                "is_checkmate": False,
                "is_stalemate": False,
                "has_legal_en_passant": False,
                "has_legal_castle": False,
                "has_legal_promotion": False,
                "is_low_material_endgame": True,
                "legal_move_count": 2,
                "piece_count": 3,
                "selected_move_is_capture": False,
                "selected_move_is_promotion": False,
                "selected_move_is_castle": False,
                "selected_move_is_en_passant": False,
                "selected_move_gives_check": False,
            },
            "result": None,
            "metadata": {},
        }
    )
    teacher_example = SearchTeacherExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "test",
            "fen": root_example.fen,
            "feature_vector": [0.0] * 230,
            "candidate_context_version": 2,
            "global_context_version": 1,
            "global_features": [0.0] * 9,
            "candidate_action_indices": [1, 2],
            "candidate_features": [[0.0] * 35, [0.0] * 35],
            "teacher_engine": "/usr/games/stockfish18",
            "teacher_nodes": 64,
            "teacher_depth": None,
            "teacher_movetime_ms": None,
            "teacher_multipv": 2,
            "teacher_coverage_ratio": 1.0,
            "teacher_root_value_cp": 20.0,
            "teacher_root_value_mate": None,
            "teacher_candidate_scores_cp": [5.0, 20.0],
            "teacher_policy": [0.1, 0.9],
            "teacher_top_k_action_indices": [2, 1],
            "teacher_pv_uci": ["a2a4"],
        }
    )

    monkeypatch.setattr(
        "train.eval.planner.load_split_examples",
        lambda dataset_dir, split: [root_example],
    )
    monkeypatch.setattr(
        "train.eval.planner.load_search_teacher_examples",
        lambda path: [teacher_example],
    )
    monkeypatch.setattr(
        "train.eval.planner.load_symbolic_proposer_checkpoint",
        lambda checkpoint_path: (object(), None),
    )
    monkeypatch.setattr(
        "train.eval.planner.load_opponent_head_checkpoint",
        lambda checkpoint_path: (object(), None),
    )
    monkeypatch.setattr(
        "train.eval.planner.score_symbolic_candidates",
        lambda model, **kwargs: ([1.0, 0.9], [0.525, 0.475]),
    )
    monkeypatch.setattr(
        "train.eval.planner.move_uci_for_action",
        lambda example, action_index: {1: "a2a3", 2: "a2a4"}[action_index],
    )
    monkeypatch.setattr(
        "train.eval.planner.label_records_with_oracle",
        lambda records, repo_root=None: [
            {"selected_move_uci": record.selected_move_uci, "fen": record.fen}
            for record in records
        ],
    )

    def fake_dataset_example_from_oracle_payload(*, sample_id, split, source, fen, payload):
        if payload.get("selected_move_uci") is not None:
            next_fen = "successor-a" if payload["selected_move_uci"] == "a2a3" else "successor-b"
            return SimpleNamespace(
                sample_id=sample_id,
                split=split,
                source=source,
                fen=fen,
                next_fen=next_fen,
                position_encoding=root_example.position_encoding,
                annotations=root_example.annotations,
                selected_action_encoding=[8, 16, 0]
                if payload["selected_move_uci"] == "a2a3"
                else [8, 24, 0],
            )
        return SimpleNamespace(
            sample_id=sample_id,
            split=split,
            source=source,
            fen=fen,
            next_fen=None,
            position_encoding=root_example.position_encoding,
            annotations=root_example.annotations,
            selected_action_encoding=None,
            legal_moves=["e8e7", "e8d7"],
            legal_action_encodings=[[60, 52, 0], [60, 51, 0]],
        )

    monkeypatch.setattr(
        "train.eval.planner.dataset_example_from_oracle_payload",
        fake_dataset_example_from_oracle_payload,
    )
    monkeypatch.setattr(
        "train.eval.planner.build_symbolic_proposer_example",
        lambda example, candidate_context_version, global_context_version: SimpleNamespace(
            feature_vector=[0.0] * 230,
            candidate_context_version=2,
            global_context_version=1,
            global_features=[0.0] * 9,
            candidate_action_indices=[10, 11],
            candidate_features=[[0.0] * 35, [0.0] * 35],
        ),
    )
    monkeypatch.setattr(
        "train.eval.planner.build_transition_context_features",
        lambda example, version: [0.0] * 45,
    )
    monkeypatch.setattr(
        "train.eval.planner.pack_position_features",
        lambda position_encoding: [0.0] * 230,
    )

    def fake_score_opponent_candidates(model, **kwargs):
        chosen_action_index = kwargs["chosen_action_index"]
        if chosen_action_index == 1:
            return [1.0, 0.0], [0.9, 0.1], 0.8, 0.7
        return [1.0, 0.0], [0.2, 0.8], 0.1, 0.1

    monkeypatch.setattr(
        "train.eval.planner.score_opponent_candidates",
        fake_score_opponent_candidates,
    )

    metrics = evaluate_two_ply_planner_baseline(
        tmp_path / "proposer.pt",
        dataset_dir=tmp_path,
        search_teacher_path=tmp_path / "search_teacher_test.jsonl",
        split="test",
        opponent_mode="learned",
        opponent_checkpoint=tmp_path / "opponent.pt",
        root_top_k=2,
        repo_root=tmp_path,
    )

    assert metrics.total_examples == 1
    assert metrics.teacher_covered_examples == 1
    assert metrics.root_top1_accuracy == 1.0
    assert metrics.teacher_root_mean_reciprocal_rank == 1.0
    assert 0.0 <= metrics.teacher_root_mean_probability <= 1.0
