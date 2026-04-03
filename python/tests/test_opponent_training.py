"""Tests for Phase-7 opponent-head training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.config import OpponentTrainConfig
from train.trainers.opponent import evaluate_opponent_checkpoint, train_opponent


def test_train_and_evaluate_opponent_head(tmp_path: Path) -> None:
    pytest.importorskip("torch")

    train_path = tmp_path / "opponent_head_train.jsonl"
    extra_train_path = tmp_path / "opponent_head_train_extra.jsonl"
    validation_path = tmp_path / "opponent_head_validation.jsonl"
    extra_validation_path = tmp_path / "opponent_head_validation_extra.jsonl"
    train_examples = [
        _example_payload("sample-1", "train", teacher_index=0),
        _example_payload("sample-2", "train", teacher_index=1),
    ]
    validation_examples = [_example_payload("sample-3", "validation", teacher_index=0)]
    train_path.write_text(
        "\n".join(json.dumps(example) for example in train_examples) + "\n",
        encoding="utf-8",
    )
    extra_train_path.write_text(
        json.dumps(_example_payload("sample-4", "train", teacher_index=0)) + "\n",
        encoding="utf-8",
    )
    validation_path.write_text(
        "\n".join(json.dumps(example) for example in validation_examples) + "\n",
        encoding="utf-8",
    )
    extra_validation_path.write_text(
        json.dumps(_example_payload("sample-5", "validation", teacher_index=1)) + "\n",
        encoding="utf-8",
    )

    config = OpponentTrainConfig.from_dict(
        {
            "seed": 0,
            "output_dir": str(tmp_path / "artifacts"),
            "data": {
                "train_path": str(train_path),
                "validation_path": str(validation_path),
                "additional_train_paths": [str(extra_train_path)],
                "additional_validation_paths": [str(extra_validation_path)],
            },
            "model": {
                "architecture": "set_v2",
                "hidden_dim": 32,
                "hidden_layers": 1,
                "action_embedding_dim": 16,
                "dropout": 0.0,
            },
            "optimization": {
                "epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "reply_policy_loss_weight": 1.0,
                "pressure_loss_weight": 0.25,
                "uncertainty_loss_weight": 0.25,
                "curriculum_priority_weight": 0.25,
            },
            "evaluation": {
                "top_k": 3,
            },
            "runtime": {
                "torch_threads": 0,
                "dataloader_workers": 0,
            },
            "export": {
                "bundle_dir": str(tmp_path / "models"),
                "checkpoint_name": "checkpoint.pt",
            },
        }
    )

    run = train_opponent(config, repo_root=tmp_path)

    checkpoint_path = tmp_path / "models" / "checkpoint.pt"
    assert checkpoint_path.exists()
    assert Path(run.summary_path).exists()
    assert run.best_epoch == 1

    metrics = evaluate_opponent_checkpoint(
        checkpoint_path,
        dataset_paths=[validation_path, extra_validation_path],
        top_k=3,
    )

    assert metrics.total_examples == 2
    assert metrics.supervised_examples == 2
    assert 0.0 <= metrics.reply_top1_accuracy <= 1.0
    assert 0.0 <= metrics.reply_top3_accuracy <= 1.0
    assert 0.0 <= metrics.teacher_reply_mean_probability <= 1.0
    assert metrics.examples_per_second > 0.0


def _example_payload(sample_id: str, split: str, *, teacher_index: int) -> dict[str, object]:
    candidate_features = [
        [1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 30,
        [0.0] * 35,
    ]
    teacher_policy = [0.0, 0.0]
    teacher_policy[teacher_index] = 1.0
    teacher_action_index = [2, 3][teacher_index]
    return {
        "sample_id": sample_id,
        "split": split,
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
        "reply_candidate_features": candidate_features,
        "teacher_reply_uci": "e7e5" if teacher_index == 0 else "d7d5",
        "teacher_reply_action_index": teacher_action_index,
        "teacher_reply_policy": teacher_policy,
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
