"""Tests for the first trainable bounded planner arm."""

from __future__ import annotations

import json
from pathlib import Path

from train.config import load_planner_train_config
from train.datasets.artifacts import SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE
from train.datasets.contracts import candidate_context_feature_dim, transition_context_feature_dim
from train.datasets.planner_head import PlannerHeadExample, write_planner_head_artifact
from train.trainers import evaluate_planner_checkpoint, train_planner


def test_load_planner_train_config_accepts_set_v1(tmp_path: Path) -> None:
    config_path = tmp_path / "planner_config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "planner_out",
                "data": {
                    "train_path": "planner_head_train.jsonl",
                    "validation_path": "planner_head_validation.jsonl",
                },
                "model": {
                    "architecture": "set_v1",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "teacher_policy_loss_weight": 1.0,
                    "teacher_kl_loss_weight": 0.25,
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {"bundle_dir": "planner_bundle"},
            }
        ),
        encoding="utf-8",
    )

    config = load_planner_train_config(config_path)

    assert config.model.architecture == "set_v1"
    assert config.runtime.torch_threads == 1


def test_train_and_evaluate_planner_checkpoint(tmp_path: Path) -> None:
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    global_dim = SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE

    train_path = tmp_path / "planner_head_train.jsonl"
    validation_path = tmp_path / "planner_head_validation.jsonl"

    train_examples = [
        _planner_example(
            sample_id="train-1",
            teacher_index=0,
            candidate_dim=candidate_dim,
            transition_dim=transition_dim,
            global_dim=global_dim,
        ),
        _planner_example(
            sample_id="train-2",
            teacher_index=1,
            candidate_dim=candidate_dim,
            transition_dim=transition_dim,
            global_dim=global_dim,
        ),
        _planner_example(
            sample_id="train-3",
            teacher_index=0,
            candidate_dim=candidate_dim,
            transition_dim=transition_dim,
            global_dim=global_dim,
        ),
        _planner_example(
            sample_id="train-4",
            teacher_index=1,
            candidate_dim=candidate_dim,
            transition_dim=transition_dim,
            global_dim=global_dim,
        ),
    ]
    validation_examples = [
        _planner_example(
            sample_id="validation-1",
            teacher_index=0,
            candidate_dim=candidate_dim,
            transition_dim=transition_dim,
            global_dim=global_dim,
        ),
        _planner_example(
            sample_id="validation-2",
            teacher_index=1,
            candidate_dim=candidate_dim,
            transition_dim=transition_dim,
            global_dim=global_dim,
        ),
    ]
    write_planner_head_artifact(train_path, train_examples)
    write_planner_head_artifact(validation_path, validation_examples)

    config_path = tmp_path / "planner_train.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": str(tmp_path / "planner_run"),
                "data": {
                    "train_path": str(train_path),
                    "validation_path": str(validation_path),
                },
                "model": {
                    "architecture": "set_v1",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 2,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "teacher_policy_loss_weight": 1.0,
                    "teacher_kl_loss_weight": 0.25,
                    "curriculum_priority_weight": 0.1,
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": str(tmp_path / "planner_bundle"),
                    "checkpoint_name": "checkpoint.pt",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_planner_train_config(config_path)
    run = train_planner(config, repo_root=tmp_path)

    checkpoint_path = Path(run.export_paths["checkpoint"])
    assert checkpoint_path.exists()
    assert Path(run.summary_path).exists()
    assert run.best_epoch >= 1

    metrics = evaluate_planner_checkpoint(checkpoint_path, dataset_path=validation_path)

    assert metrics.total_examples == 2
    assert 0.0 <= metrics.root_top1_accuracy <= 1.0
    assert 0.0 <= metrics.root_top3_accuracy <= 1.0


def _planner_example(
    *,
    sample_id: str,
    teacher_index: int,
    candidate_dim: int,
    transition_dim: int,
    global_dim: int,
) -> PlannerHeadExample:
    candidate_action_indices = [17, 42]
    teacher_policy = [1.0 if index == teacher_index else 0.0 for index in range(2)]
    feature_shift = float(teacher_index)
    return PlannerHeadExample(
        sample_id=sample_id,
        split="train" if sample_id.startswith("train") else "validation",
        fen="8/8/8/8/8/8/8/8 w - - 0 1",
        feature_vector=[feature_shift] * 230,
        candidate_context_version=2,
        global_context_version=1,
        global_features=[feature_shift] * global_dim,
        candidate_action_indices=candidate_action_indices,
        candidate_features=[
            [1.0] + [0.0] * (candidate_dim - 1),
            [0.0, 1.0] + [0.0] * (candidate_dim - 2),
        ],
        proposer_scores=[0.7, 0.3] if teacher_index == 0 else [0.3, 0.7],
        transition_context_version=1,
        transition_features=[
            [1.0] + [0.0] * (transition_dim - 1),
            [0.0, 1.0] + [0.0] * (transition_dim - 2),
        ],
        reply_peak_probabilities=[0.1, 0.4] if teacher_index == 0 else [0.4, 0.1],
        pressures=[0.1, 0.5] if teacher_index == 0 else [0.5, 0.1],
        uncertainties=[0.1, 0.2],
        curriculum_bucket_labels=["stable_agreement"],
        curriculum_priority=1.0 + feature_shift,
        teacher_top1_action_index=candidate_action_indices[teacher_index],
        teacher_top1_candidate_index=teacher_index,
        teacher_policy=teacher_policy,
        teacher_root_value_cp=25.0,
        teacher_top1_minus_top2_cp=80.0,
    )
