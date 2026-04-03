"""Tests for the first trainable bounded planner arm."""

from __future__ import annotations

import json
from pathlib import Path

from train.config import load_planner_train_config
from train.datasets.artifacts import SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE
from train.datasets.contracts import candidate_context_feature_dim, transition_context_feature_dim
from train.datasets.planner_head import (
    PlannerHeadExample,
    build_teacher_candidate_rank_bucket_targets,
    build_teacher_candidate_score_delta_targets_cp,
    materialize_planner_teacher_targets,
    write_planner_head_artifact,
)
from train.datasets.search_teacher import SearchTeacherExample
from train.trainers import evaluate_planner_checkpoint, train_planner


def test_load_planner_train_config_accepts_set_v6(tmp_path: Path) -> None:
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
                    "architecture": "set_v6",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "latent_feature_dim": 8,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "teacher_policy_loss_weight": 1.0,
                    "teacher_kl_loss_weight": 0.25,
                    "teacher_score_loss_weight": 0.1,
                    "teacher_margin_loss_weight": 0.1,
                    "teacher_rank_loss_weight": 0.1,
                    "root_value_loss_weight": 0.1,
                    "root_gap_loss_weight": 0.05,
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {"bundle_dir": "planner_bundle"},
            }
        ),
        encoding="utf-8",
    )

    config = load_planner_train_config(config_path)

    assert config.model.architecture == "set_v6"
    assert config.model.latent_feature_dim == 8
    assert config.runtime.torch_threads == 1
    assert config.optimization.teacher_score_loss_weight == 0.1
    assert config.optimization.teacher_margin_loss_weight == 0.1
    assert config.optimization.teacher_rank_loss_weight == 0.1
    assert config.optimization.root_value_loss_weight == 0.1


def test_load_planner_train_config_accepts_recurrent_v1(tmp_path: Path) -> None:
    config_path = tmp_path / "planner_recurrent_config.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 5,
                "output_dir": "planner_out",
                "data": {
                    "train_path": "planner_head_train.jsonl",
                    "validation_path": "planner_head_validation.jsonl",
                },
                "model": {
                    "architecture": "recurrent_v1",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "latent_feature_dim": 8,
                    "deliberation_steps": 2,
                    "memory_slots": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "teacher_policy_loss_weight": 1.0,
                    "teacher_kl_loss_weight": 0.25,
                    "root_value_loss_weight": 0.1,
                    "root_gap_loss_weight": 0.05,
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {"bundle_dir": "planner_bundle"},
            }
        ),
        encoding="utf-8",
    )

    config = load_planner_train_config(config_path)

    assert config.model.architecture == "recurrent_v1"
    assert config.model.deliberation_steps == 2
    assert config.model.memory_slots == 2


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
                    "architecture": "set_v3",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "latent_feature_dim": 4,
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
                    "root_value_loss_weight": 0.1,
                    "root_gap_loss_weight": 0.1,
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
    assert metrics.root_value_mae_cp >= 0.0


def test_train_and_evaluate_planner_checkpoint_with_score_aux(tmp_path: Path) -> None:
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    global_dim = SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE

    train_path = tmp_path / "planner_head_train_v6.jsonl"
    validation_path = tmp_path / "planner_head_validation_v6.jsonl"
    write_planner_head_artifact(
        train_path,
        [
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
        ],
    )
    write_planner_head_artifact(
        validation_path,
        [
            _planner_example(
                sample_id="validation-1",
                teacher_index=0,
                candidate_dim=candidate_dim,
                transition_dim=transition_dim,
                global_dim=global_dim,
            )
        ],
    )

    config_path = tmp_path / "planner_train_set_v6.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 9,
                "output_dir": str(tmp_path / "planner_run_set_v6"),
                "data": {
                    "train_path": str(train_path),
                    "validation_path": str(validation_path),
                },
                "model": {
                    "architecture": "set_v6",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "latent_feature_dim": 4,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 2,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "teacher_policy_loss_weight": 1.0,
                    "teacher_kl_loss_weight": 0.25,
                    "teacher_score_loss_weight": 0.1,
                    "teacher_margin_loss_weight": 0.1,
                    "teacher_rank_loss_weight": 0.1,
                    "curriculum_priority_weight": 0.1,
                    "root_value_loss_weight": 0.1,
                    "root_gap_loss_weight": 0.1,
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": str(tmp_path / "planner_bundle_set_v6"),
                    "checkpoint_name": "checkpoint.pt",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_planner(load_planner_train_config(config_path), repo_root=tmp_path)
    metrics = evaluate_planner_checkpoint(
        Path(run.export_paths["checkpoint"]),
        dataset_path=validation_path,
    )

    assert metrics.total_examples == 1
    assert metrics.teacher_score_loss >= 0.0
    assert metrics.teacher_score_mae_cp >= 0.0
    assert metrics.teacher_margin_loss >= 0.0
    assert metrics.teacher_margin_mae_cp >= 0.0
    assert metrics.teacher_rank_loss >= 0.0
    assert 0.0 <= metrics.teacher_rank_accuracy <= 1.0


def test_set_v1_ignores_latent_features_in_artifacts(tmp_path: Path) -> None:
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    global_dim = SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE

    train_path = tmp_path / "planner_head_train.jsonl"
    validation_path = tmp_path / "planner_head_validation.jsonl"
    examples = [
        _planner_example(
            sample_id="train-1",
            teacher_index=0,
            candidate_dim=candidate_dim,
            transition_dim=transition_dim,
            global_dim=global_dim,
        ),
        _planner_example(
            sample_id="validation-1",
            teacher_index=1,
            candidate_dim=candidate_dim,
            transition_dim=transition_dim,
            global_dim=global_dim,
        ),
    ]
    write_planner_head_artifact(train_path, [examples[0]])
    write_planner_head_artifact(validation_path, [examples[1]])

    config_path = tmp_path / "planner_train_set_v1.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 5,
                "output_dir": str(tmp_path / "planner_run_set_v1"),
                "data": {
                    "train_path": str(train_path),
                    "validation_path": str(validation_path),
                },
                "model": {
                    "architecture": "set_v1",
                    "hidden_dim": 16,
                    "hidden_layers": 1,
                    "action_embedding_dim": 8,
                    "latent_feature_dim": 0,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "teacher_policy_loss_weight": 1.0,
                    "teacher_kl_loss_weight": 0.25,
                    "curriculum_priority_weight": 0.0,
                    "root_value_loss_weight": 0.1,
                    "root_gap_loss_weight": 0.1,
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": str(tmp_path / "planner_bundle_set_v1"),
                    "checkpoint_name": "checkpoint.pt",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_planner(load_planner_train_config(config_path), repo_root=tmp_path)
    metrics = evaluate_planner_checkpoint(
        Path(run.export_paths["checkpoint"]),
        dataset_path=validation_path,
    )

    assert metrics.total_examples == 1
    assert 0.0 <= metrics.root_top1_accuracy <= 1.0


def test_train_and_evaluate_planner_checkpoint_with_recurrent_v1(tmp_path: Path) -> None:
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    global_dim = SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE

    train_path = tmp_path / "planner_head_train_recurrent.jsonl"
    validation_path = tmp_path / "planner_head_validation_recurrent.jsonl"
    write_planner_head_artifact(
        train_path,
        [
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
        ],
    )
    write_planner_head_artifact(
        validation_path,
        [
            _planner_example(
                sample_id="validation-1",
                teacher_index=0,
                candidate_dim=candidate_dim,
                transition_dim=transition_dim,
                global_dim=global_dim,
            )
        ],
    )

    config_path = tmp_path / "planner_train_recurrent.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 10,
                "output_dir": str(tmp_path / "planner_run_recurrent"),
                "data": {
                    "train_path": str(train_path),
                    "validation_path": str(validation_path),
                },
                "model": {
                    "architecture": "recurrent_v1",
                    "hidden_dim": 32,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "latent_feature_dim": 4,
                    "deliberation_steps": 2,
                    "memory_slots": 2,
                    "dropout": 0.0,
                },
                "optimization": {
                    "epochs": 2,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0,
                    "teacher_policy_loss_weight": 1.0,
                    "teacher_kl_loss_weight": 0.25,
                    "root_value_loss_weight": 0.1,
                    "root_gap_loss_weight": 0.1,
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": str(tmp_path / "planner_bundle_recurrent"),
                    "checkpoint_name": "checkpoint.pt",
                },
            }
        ),
        encoding="utf-8",
    )

    run = train_planner(load_planner_train_config(config_path), repo_root=tmp_path)
    metrics = evaluate_planner_checkpoint(
        Path(run.export_paths["checkpoint"]),
        dataset_path=validation_path,
    )

    assert metrics.total_examples == 1
    assert 0.0 <= metrics.root_top1_accuracy <= 1.0


def test_planner_head_example_accepts_optional_teacher_candidate_scores() -> None:
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    global_dim = SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE

    example = _planner_example(
        sample_id="train-1",
        teacher_index=0,
        candidate_dim=candidate_dim,
        transition_dim=transition_dim,
        global_dim=global_dim,
    )
    payload = example.to_dict()

    roundtrip = PlannerHeadExample.from_dict(payload)
    legacy_roundtrip = PlannerHeadExample.from_dict(
        {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "teacher_candidate_scores_cp",
                "teacher_candidate_score_delta_targets_cp",
                "teacher_rank_bucket_version",
                "teacher_candidate_rank_bucket_targets",
            }
        }
    )

    assert roundtrip.teacher_candidate_scores_cp == [25.0, -55.0]
    assert roundtrip.teacher_candidate_score_delta_targets_cp == [0.0, -80.0]
    assert roundtrip.teacher_rank_bucket_version == 1
    assert roundtrip.teacher_candidate_rank_bucket_targets == [0, 1]
    assert legacy_roundtrip.teacher_candidate_scores_cp is None
    assert legacy_roundtrip.teacher_candidate_score_delta_targets_cp is None
    assert legacy_roundtrip.teacher_rank_bucket_version is None
    assert legacy_roundtrip.teacher_candidate_rank_bucket_targets is None


def test_build_teacher_candidate_score_delta_targets_cp_clips_large_gaps() -> None:
    targets = build_teacher_candidate_score_delta_targets_cp(
        [180.0, -260.0, 40.0],
        considered_indices=[0, 1, 2],
        teacher_root_value_cp=40.0,
    )

    assert targets == [140.0, -256.0, 0.0]


def test_build_teacher_candidate_rank_bucket_targets_marks_top1_top3_and_tail() -> None:
    targets = build_teacher_candidate_rank_bucket_targets(
        [120.0, 80.0, 60.0, -40.0],
        considered_indices=[0, 1, 2, 3],
        teacher_top1_candidate_index=0,
    )

    assert targets == [0, 1, 1, 2]


def test_materialize_planner_teacher_targets_backfills_missing_score_fields() -> None:
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    global_dim = SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE

    legacy_example = _planner_example(
        sample_id="train-legacy",
        teacher_index=0,
        candidate_dim=candidate_dim,
        transition_dim=transition_dim,
        global_dim=global_dim,
    )
    legacy_payload = legacy_example.to_dict()
    for key in (
        "teacher_candidate_scores_cp",
        "teacher_candidate_score_delta_targets_cp",
        "teacher_rank_bucket_version",
        "teacher_candidate_rank_bucket_targets",
    ):
        legacy_payload.pop(key, None)
    legacy_roundtrip = PlannerHeadExample.from_dict(legacy_payload)

    teacher_example = SearchTeacherExample(
        sample_id="train-legacy",
        split="train",
        fen=legacy_roundtrip.fen,
        feature_vector=list(legacy_roundtrip.feature_vector),
        candidate_context_version=legacy_roundtrip.candidate_context_version,
        global_context_version=legacy_roundtrip.global_context_version,
        global_features=list(legacy_roundtrip.global_features),
        candidate_action_indices=[7, 17, 42, 99],
        candidate_features=[
            [0.0] * candidate_dim,
            [1.0] + [0.0] * (candidate_dim - 1),
            [0.0, 1.0] + [0.0] * (candidate_dim - 2),
            [0.0] * candidate_dim,
        ],
        teacher_engine="/usr/games/stockfish18",
        teacher_nodes=64,
        teacher_depth=None,
        teacher_movetime_ms=None,
        teacher_multipv=4,
        teacher_coverage_ratio=1.0,
        teacher_root_value_cp=25.0,
        teacher_root_value_mate=None,
        teacher_candidate_scores_cp=[-40.0, 25.0, -55.0, -120.0],
        teacher_policy=[0.0, 1.0, 0.0, 0.0],
        teacher_top_k_action_indices=[17, 42, 7, 99],
        teacher_pv_uci=["a2a3"],
    )

    materialized = materialize_planner_teacher_targets(
        [legacy_roundtrip],
        teacher_examples=[teacher_example],
    )

    assert len(materialized) == 1
    rebuilt = materialized[0]
    assert rebuilt.teacher_candidate_scores_cp == [25.0, -55.0]
    assert rebuilt.teacher_candidate_score_delta_targets_cp == [0.0, -80.0]
    assert rebuilt.teacher_rank_bucket_version == 1
    assert rebuilt.teacher_candidate_rank_bucket_targets == [0, 1]


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
        latent_state_version=1,
        latent_features=[
            [feature_shift, 0.1, 0.2, 0.3],
            [0.3, 0.2, 0.1, feature_shift],
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
        teacher_candidate_scores_cp=[25.0, -55.0],
        teacher_candidate_score_delta_targets_cp=[0.0, -80.0],
        teacher_rank_bucket_version=1,
        teacher_candidate_rank_bucket_targets=[0, 1],
    )
