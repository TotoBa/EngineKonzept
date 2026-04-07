from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.config import (
    PlannerDataConfig,
    PlannerEvaluationConfig,
    PlannerExportConfig,
    PlannerRuntimeConfig,
)
from train.datasets.artifacts import (
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
    pack_position_features,
)
from train.datasets.planner_head import PlannerHeadExample
from train.datasets.schema import PositionEncoding
from train.models.lapv1 import LAPv1Config
from train.trainers import LAPv1Stage2Config, LAPv1Stage2PhaseConfig, train_lapv1
from train.trainers.lapv1 import LAPv1OptimizationConfig, LAPv1TrainConfig


pytest.importorskip("torch")
pytest.importorskip("chess")


def test_train_lapv1_stage2_logs_monotonicity_and_rollback_stats(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_stage2_train.jsonl"
    validation_path = tmp_path / "lapv1_stage2_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
            _planner_example("train-3", teacher_index=0, teacher_cp=-40.0, teacher_gap=25.0),
            _planner_example("train-4", teacher_index=1, teacher_cp=35.0, teacher_gap=15.0),
        ],
    )
    _write_examples(
        validation_path,
        [
            _planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0),
            _planner_example("validation-2", teacher_index=1, teacher_cp=0.0, teacher_gap=5.0),
        ],
    )

    config = LAPv1TrainConfig(
        seed=13,
        output_dir=str(tmp_path / "lapv1_stage2_out"),
        stage="T2",
        stage2=LAPv1Stage2Config(max_inner_steps_schedule=(2,)),
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {
                    "max_inner_steps": 2,
                    "min_inner_steps": 2,
                    "memory_slots": 4,
                    "rollback_buffer_size": 4,
                },
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            sharpness_target_loss_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
            deliberation_monotonicity_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert len(run.history) == 2
    first_epoch = run.history[0]
    assert first_epoch["max_inner_steps"] == 2
    assert first_epoch["train"]["sharpness_target_loss"] >= 0.0
    assert first_epoch["train"]["deliberation_monotonicity_loss"] >= 0.0
    assert first_epoch["train"]["deliberation_step_policy_loss"] >= 0.0
    assert "rollbacks" in first_epoch["train"]
    assert "rollback_hit_rate" in first_epoch["train"]
    assert "rollback_example_rate" in first_epoch["train"]
    assert "initial_root_top1_accuracy" in first_epoch["validation"]
    assert "top1_changed_rate" in first_epoch["validation"]
    assert "step_histogram" in first_epoch["validation"]
    assert first_epoch["train"]["rollback_hit_rate"] >= 0.0
    assert first_epoch["validation"]["deliberation_monotonicity_loss"] >= 0.0


def test_train_lapv1_stage2_supports_freeze_then_joint_phases(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_stage2_train.jsonl"
    validation_path = tmp_path / "lapv1_stage2_validation.jsonl"
    hard_train_path = tmp_path / "lapv1_stage2_hard_train.jsonl"
    hard_validation_path = tmp_path / "lapv1_stage2_hard_validation.jsonl"
    _write_examples(
        train_path,
        [
            _planner_example("train-1", teacher_index=0, teacher_cp=60.0, teacher_gap=40.0),
            _planner_example("train-2", teacher_index=1, teacher_cp=10.0, teacher_gap=10.0),
            _planner_example("train-3", teacher_index=0, teacher_cp=-40.0, teacher_gap=25.0),
            _planner_example("train-4", teacher_index=1, teacher_cp=35.0, teacher_gap=15.0),
        ],
    )
    _write_examples(
        validation_path,
        [
            _planner_example("validation-1", teacher_index=0, teacher_cp=50.0, teacher_gap=30.0),
            _planner_example("validation-2", teacher_index=1, teacher_cp=0.0, teacher_gap=5.0),
        ],
    )
    _write_examples(
        hard_train_path,
        [
            _planner_example("hard-train-1", teacher_index=1, teacher_cp=5.0, teacher_gap=2.0),
            _planner_example("hard-train-2", teacher_index=0, teacher_cp=-5.0, teacher_gap=3.0),
        ],
    )
    _write_examples(
        hard_validation_path,
        [
            _planner_example(
                "hard-validation-1",
                teacher_index=1,
                teacher_cp=0.0,
                teacher_gap=1.0,
            ),
        ],
    )

    config = LAPv1TrainConfig(
        seed=29,
        output_dir=str(tmp_path / "lapv1_stage2_out"),
        stage="T2",
        stage2=LAPv1Stage2Config(
            phases=(
                LAPv1Stage2PhaseConfig(
                    name="freeze_inner",
                    epochs=1,
                    trainable_parameter_groups=("inner_loop",),
                    max_inner_steps_schedule=(1, 2),
                    min_inner_steps_schedule=(1, 1),
                    train_paths=(str(hard_train_path),),
                    validation_paths=(str(hard_validation_path),),
                ),
                LAPv1Stage2PhaseConfig(
                    name="joint_finetune",
                    epochs=1,
                    trainable_parameter_groups=("all",),
                    max_inner_steps_schedule=(4,),
                    min_inner_steps_schedule=(2,),
                ),
            )
        ),
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {
                    "max_inner_steps": 4,
                    "min_inner_steps": 1,
                    "memory_slots": 4,
                    "rollback_buffer_size": 4,
                },
                "opponent_head": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 1,
                    "action_embedding_dim": 16,
                    "dropout": 0.0,
                },
                "value_head": {"hidden_dim": 1024},
                "policy_head": {
                    "hidden_dim": 512,
                    "action_embedding_dim": 32,
                    "feedforward_dim": 1024,
                },
                "state_embedder": {"feedforward_dim": 1024},
                "intention_encoder": {"feedforward_dim": 1024},
            }
        ),
        optimization=LAPv1OptimizationConfig(
            epochs=2,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_grad_norm=1.0,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            sharpness_target_loss_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
            deliberation_monotonicity_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    assert [entry["stage2_phase"] for entry in run.history] == [
        "freeze_inner",
        "joint_finetune",
    ]
    assert run.history[0]["trainable_parameter_groups"] == ["inner_loop"]
    assert run.history[1]["trainable_parameter_groups"] == ["all"]
    assert run.history[0]["max_inner_steps"] == 2
    assert run.history[0]["min_inner_steps"] == 1
    assert run.history[1]["max_inner_steps"] == 4
    assert run.history[1]["min_inner_steps"] == 2
    assert run.history[0]["train_dataset_paths"] == [str(hard_train_path)]
    assert run.history[0]["validation_dataset_paths"] == [str(hard_validation_path)]
    assert run.history[1]["train_dataset_paths"] == [str(train_path)]
    assert run.history[1]["validation_dataset_paths"] == [str(validation_path)]
    assert "deliberation_improvement_loss" in run.history[0]["train"]
    assert "root_incorrect_improvement_rate" in run.history[0]["validation"]


def _write_examples(path: Path, examples: list[PlannerHeadExample]) -> None:
    path.write_text(
        "".join(json.dumps(example.to_dict()) + "\n" for example in examples),
        encoding="utf-8",
    )


def _planner_example(
    sample_id: str,
    *,
    teacher_index: int,
    teacher_cp: float,
    teacher_gap: float,
) -> PlannerHeadExample:
    feature_vector = pack_position_features(
        PositionEncoding(
            piece_tokens=[[4, 0, 5], [60, 1, 5], [0, 0, 3], [63, 1, 3]],
            square_tokens=[[square_index, 0] for square_index in range(64)],
            rule_token=[0, 0, -1, 0, 1, 0],
        )
    )
    candidate_features = [
        [1.0] + [0.0] * 34,
        [0.0, 1.0] + [0.0] * 33,
    ]
    teacher_policy = [0.0, 0.0]
    teacher_policy[teacher_index] = 1.0
    return PlannerHeadExample(
        sample_id=sample_id,
        split="train",
        fen="4k2r/8/8/8/8/8/8/R3K3 w - - 0 1",
        feature_vector=feature_vector,
        candidate_context_version=2,
        global_context_version=1,
        global_features=[0.0] * SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
        candidate_action_indices=[1, 2],
        candidate_features=candidate_features,
        proposer_scores=[0.1, 0.2],
        transition_context_version=1,
        transition_features=[[0.0] * 45, [0.0] * 45],
        reply_peak_probabilities=[0.5, 0.4],
        pressures=[0.2, 0.3],
        uncertainties=[0.3, 0.4],
        curriculum_bucket_labels=["lapv1_test"],
        curriculum_priority=1.0,
        teacher_top1_action_index=teacher_index + 1,
        teacher_top1_candidate_index=teacher_index,
        teacher_policy=teacher_policy,
        teacher_root_value_cp=teacher_cp,
        teacher_top1_minus_top2_cp=teacher_gap,
        teacher_candidate_scores_cp=[teacher_cp, teacher_cp - teacher_gap],
        teacher_candidate_score_delta_targets_cp=[0.0, -teacher_gap],
        teacher_rank_bucket_version=1,
        teacher_candidate_rank_bucket_targets=[0, 1],
        latent_state_version=None,
        latent_features=None,
    )
