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
from train.trainers import evaluate_lapv1_checkpoint, train_lapv1
from train.trainers.lapv1 import LAPv1OptimizationConfig, LAPv1TrainConfig


pytest.importorskip("torch")
pytest.importorskip("chess")


def test_train_and_evaluate_lapv1_stage1_on_tiny_cpu_dataset(tmp_path: Path) -> None:
    train_path = tmp_path / "lapv1_train.jsonl"
    validation_path = tmp_path / "lapv1_validation.jsonl"
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
        seed=5,
        output_dir=str(tmp_path / "lapv1_out"),
        stage="T1",
        data=PlannerDataConfig(
            train_path=str(train_path),
            validation_path=str(validation_path),
        ),
        model=LAPv1Config.from_mapping(
            {
                "deliberation": {"max_inner_steps": 0, "min_inner_steps": 0},
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
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            value_wdl_weight=1.0,
            value_cp_weight=0.25,
            sharpness_weight=0.1,
            policy_ce_weight=1.0,
            policy_kl_weight=0.25,
            policy_margin_weight=0.1,
            policy_rank_weight=0.1,
            intention_aux_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir=str(tmp_path / "bundle")),
    )

    run = train_lapv1(config, repo_root=tmp_path)

    checkpoint_path = Path(run.export_paths["checkpoint"])
    assert checkpoint_path.exists()
    assert Path(run.summary_path).exists()
    assert run.best_epoch == 1

    metrics = evaluate_lapv1_checkpoint(checkpoint_path, dataset_path=validation_path)

    assert metrics.total_examples == 2
    assert 0.0 <= metrics.root_top1_accuracy <= 1.0
    assert 0.0 <= metrics.root_top3_accuracy <= 1.0
    assert metrics.total_loss >= 0.0


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
