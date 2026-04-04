"""Tests for the offline MoE specialization analysis workflow."""

from __future__ import annotations

import json
from pathlib import Path

from train.config import (
    MoEConfig,
    PlannerDataConfig,
    PlannerEvaluationConfig,
    PlannerExportConfig,
    PlannerModelConfig,
    PlannerOptimizationConfig,
    PlannerRuntimeConfig,
    PlannerTrainConfig,
)
from train.datasets.artifacts import POSITION_FEATURE_SIZE, SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE
from train.datasets.planner_head import PlannerHeadExample, write_planner_head_artifact
from train.eval.moe_analysis import (
    analyze_moe_expert_specialization,
    visualize_moe_routing_report,
)
from train.models.moe_planner import MoEPlannerHeadModel, torch
from train.models.planner import PLANNER_MODEL_NAME


def test_moe_analysis_and_visualization_workflow(tmp_path: Path) -> None:
    assert torch is not None
    checkpoint_path = tmp_path / "moe_checkpoint.pt"
    dataset_path = tmp_path / "planner_head_test.jsonl"
    report_path = tmp_path / "moe_report.json"
    plot_dir = tmp_path / "plots"

    config = PlannerTrainConfig(
        seed=7,
        output_dir="planner_out",
        initial_checkpoint=None,
        data=PlannerDataConfig(
            train_path="planner_head_train.jsonl",
            validation_path="planner_head_validation.jsonl",
        ),
        curriculum=None,
        moe=MoEConfig(
            num_experts=4,
            top_k=2,
            load_balance_weight=0.01,
            expert_hidden_dim=32,
            enable_complexity_head=True,
            easy_threshold=0.3,
            hard_threshold=0.7,
            complexity_loss_weight=0.05,
        ),
        model=PlannerModelConfig(
            architecture="moe_v1",
            hidden_dim=32,
            hidden_layers=1,
            action_embedding_dim=16,
            latent_feature_dim=0,
            deliberation_steps=2,
            dropout=0.0,
        ),
        optimization=PlannerOptimizationConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            weight_decay=0.0,
            teacher_policy_loss_weight=1.0,
            teacher_kl_loss_weight=0.25,
            teacher_score_loss_weight=0.1,
            teacher_margin_loss_weight=0.0,
            teacher_rank_loss_weight=0.0,
            curriculum_priority_weight=0.0,
            root_value_loss_weight=0.05,
            root_gap_loss_weight=0.05,
        ),
        evaluation=PlannerEvaluationConfig(top_k=3),
        runtime=PlannerRuntimeConfig(torch_threads=1, dataloader_workers=0),
        export=PlannerExportConfig(bundle_dir="planner_bundle", checkpoint_name="checkpoint.pt"),
    )
    model = MoEPlannerHeadModel(
        hidden_dim=32,
        hidden_layers=1,
        action_embedding_dim=16,
        latent_feature_dim=0,
        deliberation_steps=2,
        dropout=0.0,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        enable_complexity_head=True,
        easy_threshold=0.3,
        hard_threshold=0.7,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": config.to_dict(),
            "model_name": PLANNER_MODEL_NAME,
        },
        checkpoint_path,
    )

    examples = [
        PlannerHeadExample(
            sample_id="opening_example",
            split="test",
            fen="rnbqkbnr/pppp1ppp/8/4p3/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
            feature_vector=[0.0] * POSITION_FEATURE_SIZE,
            candidate_context_version=2,
            global_context_version=1,
            global_features=[0.0] * SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
            candidate_action_indices=[1, 2, 3],
            candidate_features=[[0.0] * 35 for _ in range(3)],
            proposer_scores=[0.2, 0.1, -0.1],
            transition_context_version=1,
            transition_features=[[0.0] * 45 for _ in range(3)],
            reply_peak_probabilities=[0.3, 0.2, 0.1],
            pressures=[0.0, 0.0, 0.0],
            uncertainties=[0.1, 0.2, 0.3],
            curriculum_bucket_labels=["easy"],
            curriculum_priority=0.1,
            teacher_top1_action_index=1,
            teacher_top1_candidate_index=0,
            teacher_policy=[0.8, 0.15, 0.05],
            teacher_root_value_cp=60.0,
            teacher_top1_minus_top2_cp=150.0,
            teacher_candidate_scores_cp=[60.0, 10.0, -20.0],
        ),
        PlannerHeadExample(
            sample_id="endgame_example",
            split="test",
            fen="8/8/3k4/8/3K4/8/2P5/8 w - - 0 1",
            feature_vector=[0.0] * POSITION_FEATURE_SIZE,
            candidate_context_version=2,
            global_context_version=1,
            global_features=[0.0] * SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
            candidate_action_indices=[4, 5, 6],
            candidate_features=[[0.0] * 35 for _ in range(3)],
            proposer_scores=[0.0, 0.0, 0.0],
            transition_context_version=1,
            transition_features=[[0.0] * 45 for _ in range(3)],
            reply_peak_probabilities=[0.2, 0.2, 0.2],
            pressures=[0.0, 0.0, 0.0],
            uncertainties=[0.4, 0.5, 0.6],
            curriculum_bucket_labels=["hard"],
            curriculum_priority=0.9,
            teacher_top1_action_index=4,
            teacher_top1_candidate_index=0,
            teacher_policy=[0.5, 0.3, 0.2],
            teacher_root_value_cp=5.0,
            teacher_top1_minus_top2_cp=8.0,
            teacher_candidate_scores_cp=[5.0, 2.0, 0.0],
        ),
    ]
    write_planner_head_artifact(dataset_path, examples)

    report = analyze_moe_expert_specialization(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        output_path=report_path,
        batch_size=2,
    )

    assert report["example_count"] == 2
    assert report["num_experts"] == 4
    assert "opening" in report["expert_activation_by_phase"]
    assert "endgame" in report["expert_activation_by_phase"]
    assert "quiet" in report["expert_activation_by_tactical_level"]
    assert "easy" in report["expert_activation_by_difficulty"]
    assert report["router_entropy_distribution"]["count"] == 2
    assert len(report["example_records"]) == 2

    plot_paths = visualize_moe_routing_report(report=report, output_dir=plot_dir)
    assert len(plot_paths) == 3
    assert all(path.exists() for path in plot_paths)

    reloaded_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert reloaded_report["example_count"] == 2
