"""Tests for runtime loading of experimental MoE planner checkpoints."""

from __future__ import annotations

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
from train.eval.planner_runtime import load_planner_head_checkpoint
from train.models.moe_planner import MoEPlannerHeadModel, torch
from train.models.planner import PLANNER_MODEL_NAME


def test_runtime_loader_accepts_moe_checkpoint(tmp_path: Path) -> None:
    assert torch is not None
    config = PlannerTrainConfig(
        seed=11,
        output_dir="planner_out",
        initial_checkpoint=None,
        data=PlannerDataConfig(
            train_path="planner_head_train.jsonl",
            validation_path="planner_head_validation.jsonl",
        ),
        curriculum=None,
        moe=MoEConfig(num_experts=4, top_k=2, load_balance_weight=0.01, expert_hidden_dim=32),
        model=PlannerModelConfig(
            architecture="moe_v1",
            hidden_dim=32,
            hidden_layers=1,
            action_embedding_dim=16,
            latent_feature_dim=0,
            dropout=0.0,
        ),
        optimization=PlannerOptimizationConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            weight_decay=0.0,
            teacher_policy_loss_weight=1.0,
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
        deliberation_steps=1,
        dropout=0.0,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        enable_complexity_head=False,
    )
    checkpoint_path = tmp_path / "moe_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": config.to_dict(),
            "model_name": PLANNER_MODEL_NAME,
        },
        checkpoint_path,
    )

    loaded_model, loaded_config = load_planner_head_checkpoint(checkpoint_path)

    assert loaded_config.model.architecture == "moe_v1"
    assert loaded_config.moe is not None
    assert hasattr(loaded_model, "router")
