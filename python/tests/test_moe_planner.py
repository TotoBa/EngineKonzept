"""Tests for the experimental MoE planner arm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from train.config import PlannerTrainConfig, load_planner_train_config
from train.datasets.artifacts import (
    POSITION_FEATURE_SIZE,
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
)
from train.datasets.contracts import candidate_context_feature_dim, transition_context_feature_dim
from train.models.moe_planner import (
    ComplexityHead,
    MoEPlannerHeadModel,
    PositionRouter,
    compute_expert_utilization_metrics,
    torch,
)
from train.models.planner import PLANNER_RANK_BUCKET_COUNT, PlannerHeadModel


def test_moe_v1_forward_matches_set_v6_output_shapes() -> None:
    assert torch is not None
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    batch_size = 2
    candidate_count = 3

    moe_model = MoEPlannerHeadModel(
        hidden_dim=32,
        hidden_layers=1,
        action_embedding_dim=16,
        latent_feature_dim=4,
        dropout=0.0,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        enable_candidate_rank_head=True,
    )
    baseline = PlannerHeadModel(
        architecture="set_v6",
        hidden_dim=32,
        hidden_layers=1,
        action_embedding_dim=16,
        latent_feature_dim=4,
        dropout=0.0,
        enable_candidate_rank_head=True,
    )

    root_features = torch.zeros((batch_size, POSITION_FEATURE_SIZE), dtype=torch.float32)
    global_features = torch.zeros(
        (batch_size, SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE),
        dtype=torch.float32,
    )
    candidate_action_indices = torch.tensor([[1, 2, 0], [3, 4, 0]], dtype=torch.long)
    candidate_features = torch.zeros((batch_size, candidate_count, candidate_dim), dtype=torch.float32)
    proposer_scores = torch.zeros((batch_size, candidate_count), dtype=torch.float32)
    transition_features = torch.zeros(
        (batch_size, candidate_count, transition_dim),
        dtype=torch.float32,
    )
    latent_features = torch.zeros((batch_size, candidate_count, 4), dtype=torch.float32)
    reply_peak_probabilities = torch.zeros((batch_size, candidate_count), dtype=torch.float32)
    pressures = torch.zeros((batch_size, candidate_count), dtype=torch.float32)
    uncertainties = torch.zeros((batch_size, candidate_count), dtype=torch.float32)
    candidate_mask = torch.tensor([[True, True, False], [True, True, False]])

    baseline_outputs = baseline(
        root_features,
        global_features,
        candidate_action_indices,
        candidate_features,
        proposer_scores,
        transition_features,
        latent_features,
        reply_peak_probabilities,
        pressures,
        uncertainties,
        candidate_mask,
    )
    moe_outputs = moe_model(
        root_features,
        global_features,
        candidate_action_indices,
        candidate_features,
        proposer_scores,
        transition_features,
        latent_features,
        reply_peak_probabilities,
        pressures,
        uncertainties,
        candidate_mask,
    )

    assert tuple(moe_outputs["logits"].shape) == tuple(baseline_outputs["logits"].shape)
    assert tuple(moe_outputs["candidate_score_prediction"].shape) == tuple(
        baseline_outputs["candidate_score_prediction"].shape
    )
    assert tuple(moe_outputs["candidate_rank_prediction"].shape) == (
        batch_size,
        candidate_count,
        PLANNER_RANK_BUCKET_COUNT,
    )
    assert tuple(moe_outputs["root_value_prediction"].shape) == tuple(
        baseline_outputs["root_value_prediction"].shape
    )
    assert tuple(moe_outputs["root_gap_prediction"].shape) == tuple(
        baseline_outputs["root_gap_prediction"].shape
    )


def test_position_router_weights_sum_to_one_and_respect_top_k() -> None:
    assert torch is not None
    router = PositionRouter(hidden_dim=16, num_experts=4, top_k=2, dropout=0.0)
    state_embedding = torch.randn((5, 16), dtype=torch.float32)

    outputs = router(state_embedding)

    assert torch.allclose(
        outputs["router_weights"].sum(dim=1),
        torch.ones(5, dtype=torch.float32),
    )
    assert torch.allclose(
        outputs["sparse_router_weights"].sum(dim=1),
        torch.ones(5, dtype=torch.float32),
    )
    assert torch.equal(
        torch.count_nonzero(outputs["sparse_router_weights"], dim=1),
        torch.full((5,), 2, dtype=torch.int64),
    )


def test_moe_load_balance_loss_is_finite_and_non_negative() -> None:
    assert torch is not None
    router = PositionRouter(hidden_dim=16, num_experts=4, top_k=2, dropout=0.0)
    outputs = router(torch.randn((6, 16), dtype=torch.float32))

    assert torch.isfinite(outputs["load_balance_loss"])
    assert float(outputs["load_balance_loss"].item()) >= 0.0


def test_complexity_head_outputs_unit_interval_scores() -> None:
    assert torch is not None
    head = ComplexityHead(hidden_dim=16)
    scores = head(torch.randn((6, 16), dtype=torch.float32))

    assert tuple(scores.shape) == (6,)
    assert bool(torch.all(scores >= 0.0))
    assert bool(torch.all(scores <= 1.0))


def test_expert_utilization_metrics_are_computed() -> None:
    assert torch is not None
    router_weights = torch.tensor(
        [
            [0.7, 0.2, 0.1, 0.0],
            [0.1, 0.6, 0.3, 0.0],
        ],
        dtype=torch.float32,
    )
    sparse_router_weights = torch.tensor(
        [
            [0.7777778, 0.2222222, 0.0, 0.0],
            [0.0, 0.6666667, 0.3333333, 0.0],
        ],
        dtype=torch.float32,
    )

    metrics = compute_expert_utilization_metrics(router_weights, sparse_router_weights)

    assert tuple(metrics["expert_activation_counts"].tolist()) == (1.0, 2.0, 1.0, 0.0)
    assert tuple(round(value, 6) for value in metrics["expert_activation_frequencies"].tolist()) == (
        0.5,
        1.0,
        0.5,
        0.0,
    )
    assert float(metrics["router_entropy"].item()) >= 0.0


def test_complexity_routing_uses_fewer_experts_for_easy_positions() -> None:
    assert torch is not None

    class FixedComplexityHead(torch.nn.Module):
        def __init__(self, scores: Any) -> None:
            super().__init__()
            self.scores = scores

        def forward(self, _state_embedding: Any) -> Any:
            return self.scores

    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    model = MoEPlannerHeadModel(
        hidden_dim=32,
        hidden_layers=1,
        action_embedding_dim=16,
        latent_feature_dim=4,
        deliberation_steps=3,
        dropout=0.0,
        num_experts=4,
        top_k=3,
        expert_hidden_dim=32,
        enable_complexity_head=True,
        easy_threshold=0.3,
        hard_threshold=0.7,
    )
    model.complexity_head = FixedComplexityHead(
        torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    )

    batch_size = 3
    candidate_count = 4
    outputs = model(
        torch.zeros((batch_size, POSITION_FEATURE_SIZE), dtype=torch.float32),
        torch.zeros((batch_size, SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE), dtype=torch.float32),
        torch.tensor([[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0]], dtype=torch.long),
        torch.zeros((batch_size, candidate_count, candidate_dim), dtype=torch.float32),
        torch.zeros((batch_size, candidate_count), dtype=torch.float32),
        torch.zeros((batch_size, candidate_count, transition_dim), dtype=torch.float32),
        torch.zeros((batch_size, candidate_count, 4), dtype=torch.float32),
        torch.zeros((batch_size, candidate_count), dtype=torch.float32),
        torch.zeros((batch_size, candidate_count), dtype=torch.float32),
        torch.zeros((batch_size, candidate_count), dtype=torch.float32),
        torch.ones((batch_size, candidate_count), dtype=torch.bool),
    )

    assert tuple(outputs["selected_expert_counts"].tolist()) == (1, 2, 3)


def test_complexity_threshold_boundaries_route_middle_band_to_medium() -> None:
    assert torch is not None

    class FixedComplexityHead(torch.nn.Module):
        def __init__(self, scores: Any) -> None:
            super().__init__()
            self.scores = scores

        def forward(self, _state_embedding: Any) -> Any:
            return self.scores

    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    model = MoEPlannerHeadModel(
        hidden_dim=32,
        hidden_layers=1,
        action_embedding_dim=16,
        latent_feature_dim=4,
        deliberation_steps=2,
        dropout=0.0,
        num_experts=4,
        top_k=3,
        expert_hidden_dim=32,
        enable_complexity_head=True,
        easy_threshold=0.3,
        hard_threshold=0.7,
    )
    model.complexity_head = FixedComplexityHead(
        torch.tensor([0.29, 0.3, 0.7, 0.71], dtype=torch.float32)
    )

    outputs = model(
        torch.zeros((4, POSITION_FEATURE_SIZE), dtype=torch.float32),
        torch.zeros((4, SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE), dtype=torch.float32),
        torch.tensor([[1, 2], [1, 2], [1, 2], [1, 2]], dtype=torch.long),
        torch.zeros((4, 2, candidate_dim), dtype=torch.float32),
        torch.zeros((4, 2), dtype=torch.float32),
        torch.zeros((4, 2, transition_dim), dtype=torch.float32),
        torch.zeros((4, 2, 4), dtype=torch.float32),
        torch.zeros((4, 2), dtype=torch.float32),
        torch.zeros((4, 2), dtype=torch.float32),
        torch.zeros((4, 2), dtype=torch.float32),
        torch.ones((4, 2), dtype=torch.bool),
    )

    assert tuple(outputs["selected_expert_counts"].tolist()) == (1, 2, 2, 3)
    assert tuple(outputs["complexity_tier_indices"].tolist()) == (0, 1, 1, 2)


def test_moe_without_complexity_head_keeps_vanilla_top_k_behavior() -> None:
    assert torch is not None
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    model = MoEPlannerHeadModel(
        hidden_dim=32,
        hidden_layers=1,
        action_embedding_dim=16,
        latent_feature_dim=4,
        deliberation_steps=3,
        dropout=0.0,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=32,
        enable_complexity_head=False,
    )

    outputs = model(
        torch.zeros((3, POSITION_FEATURE_SIZE), dtype=torch.float32),
        torch.zeros((3, SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE), dtype=torch.float32),
        torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
        torch.zeros((3, 2, candidate_dim), dtype=torch.float32),
        torch.zeros((3, 2), dtype=torch.float32),
        torch.zeros((3, 2, transition_dim), dtype=torch.float32),
        torch.zeros((3, 2, 4), dtype=torch.float32),
        torch.zeros((3, 2), dtype=torch.float32),
        torch.zeros((3, 2), dtype=torch.float32),
        torch.zeros((3, 2), dtype=torch.float32),
        torch.ones((3, 2), dtype=torch.bool),
    )

    assert outputs["complexity_score"] is None
    assert tuple(outputs["selected_expert_counts"].tolist()) == (2, 2, 2)


def test_moe_config_validation_accepts_and_rejects_correctly(tmp_path: Path) -> None:
    accepted_path = tmp_path / "planner_moe_ok.json"
    accepted_path.write_text(
        json.dumps(
            {
                "seed": 3,
                "output_dir": "planner_out",
                "data": {
                    "train_path": "planner_head_train.jsonl",
                    "validation_path": "planner_head_validation.jsonl",
                },
                "moe": {
                    "num_experts": 4,
                    "top_k": 2,
                    "load_balance_weight": 0.01,
                    "expert_hidden_dim": 64,
                    "enable_complexity_head": True,
                    "easy_threshold": 0.25,
                    "hard_threshold": 0.75,
                    "complexity_loss_weight": 0.05,
                },
                "model": {
                    "architecture": "moe_v1",
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
    accepted = load_planner_train_config(accepted_path)
    assert accepted.model.architecture == "moe_v1"
    assert accepted.moe is not None
    assert accepted.moe.top_k == 2
    assert accepted.moe.enable_complexity_head is True

    with pytest.raises(ValueError, match="moe settings are required"):
        PlannerTrainConfig.from_dict(
            {
                "seed": 3,
                "output_dir": "planner_out",
                "data": {
                    "train_path": "planner_head_train.jsonl",
                    "validation_path": "planner_head_validation.jsonl",
                },
                "model": {
                    "architecture": "moe_v1",
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
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {"bundle_dir": "planner_bundle"},
            }
        )

    with pytest.raises(ValueError, match="easy_threshold must be smaller"):
        PlannerTrainConfig.from_dict(
            {
                "seed": 3,
                "output_dir": "planner_out",
                "data": {
                    "train_path": "planner_head_train.jsonl",
                    "validation_path": "planner_head_validation.jsonl",
                },
                "moe": {
                    "num_experts": 4,
                    "top_k": 2,
                    "load_balance_weight": 0.01,
                    "expert_hidden_dim": 64,
                    "enable_complexity_head": True,
                    "easy_threshold": 0.8,
                    "hard_threshold": 0.7,
                    "complexity_loss_weight": 0.05,
                },
                "model": {
                    "architecture": "moe_v1",
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
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {"bundle_dir": "planner_bundle"},
            }
        )

    with pytest.raises(ValueError, match="only valid when model.architecture='moe_v1'"):
        PlannerTrainConfig.from_dict(
            {
                "seed": 3,
                "output_dir": "planner_out",
                "data": {
                    "train_path": "planner_head_train.jsonl",
                    "validation_path": "planner_head_validation.jsonl",
                },
                "moe": {
                    "num_experts": 4,
                    "top_k": 2,
                    "load_balance_weight": 0.01,
                    "expert_hidden_dim": 64,
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
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {"bundle_dir": "planner_bundle"},
            }
        )

    with pytest.raises(ValueError, match="must not exceed moe.num_experts"):
        PlannerTrainConfig.from_dict(
            {
                "seed": 3,
                "output_dir": "planner_out",
                "data": {
                    "train_path": "planner_head_train.jsonl",
                    "validation_path": "planner_head_validation.jsonl",
                },
                "moe": {
                    "num_experts": 2,
                    "top_k": 3,
                    "load_balance_weight": 0.01,
                    "expert_hidden_dim": 64,
                },
                "model": {
                    "architecture": "moe_v1",
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
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {"bundle_dir": "planner_bundle"},
            }
        )
