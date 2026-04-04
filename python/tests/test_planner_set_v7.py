"""Tests for the cross-attention planner scorer arm."""

from __future__ import annotations

import json
from pathlib import Path

from train.config import load_planner_train_config
from train.datasets.artifacts import (
    POSITION_FEATURE_SIZE,
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
)
from train.datasets.contracts import candidate_context_feature_dim, transition_context_feature_dim
from train.models.planner import (
    PLANNER_RANK_BUCKET_COUNT,
    PlannerHeadModel,
    torch,
)


def test_set_v7_matches_set_v6_output_contract() -> None:
    assert torch is not None
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    batch_size = 2
    candidate_count = 3
    latent_feature_dim = 4

    common_kwargs = {
        "hidden_dim": 32,
        "hidden_layers": 1,
        "action_embedding_dim": 16,
        "latent_feature_dim": latent_feature_dim,
        "dropout": 0.0,
        "enable_candidate_rank_head": True,
    }
    model_v6 = PlannerHeadModel(architecture="set_v6", **common_kwargs)
    model_v7 = PlannerHeadModel(architecture="set_v7", **common_kwargs)

    root_features = torch.zeros((batch_size, POSITION_FEATURE_SIZE), dtype=torch.float32)
    global_features = torch.zeros(
        (batch_size, SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE),
        dtype=torch.float32,
    )
    candidate_action_indices = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long)
    candidate_features = torch.zeros(
        (batch_size, candidate_count, candidate_dim),
        dtype=torch.float32,
    )
    proposer_scores = torch.zeros((batch_size, candidate_count), dtype=torch.float32)
    transition_features = torch.zeros(
        (batch_size, candidate_count, transition_dim),
        dtype=torch.float32,
    )
    latent_features = torch.zeros(
        (batch_size, candidate_count, latent_feature_dim),
        dtype=torch.float32,
    )
    reply_peak_probabilities = torch.zeros((batch_size, candidate_count), dtype=torch.float32)
    pressures = torch.zeros((batch_size, candidate_count), dtype=torch.float32)
    uncertainties = torch.zeros((batch_size, candidate_count), dtype=torch.float32)
    candidate_mask = torch.tensor([[True, True, False], [True, False, False]])

    outputs_v6 = model_v6(
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
    outputs_v7 = model_v7(
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

    assert set(outputs_v6.keys()) == set(outputs_v7.keys())
    assert tuple(outputs_v7["logits"].shape) == tuple(outputs_v6["logits"].shape) == (batch_size, candidate_count)
    assert tuple(outputs_v7["candidate_score_prediction"].shape) == (batch_size, candidate_count)
    assert tuple(outputs_v7["candidate_rank_prediction"].shape) == (
        batch_size,
        candidate_count,
        PLANNER_RANK_BUCKET_COUNT,
    )
    assert tuple(outputs_v7["root_value_prediction"].shape) == (batch_size,)
    assert tuple(outputs_v7["root_gap_prediction"].shape) == (batch_size,)


def test_set_v7_backpropagates_through_cross_attention() -> None:
    assert torch is not None
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)
    model = PlannerHeadModel(
        architecture="set_v7",
        hidden_dim=32,
        hidden_layers=1,
        action_embedding_dim=16,
        latent_feature_dim=4,
        dropout=0.0,
        enable_candidate_rank_head=False,
    )

    root_features = torch.randn((2, POSITION_FEATURE_SIZE), dtype=torch.float32)
    global_features = torch.randn(
        (2, SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE),
        dtype=torch.float32,
    )
    candidate_action_indices = torch.tensor([[1, 2, 0], [3, 4, 0]], dtype=torch.long)
    candidate_features = torch.randn((2, 3, candidate_dim), dtype=torch.float32)
    proposer_scores = torch.randn((2, 3), dtype=torch.float32)
    transition_features = torch.randn((2, 3, transition_dim), dtype=torch.float32)
    latent_features = torch.randn((2, 3, 4), dtype=torch.float32)
    reply_peak_probabilities = torch.rand((2, 3), dtype=torch.float32)
    pressures = torch.randn((2, 3), dtype=torch.float32)
    uncertainties = torch.rand((2, 3), dtype=torch.float32)
    candidate_mask = torch.tensor([[True, True, False], [True, True, False]])

    outputs = model(
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
    loss = outputs["logits"][candidate_mask].sum()
    loss = loss + outputs["root_value_prediction"].sum() + outputs["root_gap_prediction"].sum()
    loss.backward()

    assert model.cross_attention is not None
    assert model.cross_attention.in_proj_weight.grad is not None
    assert float(model.cross_attention.in_proj_weight.grad.abs().sum()) > 0.0


def test_load_planner_train_config_accepts_set_v7(tmp_path: Path) -> None:
    config_path = tmp_path / "planner_set_v7.json"
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
                    "architecture": "set_v7",
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

    config = load_planner_train_config(config_path)

    assert config.model.architecture == "set_v7"
