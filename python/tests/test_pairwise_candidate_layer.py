"""Tests for optional pairwise planner candidate refinement."""

from __future__ import annotations

import json
from pathlib import Path

from train.config import load_planner_train_config
from train.datasets.artifacts import (
    POSITION_FEATURE_SIZE,
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
)
from train.datasets.contracts import candidate_context_feature_dim, transition_context_feature_dim
from train.models.planner import PairwiseCandidateLayer, PlannerHeadModel, torch


def test_pairwise_candidate_layer_preserves_shape() -> None:
    assert torch is not None
    layer = PairwiseCandidateLayer(hidden_dim=16, dropout=0.0)
    candidate_embeddings = torch.randn((2, 4, 16), dtype=torch.float32)
    candidate_mask = torch.tensor([[True, True, True, False], [True, False, False, False]])

    refined = layer(candidate_embeddings, candidate_mask)

    assert tuple(refined.shape) == (2, 4, 16)


def test_pairwise_candidate_layer_zeroes_masked_candidates() -> None:
    assert torch is not None
    layer = PairwiseCandidateLayer(hidden_dim=8, dropout=0.0)
    candidate_embeddings = torch.randn((1, 3, 8), dtype=torch.float32)
    candidate_mask = torch.tensor([[True, False, False]])

    refined = layer(candidate_embeddings, candidate_mask)

    assert torch.allclose(refined[:, 1:, :], torch.zeros((1, 2, 8), dtype=torch.float32))


def test_planner_without_pairwise_matches_shared_weights_baseline() -> None:
    assert torch is not None
    candidate_dim = candidate_context_feature_dim(2)
    transition_dim = transition_context_feature_dim(1)

    torch.manual_seed(1234)
    baseline = PlannerHeadModel(
        architecture="set_v6",
        hidden_dim=32,
        hidden_layers=1,
        action_embedding_dim=16,
        latent_feature_dim=4,
        dropout=0.0,
        enable_pairwise_candidates=False,
        enable_candidate_rank_head=False,
    )
    torch.manual_seed(1234)
    with_pairwise = PlannerHeadModel(
        architecture="set_v6",
        hidden_dim=32,
        hidden_layers=1,
        action_embedding_dim=16,
        latent_feature_dim=4,
        dropout=0.0,
        enable_pairwise_candidates=True,
        enable_candidate_rank_head=False,
    )
    with_pairwise.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad():
        for parameter in with_pairwise.pairwise_candidate_layer.parameters():
            parameter.zero_()

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
    candidate_mask = torch.tensor([[True, True, False], [True, False, False]])

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
    pairwise_outputs = with_pairwise(
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

    assert torch.allclose(pairwise_outputs["logits"], baseline_outputs["logits"])
    assert torch.allclose(
        pairwise_outputs["candidate_score_prediction"],
        baseline_outputs["candidate_score_prediction"],
    )
    assert torch.allclose(
        pairwise_outputs["root_value_prediction"],
        baseline_outputs["root_value_prediction"],
    )
    assert torch.allclose(
        pairwise_outputs["root_gap_prediction"],
        baseline_outputs["root_gap_prediction"],
    )


def test_load_planner_train_config_accepts_pairwise_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "planner_pairwise.json"
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
                    "enable_pairwise_candidates": True,
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

    assert config.model.enable_pairwise_candidates is True
