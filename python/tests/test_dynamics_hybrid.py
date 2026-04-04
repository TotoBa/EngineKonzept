"""Tests for the hybrid symbolic-residual dynamics arm."""

from __future__ import annotations

import json
from pathlib import Path

from train.config import load_dynamics_train_config
from train.datasets.artifacts import (
    DynamicsTrainingExample,
    PIECE_FEATURE_SIZE,
    POSITION_FEATURE_SIZE,
    RULE_FEATURE_SIZE,
    SQUARE_FEATURE_SIZE,
    SYMBOLIC_MOVE_DELTA_FEATURE_SIZE,
)
from train.models.dynamics import LatentDynamicsModel, torch


def test_hybrid_v1_forward_pass_shape() -> None:
    assert torch is not None
    model = LatentDynamicsModel(
        architecture="hybrid_v1",
        latent_dim=16,
        hidden_dim=32,
        hidden_layers=2,
        action_embedding_dim=8,
        dropout=0.0,
    )
    features = torch.zeros((2, POSITION_FEATURE_SIZE), dtype=torch.float32)
    action_indices = torch.tensor([1, 2], dtype=torch.long)
    symbolic_move_delta_features = torch.zeros(
        (2, SYMBOLIC_MOVE_DELTA_FEATURE_SIZE),
        dtype=torch.float32,
    )
    symbolic_next_features = torch.zeros((2, POSITION_FEATURE_SIZE), dtype=torch.float32)

    prediction = model.predict(
        features,
        action_indices,
        symbolic_move_delta_features=symbolic_move_delta_features,
        symbolic_next_features=symbolic_next_features,
    )

    assert tuple(prediction.next_features.shape) == (2, POSITION_FEATURE_SIZE)
    assert tuple(prediction.piece_features.shape) == (2, PIECE_FEATURE_SIZE)
    assert tuple(prediction.square_features.shape) == (2, SQUARE_FEATURE_SIZE)
    assert tuple(prediction.rule_features.shape) == (2, RULE_FEATURE_SIZE)
    assert tuple(prediction.piece_delta_features.shape) == (2, PIECE_FEATURE_SIZE)
    assert tuple(prediction.square_delta_features.shape) == (2, SQUARE_FEATURE_SIZE)
    assert tuple(prediction.rule_delta_features.shape) == (2, RULE_FEATURE_SIZE)


def test_hybrid_v1_adds_residual_to_symbolic_prediction() -> None:
    assert torch is not None
    model = LatentDynamicsModel(
        architecture="hybrid_v1",
        latent_dim=8,
        hidden_dim=16,
        hidden_layers=1,
        action_embedding_dim=4,
        dropout=0.0,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.piece_decoder[-1].bias.fill_(0.5)
        model.square_decoder[-1].bias.fill_(1.0)
        model.rule_decoder[-1].bias.fill_(-0.25)

    features = torch.zeros((1, POSITION_FEATURE_SIZE), dtype=torch.float32)
    action_indices = torch.tensor([0], dtype=torch.long)
    symbolic_next_features = torch.cat(
        (
            torch.full((1, PIECE_FEATURE_SIZE), 1.0, dtype=torch.float32),
            torch.full((1, SQUARE_FEATURE_SIZE), 2.0, dtype=torch.float32),
            torch.full((1, RULE_FEATURE_SIZE), 3.0, dtype=torch.float32),
        ),
        dim=1,
    )

    prediction = model.predict(
        features,
        action_indices,
        symbolic_move_delta_features=torch.zeros(
            (1, SYMBOLIC_MOVE_DELTA_FEATURE_SIZE),
            dtype=torch.float32,
        ),
        symbolic_next_features=symbolic_next_features,
    )

    assert torch.allclose(
        prediction.piece_features,
        torch.full((1, PIECE_FEATURE_SIZE), 1.5, dtype=torch.float32),
    )
    assert torch.allclose(
        prediction.square_features,
        torch.full((1, SQUARE_FEATURE_SIZE), 3.0, dtype=torch.float32),
    )
    assert torch.allclose(
        prediction.rule_features,
        torch.full((1, RULE_FEATURE_SIZE), 2.75, dtype=torch.float32),
    )


def test_dynamics_training_example_accepts_missing_symbolic_delta_fields() -> None:
    example = DynamicsTrainingExample.from_dict(
        {
            "sample_id": "sample-1",
            "split": "train",
            "feature_vector": [0.0] * POSITION_FEATURE_SIZE,
            "action_index": 7,
            "action_candidate_context_version": 1,
            "action_features": [0.0] * 18,
            "next_feature_vector": [1.0] * POSITION_FEATURE_SIZE,
            "is_capture": False,
            "is_promotion": False,
            "is_castle": False,
            "is_en_passant": False,
            "gives_check": False,
            "trajectory_id": "traj-1",
            "ply_index": 2,
        }
    )

    assert example.symbolic_move_delta_version is None
    assert example.symbolic_move_delta_features is None

    assert torch is not None
    model = LatentDynamicsModel(
        architecture="hybrid_v1",
        latent_dim=8,
        hidden_dim=16,
        hidden_layers=1,
        action_embedding_dim=4,
        dropout=0.0,
    )
    next_features = model(
        torch.zeros((1, POSITION_FEATURE_SIZE), dtype=torch.float32),
        torch.tensor([example.action_index], dtype=torch.long),
    )
    assert tuple(next_features.shape) == (1, POSITION_FEATURE_SIZE)


def test_load_dynamics_train_config_accepts_hybrid_v1(tmp_path: Path) -> None:
    config_path = tmp_path / "config_hybrid.json"
    config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": "artifacts/phase6/test-hybrid",
                "data": {
                    "dataset_path": "artifacts/datasets/test",
                    "train_split": "train",
                    "validation_split": "validation",
                },
                "model": {
                    "architecture": "hybrid_v1",
                    "latent_dim": 64,
                    "hidden_dim": 128,
                    "hidden_layers": 2,
                    "action_embedding_dim": 32,
                    "dropout": 0.1,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "reconstruction_loss_weight": 1.0,
                    "piece_loss_weight": 1.0,
                    "square_loss_weight": 1.0,
                    "rule_loss_weight": 1.0,
                },
                "evaluation": {"drift_horizon": 2},
                "runtime": {"torch_threads": 0, "dataloader_workers": 0},
                "export": {
                    "bundle_dir": "models/dynamics/test-hybrid",
                    "checkpoint_name": "checkpoint.pt",
                    "exported_program_name": "dynamics.pt2",
                    "metadata_name": "metadata.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_dynamics_train_config(config_path)

    assert config.model.architecture == "hybrid_v1"
