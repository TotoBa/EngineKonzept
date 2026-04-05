from __future__ import annotations

import pytest

from train.config import StateEmbedderConfig
from train.datasets.artifacts import PIECE_TOKEN_CAPACITY, SQUARE_TOKEN_COUNT
from train.models.intention_encoder import STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM, torch
from train.models.state_embedder import RelationalStateEmbedder


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _sample_piece_intentions(*, batch_size: int = 2, intention_dim: int = 64) -> torch.Tensor:
    piece_intentions = torch.zeros(
        (batch_size, PIECE_TOKEN_CAPACITY, intention_dim),
        dtype=torch.float32,
    )
    piece_intentions[:, :5, :] = torch.randn((batch_size, 5, intention_dim), dtype=torch.float32)
    return piece_intentions


def _sample_square_tokens(*, batch_size: int = 2, square_dim: int = 2) -> torch.Tensor:
    square_indices = torch.arange(SQUARE_TOKEN_COUNT, dtype=torch.float32).reshape(1, -1, 1)
    occupant_code = torch.zeros((1, SQUARE_TOKEN_COUNT, 1), dtype=torch.float32)
    tokens = torch.cat([square_indices / 63.0, occupant_code], dim=2)
    if square_dim > 2:
        extra = torch.zeros((1, SQUARE_TOKEN_COUNT, square_dim - 2), dtype=torch.float32)
        tokens = torch.cat([tokens, extra], dim=2)
    return tokens.repeat(batch_size, 1, 1)


def _sample_global_features(*, batch_size: int = 2) -> torch.Tensor:
    values = torch.linspace(
        0.0,
        1.0,
        STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM,
        dtype=torch.float32,
    )
    return values.unsqueeze(0).repeat(batch_size, 1)


def _sample_reachability_edges(*, batch_size: int = 2, edge_count: int = 16) -> torch.Tensor:
    edges = torch.full((batch_size, edge_count, 3), -1, dtype=torch.long)
    base_edges = torch.tensor(
        [
            [4, 12, 5],
            [0, 8, 3],
            [7, 15, 1],
            [60, 52, 5],
            [63, 55, 3],
            [12, 20, 5],
            [52, 44, 5],
        ],
        dtype=torch.long,
    )
    edges[:, : base_edges.shape[0], :] = base_edges
    return edges


def test_state_embedder_forward_shapes_are_correct() -> None:
    model = RelationalStateEmbedder()
    z_root, sigma_root = model(
        _sample_piece_intentions(),
        _sample_square_tokens(),
        _sample_global_features(),
        _sample_reachability_edges(),
    )

    assert tuple(z_root.shape) == (2, 512)
    assert tuple(sigma_root.shape) == (2, 1)


def test_state_embedder_sigma_root_is_positive() -> None:
    model = RelationalStateEmbedder()
    _z_root, sigma_root = model(
        _sample_piece_intentions(batch_size=1),
        _sample_square_tokens(batch_size=1),
        _sample_global_features(batch_size=1),
        _sample_reachability_edges(batch_size=1),
    )

    assert bool(torch.all(sigma_root > 0.0))


def test_state_embedder_backward_populates_layer_gradients() -> None:
    model = RelationalStateEmbedder()
    z_root, sigma_root = model(
        _sample_piece_intentions(),
        _sample_square_tokens(),
        _sample_global_features(),
        _sample_reachability_edges(),
    )
    (z_root.square().sum() + sigma_root.sum()).backward()

    assert model.piece_projection[0].weight.grad is not None
    assert model.square_projection[0].weight.grad is not None
    assert model.layers[0].self_attention.in_proj_weight.grad is not None
    assert model.layers[0].edge_attention.in_proj_weight.grad is not None
    assert model.layers[-1].feedforward[0].weight.grad is not None
    assert model.state_projection[0].weight.grad is not None
    assert model.sigma_head[0].weight.grad is not None


def test_state_embedder_parameter_budget_is_close_to_forty_megabytes() -> None:
    model = RelationalStateEmbedder()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = parameter_count * 4
    target_bytes = 40 * 1024 * 1024
    lower_bound = int(target_bytes * 0.7)
    upper_bound = int(target_bytes * 1.3)

    assert lower_bound <= parameter_bytes <= upper_bound


def test_state_embedder_config_validates_defaults() -> None:
    config = StateEmbedderConfig()

    assert config.hidden_dim == 256
    assert config.state_dim == 512
    assert config.num_layers == 6
    assert config.num_heads == 8
