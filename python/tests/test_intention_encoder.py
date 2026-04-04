from __future__ import annotations

import pytest

from train.config import IntentionEncoderConfig
from train.datasets.artifacts import PIECE_TOKEN_CAPACITY
from train.models.intention_encoder import (
    STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM,
    KingSpecialHead,
    PieceIntentionEncoder,
    torch,
)


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _sample_piece_tokens(*, batch_size: int = 2) -> torch.Tensor:
    piece_tokens = torch.full(
        (batch_size, PIECE_TOKEN_CAPACITY, 3),
        -1,
        dtype=torch.long,
    )
    piece_tokens[:, 0] = torch.tensor([4, 0, 5], dtype=torch.long)
    piece_tokens[:, 1] = torch.tensor([0, 0, 3], dtype=torch.long)
    piece_tokens[:, 2] = torch.tensor([7, 0, 1], dtype=torch.long)
    piece_tokens[:, 3] = torch.tensor([60, 1, 5], dtype=torch.long)
    piece_tokens[:, 4] = torch.tensor([63, 1, 3], dtype=torch.long)
    return piece_tokens


def _sample_global_features(*, batch_size: int = 2) -> torch.Tensor:
    values = torch.linspace(
        0.0,
        1.0,
        STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM,
        dtype=torch.float32,
    )
    return values.unsqueeze(0).repeat(batch_size, 1)


def _sample_reachability_edges(*, batch_size: int = 2, edge_count: int = 8) -> torch.Tensor:
    edges = torch.full((batch_size, edge_count, 3), -1, dtype=torch.long)
    base_edges = torch.tensor(
        [
            [4, 12, 5],
            [0, 8, 3],
            [7, 15, 1],
            [60, 52, 5],
            [63, 55, 3],
            [4, 0, 5],
        ],
        dtype=torch.long,
    )
    edges[:, : base_edges.shape[0], :] = base_edges
    return edges


def test_intention_encoder_forward_shapes_are_correct() -> None:
    model = PieceIntentionEncoder()
    piece_intentions = model(
        _sample_piece_tokens(),
        _sample_global_features(),
        _sample_reachability_edges(),
    )
    king_head = KingSpecialHead()
    own_king_will, opp_king_will = king_head(
        piece_intentions[:, 0, :],
        piece_intentions[:, 3, :],
        _sample_global_features(),
    )

    assert tuple(piece_intentions.shape) == (2, PIECE_TOKEN_CAPACITY, 64)
    assert tuple(own_king_will.shape) == (2, 64)
    assert tuple(opp_king_will.shape) == (2, 64)


def test_padding_rows_are_zeroed_after_attention() -> None:
    model = PieceIntentionEncoder()
    piece_intentions = model(
        _sample_piece_tokens(batch_size=1),
        _sample_global_features(batch_size=1),
        _sample_reachability_edges(batch_size=1),
    )

    assert torch.count_nonzero(piece_intentions[0, 5:, :]) == 0
    assert torch.count_nonzero(piece_intentions[0, :5, :]) > 0


def test_intention_encoder_is_deterministic_for_fixed_seed() -> None:
    torch.manual_seed(1234)
    first = PieceIntentionEncoder()
    torch.manual_seed(1234)
    second = PieceIntentionEncoder()

    piece_tokens = _sample_piece_tokens()
    global_features = _sample_global_features()
    reachability_edges = _sample_reachability_edges()

    first_outputs = first(piece_tokens, global_features, reachability_edges)
    second_outputs = second(piece_tokens, global_features, reachability_edges)

    assert torch.allclose(first_outputs, second_outputs)


def test_backward_pass_populates_gradients() -> None:
    model = PieceIntentionEncoder()
    outputs = model(
        _sample_piece_tokens(),
        _sample_global_features(),
        _sample_reachability_edges(),
    )
    loss = outputs.square().sum()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert any(gradient is not None for gradient in gradients)
    assert any(
        gradient is not None and torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
    )


def test_intention_encoder_parameter_budget_is_close_to_twenty_megabytes() -> None:
    model = PieceIntentionEncoder()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = parameter_count * 4
    target_bytes = 20 * 1024 * 1024
    lower_bound = int(target_bytes * 0.7)
    upper_bound = int(target_bytes * 1.3)

    assert lower_bound <= parameter_bytes <= upper_bound


def test_intention_encoder_config_validates_defaults() -> None:
    config = IntentionEncoderConfig()

    assert config.hidden_dim == 128
    assert config.intention_dim == 64
    assert config.num_layers == 4
    assert config.num_heads == 4
