from __future__ import annotations

import pytest

from train.datasets.contracts import candidate_context_feature_dim
from train.models.intention_encoder import torch
from train.models.policy_head_large import LargePolicyHead


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _sample_root(*, batch_size: int = 2) -> torch.Tensor:
    return torch.randn((batch_size, 512), dtype=torch.float32)


def _sample_candidates(*, batch_size: int = 2, candidate_count: int = 5) -> torch.Tensor:
    feature_dim = candidate_context_feature_dim(2)
    return torch.randn((batch_size, candidate_count, feature_dim), dtype=torch.float32)


def _sample_indices(*, batch_size: int = 2, candidate_count: int = 5) -> torch.Tensor:
    base = torch.tensor([1, 17, 42, 103, 511], dtype=torch.long)
    return base.unsqueeze(0).repeat(batch_size, 1)


def _sample_mask(*, batch_size: int = 2, candidate_count: int = 5) -> torch.Tensor:
    return torch.tensor(
        [[True, True, True, False, False], [True, True, False, False, False]],
        dtype=torch.bool,
    )


def test_large_policy_head_output_shape_matches_candidate_count() -> None:
    model = LargePolicyHead()
    logits = model(
        _sample_root(),
        _sample_candidates(),
        _sample_indices(),
        _sample_mask(),
    )

    assert tuple(logits.shape) == (2, 5)


def test_masked_candidates_produce_negative_infinity_logits() -> None:
    model = LargePolicyHead()
    logits = model(
        _sample_root(),
        _sample_candidates(),
        _sample_indices(),
        _sample_mask(),
    )

    assert torch.isneginf(logits[0, 3])
    assert torch.isneginf(logits[0, 4])
    assert torch.isneginf(logits[1, 2])
    assert torch.isfinite(logits[0, 0])


def test_large_policy_head_backward_populates_gradients() -> None:
    model = LargePolicyHead()
    logits = model(
        _sample_root(),
        _sample_candidates(),
        _sample_indices(),
        _sample_mask(),
    )
    finite_logits = torch.where(torch.isfinite(logits), logits, torch.zeros_like(logits))
    finite_logits.sum().backward()

    assert model.root_projection[0].weight.grad is not None
    assert model.action_embedding.weight.grad is not None
    assert model.layers[0].cross_attention.in_proj_weight.grad is not None
    assert model.layers[-1].feedforward[0].weight.grad is not None
    assert model.scorer[0].weight.grad is not None


def test_large_policy_head_parameter_budget_is_close_to_hundred_megabytes() -> None:
    model = LargePolicyHead()
    parameter_bytes = sum(parameter.numel() for parameter in model.parameters()) * 4
    target_bytes = 100 * 1024 * 1024
    lower_bound = int(target_bytes * 0.7)
    upper_bound = int(target_bytes * 1.3)

    assert lower_bound <= parameter_bytes <= upper_bound
