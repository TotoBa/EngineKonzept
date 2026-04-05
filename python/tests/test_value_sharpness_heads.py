from __future__ import annotations

import pytest

from train.config import SharpnessHeadConfig, ValueHeadConfig
from train.models.intention_encoder import torch
from train.models.value_head import SharpnessHead, ValueHead


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _sample_z_root(*, batch_size: int = 2, state_dim: int = 512) -> torch.Tensor:
    return torch.randn((batch_size, state_dim), dtype=torch.float32)


def _sample_memory(*, batch_size: int = 2, slots: int = 4, memory_dim: int = 256) -> torch.Tensor:
    return torch.randn((batch_size, slots, memory_dim), dtype=torch.float32)


def test_value_head_output_shapes_are_correct() -> None:
    model = ValueHead()
    wdl_logits, cp_score, sigma_value = model(_sample_z_root(), _sample_memory())

    assert tuple(wdl_logits.shape) == (2, 3)
    assert tuple(cp_score.shape) == (2, 1)
    assert tuple(sigma_value.shape) == (2, 1)


def test_wdl_logits_softmax_sums_to_one() -> None:
    model = ValueHead()
    wdl_logits, _cp_score, _sigma_value = model(_sample_z_root())
    probabilities = torch.softmax(wdl_logits, dim=1)

    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(2, dtype=torch.float32),
        atol=1e-5,
    )


def test_sharpness_head_outputs_unit_interval_scores() -> None:
    model = SharpnessHead()
    sharpness = model(_sample_z_root())

    assert tuple(sharpness.shape) == (2, 1)
    assert bool(torch.all(sharpness >= 0.0))
    assert bool(torch.all(sharpness <= 1.0))


def test_value_and_sharpness_parameter_counts_match_targets() -> None:
    value_head = ValueHead()
    sharpness_head = SharpnessHead()

    value_parameter_bytes = sum(parameter.numel() for parameter in value_head.parameters()) * 4
    sharpness_parameter_bytes = (
        sum(parameter.numel() for parameter in sharpness_head.parameters()) * 4
    )

    value_target_bytes = 100 * 1024 * 1024
    value_lower_bound = int(value_target_bytes * 0.7)
    value_upper_bound = int(value_target_bytes * 1.3)

    assert value_lower_bound <= value_parameter_bytes <= value_upper_bound
    assert sharpness_parameter_bytes < 1 * 1024 * 1024


def test_value_and_sharpness_configs_validate_defaults() -> None:
    value_config = ValueHeadConfig()
    sharpness_config = SharpnessHeadConfig()

    assert value_config.hidden_dim == 2816
    assert value_config.hidden_layers == 4
    assert sharpness_config.hidden_dim == 128
