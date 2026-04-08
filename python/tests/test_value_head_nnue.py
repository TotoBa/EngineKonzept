from __future__ import annotations

import pytest

from train.models.intention_encoder import torch
from train.models.value_head_nnue import NNUEValueHead


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def test_nnue_value_forward_shape() -> None:
    model = NNUEValueHead(accumulator_dim=8, hidden_dim=4)
    a_stm = torch.randn((3, 8), dtype=torch.float32)
    a_other = torch.randn((3, 8), dtype=torch.float32)

    wdl_logits, cp_score, sigma_value = model(a_stm, a_other)

    assert tuple(wdl_logits.shape) == (3, 3)
    assert tuple(cp_score.shape) == (3, 1)
    assert tuple(sigma_value.shape) == (3, 1)


def test_nnue_value_start_position_finite() -> None:
    model = NNUEValueHead(accumulator_dim=8, hidden_dim=4)
    a_stm = torch.zeros((1, 8), dtype=torch.float32)
    a_other = torch.zeros((1, 8), dtype=torch.float32)

    wdl_logits, cp_score, sigma_value = model(a_stm, a_other)

    assert torch.isfinite(wdl_logits).all()
    assert torch.isfinite(cp_score).all()
    assert torch.isfinite(sigma_value).all()


def test_nnue_value_loss_decreases() -> None:
    torch.manual_seed(7)
    model = NNUEValueHead(accumulator_dim=8, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    a_stm = torch.randn((8, 8), dtype=torch.float32)
    a_other = torch.randn((8, 8), dtype=torch.float32)
    cp_target = ((a_stm - a_other).sum(dim=1, keepdim=True) * 25.0).clamp(-200.0, 200.0)
    wdl_target = torch.where(
        cp_target.squeeze(1) > 50.0,
        torch.full((8,), 2, dtype=torch.long),
        torch.where(
            cp_target.squeeze(1) < -50.0,
            torch.zeros((8,), dtype=torch.long),
            torch.ones((8,), dtype=torch.long),
        ),
    )

    losses: list[float] = []
    for _step in range(10):
        optimizer.zero_grad()
        wdl_logits, cp_score, _sigma_value = model(a_stm, a_other)
        loss = torch.nn.functional.cross_entropy(wdl_logits, wdl_target)
        loss = loss + torch.nn.functional.mse_loss(cp_score / 256.0, cp_target / 256.0)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0]
