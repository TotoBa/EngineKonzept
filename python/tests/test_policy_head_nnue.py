from __future__ import annotations

import pytest

from train.models.intention_encoder import torch
from train.models.policy_head_nnue import NNUEPolicyHead


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def test_policy_nnue_forward_shape() -> None:
    model = NNUEPolicyHead(accumulator_dim=8, move_type_dim=4, hidden_dim=8)
    a_root_stm = torch.randn((3, 8), dtype=torch.float32)
    a_succ_other = torch.randn((3, 5, 8), dtype=torch.float32)
    move_type_ids = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 1, 1, 1, 1], [4, 3, 2, 1, 0]],
        dtype=torch.long,
    )

    logits = model(a_root_stm, a_succ_other, move_type_ids)

    assert tuple(logits.shape) == (3, 5)


def test_policy_nnue_logits_change_with_move_type_embed() -> None:
    model = NNUEPolicyHead(accumulator_dim=4, move_type_dim=2, hidden_dim=4)
    with torch.no_grad():
        model.adapter.weight.zero_()
        model.adapter.bias.zero_()
        model.move_type_emb.weight.zero_()
        model.move_type_emb.weight[1] = torch.tensor([1.0, -1.0])
        first_linear = model.move_head[0]
        first_linear.weight.zero_()
        first_linear.bias.zero_()
        first_linear.weight[0, -2:] = torch.tensor([1.0, -1.0])
        final_linear = model.move_head[2]
        final_linear.weight.zero_()
        final_linear.bias.zero_()
        final_linear.weight[0, 0] = 1.0
    logits = model(
        torch.zeros((1, 4), dtype=torch.float32),
        torch.zeros((1, 2, 4), dtype=torch.float32),
        torch.tensor([[0, 1]], dtype=torch.long),
    )

    assert float(logits[0, 0].item()) != float(logits[0, 1].item())


def test_policy_nnue_loss_decreases() -> None:
    torch.manual_seed(7)
    model = NNUEPolicyHead(accumulator_dim=8, move_type_dim=4, hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    a_root_stm = torch.randn((8, 8), dtype=torch.float32)
    a_succ_other = torch.randn((8, 3, 8), dtype=torch.float32)
    move_type_ids = torch.randint(0, 8, (8, 3), dtype=torch.long)
    target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)

    losses: list[float] = []
    for _step in range(10):
        optimizer.zero_grad()
        logits = model(a_root_stm, a_succ_other, move_type_ids)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0]
