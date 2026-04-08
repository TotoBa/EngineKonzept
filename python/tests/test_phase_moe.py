from __future__ import annotations

import pytest

from train.models.intention_encoder import torch
from train.models.phase_moe import PhaseMoE
from train.models.phase_router import PhaseRouter


pytest.importorskip("torch")


def test_phase_router_passthrough() -> None:
    router = PhaseRouter()
    phase_idx = router({"phase_index": torch.tensor([0, 2, 1, 3], dtype=torch.long)})
    assert torch.equal(phase_idx, torch.tensor([0, 2, 1, 3], dtype=torch.long))


def test_phase_moe_equivalent_to_single_after_init() -> None:
    torch.manual_seed(7)
    single = torch.nn.Linear(3, 2)
    moe = PhaseMoE.from_single(single, num_phases=4)
    inputs = torch.randn(5, 3)
    phase_idx = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)

    expected = single(inputs)
    actual = moe(inputs, phase_idx=phase_idx)

    assert torch.allclose(actual, expected)


def test_phase_moe_routes_correctly() -> None:
    class Expert(torch.nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.offset = offset

        def forward(self, values: torch.Tensor, *, scale: torch.Tensor, bias: float) -> dict[str, torch.Tensor]:
            return {"values": values * scale + self.offset + bias}

    offsets = iter((0.0, 10.0, 20.0, 30.0))
    moe = PhaseMoE(lambda: Expert(next(offsets)), num_phases=4)
    values = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    scale = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    phase_idx = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    outputs = moe(values, phase_idx=phase_idx, scale=scale, bias=0.5)

    assert torch.allclose(
        outputs["values"],
        torch.tensor([[1.5], [14.5], [29.5], [46.5]], dtype=torch.float32),
    )


def test_phase_moe_gradients_only_flow_to_active_expert() -> None:
    torch.manual_seed(13)
    moe = PhaseMoE(lambda: torch.nn.Linear(2, 1, bias=False), num_phases=4)
    inputs = torch.randn(6, 2)
    phase_idx = torch.tensor([0, 0, 1, 1, 1, 0], dtype=torch.long)

    loss = moe(inputs, phase_idx=phase_idx).sum()
    loss.backward()

    grad_norms = []
    for expert in moe.experts:
        grad = expert.weight.grad
        grad_norms.append(0.0 if grad is None else float(torch.norm(grad).item()))

    assert grad_norms[0] > 0.0
    assert grad_norms[1] > 0.0
    assert grad_norms[2] == 0.0
    assert grad_norms[3] == 0.0
