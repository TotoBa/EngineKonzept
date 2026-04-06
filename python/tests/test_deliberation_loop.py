from __future__ import annotations

import pytest

from train.models.deliberation import (
    CandidateSelector,
    DeliberationLoop,
    torch,
)


pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


class ConstantSharpness(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return torch.full((z_t.shape[0],), self.value, dtype=z_t.dtype, device=z_t.device)


class ScriptedValueProjector(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = values
        self.calls = 0

    def forward(
        self,
        z_t: torch.Tensor,
        M_t: torch.Tensor,
        C_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del M_t, C_t
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return (
            torch.full((z_t.shape[0],), value, dtype=z_t.dtype, device=z_t.device),
            torch.full((z_t.shape[0],), 0.5, dtype=z_t.dtype, device=z_t.device),
        )


class PerExampleSharpness(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.values = values

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            self.values,
            dtype=z_t.dtype,
            device=z_t.device,
        )


class ScriptedPerExampleValueProjector(torch.nn.Module):
    def __init__(self, values: list[list[float]]) -> None:
        super().__init__()
        self.values = values
        self.calls = 0

    def forward(
        self,
        z_t: torch.Tensor,
        M_t: torch.Tensor,
        C_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del M_t, C_t
        values = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return (
            torch.tensor(values, dtype=z_t.dtype, device=z_t.device),
            torch.full((z_t.shape[0],), 0.5, dtype=z_t.dtype, device=z_t.device),
        )


def _sample_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z_root = torch.randn((2, 512), dtype=torch.float32)
    candidate_action_indices = torch.tensor(
        [[3, 7, 11, 0], [5, 9, 0, 0]],
        dtype=torch.long,
    )
    initial_candidate_scores = torch.tensor(
        [[0.8, 0.5, 0.1, -1.0], [0.7, 0.2, -1.0, -1.0]],
        dtype=torch.float32,
    )
    candidate_mask = torch.tensor(
        [[True, True, True, False], [True, True, False, False]],
        dtype=torch.bool,
    )
    return z_root, candidate_action_indices, initial_candidate_scores, candidate_mask


def test_single_legal_move_early_exits_at_t0() -> None:
    loop = DeliberationLoop()
    z_root, candidate_action_indices, initial_candidate_scores, candidate_mask = _sample_inputs()
    candidate_mask = torch.tensor(
        [[True, False, False, False], [True, False, False, False]],
        dtype=torch.bool,
    )

    outputs = loop(
        z_root,
        candidate_action_indices,
        initial_candidate_scores,
        candidate_mask,
        single_legal_move=True,
    )

    assert outputs["step_count"] == 0
    assert len(outputs["trace"].steps) == 0


def test_max_inner_steps_is_a_hard_cap() -> None:
    loop = DeliberationLoop(
        max_inner_steps=3,
        min_inner_steps=3,
        sharpness_projector=ConstantSharpness(1.0),
    )
    outputs = loop(*_sample_inputs())

    assert outputs["step_count"] == 3
    assert len(outputs["trace"].steps) == 3


def test_rollback_fires_on_synthetic_value_regression() -> None:
    loop = DeliberationLoop(
        max_inner_steps=2,
        min_inner_steps=0,
        rollback_threshold=40.0,
        sharpness_projector=ConstantSharpness(1.0),
        value_projector=ScriptedValueProjector([100.0, 10.0, 100.0]),
    )
    outputs = loop(*_sample_inputs())

    assert any(step.rollback_fired for step in outputs["trace"].steps)


def test_halt_is_applied_per_example_not_batch_global() -> None:
    loop = DeliberationLoop(
        max_inner_steps=2,
        min_inner_steps=0,
        q_threshold=0.3,
        sharpness_projector=PerExampleSharpness([0.1, 1.0]),
    )

    outputs = loop(*_sample_inputs())

    assert outputs["step_count"] == 2
    assert outputs["step_active_masks"][0].tolist() == [False, True]
    assert outputs["step_active_masks"][1].tolist() == [False, True]


def test_rollback_is_applied_per_example_not_batch_global() -> None:
    loop = DeliberationLoop(
        max_inner_steps=1,
        min_inner_steps=0,
        rollback_threshold=40.0,
        sharpness_projector=ConstantSharpness(1.0),
        value_projector=ScriptedPerExampleValueProjector(
            [
                [100.0, 100.0],
                [10.0, 95.0],
            ]
        ),
    )

    outputs = loop(*_sample_inputs())

    assert outputs["step_count"] == 1
    assert outputs["step_rollback_masks"][0].tolist() == [True, False]
    assert outputs["step_rollback_flags"] == (True,)


def test_deliberation_loop_is_deterministic_for_fixed_seed() -> None:
    torch.manual_seed(42)
    first = DeliberationLoop()
    torch.manual_seed(42)
    second = DeliberationLoop()
    inputs = _sample_inputs()

    first_outputs = first(*inputs)
    second_outputs = second(*inputs)

    assert torch.allclose(
        first_outputs["final_candidate_scores"],
        second_outputs["final_candidate_scores"],
        equal_nan=True,
    )
    assert first_outputs["refined_top1_action_index"].tolist() == second_outputs[
        "refined_top1_action_index"
    ].tolist()
    assert [
        step.selected_candidates for step in first_outputs["trace"].steps
    ] == [
        step.selected_candidates for step in second_outputs["trace"].steps
    ]


def test_trace_length_matches_actual_step_count() -> None:
    loop = DeliberationLoop(max_inner_steps=4, min_inner_steps=4)
    outputs = loop(*_sample_inputs())

    assert len(outputs["trace"].steps) == outputs["step_count"]
    assert [step.step for step in outputs["trace"].steps] == list(
        range(outputs["step_count"])
    )


def test_candidate_selector_returns_top_k_indices() -> None:
    selector = CandidateSelector(top_k=2)
    indices = selector(
        torch.zeros((1, 512), dtype=torch.float32),
        torch.tensor([[0.1, 0.9, 0.3]], dtype=torch.float32),
        torch.tensor([[0.5]], dtype=torch.float32),
        torch.tensor([[True, True, True]], dtype=torch.bool),
    )

    assert tuple(indices[0].tolist()) == (1, 2)
