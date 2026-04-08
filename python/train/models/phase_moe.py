"""Generic hard-routed phase mixture-of-experts wrapper."""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


PHASE_MOE_MODEL_NAME = "lapv2_phase_moe_v1"


if torch is not None and nn is not None:

    class PhaseMoE(nn.Module):
        """Route each batch item onto one hard-selected phase expert."""

        def __init__(
            self,
            make_expert: Callable[[], nn.Module],
            num_phases: int = 4,
        ) -> None:
            super().__init__()
            if num_phases <= 0:
                raise ValueError("num_phases must be positive")
            self.num_phases = int(num_phases)
            self.experts = nn.ModuleList(make_expert() for _ in range(num_phases))

        def forward(
            self,
            *args: Any,
            phase_idx: torch.LongTensor,
            **kwargs: Any,
        ) -> Any:
            if phase_idx.ndim != 1:
                raise ValueError("phase_idx must be a rank-1 tensor")
            batch_size = int(phase_idx.shape[0])
            outputs_by_phase: dict[int, tuple[torch.Tensor, Any]] = {}
            for phase in range(self.num_phases):
                mask = phase_idx == phase
                if not bool(mask.any().item()):
                    continue
                expert_args = tuple(_slice_input(value, mask, batch_size) for value in args)
                expert_kwargs = {
                    key: _slice_input(value, mask, batch_size)
                    for key, value in kwargs.items()
                }
                outputs_by_phase[phase] = (mask, self.experts[phase](*expert_args, **expert_kwargs))
            if not outputs_by_phase:
                raise ValueError("phase_idx selected no active experts")
            return _merge_outputs(outputs_by_phase, batch_size)

        @classmethod
        def from_single(
            cls,
            pretrained: nn.Module,
            num_phases: int = 4,
        ) -> "PhaseMoE":
            if num_phases <= 0:
                raise ValueError("num_phases must be positive")
            moe = cls.__new__(cls)
            nn.Module.__init__(moe)
            moe.num_phases = int(num_phases)
            moe.experts = nn.ModuleList(
                copy.deepcopy(pretrained)
                for _ in range(num_phases)
            )
            return moe


    def _slice_input(value: Any, mask: torch.Tensor, batch_size: int) -> Any:
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.shape[0]) == batch_size:
            return value[mask]
        return value


    def _merge_outputs(
        outputs_by_phase: Mapping[int, tuple[torch.Tensor, Any]],
        batch_size: int,
    ) -> Any:
        example_output = next(iter(outputs_by_phase.values()))[1]
        return _merge_value(example_output, outputs_by_phase, batch_size)


    def _merge_value(
        example_value: Any,
        outputs_by_phase: Mapping[int, tuple[torch.Tensor, Any]],
        batch_size: int,
    ) -> Any:
        if isinstance(example_value, torch.Tensor):
            if example_value.ndim == 0:
                raise ValueError("PhaseMoE experts must return batched tensors")
            merged = torch.zeros(
                (batch_size, *example_value.shape[1:]),
                dtype=example_value.dtype,
                device=example_value.device,
            )
            for mask, value in outputs_by_phase.values():
                merged[mask] = value
            return merged
        if isinstance(example_value, tuple):
            return tuple(
                _merge_value(
                    example_value[index],
                    {
                        phase: (mask, value[index])
                        for phase, (mask, value) in outputs_by_phase.items()
                    },
                    batch_size,
                )
                for index in range(len(example_value))
            )
        if isinstance(example_value, list):
            return [
                _merge_value(
                    example_value[index],
                    {
                        phase: (mask, value[index])
                        for phase, (mask, value) in outputs_by_phase.items()
                    },
                    batch_size,
                )
                for index in range(len(example_value))
            ]
        if isinstance(example_value, dict):
            return {
                key: _merge_value(
                    example_value[key],
                    {
                        phase: (mask, value[key])
                        for phase, (mask, value) in outputs_by_phase.items()
                    },
                    batch_size,
                )
                for key in example_value
            }
        raise TypeError(
            "PhaseMoE only supports tensor, tuple, list, and dict expert outputs"
        )

else:  # pragma: no cover - exercised when torch is absent

    class PhaseMoE:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("torch is required to instantiate PhaseMoE.")
