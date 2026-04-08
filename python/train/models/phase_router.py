"""Deterministic phase router for future LAPv2 hard expert routing."""

from __future__ import annotations

from typing import Mapping

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


PHASE_ROUTER_MODEL_NAME = "lapv2_phase_router_v1"


if torch is not None and nn is not None:

    class PhaseRouter(nn.Module):
        """Pass through the precomputed hard phase index tensor."""

        NUM_PHASES = 4

        def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.LongTensor:
            phase_idx = batch["phase_index"].to(dtype=torch.long)
            if phase_idx.ndim != 1:
                raise ValueError("phase_index must be a rank-1 tensor")
            return phase_idx

else:  # pragma: no cover - exercised when torch is absent

    class PhaseRouter:  # type: ignore[no-redef]
        NUM_PHASES = 4

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("torch is required to instantiate PhaseRouter.")
