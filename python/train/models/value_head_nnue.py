"""NNUE-style value readout over shared sparse LAPv2 accumulators."""

from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


NNUE_VALUE_HEAD_MODEL_NAME = "lapv2_nnue_value_head_v1"


if torch is not None and nn is not None:

    class ClippedReLU(nn.Module):
        """Clamp activations into the classic NNUE [0, 1] band."""

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.clamp(values, 0.0, 1.0)


    class NNUEValueHead(nn.Module):
        """Predict WDL, cp, and sigma from side-to-move-oriented accumulators."""

        def __init__(
            self,
            *,
            accumulator_dim: int = 64,
            hidden_dim: int = 32,
            cp_score_cap: float = 1024.0,
        ) -> None:
            super().__init__()
            if accumulator_dim <= 0:
                raise ValueError("accumulator_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if cp_score_cap <= 0.0:
                raise ValueError("cp_score_cap must be positive")
            self.accumulator_dim = int(accumulator_dim)
            self.hidden_dim = int(hidden_dim)
            self.cp_score_cap = float(cp_score_cap)
            self.adapter = nn.Linear(accumulator_dim, accumulator_dim)
            self.clipped_relu = ClippedReLU()
            self.backbone = nn.Sequential(
                nn.Linear(2 * accumulator_dim, hidden_dim),
                ClippedReLU(),
            )
            self.wdl_head = nn.Linear(hidden_dim, 3)
            self.cp_head = nn.Linear(hidden_dim, 1)
            self.sigma_head = nn.Linear(hidden_dim, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            std = 1.0 / math.sqrt(float(self.accumulator_dim))
            nn.init.normal_(self.adapter.weight, mean=0.0, std=std)
            nn.init.zeros_(self.adapter.bias)
            for module in (self.backbone[0], self.wdl_head, self.cp_head, self.sigma_head):
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                        nn.init.uniform_(module.bias, -bound, bound)

        def forward(
            self,
            a_stm: torch.Tensor,
            a_other: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if a_stm.ndim != 2 or a_stm.shape[1] != self.accumulator_dim:
                raise ValueError(
                    f"a_stm must have shape (batch, {self.accumulator_dim})"
                )
            if a_other.ndim != 2 or a_other.shape[1] != self.accumulator_dim:
                raise ValueError(
                    f"a_other must have shape (batch, {self.accumulator_dim})"
                )
            stm_hidden = self.clipped_relu(self.adapter(a_stm))
            other_hidden = self.clipped_relu(self.adapter(a_other))
            hidden = self.backbone(torch.cat([stm_hidden, other_hidden], dim=1))
            wdl_logits = self.wdl_head(hidden)
            cp_score = torch.tanh(self.cp_head(hidden)) * self.cp_score_cap
            sigma_value = torch.nn.functional.softplus(self.sigma_head(hidden)) + 1e-6
            return wdl_logits, cp_score, sigma_value

else:  # pragma: no cover - exercised when torch is absent

    class ClippedReLU:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("torch is required to instantiate ClippedReLU.")


    class NNUEValueHead:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("torch is required to instantiate NNUEValueHead.")
