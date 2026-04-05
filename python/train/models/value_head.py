"""Model-only LAPv1 value and sharpness heads."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


VALUE_HEAD_MODEL_NAME = "lapv1_value_head"
SHARPNESS_HEAD_MODEL_NAME = "lapv1_sharpness_head"
DEFAULT_STATE_DIM = 512
DEFAULT_MEMORY_DIM = 256


if torch is not None and nn is not None:

    def _build_mlp(
        *,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        return nn.Sequential(*layers)


    class ValueHead(nn.Module):
        """Predict WDL logits, cp score, and positive value uncertainty from z_root."""

        def __init__(
            self,
            *,
            state_dim: int = DEFAULT_STATE_DIM,
            memory_dim: int = DEFAULT_MEMORY_DIM,
            hidden_dim: int = 2816,
            hidden_layers: int = 4,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if state_dim <= 0:
                raise ValueError("state_dim must be positive")
            if memory_dim <= 0:
                raise ValueError("memory_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if hidden_layers <= 0:
                raise ValueError("hidden_layers must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0.0, 1.0)")

            self.state_dim = state_dim
            self.memory_dim = memory_dim
            self.hidden_dim = hidden_dim
            self.hidden_layers = hidden_layers
            self.memory_projection = nn.Sequential(
                nn.Linear(memory_dim, state_dim),
                nn.LayerNorm(state_dim),
                nn.Tanh(),
            )
            self.backbone = _build_mlp(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                dropout=dropout,
            )
            self.wdl_head = nn.Linear(hidden_dim, 3)
            self.cp_head = nn.Linear(hidden_dim, 1)
            self.sigma_head = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            z_root: torch.Tensor,
            memory: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Return WDL logits, cp score, and positive sigma_value."""
            if z_root.ndim != 2 or z_root.shape[1] != self.state_dim:
                raise ValueError(f"z_root must have shape (batch, {self.state_dim})")
            if memory is not None:
                if memory.ndim != 3 or memory.shape[2] != self.memory_dim:
                    raise ValueError(
                        f"memory must have shape (batch, slots, {self.memory_dim})"
                    )
                memory_summary = self.memory_projection(memory.mean(dim=1))
                z_root = z_root + memory_summary

            hidden = self.backbone(z_root)
            wdl_logits = self.wdl_head(hidden)
            cp_score = self.cp_head(hidden)
            sigma_value = torch.nn.functional.softplus(self.sigma_head(hidden)) + 1e-6
            return wdl_logits, cp_score, sigma_value


    class SharpnessHead(nn.Module):
        """Predict whether more bounded deliberation is likely worthwhile."""

        def __init__(
            self,
            *,
            state_dim: int = DEFAULT_STATE_DIM,
            hidden_dim: int = 128,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if state_dim <= 0:
                raise ValueError("state_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0.0, 1.0)")

            self.state_dim = state_dim
            self.hidden_dim = hidden_dim
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, z_root: torch.Tensor) -> torch.Tensor:
            """Return a [0, 1] sharpness scalar per batch row."""
            if z_root.ndim != 2 or z_root.shape[1] != self.state_dim:
                raise ValueError(f"z_root must have shape (batch, {self.state_dim})")
            return torch.sigmoid(self.network(z_root))
