"""PyTorch proposer model for legality and policy prediction."""

from __future__ import annotations

from typing import Any

from train.action_space import ACTION_SPACE_SIZE
from train.datasets.artifacts import POSITION_FEATURE_SIZE

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


MODEL_NAME = "legality_policy_proposer_v1"


def torch_is_available() -> bool:
    """Report whether PyTorch is available in the current environment."""
    return torch is not None and nn is not None


if nn is not None:

    class LegalityPolicyProposer(nn.Module):
        """Simple MLP proposer over the deterministic Phase-3 encoder features."""

        def __init__(self, *, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
            super().__init__()
            if hidden_layers <= 0:
                raise ValueError("hidden_layers must be positive")

            layers: list[nn.Module] = []
            input_dim = POSITION_FEATURE_SIZE
            for _ in range(hidden_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)
            self.legality_head = nn.Linear(input_dim, ACTION_SPACE_SIZE)
            self.policy_head = nn.Linear(input_dim, ACTION_SPACE_SIZE)

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Return legality logits and policy logits for the flat action space."""
            hidden = self.backbone(features)
            return self.legality_head(hidden), self.policy_head(hidden)

else:

    class LegalityPolicyProposer:  # pragma: no cover - exercised when torch is absent
        """Import-safe fallback when PyTorch is not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for proposer training. Install the 'train' extra or torch."
            )
