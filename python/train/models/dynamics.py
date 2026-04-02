"""Phase-6 latent dynamics model definitions."""

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


DYNAMICS_MODEL_NAME = "latent_dynamics_v1"


def torch_is_available() -> bool:
    """Return whether PyTorch is importable in the current environment."""
    return torch is not None and nn is not None


if torch is not None and nn is not None:

    class LatentDynamicsModel(nn.Module):
        """Encode state features, apply an action-conditioned latent step, and decode next state."""

        def __init__(
            self,
            *,
            latent_dim: int,
            hidden_dim: int,
            hidden_layers: int,
            action_embedding_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.action_embedding_dim = action_embedding_dim
            self.encoder = _build_mlp(
                input_dim=POSITION_FEATURE_SIZE,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_dim=latent_dim,
                dropout=dropout,
            )
            self.action_embedding = nn.Embedding(ACTION_SPACE_SIZE, action_embedding_dim)
            self.transition = _build_mlp(
                input_dim=latent_dim + action_embedding_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_dim=latent_dim,
                dropout=dropout,
            )
            self.decoder = _build_mlp(
                input_dim=latent_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_dim=POSITION_FEATURE_SIZE,
                dropout=dropout,
            )

        def encode(self, features: Any) -> Any:
            """Map flat state features to a latent vector."""
            return self.encoder(features)

        def step(self, latent: Any, action_indices: Any) -> Any:
            """Apply one residual action-conditioned latent transition."""
            action_embedding = self.action_embedding(action_indices)
            delta = self.transition(torch.cat((latent, action_embedding), dim=1))
            return latent + delta

        def decode(self, latent: Any) -> Any:
            """Decode a latent state back into packed next-state features."""
            return self.decoder(latent)

        def forward(self, features: Any, action_indices: Any) -> Any:
            """Predict packed next-state features for one action-conditioned transition."""
            latent = self.encode(features)
            next_latent = self.step(latent, action_indices)
            return self.decode(next_latent)


    def _build_mlp(
        *,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

else:

    class LatentDynamicsModel:  # pragma: no cover - exercised when torch is absent
        """Import-safe fallback when PyTorch is unavailable."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for dynamics training. Install the 'train' extra or torch."
            )
