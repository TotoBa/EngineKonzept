"""Phase-6 latent dynamics model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from train.action_space import ACTION_SPACE_SIZE
from train.datasets.artifacts import (
    PIECE_FEATURE_SIZE,
    POSITION_FEATURE_SIZE,
    RULE_FEATURE_SIZE,
    SQUARE_FEATURE_SIZE,
)

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


@dataclass(frozen=True)
class DynamicsPrediction:
    """Structured next-state prediction for one action-conditioned transition."""

    next_features: Any
    piece_features: Any
    square_features: Any
    rule_features: Any


if torch is not None and nn is not None:

    class LatentDynamicsModel(nn.Module):
        """Encode state features, apply an action-conditioned latent step, and decode next state."""

        def __init__(
            self,
            *,
            architecture: str,
            latent_dim: int,
            hidden_dim: int,
            hidden_layers: int,
            action_embedding_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.architecture = architecture
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
            self.transition_input_dim = latent_dim + action_embedding_dim
            self.transition = _build_mlp(
                input_dim=self.transition_input_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_dim=latent_dim,
                dropout=dropout,
            )
            if architecture == "mlp_v1":
                self.decoder = _build_mlp(
                    input_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    output_dim=POSITION_FEATURE_SIZE,
                    dropout=dropout,
                )
                self.piece_decoder = None
                self.square_decoder = None
                self.rule_decoder = None
            elif architecture == "structured_v2":
                self.decoder = None
                self.piece_decoder = _build_mlp(
                    input_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=max(1, hidden_layers - 1),
                    output_dim=PIECE_FEATURE_SIZE,
                    dropout=dropout,
                )
                self.square_decoder = _build_mlp(
                    input_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=max(1, hidden_layers - 1),
                    output_dim=SQUARE_FEATURE_SIZE,
                    dropout=dropout,
                )
                self.rule_decoder = _build_mlp(
                    input_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=max(1, hidden_layers - 1),
                    output_dim=RULE_FEATURE_SIZE,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"unsupported dynamics architecture: {architecture}")

        def encode(self, features: Any) -> Any:
            """Map flat state features to a latent vector."""
            return self.encoder(features)

        def step(self, latent: Any, action_indices: Any) -> Any:
            """Apply one residual action-conditioned latent transition."""
            action_embedding = self.action_embedding(action_indices)
            transition_input = torch.cat((latent, action_embedding), dim=1)
            delta = self.transition(transition_input)
            return latent + delta

        def predict(self, features: Any, action_indices: Any) -> DynamicsPrediction:
            """Predict structured next-state sections for one action-conditioned transition."""
            latent = self.encode(features)
            next_latent = self.step(latent, action_indices)
            decoded = self.decode(next_latent)
            return DynamicsPrediction(
                next_features=decoded.next_features,
                piece_features=decoded.piece_features,
                square_features=decoded.square_features,
                rule_features=decoded.rule_features,
            )

        def decode(self, latent: Any) -> DynamicsPrediction:
            """Decode a latent state into structured next-state sections."""
            if self.architecture == "mlp_v1":
                next_features = self.decoder(latent)
                piece_features, square_features, rule_features = torch.split(
                    next_features,
                    [PIECE_FEATURE_SIZE, SQUARE_FEATURE_SIZE, RULE_FEATURE_SIZE],
                    dim=1,
                )
            else:
                piece_features = self.piece_decoder(latent)
                square_features = self.square_decoder(latent)
                rule_features = self.rule_decoder(latent)
                next_features = torch.cat(
                    (piece_features, square_features, rule_features),
                    dim=1,
                )
            return DynamicsPrediction(
                next_features=next_features,
                piece_features=piece_features,
                square_features=square_features,
                rule_features=rule_features,
            )

        def forward(self, features: Any, action_indices: Any) -> Any:
            """Predict packed next-state features for one action-conditioned transition."""
            return self.predict(features, action_indices).next_features


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
