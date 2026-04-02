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
    next_latent: Any | None = None
    piece_delta_features: Any | None = None
    square_delta_features: Any | None = None
    rule_delta_features: Any | None = None


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
                self.piece_delta_decoder = None
                self.square_delta_decoder = None
                self.rule_delta_decoder = None
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
                self.piece_delta_decoder = None
                self.square_delta_decoder = None
                self.rule_delta_decoder = None
            elif architecture in {"structured_v3", "structured_v4"}:
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
                self.piece_delta_decoder = _build_mlp(
                    input_dim=self.transition_input_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=1,
                    output_dim=PIECE_FEATURE_SIZE,
                    dropout=dropout,
                )
                self.square_delta_decoder = _build_mlp(
                    input_dim=self.transition_input_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=1,
                    output_dim=SQUARE_FEATURE_SIZE,
                    dropout=dropout,
                )
                self.rule_delta_decoder = _build_mlp(
                    input_dim=self.transition_input_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=1,
                    output_dim=RULE_FEATURE_SIZE,
                    dropout=dropout,
                )
            elif architecture == "edit_v1":
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
                self.piece_delta_decoder = _build_mlp(
                    input_dim=self.transition_input_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=1,
                    output_dim=PIECE_FEATURE_SIZE,
                    dropout=dropout,
                )
                self.square_delta_decoder = _build_mlp(
                    input_dim=self.transition_input_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=1,
                    output_dim=SQUARE_FEATURE_SIZE,
                    dropout=dropout,
                )
                self.rule_delta_decoder = _build_mlp(
                    input_dim=self.transition_input_dim,
                    hidden_dim=hidden_dim,
                    hidden_layers=1,
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
            return self.step_from_action_embedding(latent, action_embedding)

        def step_from_action_embedding(self, latent: Any, action_embedding: Any) -> Any:
            """Apply one residual action-conditioned latent transition from a precomputed embedding."""
            transition_input = torch.cat((latent, action_embedding), dim=1)
            return self.step_from_transition_input(latent, transition_input)

        def step_from_transition_input(self, latent: Any, transition_input: Any) -> Any:
            """Apply one residual action-conditioned latent transition from a prebuilt transition input."""
            delta = self.transition(transition_input)
            return latent + delta

        def predict(self, features: Any, action_indices: Any) -> DynamicsPrediction:
            """Predict structured next-state sections for one action-conditioned transition."""
            latent = self.encode(features)
            action_embedding = self.action_embedding(action_indices)
            transition_input = torch.cat((latent, action_embedding), dim=1)
            next_latent = self.step_from_transition_input(latent, transition_input)
            decoded = self.decode(next_latent)
            piece_delta_features = None
            square_delta_features = None
            rule_delta_features = None
            if self.architecture in {"structured_v3", "structured_v4", "edit_v1"}:
                piece_delta_features = self.piece_delta_decoder(transition_input)
                square_delta_features = self.square_delta_decoder(transition_input)
                rule_delta_features = self.rule_delta_decoder(transition_input)
            if self.architecture == "edit_v1":
                current_piece, current_square, current_rule = torch.split(
                    features,
                    [PIECE_FEATURE_SIZE, SQUARE_FEATURE_SIZE, RULE_FEATURE_SIZE],
                    dim=1,
                )
                piece_features = current_piece + piece_delta_features
                square_features = current_square + square_delta_features
                rule_features = current_rule + rule_delta_features
                decoded = DynamicsPrediction(
                    next_features=torch.cat(
                        (piece_features, square_features, rule_features),
                        dim=1,
                    ),
                    piece_features=piece_features,
                    square_features=square_features,
                    rule_features=rule_features,
                )
            return DynamicsPrediction(
                next_features=decoded.next_features,
                piece_features=decoded.piece_features,
                square_features=decoded.square_features,
                rule_features=decoded.rule_features,
                next_latent=next_latent,
                piece_delta_features=piece_delta_features,
                square_delta_features=square_delta_features,
                rule_delta_features=rule_delta_features,
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
                next_latent=None,
                piece_delta_features=None,
                square_delta_features=None,
                rule_delta_features=None,
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
