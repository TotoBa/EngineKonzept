"""Loss functions for the Phase-6 latent dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    F = None


@dataclass(frozen=True)
class DynamicsLossBreakdown:
    """Individual dynamics loss components."""

    total: Any
    reconstruction: Any
    piece: Any
    square: Any
    rule: Any
    latent_consistency: Any
    delta: Any


def compute_dynamics_losses(
    predicted_next_features: Any,
    next_feature_targets: Any,
    *,
    predicted_piece_features: Any,
    predicted_square_features: Any,
    predicted_rule_features: Any,
    target_piece_features: Any,
    target_square_features: Any,
    target_rule_features: Any,
    predicted_piece_delta_features: Any | None,
    predicted_square_delta_features: Any | None,
    predicted_rule_delta_features: Any | None,
    target_piece_delta_features: Any,
    target_square_delta_features: Any,
    target_rule_delta_features: Any,
    predicted_next_latent: Any | None,
    target_next_latent: Any | None,
    reconstruction_weight: float,
    piece_weight: float,
    square_weight: float,
    rule_weight: float,
    delta_loss_weight: float,
    latent_consistency_weight: float,
) -> DynamicsLossBreakdown:
    """Combine section-wise one-step next-state reconstruction terms into a training loss."""
    if torch is None or F is None:  # pragma: no cover - exercised when torch is absent
        raise RuntimeError(
            "PyTorch is required for dynamics training. Install the 'train' extra or torch."
        )

    reconstruction_loss = F.mse_loss(predicted_next_features, next_feature_targets)
    piece_loss = F.mse_loss(predicted_piece_features, target_piece_features)
    square_loss = F.mse_loss(predicted_square_features, target_square_features)
    rule_loss = F.mse_loss(predicted_rule_features, target_rule_features)
    if (
        predicted_piece_delta_features is None
        or predicted_square_delta_features is None
        or predicted_rule_delta_features is None
    ):
        delta_loss = reconstruction_loss * 0.0
    else:
        delta_loss = (
            F.mse_loss(predicted_piece_delta_features, target_piece_delta_features)
            + F.mse_loss(predicted_square_delta_features, target_square_delta_features)
            + F.mse_loss(predicted_rule_delta_features, target_rule_delta_features)
        )
    if predicted_next_latent is None or target_next_latent is None:
        latent_consistency_loss = reconstruction_loss * 0.0
    else:
        latent_consistency_loss = F.mse_loss(predicted_next_latent, target_next_latent)
    total_loss = reconstruction_weight * (
        piece_weight * piece_loss
        + square_weight * square_loss
        + rule_weight * rule_loss
    ) + (delta_loss_weight * delta_loss) + (latent_consistency_weight * latent_consistency_loss)
    return DynamicsLossBreakdown(
        total=total_loss,
        reconstruction=reconstruction_loss,
        piece=piece_loss,
        square=square_loss,
        rule=rule_loss,
        latent_consistency=latent_consistency_loss,
        delta=delta_loss,
    )
