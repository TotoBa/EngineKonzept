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
    reconstruction_weight: float,
    piece_weight: float,
    square_weight: float,
    rule_weight: float,
) -> DynamicsLossBreakdown:
    """Combine section-wise one-step next-state reconstruction terms into a training loss."""
    if torch is None or F is None:  # pragma: no cover - exercised when torch is absent
        raise RuntimeError(
            "PyTorch is required for dynamics training. Install the 'train' extra or torch."
        )

    piece_loss = F.mse_loss(predicted_piece_features, target_piece_features)
    square_loss = F.mse_loss(predicted_square_features, target_square_features)
    rule_loss = F.mse_loss(predicted_rule_features, target_rule_features)
    reconstruction_loss = F.mse_loss(predicted_next_features, next_feature_targets)
    total_loss = reconstruction_weight * (
        piece_weight * piece_loss
        + square_weight * square_loss
        + rule_weight * rule_loss
    )
    return DynamicsLossBreakdown(
        total=total_loss,
        reconstruction=reconstruction_loss,
        piece=piece_loss,
        square=square_loss,
        rule=rule_loss,
    )
