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


def compute_dynamics_losses(
    predicted_next_features: Any,
    next_feature_targets: Any,
    *,
    reconstruction_weight: float,
) -> DynamicsLossBreakdown:
    """Combine one-step next-state reconstruction terms into a training loss."""
    if torch is None or F is None:  # pragma: no cover - exercised when torch is absent
        raise RuntimeError(
            "PyTorch is required for dynamics training. Install the 'train' extra or torch."
        )

    reconstruction_loss = F.mse_loss(predicted_next_features, next_feature_targets)
    total_loss = reconstruction_weight * reconstruction_loss
    return DynamicsLossBreakdown(total=total_loss, reconstruction=reconstruction_loss)
