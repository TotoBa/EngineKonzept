"""Loss functions for the Phase-7 opponent head."""

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
class OpponentLossBreakdown:
    """Individual opponent-head loss components."""

    total: Any
    reply_policy: Any
    pressure: Any
    uncertainty: Any


def compute_opponent_losses(
    predicted_reply_logits: Any,
    *,
    reply_mask: Any,
    target_reply_policy: Any,
    supervised_mask: Any,
    reply_example_weights: Any | None,
    predicted_pressure: Any,
    target_pressure: Any,
    predicted_uncertainty: Any,
    target_uncertainty: Any,
    reply_policy_weight: float,
    pressure_weight: float,
    uncertainty_weight: float,
) -> OpponentLossBreakdown:
    """Combine reply-policy and scalar auxiliary losses into one training objective."""
    if torch is None or F is None:  # pragma: no cover - exercised when torch is absent
        raise RuntimeError(
            "PyTorch is required for opponent-head training. Install the 'train' extra or torch."
        )

    masked_logits = predicted_reply_logits.masked_fill(~reply_mask, -1e9)
    reply_log_probs = torch.log_softmax(masked_logits, dim=1)
    reply_loss_per_example = -(target_reply_policy * reply_log_probs).sum(dim=1)
    if bool(supervised_mask.any()):
        supervised_losses = reply_loss_per_example[supervised_mask]
        if reply_example_weights is None:
            reply_policy_loss = supervised_losses.mean()
        else:
            supervised_weights = reply_example_weights[supervised_mask].clamp_min(1e-6)
            reply_policy_loss = (supervised_losses * supervised_weights).sum() / supervised_weights.sum()
    else:
        reply_policy_loss = predicted_reply_logits.sum() * 0.0
    pressure_loss = F.mse_loss(predicted_pressure, target_pressure)
    uncertainty_loss = F.mse_loss(predicted_uncertainty, target_uncertainty)
    total_loss = (
        reply_policy_weight * reply_policy_loss
        + pressure_weight * pressure_loss
        + uncertainty_weight * uncertainty_loss
    )
    return OpponentLossBreakdown(
        total=total_loss,
        reply_policy=reply_policy_loss,
        pressure=pressure_loss,
        uncertainty=uncertainty_loss,
    )
