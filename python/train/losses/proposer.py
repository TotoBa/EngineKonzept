"""Loss functions for the Phase-5 legality/policy proposer."""

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
class ProposerLossBreakdown:
    """Individual proposer loss components."""

    total: Any
    legality: Any
    policy: Any


def compute_proposer_losses(
    legality_logits: Any,
    policy_logits: Any,
    legal_targets: Any,
    selected_action_indices: Any,
    *,
    legality_weight: float,
    policy_weight: float,
) -> ProposerLossBreakdown:
    """Combine legality BCE and policy cross-entropy into a single training loss."""
    if torch is None or F is None:  # pragma: no cover - exercised when torch is absent
        raise RuntimeError(
            "PyTorch is required for proposer training. Install the 'train' extra or torch."
        )

    legality_loss = F.binary_cross_entropy_with_logits(legality_logits, legal_targets)

    policy_mask = selected_action_indices != -100
    if bool(policy_mask.any()):
        policy_loss = F.cross_entropy(
            policy_logits[policy_mask],
            selected_action_indices[policy_mask],
        )
    else:
        policy_loss = policy_logits.sum() * 0.0

    total_loss = legality_weight * legality_loss + policy_weight * policy_loss
    return ProposerLossBreakdown(
        total=total_loss,
        legality=legality_loss,
        policy=policy_loss,
    )
