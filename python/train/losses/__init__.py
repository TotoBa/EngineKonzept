"""Loss helpers for proposer and later planner training."""

from train.losses.proposer import ProposerLossBreakdown, compute_proposer_losses


def module_purpose() -> str:
    """Describe the current responsibility of the loss package."""
    return "Legality and policy loss functions"


__all__ = ["ProposerLossBreakdown", "compute_proposer_losses", "module_purpose"]
