"""Loss helpers for proposer and later planner training."""

from train.losses.dynamics import DynamicsLossBreakdown, compute_dynamics_losses
from train.losses.proposer import ProposerLossBreakdown, compute_proposer_losses


def module_purpose() -> str:
    """Describe the current responsibility of the loss package."""
    return "Legality/policy proposer losses and latent dynamics reconstruction losses"

__all__ = [
    "DynamicsLossBreakdown",
    "ProposerLossBreakdown",
    "compute_dynamics_losses",
    "compute_proposer_losses",
    "module_purpose",
]
