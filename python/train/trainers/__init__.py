"""Training loops and evaluation helpers for EngineKonzept models."""

from train.trainers.dynamics import (
    DynamicsMetrics,
    DynamicsTrainingRun,
    evaluate_dynamics_checkpoint,
    train_dynamics,
)
from train.trainers.proposer import (
    ProposerMetrics,
    ProposerTrainingRun,
    evaluate_proposer_checkpoint,
    train_proposer,
)


def module_purpose() -> str:
    """Describe the current responsibility of the trainer package."""
    return "Legality/policy proposer and latent dynamics training loops and metrics"


__all__ = [
    "DynamicsMetrics",
    "DynamicsTrainingRun",
    "ProposerMetrics",
    "ProposerTrainingRun",
    "evaluate_dynamics_checkpoint",
    "evaluate_proposer_checkpoint",
    "module_purpose",
    "train_dynamics",
    "train_proposer",
]
