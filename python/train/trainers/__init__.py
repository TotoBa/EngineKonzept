"""Training loops and evaluation helpers for EngineKonzept models."""

from train.trainers.dynamics import (
    DynamicsMetrics,
    DynamicsTrainingRun,
    evaluate_dynamics_checkpoint,
    train_dynamics,
)
from train.trainers.opponent import (
    OpponentMetrics,
    OpponentTrainingRun,
    evaluate_opponent_checkpoint,
    train_opponent,
)
from train.trainers.proposer import (
    ProposerMetrics,
    ProposerTrainingRun,
    evaluate_proposer_checkpoint,
    train_proposer,
)


def module_purpose() -> str:
    """Describe the current responsibility of the trainer package."""
    return "Legality/policy proposer, latent dynamics, and opponent-head training loops and metrics"


__all__ = [
    "DynamicsMetrics",
    "DynamicsTrainingRun",
    "OpponentMetrics",
    "OpponentTrainingRun",
    "ProposerMetrics",
    "ProposerTrainingRun",
    "evaluate_dynamics_checkpoint",
    "evaluate_opponent_checkpoint",
    "evaluate_proposer_checkpoint",
    "module_purpose",
    "train_opponent",
    "train_dynamics",
    "train_proposer",
]
