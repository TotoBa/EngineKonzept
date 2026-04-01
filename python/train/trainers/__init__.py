"""Training loops and evaluation helpers for EngineKonzept models."""

from train.trainers.proposer import (
    ProposerMetrics,
    ProposerTrainingRun,
    evaluate_proposer_checkpoint,
    train_proposer,
)


def module_purpose() -> str:
    """Describe the current responsibility of the trainer package."""
    return "Legality/policy proposer training loops and metrics"


__all__ = [
    "ProposerMetrics",
    "ProposerTrainingRun",
    "evaluate_proposer_checkpoint",
    "module_purpose",
    "train_proposer",
]
