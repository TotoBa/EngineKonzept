"""Training loops and evaluation helpers for EngineKonzept models."""

from train.trainers.dynamics import (
    DynamicsMetrics,
    DynamicsTrainingRun,
    evaluate_dynamics_checkpoint,
    train_dynamics,
)
from train.trainers.lapv1 import (
    LAPv1Metrics,
    LAPv1TrainingRun,
    count_lapv1_model_parameters,
    evaluate_lapv1_checkpoint,
    load_lapv1_train_config,
    train_lapv1,
)
from train.trainers.opponent import (
    OpponentMetrics,
    OpponentTrainingRun,
    evaluate_opponent_checkpoint,
    train_opponent,
)
from train.trainers.planner import (
    PlannerMetrics,
    PlannerTrainingRun,
    evaluate_planner_checkpoint,
    train_planner,
)
from train.trainers.proposer import (
    ProposerMetrics,
    ProposerTrainingRun,
    evaluate_proposer_checkpoint,
    train_proposer,
)


def module_purpose() -> str:
    """Describe the current responsibility of the trainer package."""
    return "Legality/policy proposer, LAPv1 static-head, latent dynamics, opponent-head, and planner-head training loops and metrics"


__all__ = [
    "DynamicsMetrics",
    "DynamicsTrainingRun",
    "LAPv1Metrics",
    "LAPv1TrainingRun",
    "count_lapv1_model_parameters",
    "OpponentMetrics",
    "OpponentTrainingRun",
    "PlannerMetrics",
    "PlannerTrainingRun",
    "ProposerMetrics",
    "ProposerTrainingRun",
    "evaluate_dynamics_checkpoint",
    "evaluate_lapv1_checkpoint",
    "load_lapv1_train_config",
    "evaluate_opponent_checkpoint",
    "evaluate_planner_checkpoint",
    "evaluate_proposer_checkpoint",
    "module_purpose",
    "train_opponent",
    "train_lapv1",
    "train_dynamics",
    "train_planner",
    "train_proposer",
]
