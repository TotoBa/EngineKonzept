"""Model definitions for learned EngineKonzept components."""

from train.models.dynamics import DYNAMICS_MODEL_NAME, LatentDynamicsModel
from train.models.opponent import OPPONENT_MODEL_NAME, OpponentHeadModel
from train.models.planner import PLANNER_MODEL_NAME, PlannerHeadModel
from train.models.proposer import LegalityPolicyProposer, MODEL_NAME, torch_is_available


def module_purpose() -> str:
    """Describe the current responsibility of the model package."""
    return "Legality/policy proposer, latent dynamics, opponent-head, and planner-head model definitions"


__all__ = [
    "DYNAMICS_MODEL_NAME",
    "LegalityPolicyProposer",
    "LatentDynamicsModel",
    "MODEL_NAME",
    "OPPONENT_MODEL_NAME",
    "PLANNER_MODEL_NAME",
    "PlannerHeadModel",
    "OpponentHeadModel",
    "module_purpose",
    "torch_is_available",
]
