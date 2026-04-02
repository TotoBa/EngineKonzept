"""Model definitions for learned EngineKonzept components."""

from train.models.dynamics import DYNAMICS_MODEL_NAME, LatentDynamicsModel
from train.models.proposer import LegalityPolicyProposer, MODEL_NAME, torch_is_available


def module_purpose() -> str:
    """Describe the current responsibility of the model package."""
    return "Legality/policy proposer and latent dynamics model definitions"


__all__ = [
    "DYNAMICS_MODEL_NAME",
    "LegalityPolicyProposer",
    "LatentDynamicsModel",
    "MODEL_NAME",
    "module_purpose",
    "torch_is_available",
]
