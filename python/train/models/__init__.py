"""Model definitions for learned EngineKonzept components."""

from train.models.proposer import LegalityPolicyProposer, MODEL_NAME, torch_is_available


def module_purpose() -> str:
    """Describe the current responsibility of the model package."""
    return "Legality/policy proposer model definitions"


__all__ = [
    "LegalityPolicyProposer",
    "MODEL_NAME",
    "module_purpose",
    "torch_is_available",
]
