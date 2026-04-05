"""Model definitions for learned EngineKonzept components."""

from train.models.dynamics import DYNAMICS_MODEL_NAME, LatentDynamicsModel
from train.models.intention_encoder import (
    INTENTION_ENCODER_MODEL_NAME,
    KingSpecialHead,
    PieceIntentionEncoder,
)
from train.models.opponent import OPPONENT_MODEL_NAME, OpponentHeadModel
from train.models.planner import PLANNER_MODEL_NAME, PlannerHeadModel
from train.models.policy_head_large import (
    LARGE_POLICY_HEAD_MODEL_NAME,
    LargePolicyHead,
)
from train.models.proposer import LegalityPolicyProposer, MODEL_NAME, torch_is_available
from train.models.state_embedder import RelationalStateEmbedder, STATE_EMBEDDER_MODEL_NAME
from train.models.value_head import (
    SHARPNESS_HEAD_MODEL_NAME,
    VALUE_HEAD_MODEL_NAME,
    SharpnessHead,
    ValueHead,
)


def module_purpose() -> str:
    """Describe the current responsibility of the model package."""
    return "Legality/policy proposer, piece-intention encoder, state embedder, value/sharpness heads, large policy head, latent dynamics, opponent-head, and planner-head model definitions"


__all__ = [
    "DYNAMICS_MODEL_NAME",
    "INTENTION_ENCODER_MODEL_NAME",
    "KingSpecialHead",
    "LegalityPolicyProposer",
    "LARGE_POLICY_HEAD_MODEL_NAME",
    "LargePolicyHead",
    "LatentDynamicsModel",
    "MODEL_NAME",
    "OPPONENT_MODEL_NAME",
    "PLANNER_MODEL_NAME",
    "PlannerHeadModel",
    "PieceIntentionEncoder",
    "OpponentHeadModel",
    "RelationalStateEmbedder",
    "STATE_EMBEDDER_MODEL_NAME",
    "SHARPNESS_HEAD_MODEL_NAME",
    "SharpnessHead",
    "VALUE_HEAD_MODEL_NAME",
    "ValueHead",
    "module_purpose",
    "torch_is_available",
]
