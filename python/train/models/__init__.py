"""Model definitions for learned EngineKonzept components.

This package intentionally uses lazy exports so importing one model module does
not eagerly import the full training stack and trigger avoidable cycles.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from train.models.deliberation import (
        DELIBERATION_MODEL_NAME,
        CandidateSelector,
        DeliberationCell,
        DeliberationLoop,
        DeliberationTrace,
        DeliberationTraceStep,
        LatentTransition,
    )
    from train.models.dynamics import DYNAMICS_MODEL_NAME, LatentDynamicsModel
    from train.models.intention_encoder import (
        INTENTION_ENCODER_MODEL_NAME,
        KingSpecialHead,
        PieceIntentionEncoder,
    )
    from train.models.lapv1 import LAPV1_MODEL_NAME, LAPv1Config, LAPv1Model
    from train.models.opponent import OPPONENT_MODEL_NAME, OpponentHeadModel
    from train.models.planner import PLANNER_MODEL_NAME, PlannerHeadModel
    from train.models.policy_head_large import (
        LARGE_POLICY_HEAD_MODEL_NAME,
        LargePolicyHead,
    )
    from train.models.policy_head_nnue import (
        NNUE_POLICY_HEAD_MODEL_NAME,
        NNUEPolicyHead,
    )
    from train.models.proposer import LegalityPolicyProposer, MODEL_NAME, torch_is_available
    from train.models.state_embedder import RelationalStateEmbedder, STATE_EMBEDDER_MODEL_NAME
    from train.models.value_head import (
        SHARPNESS_HEAD_MODEL_NAME,
        VALUE_HEAD_MODEL_NAME,
        SharpnessHead,
        ValueHead,
    )
    from train.models.value_head_nnue import (
        NNUE_VALUE_HEAD_MODEL_NAME,
        ClippedReLU,
        NNUEValueHead,
    )


_EXPORTS: dict[str, tuple[str, str]] = {
    "DYNAMICS_MODEL_NAME": ("train.models.dynamics", "DYNAMICS_MODEL_NAME"),
    "LatentDynamicsModel": ("train.models.dynamics", "LatentDynamicsModel"),
    "DELIBERATION_MODEL_NAME": ("train.models.deliberation", "DELIBERATION_MODEL_NAME"),
    "CandidateSelector": ("train.models.deliberation", "CandidateSelector"),
    "DeliberationCell": ("train.models.deliberation", "DeliberationCell"),
    "DeliberationLoop": ("train.models.deliberation", "DeliberationLoop"),
    "DeliberationTrace": ("train.models.deliberation", "DeliberationTrace"),
    "DeliberationTraceStep": ("train.models.deliberation", "DeliberationTraceStep"),
    "LatentTransition": ("train.models.deliberation", "LatentTransition"),
    "INTENTION_ENCODER_MODEL_NAME": (
        "train.models.intention_encoder",
        "INTENTION_ENCODER_MODEL_NAME",
    ),
    "KingSpecialHead": ("train.models.intention_encoder", "KingSpecialHead"),
    "PieceIntentionEncoder": ("train.models.intention_encoder", "PieceIntentionEncoder"),
    "OPPONENT_MODEL_NAME": ("train.models.opponent", "OPPONENT_MODEL_NAME"),
    "OpponentHeadModel": ("train.models.opponent", "OpponentHeadModel"),
    "PLANNER_MODEL_NAME": ("train.models.planner", "PLANNER_MODEL_NAME"),
    "PlannerHeadModel": ("train.models.planner", "PlannerHeadModel"),
    "LARGE_POLICY_HEAD_MODEL_NAME": (
        "train.models.policy_head_large",
        "LARGE_POLICY_HEAD_MODEL_NAME",
    ),
    "LargePolicyHead": ("train.models.policy_head_large", "LargePolicyHead"),
    "NNUE_POLICY_HEAD_MODEL_NAME": (
        "train.models.policy_head_nnue",
        "NNUE_POLICY_HEAD_MODEL_NAME",
    ),
    "NNUEPolicyHead": ("train.models.policy_head_nnue", "NNUEPolicyHead"),
    "MODEL_NAME": ("train.models.proposer", "MODEL_NAME"),
    "LegalityPolicyProposer": ("train.models.proposer", "LegalityPolicyProposer"),
    "torch_is_available": ("train.models.proposer", "torch_is_available"),
    "STATE_EMBEDDER_MODEL_NAME": (
        "train.models.state_embedder",
        "STATE_EMBEDDER_MODEL_NAME",
    ),
    "RelationalStateEmbedder": ("train.models.state_embedder", "RelationalStateEmbedder"),
    "SHARPNESS_HEAD_MODEL_NAME": (
        "train.models.value_head",
        "SHARPNESS_HEAD_MODEL_NAME",
    ),
    "VALUE_HEAD_MODEL_NAME": ("train.models.value_head", "VALUE_HEAD_MODEL_NAME"),
    "SharpnessHead": ("train.models.value_head", "SharpnessHead"),
    "ValueHead": ("train.models.value_head", "ValueHead"),
    "NNUE_VALUE_HEAD_MODEL_NAME": (
        "train.models.value_head_nnue",
        "NNUE_VALUE_HEAD_MODEL_NAME",
    ),
    "ClippedReLU": ("train.models.value_head_nnue", "ClippedReLU"),
    "NNUEValueHead": ("train.models.value_head_nnue", "NNUEValueHead"),
}


def module_purpose() -> str:
    """Describe the current responsibility of the model package."""
    return "Legality/policy proposer, piece-intention encoder, state embedder, value/sharpness heads, large policy head, bounded deliberation, latent dynamics, opponent-head, and planner-head model definitions"


def __getattr__(name: str) -> Any:
    if name == "module_purpose":
        return module_purpose
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute_name = target
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "DYNAMICS_MODEL_NAME",
    "DELIBERATION_MODEL_NAME",
    "CandidateSelector",
    "DeliberationCell",
    "DeliberationLoop",
    "DeliberationTrace",
    "DeliberationTraceStep",
    "INTENTION_ENCODER_MODEL_NAME",
    "KingSpecialHead",
    "LAPV1_MODEL_NAME",
    "LAPv1Config",
    "LAPv1Model",
    "LegalityPolicyProposer",
    "LatentTransition",
    "LARGE_POLICY_HEAD_MODEL_NAME",
    "LargePolicyHead",
    "NNUE_POLICY_HEAD_MODEL_NAME",
    "NNUEPolicyHead",
    "LatentDynamicsModel",
    "MODEL_NAME",
    "OPPONENT_MODEL_NAME",
    "PLANNER_MODEL_NAME",
    "PlannerHeadModel",
    "PieceIntentionEncoder",
    "OpponentHeadModel",
    "RelationalStateEmbedder",
    "STATE_EMBEDDER_MODEL_NAME",
    "NNUE_VALUE_HEAD_MODEL_NAME",
    "SHARPNESS_HEAD_MODEL_NAME",
    "SharpnessHead",
    "VALUE_HEAD_MODEL_NAME",
    "ValueHead",
    "ClippedReLU",
    "NNUEValueHead",
    "module_purpose",
    "torch_is_available",
]
