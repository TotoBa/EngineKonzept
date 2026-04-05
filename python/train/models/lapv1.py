"""Model-only LAPv1 wrapper composing the new planner stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from train.config import (
    DeliberationConfig,
    IntentionEncoderConfig,
    LargePolicyHeadConfig,
    OpponentModelConfig,
    SharpnessHeadConfig,
    StateEmbedderConfig,
    ValueHeadConfig,
)
from train.datasets.artifacts import (
    POSITION_FEATURE_SIZE,
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
    TRANSITION_CONTEXT_FEATURE_SIZE,
)
from train.models.deliberation import DeliberationLoop
from train.models.intention_encoder import PieceIntentionEncoder
from train.models.opponent import OpponentHeadModel
from train.models.policy_head_large import LargePolicyHead
from train.models.state_embedder import RelationalStateEmbedder
from train.models.value_head import SharpnessHead, ValueHead

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


LAPV1_MODEL_NAME = "lapv1_wrapper"


@dataclass(frozen=True)
class LAPv1Config:
    """Assemble all standalone LAPv1 submodule configs in one wrapper."""

    intention_encoder: IntentionEncoderConfig = field(default_factory=IntentionEncoderConfig)
    state_embedder: StateEmbedderConfig = field(default_factory=StateEmbedderConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    sharpness_head: SharpnessHeadConfig = field(default_factory=SharpnessHeadConfig)
    policy_head: LargePolicyHeadConfig = field(default_factory=LargePolicyHeadConfig)
    opponent_head: OpponentModelConfig = field(
        default_factory=lambda: OpponentModelConfig(architecture="set_v2")
    )
    deliberation: DeliberationConfig = field(default_factory=DeliberationConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LAPv1Config":
        """Parse one nested LAPv1 wrapper config from a JSON-like mapping."""
        return cls(
            intention_encoder=IntentionEncoderConfig(
                **dict(payload.get("intention_encoder", {}))
            ),
            state_embedder=StateEmbedderConfig(**dict(payload.get("state_embedder", {}))),
            value_head=ValueHeadConfig(**dict(payload.get("value_head", {}))),
            sharpness_head=SharpnessHeadConfig(**dict(payload.get("sharpness_head", {}))),
            policy_head=LargePolicyHeadConfig(**dict(payload.get("policy_head", {}))),
            opponent_head=OpponentModelConfig(**dict(payload.get("opponent_head", {}))),
            deliberation=DeliberationConfig(**dict(payload.get("deliberation", {}))),
        )


if torch is not None and nn is not None:

    class _ValueProjectorAdapter(nn.Module):
        def __init__(self, value_head: ValueHead) -> None:
            super().__init__()
            self.value_head = value_head

        def forward(
            self,
            z_t: torch.Tensor,
            M_t: torch.Tensor,
            C_t: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del C_t
            _wdl_logits, cp_score, sigma_value = self.value_head(z_t, M_t)
            return cp_score.squeeze(1), sigma_value.squeeze(1)


    class _SharpnessProjectorAdapter(nn.Module):
        def __init__(self, sharpness_head: SharpnessHead) -> None:
            super().__init__()
            self.sharpness_head = sharpness_head

        def forward(self, z_t: torch.Tensor) -> torch.Tensor:
            return self.sharpness_head(z_t).squeeze(1)


    class _OpponentReplySignalProjector(nn.Module):
        def __init__(
            self,
            *,
            state_dim: int,
            global_dim: int,
            opponent_head: OpponentHeadModel,
        ) -> None:
            super().__init__()
            self.opponent_head = opponent_head
            self.root_projection = nn.Linear(state_dim, POSITION_FEATURE_SIZE)
            self.next_projection = nn.Linear(state_dim, POSITION_FEATURE_SIZE)
            self.transition_projection = nn.Linear(
                state_dim * 2,
                TRANSITION_CONTEXT_FEATURE_SIZE,
            )
            self.reply_global_projection = nn.Linear(
                global_dim,
                SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
            )

        def forward(
            self,
            transitioned_latents: torch.Tensor,
            *,
            z_t: torch.Tensor,
            selected_action_indices: torch.Tensor,
            candidate_action_indices: torch.Tensor,
            candidate_features: torch.Tensor,
            candidate_mask: torch.Tensor,
            global_features: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, selected_count, state_dim = transitioned_latents.shape
            candidate_count = candidate_action_indices.shape[1]
            root_expanded = z_t.unsqueeze(1).expand(-1, selected_count, -1)
            flat_root = self.root_projection(root_expanded.reshape(-1, state_dim))
            flat_next = self.next_projection(transitioned_latents.reshape(-1, state_dim))
            flat_transition = self.transition_projection(
                torch.cat([root_expanded, transitioned_latents], dim=2).reshape(
                    -1, state_dim * 2
                )
            )
            flat_reply_global = self.reply_global_projection(global_features).unsqueeze(1).expand(
                -1, selected_count, -1
            ).reshape(-1, SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE)
            flat_action_indices = selected_action_indices.reshape(-1)
            flat_candidate_action_indices = candidate_action_indices.unsqueeze(1).expand(
                -1, selected_count, -1
            ).reshape(-1, candidate_count)
            flat_candidate_features = candidate_features.unsqueeze(1).expand(
                -1,
                selected_count,
                -1,
                -1,
            ).reshape(-1, candidate_count, candidate_features.shape[2])
            flat_candidate_mask = candidate_mask.unsqueeze(1).expand(
                -1,
                selected_count,
                -1,
            ).reshape(-1, candidate_count)

            prediction = self.opponent_head(
                flat_root,
                flat_next,
                flat_action_indices,
                flat_transition,
                flat_reply_global,
                flat_candidate_action_indices,
                flat_candidate_features,
                flat_candidate_mask,
            )
            best_reply = prediction.reply_logits.masked_fill(
                ~flat_candidate_mask,
                float("-inf"),
            ).max(dim=1).values
            signal = best_reply - (prediction.pressure * 10.0) - (prediction.uncertainty * 10.0)
            return signal.reshape(batch_size, selected_count)


    class LAPv1Model(nn.Module):
        """Compose the LAPv1 stack up to bounded deliberation."""

        def __init__(self, config: LAPv1Config) -> None:
            super().__init__()
            self.config = config
            self.intention_encoder = PieceIntentionEncoder(
                hidden_dim=config.intention_encoder.hidden_dim,
                intention_dim=config.intention_encoder.intention_dim,
                num_layers=config.intention_encoder.num_layers,
                num_heads=config.intention_encoder.num_heads,
                feedforward_dim=config.intention_encoder.feedforward_dim,
                dropout=config.intention_encoder.dropout,
                max_edge_count=config.intention_encoder.max_edge_count,
            )
            self.state_embedder = RelationalStateEmbedder(
                intention_dim=config.state_embedder.intention_dim,
                square_input_dim=config.state_embedder.square_input_dim,
                global_dim=config.state_embedder.global_dim,
                hidden_dim=config.state_embedder.hidden_dim,
                state_dim=config.state_embedder.state_dim,
                num_layers=config.state_embedder.num_layers,
                num_heads=config.state_embedder.num_heads,
                feedforward_dim=config.state_embedder.feedforward_dim,
                dropout=config.state_embedder.dropout,
                max_edge_count=config.state_embedder.max_edge_count,
            )
            self.value_head = ValueHead(
                state_dim=config.value_head.state_dim,
                memory_dim=config.value_head.memory_dim,
                hidden_dim=config.value_head.hidden_dim,
                hidden_layers=config.value_head.hidden_layers,
                cp_score_cap=config.value_head.cp_score_cap,
                dropout=config.value_head.dropout,
            )
            self.sharpness_head = SharpnessHead(
                state_dim=config.sharpness_head.state_dim,
                hidden_dim=config.sharpness_head.hidden_dim,
                dropout=config.sharpness_head.dropout,
            )
            self.policy_head = LargePolicyHead(
                state_dim=config.policy_head.state_dim,
                hidden_dim=config.policy_head.hidden_dim,
                action_embedding_dim=config.policy_head.action_embedding_dim,
                num_layers=config.policy_head.num_layers,
                num_heads=config.policy_head.num_heads,
                feedforward_dim=config.policy_head.feedforward_dim,
                dropout=config.policy_head.dropout,
            )
            self.opponent_head = OpponentHeadModel(
                architecture=config.opponent_head.architecture,
                hidden_dim=config.opponent_head.hidden_dim,
                hidden_layers=config.opponent_head.hidden_layers,
                action_embedding_dim=config.opponent_head.action_embedding_dim,
                dropout=config.opponent_head.dropout,
            )
            self.deliberation_loop = DeliberationLoop(
                state_dim=config.deliberation.state_dim,
                memory_dim=config.deliberation.memory_dim,
                memory_slots=config.deliberation.memory_slots,
                action_embedding_dim=config.deliberation.action_embedding_dim,
                top_k_refine=config.deliberation.top_k_refine,
                max_inner_steps=config.deliberation.max_inner_steps,
                min_inner_steps=config.deliberation.min_inner_steps,
                q_threshold=config.deliberation.q_threshold,
                rollback_threshold=config.deliberation.rollback_threshold,
                top1_stable_steps=config.deliberation.top1_stable_steps,
                rollback_buffer_size=config.deliberation.rollback_buffer_size,
                value_projector=_ValueProjectorAdapter(self.value_head),
                sharpness_projector=_SharpnessProjectorAdapter(self.sharpness_head),
                reply_signal_projector=_OpponentReplySignalProjector(
                    state_dim=config.deliberation.state_dim,
                    global_dim=config.state_embedder.global_dim,
                    opponent_head=self.opponent_head,
                ),
            )

        def forward(
            self,
            piece_tokens: torch.Tensor,
            square_tokens: torch.Tensor,
            state_context_v1_global: torch.Tensor,
            reachability_edges: torch.Tensor,
            candidate_context_v2: torch.Tensor,
            candidate_action_indices: torch.Tensor,
            candidate_mask: torch.Tensor,
            *,
            candidate_uci: list[list[str]] | None = None,
            single_legal_move: bool = False,
        ) -> dict[str, Any]:
            """Run the full LAPv1 wrapper forward pass without trainer glue."""
            piece_intentions = self.intention_encoder(
                piece_tokens,
                state_context_v1_global,
                reachability_edges,
            )
            z_root, _sigma_root = self.state_embedder(
                piece_intentions,
                square_tokens,
                state_context_v1_global,
                reachability_edges,
            )
            initial_policy_logits = self.policy_head(
                z_root,
                candidate_context_v2,
                candidate_action_indices,
                candidate_mask,
            )
            deliberation_outputs = self.deliberation_loop(
                z_root,
                candidate_action_indices,
                initial_policy_logits,
                candidate_mask,
                single_legal_move=single_legal_move,
                candidate_uci=candidate_uci,
                candidate_features=candidate_context_v2,
                global_features=state_context_v1_global,
            )
            wdl_logits, cp_score, sigma_value = self.value_head(
                deliberation_outputs["final_z"],
                deliberation_outputs["final_memory"],
            )
            return {
                "final_policy_logits": deliberation_outputs["final_candidate_scores"],
                "final_value": {
                    "wdl_logits": wdl_logits,
                    "cp_score": cp_score,
                    "sigma_value": sigma_value,
                },
                "deliberation_trace": deliberation_outputs["trace"],
                "step_value_cp_tensors": deliberation_outputs["step_value_cp_tensors"],
                "step_sharpness_tensors": deliberation_outputs["step_sharpness_tensors"],
                "step_rollback_flags": deliberation_outputs["step_rollback_flags"],
                "refined_top1_action_index": deliberation_outputs[
                    "refined_top1_action_index"
                ],
                "piece_intentions": piece_intentions,
                "z_root": z_root,
            }

else:  # pragma: no cover - exercised when torch is absent

    class LAPv1Model:  # type: ignore[no-redef]
        """Stub placeholder when torch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for LAPv1Model. Install the 'train' extra or torch."
            )
