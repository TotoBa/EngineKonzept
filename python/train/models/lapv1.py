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
from train.datasets.nnue_features import TOTAL_FEATURES
from train.models.deliberation import DeliberationLoop
from train.models.dual_accumulator import (
    AccumulatorCache,
    DualAccumulatorBuilder,
    build_sparse_rows,
    pack_sparse_feature_lists,
    unpack_sparse_row,
)
from train.models.feature_transformer import FeatureTransformer
from train.models.intention_encoder import PieceIntentionEncoder
from train.models.opponent import OpponentHeadModel
from train.models.opponent_readout import OpponentReadout
from train.models.phase_moe import PhaseMoE
from train.models.phase_router import PhaseRouter
from train.models.policy_head_large import LargePolicyHead
from train.models.policy_head_nnue import NNUEPolicyHead
from train.models.state_embedder import RelationalStateEmbedder
from train.models.value_head import SharpnessHead, ValueHead
from train.models.value_head_nnue import NNUEValueHead

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


LAPV1_MODEL_NAME = "lapv1_wrapper"
LAPV2_MODEL_VERSION = 1


@dataclass(frozen=True)
class LAPv2LossBalanceConfig:
    """Shared-FT loss balancing controls for LAPv2 NNUE heads."""

    value_loss_norm: str = "none"
    policy_loss_norm: str = "none"
    adapter_decoupling: float = 0.0

    def __post_init__(self) -> None:
        allowed_norms = {"none", "ema"}
        if self.value_loss_norm not in allowed_norms:
            raise ValueError("lapv2.loss_balance.value_loss_norm must be 'none' or 'ema'")
        if self.policy_loss_norm not in allowed_norms:
            raise ValueError("lapv2.loss_balance.policy_loss_norm must be 'none' or 'ema'")
        if self.adapter_decoupling < 0.0:
            raise ValueError("lapv2.loss_balance.adapter_decoupling must be non-negative")


@dataclass(frozen=True)
class LAPv2Config:
    """Feature-flagged LAPv2 upgrades layered onto the LAPv1 wrapper."""

    enabled: bool = False
    phase_moe: bool = False
    dual_accumulator: bool = False
    nnue_value: bool = False
    nnue_value_phase_moe: bool = False
    nnue_phase_gate_steps: int = 0
    nnue_policy: bool = False
    sharpness_phase_moe: bool = False
    shared_opponent_readout: bool = False
    distill_opponent: bool = False
    distill_fraction: float = 0.25
    distill_reply_weight: float = 1.0
    distill_pressure_weight: float = 0.5
    distill_uncertainty_weight: float = 0.5
    accumulator_cache: bool = False
    N_accumulator: int = 64
    loss_balance: LAPv2LossBalanceConfig = field(default_factory=LAPv2LossBalanceConfig)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LAPv2Config":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            phase_moe=bool(payload.get("phase_moe", False)),
            dual_accumulator=bool(payload.get("dual_accumulator", False)),
            nnue_value=bool(payload.get("nnue_value", False)),
            nnue_value_phase_moe=bool(payload.get("nnue_value_phase_moe", False)),
            nnue_phase_gate_steps=int(payload.get("nnue_phase_gate_steps", 0)),
            nnue_policy=bool(payload.get("nnue_policy", False)),
            sharpness_phase_moe=bool(payload.get("sharpness_phase_moe", False)),
            shared_opponent_readout=bool(payload.get("shared_opponent_readout", False)),
            distill_opponent=bool(payload.get("distill_opponent", False)),
            distill_fraction=float(payload.get("distill_fraction", 0.25)),
            distill_reply_weight=float(payload.get("distill_reply_weight", 1.0)),
            distill_pressure_weight=float(payload.get("distill_pressure_weight", 0.5)),
            distill_uncertainty_weight=float(payload.get("distill_uncertainty_weight", 0.5)),
            accumulator_cache=bool(payload.get("accumulator_cache", False)),
            N_accumulator=int(payload.get("N_accumulator", 64)),
            loss_balance=LAPv2LossBalanceConfig(
                **dict(payload.get("loss_balance", {}))
            ),
        )

    def __post_init__(self) -> None:
        if self.N_accumulator <= 0:
            raise ValueError("lapv2.N_accumulator must be positive")
        if self.nnue_phase_gate_steps < 0:
            raise ValueError("lapv2.nnue_phase_gate_steps must be non-negative")
        if self.nnue_policy_enabled and not self.nnue_value_enabled:
            raise ValueError("lapv2.nnue_policy requires lapv2.nnue_value")
        if self.distill_opponent and not self.shared_opponent_readout:
            raise ValueError("lapv2.distill_opponent requires lapv2.shared_opponent_readout")
        if not 0.0 <= self.distill_fraction <= 1.0:
            raise ValueError("lapv2.distill_fraction must be in [0.0, 1.0]")
        if self.distill_reply_weight < 0.0:
            raise ValueError("lapv2.distill_reply_weight must be non-negative")
        if self.distill_pressure_weight < 0.0:
            raise ValueError("lapv2.distill_pressure_weight must be non-negative")
        if self.distill_uncertainty_weight < 0.0:
            raise ValueError("lapv2.distill_uncertainty_weight must be non-negative")

    @property
    def phase_moe_enabled(self) -> bool:
        return self.enabled and self.phase_moe

    @property
    def nnue_value_enabled(self) -> bool:
        return self.enabled and self.nnue_value

    @property
    def nnue_value_phase_moe_enabled(self) -> bool:
        return self.nnue_value_enabled and self.nnue_value_phase_moe

    @property
    def nnue_policy_enabled(self) -> bool:
        return self.enabled and self.nnue_policy

    @property
    def sharpness_phase_moe_enabled(self) -> bool:
        return self.enabled and self.sharpness_phase_moe


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
    lapv2: LAPv2Config = field(default_factory=LAPv2Config)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LAPv1Config":
        """Parse one nested LAPv1 wrapper config from a JSON-like mapping."""
        opponent_payload = {
            "architecture": "set_v2",
            **dict(payload.get("opponent_head", {})),
        }
        return cls(
            intention_encoder=IntentionEncoderConfig(
                **dict(payload.get("intention_encoder", {}))
            ),
            state_embedder=StateEmbedderConfig(**dict(payload.get("state_embedder", {}))),
            value_head=ValueHeadConfig(**dict(payload.get("value_head", {}))),
            sharpness_head=SharpnessHeadConfig(**dict(payload.get("sharpness_head", {}))),
            policy_head=LargePolicyHeadConfig(**dict(payload.get("policy_head", {}))),
            opponent_head=OpponentModelConfig(**opponent_payload),
            deliberation=DeliberationConfig(**dict(payload.get("deliberation", {}))),
            lapv2=LAPv2Config.from_mapping(dict(payload.get("lapv2", {}))),
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
        def __init__(self, sharpness_head: SharpnessHead | PhaseMoE) -> None:
            super().__init__()
            self.sharpness_head = sharpness_head
            self._phase_idx: torch.Tensor | None = None

        def set_phase_idx(self, phase_idx: torch.Tensor | None) -> None:
            self._phase_idx = phase_idx

        def forward(self, z_t: torch.Tensor) -> torch.Tensor:
            if isinstance(self.sharpness_head, PhaseMoE):
                if self._phase_idx is None:
                    raise ValueError(
                        "phase_idx is required when sharpness head is phase-routed"
                    )
                return self.sharpness_head(z_t, phase_idx=self._phase_idx).squeeze(1)
            return self.sharpness_head(z_t).squeeze(1)


    class _OpponentReplySignalProjector(nn.Module):
        def __init__(
            self,
            *,
            state_dim: int,
            global_dim: int,
            opponent_head: OpponentHeadModel | None = None,
            opponent_readout: OpponentReadout | None = None,
        ) -> None:
            super().__init__()
            if opponent_head is None and opponent_readout is None:
                raise ValueError(
                    "at least one of opponent_head or opponent_readout must be provided"
                )
            self.opponent_head = opponent_head
            self.opponent_readout = opponent_readout
            if opponent_head is not None:
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
            else:
                self.root_projection = None
                self.next_projection = None
                self.transition_projection = None
                self.reply_global_projection = None

        def _legacy_prediction(
            self,
            *,
            transitioned_latents: torch.Tensor,
            z_t: torch.Tensor,
            selected_count: int,
            state_dim: int,
            flat_action_indices: torch.Tensor,
            flat_candidate_action_indices: torch.Tensor,
            flat_candidate_features: torch.Tensor,
            flat_candidate_mask: torch.Tensor,
            flat_reply_global: torch.Tensor,
        ) -> Any:
            assert self.opponent_head is not None
            assert self.root_projection is not None
            assert self.next_projection is not None
            assert self.transition_projection is not None
            assert self.reply_global_projection is not None
            root_expanded = z_t.unsqueeze(1).expand(-1, selected_count, -1)
            flat_root = self.root_projection(root_expanded.reshape(-1, state_dim))
            flat_next = self.next_projection(
                transitioned_latents.reshape(-1, state_dim)
            )
            flat_transition = self.transition_projection(
                torch.cat([root_expanded, transitioned_latents], dim=2).reshape(
                    -1, state_dim * 2
                )
            )
            return self.opponent_head(
                flat_root,
                flat_next,
                flat_action_indices,
                flat_transition,
                self.reply_global_projection(flat_reply_global),
                flat_candidate_action_indices,
                flat_candidate_features,
                flat_candidate_mask,
            )

        def predict(
            self,
            transitioned_latents: torch.Tensor,
            *,
            z_t: torch.Tensor,
            selected_action_indices: torch.Tensor,
            candidate_action_indices: torch.Tensor,
            candidate_features: torch.Tensor,
            candidate_mask: torch.Tensor,
            global_features: torch.Tensor,
            collect_distill_targets: bool = False,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
            batch_size, selected_count, state_dim = transitioned_latents.shape
            candidate_count = candidate_action_indices.shape[1]
            flat_reply_global = global_features.unsqueeze(1).expand(
                -1, selected_count, -1
            ).reshape(-1, global_features.shape[1])
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
            if self.opponent_readout is not None:
                active_prediction = self.opponent_readout(
                    transitioned_latents.reshape(-1, state_dim),
                    flat_action_indices,
                    flat_reply_global,
                    flat_candidate_action_indices,
                    flat_candidate_features,
                    flat_candidate_mask,
                )
            else:
                active_prediction = self._legacy_prediction(
                    transitioned_latents=transitioned_latents,
                    z_t=z_t,
                    selected_count=selected_count,
                    state_dim=state_dim,
                    flat_action_indices=flat_action_indices,
                    flat_candidate_action_indices=flat_candidate_action_indices,
                    flat_candidate_features=flat_candidate_features,
                    flat_candidate_mask=flat_candidate_mask,
                    flat_reply_global=flat_reply_global,
                )
            teacher_prediction = None
            if (
                collect_distill_targets
                and self.opponent_readout is not None
                and self.opponent_head is not None
            ):
                teacher_prediction = self._legacy_prediction(
                    transitioned_latents=transitioned_latents,
                    z_t=z_t,
                    selected_count=selected_count,
                    state_dim=state_dim,
                    flat_action_indices=flat_action_indices,
                    flat_candidate_action_indices=flat_candidate_action_indices,
                    flat_candidate_features=flat_candidate_features,
                    flat_candidate_mask=flat_candidate_mask,
                    flat_reply_global=flat_reply_global,
                )
            best_reply = active_prediction.reply_logits.masked_fill(
                ~flat_candidate_mask,
                float("-inf"),
            ).max(dim=1).values
            signal = best_reply - (active_prediction.pressure * 10.0) - (
                active_prediction.uncertainty * 10.0
            )
            return signal.reshape(batch_size, selected_count), {
                "student_reply_logits": active_prediction.reply_logits.reshape(
                    batch_size,
                    selected_count,
                    candidate_count,
                ),
                "student_pressure": active_prediction.pressure.reshape(
                    batch_size,
                    selected_count,
                ),
                "student_uncertainty": active_prediction.uncertainty.reshape(
                    batch_size,
                    selected_count,
                ),
                "teacher_reply_logits": (
                    None
                    if teacher_prediction is None
                    else teacher_prediction.reply_logits.reshape(
                        batch_size,
                        selected_count,
                        candidate_count,
                    )
                ),
                "teacher_pressure": (
                    None
                    if teacher_prediction is None
                    else teacher_prediction.pressure.reshape(batch_size, selected_count)
                ),
                "teacher_uncertainty": (
                    None
                    if teacher_prediction is None
                    else teacher_prediction.uncertainty.reshape(batch_size, selected_count)
                ),
            }

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
            signal, _ = self.predict(
                transitioned_latents,
                z_t=z_t,
                selected_action_indices=selected_action_indices,
                candidate_action_indices=candidate_action_indices,
                candidate_features=candidate_features,
                candidate_mask=candidate_mask,
                global_features=global_features,
                collect_distill_targets=False,
            )
            return signal


    class LAPv1Model(nn.Module):
        """Compose the LAPv1 stack up to bounded deliberation."""

        def __init__(self, config: LAPv1Config) -> None:
            super().__init__()
            self.config = config
            intention_encoder = PieceIntentionEncoder(
                hidden_dim=config.intention_encoder.hidden_dim,
                intention_dim=config.intention_encoder.intention_dim,
                num_layers=config.intention_encoder.num_layers,
                num_heads=config.intention_encoder.num_heads,
                feedforward_dim=config.intention_encoder.feedforward_dim,
                dropout=config.intention_encoder.dropout,
                max_edge_count=config.intention_encoder.max_edge_count,
            )
            state_embedder = RelationalStateEmbedder(
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
            if config.lapv2.phase_moe_enabled:
                self.phase_router: PhaseRouter | None = PhaseRouter()
                self.intention_encoder = PhaseMoE.from_single(intention_encoder)
                self.state_embedder = PhaseMoE.from_single(state_embedder)
            else:
                self.phase_router = None
                self.intention_encoder = intention_encoder
                self.state_embedder = state_embedder
            self.value_head = ValueHead(
                state_dim=config.value_head.state_dim,
                memory_dim=config.value_head.memory_dim,
                hidden_dim=config.value_head.hidden_dim,
                hidden_layers=config.value_head.hidden_layers,
                cp_score_cap=config.value_head.cp_score_cap,
                dropout=config.value_head.dropout,
            )
            if config.lapv2.nnue_value_enabled:
                ft: FeatureTransformer | PhaseMoE = FeatureTransformer(
                    num_features=TOTAL_FEATURES,
                    accumulator_dim=config.lapv2.N_accumulator,
                )
                value_head_nnue: NNUEValueHead | PhaseMoE = NNUEValueHead(
                    accumulator_dim=config.lapv2.N_accumulator,
                    hidden_dim=32,
                    cp_score_cap=config.value_head.cp_score_cap,
                )
                if config.lapv2.nnue_value_phase_moe_enabled:
                    if self.phase_router is None:
                        self.phase_router = PhaseRouter()
                    ft = PhaseMoE.from_single(ft)
                    value_head_nnue = PhaseMoE.from_single(value_head_nnue)
                self.ft: FeatureTransformer | PhaseMoE | None = ft
                self.dual_acc_builder: DualAccumulatorBuilder | None = DualAccumulatorBuilder()
                self.value_head_nnue: NNUEValueHead | PhaseMoE | None = value_head_nnue
            else:
                self.ft = None
                self.dual_acc_builder = None
                self.value_head_nnue = None
            sharpness_head: SharpnessHead | PhaseMoE = SharpnessHead(
                state_dim=config.sharpness_head.state_dim,
                hidden_dim=config.sharpness_head.hidden_dim,
                dropout=config.sharpness_head.dropout,
            )
            if config.lapv2.enabled and config.lapv2.sharpness_phase_moe:
                if self.phase_router is None:
                    self.phase_router = PhaseRouter()
                sharpness_head = PhaseMoE.from_single(sharpness_head)
            self.sharpness_head = sharpness_head
            self._sharpness_projector = _SharpnessProjectorAdapter(self.sharpness_head)
            self.policy_head = LargePolicyHead(
                state_dim=config.policy_head.state_dim,
                hidden_dim=config.policy_head.hidden_dim,
                action_embedding_dim=config.policy_head.action_embedding_dim,
                num_layers=config.policy_head.num_layers,
                num_heads=config.policy_head.num_heads,
                feedforward_dim=config.policy_head.feedforward_dim,
                dropout=config.policy_head.dropout,
            )
            if config.lapv2.nnue_policy_enabled:
                policy_head_nnue: NNUEPolicyHead | PhaseMoE = NNUEPolicyHead(
                    accumulator_dim=config.lapv2.N_accumulator,
                    move_type_vocab=128,
                    move_type_dim=16,
                    hidden_dim=32,
                )
                if config.lapv2.nnue_value_phase_moe_enabled:
                    policy_head_nnue = PhaseMoE.from_single(policy_head_nnue)
                self.policy_head_nnue: NNUEPolicyHead | PhaseMoE | None = policy_head_nnue
            else:
                self.policy_head_nnue = None
            self.opponent_head = OpponentHeadModel(
                architecture=config.opponent_head.architecture,
                hidden_dim=config.opponent_head.hidden_dim,
                hidden_layers=config.opponent_head.hidden_layers,
                action_embedding_dim=config.opponent_head.action_embedding_dim,
                dropout=config.opponent_head.dropout,
            )
            opponent_readout: OpponentReadout | None = None
            if config.lapv2.enabled and config.lapv2.shared_opponent_readout:
                opponent_readout = OpponentReadout(
                    state_dim=config.deliberation.state_dim,
                    global_dim=config.state_embedder.global_dim,
                    action_embedding_dim=config.opponent_head.action_embedding_dim,
                    hidden_dim=config.opponent_head.hidden_dim,
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
                sharpness_projector=self._sharpness_projector,
                reply_signal_projector=_OpponentReplySignalProjector(
                    state_dim=config.deliberation.state_dim,
                    global_dim=config.state_embedder.global_dim,
                    opponent_head=self.opponent_head,
                    opponent_readout=opponent_readout,
                ),
            )

        @property
        def opponent_readout(self) -> OpponentReadout | None:
            projector = self.deliberation_loop.reply_signal_projector
            return getattr(projector, "opponent_readout", None)

        def _root_dual_accumulators(
            self,
            *,
            phase_idx: torch.Tensor | None,
            nnue_feat_white_indices: torch.Tensor,
            nnue_feat_white_offsets: torch.Tensor,
            nnue_feat_black_indices: torch.Tensor,
            nnue_feat_black_offsets: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if self.ft is None or self.dual_acc_builder is None:
                raise RuntimeError("LAPv2 FT path is not initialized")
            return self.dual_acc_builder(
                self.ft,
                {
                    "nnue_feat_white_indices": nnue_feat_white_indices,
                    "nnue_feat_white_offsets": nnue_feat_white_offsets,
                    "nnue_feat_black_indices": nnue_feat_black_indices,
                    "nnue_feat_black_offsets": nnue_feat_black_offsets,
                },
                phase_idx=phase_idx,
            )

        def _nnue_policy_logits(
            self,
            *,
            a_white: torch.Tensor,
            a_black: torch.Tensor,
            phase_idx: torch.Tensor | None,
            side_to_move: torch.Tensor,
            candidate_mask: torch.Tensor,
            candidate_move_types: torch.Tensor,
            candidate_delta_white_leave_indices: torch.Tensor,
            candidate_delta_white_leave_offsets: torch.Tensor,
            candidate_delta_white_enter_indices: torch.Tensor,
            candidate_delta_white_enter_offsets: torch.Tensor,
            candidate_delta_black_leave_indices: torch.Tensor,
            candidate_delta_black_leave_offsets: torch.Tensor,
            candidate_delta_black_enter_indices: torch.Tensor,
            candidate_delta_black_enter_offsets: torch.Tensor,
            candidate_nnue_feat_white_after_move_indices: torch.Tensor,
            candidate_nnue_feat_white_after_move_offsets: torch.Tensor,
            candidate_nnue_feat_black_after_move_indices: torch.Tensor,
            candidate_nnue_feat_black_after_move_offsets: torch.Tensor,
            candidate_has_king_move: torch.Tensor,
        ) -> tuple[torch.Tensor, list[AccumulatorCache] | None, dict[str, Any]]:
            if self.ft is None or self.policy_head_nnue is None:
                raise RuntimeError("lapv2.nnue_policy is enabled but policy modules are missing")
            batch_size, candidate_count = candidate_move_types.shape
            use_cache = bool(self.config.lapv2.accumulator_cache and not self.training)
            if use_cache:
                succ_white_rows: list[torch.Tensor] = []
                succ_black_rows: list[torch.Tensor] = []
                caches: list[AccumulatorCache] = []
                for batch_index in range(batch_size):
                    sample_phase = (
                        None
                        if phase_idx is None
                        else int(phase_idx[batch_index].item())
                    )
                    cache = AccumulatorCache(self.ft, phase_idx=sample_phase)
                    cache.init_from_accumulators(
                        a_white[batch_index : batch_index + 1],
                        a_black[batch_index : batch_index + 1],
                    )
                    caches.append(cache)
                    candidate_white_rows: list[torch.Tensor] = []
                    candidate_black_rows: list[torch.Tensor] = []
                    for candidate_index in range(candidate_count):
                        if not bool(candidate_mask[batch_index, candidate_index].item()):
                            candidate_white_rows.append(
                                a_white[batch_index : batch_index + 1].clone()
                            )
                            candidate_black_rows.append(
                                a_black[batch_index : batch_index + 1].clone()
                            )
                            continue
                        flat_index = batch_index * candidate_count + candidate_index
                        has_king_move = bool(
                            candidate_has_king_move[batch_index, candidate_index].item()
                        )

                        def full_rebuild(
                            *,
                            sample_phase_index: int | None = sample_phase,
                            row_index: int = flat_index,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
                            white_after = self._build_single_sparse_accumulator(
                                candidate_nnue_feat_white_after_move_indices,
                                candidate_nnue_feat_white_after_move_offsets,
                                row_index=row_index,
                                phase_value=sample_phase_index,
                                device=a_white.device,
                            )
                            black_after = self._build_single_sparse_accumulator(
                                candidate_nnue_feat_black_after_move_indices,
                                candidate_nnue_feat_black_after_move_offsets,
                                row_index=row_index,
                                phase_value=sample_phase_index,
                                device=a_white.device,
                            )
                            return white_after, black_after

                        succ_white, succ_black = cache.successor(
                            candidate_index,
                            unpack_sparse_row(
                                candidate_delta_white_leave_indices,
                                candidate_delta_white_leave_offsets,
                                flat_index,
                            ),
                            unpack_sparse_row(
                                candidate_delta_white_enter_indices,
                                candidate_delta_white_enter_offsets,
                                flat_index,
                            ),
                            unpack_sparse_row(
                                candidate_delta_black_leave_indices,
                                candidate_delta_black_leave_offsets,
                                flat_index,
                            ),
                            unpack_sparse_row(
                                candidate_delta_black_enter_indices,
                                candidate_delta_black_enter_offsets,
                                flat_index,
                            ),
                            is_king_w=has_king_move,
                            is_king_b=has_king_move,
                            full_rebuild_fn=full_rebuild if has_king_move else None,
                        )
                        candidate_white_rows.append(succ_white)
                        candidate_black_rows.append(succ_black)
                    succ_white_rows.append(torch.cat(candidate_white_rows, dim=0).unsqueeze(0))
                    succ_black_rows.append(torch.cat(candidate_black_rows, dim=0).unsqueeze(0))
                succ_white = torch.cat(succ_white_rows, dim=0)
                succ_black = torch.cat(succ_black_rows, dim=0)
            else:
                caches = None
            candidate_phase_idx = (
                None
                if phase_idx is None
                else phase_idx.repeat_interleave(candidate_count)
            )
            if not use_cache:
                leave_white = build_sparse_rows(
                    self.ft,
                    candidate_delta_white_leave_indices,
                    candidate_delta_white_leave_offsets,
                    phase_idx=candidate_phase_idx,
                )
                enter_white = build_sparse_rows(
                    self.ft,
                    candidate_delta_white_enter_indices,
                    candidate_delta_white_enter_offsets,
                    phase_idx=candidate_phase_idx,
                )
                leave_black = build_sparse_rows(
                    self.ft,
                    candidate_delta_black_leave_indices,
                    candidate_delta_black_leave_offsets,
                    phase_idx=candidate_phase_idx,
                )
                enter_black = build_sparse_rows(
                    self.ft,
                    candidate_delta_black_enter_indices,
                    candidate_delta_black_enter_offsets,
                    phase_idx=candidate_phase_idx,
                )
                flat_white_root = a_white.repeat_interleave(candidate_count, dim=0)
                flat_black_root = a_black.repeat_interleave(candidate_count, dim=0)
                succ_white = flat_white_root - leave_white + enter_white
                succ_black = flat_black_root - leave_black + enter_black
                flat_candidate_has_king_move = candidate_has_king_move.reshape(-1)
                if bool(flat_candidate_has_king_move.any().item()):
                    rebuilt_white = build_sparse_rows(
                        self.ft,
                        candidate_nnue_feat_white_after_move_indices,
                        candidate_nnue_feat_white_after_move_offsets,
                        phase_idx=candidate_phase_idx,
                    )
                    rebuilt_black = build_sparse_rows(
                        self.ft,
                        candidate_nnue_feat_black_after_move_indices,
                        candidate_nnue_feat_black_after_move_offsets,
                        phase_idx=candidate_phase_idx,
                    )
                    king_mask = flat_candidate_has_king_move.unsqueeze(1)
                    succ_white = torch.where(king_mask, rebuilt_white, succ_white)
                    succ_black = torch.where(king_mask, rebuilt_black, succ_black)
                succ_white = succ_white.reshape(batch_size, candidate_count, -1)
                succ_black = succ_black.reshape(batch_size, candidate_count, -1)
            stm_white_mask = (side_to_move == 0).unsqueeze(1)
            a_root_stm = torch.where(stm_white_mask, a_white, a_black)
            a_succ_other = torch.where(
                stm_white_mask.unsqueeze(2),
                succ_black,
                succ_white,
            )
            if self.config.lapv2.nnue_value_phase_moe_enabled:
                assert phase_idx is not None
                logits = self.policy_head_nnue(
                    a_root_stm,
                    a_succ_other,
                    candidate_move_types,
                    phase_idx=phase_idx,
                )
            else:
                logits = self.policy_head_nnue(
                    a_root_stm,
                    a_succ_other,
                    candidate_move_types,
                )
            masked_logits = logits.masked_fill(~candidate_mask, float("-1e9"))
            return masked_logits, caches, {
                "enabled": use_cache,
                "phase_indices": ()
                if phase_idx is None
                else tuple(int(value) for value in phase_idx.tolist()),
            }

        def _build_single_sparse_accumulator(
            self,
            indices: torch.Tensor,
            offsets: torch.Tensor,
            *,
            row_index: int,
            phase_value: int | None,
            device: torch.device,
        ) -> torch.Tensor:
            if self.ft is None:
                raise RuntimeError("LAPv2 FT path is not initialized")
            row_features = unpack_sparse_row(indices, offsets, row_index)
            row_indices, row_offsets = pack_sparse_feature_lists([row_features])
            row_indices = row_indices.to(device=device)
            row_offsets = row_offsets.to(device=device)
            if phase_value is None:
                return build_sparse_rows(self.ft, row_indices, row_offsets)
            return build_sparse_rows(
                self.ft,
                row_indices,
                row_offsets,
                phase_idx=torch.tensor([phase_value], dtype=torch.long, device=device),
            )

        def _accumulator_cache_stats(
            self,
            caches: list[AccumulatorCache] | None,
            *,
            phase_idx: torch.Tensor | None,
            step_selected_candidate_tensors: tuple[torch.Tensor, ...],
            step_active_masks: tuple[torch.Tensor, ...],
        ) -> dict[str, Any]:
            if caches is None:
                return {
                    "enabled": False,
                    "phase_fixed": False,
                    "phase_indices": ()
                    if phase_idx is None
                    else tuple(int(value) for value in phase_idx.tolist()),
                    "lookup_hits": 0,
                    "lookup_misses": 0,
                    "touch_hits": 0,
                    "touch_misses": 0,
                    "cached_candidate_count": 0,
                }
            for selected_indices, active_mask in zip(
                step_selected_candidate_tensors,
                step_active_masks,
                strict=True,
            ):
                for batch_index, cache in enumerate(caches):
                    if not bool(active_mask[batch_index].item()):
                        continue
                    for candidate_index in selected_indices[batch_index].tolist():
                        cache.touch(int(candidate_index))
            return {
                "enabled": True,
                "phase_fixed": phase_idx is not None,
                "phase_indices": ()
                if phase_idx is None
                else tuple(int(value) for value in phase_idx.tolist()),
                "lookup_hits": sum(cache.lookup_hits for cache in caches),
                "lookup_misses": sum(cache.lookup_misses for cache in caches),
                "touch_hits": sum(cache.touch_hits for cache in caches),
                "touch_misses": sum(cache.touch_misses for cache in caches),
                "cached_candidate_count": sum(
                    cache.cached_candidate_count for cache in caches
                ),
            }

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
            phase_index: torch.Tensor | None = None,
            side_to_move: torch.Tensor | None = None,
            nnue_feat_white_indices: torch.Tensor | None = None,
            nnue_feat_white_offsets: torch.Tensor | None = None,
            nnue_feat_black_indices: torch.Tensor | None = None,
            nnue_feat_black_offsets: torch.Tensor | None = None,
            candidate_move_types: torch.Tensor | None = None,
            candidate_delta_white_leave_indices: torch.Tensor | None = None,
            candidate_delta_white_leave_offsets: torch.Tensor | None = None,
            candidate_delta_white_enter_indices: torch.Tensor | None = None,
            candidate_delta_white_enter_offsets: torch.Tensor | None = None,
            candidate_delta_black_leave_indices: torch.Tensor | None = None,
            candidate_delta_black_leave_offsets: torch.Tensor | None = None,
            candidate_delta_black_enter_indices: torch.Tensor | None = None,
            candidate_delta_black_enter_offsets: torch.Tensor | None = None,
            candidate_nnue_feat_white_after_move_indices: torch.Tensor | None = None,
            candidate_nnue_feat_white_after_move_offsets: torch.Tensor | None = None,
            candidate_nnue_feat_black_after_move_indices: torch.Tensor | None = None,
            candidate_nnue_feat_black_after_move_offsets: torch.Tensor | None = None,
            candidate_has_king_move: torch.Tensor | None = None,
            candidate_uci: list[list[str]] | None = None,
            single_legal_move: bool = False,
            collect_opponent_distill: bool = False,
        ) -> dict[str, Any]:
            """Run the full LAPv1 wrapper forward pass without trainer glue."""
            phase_idx: torch.Tensor | None = None
            if (
                self.config.lapv2.phase_moe_enabled
                or self.config.lapv2.nnue_value_phase_moe_enabled
                or self.config.lapv2.sharpness_phase_moe_enabled
            ):
                if phase_index is None:
                    raise ValueError(
                        "phase_index is required when LAPv2 phase-routed modules are enabled"
                    )
                assert self.phase_router is not None
                phase_idx = self.phase_router({"phase_index": phase_index})
            if self.config.lapv2.sharpness_phase_moe_enabled:
                if phase_idx is None:
                    raise ValueError(
                        "phase_index is required when lapv2.sharpness_phase_moe is enabled"
                    )
                self._sharpness_projector.set_phase_idx(phase_idx)
            else:
                self._sharpness_projector.set_phase_idx(None)
            if self.config.lapv2.phase_moe_enabled:
                piece_intentions = self.intention_encoder(
                    piece_tokens,
                    state_context_v1_global,
                    reachability_edges,
                    phase_idx=phase_idx,
                )
                z_root, _sigma_root = self.state_embedder(
                    piece_intentions,
                    square_tokens,
                    state_context_v1_global,
                    reachability_edges,
                    phase_idx=phase_idx,
                )
            else:
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
            if self.config.lapv2.sharpness_phase_moe_enabled:
                assert phase_idx is not None
                root_sharpness = self.sharpness_head(z_root, phase_idx=phase_idx).squeeze(1)
            else:
                root_sharpness = self.sharpness_head(z_root).squeeze(1)
            a_white: torch.Tensor | None = None
            a_black: torch.Tensor | None = None
            if self.config.lapv2.nnue_value_enabled:
                if self.ft is None or self.dual_acc_builder is None or self.value_head_nnue is None:
                    raise RuntimeError("lapv2.nnue_value is enabled but NNUE modules are missing")
                if side_to_move is None:
                    raise ValueError("side_to_move is required when lapv2.nnue_value is enabled")
                if (
                    nnue_feat_white_indices is None
                    or nnue_feat_white_offsets is None
                    or nnue_feat_black_indices is None
                    or nnue_feat_black_offsets is None
                ):
                    raise ValueError(
                        "nnue sparse feature inputs are required when lapv2.nnue_value is enabled"
                    )
                a_white, a_black = self._root_dual_accumulators(
                    phase_idx=phase_idx,
                    nnue_feat_white_indices=nnue_feat_white_indices,
                    nnue_feat_white_offsets=nnue_feat_white_offsets,
                    nnue_feat_black_indices=nnue_feat_black_indices,
                    nnue_feat_black_offsets=nnue_feat_black_offsets,
                )
            if self.config.lapv2.nnue_policy_enabled:
                if side_to_move is None:
                    raise ValueError("side_to_move is required when lapv2.nnue_policy is enabled")
                required_policy_inputs = (
                    candidate_move_types,
                    candidate_delta_white_leave_indices,
                    candidate_delta_white_leave_offsets,
                    candidate_delta_white_enter_indices,
                    candidate_delta_white_enter_offsets,
                    candidate_delta_black_leave_indices,
                    candidate_delta_black_leave_offsets,
                    candidate_delta_black_enter_indices,
                    candidate_delta_black_enter_offsets,
                    candidate_nnue_feat_white_after_move_indices,
                    candidate_nnue_feat_white_after_move_offsets,
                    candidate_nnue_feat_black_after_move_indices,
                    candidate_nnue_feat_black_after_move_offsets,
                    candidate_has_king_move,
                    a_white,
                    a_black,
                )
                if any(value is None for value in required_policy_inputs):
                    raise ValueError(
                        "candidate move-type and sparse successor inputs are required when "
                        "lapv2.nnue_policy is enabled"
                    )
                initial_policy_logits, policy_caches, accumulator_cache_meta = self._nnue_policy_logits(
                    a_white=a_white,
                    a_black=a_black,
                    phase_idx=phase_idx,
                    side_to_move=side_to_move,
                    candidate_mask=candidate_mask,
                    candidate_move_types=candidate_move_types,
                    candidate_delta_white_leave_indices=candidate_delta_white_leave_indices,
                    candidate_delta_white_leave_offsets=candidate_delta_white_leave_offsets,
                    candidate_delta_white_enter_indices=candidate_delta_white_enter_indices,
                    candidate_delta_white_enter_offsets=candidate_delta_white_enter_offsets,
                    candidate_delta_black_leave_indices=candidate_delta_black_leave_indices,
                    candidate_delta_black_leave_offsets=candidate_delta_black_leave_offsets,
                    candidate_delta_black_enter_indices=candidate_delta_black_enter_indices,
                    candidate_delta_black_enter_offsets=candidate_delta_black_enter_offsets,
                    candidate_nnue_feat_white_after_move_indices=(
                        candidate_nnue_feat_white_after_move_indices
                    ),
                    candidate_nnue_feat_white_after_move_offsets=(
                        candidate_nnue_feat_white_after_move_offsets
                    ),
                    candidate_nnue_feat_black_after_move_indices=(
                        candidate_nnue_feat_black_after_move_indices
                    ),
                    candidate_nnue_feat_black_after_move_offsets=(
                        candidate_nnue_feat_black_after_move_offsets
                    ),
                    candidate_has_king_move=candidate_has_king_move,
                )
            else:
                policy_caches = None
                accumulator_cache_meta = {
                    "enabled": False,
                    "phase_fixed": False,
                    "phase_indices": ()
                    if phase_idx is None
                    else tuple(int(value) for value in phase_idx.tolist()),
                    "lookup_hits": 0,
                    "lookup_misses": 0,
                    "touch_hits": 0,
                    "touch_misses": 0,
                    "cached_candidate_count": 0,
                }
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
                phase_idx=phase_idx,
                single_legal_move=single_legal_move,
                candidate_uci=candidate_uci,
                candidate_features=candidate_context_v2,
                global_features=state_context_v1_global,
                collect_reply_diagnostics=collect_opponent_distill,
            )
            accumulator_cache_stats = self._accumulator_cache_stats(
                policy_caches,
                phase_idx=phase_idx,
                step_selected_candidate_tensors=deliberation_outputs[
                    "step_selected_candidate_tensors"
                ],
                step_active_masks=deliberation_outputs["step_active_masks"],
            )
            accumulator_cache_stats.update(accumulator_cache_meta)
            if self.config.lapv2.nnue_value_enabled:
                assert a_white is not None and a_black is not None and side_to_move is not None
                stm_white_mask = (side_to_move == 0).unsqueeze(1)
                a_stm = torch.where(stm_white_mask, a_white, a_black)
                a_other = torch.where(stm_white_mask, a_black, a_white)
                if self.config.lapv2.nnue_value_phase_moe_enabled:
                    assert phase_idx is not None
                    wdl_logits, cp_score, sigma_value = self.value_head_nnue(
                        a_stm,
                        a_other,
                        phase_idx=phase_idx,
                    )
                else:
                    wdl_logits, cp_score, sigma_value = self.value_head_nnue(a_stm, a_other)
            else:
                wdl_logits, cp_score, sigma_value = self.value_head(
                    deliberation_outputs["final_z"],
                    deliberation_outputs["final_memory"],
                )
            return {
                "initial_policy_logits": initial_policy_logits,
                "final_policy_logits": deliberation_outputs["final_candidate_scores"],
                "final_policy_deltas": deliberation_outputs["final_candidate_deltas"],
                "final_value": {
                    "wdl_logits": wdl_logits,
                    "cp_score": cp_score,
                    "sigma_value": sigma_value,
                },
                "deliberation_trace": deliberation_outputs["trace"],
                "step_value_cp_tensors": deliberation_outputs["step_value_cp_tensors"],
                "step_sharpness_tensors": deliberation_outputs["step_sharpness_tensors"],
                "step_candidate_score_tensors": deliberation_outputs[
                    "step_candidate_score_tensors"
                ],
                "step_selected_candidate_tensors": deliberation_outputs[
                    "step_selected_candidate_tensors"
                ],
                "step_selected_candidate_masks": deliberation_outputs[
                    "step_selected_candidate_masks"
                ],
                "step_phase_indices": deliberation_outputs["step_phase_indices"],
                "step_active_masks": deliberation_outputs["step_active_masks"],
                "step_rollback_masks": deliberation_outputs["step_rollback_masks"],
                "step_frontier_turnover_tensors": deliberation_outputs[
                    "step_frontier_turnover_tensors"
                ],
                "step_frontier_revisit_tensors": deliberation_outputs[
                    "step_frontier_revisit_tensors"
                ],
                "step_frontier_stable_masks": deliberation_outputs[
                    "step_frontier_stable_masks"
                ],
                "step_frontier_gate_tensors": deliberation_outputs[
                    "step_frontier_gate_tensors"
                ],
                "step_frontier_pressure_tensors": deliberation_outputs[
                    "step_frontier_pressure_tensors"
                ],
                "step_frontier_uncertainty_tensors": deliberation_outputs[
                    "step_frontier_uncertainty_tensors"
                ],
                "step_rollback_flags": deliberation_outputs["step_rollback_flags"],
                "step_student_reply_logits_tensors": deliberation_outputs[
                    "step_student_reply_logits_tensors"
                ],
                "step_student_pressure_tensors": deliberation_outputs[
                    "step_student_pressure_tensors"
                ],
                "step_student_uncertainty_tensors": deliberation_outputs[
                    "step_student_uncertainty_tensors"
                ],
                "step_teacher_reply_logits_tensors": deliberation_outputs[
                    "step_teacher_reply_logits_tensors"
                ],
                "step_teacher_pressure_tensors": deliberation_outputs[
                    "step_teacher_pressure_tensors"
                ],
                "step_teacher_uncertainty_tensors": deliberation_outputs[
                    "step_teacher_uncertainty_tensors"
                ],
                "root_candidate_scores": deliberation_outputs["root_candidate_scores"],
                "root_sharpness": root_sharpness,
                "refined_top1_action_index": deliberation_outputs[
                    "refined_top1_action_index"
                ],
                "frontier_visit_counts": deliberation_outputs["frontier_visit_counts"],
                "frontier_unique_candidate_counts": deliberation_outputs[
                    "frontier_unique_candidate_counts"
                ],
                "candidate_frontier_states": deliberation_outputs[
                    "candidate_frontier_states"
                ],
                "candidate_frontier_memory": deliberation_outputs[
                    "candidate_frontier_memory"
                ],
                "frontier_state_drift": deliberation_outputs["frontier_state_drift"],
                "frontier_memory_norm": deliberation_outputs["frontier_memory_norm"],
                "frontier_update_gate_mean": deliberation_outputs[
                    "frontier_update_gate_mean"
                ],
                "frontier_reply_pressure_mean": deliberation_outputs[
                    "frontier_reply_pressure_mean"
                ],
                "frontier_reply_uncertainty_mean": deliberation_outputs[
                    "frontier_reply_uncertainty_mean"
                ],
                "piece_intentions": piece_intentions,
                "z_root": z_root,
                "accumulator_cache_stats": accumulator_cache_stats,
            }

else:  # pragma: no cover - exercised when torch is absent

    class LAPv1Model:  # type: ignore[no-redef]
        """Stub placeholder when torch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for LAPv1Model. Install the 'train' extra or torch."
            )
