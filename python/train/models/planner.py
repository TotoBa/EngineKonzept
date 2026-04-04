"""Phase-8 bounded planner-head model definitions."""

from __future__ import annotations

import math
from typing import Any

from train.action_space import ACTION_SPACE_SIZE
from train.datasets.artifacts import (
    POSITION_FEATURE_SIZE,
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
    TRANSITION_CONTEXT_FEATURE_SIZE,
)
from train.datasets.contracts import candidate_context_feature_dim

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


PLANNER_MODEL_NAME = "planner_head"
PLANNER_CANDIDATE_FEATURE_SIZE = candidate_context_feature_dim(2)
PLANNER_RANK_BUCKET_COUNT = 3


if torch is not None and nn is not None:

    def _build_mlp(
        *,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        output_dim: int,
        dropout: float,
    ) -> Any:
        layers: list[Any] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    class PairwiseCandidateLayer(nn.Module):
        """Single-layer pairwise candidate refinement with masked self-attention."""

        def __init__(self, *, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.attention = nn.MultiheadAttention(
                hidden_dim,
                2,
                dropout=dropout,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, candidate_embeddings: Any, candidate_mask: Any) -> Any:
            key_padding_mask = ~candidate_mask
            attended, _ = self.attention(
                candidate_embeddings,
                candidate_embeddings,
                candidate_embeddings,
                key_padding_mask=key_padding_mask,
            )
            refined = candidate_embeddings + self.dropout(attended)
            return refined * candidate_mask.unsqueeze(2).to(refined.dtype)

    class PlannerHeadModel(nn.Module):
        """Score bounded exact root candidates for the first trainable planner arm."""

        def __init__(
            self,
            *,
            architecture: str,
            hidden_dim: int,
            hidden_layers: int,
            action_embedding_dim: int,
            latent_feature_dim: int,
            deliberation_steps: int = 1,
            memory_slots: int = 0,
            dropout: float,
            enable_pairwise_candidates: bool = False,
            enable_candidate_rank_head: bool = False,
        ) -> None:
            super().__init__()
            if architecture not in {"set_v1", "set_v2", "set_v3", "set_v5", "set_v6", "set_v7", "recurrent_v1"}:
                raise ValueError(f"unsupported planner architecture: {architecture}")
            self.architecture = architecture
            self.latent_feature_dim = latent_feature_dim
            self.deliberation_steps = deliberation_steps
            self.memory_slots = memory_slots
            candidate_input_dim = (
                action_embedding_dim
                + PLANNER_CANDIDATE_FEATURE_SIZE
                + TRANSITION_CONTEXT_FEATURE_SIZE
                + latent_feature_dim
                + 4
            )
            self.state_backbone = _build_mlp(
                input_dim=POSITION_FEATURE_SIZE + SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            self.action_embedding = nn.Embedding(ACTION_SPACE_SIZE, action_embedding_dim)
            self.candidate_query_projection = None
            if architecture == "set_v7":
                self.candidate_projection = None
                self.candidate_query_projection = nn.Sequential(
                    nn.Linear(candidate_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                )
            else:
                self.candidate_projection = nn.Sequential(
                    nn.Linear(candidate_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                )
            self.context_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            if architecture == "set_v5":
                num_heads = 4
                self.self_attention = nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True,
                )
                self.self_attn_norm = nn.LayerNorm(hidden_dim)
                self.cross_attention = nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True,
                )
                self.cross_attn_norm = nn.LayerNorm(hidden_dim)
                self.set_query = None
                self.set_key = None
                self.set_value = None
            else:
                self.self_attention = None
                self.self_attn_norm = None
                self.cross_attention = None
                self.cross_attn_norm = None
                self.set_query = nn.Linear(hidden_dim, hidden_dim)
                self.set_key = nn.Linear(hidden_dim, hidden_dim)
                self.set_value = nn.Linear(hidden_dim, hidden_dim)
            if architecture == "set_v7":
                self.self_attention = None
                self.self_attn_norm = None
                self.cross_attention = nn.MultiheadAttention(
                    hidden_dim, 4, dropout=dropout, batch_first=True,
                )
                self.cross_attn_norm = nn.LayerNorm(hidden_dim)
                self.set_query = None
                self.set_key = None
                self.set_value = None
            if architecture == "recurrent_v1":
                self.memory_slot_embeddings = nn.Parameter(torch.randn(memory_slots, hidden_dim) * 0.02)
                self.memory_cell = nn.GRUCell(hidden_dim * 2, hidden_dim)
                self.candidate_refinement = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.recurrent_norm = nn.LayerNorm(hidden_dim)
            else:
                self.memory_slot_embeddings = None
                self.memory_cell = None
                self.candidate_refinement = None
                self.recurrent_norm = None
            self.pairwise_candidate_layer = (
                PairwiseCandidateLayer(hidden_dim=hidden_dim, dropout=dropout)
                if enable_pairwise_candidates
                else None
            )
            candidate_factor_count = 5 if architecture == "set_v1" else 6
            self.candidate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * candidate_factor_count, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            if architecture in {"set_v2", "set_v3", "set_v5", "set_v6", "set_v7", "recurrent_v1"}:
                self.root_value_head = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(hidden_dim, 1),
                )
                self.root_gap_head = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(hidden_dim, 1),
                )
                self.candidate_score_head = (
                    nn.Sequential(
                        nn.Linear(hidden_dim * candidate_factor_count, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, 1),
                    )
                    if architecture in {"set_v6", "set_v7"}
                    else None
                )
                self.candidate_rank_head = (
                    nn.Sequential(
                        nn.Linear(hidden_dim * candidate_factor_count, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, PLANNER_RANK_BUCKET_COUNT),
                    )
                    if architecture in {"set_v6", "set_v7"} and enable_candidate_rank_head
                    else None
                )
            else:
                self.root_value_head = None
                self.root_gap_head = None
                self.candidate_score_head = None
                self.candidate_rank_head = None

        def forward(
            self,
            root_features: Any,
            global_features: Any,
            candidate_action_indices: Any,
            candidate_features: Any,
            proposer_scores: Any,
            transition_features: Any,
            latent_features: Any,
            reply_peak_probabilities: Any,
            pressures: Any,
            uncertainties: Any,
            candidate_mask: Any,
        ) -> Any:
            """Score the bounded exact root candidate set."""
            state_hidden = self.context_projection(
                self.state_backbone(torch.cat([root_features, global_features], dim=1))
            )
            safe_candidate_indices = candidate_action_indices.clamp_min(0)
            action_hidden = self.action_embedding(safe_candidate_indices)
            scalar_features = torch.stack(
                [
                    proposer_scores,
                    reply_peak_probabilities,
                    pressures,
                    uncertainties,
                ],
                dim=2,
            )
            candidate_inputs = torch.cat(
                [
                    action_hidden,
                    candidate_features,
                    transition_features,
                    latent_features,
                    scalar_features,
                ],
                dim=2,
            )
            if self.architecture == "set_v7":
                assert self.candidate_query_projection is not None
                candidate_token = self.candidate_query_projection(candidate_inputs)
            else:
                candidate_token = self.candidate_projection(candidate_inputs)
            if self.architecture == "set_v5":
                key_padding_mask = ~candidate_mask
                residual = candidate_token
                candidate_token = self.self_attn_norm(candidate_token)
                self_attended, _ = self.self_attention(
                    candidate_token, candidate_token, candidate_token,
                    key_padding_mask=key_padding_mask,
                )
                candidate_token = residual + self_attended
                state_seq = state_hidden.unsqueeze(1).expand(-1, candidate_token.size(1), -1)
                residual2 = candidate_token
                candidate_token = self.cross_attn_norm(candidate_token)
                cross_attended, _ = self.cross_attention(
                    candidate_token, state_seq, state_seq,
                )
                candidate_token = residual2 + cross_attended
                mask = candidate_mask.unsqueeze(2).to(candidate_token.dtype)
                candidate_count = mask.sum(dim=1).clamp_min(1.0)
                attended = torch.sum(candidate_token * mask, dim=1) / candidate_count
                summary = attended
                context_hidden = state_hidden
            elif self.architecture == "set_v7":
                assert self.cross_attention is not None
                assert self.cross_attn_norm is not None
                residual = candidate_token
                candidate_token = self.cross_attn_norm(candidate_token)
                state_seq = state_hidden.unsqueeze(1)
                cross_attended, _ = self.cross_attention(
                    candidate_token,
                    state_seq,
                    state_seq,
                )
                candidate_token = residual + cross_attended
                mask = candidate_mask.unsqueeze(2).to(candidate_token.dtype)
                candidate_count = mask.sum(dim=1).clamp_min(1.0)
                attended = torch.sum(candidate_token * mask, dim=1) / candidate_count
                summary = attended
                context_hidden = state_hidden
            elif self.architecture == "recurrent_v1":
                assert self.memory_slot_embeddings is not None
                assert self.memory_cell is not None
                assert self.candidate_refinement is not None
                assert self.recurrent_norm is not None
                batch_size, _, hidden_size = candidate_token.shape
                memory_slots = (
                    state_hidden.unsqueeze(1)
                    + self.memory_slot_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                )
                mask = candidate_mask.unsqueeze(2).to(candidate_token.dtype)
                candidate_count = mask.sum(dim=1).clamp_min(1.0)
                summary = torch.sum(candidate_token * mask, dim=1) / candidate_count
                attended = summary
                context_hidden = state_hidden
                for _ in range(self.deliberation_steps):
                    memory_summary = memory_slots.mean(dim=1)
                    query = self.set_query(memory_summary).unsqueeze(1)
                    keys = self.set_key(candidate_token)
                    values = self.set_value(candidate_token)
                    attention_logits = (query * keys).sum(dim=2) / math.sqrt(float(keys.shape[2]))
                    attention_logits = attention_logits.masked_fill(~candidate_mask, -1e9)
                    attention_weights = torch.softmax(attention_logits, dim=1)
                    attention_weights = attention_weights.masked_fill(~candidate_mask, 0.0)
                    attended = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)
                    summary = torch.sum(candidate_token * mask, dim=1) / candidate_count
                    memory_input = torch.cat([attended, summary], dim=1)
                    repeated_memory_input = memory_input.unsqueeze(1).expand(-1, self.memory_slots, -1)
                    memory_slots = self.memory_cell(
                        repeated_memory_input.reshape(-1, hidden_size * 2),
                        memory_slots.reshape(-1, hidden_size),
                    ).reshape(batch_size, self.memory_slots, hidden_size)
                    context_hidden = memory_slots.mean(dim=1)
                    refinement_input = torch.cat(
                        [
                            candidate_token,
                            context_hidden.unsqueeze(1).expand_as(candidate_token),
                            attended.unsqueeze(1).expand_as(candidate_token),
                        ],
                        dim=2,
                    )
                    candidate_token = self.recurrent_norm(
                        candidate_token + self.candidate_refinement(refinement_input) * mask
                    )
            else:
                query = self.set_query(state_hidden).unsqueeze(1)
                keys = self.set_key(candidate_token)
                values = self.set_value(candidate_token)
                attention_logits = (query * keys).sum(dim=2) / math.sqrt(float(keys.shape[2]))
                attention_logits = attention_logits.masked_fill(~candidate_mask, -1e9)
                attention_weights = torch.softmax(attention_logits, dim=1)
                attention_weights = attention_weights.masked_fill(~candidate_mask, 0.0)
                attended = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)
                mask = candidate_mask.unsqueeze(2).to(candidate_token.dtype)
                candidate_count = mask.sum(dim=1).clamp_min(1.0)
                summary = torch.sum(candidate_token * mask, dim=1) / candidate_count
                context_hidden = state_hidden
            if self.pairwise_candidate_layer is not None:
                candidate_token = self.pairwise_candidate_layer(candidate_token, candidate_mask)
            repeated_context = context_hidden.unsqueeze(1).expand_as(candidate_token)
            repeated_attended = attended.unsqueeze(1).expand_as(candidate_token)
            repeated_summary = summary.unsqueeze(1).expand_as(candidate_token)
            candidate_hidden_parts = [
                candidate_token,
                repeated_context,
                repeated_attended,
                repeated_summary,
                candidate_token * repeated_context,
            ]
            if self.architecture in {"set_v2", "set_v3", "set_v5", "set_v6", "set_v7", "recurrent_v1"}:
                candidate_hidden_parts.append(candidate_token * repeated_attended)
            candidate_hidden = torch.cat(candidate_hidden_parts, dim=2)
            logits = self.candidate_mlp(candidate_hidden).squeeze(-1)
            candidate_score_prediction = None
            if self.candidate_score_head is not None:
                candidate_score_prediction = self.candidate_score_head(candidate_hidden).squeeze(-1)
            candidate_rank_prediction = None
            if self.candidate_rank_head is not None:
                candidate_rank_prediction = self.candidate_rank_head(candidate_hidden)
            root_summary = torch.cat([context_hidden, attended, summary], dim=1)
            root_value_prediction = None
            root_gap_prediction = None
            if self.root_value_head is not None:
                root_value_prediction = self.root_value_head(root_summary).squeeze(-1)
            if self.root_gap_head is not None:
                root_gap_prediction = self.root_gap_head(root_summary).squeeze(-1)
            return {
                "logits": logits.masked_fill(~candidate_mask, -20.0),
                "candidate_score_prediction": (
                    candidate_score_prediction.masked_fill(~candidate_mask, 0.0)
                    if candidate_score_prediction is not None
                    else None
                ),
                "candidate_rank_prediction": (
                    candidate_rank_prediction.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
                    if candidate_rank_prediction is not None
                    else None
                ),
                "root_value_prediction": root_value_prediction,
                "root_gap_prediction": root_gap_prediction,
            }

else:  # pragma: no cover - exercised when torch is absent

    class PlannerHeadModel:  # type: ignore[no-redef]
        """Stub placeholder when torch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for planner-head models. Install the 'train' extra or torch."
            )
