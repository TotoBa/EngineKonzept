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
            dropout: float,
        ) -> None:
            super().__init__()
            if architecture not in {"set_v1", "set_v2", "set_v3", "set_v5", "set_v6"}:
                raise ValueError(f"unsupported planner architecture: {architecture}")
            self.architecture = architecture
            self.latent_feature_dim = latent_feature_dim
            self.state_backbone = _build_mlp(
                input_dim=POSITION_FEATURE_SIZE + SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            self.action_embedding = nn.Embedding(ACTION_SPACE_SIZE, action_embedding_dim)
            self.candidate_projection = nn.Sequential(
                nn.Linear(
                    action_embedding_dim
                    + PLANNER_CANDIDATE_FEATURE_SIZE
                    + TRANSITION_CONTEXT_FEATURE_SIZE
                    + latent_feature_dim
                    + 4,
                    hidden_dim,
                ),
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
            candidate_factor_count = 5 if architecture == "set_v1" else 6
            self.candidate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * candidate_factor_count, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            if architecture in {"set_v2", "set_v3", "set_v5", "set_v6"}:
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
                    if architecture == "set_v6"
                    else None
                )
            else:
                self.root_value_head = None
                self.root_gap_head = None
                self.candidate_score_head = None

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
            candidate_token = self.candidate_projection(
                torch.cat(
                    [
                        action_hidden,
                        candidate_features,
                        transition_features,
                        latent_features,
                        scalar_features,
                    ],
                    dim=2,
                )
            )
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
            repeated_context = state_hidden.unsqueeze(1).expand_as(candidate_token)
            repeated_attended = attended.unsqueeze(1).expand_as(candidate_token)
            repeated_summary = summary.unsqueeze(1).expand_as(candidate_token)
            candidate_hidden_parts = [
                candidate_token,
                repeated_context,
                repeated_attended,
                repeated_summary,
                candidate_token * repeated_context,
            ]
            if self.architecture in {"set_v2", "set_v3", "set_v5", "set_v6"}:
                candidate_hidden_parts.append(candidate_token * repeated_attended)
            candidate_hidden = torch.cat(candidate_hidden_parts, dim=2)
            logits = self.candidate_mlp(candidate_hidden).squeeze(-1)
            candidate_score_prediction = None
            if self.candidate_score_head is not None:
                candidate_score_prediction = self.candidate_score_head(candidate_hidden).squeeze(-1)
            root_summary = torch.cat([state_hidden, attended, summary], dim=1)
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
