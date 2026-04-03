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


PLANNER_MODEL_NAME = "planner_head_v1"
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
            dropout: float,
        ) -> None:
            super().__init__()
            if architecture != "set_v1":
                raise ValueError(f"unsupported planner architecture: {architecture}")
            self.architecture = architecture
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
            self.set_query = nn.Linear(hidden_dim, hidden_dim)
            self.set_key = nn.Linear(hidden_dim, hidden_dim)
            self.set_value = nn.Linear(hidden_dim, hidden_dim)
            self.candidate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 5, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(
            self,
            root_features: Any,
            global_features: Any,
            candidate_action_indices: Any,
            candidate_features: Any,
            proposer_scores: Any,
            transition_features: Any,
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
                    [action_hidden, candidate_features, transition_features, scalar_features],
                    dim=2,
                )
            )
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
            candidate_hidden = torch.cat(
                [
                    candidate_token,
                    repeated_context,
                    repeated_attended,
                    repeated_summary,
                    candidate_token * repeated_context,
                ],
                dim=2,
            )
            logits = self.candidate_mlp(candidate_hidden).squeeze(-1)
            return logits.masked_fill(~candidate_mask, -20.0)

else:  # pragma: no cover - exercised when torch is absent

    class PlannerHeadModel:  # type: ignore[no-redef]
        """Stub placeholder when torch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for planner-head models. Install the 'train' extra or torch."
            )
