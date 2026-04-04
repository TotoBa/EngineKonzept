"""Experimental MoE planner arm for bounded candidate scoring."""

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
from train.models.planner import PLANNER_RANK_BUCKET_COUNT

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


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


    def compute_expert_utilization_metrics(
        router_weights: Any,
        sparse_router_weights: Any,
    ) -> dict[str, Any]:
        """Summarize router balance and expert usage for one batch."""
        num_experts = int(router_weights.shape[1])
        entropy = -torch.sum(
            router_weights * torch.log(router_weights.clamp_min(1e-12)),
            dim=1,
        )
        activation_counts = torch.count_nonzero(sparse_router_weights, dim=0).to(torch.float32)
        return {
            "router_entropy": entropy.mean(),
            "expert_activation_counts": activation_counts,
            "expert_activation_frequencies": activation_counts / router_weights.shape[0],
            "expert_importance": sparse_router_weights.sum(dim=0) / router_weights.shape[0],
            "num_experts": num_experts,
        }


    class PositionRouter(nn.Module):
        """Route a state embedding onto a sparse Top-2 expert mixture."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            num_experts: int,
            top_k: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if num_experts <= 0:
                raise ValueError("num_experts must be positive")
            if top_k <= 0 or top_k > num_experts:
                raise ValueError("top_k must be in [1, num_experts]")
            self.num_experts = num_experts
            self.top_k = top_k
            self.router = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, num_experts),
            )

        def forward(self, state_embedding: Any) -> dict[str, Any]:
            logits = self.router(state_embedding)
            router_weights = torch.softmax(logits, dim=1)
            topk_values, topk_indices = torch.topk(router_weights, self.top_k, dim=1)
            sparse_router_weights = torch.zeros_like(router_weights)
            sparse_router_weights.scatter_(1, topk_indices, topk_values)
            sparse_router_weights = sparse_router_weights / sparse_router_weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)

            ideal = 1.0 / float(self.num_experts)
            importance = sparse_router_weights.mean(dim=0)
            load = torch.count_nonzero(sparse_router_weights, dim=0).to(torch.float32)
            load = load / state_embedding.shape[0]
            load_balance_loss = torch.mean((importance - ideal) ** 2) + torch.mean(
                (load - ideal) ** 2
            )
            return {
                "router_logits": logits,
                "router_weights": router_weights,
                "sparse_router_weights": sparse_router_weights,
                "topk_indices": topk_indices,
                "load_balance_loss": load_balance_loss,
                **compute_expert_utilization_metrics(router_weights, sparse_router_weights),
            }


    class CandidateExpert(nn.Module):
        """Per-expert bounded candidate scorer with the set_v6 interface."""

        def __init__(
            self,
            *,
            input_dim: int,
            expert_hidden_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(expert_hidden_dim, expert_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim // 2, 1),
            )

        def forward(self, candidate_hidden: Any) -> Any:
            return self.network(candidate_hidden).squeeze(-1)


    class MoEPlannerHeadModel(nn.Module):
        """Experimental MoE bounded planner with Top-2 routed candidate experts."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            hidden_layers: int,
            action_embedding_dim: int,
            latent_feature_dim: int,
            dropout: float,
            num_experts: int = 4,
            top_k: int = 2,
            expert_hidden_dim: int = 128,
            enable_candidate_rank_head: bool = False,
        ) -> None:
            super().__init__()
            self.num_experts = num_experts
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
            self.set_query = nn.Linear(hidden_dim, hidden_dim)
            self.set_key = nn.Linear(hidden_dim, hidden_dim)
            self.set_value = nn.Linear(hidden_dim, hidden_dim)
            self.router = PositionRouter(
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
            )
            candidate_factor_count = 6
            candidate_hidden_dim = hidden_dim * candidate_factor_count
            self.experts = nn.ModuleList(
                CandidateExpert(
                    input_dim=candidate_hidden_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            )
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
            self.candidate_score_head = nn.Sequential(
                nn.Linear(candidate_hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.candidate_rank_head = (
                nn.Sequential(
                    nn.Linear(candidate_hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, PLANNER_RANK_BUCKET_COUNT),
                )
                if enable_candidate_rank_head
                else None
            )

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
        ) -> dict[str, Any]:
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
                    candidate_token * repeated_attended,
                ],
                dim=2,
            )

            router_outputs = self.router(state_hidden)
            expert_logits = torch.stack(
                [expert(candidate_hidden) for expert in self.experts],
                dim=1,
            )
            fused_logits = torch.sum(
                expert_logits * router_outputs["sparse_router_weights"].unsqueeze(2),
                dim=1,
            )
            candidate_score_prediction = self.candidate_score_head(candidate_hidden).squeeze(-1)
            candidate_rank_prediction = (
                self.candidate_rank_head(candidate_hidden)
                if self.candidate_rank_head is not None
                else None
            )
            root_summary = torch.cat([state_hidden, attended, summary], dim=1)
            return {
                "logits": fused_logits.masked_fill(~candidate_mask, -20.0),
                "candidate_score_prediction": candidate_score_prediction.masked_fill(
                    ~candidate_mask,
                    0.0,
                ),
                "candidate_rank_prediction": (
                    candidate_rank_prediction.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
                    if candidate_rank_prediction is not None
                    else None
                ),
                "root_value_prediction": self.root_value_head(root_summary).squeeze(-1),
                "root_gap_prediction": self.root_gap_head(root_summary).squeeze(-1),
                "load_balance_loss": router_outputs["load_balance_loss"],
                "router_entropy": router_outputs["router_entropy"],
                "expert_activation_counts": router_outputs["expert_activation_counts"],
                "expert_activation_frequencies": router_outputs["expert_activation_frequencies"],
                "router_weights": router_outputs["router_weights"],
                "sparse_router_weights": router_outputs["sparse_router_weights"],
            }


else:

    class PositionRouter:  # pragma: no cover - exercised when torch is absent
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for MoE planner models. Install the 'train' extra or torch."
            )


    class CandidateExpert:  # pragma: no cover - exercised when torch is absent
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for MoE planner models. Install the 'train' extra or torch."
            )


    class MoEPlannerHeadModel:  # pragma: no cover - exercised when torch is absent
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for MoE planner models. Install the 'train' extra or torch."
            )


    def compute_expert_utilization_metrics(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(
            "PyTorch is required for MoE planner models. Install the 'train' extra or torch."
        )
