"""Phase-7 opponent-head model definitions."""

from __future__ import annotations

from dataclasses import dataclass
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


OPPONENT_MODEL_NAME = "opponent_head_v1"
OPPONENT_CANDIDATE_FEATURE_SIZE = candidate_context_feature_dim(2)


def torch_is_available() -> bool:
    """Return whether PyTorch is importable in the current environment."""
    return torch is not None and nn is not None


@dataclass(frozen=True)
class OpponentHeadPrediction:
    """Structured opponent-head outputs for one successor state."""

    reply_logits: Any
    pressure: Any
    uncertainty: Any


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

    class OpponentHeadModel(nn.Module):
        """Predict reply ranking, pressure, and uncertainty over exact legal replies."""

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
            if architecture not in {"mlp_v1", "set_v2"}:
                raise ValueError(f"unsupported opponent architecture: {architecture}")
            self.architecture = architecture
            state_input_dim = (
                POSITION_FEATURE_SIZE * 2
                + TRANSITION_CONTEXT_FEATURE_SIZE
            )
            self.state_backbone = _build_mlp(
                input_dim=state_input_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                output_dim=hidden_dim,
                dropout=dropout,
            )
            self.action_embedding = nn.Embedding(ACTION_SPACE_SIZE, action_embedding_dim)
            self.context_projection = nn.Sequential(
                nn.Linear(
                    hidden_dim + action_embedding_dim + SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
                    hidden_dim,
                ),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            if architecture == "mlp_v1":
                self.candidate_mlp = nn.Sequential(
                    nn.Linear(
                        hidden_dim
                        + action_embedding_dim
                        + OPPONENT_CANDIDATE_FEATURE_SIZE,
                        hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
            else:
                self.candidate_projection = nn.Sequential(
                    nn.Linear(
                        action_embedding_dim + OPPONENT_CANDIDATE_FEATURE_SIZE,
                        hidden_dim,
                    ),
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
            self.pressure_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(
            self,
            root_features: Any,
            next_features: Any,
            chosen_action_indices: Any,
            transition_features: Any,
            reply_global_features: Any,
            reply_candidate_action_indices: Any,
            reply_candidate_features: Any,
            reply_candidate_mask: Any,
        ) -> OpponentHeadPrediction:
            """Score exact reply candidates for one successor state."""
            state_input = torch.cat(
                [root_features, next_features, transition_features],
                dim=1,
            )
            state_hidden = self.state_backbone(state_input)
            chosen_action_hidden = self.action_embedding(chosen_action_indices.clamp_min(0))
            context_hidden = self.context_projection(
                torch.cat(
                    [state_hidden, chosen_action_hidden, reply_global_features],
                    dim=1,
                )
            )

            safe_candidate_indices = reply_candidate_action_indices.clamp_min(0)
            reply_action_hidden = self.action_embedding(safe_candidate_indices)
            repeated_context = context_hidden.unsqueeze(1).expand(
                -1,
                safe_candidate_indices.shape[1],
                -1,
            )
            if self.architecture == "mlp_v1":
                candidate_hidden = torch.cat(
                    [repeated_context, reply_action_hidden, reply_candidate_features],
                    dim=2,
                )
            else:
                candidate_token = self.candidate_projection(
                    torch.cat([reply_action_hidden, reply_candidate_features], dim=2)
                )
                query = self.set_query(context_hidden).unsqueeze(1)
                keys = self.set_key(candidate_token)
                values = self.set_value(candidate_token)
                attention_logits = (query * keys).sum(dim=2) / math.sqrt(float(keys.shape[2]))
                attention_logits = attention_logits.masked_fill(~reply_candidate_mask, -1e9)
                attention_weights = torch.softmax(attention_logits, dim=1)
                attention_weights = attention_weights.masked_fill(
                    ~reply_candidate_mask,
                    0.0,
                )
                attended = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)
                mask = reply_candidate_mask.unsqueeze(2).to(candidate_token.dtype)
                candidate_count = mask.sum(dim=1).clamp_min(1.0)
                summary = torch.sum(candidate_token * mask, dim=1) / candidate_count
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
            reply_logits = self.candidate_mlp(candidate_hidden).squeeze(-1)
            reply_logits = reply_logits.masked_fill(~reply_candidate_mask, -20.0)
            pressure = torch.sigmoid(self.pressure_head(context_hidden)).squeeze(1)
            uncertainty = torch.sigmoid(self.uncertainty_head(context_hidden)).squeeze(1)
            return OpponentHeadPrediction(
                reply_logits=reply_logits,
                pressure=pressure,
                uncertainty=uncertainty,
            )

else:  # pragma: no cover - exercised when torch is absent

    class OpponentHeadModel:  # type: ignore[no-redef]
        """Stub placeholder when torch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for opponent-head models. Install the 'train' extra or torch."
            )
