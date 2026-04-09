"""Shared-backbone opponent readout modules for LAPv2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from train.action_space import ACTION_SPACE_SIZE
from train.datasets.contracts import candidate_context_feature_dim

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


OPPONENT_READOUT_MODEL_NAME = "opponent_readout_v1"
OPPONENT_READOUT_CANDIDATE_FEATURE_SIZE = candidate_context_feature_dim(2)


@dataclass(frozen=True)
class OpponentReadoutPrediction:
    """Structured reply/pressure/uncertainty outputs for one successor state."""

    reply_logits: Any
    pressure: Any
    uncertainty: Any


if torch is not None and nn is not None:

    class DeltaOperator(nn.Module):
        """Residual move-conditioned latent update used by the shared readout."""

        def __init__(self, d_model: int, *, hidden_dim: int | None = None) -> None:
            super().__init__()
            if d_model <= 0:
                raise ValueError("d_model must be positive")
            hidden_dim = d_model * 2 if hidden_dim is None else hidden_dim
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            self.d_model = d_model
            self.root_norm = nn.LayerNorm(d_model)
            self.move_norm = nn.LayerNorm(d_model)
            self.query = nn.Linear(d_model, d_model)
            self.key = nn.Linear(d_model, d_model)
            self.value = nn.Linear(d_model, d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, d_model),
            )
            self.output_norm = nn.LayerNorm(d_model)

        def forward(self, h_root: torch.Tensor, move_embed: torch.Tensor) -> torch.Tensor:
            if h_root.ndim != 2 or move_embed.ndim != 2:
                raise ValueError("h_root and move_embed must have shape (batch, d_model)")
            if h_root.shape != move_embed.shape:
                raise ValueError("h_root and move_embed must align")
            query = self.query(self.root_norm(h_root)).unsqueeze(1)
            key = self.key(self.move_norm(move_embed)).unsqueeze(1)
            value = self.value(self.move_norm(move_embed)).unsqueeze(1)
            attention_logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(
                float(self.d_model)
            )
            attention_weights = torch.softmax(attention_logits, dim=-1)
            attended = torch.matmul(attention_weights, value).squeeze(1)
            residual = self.mlp(
                torch.cat(
                    [
                        h_root,
                        move_embed,
                        h_root - move_embed,
                        h_root * move_embed,
                    ],
                    dim=1,
                )
            )
            return self.output_norm(h_root + attended + residual)


    class OpponentReadout(nn.Module):
        """Reply readout over the shared latent backbone instead of a separate head."""

        def __init__(
            self,
            *,
            state_dim: int,
            global_dim: int,
            action_embedding_dim: int,
            hidden_dim: int,
        ) -> None:
            super().__init__()
            if state_dim <= 0:
                raise ValueError("state_dim must be positive")
            if global_dim <= 0:
                raise ValueError("global_dim must be positive")
            if action_embedding_dim <= 0:
                raise ValueError("action_embedding_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            self.state_dim = state_dim
            self.global_dim = global_dim
            self.selected_action_embedding = nn.Embedding(
                ACTION_SPACE_SIZE,
                action_embedding_dim,
            )
            self.reply_action_embedding = nn.Embedding(
                ACTION_SPACE_SIZE,
                action_embedding_dim,
            )
            self.move_projection = nn.Sequential(
                nn.Linear(
                    action_embedding_dim + global_dim,
                    state_dim,
                ),
                nn.GELU(),
                nn.Linear(state_dim, state_dim),
            )
            self.delta = DeltaOperator(state_dim, hidden_dim=hidden_dim)
            self.candidate_projection = nn.Sequential(
                nn.Linear(
                    action_embedding_dim + OPPONENT_READOUT_CANDIDATE_FEATURE_SIZE,
                    hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
                nn.GELU(),
            )
            self.reply_head = nn.Sequential(
                nn.Linear(state_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            pooled_hidden_dim = max(hidden_dim // 2, 1)
            self.pressure_head = nn.Sequential(
                nn.Linear(state_dim, pooled_hidden_dim),
                nn.GELU(),
                nn.Linear(pooled_hidden_dim, 1),
            )
            self.uncertainty_head = nn.Sequential(
                nn.Linear(state_dim, pooled_hidden_dim),
                nn.GELU(),
                nn.Linear(pooled_hidden_dim, 1),
            )

        def forward(
            self,
            h_root: torch.Tensor,
            selected_action_indices: torch.Tensor,
            reply_global_features: torch.Tensor,
            reply_candidate_action_indices: torch.Tensor,
            reply_candidate_features: torch.Tensor,
            reply_candidate_mask: torch.Tensor,
        ) -> OpponentReadoutPrediction:
            if h_root.ndim != 2:
                raise ValueError("h_root must have shape (batch, state_dim)")
            if selected_action_indices.ndim != 1:
                raise ValueError("selected_action_indices must have shape (batch,)")
            if reply_global_features.ndim != 2:
                raise ValueError(
                    "reply_global_features must have shape (batch, global_dim)"
                )
            if reply_candidate_action_indices.ndim != 2:
                raise ValueError(
                    "reply_candidate_action_indices must have shape (batch, num_candidates)"
                )
            if reply_candidate_features.ndim != 3:
                raise ValueError(
                    "reply_candidate_features must have shape (batch, num_candidates, feature_dim)"
                )
            if reply_candidate_mask.shape != reply_candidate_action_indices.shape:
                raise ValueError(
                    "reply_candidate_mask must align with reply_candidate_action_indices"
                )
            selected_action_hidden = self.selected_action_embedding(
                selected_action_indices.clamp_min(0)
            )
            move_embed = self.move_projection(
                torch.cat([selected_action_hidden, reply_global_features], dim=1)
            )
            context_hidden = self.delta(h_root, move_embed)
            safe_candidate_indices = reply_candidate_action_indices.clamp_min(0)
            reply_action_hidden = self.reply_action_embedding(safe_candidate_indices)
            candidate_hidden = self.candidate_projection(
                torch.cat([reply_action_hidden, reply_candidate_features], dim=2)
            )
            repeated_context = context_hidden.unsqueeze(1).expand_as(candidate_hidden)
            reply_logits = self.reply_head(
                torch.cat(
                    [
                        repeated_context,
                        candidate_hidden,
                        repeated_context - candidate_hidden,
                        repeated_context * candidate_hidden,
                    ],
                    dim=2,
                )
            ).squeeze(2)
            reply_logits = reply_logits.masked_fill(~reply_candidate_mask, -20.0)
            pressure = torch.sigmoid(self.pressure_head(context_hidden)).squeeze(1)
            uncertainty = torch.sigmoid(self.uncertainty_head(context_hidden)).squeeze(1)
            return OpponentReadoutPrediction(
                reply_logits=reply_logits,
                pressure=pressure,
                uncertainty=uncertainty,
            )

else:  # pragma: no cover - exercised when torch is absent

    class DeltaOperator:  # type: ignore[no-redef]
        """Stub placeholder when torch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for LAPv2 opponent-readout models. Install the 'train' extra or torch."
            )


    class OpponentReadout:  # type: ignore[no-redef]
        """Stub placeholder when torch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for LAPv2 opponent-readout models. Install the 'train' extra or torch."
            )
