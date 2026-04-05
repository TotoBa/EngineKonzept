"""Model-only LAPv1 large policy head over exact legal candidates."""

from __future__ import annotations

from train.action_space import ACTION_SPACE_SIZE
from train.datasets.contracts import candidate_context_feature_dim

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


LARGE_POLICY_HEAD_MODEL_NAME = "lapv1_large_policy_head"
LARGE_POLICY_CANDIDATE_FEATURE_DIM = candidate_context_feature_dim(2)
DEFAULT_STATE_DIM = 512
MASKED_CANDIDATE_LOGIT_VALUE = -1e9


if torch is not None and nn is not None:

    class _CrossAttentionBlock(nn.Module):
        """Candidate-token refinement against the shared root latent."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            num_heads: int,
            feedforward_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(hidden_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden_dim, feedforward_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(feedforward_dim, hidden_dim),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.feedforward_norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            candidate_hidden: torch.Tensor,
            *,
            root_hidden: torch.Tensor,
            candidate_mask: torch.Tensor,
        ) -> torch.Tensor:
            attended, _ = self.cross_attention(
                candidate_hidden,
                root_hidden,
                root_hidden,
                need_weights=False,
            )
            candidate_hidden = self.cross_norm(candidate_hidden + attended)
            candidate_hidden = candidate_hidden * candidate_mask.unsqueeze(2).to(
                candidate_hidden.dtype
            )
            fed_forward = self.feedforward(candidate_hidden)
            candidate_hidden = self.feedforward_norm(candidate_hidden + fed_forward)
            return candidate_hidden * candidate_mask.unsqueeze(2).to(
                candidate_hidden.dtype
            )


    class LargePolicyHead(nn.Module):
        """Score exact legal candidates with a wide candidate-conditioned network."""

        def __init__(
            self,
            *,
            state_dim: int = DEFAULT_STATE_DIM,
            hidden_dim: int = 896,
            action_embedding_dim: int = 96,
            num_layers: int = 4,
            num_heads: int = 8,
            feedforward_dim: int = 1792,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if state_dim <= 0:
                raise ValueError("state_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if action_embedding_dim <= 0:
                raise ValueError("action_embedding_dim must be positive")
            if num_layers <= 0:
                raise ValueError("num_layers must be positive")
            if num_heads <= 0:
                raise ValueError("num_heads must be positive")
            if hidden_dim % num_heads != 0:
                raise ValueError("hidden_dim must be divisible by num_heads")
            if feedforward_dim <= hidden_dim:
                raise ValueError("feedforward_dim must be larger than hidden_dim")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0.0, 1.0)")

            self.state_dim = state_dim
            self.hidden_dim = hidden_dim
            self.action_embedding_dim = action_embedding_dim
            self.num_layers = num_layers

            self.root_projection = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.action_embedding = nn.Embedding(ACTION_SPACE_SIZE, action_embedding_dim)
            self.candidate_projection = nn.Sequential(
                nn.Linear(
                    LARGE_POLICY_CANDIDATE_FEATURE_DIM + action_embedding_dim + hidden_dim,
                    hidden_dim,
                ),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.layers = nn.ModuleList(
                [
                    _CrossAttentionBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        feedforward_dim=feedforward_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim, 1024),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(1024, 1024),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Linear(512, 1),
            )

        def forward(
            self,
            z_root: torch.Tensor,
            candidate_context_v2: torch.Tensor,
            action_embedding_indices: torch.Tensor,
            candidate_mask: torch.Tensor,
        ) -> torch.Tensor:
            """Return masked candidate logits for exact legal moves only."""
            if z_root.ndim != 2 or z_root.shape[1] != self.state_dim:
                raise ValueError(f"z_root must have shape (batch, {self.state_dim})")
            if candidate_context_v2.ndim != 3 or candidate_context_v2.shape[2] != LARGE_POLICY_CANDIDATE_FEATURE_DIM:
                raise ValueError(
                    "candidate_context_v2 must have shape "
                    f"(batch, num_candidates, {LARGE_POLICY_CANDIDATE_FEATURE_DIM})"
                )
            if action_embedding_indices.ndim != 2:
                raise ValueError(
                    "action_embedding_indices must have shape (batch, num_candidates)"
                )
            if candidate_mask.ndim != 2:
                raise ValueError("candidate_mask must have shape (batch, num_candidates)")
            if candidate_context_v2.shape[:2] != action_embedding_indices.shape:
                raise ValueError(
                    "candidate_context_v2 and action_embedding_indices must align on batch and candidate axes"
                )
            if candidate_mask.shape != action_embedding_indices.shape:
                raise ValueError(
                    "candidate_mask and action_embedding_indices must have the same shape"
                )

            root_hidden = self.root_projection(z_root).unsqueeze(1)
            candidate_hidden = self.candidate_projection(
                torch.cat(
                    [
                        candidate_context_v2,
                        self.action_embedding(action_embedding_indices),
                        root_hidden.expand(-1, action_embedding_indices.shape[1], -1),
                    ],
                    dim=2,
                )
            )
            candidate_hidden = candidate_hidden * candidate_mask.unsqueeze(2).to(
                candidate_hidden.dtype
            )
            for layer in self.layers:
                candidate_hidden = layer(
                    candidate_hidden,
                    root_hidden=root_hidden,
                    candidate_mask=candidate_mask,
                )
            logits = self.scorer(candidate_hidden).squeeze(2)
            return logits.masked_fill(~candidate_mask, MASKED_CANDIDATE_LOGIT_VALUE)
