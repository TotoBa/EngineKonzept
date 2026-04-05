"""Model-only LAPv1 relational state embedder components."""

from __future__ import annotations

from train.datasets.artifacts import PIECE_TOKEN_CAPACITY, SQUARE_TOKEN_COUNT
from train.models.intention_encoder import STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


STATE_EMBEDDER_MODEL_NAME = "relational_state_embedder"
DEFAULT_PIECE_INTENTION_DIM = 64
DEFAULT_SQUARE_INPUT_DIM = 2
DEFAULT_STATE_DIM = 512


if torch is not None and nn is not None:

    class _RelationalEmbedderLayer(nn.Module):
        """One mixed self-attention plus edge-attention transformer block."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            num_heads: int,
            feedforward_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.self_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.self_norm = nn.LayerNorm(hidden_dim)
            self.edge_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.edge_norm = nn.LayerNorm(hidden_dim)
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
            hidden: torch.Tensor,
            *,
            token_mask: torch.Tensor,
            full_attention_mask: torch.Tensor,
            edge_attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            self_attended, _ = self.self_attention(
                hidden,
                hidden,
                hidden,
                attn_mask=full_attention_mask,
                need_weights=False,
            )
            hidden = self.self_norm(hidden + self_attended)
            hidden = hidden * token_mask.unsqueeze(2).to(hidden.dtype)

            edge_attended, _ = self.edge_attention(
                hidden,
                hidden,
                hidden,
                attn_mask=edge_attention_mask,
                need_weights=False,
            )
            hidden = self.edge_norm(hidden + edge_attended)
            hidden = hidden * token_mask.unsqueeze(2).to(hidden.dtype)

            fed_forward = self.feedforward(hidden)
            hidden = self.feedforward_norm(hidden + fed_forward)
            return hidden * token_mask.unsqueeze(2).to(hidden.dtype)


    class RelationalStateEmbedder(nn.Module):
        """Pool piece intentions, square tokens, and graph structure into z_root."""

        def __init__(
            self,
            *,
            intention_dim: int = DEFAULT_PIECE_INTENTION_DIM,
            square_input_dim: int = DEFAULT_SQUARE_INPUT_DIM,
            global_dim: int = STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM,
            hidden_dim: int = 256,
            state_dim: int = DEFAULT_STATE_DIM,
            num_layers: int = 6,
            num_heads: int = 8,
            feedforward_dim: int = 2816,
            dropout: float = 0.0,
            max_edge_count: int = 128,
        ) -> None:
            super().__init__()
            if intention_dim <= 0:
                raise ValueError("intention_dim must be positive")
            if square_input_dim <= 0:
                raise ValueError("square_input_dim must be positive")
            if global_dim <= 0:
                raise ValueError("global_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if state_dim <= 0:
                raise ValueError("state_dim must be positive")
            if num_layers <= 0:
                raise ValueError("num_layers must be positive")
            if num_heads <= 0:
                raise ValueError("num_heads must be positive")
            if hidden_dim % num_heads != 0:
                raise ValueError("hidden_dim must be divisible by num_heads")
            if feedforward_dim <= hidden_dim:
                raise ValueError("feedforward_dim must be larger than hidden_dim")
            if max_edge_count <= 0:
                raise ValueError("max_edge_count must be positive")
            if not 0.0 <= dropout < 1.0:
                raise ValueError("dropout must be in [0.0, 1.0)")

            self.intention_dim = intention_dim
            self.square_input_dim = square_input_dim
            self.global_dim = global_dim
            self.hidden_dim = hidden_dim
            self.state_dim = state_dim
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.max_edge_count = max_edge_count

            self.piece_projection = nn.Sequential(
                nn.Linear(intention_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.square_projection = nn.Sequential(
                nn.Linear(square_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.global_projection = nn.Sequential(
                nn.Linear(global_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )
            self.layers = nn.ModuleList(
                [
                    _RelationalEmbedderLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        feedforward_dim=feedforward_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.pool_query = nn.Parameter(torch.randn(hidden_dim) * 0.02)
            self.pool_norm = nn.LayerNorm(hidden_dim)
            self.state_projection = nn.Sequential(
                nn.Linear(hidden_dim + global_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, state_dim),
            )
            self.sigma_head = nn.Sequential(
                nn.Linear(hidden_dim + global_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(
            self,
            piece_intentions: torch.Tensor,
            square_tokens: torch.Tensor,
            global_features: torch.Tensor,
            reachability_edges: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return root latent state and positive root uncertainty."""
            if piece_intentions.ndim != 3 or piece_intentions.shape[1:] != (
                PIECE_TOKEN_CAPACITY,
                self.intention_dim,
            ):
                raise ValueError(
                    "piece_intentions must have shape "
                    f"(batch, {PIECE_TOKEN_CAPACITY}, {self.intention_dim})"
                )
            if square_tokens.ndim != 3 or square_tokens.shape[1:] != (
                SQUARE_TOKEN_COUNT,
                self.square_input_dim,
            ):
                raise ValueError(
                    "square_tokens must have shape "
                    f"(batch, {SQUARE_TOKEN_COUNT}, {self.square_input_dim})"
                )
            if global_features.ndim != 2 or global_features.shape[1] != self.global_dim:
                raise ValueError(
                    f"global_features must have shape (batch, {self.global_dim})"
                )
            if reachability_edges.ndim != 3 or reachability_edges.shape[2] != 3:
                raise ValueError(
                    "reachability_edges must have shape (batch, edge_count, 3)"
                )
            if reachability_edges.shape[1] > self.max_edge_count:
                raise ValueError(
                    "reachability_edges edge_count exceeds configured max_edge_count"
                )

            piece_mask = piece_intentions.abs().sum(dim=2) > 0.0
            square_mask = torch.ones(
                (piece_intentions.shape[0], SQUARE_TOKEN_COUNT),
                dtype=torch.bool,
                device=piece_intentions.device,
            )
            token_mask = torch.cat([piece_mask, square_mask], dim=1)

            hidden = torch.cat(
                [
                    self.piece_projection(piece_intentions),
                    self.square_projection(square_tokens),
                ],
                dim=1,
            )
            hidden = hidden + self.global_projection(global_features).unsqueeze(1)
            hidden = hidden * token_mask.unsqueeze(2).to(hidden.dtype)

            full_attention_mask = _build_full_attention_mask(
                token_mask=token_mask,
                num_heads=self.num_heads,
            )
            edge_attention_mask = _build_edge_attention_mask(
                token_mask=token_mask,
                reachability_edges=reachability_edges,
                num_heads=self.num_heads,
            )
            for layer in self.layers:
                hidden = layer(
                    hidden,
                    token_mask=token_mask,
                    full_attention_mask=full_attention_mask,
                    edge_attention_mask=edge_attention_mask,
                )

            pooling_scores = torch.matmul(self.pool_norm(hidden), self.pool_query)
            pooling_scores = pooling_scores.masked_fill(~token_mask, float("-inf"))
            pooling_weights = torch.softmax(pooling_scores, dim=1)
            pooled_hidden = torch.sum(
                hidden * pooling_weights.unsqueeze(2),
                dim=1,
            )
            pooled_with_global = torch.cat([pooled_hidden, global_features], dim=1)
            z_root = self.state_projection(pooled_with_global)
            sigma_root = torch.nn.functional.softplus(
                self.sigma_head(pooled_with_global)
            ) + 1e-6
            return z_root, sigma_root


def _build_full_attention_mask(
    *,
    token_mask: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    batch_size, token_count = token_mask.shape
    live_connections = token_mask.unsqueeze(1) & token_mask.unsqueeze(2)
    allowed = live_connections | torch.eye(
        token_count,
        dtype=torch.bool,
        device=token_mask.device,
    ).unsqueeze(0)
    return (
        (~allowed)
        .unsqueeze(1)
        .expand(batch_size, num_heads, token_count, token_count)
        .reshape(batch_size * num_heads, token_count, token_count)
    )


def _build_edge_attention_mask(
    *,
    token_mask: torch.Tensor,
    reachability_edges: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    batch_size, token_count = token_mask.shape
    allowed = torch.eye(
        token_count,
        dtype=torch.bool,
        device=token_mask.device,
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    square_offset = PIECE_TOKEN_CAPACITY
    edge_src = reachability_edges[:, :, 0].to(dtype=torch.long)
    edge_dst = reachability_edges[:, :, 1].to(dtype=torch.long)
    valid_edges = (edge_src >= 0) & (edge_dst >= 0)

    for batch_index in range(batch_size):
        for edge_index in range(reachability_edges.shape[1]):
            if not bool(valid_edges[batch_index, edge_index].item()):
                continue
            src_square = int(edge_src[batch_index, edge_index].item())
            dst_square = int(edge_dst[batch_index, edge_index].item())
            src_token = square_offset + src_square
            dst_token = square_offset + dst_square
            allowed[batch_index, src_token, dst_token] = True
            allowed[batch_index, dst_token, src_token] = True

        live_tokens = token_mask[batch_index]
        allowed[batch_index, live_tokens, :] &= live_tokens.unsqueeze(0)

    return (
        (~allowed)
        .unsqueeze(1)
        .expand(batch_size, num_heads, token_count, token_count)
        .reshape(batch_size * num_heads, token_count, token_count)
    )
