"""Model-only LAPv1 piece-intention encoder components."""

from __future__ import annotations

from train.datasets.artifacts import PIECE_TOKEN_CAPACITY
from train.datasets.contracts import state_context_v1_feature_spec

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM = len(state_context_v1_feature_spec()["global_feature_order"])
INTENTION_ENCODER_MODEL_NAME = "piece_intention_encoder"


if torch is not None and nn is not None:

    class _EdgeMaskedAttentionLayer(nn.Module):
        """Transformer block with graph-derived attention masking."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            num_heads: int,
            feedforward_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
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
            piece_mask: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            attended, _ = self.attention(
                hidden,
                hidden,
                hidden,
                attn_mask=attention_mask,
                need_weights=False,
            )
            hidden = self.attention_norm(hidden + attended)
            hidden = hidden * piece_mask.unsqueeze(2).to(hidden.dtype)
            fed_forward = self.feedforward(hidden)
            hidden = self.feedforward_norm(hidden + fed_forward)
            return hidden * piece_mask.unsqueeze(2).to(hidden.dtype)


    class PieceIntentionEncoder(nn.Module):
        """Encode exact piece tokens into piece-local intention vectors."""

        def __init__(
            self,
            *,
            hidden_dim: int = 128,
            intention_dim: int = 64,
            num_layers: int = 4,
            num_heads: int = 4,
            feedforward_dim: int = 4096,
            dropout: float = 0.0,
            max_edge_count: int = 128,
        ) -> None:
            super().__init__()
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if intention_dim <= 0:
                raise ValueError("intention_dim must be positive")
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

            self.hidden_dim = hidden_dim
            self.intention_dim = intention_dim
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.feedforward_dim = feedforward_dim
            self.max_edge_count = max_edge_count
            self.square_embedding = nn.Embedding(65, hidden_dim, padding_idx=0)
            self.color_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
            self.piece_type_embedding = nn.Embedding(7, hidden_dim, padding_idx=0)
            self.attack_count_embedding = nn.Embedding(34, hidden_dim, padding_idx=0)
            self.global_projection = nn.Sequential(
                nn.Linear(STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )
            self.layers = nn.ModuleList(
                [
                    _EdgeMaskedAttentionLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        feedforward_dim=feedforward_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.output_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, intention_dim),
            )

        def forward(
            self,
            piece_tokens: torch.Tensor,
            state_context_v1_global: torch.Tensor,
            reachability_edges: torch.Tensor,
        ) -> torch.Tensor:
            """Encode piece tokens and reachability edges into piece intentions."""
            if piece_tokens.ndim != 3 or piece_tokens.shape[1:] != (PIECE_TOKEN_CAPACITY, 3):
                raise ValueError(
                    "piece_tokens must have shape (batch, 32, 3)"
                )
            if state_context_v1_global.ndim != 2 or state_context_v1_global.shape[1] != STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM:
                raise ValueError(
                    "state_context_v1_global must have shape "
                    f"(batch, {STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM})"
                )
            if reachability_edges.ndim != 3 or reachability_edges.shape[2] != 3:
                raise ValueError(
                    "reachability_edges must have shape (batch, edge_count, 3)"
                )
            if reachability_edges.shape[1] > self.max_edge_count:
                raise ValueError(
                    "reachability_edges edge_count exceeds configured max_edge_count"
                )

            piece_mask = piece_tokens[:, :, 0] >= 0
            square_indices = _masked_embedding_indices(piece_tokens[:, :, 0], piece_mask)
            color_indices = _masked_embedding_indices(piece_tokens[:, :, 1], piece_mask)
            piece_type_indices = _masked_embedding_indices(piece_tokens[:, :, 2], piece_mask)
            attack_count_indices, attention_mask = _graph_features(
                square_indices=square_indices,
                piece_mask=piece_mask,
                reachability_edges=reachability_edges,
                num_heads=self.num_heads,
            )
            token_hidden = (
                self.square_embedding(square_indices)
                + self.color_embedding(color_indices)
                + self.piece_type_embedding(piece_type_indices)
                + self.attack_count_embedding(attack_count_indices)
                + self.global_projection(state_context_v1_global).unsqueeze(1)
            )
            token_hidden = token_hidden * piece_mask.unsqueeze(2).to(token_hidden.dtype)
            for layer in self.layers:
                token_hidden = layer(
                    token_hidden,
                    piece_mask=piece_mask,
                    attention_mask=attention_mask,
                )
            return self.output_projection(token_hidden) * piece_mask.unsqueeze(2).to(
                token_hidden.dtype
            )


    class KingSpecialHead(nn.Module):
        """Produce dedicated king-side intention summaries from king token rows."""

        def __init__(
            self,
            *,
            intention_dim: int = 64,
            global_dim: int = STATE_CONTEXT_V1_GLOBAL_FEATURE_DIM,
            hidden_dim: int = 256,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.own_tower = nn.Sequential(
                nn.Linear(intention_dim + global_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, intention_dim),
            )
            self.opp_tower = nn.Sequential(
                nn.Linear(intention_dim + global_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, intention_dim),
            )

        def forward(
            self,
            own_king_rows: torch.Tensor,
            opp_king_rows: torch.Tensor,
            global_features: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Return dedicated own/opponent king intention summaries."""
            own = self.own_tower(torch.cat([own_king_rows, global_features], dim=1))
            opp = self.opp_tower(torch.cat([opp_king_rows, global_features], dim=1))
            return own, opp


def _masked_embedding_indices(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    safe_values = values.clamp_min(0).to(dtype=torch.long) + 1
    zeros = torch.zeros_like(safe_values)
    return torch.where(mask, safe_values, zeros)


def _graph_features(
    *,
    square_indices: torch.Tensor,
    piece_mask: torch.Tensor,
    reachability_edges: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, token_count = square_indices.shape
    edge_counts = torch.zeros_like(square_indices, dtype=torch.long)
    allowed = (
        torch.eye(token_count, dtype=torch.bool, device=square_indices.device)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )

    edge_src = reachability_edges[:, :, 0].to(dtype=torch.long)
    edge_dst = reachability_edges[:, :, 1].to(dtype=torch.long)
    valid_edges = (edge_src >= 0) & (edge_dst >= 0)

    # The edge list is square-index based. We project it into piece-token adjacency
    # by matching source and destination squares against the live piece-token squares.
    for batch_index in range(batch_size):
        square_to_token = {
            int(square_indices[batch_index, token_index].item()) - 1: token_index
            for token_index in range(token_count)
            if bool(piece_mask[batch_index, token_index].item())
        }
        for edge_index in range(reachability_edges.shape[1]):
            if not bool(valid_edges[batch_index, edge_index].item()):
                continue
            src_square = int(edge_src[batch_index, edge_index].item())
            dst_square = int(edge_dst[batch_index, edge_index].item())
            if src_square in square_to_token:
                source_token = square_to_token[src_square]
                edge_counts[batch_index, source_token] = torch.clamp(
                    edge_counts[batch_index, source_token] + 1,
                    max=32,
                )
                destination_token = square_to_token.get(dst_square)
                if destination_token is not None:
                    allowed[batch_index, source_token, destination_token] = True
                    allowed[batch_index, destination_token, source_token] = True

        live_tokens = piece_mask[batch_index]
        allowed[batch_index, live_tokens, :] &= live_tokens.unsqueeze(0)

    edge_count_indices = torch.where(
        piece_mask,
        edge_counts + 1,
        torch.zeros_like(edge_counts),
    )
    attention_mask = ~allowed
    attention_mask = (
        attention_mask.unsqueeze(1)
        .expand(batch_size, num_heads, token_count, token_count)
        .reshape(batch_size * num_heads, token_count, token_count)
    )
    return edge_count_indices, attention_mask
