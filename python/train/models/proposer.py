"""PyTorch proposer models for legality and policy prediction."""

from __future__ import annotations

from typing import Any

from train.action_space import (
    ACTION_SPACE_SIZE,
    FROM_HEAD_SIZE,
    PROMOTION_HEAD_SIZE,
    TO_HEAD_SIZE,
    unflatten_action,
)
from train.datasets.artifacts import (
    PIECE_TOKEN_CAPACITY,
    PIECE_TOKEN_PADDING_VALUE,
    PIECE_TOKEN_WIDTH,
    POSITION_FEATURE_SIZE,
    RULE_TOKEN_WIDTH,
    SQUARE_TOKEN_COUNT,
    SQUARE_TOKEN_WIDTH,
)

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


MODEL_NAME = "legality_policy_proposer"


def torch_is_available() -> bool:
    """Report whether PyTorch is available in the current environment."""
    return torch is not None and nn is not None


if nn is not None:

    class LegalityPolicyProposer(nn.Module):
        """Configurable proposer over the deterministic Phase-3 encoder features."""

        def __init__(
            self,
            *,
            architecture: str,
            hidden_dim: int,
            hidden_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if hidden_layers <= 0:
                raise ValueError("hidden_layers must be positive")
            if architecture == "mlp_v1":
                self._impl = _MlpProposer(
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
            elif architecture == "multistream_v2":
                self._impl = _MultiStreamProposer(
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
            elif architecture == "factorized_v3":
                self._impl = _FactorizedMlpProposer(
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
            elif architecture == "factorized_v4":
                self._impl = _ConditionalFactorizedMlpProposer(
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
            elif architecture == "factorized_v5":
                self._impl = _PolicyResidualFactorizedMlpProposer(
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"unsupported proposer architecture: {architecture}")

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Return legality logits and policy logits for the flat action space."""
            return self._impl(features)


    class _MlpProposer(nn.Module):
        def __init__(self, *, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            input_dim = POSITION_FEATURE_SIZE
            for _ in range(hidden_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)
            self.legality_head = nn.Linear(input_dim, ACTION_SPACE_SIZE)
            self.policy_head = nn.Linear(input_dim, ACTION_SPACE_SIZE)

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.backbone(features)
            return self.legality_head(hidden), self.policy_head(hidden)


    class _FactorizedActionHead(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.from_head = nn.Linear(hidden_dim, FROM_HEAD_SIZE)
            self.to_head = nn.Linear(hidden_dim, TO_HEAD_SIZE)
            self.promotion_head = nn.Linear(hidden_dim, PROMOTION_HEAD_SIZE)
            from_indices, to_indices, promotion_indices = _build_action_component_indices()
            self.register_buffer("from_indices", from_indices, persistent=False)
            self.register_buffer("to_indices", to_indices, persistent=False)
            self.register_buffer("promotion_indices", promotion_indices, persistent=False)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            from_logits = self.from_head(hidden).index_select(1, self.from_indices)
            to_logits = self.to_head(hidden).index_select(1, self.to_indices)
            promotion_logits = self.promotion_head(hidden).index_select(
                1, self.promotion_indices
            )
            return from_logits + to_logits + promotion_logits


    class _FactorizedMlpProposer(nn.Module):
        """MLP proposer with a factorized decoder over the existing action schema."""

        def __init__(self, *, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            input_dim = POSITION_FEATURE_SIZE
            for _ in range(hidden_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)
            self.legality_tower = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.policy_tower = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.legality_head = _FactorizedActionHead(hidden_dim)
            self.policy_head = _FactorizedActionHead(hidden_dim)

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.backbone(features)
            legality_hidden = self.legality_tower(hidden)
            policy_hidden = self.policy_tower(hidden)
            return self.legality_head(legality_hidden), self.policy_head(policy_hidden)


    class _ConditionalFactorizedActionHead(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.from_head = nn.Linear(hidden_dim, FROM_HEAD_SIZE)
            self.from_condition = nn.Linear(hidden_dim, FROM_HEAD_SIZE * hidden_dim)
            self.to_base = nn.Linear(hidden_dim, TO_HEAD_SIZE)
            self.to_condition = nn.Linear(hidden_dim, TO_HEAD_SIZE * hidden_dim)
            self.promotion_base = nn.Linear(hidden_dim, PROMOTION_HEAD_SIZE)
            self.to_keys = nn.Parameter(torch.empty(TO_HEAD_SIZE, hidden_dim))
            self.promotion_from_keys = nn.Parameter(torch.empty(PROMOTION_HEAD_SIZE, hidden_dim))
            self.promotion_to_keys = nn.Parameter(torch.empty(PROMOTION_HEAD_SIZE, hidden_dim))
            nn.init.xavier_uniform_(self.to_keys)
            nn.init.xavier_uniform_(self.promotion_from_keys)
            nn.init.xavier_uniform_(self.promotion_to_keys)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            batch_size = hidden.shape[0]
            from_logits = self.from_head(hidden)
            from_queries = self.from_condition(hidden).reshape(
                batch_size, FROM_HEAD_SIZE, self.hidden_dim
            )
            to_queries = self.to_condition(hidden).reshape(
                batch_size, TO_HEAD_SIZE, self.hidden_dim
            )

            to_logits = (
                from_logits.unsqueeze(2)
                + self.to_base(hidden).unsqueeze(1)
                + torch.matmul(from_queries, self.to_keys.transpose(0, 1))
            )
            promotion_logits = (
                to_logits.unsqueeze(-1)
                + self.promotion_base(hidden).reshape(batch_size, 1, 1, PROMOTION_HEAD_SIZE)
                + torch.matmul(
                    from_queries, self.promotion_from_keys.transpose(0, 1)
                ).unsqueeze(2)
                + torch.matmul(to_queries, self.promotion_to_keys.transpose(0, 1)).unsqueeze(1)
            )
            return promotion_logits.reshape(batch_size, ACTION_SPACE_SIZE)


    class _ConditionalFactorizedMlpProposer(nn.Module):
        """MLP proposer with conditional coupling between move components."""

        def __init__(self, *, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            input_dim = POSITION_FEATURE_SIZE
            for _ in range(hidden_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            self.backbone = nn.Sequential(*layers)
            self.legality_tower = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.policy_tower = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.legality_head = _ConditionalFactorizedActionHead(hidden_dim)
            self.policy_head = _ConditionalFactorizedActionHead(hidden_dim)

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.backbone(features)
            legality_hidden = self.legality_tower(hidden)
            policy_hidden = self.policy_tower(hidden)
            return self.legality_head(legality_hidden), self.policy_head(policy_hidden)


    class _LowRankActionResidual(nn.Module):
        def __init__(self, hidden_dim: int, residual_rank: int) -> None:
            super().__init__()
            self.down = nn.Linear(hidden_dim, residual_rank)
            self.up = nn.Linear(residual_rank, ACTION_SPACE_SIZE)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.up(self.down(hidden))


    class _PolicyResidualFactorizedMlpProposer(nn.Module):
        """Conditional factorized proposer with extra policy-specific flat residual capacity."""

        def __init__(self, *, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            input_dim = POSITION_FEATURE_SIZE
            for _ in range(hidden_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            residual_rank = max(hidden_dim // 8, 16)
            self.backbone = nn.Sequential(*layers)
            self.legality_tower = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.policy_tower = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.legality_head = _ConditionalFactorizedActionHead(hidden_dim)
            self.policy_factorized_head = _ConditionalFactorizedActionHead(hidden_dim)
            self.policy_residual = _LowRankActionResidual(hidden_dim, residual_rank)

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.backbone(features)
            legality_hidden = self.legality_tower(hidden)
            policy_hidden = self.policy_tower(hidden)
            legality_logits = self.legality_head(legality_hidden)
            policy_logits = self.policy_factorized_head(policy_hidden) + self.policy_residual(
                policy_hidden
            )
            return legality_logits, policy_logits


    class _CrossAttentionBlock(nn.Module):
        def __init__(self, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            num_heads = 4 if hidden_dim % 4 == 0 else 2
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.ffn_norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            query: torch.Tensor,
            *,
            query_padding_mask: torch.Tensor | None,
            key_value: torch.Tensor,
            key_padding_mask: torch.Tensor | None,
        ) -> torch.Tensor:
            attended, _ = self.attention(
                query=query,
                key=key_value,
                value=key_value,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hidden = self.norm(query + attended)
            hidden = self.ffn_norm(hidden + self.ffn(hidden))
            if query_padding_mask is not None:
                hidden = hidden.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)
            return hidden


    class _MultiStreamProposer(nn.Module):
        """Structured proposer that preserves piece/square/rule streams before fusion."""

        def __init__(self, *, hidden_dim: int, hidden_layers: int, dropout: float) -> None:
            super().__init__()
            self.piece_projection = nn.Sequential(
                nn.Linear(PIECE_TOKEN_WIDTH, hidden_dim),
                nn.ReLU(),
            )
            self.square_projection = nn.Sequential(
                nn.Linear(SQUARE_TOKEN_WIDTH, hidden_dim),
                nn.ReLU(),
            )
            self.rule_projection = nn.Sequential(
                nn.Linear(RULE_TOKEN_WIDTH, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.piece_reads_square = _CrossAttentionBlock(hidden_dim, dropout)
            self.square_reads_piece = _CrossAttentionBlock(hidden_dim, dropout)

            fusion_layers: list[nn.Module] = []
            fusion_input_dim = hidden_dim * 5
            for _ in range(hidden_layers):
                fusion_layers.append(nn.Linear(fusion_input_dim, hidden_dim))
                fusion_layers.append(nn.ReLU())
                if dropout > 0.0:
                    fusion_layers.append(nn.Dropout(dropout))
                fusion_input_dim = hidden_dim
            self.fusion = nn.Sequential(*fusion_layers)

            self.legality_tower = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.policy_tower = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            )
            self.legality_head = nn.Linear(hidden_dim, ACTION_SPACE_SIZE)
            self.policy_head = nn.Linear(hidden_dim, ACTION_SPACE_SIZE)

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            piece_features, square_features, rule_features = _unpack_feature_streams(features)
            piece_padding_mask = piece_features[:, :, 0] == float(PIECE_TOKEN_PADDING_VALUE)
            piece_hidden = self.piece_projection(piece_features)
            square_hidden = self.square_projection(square_features)
            rule_hidden = self.rule_projection(rule_features)

            piece_hidden = self.piece_reads_square(
                piece_hidden,
                query_padding_mask=piece_padding_mask,
                key_value=square_hidden,
                key_padding_mask=None,
            )
            square_hidden = self.square_reads_piece(
                square_hidden,
                query_padding_mask=None,
                key_value=piece_hidden,
                key_padding_mask=piece_padding_mask,
            )

            piece_pooled = _masked_mean(piece_hidden, piece_padding_mask)
            square_pooled = square_hidden.mean(dim=1)
            square_max = square_hidden.amax(dim=1)
            fused = torch.cat(
                [
                    piece_pooled,
                    square_pooled,
                    square_max,
                    rule_hidden,
                    piece_pooled * rule_hidden,
                ],
                dim=1,
            )
            hidden = self.fusion(fused)
            legality_hidden = self.legality_tower(hidden)
            policy_hidden = self.policy_tower(hidden)
            return self.legality_head(legality_hidden), self.policy_head(policy_hidden)


    def _unpack_feature_streams(
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        piece_width = PIECE_TOKEN_CAPACITY * PIECE_TOKEN_WIDTH
        square_width = SQUARE_TOKEN_COUNT * SQUARE_TOKEN_WIDTH
        piece_block = features[:, :piece_width].reshape(
            -1, PIECE_TOKEN_CAPACITY, PIECE_TOKEN_WIDTH
        )
        square_block = features[:, piece_width : piece_width + square_width].reshape(
            -1, SQUARE_TOKEN_COUNT, SQUARE_TOKEN_WIDTH
        )
        rule_block = features[:, piece_width + square_width :]
        return piece_block, square_block, rule_block


    def _masked_mean(values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        keep_mask = (~padding_mask).unsqueeze(-1)
        summed = (values * keep_mask).sum(dim=1)
        counts = keep_mask.sum(dim=1).clamp_min(1)
        return summed / counts


    def _build_action_component_indices() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_components = [unflatten_action(index) for index in range(ACTION_SPACE_SIZE)]
        return (
            torch.tensor([parts[0] for parts in action_components], dtype=torch.long),
            torch.tensor([parts[1] for parts in action_components], dtype=torch.long),
            torch.tensor([parts[2] for parts in action_components], dtype=torch.long),
        )


else:

    class LegalityPolicyProposer:  # pragma: no cover - exercised when torch is absent
        """Import-safe fallback when PyTorch is not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for proposer training. Install the 'train' extra or torch."
            )
