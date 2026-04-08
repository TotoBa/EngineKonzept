"""NNUE-style policy readout over shared sparse LAPv2 accumulators."""

from __future__ import annotations

import math

from train.models.value_head_nnue import ClippedReLU

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


NNUE_POLICY_HEAD_MODEL_NAME = "lapv2_nnue_policy_head_v1"


if torch is not None and nn is not None:

    class NNUEPolicyHead(nn.Module):
        """Score candidate successors from root and successor accumulators."""

        def __init__(
            self,
            *,
            accumulator_dim: int = 64,
            move_type_vocab: int = 128,
            move_type_dim: int = 16,
            hidden_dim: int = 32,
        ) -> None:
            super().__init__()
            if accumulator_dim <= 0:
                raise ValueError("accumulator_dim must be positive")
            if move_type_vocab <= 0:
                raise ValueError("move_type_vocab must be positive")
            if move_type_dim <= 0:
                raise ValueError("move_type_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            self.accumulator_dim = int(accumulator_dim)
            self.move_type_vocab = int(move_type_vocab)
            self.move_type_dim = int(move_type_dim)
            self.hidden_dim = int(hidden_dim)
            self.adapter = nn.Linear(accumulator_dim, accumulator_dim)
            self.move_type_emb = nn.Embedding(move_type_vocab, move_type_dim)
            self.clipped_relu = ClippedReLU()
            self.move_head = nn.Sequential(
                nn.Linear((2 * accumulator_dim) + move_type_dim, hidden_dim),
                ClippedReLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.reset_parameters()

        def reset_parameters(self) -> None:
            std = 1.0 / math.sqrt(float(self.accumulator_dim))
            nn.init.normal_(self.adapter.weight, mean=0.0, std=std)
            nn.init.zeros_(self.adapter.bias)
            nn.init.normal_(self.move_type_emb.weight, mean=0.0, std=std)
            for module in self.move_head:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                        nn.init.uniform_(module.bias, -bound, bound)

        def forward(
            self,
            a_root_stm: torch.Tensor,
            a_succ_other: torch.Tensor,
            move_type_ids: torch.Tensor,
        ) -> torch.Tensor:
            if a_root_stm.ndim != 2 or a_root_stm.shape[1] != self.accumulator_dim:
                raise ValueError(
                    f"a_root_stm must have shape (batch, {self.accumulator_dim})"
                )
            if a_succ_other.ndim != 3 or a_succ_other.shape[2] != self.accumulator_dim:
                raise ValueError(
                    "a_succ_other must have shape "
                    f"(batch, num_candidates, {self.accumulator_dim})"
                )
            if move_type_ids.ndim != 2:
                raise ValueError("move_type_ids must have shape (batch, num_candidates)")
            if a_succ_other.shape[:2] != move_type_ids.shape:
                raise ValueError(
                    "a_succ_other and move_type_ids must align on batch and candidate axes"
                )
            root_hidden = self.clipped_relu(self.adapter(a_root_stm)).unsqueeze(1).expand(
                -1,
                a_succ_other.shape[1],
                -1,
            )
            succ_hidden = self.clipped_relu(
                self.adapter(a_succ_other.reshape(-1, self.accumulator_dim))
            ).reshape_as(a_succ_other)
            move_type_hidden = self.move_type_emb(move_type_ids)
            logits = self.move_head(
                torch.cat([root_hidden, succ_hidden, move_type_hidden], dim=2)
            ).squeeze(2)
            return logits

else:  # pragma: no cover - exercised when torch is absent

    class NNUEPolicyHead:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("torch is required to instantiate NNUEPolicyHead.")
