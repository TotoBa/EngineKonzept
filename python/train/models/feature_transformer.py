"""Sparse HalfKA feature transformer for future NNUE-style LAPv2 heads."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None
    nn = None


FEATURE_TRANSFORMER_MODEL_NAME = "lapv2_feature_transformer_v1"


if torch is not None and nn is not None:

    class FeatureTransformer(nn.Module):
        """EmbeddingBag-based sparse feature transformer."""

        def __init__(
            self,
            *,
            num_features: int = 49152,
            accumulator_dim: int = 64,
        ) -> None:
            super().__init__()
            if num_features <= 0:
                raise ValueError("num_features must be positive")
            if accumulator_dim <= 0:
                raise ValueError("accumulator_dim must be positive")
            self.num_features = int(num_features)
            self.accumulator_dim = int(accumulator_dim)
            self.ft = nn.EmbeddingBag(num_features, accumulator_dim, mode="sum")

        def build(
            self,
            indices: torch.Tensor,
            offsets: torch.Tensor,
        ) -> torch.Tensor:
            """Build one accumulator row per sparse feature list."""
            if indices.ndim != 1:
                raise ValueError("indices must be rank-1")
            if offsets.ndim != 1:
                raise ValueError("offsets must be rank-1")
            if offsets.numel() == 0:
                return torch.zeros(
                    (0, self.accumulator_dim),
                    dtype=self.ft.weight.dtype,
                    device=self.ft.weight.device,
                )
            if indices.numel() == 0:
                return torch.zeros(
                    (int(offsets.shape[0]), self.accumulator_dim),
                    dtype=self.ft.weight.dtype,
                    device=self.ft.weight.device,
                )
            return self.ft(indices.to(torch.long), offsets.to(torch.long))

        def gather_rows(self, indices: torch.Tensor) -> torch.Tensor:
            """Return raw FT rows for incremental updates."""
            if indices.ndim != 1:
                raise ValueError("indices must be rank-1")
            return self.ft.weight[indices.to(torch.long)]

else:  # pragma: no cover - exercised when torch is absent

    class FeatureTransformer:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("torch is required to instantiate FeatureTransformer.")
