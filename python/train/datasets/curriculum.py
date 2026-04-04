"""Curriculum-weighted sampling helpers for planner-head training."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, Iterator, Sequence

try:  # pragma: no cover - exercised through trainer/tests when torch is installed
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from train.datasets.planner_head import PlannerHeadExample


SUPPORTED_CURRICULUM_STRATEGIES = {"uniform", "linear_ramp", "sqrt_ramp"}
_WEIGHT_FLOOR = 1e-3


def compute_curriculum_weights(
    examples: Sequence["PlannerHeadExample"],
    strategy: str = "linear_ramp",
    *,
    epoch: int = 0,
    total_epochs: int = 1,
    value_spread_weight: float = 1.0,
    candidate_count_weight: float = 1.0,
    agreement_weight: float = 1.0,
) -> list[float]:
    """Proxy to the canonical planner-head curriculum weighting function."""
    from train.datasets.planner_head import compute_curriculum_weights as _compute_weights

    return _compute_weights(
        examples,
        strategy=strategy,
        epoch=epoch,
        total_epochs=total_epochs,
        value_spread_weight=value_spread_weight,
        candidate_count_weight=candidate_count_weight,
        agreement_weight=agreement_weight,
    )


class CurriculumSampler:
    """Deterministic epoch-aware sampler over planner-head examples."""

    def __init__(
        self,
        dataset: Sequence["PlannerHeadExample"],
        *,
        strategy: str = "uniform",
        seed: int = 0,
        total_epochs: int = 1,
        value_spread_weight: float = 1.0,
        candidate_count_weight: float = 1.0,
        agreement_weight: float = 1.0,
    ) -> None:
        if strategy not in SUPPORTED_CURRICULUM_STRATEGIES:
            raise ValueError(
                f"unsupported curriculum strategy: {strategy!r}; expected one of "
                f"{sorted(SUPPORTED_CURRICULUM_STRATEGIES)}"
            )
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        self._dataset = dataset
        self._strategy = strategy
        self._seed = int(seed)
        self._total_epochs = int(total_epochs)
        self._epoch = 0
        self._value_spread_weight = float(value_spread_weight)
        self._candidate_count_weight = float(candidate_count_weight)
        self._agreement_weight = float(agreement_weight)

    @property
    def strategy(self) -> str:
        return self._strategy

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = int(epoch)

    def current_weights(self) -> list[float]:
        return compute_curriculum_weights(
            self._dataset,
            strategy=self._strategy,
            epoch=self._epoch,
            total_epochs=self._total_epochs,
            value_spread_weight=self._value_spread_weight,
            candidate_count_weight=self._candidate_count_weight,
            agreement_weight=self._agreement_weight,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __iter__(self) -> Iterator[int]:
        population = len(self._dataset)
        if population == 0:
            return iter(())
        if self._strategy == "uniform":
            return iter(self._uniform_indices())
        return iter(self._weighted_indices())

    def _uniform_indices(self) -> list[int]:
        if torch is not None:
            generator = torch.Generator()
            generator.manual_seed(self._seed + self._epoch)
            return [int(index) for index in torch.randperm(len(self), generator=generator).tolist()]
        indices = list(range(len(self)))
        random.Random(self._seed + self._epoch).shuffle(indices)
        return indices

    def _weighted_indices(self) -> list[int]:
        weights = self.current_weights()
        if torch is not None:
            generator = torch.Generator()
            generator.manual_seed(self._seed + self._epoch)
            return [
                int(index)
                for index in torch.multinomial(
                    torch.tensor(weights, dtype=torch.float32),
                    len(weights),
                    replacement=False,
                    generator=generator,
                ).tolist()
            ]
        return _weighted_sample_without_replacement(
            weights=weights,
            seed=self._seed + self._epoch,
        )


def _weighted_sample_without_replacement(*, weights: Sequence[float], seed: int) -> list[int]:
    keyed_indices: list[tuple[float, int]] = []
    rng = random.Random(seed)
    for index, weight in enumerate(weights):
        clamped_weight = max(float(weight), _WEIGHT_FLOOR)
        key = math.log(max(rng.random(), 1e-12)) / clamped_weight
        keyed_indices.append((key, index))
    keyed_indices.sort(reverse=True)
    return [index for _, index in keyed_indices]
