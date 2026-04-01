"""Python-side action-space helpers aligned with the Rust factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

FROM_HEAD_SIZE = 64
TO_HEAD_SIZE = 64
PROMOTION_HEAD_SIZE = 5
ACTION_SPACE_SIZE = FROM_HEAD_SIZE * TO_HEAD_SIZE * PROMOTION_HEAD_SIZE


@dataclass(frozen=True)
class ActionEncoding:
    """Validated factorized action encoding."""

    from_index: int
    to_index: int
    promotion_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.from_index < FROM_HEAD_SIZE:
            raise ValueError(f"from_index out of range: {self.from_index}")
        if not 0 <= self.to_index < TO_HEAD_SIZE:
            raise ValueError(f"to_index out of range: {self.to_index}")
        if not 0 <= self.promotion_index < PROMOTION_HEAD_SIZE:
            raise ValueError(f"promotion_index out of range: {self.promotion_index}")

    @classmethod
    def from_sequence(cls, values: Sequence[int]) -> "ActionEncoding":
        """Construct a validated action encoding from a sequence of indices."""
        if len(values) != 3:
            raise ValueError(f"expected 3 action indices, got {len(values)}")
        return cls(
            from_index=int(values[0]),
            to_index=int(values[1]),
            promotion_index=int(values[2]),
        )

    def as_list(self) -> list[int]:
        """Return the JSON-friendly list representation."""
        return [self.from_index, self.to_index, self.promotion_index]

    def flat_index(self) -> int:
        """Return the flat joint-action index used by the proposer heads."""
        return flatten_action(self.as_list())


def flatten_action(values: Sequence[int]) -> int:
    """Flatten a factorized action encoding into a single vocabulary index."""
    encoding = ActionEncoding.from_sequence(values)
    return (
        encoding.from_index * TO_HEAD_SIZE + encoding.to_index
    ) * PROMOTION_HEAD_SIZE + encoding.promotion_index


def unflatten_action(index: int) -> list[int]:
    """Recover the factorized action encoding from a flat vocabulary index."""
    if not 0 <= index < ACTION_SPACE_SIZE:
        raise ValueError(f"action index out of range: {index}")
    promotion_index = index % PROMOTION_HEAD_SIZE
    to_bucket = index // PROMOTION_HEAD_SIZE
    to_index = to_bucket % TO_HEAD_SIZE
    from_index = to_bucket // TO_HEAD_SIZE
    return [from_index, to_index, promotion_index]


def flatten_legal_actions(actions: Sequence[Sequence[int]]) -> list[int]:
    """Flatten and deterministically sort a legal-action set."""
    return sorted({flatten_action(action) for action in actions})


def action_space_metadata() -> dict[str, object]:
    """Describe the proposer action layout for exported metadata."""
    return {
        "from_head_size": FROM_HEAD_SIZE,
        "to_head_size": TO_HEAD_SIZE,
        "promotion_head_size": PROMOTION_HEAD_SIZE,
        "flat_size": ACTION_SPACE_SIZE,
        "flatten_formula": "((from_index * 64) + to_index) * 5 + promotion_index",
    }
