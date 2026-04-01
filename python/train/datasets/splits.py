"""Deterministic dataset split assignment."""

from __future__ import annotations

import hashlib
from typing import Sequence

from train.datasets.schema import RawPositionRecord, SplitRatios

SPLIT_NAMES = ("train", "validation", "test")


def assign_splits(
    records: Sequence[RawPositionRecord],
    *,
    ratios: SplitRatios,
    seed: str,
) -> list[str]:
    """Assign deterministic train/validation/test labels."""
    if not records:
        return []

    counts = _split_counts(len(records), ratios)
    ranked_indices = sorted(
        range(len(records)),
        key=lambda index: _stable_digest(records[index].sample_id, seed),
    )

    assignments = [""] * len(records)
    offset = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        for index in ranked_indices[offset : offset + count]:
            assignments[index] = split_name
        offset += count

    return assignments


def _split_counts(record_count: int, ratios: SplitRatios) -> tuple[int, int, int]:
    exact_counts = (
        record_count * ratios.train,
        record_count * ratios.validation,
        record_count * ratios.test,
    )
    floor_counts = [int(value) for value in exact_counts]
    remainder = record_count - sum(floor_counts)
    fractional_order = sorted(
        range(3),
        key=lambda index: (exact_counts[index] - floor_counts[index], -index),
        reverse=True,
    )

    for index in fractional_order[:remainder]:
        floor_counts[index] += 1

    return floor_counts[0], floor_counts[1], floor_counts[2]


def _stable_digest(sample_id: str, seed: str) -> bytes:
    return hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
