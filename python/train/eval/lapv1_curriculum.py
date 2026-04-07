"""Helpers for building hard-position LAPv1 Stage-T2 curricula."""

from __future__ import annotations

from math import log1p

from train.datasets.lapv1_training import LAPv1TrainingExample


def lapv1_hardness_score(
    example: LAPv1TrainingExample,
    *,
    gap_cap_cp: float = 128.0,
    value_cap_cp: float = 256.0,
    candidate_count_cap: int = 8,
) -> float:
    """Return a deterministic hardness score for one precomputed LAPv1 row.

    The score favors:
    - rows with small top1-vs-top2 gaps
    - rows near value balance
    - rows that already carried high curriculum priority upstream
    - rows with broader legal choice sets
    """

    if gap_cap_cp <= 0.0:
        raise ValueError("gap_cap_cp must be positive")
    if value_cap_cp <= 0.0:
        raise ValueError("value_cap_cp must be positive")
    if candidate_count_cap <= 0:
        raise ValueError("candidate_count_cap must be positive")

    gap_cp = float(abs(example.teacher_top1_minus_top2_cp or 0.0))
    gap_term = 1.0 - min(gap_cp, gap_cap_cp) / gap_cap_cp
    value_term = 1.0 - min(abs(example.teacher_root_value_cp), value_cap_cp) / value_cap_cp
    candidate_term = min(
        len(example.candidate_action_indices),
        candidate_count_cap,
    ) / candidate_count_cap
    curriculum_term = log1p(max(example.curriculum_priority, 0.0))
    return (
        1.5 * gap_term
        + 1.25 * float(example.sharpness_target)
        + 1.0 * curriculum_term
        + 0.5 * value_term
        + 0.25 * candidate_term
    )
