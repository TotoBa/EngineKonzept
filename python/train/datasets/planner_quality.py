"""Quality filters for planner-head workflow artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from train.datasets.planner_head import PlannerHeadExample


@dataclass(frozen=True)
class PlannerHeadFilterSummary:
    """Summary statistics for planner-head quality filtering."""

    total_examples: int
    kept_examples: int
    dropped_examples: int
    dropped_nan_or_extreme_root_value: int
    dropped_ambiguous_scores: int
    dropped_too_few_candidates: int
    kept_average_candidate_count: float
    dropped_average_candidate_count: float
    kept_average_abs_root_value_cp: float
    dropped_average_abs_root_value_cp: float
    kept_average_score_span_cp: float
    dropped_average_score_span_cp: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_examples": self.total_examples,
            "kept_examples": self.kept_examples,
            "dropped_examples": self.dropped_examples,
            "dropped_nan_or_extreme_root_value": self.dropped_nan_or_extreme_root_value,
            "dropped_ambiguous_scores": self.dropped_ambiguous_scores,
            "dropped_too_few_candidates": self.dropped_too_few_candidates,
            "kept_average_candidate_count": self.kept_average_candidate_count,
            "dropped_average_candidate_count": self.dropped_average_candidate_count,
            "kept_average_abs_root_value_cp": self.kept_average_abs_root_value_cp,
            "dropped_average_abs_root_value_cp": self.dropped_average_abs_root_value_cp,
            "kept_average_score_span_cp": self.kept_average_score_span_cp,
            "dropped_average_score_span_cp": self.dropped_average_score_span_cp,
        }


def filter_planner_head_examples(
    examples: Sequence[PlannerHeadExample],
    *,
    max_abs_root_value_cp: float = 2000.0,
    ambiguous_score_span_cp: float = 5.0,
    min_candidate_count: int = 2,
) -> tuple[list[PlannerHeadExample], PlannerHeadFilterSummary]:
    """Filter planner-head rows by teacher-signal quality."""
    if max_abs_root_value_cp <= 0.0:
        raise ValueError("max_abs_root_value_cp must be positive")
    if ambiguous_score_span_cp < 0.0:
        raise ValueError("ambiguous_score_span_cp must be non-negative")
    if min_candidate_count <= 0:
        raise ValueError("min_candidate_count must be positive")

    kept: list[PlannerHeadExample] = []
    dropped: list[PlannerHeadExample] = []
    dropped_nan_or_extreme_root_value = 0
    dropped_ambiguous_scores = 0
    dropped_too_few_candidates = 0

    for example in examples:
        reasons = _drop_reasons(
            example,
            max_abs_root_value_cp=max_abs_root_value_cp,
            ambiguous_score_span_cp=ambiguous_score_span_cp,
            min_candidate_count=min_candidate_count,
        )
        if reasons:
            dropped.append(example)
            dropped_nan_or_extreme_root_value += int("nan_or_extreme_root_value" in reasons)
            dropped_ambiguous_scores += int("ambiguous_scores" in reasons)
            dropped_too_few_candidates += int("too_few_candidates" in reasons)
            continue
        kept.append(example)

    summary = PlannerHeadFilterSummary(
        total_examples=len(examples),
        kept_examples=len(kept),
        dropped_examples=len(dropped),
        dropped_nan_or_extreme_root_value=dropped_nan_or_extreme_root_value,
        dropped_ambiguous_scores=dropped_ambiguous_scores,
        dropped_too_few_candidates=dropped_too_few_candidates,
        kept_average_candidate_count=_average_candidate_count(kept),
        dropped_average_candidate_count=_average_candidate_count(dropped),
        kept_average_abs_root_value_cp=_average_abs_root_value_cp(kept),
        dropped_average_abs_root_value_cp=_average_abs_root_value_cp(dropped),
        kept_average_score_span_cp=_average_score_span_cp(kept),
        dropped_average_score_span_cp=_average_score_span_cp(dropped),
    )
    return kept, summary


def _drop_reasons(
    example: PlannerHeadExample,
    *,
    max_abs_root_value_cp: float,
    ambiguous_score_span_cp: float,
    min_candidate_count: int,
) -> set[str]:
    reasons: set[str] = set()
    abs_root_value_cp = abs(float(example.teacher_root_value_cp))
    if math.isnan(abs_root_value_cp) or abs_root_value_cp > max_abs_root_value_cp:
        reasons.add("nan_or_extreme_root_value")

    candidate_count = len(example.candidate_action_indices)
    if candidate_count < min_candidate_count:
        reasons.add("too_few_candidates")

    score_span_cp = planner_head_score_span_cp(example)
    if (
        candidate_count >= min_candidate_count
        and score_span_cp is not None
        and score_span_cp <= ambiguous_score_span_cp
    ):
        reasons.add("ambiguous_scores")
    return reasons


def planner_head_score_span_cp(example: PlannerHeadExample) -> float | None:
    """Return the teacher score spread for one bounded candidate slice."""
    if not example.teacher_candidate_scores_cp:
        return None
    return float(max(example.teacher_candidate_scores_cp) - min(example.teacher_candidate_scores_cp))


def _average_candidate_count(examples: Sequence[PlannerHeadExample]) -> float:
    if not examples:
        return 0.0
    return float(sum(len(example.candidate_action_indices) for example in examples) / len(examples))


def _average_abs_root_value_cp(examples: Sequence[PlannerHeadExample]) -> float:
    if not examples:
        return 0.0
    return float(sum(abs(float(example.teacher_root_value_cp)) for example in examples) / len(examples))


def _average_score_span_cp(examples: Sequence[PlannerHeadExample]) -> float:
    spans = [
        planner_head_score_span_cp(example)
        for example in examples
    ]
    finite_spans = [span for span in spans if span is not None]
    if not finite_spans:
        return 0.0
    return float(sum(finite_spans) / len(finite_spans))
