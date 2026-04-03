"""Phase-7 baseline evaluation over exact legal replies with the symbolic proposer scorer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Sequence

from train.datasets.opponent_head import (
    load_opponent_head_examples,
    opponent_head_artifact_name,
)
from train.eval.symbolic_proposer import (
    load_symbolic_proposer_checkpoint,
    score_symbolic_candidates,
)


_CAPTURE_FEATURE_INDEX = 0
_PROMOTION_FEATURE_INDEX = 1
_EN_PASSANT_FEATURE_INDEX = 3
_GIVES_CHECK_FEATURE_INDEX = 4


@dataclass(frozen=True)
class OpponentBaselineMetrics:
    """Aggregate metrics for the exact symbolic reply-scorer baseline."""

    total_examples: int
    supervised_examples: int
    reply_top1_accuracy: float
    reply_top3_accuracy: float
    teacher_reply_mean_reciprocal_rank: float
    teacher_reply_mean_probability: float
    pressure_mae: float
    uncertainty_mae: float
    examples_per_second: float

    def to_dict(self) -> dict[str, float | int]:
        """Return the JSON-friendly representation."""
        return {
            "total_examples": self.total_examples,
            "supervised_examples": self.supervised_examples,
            "reply_top1_accuracy": round(self.reply_top1_accuracy, 6),
            "reply_top3_accuracy": round(self.reply_top3_accuracy, 6),
            "teacher_reply_mean_reciprocal_rank": round(
                self.teacher_reply_mean_reciprocal_rank,
                6,
            ),
            "teacher_reply_mean_probability": round(self.teacher_reply_mean_probability, 6),
            "pressure_mae": round(self.pressure_mae, 6),
            "uncertainty_mae": round(self.uncertainty_mae, 6),
            "examples_per_second": round(self.examples_per_second, 3),
        }


def evaluate_symbolic_opponent_baseline(
    checkpoint_path: Path,
    *,
    dataset_path: Path,
    split: str,
) -> OpponentBaselineMetrics:
    """Evaluate the exact symbolic reply-scorer baseline on OpponentHead examples."""
    artifact_path = (
        dataset_path / opponent_head_artifact_name(split)
        if dataset_path.is_dir()
        else dataset_path
    )
    examples = load_opponent_head_examples(artifact_path)
    model, _config = load_symbolic_proposer_checkpoint(checkpoint_path)

    started_at = time.perf_counter()
    supervised_examples = 0
    reply_top1_correct = 0
    reply_top3_correct = 0
    reciprocal_rank_total = 0.0
    teacher_reply_probability_total = 0.0
    pressure_error_total = 0.0
    uncertainty_error_total = 0.0

    for example in examples:
        if example.reply_candidate_action_indices:
            candidate_scores, candidate_policy = score_symbolic_candidates(
                model,
                feature_vector=example.next_feature_vector,
                candidate_action_indices=example.reply_candidate_action_indices,
                candidate_features=example.reply_candidate_features,
                global_features=example.reply_global_features,
                candidate_context_version=example.reply_candidate_context_version,
            )
            ranked_indices = sorted(
                range(len(example.reply_candidate_action_indices)),
                key=lambda index: (-candidate_scores[index], index),
            )
            predicted_top_index = ranked_indices[0]
            pressure_prediction = _pressure_from_candidate_features(
                example.reply_candidate_features[predicted_top_index]
            )
            uncertainty_prediction = 1.0 - max(candidate_policy)
        else:
            candidate_scores = []
            candidate_policy = []
            ranked_indices = []
            pressure_prediction = 0.0
            uncertainty_prediction = 0.0

        if example.teacher_reply_action_index is not None:
            supervised_examples += 1
            try:
                teacher_candidate_index = example.reply_candidate_action_indices.index(
                    example.teacher_reply_action_index
                )
            except ValueError as error:
                raise ValueError(
                    f"{example.sample_id}: teacher reply action missing from exact candidate set"
                ) from error
            reply_top1_correct += int(ranked_indices[:1] == [teacher_candidate_index])
            reply_top3_correct += int(teacher_candidate_index in ranked_indices[:3])
            reciprocal_rank_total += 1.0 / (ranked_indices.index(teacher_candidate_index) + 1)
            teacher_reply_probability_total += candidate_policy[teacher_candidate_index]

        pressure_error_total += abs(pressure_prediction - example.pressure_target)
        uncertainty_error_total += abs(uncertainty_prediction - example.uncertainty_target)

    total_examples = len(examples)
    elapsed = time.perf_counter() - started_at
    return OpponentBaselineMetrics(
        total_examples=total_examples,
        supervised_examples=supervised_examples,
        reply_top1_accuracy=_ratio(reply_top1_correct, supervised_examples),
        reply_top3_accuracy=_ratio(reply_top3_correct, supervised_examples),
        teacher_reply_mean_reciprocal_rank=_ratio(reciprocal_rank_total, supervised_examples),
        teacher_reply_mean_probability=_ratio(
            teacher_reply_probability_total,
            supervised_examples,
        ),
        pressure_mae=_ratio(pressure_error_total, total_examples),
        uncertainty_mae=_ratio(uncertainty_error_total, total_examples),
        examples_per_second=_ratio(total_examples, elapsed),
    )


def _pressure_from_candidate_features(candidate_features: Sequence[float]) -> float:
    if not candidate_features:
        return 0.0
    if bool(candidate_features[_GIVES_CHECK_FEATURE_INDEX]):
        return 1.0
    if bool(candidate_features[_PROMOTION_FEATURE_INDEX]):
        return 0.75
    if bool(candidate_features[_CAPTURE_FEATURE_INDEX]) or bool(
        candidate_features[_EN_PASSANT_FEATURE_INDEX]
    ):
        return 0.5
    return 0.0


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
