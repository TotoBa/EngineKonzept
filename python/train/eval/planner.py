"""Bounded planner-style offline evaluation over proposer and opponent contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from train.datasets import (
    build_symbolic_proposer_example,
    build_transition_context_features,
    dataset_example_from_oracle_payload,
    load_search_teacher_examples,
    load_split_examples,
    move_uci_for_action,
    pack_position_features,
)
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import RawPositionRecord
from train.eval.opponent import (
    load_opponent_head_checkpoint,
    load_symbolic_proposer_checkpoint,
    score_opponent_candidates,
    score_symbolic_candidates,
)


@dataclass(frozen=True)
class PlannerBaselineMetrics:
    """Aggregate metrics for the first bounded opponent-aware planner baseline."""

    total_examples: int
    teacher_covered_examples: int
    root_top1_accuracy: float
    teacher_root_mean_reciprocal_rank: float
    teacher_root_mean_probability: float
    mean_reply_peak_probability: float
    mean_pressure: float
    mean_uncertainty: float
    examples_per_second: float

    def to_dict(self) -> dict[str, float | int]:
        """Return the JSON-friendly representation."""
        return {
            "total_examples": self.total_examples,
            "teacher_covered_examples": self.teacher_covered_examples,
            "root_top1_accuracy": round(self.root_top1_accuracy, 6),
            "teacher_root_mean_reciprocal_rank": round(
                self.teacher_root_mean_reciprocal_rank,
                6,
            ),
            "teacher_root_mean_probability": round(
                self.teacher_root_mean_probability,
                6,
            ),
            "mean_reply_peak_probability": round(self.mean_reply_peak_probability, 6),
            "mean_pressure": round(self.mean_pressure, 6),
            "mean_uncertainty": round(self.mean_uncertainty, 6),
            "examples_per_second": round(self.examples_per_second, 3),
        }


def evaluate_two_ply_planner_baseline(
    proposer_checkpoint: Path,
    *,
    dataset_dir: Path,
    search_teacher_path: Path,
    split: str,
    opponent_mode: str,
    opponent_checkpoint: Path | None = None,
    root_top_k: int = 4,
    reply_peak_weight: float = 0.5,
    pressure_weight: float = 0.25,
    uncertainty_weight: float = 0.25,
    repo_root: Path,
) -> PlannerBaselineMetrics:
    """Evaluate a bounded two-ply planner-style aggregation over exact legal moves."""
    if opponent_mode not in {"none", "symbolic", "learned"}:
        raise ValueError("opponent_mode must be 'none', 'symbolic', or 'learned'")
    if root_top_k <= 0:
        raise ValueError("root_top_k must be positive")
    if opponent_mode == "learned" and opponent_checkpoint is None:
        raise ValueError("opponent_checkpoint is required when opponent_mode='learned'")

    proposer_model, _ = load_symbolic_proposer_checkpoint(proposer_checkpoint)
    opponent_model = None
    if opponent_mode == "learned":
        opponent_model, _ = load_opponent_head_checkpoint(opponent_checkpoint)

    dataset_examples = load_split_examples(dataset_dir, split)
    if not dataset_examples:
        raise ValueError(f"dataset split is empty: {split}")
    search_teacher_examples = load_search_teacher_examples(search_teacher_path)
    if not search_teacher_examples:
        raise ValueError(f"search-teacher artifact is empty: {search_teacher_path}")

    dataset_by_sample_id = {example.sample_id: example for example in dataset_examples}
    started_at = time.perf_counter()
    teacher_covered_examples = 0
    root_top1_correct = 0
    reciprocal_rank_total = 0.0
    teacher_root_probability_total = 0.0
    reply_peak_total = 0.0
    pressure_total = 0.0
    uncertainty_total = 0.0

    for teacher_example in search_teacher_examples:
        dataset_example = dataset_by_sample_id.get(teacher_example.sample_id)
        if dataset_example is None:
            raise ValueError(
                f"{teacher_example.sample_id}: missing dataset example for planner evaluation"
            )
        root_scores, _root_policy = score_symbolic_candidates(
            proposer_model,
            feature_vector=teacher_example.feature_vector,
            candidate_action_indices=teacher_example.candidate_action_indices,
            candidate_features=teacher_example.candidate_features,
            global_features=teacher_example.global_features,
            candidate_context_version=teacher_example.candidate_context_version,
        )
        ranked_root_indices = sorted(
            range(len(teacher_example.candidate_action_indices)),
            key=lambda index: (-root_scores[index], index),
        )
        considered_indices = ranked_root_indices[: min(root_top_k, len(ranked_root_indices))]
        planner_rows = _evaluate_root_candidates(
            dataset_example,
            teacher_example=teacher_example,
            considered_indices=considered_indices,
            root_scores=root_scores,
            proposer_model=proposer_model,
            opponent_model=opponent_model,
            opponent_mode=opponent_mode,
            reply_peak_weight=reply_peak_weight,
            pressure_weight=pressure_weight,
            uncertainty_weight=uncertainty_weight,
            repo_root=repo_root,
        )
        planner_rows.sort(key=lambda row: (-row["planner_score"], row["candidate_list_index"]))
        planner_policy = _planner_policy([row["planner_score"] for row in planner_rows])
        predicted_action_index = int(planner_rows[0]["action_index"])
        reply_peak_total += float(planner_rows[0]["reply_peak_probability"])
        pressure_total += float(planner_rows[0]["pressure"])
        uncertainty_total += float(planner_rows[0]["uncertainty"])

        if teacher_example.teacher_top_k_action_indices:
            teacher_covered_examples += 1
            teacher_action_index = int(teacher_example.teacher_top_k_action_indices[0])
            root_top1_correct += int(predicted_action_index == teacher_action_index)
            teacher_rank = next(
                (
                    rank
                    for rank, row in enumerate(planner_rows, 1)
                    if int(row["action_index"]) == teacher_action_index
                ),
                None,
            )
            if teacher_rank is not None:
                reciprocal_rank_total += 1.0 / float(teacher_rank)
                teacher_probability = next(
                    probability
                    for probability, row in zip(planner_policy, planner_rows, strict=True)
                    if int(row["action_index"]) == teacher_action_index
                )
                teacher_root_probability_total += float(teacher_probability)

    total_examples = len(search_teacher_examples)
    elapsed = time.perf_counter() - started_at
    return PlannerBaselineMetrics(
        total_examples=total_examples,
        teacher_covered_examples=teacher_covered_examples,
        root_top1_accuracy=_ratio(root_top1_correct, teacher_covered_examples),
        teacher_root_mean_reciprocal_rank=_ratio(
            reciprocal_rank_total,
            teacher_covered_examples,
        ),
        teacher_root_mean_probability=_ratio(
            teacher_root_probability_total,
            teacher_covered_examples,
        ),
        mean_reply_peak_probability=_ratio(reply_peak_total, total_examples),
        mean_pressure=_ratio(pressure_total, total_examples),
        mean_uncertainty=_ratio(uncertainty_total, total_examples),
        examples_per_second=_ratio(total_examples, elapsed),
    )


def _evaluate_root_candidates(
    dataset_example: Any,
    *,
    teacher_example: Any,
    considered_indices: list[int],
    root_scores: list[float],
    proposer_model: Any,
    opponent_model: Any,
    opponent_mode: str,
    reply_peak_weight: float,
    pressure_weight: float,
    uncertainty_weight: float,
    repo_root: Path,
) -> list[dict[str, float | int]]:
    root_records: list[RawPositionRecord] = []
    for candidate_list_index in considered_indices:
        action_index = int(teacher_example.candidate_action_indices[candidate_list_index])
        root_records.append(
            RawPositionRecord(
                sample_id=f"{dataset_example.sample_id}:planner_root:{action_index}",
                fen=dataset_example.fen,
                source=dataset_example.source,
                selected_move_uci=move_uci_for_action(dataset_example, action_index),
            )
        )
    root_payloads = label_records_with_oracle(root_records, repo_root=repo_root)

    root_selected_examples = [
        dataset_example_from_oracle_payload(
            sample_id=dataset_example.sample_id,
            split=dataset_example.split,
            source=dataset_example.source,
            fen=dataset_example.fen,
            payload=payload,
        )
        for payload in root_payloads
    ]
    successor_records = [
        RawPositionRecord(
            sample_id=f"{dataset_example.sample_id}:planner_successor:{index}",
            fen=str(root_selected_example.next_fen),
            source=dataset_example.source,
        )
        for index, root_selected_example in enumerate(root_selected_examples)
    ]
    successor_payloads = label_records_with_oracle(successor_records, repo_root=repo_root)

    planner_rows: list[dict[str, float | int]] = []
    for candidate_list_index, root_selected_example, successor_payload in zip(
        considered_indices,
        root_selected_examples,
        successor_payloads,
        strict=True,
    ):
        successor_example = dataset_example_from_oracle_payload(
            sample_id=dataset_example.sample_id,
            split=dataset_example.split,
            source="planner_eval",
            fen=str(root_selected_example.next_fen),
            payload=successor_payload,
        )
        successor_symbolic = build_symbolic_proposer_example(
            successor_example,
            candidate_context_version=2,
            global_context_version=1,
        )
        transition_features = build_transition_context_features(root_selected_example, version=1)
        if opponent_mode == "none":
            reply_peak_probability = 0.0
            pressure = 0.0
            uncertainty = 0.0
        elif opponent_mode == "symbolic":
            reply_scores, reply_policy = score_symbolic_candidates(
                proposer_model,
                feature_vector=successor_symbolic.feature_vector,
                candidate_action_indices=successor_symbolic.candidate_action_indices,
                candidate_features=successor_symbolic.candidate_features,
                global_features=successor_symbolic.global_features,
                candidate_context_version=successor_symbolic.candidate_context_version,
            )
            if reply_scores:
                best_reply_index = max(
                    range(len(reply_scores)),
                    key=lambda index: reply_scores[index],
                )
                reply_peak_probability = max(reply_policy)
                pressure = _pressure_from_candidate_features(
                    successor_symbolic.candidate_features[best_reply_index]
                )
                uncertainty = 1.0 - reply_peak_probability
            else:
                reply_peak_probability = 0.0
                pressure = 0.0
                uncertainty = 0.0
        else:
            _reply_scores, reply_policy, pressure, uncertainty = score_opponent_candidates(
                opponent_model,
                root_feature_vector=pack_position_features(root_selected_example.position_encoding),
                next_feature_vector=successor_symbolic.feature_vector,
                chosen_action_index=int(
                    teacher_example.candidate_action_indices[candidate_list_index]
                ),
                transition_features=transition_features,
                reply_candidate_action_indices=successor_symbolic.candidate_action_indices,
                reply_candidate_features=successor_symbolic.candidate_features,
                reply_global_features=successor_symbolic.global_features,
            )
            reply_peak_probability = max(reply_policy) if reply_policy else 0.0

        root_score = float(root_scores[candidate_list_index])
        planner_score = (
            root_score
            - (reply_peak_weight * reply_peak_probability)
            - (pressure_weight * pressure)
            - (uncertainty_weight * uncertainty)
        )
        planner_rows.append(
            {
                "candidate_list_index": int(candidate_list_index),
                "action_index": int(teacher_example.candidate_action_indices[candidate_list_index]),
                "planner_score": float(planner_score),
                "reply_peak_probability": float(reply_peak_probability),
                "pressure": float(pressure),
                "uncertainty": float(uncertainty),
            }
        )
    return planner_rows


def _planner_policy(scores: list[float]) -> list[float]:
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    import math

    max_score = max(scores)
    exps = [math.exp(score - max_score) for score in scores]
    total = sum(exps)
    if total <= 0.0:
        return [0.0 for _ in scores]
    return [value / total for value in exps]


def _pressure_from_candidate_features(candidate_features: list[float]) -> float:
    if not candidate_features:
        return 0.0
    if bool(candidate_features[4]):
        return 1.0
    if bool(candidate_features[1]):
        return 0.75
    if bool(candidate_features[0]) or bool(candidate_features[3]):
        return 0.5
    return 0.0


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
