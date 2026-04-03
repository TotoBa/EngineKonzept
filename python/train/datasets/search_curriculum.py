"""Offline curriculum buckets derived from search traces and disagreement artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from train.datasets.schema import SUPPORTED_SPLITS
from train.datasets.search_disagreements import SearchDisagreementExample
from train.datasets.search_traces import SearchTraceExample


SEARCH_CURRICULUM_ARTIFACT_PREFIX = "search_curriculum_"
_CAPTURE_FEATURE_INDEX = 0
_PROMOTION_FEATURE_INDEX = 1
_GIVES_CHECK_FEATURE_INDEX = 4


@dataclass(frozen=True)
class SearchCurriculumExample:
    """Curriculum bucket assignment over one root exact-candidate position."""

    sample_id: str
    split: str
    fen: str
    teacher_top1_action_index: int
    best_reply_action_index: int | None
    pv_length: int
    bucket_labels: list[str]
    curriculum_priority: float
    teacher_top1_minus_top2_cp: float | None
    proposer_rank_of_teacher_top1: int
    teacher_rank_of_proposer_top1: int
    teacher_top1_advantage_cp: float
    policy_l1_distance: float
    top1_disagrees: bool

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "fen": self.fen,
            "teacher_top1_action_index": self.teacher_top1_action_index,
            "best_reply_action_index": self.best_reply_action_index,
            "pv_length": self.pv_length,
            "bucket_labels": self.bucket_labels,
            "curriculum_priority": self.curriculum_priority,
            "teacher_top1_minus_top2_cp": self.teacher_top1_minus_top2_cp,
            "proposer_rank_of_teacher_top1": self.proposer_rank_of_teacher_top1,
            "teacher_rank_of_proposer_top1": self.teacher_rank_of_proposer_top1,
            "teacher_top1_advantage_cp": self.teacher_top1_advantage_cp,
            "policy_l1_distance": self.policy_l1_distance,
            "top1_disagrees": self.top1_disagrees,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SearchCurriculumExample":
        """Construct the curriculum example from JSON."""
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            fen=str(payload["fen"]),
            teacher_top1_action_index=int(payload["teacher_top1_action_index"]),
            best_reply_action_index=_optional_int(payload.get("best_reply_action_index")),
            pv_length=int(payload["pv_length"]),
            bucket_labels=[str(value) for value in list(payload["bucket_labels"])],
            curriculum_priority=float(payload["curriculum_priority"]),
            teacher_top1_minus_top2_cp=_optional_float(payload.get("teacher_top1_minus_top2_cp")),
            proposer_rank_of_teacher_top1=int(payload["proposer_rank_of_teacher_top1"]),
            teacher_rank_of_proposer_top1=int(payload["teacher_rank_of_proposer_top1"]),
            teacher_top1_advantage_cp=float(payload["teacher_top1_advantage_cp"]),
            policy_l1_distance=float(payload["policy_l1_distance"]),
            top1_disagrees=bool(payload["top1_disagrees"]),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "SearchCurriculumExample":
        """Parse one JSONL row."""
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: search curriculum example must be a JSON object")
        return cls.from_dict(payload)


def search_curriculum_artifact_name(split: str) -> str:
    """Return the canonical curriculum artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{SEARCH_CURRICULUM_ARTIFACT_PREFIX}{split}.jsonl"


def load_search_curriculum_examples(path: Path) -> list[SearchCurriculumExample]:
    """Load curriculum examples from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"search curriculum artifact not found: {path}")
    examples: list[SearchCurriculumExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(SearchCurriculumExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def write_search_curriculum_artifact(
    path: Path,
    examples: Sequence[SearchCurriculumExample],
) -> None:
    """Write curriculum examples as JSONL."""
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_search_curriculum_examples(
    trace_examples: Sequence[SearchTraceExample],
    disagreement_examples: Sequence[SearchDisagreementExample],
    *,
    forced_gap_cp: float = 80.0,
    unstable_gap_cp: float = 20.0,
    large_rank_threshold: int = 4,
    disagreement_advantage_cp: float = 80.0,
    high_policy_l1: float = 0.75,
) -> list[SearchCurriculumExample]:
    """Derive curriculum buckets from trace and disagreement artifacts."""
    if forced_gap_cp <= 0.0:
        raise ValueError("forced_gap_cp must be positive")
    if unstable_gap_cp < 0.0:
        raise ValueError("unstable_gap_cp must be non-negative")
    if large_rank_threshold <= 1:
        raise ValueError("large_rank_threshold must be greater than 1")
    if disagreement_advantage_cp < 0.0:
        raise ValueError("disagreement_advantage_cp must be non-negative")
    if high_policy_l1 < 0.0:
        raise ValueError("high_policy_l1 must be non-negative")

    disagreement_by_sample_id = {
        example.sample_id: example for example in disagreement_examples
    }
    built: list[SearchCurriculumExample] = []
    for trace_example in trace_examples:
        disagreement_example = disagreement_by_sample_id.get(trace_example.sample_id)
        if disagreement_example is None:
            raise ValueError(
                f"{trace_example.sample_id}: disagreement artifact missing for curriculum build"
            )
        if disagreement_example.fen != trace_example.fen:
            raise ValueError(
                f"{trace_example.sample_id}: disagreement and trace FENs do not match"
            )

        teacher_top1_action_index = trace_example.teacher_top_k_action_indices[0]
        root_feature_row = _feature_row_for_action(
            trace_example.candidate_action_indices,
            trace_example.candidate_features,
            teacher_top1_action_index,
        )
        bucket_labels = _bucket_labels(
            trace_example,
            disagreement_example,
            root_feature_row=root_feature_row,
            forced_gap_cp=forced_gap_cp,
            unstable_gap_cp=unstable_gap_cp,
            large_rank_threshold=large_rank_threshold,
            disagreement_advantage_cp=disagreement_advantage_cp,
            high_policy_l1=high_policy_l1,
        )
        built.append(
            SearchCurriculumExample(
                sample_id=trace_example.sample_id,
                split=trace_example.split,
                fen=trace_example.fen,
                teacher_top1_action_index=teacher_top1_action_index,
                best_reply_action_index=trace_example.best_reply_action_index,
                pv_length=trace_example.pv_length,
                bucket_labels=bucket_labels,
                curriculum_priority=_priority(
                    disagreement_example,
                    teacher_gap_cp=trace_example.top1_minus_top2_cp,
                ),
                teacher_top1_minus_top2_cp=trace_example.top1_minus_top2_cp,
                proposer_rank_of_teacher_top1=disagreement_example.proposer_rank_of_teacher_top1,
                teacher_rank_of_proposer_top1=disagreement_example.teacher_rank_of_proposer_top1,
                teacher_top1_advantage_cp=disagreement_example.teacher_top1_advantage_cp,
                policy_l1_distance=disagreement_example.policy_l1_distance,
                top1_disagrees=disagreement_example.top1_disagrees,
            )
        )

    return built


def _bucket_labels(
    trace_example: SearchTraceExample,
    disagreement_example: SearchDisagreementExample,
    *,
    root_feature_row: Sequence[float],
    forced_gap_cp: float,
    unstable_gap_cp: float,
    large_rank_threshold: int,
    disagreement_advantage_cp: float,
    high_policy_l1: float,
) -> list[str]:
    labels: list[str] = []
    teacher_gap_cp = trace_example.top1_minus_top2_cp

    if teacher_gap_cp is not None and teacher_gap_cp >= forced_gap_cp:
        labels.append("forced_teacher")
    if teacher_gap_cp is not None and teacher_gap_cp <= unstable_gap_cp:
        labels.append("unstable_teacher")
    if trace_example.best_reply_action_index is not None:
        labels.append("reply_supervised")
    if disagreement_example.top1_disagrees:
        labels.append("top1_disagreement")
    if (
        disagreement_example.proposer_rank_of_teacher_top1 >= large_rank_threshold
        or disagreement_example.teacher_rank_of_proposer_top1 >= large_rank_threshold
    ):
        labels.append("large_rank_mismatch")
    if disagreement_example.teacher_top1_advantage_cp >= disagreement_advantage_cp:
        labels.append("teacher_punishes_proposer")
    if disagreement_example.policy_l1_distance >= high_policy_l1:
        labels.append("policy_shape_mismatch")
    if bool(root_feature_row[_CAPTURE_FEATURE_INDEX]):
        labels.append("capture_line")
    if bool(root_feature_row[_PROMOTION_FEATURE_INDEX]):
        labels.append("promotion_line")
    if bool(root_feature_row[_GIVES_CHECK_FEATURE_INDEX]):
        labels.append("checking_line")
    if not labels:
        labels.append("stable_agreement")
    return labels


def _feature_row_for_action(
    candidate_action_indices: Sequence[int],
    candidate_features: Sequence[Sequence[float]],
    action_index: int,
) -> Sequence[float]:
    for index, candidate_action_index in enumerate(candidate_action_indices):
        if candidate_action_index == action_index:
            return candidate_features[index]
    raise ValueError(f"candidate feature row missing for action {action_index}")


def _priority(
    disagreement_example: SearchDisagreementExample,
    *,
    teacher_gap_cp: float | None,
) -> float:
    return round(
        (1.0 if disagreement_example.top1_disagrees else 0.0)
        + min(float(disagreement_example.proposer_rank_of_teacher_top1), 16.0) / 8.0
        + min(float(disagreement_example.teacher_rank_of_proposer_top1), 16.0) / 8.0
        + max(disagreement_example.teacher_top1_advantage_cp, 0.0) / 100.0
        + disagreement_example.policy_l1_distance
        + (0.5 if teacher_gap_cp is not None and teacher_gap_cp >= 80.0 else 0.0),
        6,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
