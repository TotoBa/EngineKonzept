"""Offline disagreement artifacts between symbolic proposer ranking and search teachers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from train.datasets.artifacts import (
    DEFAULT_CANDIDATE_CONTEXT_VERSION,
    DEFAULT_GLOBAL_CONTEXT_VERSION,
    build_symbolic_proposer_example,
)
from train.eval.symbolic_proposer import (
    load_symbolic_proposer_checkpoint,
    score_symbolic_candidates,
)
from train.datasets.schema import DatasetExample, SUPPORTED_SPLITS
from train.datasets.search_teacher import SearchTeacherExample
from train.models.proposer import torch_is_available

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None


SEARCH_DISAGREEMENTS_ARTIFACT_PREFIX = "search_disagreements_"


@dataclass(frozen=True)
class SearchDisagreementExample:
    """Offline disagreement record over exact legal candidates."""

    sample_id: str
    split: str
    fen: str
    feature_vector: list[float]
    candidate_context_version: int
    global_context_version: int
    global_features: list[float]
    candidate_action_indices: list[int]
    candidate_features: list[list[float]]
    teacher_engine: str
    teacher_nodes: int | None
    teacher_depth: int | None
    teacher_movetime_ms: int | None
    teacher_multipv: int
    teacher_coverage_ratio: float
    teacher_root_value_cp: float
    teacher_root_value_mate: int | None
    teacher_candidate_scores_cp: list[float]
    teacher_policy: list[float]
    teacher_top_k_action_indices: list[int]
    proposer_checkpoint: str
    proposer_candidate_scores: list[float]
    proposer_policy: list[float]
    proposer_top_k_action_indices: list[int]
    teacher_top1_action_index: int
    proposer_top1_action_index: int
    teacher_rank_of_proposer_top1: int
    proposer_rank_of_teacher_top1: int
    top1_disagrees: bool
    teacher_top1_minus_top2_cp: float | None
    proposer_top1_minus_top2_logit: float | None
    teacher_top1_advantage_cp: float
    policy_l1_distance: float

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "fen": self.fen,
            "feature_vector": self.feature_vector,
            "candidate_context_version": self.candidate_context_version,
            "global_context_version": self.global_context_version,
            "global_features": self.global_features,
            "candidate_action_indices": self.candidate_action_indices,
            "candidate_features": self.candidate_features,
            "teacher_engine": self.teacher_engine,
            "teacher_nodes": self.teacher_nodes,
            "teacher_depth": self.teacher_depth,
            "teacher_movetime_ms": self.teacher_movetime_ms,
            "teacher_multipv": self.teacher_multipv,
            "teacher_coverage_ratio": self.teacher_coverage_ratio,
            "teacher_root_value_cp": self.teacher_root_value_cp,
            "teacher_root_value_mate": self.teacher_root_value_mate,
            "teacher_candidate_scores_cp": self.teacher_candidate_scores_cp,
            "teacher_policy": self.teacher_policy,
            "teacher_top_k_action_indices": self.teacher_top_k_action_indices,
            "proposer_checkpoint": self.proposer_checkpoint,
            "proposer_candidate_scores": self.proposer_candidate_scores,
            "proposer_policy": self.proposer_policy,
            "proposer_top_k_action_indices": self.proposer_top_k_action_indices,
            "teacher_top1_action_index": self.teacher_top1_action_index,
            "proposer_top1_action_index": self.proposer_top1_action_index,
            "teacher_rank_of_proposer_top1": self.teacher_rank_of_proposer_top1,
            "proposer_rank_of_teacher_top1": self.proposer_rank_of_teacher_top1,
            "top1_disagrees": self.top1_disagrees,
            "teacher_top1_minus_top2_cp": self.teacher_top1_minus_top2_cp,
            "proposer_top1_minus_top2_logit": self.proposer_top1_minus_top2_logit,
            "teacher_top1_advantage_cp": self.teacher_top1_advantage_cp,
            "policy_l1_distance": self.policy_l1_distance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SearchDisagreementExample":
        """Construct the disagreement example from JSON."""
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            fen=str(payload["fen"]),
            feature_vector=[float(value) for value in list(payload["feature_vector"])],
            candidate_context_version=int(payload["candidate_context_version"]),
            global_context_version=int(payload["global_context_version"]),
            global_features=[float(value) for value in list(payload["global_features"])],
            candidate_action_indices=[
                int(value) for value in list(payload["candidate_action_indices"])
            ],
            candidate_features=[
                [float(value) for value in row] for row in list(payload["candidate_features"])
            ],
            teacher_engine=str(payload["teacher_engine"]),
            teacher_nodes=_optional_int(payload.get("teacher_nodes")),
            teacher_depth=_optional_int(payload.get("teacher_depth")),
            teacher_movetime_ms=_optional_int(payload.get("teacher_movetime_ms")),
            teacher_multipv=int(payload["teacher_multipv"]),
            teacher_coverage_ratio=float(payload["teacher_coverage_ratio"]),
            teacher_root_value_cp=float(payload["teacher_root_value_cp"]),
            teacher_root_value_mate=_optional_int(payload.get("teacher_root_value_mate")),
            teacher_candidate_scores_cp=[
                float(value) for value in list(payload["teacher_candidate_scores_cp"])
            ],
            teacher_policy=[float(value) for value in list(payload["teacher_policy"])],
            teacher_top_k_action_indices=[
                int(value) for value in list(payload["teacher_top_k_action_indices"])
            ],
            proposer_checkpoint=str(payload["proposer_checkpoint"]),
            proposer_candidate_scores=[
                float(value) for value in list(payload["proposer_candidate_scores"])
            ],
            proposer_policy=[float(value) for value in list(payload["proposer_policy"])],
            proposer_top_k_action_indices=[
                int(value) for value in list(payload["proposer_top_k_action_indices"])
            ],
            teacher_top1_action_index=int(payload["teacher_top1_action_index"]),
            proposer_top1_action_index=int(payload["proposer_top1_action_index"]),
            teacher_rank_of_proposer_top1=int(payload["teacher_rank_of_proposer_top1"]),
            proposer_rank_of_teacher_top1=int(payload["proposer_rank_of_teacher_top1"]),
            top1_disagrees=bool(payload["top1_disagrees"]),
            teacher_top1_minus_top2_cp=_optional_float(payload.get("teacher_top1_minus_top2_cp")),
            proposer_top1_minus_top2_logit=_optional_float(
                payload.get("proposer_top1_minus_top2_logit")
            ),
            teacher_top1_advantage_cp=float(payload["teacher_top1_advantage_cp"]),
            policy_l1_distance=float(payload["policy_l1_distance"]),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "SearchDisagreementExample":
        """Parse the example from one JSON line."""
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: search disagreement example must be a JSON object")
        return cls.from_dict(payload)


def search_disagreements_artifact_name(split: str) -> str:
    """Return the canonical disagreement artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{SEARCH_DISAGREEMENTS_ARTIFACT_PREFIX}{split}.jsonl"


def load_search_disagreement_examples(path: Path) -> list[SearchDisagreementExample]:
    """Load search-disagreement examples from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"search disagreement artifact not found: {path}")

    examples: list[SearchDisagreementExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(SearchDisagreementExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def write_search_disagreement_artifact(
    path: Path,
    examples: Sequence[SearchDisagreementExample],
) -> None:
    """Write search-disagreement examples as JSONL."""
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_search_disagreement_examples(
    dataset_examples: Sequence[DatasetExample],
    teacher_examples: Sequence[SearchTeacherExample],
    *,
    checkpoint_path: Path,
    top_k: int = 8,
    max_examples: int | None = None,
) -> list[SearchDisagreementExample]:
    """Materialize proposer-vs-teacher disagreement examples."""
    if torch is None or not torch_is_available():  # pragma: no cover - torch absent
        raise RuntimeError(
            "PyTorch is required for search disagreement workflows. Install the 'train' extra or torch."
        )
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    selected_teacher_examples = list(
        teacher_examples[:max_examples] if max_examples is not None else teacher_examples
    )
    if not selected_teacher_examples:
        return []

    model, _config = load_symbolic_proposer_checkpoint(checkpoint_path)

    dataset_by_sample_id = {example.sample_id: example for example in dataset_examples}
    built: list[SearchDisagreementExample] = []
    with torch.inference_mode():
        for teacher_example in selected_teacher_examples:
            dataset_example = dataset_by_sample_id.get(teacher_example.sample_id)
            if dataset_example is None:
                raise ValueError(
                    f"dataset example missing for teacher sample {teacher_example.sample_id}"
                )
            if dataset_example.fen != teacher_example.fen:
                raise ValueError(
                    f"{teacher_example.sample_id}: teacher artifact FEN does not match dataset split"
                )

            symbolic_example = build_symbolic_proposer_example(
                dataset_example,
                candidate_context_version=DEFAULT_CANDIDATE_CONTEXT_VERSION,
                global_context_version=DEFAULT_GLOBAL_CONTEXT_VERSION,
            )
            proposer_scores, proposer_policy = score_symbolic_candidates(
                model,
                feature_vector=symbolic_example.feature_vector,
                candidate_action_indices=symbolic_example.candidate_action_indices,
                candidate_features=symbolic_example.candidate_features,
                global_features=symbolic_example.global_features,
                candidate_context_version=symbolic_example.candidate_context_version,
            )
            if len(proposer_scores) != len(symbolic_example.candidate_action_indices):
                raise AssertionError("proposer candidate score count must match candidate count")

            teacher_scores, teacher_policy = _align_teacher_outputs(
                teacher_example,
                candidate_action_indices=symbolic_example.candidate_action_indices,
            )
            teacher_ranking = _rank_actions(
                symbolic_example.candidate_action_indices,
                teacher_scores,
            )
            proposer_ranking = _rank_actions(
                symbolic_example.candidate_action_indices,
                proposer_scores,
            )
            teacher_rank_by_action = {
                action_index: rank for rank, action_index in enumerate(teacher_ranking, start=1)
            }
            proposer_rank_by_action = {
                action_index: rank for rank, action_index in enumerate(proposer_ranking, start=1)
            }
            teacher_top1_action_index = teacher_ranking[0]
            proposer_top1_action_index = proposer_ranking[0]
            teacher_score_by_action = {
                action_index: score
                for action_index, score in zip(
                    symbolic_example.candidate_action_indices,
                    teacher_scores,
                    strict=True,
                )
            }

            built.append(
                SearchDisagreementExample(
                    sample_id=symbolic_example.sample_id,
                    split=symbolic_example.split,
                    fen=dataset_example.fen,
                    feature_vector=list(symbolic_example.feature_vector),
                    candidate_context_version=symbolic_example.candidate_context_version,
                    global_context_version=symbolic_example.global_context_version,
                    global_features=list(symbolic_example.global_features),
                    candidate_action_indices=list(symbolic_example.candidate_action_indices),
                    candidate_features=[list(row) for row in symbolic_example.candidate_features],
                    teacher_engine=teacher_example.teacher_engine,
                    teacher_nodes=teacher_example.teacher_nodes,
                    teacher_depth=teacher_example.teacher_depth,
                    teacher_movetime_ms=teacher_example.teacher_movetime_ms,
                    teacher_multipv=teacher_example.teacher_multipv,
                    teacher_coverage_ratio=teacher_example.teacher_coverage_ratio,
                    teacher_root_value_cp=teacher_example.teacher_root_value_cp,
                    teacher_root_value_mate=teacher_example.teacher_root_value_mate,
                    teacher_candidate_scores_cp=teacher_scores,
                    teacher_policy=teacher_policy,
                    teacher_top_k_action_indices=list(teacher_example.teacher_top_k_action_indices),
                    proposer_checkpoint=str(checkpoint_path),
                    proposer_candidate_scores=proposer_scores,
                    proposer_policy=proposer_policy,
                    proposer_top_k_action_indices=proposer_ranking[: min(top_k, len(proposer_ranking))],
                    teacher_top1_action_index=teacher_top1_action_index,
                    proposer_top1_action_index=proposer_top1_action_index,
                    teacher_rank_of_proposer_top1=teacher_rank_by_action[proposer_top1_action_index],
                    proposer_rank_of_teacher_top1=proposer_rank_by_action[teacher_top1_action_index],
                    top1_disagrees=teacher_top1_action_index != proposer_top1_action_index,
                    teacher_top1_minus_top2_cp=_top1_minus_top2(teacher_scores),
                    proposer_top1_minus_top2_logit=_top1_minus_top2(proposer_scores),
                    teacher_top1_advantage_cp=(
                        teacher_score_by_action[teacher_top1_action_index]
                        - teacher_score_by_action[proposer_top1_action_index]
                    ),
                    policy_l1_distance=sum(
                        abs(teacher_value - proposer_value)
                        for teacher_value, proposer_value in zip(
                            teacher_policy,
                            proposer_policy,
                            strict=True,
                        )
                    ),
                )
            )

    return built


def _align_teacher_outputs(
    teacher_example: SearchTeacherExample,
    *,
    candidate_action_indices: Sequence[int],
) -> tuple[list[float], list[float]]:
    teacher_score_by_action = {
        action_index: score
        for action_index, score in zip(
            teacher_example.candidate_action_indices,
            teacher_example.teacher_candidate_scores_cp,
            strict=True,
        )
    }
    teacher_policy_by_action = {
        action_index: value
        for action_index, value in zip(
            teacher_example.candidate_action_indices,
            teacher_example.teacher_policy,
            strict=True,
        )
    }
    missing = [action for action in candidate_action_indices if action not in teacher_score_by_action]
    if missing:
        raise ValueError(
            f"{teacher_example.sample_id}: teacher artifact is missing {len(missing)} candidate actions"
        )
    return (
        [teacher_score_by_action[action_index] for action_index in candidate_action_indices],
        [teacher_policy_by_action[action_index] for action_index in candidate_action_indices],
    )


def _rank_actions(action_indices: Sequence[int], scores: Sequence[float]) -> list[int]:
    ranked_positions = sorted(range(len(action_indices)), key=lambda index: (-scores[index], index))
    return [int(action_indices[index]) for index in ranked_positions]


def _top1_minus_top2(scores: Sequence[float]) -> float | None:
    if len(scores) < 2:
        return None
    ranked = sorted((float(score) for score in scores), reverse=True)
    return ranked[0] - ranked[1]


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)
