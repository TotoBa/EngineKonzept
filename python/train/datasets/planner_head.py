"""PlannerHeadV1 artifacts for a trainable bounded root scorer."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Sequence

from train.datasets import (
    build_symbolic_proposer_example,
    build_selected_move_action_features,
    build_transition_context_features,
    dataset_example_from_oracle_payload,
    load_search_curriculum_examples,
    load_search_teacher_examples,
    load_split_examples,
    move_uci_for_action,
    pack_position_features,
)
from train.datasets.contracts import project_candidate_context_to_v1
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import DatasetExample, RawPositionRecord, SUPPORTED_SPLITS
from train.eval.opponent import (
    load_opponent_head_checkpoint,
    score_opponent_candidates,
)
from train.eval.dynamics import (
    load_dynamics_checkpoint,
    predict_dynamics_latent,
)
from train.eval.symbolic_proposer import (
    load_symbolic_proposer_checkpoint,
    score_symbolic_candidates,
)


PLANNER_HEAD_ARTIFACT_PREFIX = "planner_head_"
PLANNER_LATENT_STATE_VERSION = 1
PLANNER_SCORE_TARGET_CLIP_CP = 256.0


@dataclass(frozen=True)
class PlannerHeadExample:
    """One bounded root-planning training example over exact root candidates."""

    sample_id: str
    split: str
    fen: str
    feature_vector: list[float]
    candidate_context_version: int
    global_context_version: int
    global_features: list[float]
    candidate_action_indices: list[int]
    candidate_features: list[list[float]]
    proposer_scores: list[float]
    transition_context_version: int
    transition_features: list[list[float]]
    reply_peak_probabilities: list[float]
    pressures: list[float]
    uncertainties: list[float]
    curriculum_bucket_labels: list[str]
    curriculum_priority: float
    teacher_top1_action_index: int
    teacher_top1_candidate_index: int
    teacher_policy: list[float]
    teacher_root_value_cp: float
    teacher_top1_minus_top2_cp: float | None
    teacher_candidate_scores_cp: list[float] | None = None
    teacher_candidate_score_delta_targets_cp: list[float] | None = None
    latent_state_version: int | None = None
    latent_features: list[list[float]] | None = None

    def to_dict(self) -> dict[str, object]:
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
            "proposer_scores": self.proposer_scores,
            "transition_context_version": self.transition_context_version,
            "transition_features": self.transition_features,
            "latent_state_version": self.latent_state_version,
            "latent_features": self.latent_features,
            "reply_peak_probabilities": self.reply_peak_probabilities,
            "pressures": self.pressures,
            "uncertainties": self.uncertainties,
            "curriculum_bucket_labels": self.curriculum_bucket_labels,
            "curriculum_priority": self.curriculum_priority,
            "teacher_top1_action_index": self.teacher_top1_action_index,
            "teacher_top1_candidate_index": self.teacher_top1_candidate_index,
            "teacher_policy": self.teacher_policy,
            "teacher_candidate_scores_cp": self.teacher_candidate_scores_cp,
            "teacher_candidate_score_delta_targets_cp": self.teacher_candidate_score_delta_targets_cp,
            "teacher_root_value_cp": self.teacher_root_value_cp,
            "teacher_top1_minus_top2_cp": self.teacher_top1_minus_top2_cp,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlannerHeadExample":
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        candidate_action_indices = [int(value) for value in list(payload["candidate_action_indices"])]
        candidate_features = [
            [float(value) for value in row] for row in list(payload["candidate_features"])
        ]
        proposer_scores = [float(value) for value in list(payload["proposer_scores"])]
        transition_features = [
            [float(value) for value in row] for row in list(payload["transition_features"])
        ]
        latent_state_version = _optional_int(payload.get("latent_state_version"))
        latent_features_payload = payload.get("latent_features")
        latent_features = (
            [[float(value) for value in row] for row in list(latent_features_payload)]
            if latent_features_payload is not None
            else None
        )
        reply_peak_probabilities = [
            float(value) for value in list(payload["reply_peak_probabilities"])
        ]
        pressures = [float(value) for value in list(payload["pressures"])]
        uncertainties = [float(value) for value in list(payload["uncertainties"])]
        teacher_policy = [float(value) for value in list(payload["teacher_policy"])]
        teacher_candidate_scores_cp = _optional_float_list(payload.get("teacher_candidate_scores_cp"))
        teacher_candidate_score_delta_targets_cp = _optional_float_list(
            payload.get("teacher_candidate_score_delta_targets_cp")
        )
        expected_length = len(candidate_action_indices)
        for name, values in (
            ("candidate_features", candidate_features),
            ("proposer_scores", proposer_scores),
            ("transition_features", transition_features),
            ("reply_peak_probabilities", reply_peak_probabilities),
            ("pressures", pressures),
            ("uncertainties", uncertainties),
            ("teacher_policy", teacher_policy),
        ):
            if len(values) != expected_length:
                raise ValueError(
                    f"{name} must have the same length as candidate_action_indices"
                )
        if teacher_candidate_scores_cp is not None and len(teacher_candidate_scores_cp) != expected_length:
            raise ValueError(
                "teacher_candidate_scores_cp must have the same length as candidate_action_indices"
            )
        if (
            teacher_candidate_score_delta_targets_cp is not None
            and len(teacher_candidate_score_delta_targets_cp) != expected_length
        ):
            raise ValueError(
                "teacher_candidate_score_delta_targets_cp must have the same length as candidate_action_indices"
            )
        if latent_features is not None and len(latent_features) != expected_length:
            raise ValueError(
                "latent_features must have the same length as candidate_action_indices"
            )
        teacher_top1_candidate_index = int(payload["teacher_top1_candidate_index"])
        if not 0 <= teacher_top1_candidate_index < expected_length:
            raise ValueError("teacher_top1_candidate_index out of range")
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            fen=str(payload["fen"]),
            feature_vector=[float(value) for value in list(payload["feature_vector"])],
            candidate_context_version=int(payload["candidate_context_version"]),
            global_context_version=int(payload["global_context_version"]),
            global_features=[float(value) for value in list(payload["global_features"])],
            candidate_action_indices=candidate_action_indices,
            candidate_features=candidate_features,
            proposer_scores=proposer_scores,
            transition_context_version=int(payload["transition_context_version"]),
            transition_features=transition_features,
            latent_state_version=latent_state_version,
            latent_features=latent_features,
            reply_peak_probabilities=reply_peak_probabilities,
            pressures=pressures,
            uncertainties=uncertainties,
            curriculum_bucket_labels=[
                str(value) for value in list(payload["curriculum_bucket_labels"])
            ],
            curriculum_priority=float(payload["curriculum_priority"]),
            teacher_top1_action_index=int(payload["teacher_top1_action_index"]),
            teacher_top1_candidate_index=teacher_top1_candidate_index,
            teacher_policy=teacher_policy,
            teacher_candidate_scores_cp=teacher_candidate_scores_cp,
            teacher_candidate_score_delta_targets_cp=teacher_candidate_score_delta_targets_cp,
            teacher_root_value_cp=float(payload["teacher_root_value_cp"]),
            teacher_top1_minus_top2_cp=_optional_float(payload.get("teacher_top1_minus_top2_cp")),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "PlannerHeadExample":
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: planner head example must be a JSON object")
        return cls.from_dict(payload)


def planner_head_artifact_name(split: str) -> str:
    """Return the canonical planner-head artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{PLANNER_HEAD_ARTIFACT_PREFIX}{split}.jsonl"


def load_planner_head_examples(path: Path) -> list[PlannerHeadExample]:
    """Load planner-head examples from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"planner head artifact not found: {path}")
    examples: list[PlannerHeadExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(PlannerHeadExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def write_planner_head_artifact(path: Path, examples: Sequence[PlannerHeadExample]) -> None:
    """Write planner-head examples as JSONL."""
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_planner_head_examples(
    *,
    dataset_dir: Path,
    split: str,
    search_teacher_path: Path,
    search_curriculum_path: Path | None,
    proposer_checkpoint: Path,
    dynamics_checkpoint: Path | None,
    opponent_mode: str,
    opponent_checkpoint: Path | None,
    root_top_k: int,
    max_examples: int | None,
    repo_root: Path,
) -> list[PlannerHeadExample]:
    """Build bounded root-planning examples from the current workflow artifacts."""

    if root_top_k <= 0:
        raise ValueError("root_top_k must be positive")
    if opponent_mode not in {"none", "symbolic", "learned"}:
        raise ValueError("opponent_mode must be 'none', 'symbolic', or 'learned'")
    if opponent_mode == "learned" and opponent_checkpoint is None:
        raise ValueError("opponent_checkpoint is required when opponent_mode='learned'")

    dataset_examples = load_split_examples(dataset_dir, split)
    if not dataset_examples:
        raise ValueError(f"dataset split is empty: {split}")
    dataset_by_sample_id = {example.sample_id: example for example in dataset_examples}
    teacher_examples = load_search_teacher_examples(search_teacher_path)
    curriculum_by_sample_id: dict[str, Any] = {}
    if search_curriculum_path is not None and search_curriculum_path.exists():
        curriculum_by_sample_id = {
            example.sample_id: example
            for example in load_search_curriculum_examples(search_curriculum_path)
        }

    proposer_model, _ = load_symbolic_proposer_checkpoint(proposer_checkpoint)
    dynamics_model = None
    if dynamics_checkpoint is not None:
        dynamics_model, _ = load_dynamics_checkpoint(dynamics_checkpoint)
    opponent_model = None
    if opponent_mode == "learned":
        opponent_model, _ = load_opponent_head_checkpoint(opponent_checkpoint)

    built: list[PlannerHeadExample] = []
    selected_teacher_examples = (
        teacher_examples[:max_examples] if max_examples is not None else teacher_examples
    )
    for teacher_example in selected_teacher_examples:
        dataset_example = dataset_by_sample_id.get(teacher_example.sample_id)
        if dataset_example is None:
            raise ValueError(
                f"{teacher_example.sample_id}: missing dataset example for planner-head build"
            )
        curriculum_example = curriculum_by_sample_id.get(teacher_example.sample_id)
        root_scores, _ = score_symbolic_candidates(
            proposer_model,
            feature_vector=teacher_example.feature_vector,
            candidate_action_indices=teacher_example.candidate_action_indices,
            candidate_features=teacher_example.candidate_features,
            global_features=teacher_example.global_features,
            candidate_context_version=teacher_example.candidate_context_version,
        )
        teacher_top1_action_index = int(teacher_example.teacher_top_k_action_indices[0])
        teacher_top1_root_index = teacher_example.candidate_action_indices.index(
            teacher_top1_action_index
        )
        considered_indices = _select_root_candidate_indices(
            root_scores,
            teacher_top1_root_index=teacher_top1_root_index,
            root_top_k=root_top_k,
        )
        candidate_rows = _build_root_candidate_rows(
            dataset_example,
            teacher_example=teacher_example,
            considered_indices=considered_indices,
            root_scores=root_scores,
            opponent_mode=opponent_mode,
            proposer_model=proposer_model,
            dynamics_model=dynamics_model,
            opponent_model=opponent_model,
            repo_root=repo_root,
        )
        candidate_action_indices = [row["action_index"] for row in candidate_rows]
        teacher_top1_candidate_index = candidate_action_indices.index(teacher_top1_action_index)
        teacher_policy = _restricted_teacher_policy(
            teacher_example.teacher_policy,
            considered_indices=considered_indices,
            teacher_top1_candidate_index=teacher_top1_candidate_index,
        )
        teacher_candidate_scores_cp = _restricted_teacher_candidate_scores(
            teacher_example.teacher_candidate_scores_cp,
            considered_indices=considered_indices,
        )
        teacher_candidate_score_delta_targets_cp = build_teacher_candidate_score_delta_targets_cp(
            teacher_example.teacher_candidate_scores_cp,
            considered_indices=considered_indices,
            teacher_root_value_cp=float(teacher_example.teacher_root_value_cp),
        )
        built.append(
            PlannerHeadExample(
                sample_id=teacher_example.sample_id,
                split=teacher_example.split,
                fen=teacher_example.fen,
                feature_vector=list(teacher_example.feature_vector),
                candidate_context_version=teacher_example.candidate_context_version,
                global_context_version=teacher_example.global_context_version,
                global_features=list(teacher_example.global_features),
                candidate_action_indices=candidate_action_indices,
                candidate_features=[
                    list(teacher_example.candidate_features[index])
                    for index in considered_indices
                ],
                proposer_scores=[row["proposer_score"] for row in candidate_rows],
                transition_context_version=1,
                transition_features=[row["transition_features"] for row in candidate_rows],
                latent_state_version=(
                    PLANNER_LATENT_STATE_VERSION if dynamics_model is not None else None
                ),
                latent_features=(
                    [row["latent_features"] for row in candidate_rows]
                    if dynamics_model is not None
                    else None
                ),
                reply_peak_probabilities=[
                    row["reply_peak_probability"] for row in candidate_rows
                ],
                pressures=[row["pressure"] for row in candidate_rows],
                uncertainties=[row["uncertainty"] for row in candidate_rows],
                curriculum_bucket_labels=(
                    list(curriculum_example.bucket_labels)
                    if curriculum_example is not None
                    else []
                ),
                curriculum_priority=(
                    float(curriculum_example.curriculum_priority)
                    if curriculum_example is not None
                    else 0.0
                ),
                teacher_top1_action_index=teacher_top1_action_index,
                teacher_top1_candidate_index=teacher_top1_candidate_index,
                teacher_policy=teacher_policy,
                teacher_candidate_scores_cp=teacher_candidate_scores_cp,
                teacher_candidate_score_delta_targets_cp=teacher_candidate_score_delta_targets_cp,
                teacher_root_value_cp=float(teacher_example.teacher_root_value_cp),
                teacher_top1_minus_top2_cp=_optional_float(
                    getattr(curriculum_example, "teacher_top1_minus_top2_cp", None)
                ),
            )
        )
    return built


def materialize_planner_latent_features(
    examples: Sequence[PlannerHeadExample],
    *,
    dynamics_model: Any,
    latent_state_version: int = PLANNER_LATENT_STATE_VERSION,
    predictor: Any = None,
) -> list[PlannerHeadExample]:
    """Attach latent successor features to existing planner-head artifacts."""
    if predictor is None:
        predictor = predict_dynamics_latent
    materialized: list[PlannerHeadExample] = []
    for example in examples:
        latent_rows: list[list[float]] = []
        for action_index, candidate_row, transition_row in zip(
            example.candidate_action_indices,
            example.candidate_features,
            example.transition_features,
            strict=True,
        ):
            action_features = project_candidate_context_to_v1(
                candidate_row,
                version=example.candidate_context_version,
            )
            latent_rows.append(
                predictor(
                    dynamics_model,
                    feature_vector=example.feature_vector,
                    action_index=int(action_index),
                    action_features=action_features,
                    transition_features=transition_row,
                )
            )
        materialized.append(
            replace(
                example,
                latent_state_version=latent_state_version,
                latent_features=latent_rows,
            )
        )
    return materialized


def load_planner_head_examples_for_split(dataset_dir: Path, split: str) -> list[PlannerHeadExample]:
    """Load the canonical planner-head artifact for one dataset directory split."""
    return load_planner_head_examples(dataset_dir / planner_head_artifact_name(split))


def _select_root_candidate_indices(
    root_scores: Sequence[float],
    *,
    teacher_top1_root_index: int,
    root_top_k: int,
) -> list[int]:
    ranked = sorted(range(len(root_scores)), key=lambda index: (-root_scores[index], index))
    selected = ranked[: min(root_top_k, len(ranked))]
    if teacher_top1_root_index not in selected:
        if len(selected) < root_top_k:
            selected.append(teacher_top1_root_index)
        else:
            selected[-1] = teacher_top1_root_index
    deduped = list(dict.fromkeys(selected))
    deduped.sort(key=lambda index: (-root_scores[index], index))
    return deduped


def _build_root_candidate_rows(
    dataset_example: DatasetExample,
    *,
    teacher_example: Any,
    considered_indices: Sequence[int],
    root_scores: Sequence[float],
    opponent_mode: str,
    proposer_model: Any,
    dynamics_model: Any,
    opponent_model: Any,
    repo_root: Path,
) -> list[dict[str, Any]]:
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

    rows: list[dict[str, Any]] = []
    for candidate_list_index, root_selected_example, successor_payload in zip(
        considered_indices,
        root_selected_examples,
        successor_payloads,
        strict=True,
    ):
        successor_example = dataset_example_from_oracle_payload(
            sample_id=dataset_example.sample_id,
            split=dataset_example.split,
            source="planner_head",
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
            _reply_scores, reply_policy = score_symbolic_candidates(
                proposer_model,
                feature_vector=successor_symbolic.feature_vector,
                candidate_action_indices=successor_symbolic.candidate_action_indices,
                candidate_features=successor_symbolic.candidate_features,
                global_features=successor_symbolic.global_features,
                candidate_context_version=successor_symbolic.candidate_context_version,
            )
            reply_peak_probability = max(reply_policy) if reply_policy else 0.0
            if reply_policy:
                top_index = max(range(len(reply_policy)), key=lambda index: reply_policy[index])
                pressure = _pressure_from_candidate_features(
                    successor_symbolic.candidate_features[top_index]
                )
                uncertainty = 1.0 - reply_peak_probability
            else:
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
        latent_features = None
        if dynamics_model is not None:
            latent_features = predict_dynamics_latent(
                dynamics_model,
                feature_vector=teacher_example.feature_vector,
                action_index=int(teacher_example.candidate_action_indices[candidate_list_index]),
                action_features=build_selected_move_action_features(
                    root_selected_example,
                    candidate_context_version=1,
                ),
                transition_features=transition_features,
            )

        rows.append(
            {
                "action_index": int(teacher_example.candidate_action_indices[candidate_list_index]),
                "proposer_score": float(root_scores[candidate_list_index]),
                "transition_features": [float(value) for value in transition_features],
                "latent_features": latent_features,
                "reply_peak_probability": float(reply_peak_probability),
                "pressure": float(pressure),
                "uncertainty": float(uncertainty),
            }
        )
    return rows


def _restricted_teacher_policy(
    teacher_policy: Sequence[float],
    *,
    considered_indices: Sequence[int],
    teacher_top1_candidate_index: int,
) -> list[float]:
    restricted = [float(teacher_policy[index]) for index in considered_indices]
    total = sum(restricted)
    if total <= 0.0:
        one_hot = [0.0 for _ in restricted]
        one_hot[teacher_top1_candidate_index] = 1.0
        return one_hot
    return [value / total for value in restricted]


def _restricted_teacher_candidate_scores(
    teacher_candidate_scores_cp: Sequence[float],
    *,
    considered_indices: Sequence[int],
) -> list[float]:
    return [float(teacher_candidate_scores_cp[index]) for index in considered_indices]


def build_teacher_candidate_score_delta_targets_cp(
    teacher_candidate_scores_cp: Sequence[float],
    *,
    considered_indices: Sequence[int],
    teacher_root_value_cp: float,
    clip_cp: float = PLANNER_SCORE_TARGET_CLIP_CP,
) -> list[float]:
    """Return clipped per-candidate score deltas relative to the teacher root value."""
    if clip_cp <= 0.0:
        raise ValueError("clip_cp must be positive")
    return [
        max(
            -clip_cp,
            min(clip_cp, float(teacher_candidate_scores_cp[index]) - float(teacher_root_value_cp)),
        )
        for index in considered_indices
    ]


def _pressure_from_candidate_features(candidate_features: Sequence[float]) -> float:
    if not candidate_features:
        return 0.0
    if bool(candidate_features[4]):
        return 1.0
    if bool(candidate_features[1]):
        return 0.75
    if bool(candidate_features[0]) or bool(candidate_features[3]):
        return 0.5
    return 0.0


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_float_list(value: object | None) -> list[float] | None:
    if value is None:
        return None
    return [float(entry) for entry in list(value)]


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)
