"""OpponentHeadV1 dataset artifacts derived from exact successor states and search workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from train.action_space import flatten_action
from train.datasets.artifacts import (
    DEFAULT_GLOBAL_CONTEXT_VERSION,
    POSITION_FEATURE_SIZE,
    build_symbolic_proposer_example,
    build_transition_context_features,
    pack_position_features,
)
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import (
    DatasetExample,
    PositionEncoding,
    RawPositionRecord,
    SUPPORTED_SPLITS,
    TacticalAnnotations,
)
from train.datasets.search_curriculum import SearchCurriculumExample
from train.datasets.search_traces import SearchTraceExample


OPPONENT_HEAD_ARTIFACT_PREFIX = "opponent_head_"


@dataclass(frozen=True)
class OpponentHeadExample:
    """First OpponentHeadV1 training example over exact legal reply candidates."""

    sample_id: str
    split: str
    root_fen: str
    root_feature_vector: list[float]
    curriculum_bucket_labels: list[str]
    curriculum_priority: float
    chosen_move_uci: str
    chosen_action_index: int
    transition_context_version: int
    transition_features: list[float]
    next_fen: str
    next_feature_vector: list[float]
    reply_candidate_context_version: int
    reply_global_context_version: int
    reply_global_features: list[float]
    reply_candidate_action_indices: list[int]
    reply_candidate_features: list[list[float]]
    teacher_reply_uci: str | None
    teacher_reply_action_index: int | None
    teacher_reply_policy: list[float]
    teacher_root_value_cp: float
    teacher_root_value_mate: int | None
    teacher_top1_minus_top2_cp: float | None
    pressure_target: float
    uncertainty_target: float
    reply_is_capture: bool
    reply_is_promotion: bool
    reply_is_castle: bool
    reply_is_en_passant: bool
    reply_gives_check: bool

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "root_fen": self.root_fen,
            "root_feature_vector": self.root_feature_vector,
            "curriculum_bucket_labels": self.curriculum_bucket_labels,
            "curriculum_priority": self.curriculum_priority,
            "chosen_move_uci": self.chosen_move_uci,
            "chosen_action_index": self.chosen_action_index,
            "transition_context_version": self.transition_context_version,
            "transition_features": self.transition_features,
            "next_fen": self.next_fen,
            "next_feature_vector": self.next_feature_vector,
            "reply_candidate_context_version": self.reply_candidate_context_version,
            "reply_global_context_version": self.reply_global_context_version,
            "reply_global_features": self.reply_global_features,
            "reply_candidate_action_indices": self.reply_candidate_action_indices,
            "reply_candidate_features": self.reply_candidate_features,
            "teacher_reply_uci": self.teacher_reply_uci,
            "teacher_reply_action_index": self.teacher_reply_action_index,
            "teacher_reply_policy": self.teacher_reply_policy,
            "teacher_root_value_cp": self.teacher_root_value_cp,
            "teacher_root_value_mate": self.teacher_root_value_mate,
            "teacher_top1_minus_top2_cp": self.teacher_top1_minus_top2_cp,
            "pressure_target": self.pressure_target,
            "uncertainty_target": self.uncertainty_target,
            "reply_is_capture": self.reply_is_capture,
            "reply_is_promotion": self.reply_is_promotion,
            "reply_is_castle": self.reply_is_castle,
            "reply_is_en_passant": self.reply_is_en_passant,
            "reply_gives_check": self.reply_gives_check,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OpponentHeadExample":
        """Construct the example from JSON."""
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        root_feature_vector = [float(value) for value in list(payload["root_feature_vector"])]
        if len(root_feature_vector) != POSITION_FEATURE_SIZE:
            raise ValueError(
                f"root_feature_vector must have width {POSITION_FEATURE_SIZE}"
            )
        next_feature_vector = [float(value) for value in list(payload["next_feature_vector"])]
        if len(next_feature_vector) != POSITION_FEATURE_SIZE:
            raise ValueError(
                f"next_feature_vector must have width {POSITION_FEATURE_SIZE}"
            )
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            root_fen=str(payload["root_fen"]),
            root_feature_vector=root_feature_vector,
            curriculum_bucket_labels=[
                str(value) for value in list(payload["curriculum_bucket_labels"])
            ],
            curriculum_priority=float(payload["curriculum_priority"]),
            chosen_move_uci=str(payload["chosen_move_uci"]),
            chosen_action_index=int(payload["chosen_action_index"]),
            transition_context_version=int(payload["transition_context_version"]),
            transition_features=[float(value) for value in list(payload["transition_features"])],
            next_fen=str(payload["next_fen"]),
            next_feature_vector=next_feature_vector,
            reply_candidate_context_version=int(payload["reply_candidate_context_version"]),
            reply_global_context_version=int(payload["reply_global_context_version"]),
            reply_global_features=[float(value) for value in list(payload["reply_global_features"])],
            reply_candidate_action_indices=[
                int(value) for value in list(payload["reply_candidate_action_indices"])
            ],
            reply_candidate_features=[
                [float(value) for value in row]
                for row in list(payload["reply_candidate_features"])
            ],
            teacher_reply_uci=_optional_str(payload.get("teacher_reply_uci")),
            teacher_reply_action_index=_optional_int(payload.get("teacher_reply_action_index")),
            teacher_reply_policy=[float(value) for value in list(payload["teacher_reply_policy"])],
            teacher_root_value_cp=float(payload["teacher_root_value_cp"]),
            teacher_root_value_mate=_optional_int(payload.get("teacher_root_value_mate")),
            teacher_top1_minus_top2_cp=_optional_float(payload.get("teacher_top1_minus_top2_cp")),
            pressure_target=float(payload["pressure_target"]),
            uncertainty_target=float(payload["uncertainty_target"]),
            reply_is_capture=bool(payload["reply_is_capture"]),
            reply_is_promotion=bool(payload["reply_is_promotion"]),
            reply_is_castle=bool(payload["reply_is_castle"]),
            reply_is_en_passant=bool(payload["reply_is_en_passant"]),
            reply_gives_check=bool(payload["reply_gives_check"]),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "OpponentHeadExample":
        """Parse one JSONL row."""
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: opponent head example must be a JSON object")
        return cls.from_dict(payload)


def opponent_head_artifact_name(split: str) -> str:
    """Return the canonical OpponentHead artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{OPPONENT_HEAD_ARTIFACT_PREFIX}{split}.jsonl"


def load_opponent_head_examples(path: Path) -> list[OpponentHeadExample]:
    """Load OpponentHead examples from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"opponent head artifact not found: {path}")
    examples: list[OpponentHeadExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(OpponentHeadExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def write_opponent_head_artifact(path: Path, examples: Sequence[OpponentHeadExample]) -> None:
    """Write OpponentHead examples as JSONL."""
    lines = [json.dumps(example.to_dict(), sort_keys=True) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_opponent_head_examples(
    dataset_examples: Sequence[DatasetExample],
    trace_examples: Sequence[SearchTraceExample],
    curriculum_examples: Sequence[SearchCurriculumExample],
    *,
    repo_root: Path | None = None,
    oracle_command: Sequence[str] | None = None,
) -> list[OpponentHeadExample]:
    """Derive the first OpponentHeadV1 dataset from traces, curriculum, and exact symbolic states."""
    dataset_by_sample_id = {example.sample_id: example for example in dataset_examples}
    curriculum_by_sample_id = {example.sample_id: example for example in curriculum_examples}

    root_records: list[RawPositionRecord] = []
    root_trace_examples: list[SearchTraceExample] = []
    root_dataset_examples: list[DatasetExample] = []
    root_curriculum_examples: list[SearchCurriculumExample] = []
    chosen_move_ucis: list[str] = []

    for trace_example in trace_examples:
        dataset_example = dataset_by_sample_id.get(trace_example.sample_id)
        curriculum_example = curriculum_by_sample_id.get(trace_example.sample_id)
        if dataset_example is None or curriculum_example is None:
            raise ValueError(
                f"{trace_example.sample_id}: dataset or curriculum artifact missing for opponent build"
            )
        if dataset_example.fen != trace_example.fen or curriculum_example.fen != trace_example.fen:
            raise ValueError(f"{trace_example.sample_id}: root FEN mismatch across artifacts")

        chosen_action_index = trace_example.teacher_top_k_action_indices[0]
        chosen_move_uci = _move_uci_for_action(dataset_example, chosen_action_index)
        root_records.append(
            RawPositionRecord(
                sample_id=f"{trace_example.sample_id}:opponent_root",
                fen=trace_example.fen,
                source=dataset_example.source,
                selected_move_uci=chosen_move_uci,
            )
        )
        root_trace_examples.append(trace_example)
        root_dataset_examples.append(dataset_example)
        root_curriculum_examples.append(curriculum_example)
        chosen_move_ucis.append(chosen_move_uci)

    root_payloads = label_records_with_oracle(
        root_records,
        repo_root=repo_root,
        command=oracle_command,
    )

    successor_records: list[RawPositionRecord] = []
    root_selected_examples: list[DatasetExample] = []
    for trace_example, dataset_example, root_payload in zip(
        root_trace_examples,
        root_dataset_examples,
        root_payloads,
        strict=True,
    ):
        root_selected_example = _dataset_example_from_oracle_payload(
            sample_id=dataset_example.sample_id,
            split=dataset_example.split,
            source=dataset_example.source,
            fen=trace_example.fen,
            payload=root_payload,
        )
        root_selected_examples.append(root_selected_example)
        successor_records.append(
            RawPositionRecord(
                sample_id=f"{trace_example.sample_id}:opponent_successor",
                fen=str(root_selected_example.next_fen),
                source=dataset_example.source,
                selected_move_uci=trace_example.best_reply_uci,
            )
        )

    successor_payloads = label_records_with_oracle(
        successor_records,
        repo_root=repo_root,
        command=oracle_command,
    )

    built: list[OpponentHeadExample] = []
    for (
        trace_example,
        curriculum_example,
        chosen_move_uci,
        root_selected_example,
        successor_payload,
    ) in zip(
        root_trace_examples,
        root_curriculum_examples,
        chosen_move_ucis,
        root_selected_examples,
        successor_payloads,
        strict=True,
    ):
        successor_example = _dataset_example_from_oracle_payload(
            sample_id=trace_example.sample_id,
            split=trace_example.split,
            source="opponent_head",
            fen=str(root_selected_example.next_fen),
            payload=successor_payload,
        )
        successor_symbolic = build_symbolic_proposer_example(
            successor_example,
            candidate_context_version=2,
            global_context_version=DEFAULT_GLOBAL_CONTEXT_VERSION,
        )
        teacher_reply_action_index = (
            None
            if successor_example.selected_action_encoding is None
            else flatten_action(successor_example.selected_action_encoding)
        )
        teacher_reply_policy = _teacher_reply_policy(
            successor_symbolic.candidate_action_indices,
            teacher_reply_action_index,
        )
        pressure_target = _pressure_target(successor_example)
        uncertainty_target = _uncertainty_target(curriculum_example)

        built.append(
            OpponentHeadExample(
                sample_id=trace_example.sample_id,
                split=trace_example.split,
                root_fen=trace_example.fen,
                root_feature_vector=pack_position_features(root_selected_example.position_encoding),
                curriculum_bucket_labels=list(curriculum_example.bucket_labels),
                curriculum_priority=curriculum_example.curriculum_priority,
                chosen_move_uci=chosen_move_uci,
                chosen_action_index=flatten_action(root_selected_example.selected_action_encoding),
                transition_context_version=1,
                transition_features=build_transition_context_features(
                    root_selected_example,
                    version=1,
                ),
                next_fen=str(root_selected_example.next_fen),
                next_feature_vector=list(successor_symbolic.feature_vector),
                reply_candidate_context_version=successor_symbolic.candidate_context_version,
                reply_global_context_version=successor_symbolic.global_context_version,
                reply_global_features=list(successor_symbolic.global_features),
                reply_candidate_action_indices=list(successor_symbolic.candidate_action_indices),
                reply_candidate_features=[list(row) for row in successor_symbolic.candidate_features],
                teacher_reply_uci=trace_example.best_reply_uci,
                teacher_reply_action_index=teacher_reply_action_index,
                teacher_reply_policy=teacher_reply_policy,
                teacher_root_value_cp=trace_example.teacher_root_value_cp,
                teacher_root_value_mate=trace_example.teacher_root_value_mate,
                teacher_top1_minus_top2_cp=trace_example.top1_minus_top2_cp,
                pressure_target=pressure_target,
                uncertainty_target=uncertainty_target,
                reply_is_capture=bool(successor_example.annotations.selected_move_is_capture),
                reply_is_promotion=bool(successor_example.annotations.selected_move_is_promotion),
                reply_is_castle=bool(successor_example.annotations.selected_move_is_castle),
                reply_is_en_passant=bool(successor_example.annotations.selected_move_is_en_passant),
                reply_gives_check=bool(successor_example.annotations.selected_move_gives_check),
            )
        )

    return built


def _dataset_example_from_oracle_payload(
    *,
    sample_id: str,
    split: str,
    source: str,
    fen: str,
    payload: dict[str, object],
) -> DatasetExample:
    return DatasetExample(
        sample_id=sample_id,
        split=split,
        source=source,
        fen=fen,
        side_to_move=str(payload["side_to_move"]),
        selected_move_uci=_optional_str(payload.get("selected_move_uci")),
        selected_action_encoding=_optional_int_list(payload.get("selected_action_encoding")),
        next_fen=_optional_str(payload.get("next_fen")),
        legal_moves=[str(value) for value in list(payload["legal_moves"])],
        legal_action_encodings=[
            [int(component) for component in action]
            for action in list(payload["legal_action_encodings"])
        ],
        position_encoding=PositionEncoding.from_oracle_dict(dict(payload["position_encoding"])),
        wdl_target=None,
        annotations=TacticalAnnotations.from_oracle_dict(dict(payload["annotations"])),
        result=None,
        metadata={},
    )


def _move_uci_for_action(example: DatasetExample, action_index: int) -> str:
    for move_uci, action in zip(example.legal_moves, example.legal_action_encodings, strict=True):
        if flatten_action(action) == action_index:
            return move_uci
    raise ValueError(f"{example.sample_id}: no legal move for action index {action_index}")


def _teacher_reply_policy(
    reply_candidate_action_indices: Sequence[int],
    teacher_reply_action_index: int | None,
) -> list[float]:
    if not reply_candidate_action_indices:
        return []
    if teacher_reply_action_index is None:
        return [0.0] * len(reply_candidate_action_indices)
    policy = [0.0] * len(reply_candidate_action_indices)
    try:
        policy[reply_candidate_action_indices.index(teacher_reply_action_index)] = 1.0
    except ValueError as error:
        raise ValueError(
            f"teacher reply action {teacher_reply_action_index} missing from exact reply candidate set"
        ) from error
    return policy


def _pressure_target(example: DatasetExample) -> float:
    annotations = example.annotations
    if bool(annotations.selected_move_gives_check):
        return 1.0
    if bool(annotations.selected_move_is_promotion):
        return 0.75
    if bool(annotations.selected_move_is_capture):
        return 0.5
    if bool(annotations.selected_move_is_en_passant):
        return 0.5
    return 0.0


def _uncertainty_target(example: SearchCurriculumExample) -> float:
    if "forced_teacher" in example.bucket_labels:
        return 0.0
    if "unstable_teacher" in example.bucket_labels:
        return 1.0
    gap_cp = example.teacher_top1_minus_top2_cp
    if gap_cp is None:
        return 0.5
    return max(0.0, min(1.0, 1.0 - (gap_cp / 80.0)))


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int_list(value: object) -> list[int] | None:
    if value is None:
        return None
    return [int(element) for element in list(value)]
