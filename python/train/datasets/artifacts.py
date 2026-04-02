"""Dataset artifact loaders and feature packing for proposer and dynamics training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from train.action_space import flatten_action, flatten_legal_actions
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import DatasetExample, PositionEncoding, RawPositionRecord, SUPPORTED_SPLITS

PIECE_TOKEN_CAPACITY = 32
PIECE_TOKEN_WIDTH = 3
PIECE_TOKEN_PADDING_VALUE = -1
SQUARE_TOKEN_COUNT = 64
SQUARE_TOKEN_WIDTH = 2
RULE_TOKEN_WIDTH = 6
POSITION_FEATURE_SIZE = (
    PIECE_TOKEN_CAPACITY * PIECE_TOKEN_WIDTH
    + SQUARE_TOKEN_COUNT * SQUARE_TOKEN_WIDTH
    + RULE_TOKEN_WIDTH
)
PIECE_FEATURE_SIZE = PIECE_TOKEN_CAPACITY * PIECE_TOKEN_WIDTH
SQUARE_FEATURE_SIZE = SQUARE_TOKEN_COUNT * SQUARE_TOKEN_WIDTH
RULE_FEATURE_SIZE = RULE_TOKEN_WIDTH
PIECE_FEATURE_SLICE = slice(0, PIECE_FEATURE_SIZE)
SQUARE_FEATURE_SLICE = slice(PIECE_FEATURE_SIZE, PIECE_FEATURE_SIZE + SQUARE_FEATURE_SIZE)
RULE_FEATURE_SLICE = slice(PIECE_FEATURE_SIZE + SQUARE_FEATURE_SIZE, POSITION_FEATURE_SIZE)
PROPOSER_ARTIFACT_PREFIX = "proposer_"
DYNAMICS_ARTIFACT_PREFIX = "dynamics_"


@dataclass(frozen=True)
class ProposerTrainingExample:
    """Training-ready proposer example with packed numeric features."""

    sample_id: str
    split: str
    feature_vector: list[float]
    legal_action_indices: list[int]
    selected_action_index: int | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "feature_vector": self.feature_vector,
            "legal_action_indices": self.legal_action_indices,
            "selected_action_index": self.selected_action_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProposerTrainingExample":
        """Construct the training example from JSON."""
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        feature_vector = [float(value) for value in list(payload["feature_vector"])]
        legal_action_indices = sorted(int(value) for value in list(payload["legal_action_indices"]))
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            feature_vector=feature_vector,
            legal_action_indices=legal_action_indices,
            selected_action_index=_optional_int(payload.get("selected_action_index")),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "ProposerTrainingExample":
        """Parse the training example from one JSON line."""
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: proposer training example must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class DynamicsTrainingExample:
    """Training-ready dynamics example with exact one-step transition targets."""

    sample_id: str
    split: str
    feature_vector: list[float]
    action_index: int
    next_feature_vector: list[float]
    is_capture: bool
    is_promotion: bool
    is_castle: bool
    is_en_passant: bool
    gives_check: bool
    trajectory_id: str | None
    ply_index: int | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "feature_vector": self.feature_vector,
            "action_index": self.action_index,
            "next_feature_vector": self.next_feature_vector,
            "is_capture": self.is_capture,
            "is_promotion": self.is_promotion,
            "is_castle": self.is_castle,
            "is_en_passant": self.is_en_passant,
            "gives_check": self.gives_check,
            "trajectory_id": self.trajectory_id,
            "ply_index": self.ply_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DynamicsTrainingExample":
        """Construct the training example from JSON."""
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            feature_vector=[float(value) for value in list(payload["feature_vector"])],
            action_index=int(payload["action_index"]),
            next_feature_vector=[
                float(value) for value in list(payload["next_feature_vector"])
            ],
            is_capture=bool(payload["is_capture"]),
            is_promotion=bool(payload["is_promotion"]),
            is_castle=bool(payload["is_castle"]),
            is_en_passant=bool(payload["is_en_passant"]),
            gives_check=bool(payload["gives_check"]),
            trajectory_id=_optional_str(payload.get("trajectory_id")),
            ply_index=_optional_int(payload.get("ply_index")),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "DynamicsTrainingExample":
        """Parse the training example from one JSON line."""
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: dynamics training example must be a JSON object")
        return cls.from_dict(payload)


def load_dataset_examples(dataset_path: Path) -> list[DatasetExample]:
    """Load serialized dataset examples from a dataset directory or JSONL file."""
    target_path = dataset_path / "dataset.jsonl" if dataset_path.is_dir() else dataset_path
    return _load_examples_from_jsonl(target_path)


def load_split_examples(dataset_path: Path, split: str) -> list[DatasetExample]:
    """Load examples for a specific split from dataset artifacts."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")

    if dataset_path.is_dir():
        split_path = dataset_path / f"{split}.jsonl"
        if split_path.exists():
            return _load_examples_from_jsonl(split_path)

    return [example for example in load_dataset_examples(dataset_path) if example.split == split]


def load_proposer_examples(dataset_path: Path, split: str) -> list[ProposerTrainingExample]:
    """Load split examples and convert them into proposer-ready features."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")

    if dataset_path.is_dir():
        proposer_path = dataset_path / proposer_artifact_name(split)
        if proposer_path.exists():
            return _load_proposer_examples_from_jsonl(proposer_path)

    return [to_proposer_example(example) for example in load_split_examples(dataset_path, split)]


def load_dynamics_examples(
    dataset_path: Path,
    split: str,
    *,
    repo_root: Path | None = None,
    oracle_command: Sequence[str] | None = None,
) -> list[DynamicsTrainingExample]:
    """Load split examples and convert them into dynamics-ready transition records."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")

    if dataset_path.is_dir():
        dynamics_path = dataset_path / dynamics_artifact_name(split)
        if dynamics_path.exists():
            return _load_dynamics_examples_from_jsonl(dynamics_path)

    return _build_dynamics_examples(
        load_split_examples(dataset_path, split),
        repo_root=repo_root,
        oracle_command=oracle_command,
    )


def materialize_proposer_artifacts(dataset_path: Path) -> dict[str, int]:
    """Write proposer_<split>.jsonl files next to full dataset split artifacts."""
    if not dataset_path.is_dir():
        raise ValueError("dataset_path must be a directory")

    written_counts: dict[str, int] = {}
    for split in sorted(SUPPORTED_SPLITS):
        split_examples = load_split_examples(dataset_path, split)
        proposer_examples = [to_proposer_example(example) for example in split_examples]
        _write_jsonl(path=dataset_path / proposer_artifact_name(split), records=proposer_examples)
        written_counts[split] = len(proposer_examples)
    return written_counts


def materialize_dynamics_artifacts(
    dataset_path: Path,
    *,
    repo_root: Path | None = None,
    oracle_command: Sequence[str] | None = None,
) -> dict[str, int]:
    """Write dynamics_<split>.jsonl files next to full dataset split artifacts."""
    if not dataset_path.is_dir():
        raise ValueError("dataset_path must be a directory")

    written_counts: dict[str, int] = {}
    for split in sorted(SUPPORTED_SPLITS):
        split_examples = load_split_examples(dataset_path, split)
        dynamics_examples = _build_dynamics_examples(
            split_examples,
            repo_root=repo_root,
            oracle_command=oracle_command,
        )
        _write_jsonl(path=dataset_path / dynamics_artifact_name(split), records=dynamics_examples)
        written_counts[split] = len(dynamics_examples)
    return written_counts


def to_proposer_example(example: DatasetExample) -> ProposerTrainingExample:
    """Convert a dataset example into proposer-ready supervision tensors."""
    legal_action_indices = flatten_legal_actions(example.legal_action_encodings)
    selected_action_index = (
        None
        if example.selected_action_encoding is None
        else flatten_action(example.selected_action_encoding)
    )
    if selected_action_index is not None and selected_action_index not in legal_action_indices:
        raise ValueError(
            f"{example.sample_id}: selected action is not part of the legal action set"
        )

    return ProposerTrainingExample(
        sample_id=example.sample_id,
        split=example.split,
        feature_vector=pack_position_features(example.position_encoding),
        legal_action_indices=legal_action_indices,
        selected_action_index=selected_action_index,
    )


def pack_position_features(position_encoding: PositionEncoding) -> list[float]:
    """Flatten the deterministic encoder output into a fixed-width numeric vector."""
    if len(position_encoding.piece_tokens) > PIECE_TOKEN_CAPACITY:
        raise ValueError(
            f"piece token count exceeds capacity: {len(position_encoding.piece_tokens)}"
        )

    features: list[float] = []
    features.extend(_pad_piece_tokens(position_encoding.piece_tokens))
    for token in position_encoding.square_tokens:
        features.extend(float(value) for value in token)
    features.extend(float(value) for value in position_encoding.rule_token)

    if len(features) != POSITION_FEATURE_SIZE:
        raise AssertionError(
            f"expected {POSITION_FEATURE_SIZE} packed features, got {len(features)}"
        )

    return features


def position_feature_spec() -> dict[str, object]:
    """Describe the fixed-width proposer input layout."""
    return {
        "feature_dim": POSITION_FEATURE_SIZE,
        "sections": {
            "piece": {"offset": PIECE_FEATURE_SLICE.start, "size": PIECE_FEATURE_SIZE},
            "square": {"offset": SQUARE_FEATURE_SLICE.start, "size": SQUARE_FEATURE_SIZE},
            "rule": {"offset": RULE_FEATURE_SLICE.start, "size": RULE_FEATURE_SIZE},
        },
        "layout": {
            "piece_token_capacity": PIECE_TOKEN_CAPACITY,
            "piece_token_width": PIECE_TOKEN_WIDTH,
            "piece_padding_value": PIECE_TOKEN_PADDING_VALUE,
            "square_token_count": SQUARE_TOKEN_COUNT,
            "square_token_width": SQUARE_TOKEN_WIDTH,
            "rule_token_width": RULE_TOKEN_WIDTH,
            "flatten_order": [
                "piece_tokens padded to 32 rows with [-1, -1, -1]",
                "square_tokens[64][square_index, occupant_code]",
                "rule_token[side_to_move, castling_bits, en_passant_square, halfmove_clock, fullmove_number, repetition_count]",
            ],
        },
    }


def proposer_artifact_name(split: str) -> str:
    """Return the canonical proposer artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{PROPOSER_ARTIFACT_PREFIX}{split}.jsonl"


def dynamics_artifact_name(split: str) -> str:
    """Return the canonical dynamics artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{DYNAMICS_ARTIFACT_PREFIX}{split}.jsonl"


def split_position_features(feature_vector: Sequence[float]) -> dict[str, list[float]]:
    """Split one packed feature vector into deterministic piece/square/rule sections."""
    values = [float(value) for value in feature_vector]
    if len(values) != POSITION_FEATURE_SIZE:
        raise ValueError(
            f"expected {POSITION_FEATURE_SIZE} packed features, got {len(values)}"
        )
    return {
        "piece": values[PIECE_FEATURE_SLICE],
        "square": values[SQUARE_FEATURE_SLICE],
        "rule": values[RULE_FEATURE_SLICE],
    }


def _load_examples_from_jsonl(path: Path) -> list[DatasetExample]:
    if not path.exists():
        raise FileNotFoundError(f"dataset artifact not found: {path}")

    examples: list[DatasetExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(DatasetExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def _load_proposer_examples_from_jsonl(path: Path) -> list[ProposerTrainingExample]:
    if not path.exists():
        raise FileNotFoundError(f"dataset artifact not found: {path}")

    examples: list[ProposerTrainingExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(ProposerTrainingExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def _load_dynamics_examples_from_jsonl(path: Path) -> list[DynamicsTrainingExample]:
    if not path.exists():
        raise FileNotFoundError(f"dataset artifact not found: {path}")

    examples: list[DynamicsTrainingExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(DynamicsTrainingExample.from_json(line, source=f"{path}:{line_number}"))
    return examples


def _write_jsonl(path: Path, records: Sequence[object]) -> None:
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _pad_piece_tokens(piece_tokens: Sequence[Sequence[int]]) -> list[float]:
    padded: list[float] = []
    for token in piece_tokens:
        if len(token) != PIECE_TOKEN_WIDTH:
            raise ValueError("piece tokens must have width 3")
        padded.extend(float(value) for value in token)

    remaining_rows = PIECE_TOKEN_CAPACITY - len(piece_tokens)
    padded.extend(
        float(PIECE_TOKEN_PADDING_VALUE) for _ in range(remaining_rows * PIECE_TOKEN_WIDTH)
    )
    return padded


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _build_dynamics_examples(
    split_examples: list[DatasetExample],
    *,
    repo_root: Path | None,
    oracle_command: Sequence[str] | None,
) -> list[DynamicsTrainingExample]:
    transition_examples = [
        example
        for example in split_examples
        if example.selected_action_encoding is not None and example.next_fen is not None
    ]
    if not transition_examples:
        return []

    next_records = [
        RawPositionRecord(
            sample_id=f"{example.sample_id}:next",
            fen=str(example.next_fen),
            source=example.source,
        )
        for example in transition_examples
    ]
    next_payloads = label_records_with_oracle(
        next_records,
        repo_root=repo_root,
        command=oracle_command,
    )

    dynamics_examples: list[DynamicsTrainingExample] = []
    for example, next_payload in zip(transition_examples, next_payloads, strict=True):
        next_position = PositionEncoding.from_oracle_dict(dict(next_payload["position_encoding"]))
        dynamics_examples.append(
            DynamicsTrainingExample(
                sample_id=example.sample_id,
                split=example.split,
                feature_vector=pack_position_features(example.position_encoding),
                action_index=flatten_action(example.selected_action_encoding),
                next_feature_vector=pack_position_features(next_position),
                is_capture=bool(example.annotations.selected_move_is_capture),
                is_promotion=bool(example.annotations.selected_move_is_promotion),
                is_castle=bool(example.annotations.selected_move_is_castle),
                is_en_passant=bool(example.annotations.selected_move_is_en_passant),
                gives_check=bool(example.annotations.selected_move_gives_check),
                trajectory_id=_trajectory_id(example),
                ply_index=_trajectory_ply(example),
            )
        )
    return dynamics_examples


def _trajectory_id(example: DatasetExample) -> str | None:
    source_pgn = example.metadata.get("source_pgn")
    game_index = example.metadata.get("game_index")
    if source_pgn is None or game_index is None:
        return None
    return f"{source_pgn}:{int(game_index)}"


def _trajectory_ply(example: DatasetExample) -> int | None:
    ply = example.metadata.get("ply")
    if ply is None:
        return None
    return int(ply)
