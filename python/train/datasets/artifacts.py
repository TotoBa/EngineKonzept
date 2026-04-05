"""Dataset artifact loaders and feature packing for proposer and dynamics training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from train.action_space import flatten_action, flatten_legal_actions
from train.datasets.contracts import (
    DEFAULT_CANDIDATE_CONTEXT_VERSION,
    DEFAULT_GLOBAL_CONTEXT_VERSION,
    SYMBOLIC_MAX_LEGAL_CANDIDATES as CONTRACT_SYMBOLIC_MAX_LEGAL_CANDIDATES,
    candidate_context_feature_dim,
    global_context_feature_dim,
    symbolic_move_delta_feature_dim,
    symbolic_move_delta_spec,
    symbolic_candidate_context_spec,
    transition_context_feature_dim,
    transition_context_spec,
)
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import DatasetExample, PositionEncoding, RawPositionRecord, SUPPORTED_SPLITS

try:
    import chess
except ModuleNotFoundError:  # pragma: no cover - exercised when chess is absent
    chess = None

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
SYMBOLIC_PROPOSER_ARTIFACT_PREFIX = "proposer_symbolic_"
DYNAMICS_ARTIFACT_PREFIX = "dynamics_"
SYMBOLIC_PROPOSER_CANDIDATE_FEATURE_SIZE = candidate_context_feature_dim(
    DEFAULT_CANDIDATE_CONTEXT_VERSION
)
SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE = global_context_feature_dim(
    DEFAULT_GLOBAL_CONTEXT_VERSION
)
SYMBOLIC_MAX_LEGAL_CANDIDATES = CONTRACT_SYMBOLIC_MAX_LEGAL_CANDIDATES
SYMBOLIC_MOVE_DELTA_FEATURE_SIZE = symbolic_move_delta_feature_dim(1)
TRANSITION_CONTEXT_FEATURE_SIZE = transition_context_feature_dim(1)


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
class SymbolicProposerTrainingExample:
    """Training-ready proposer example with exact symbolic candidate features."""

    sample_id: str
    split: str
    feature_vector: list[float]
    candidate_context_version: int
    global_context_version: int
    global_features: list[float]
    legal_action_indices: list[int]
    candidate_action_indices: list[int]
    candidate_features: list[list[float]]
    selected_action_index: int | None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "feature_vector": self.feature_vector,
            "candidate_context_version": self.candidate_context_version,
            "global_context_version": self.global_context_version,
            "global_features": self.global_features,
            "legal_action_indices": self.legal_action_indices,
            "candidate_action_indices": self.candidate_action_indices,
            "candidate_features": self.candidate_features,
            "selected_action_index": self.selected_action_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SymbolicProposerTrainingExample":
        """Construct the symbolic training example from JSON."""
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        candidate_context_version = int(
            payload.get("candidate_context_version", DEFAULT_CANDIDATE_CONTEXT_VERSION)
        )
        global_context_version = int(
            payload.get("global_context_version", DEFAULT_GLOBAL_CONTEXT_VERSION)
        )
        global_features = [float(value) for value in list(payload["global_features"])]
        if len(global_features) != global_context_feature_dim(global_context_version):
            raise ValueError(
                "global_features must have width "
                f"{global_context_feature_dim(global_context_version)}"
            )
        candidate_features = [
            [float(value) for value in row] for row in list(payload["candidate_features"])
        ]
        for row in candidate_features:
            if len(row) != candidate_context_feature_dim(candidate_context_version):
                raise ValueError(
                    "candidate_features rows must have width "
                    f"{candidate_context_feature_dim(candidate_context_version)}"
                )
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            feature_vector=[float(value) for value in list(payload["feature_vector"])],
            candidate_context_version=candidate_context_version,
            global_context_version=global_context_version,
            global_features=global_features,
            legal_action_indices=sorted(int(value) for value in list(payload["legal_action_indices"])),
            candidate_action_indices=[
                int(value) for value in list(payload["candidate_action_indices"])
            ],
            candidate_features=candidate_features,
            selected_action_index=_optional_int(payload.get("selected_action_index")),
        )

    @classmethod
    def from_json(cls, line: str, *, source: str = "<jsonl>") -> "SymbolicProposerTrainingExample":
        """Parse the symbolic training example from one JSON line."""
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{source}: symbolic proposer training example must be a JSON object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class DynamicsTrainingExample:
    """Training-ready dynamics example with exact one-step transition targets."""

    sample_id: str
    split: str
    feature_vector: list[float]
    action_index: int
    action_candidate_context_version: int | None
    action_features: list[float] | None
    next_feature_vector: list[float]
    is_capture: bool
    is_promotion: bool
    is_castle: bool
    is_en_passant: bool
    gives_check: bool
    trajectory_id: str | None
    ply_index: int | None
    transition_context_version: int | None = None
    transition_features: list[float] | None = None
    symbolic_move_delta_version: int | None = None
    symbolic_move_delta_features: list[float] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "feature_vector": self.feature_vector,
            "action_index": self.action_index,
            "action_candidate_context_version": self.action_candidate_context_version,
            "action_features": self.action_features,
            "next_feature_vector": self.next_feature_vector,
            "is_capture": self.is_capture,
            "is_promotion": self.is_promotion,
            "is_castle": self.is_castle,
            "is_en_passant": self.is_en_passant,
            "gives_check": self.gives_check,
            "trajectory_id": self.trajectory_id,
            "ply_index": self.ply_index,
            "transition_context_version": self.transition_context_version,
            "transition_features": self.transition_features,
            "symbolic_move_delta_version": self.symbolic_move_delta_version,
            "symbolic_move_delta_features": self.symbolic_move_delta_features,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "DynamicsTrainingExample":
        """Construct the training example from JSON."""
        split = str(payload["split"])
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported split: {split}")
        action_candidate_context_version = _optional_int(
            payload.get("action_candidate_context_version")
        )
        action_features = _optional_float_list(payload.get("action_features"))
        if action_features is not None:
            version = (
                DEFAULT_CANDIDATE_CONTEXT_VERSION
                if action_candidate_context_version is None
                else action_candidate_context_version
            )
            expected_width = candidate_context_feature_dim(version)
            if len(action_features) != expected_width:
                raise ValueError(
                    "action_features must have width "
                    f"{expected_width}"
                )
        if action_features is None:
            action_candidate_context_version = None
        elif action_candidate_context_version is None:
            action_candidate_context_version = DEFAULT_CANDIDATE_CONTEXT_VERSION
        transition_context_version = _optional_int(payload.get("transition_context_version"))
        transition_features = _optional_float_list(payload.get("transition_features"))
        if transition_features is not None:
            version = 1 if transition_context_version is None else transition_context_version
            expected_width = transition_context_feature_dim(version)
            if len(transition_features) != expected_width:
                raise ValueError(
                    "transition_features must have width "
                    f"{expected_width}"
                )
        if transition_features is None:
            transition_context_version = None
        elif transition_context_version is None:
            transition_context_version = 1
        symbolic_move_delta_version = _optional_int(payload.get("symbolic_move_delta_version"))
        symbolic_move_delta_features = _optional_float_list(payload.get("symbolic_move_delta_features"))
        if symbolic_move_delta_features is not None:
            version = 1 if symbolic_move_delta_version is None else symbolic_move_delta_version
            expected_width = symbolic_move_delta_feature_dim(version)
            if len(symbolic_move_delta_features) != expected_width:
                raise ValueError(
                    "symbolic_move_delta_features must have width "
                    f"{expected_width}"
                )
        if symbolic_move_delta_features is None:
            symbolic_move_delta_version = None
        elif symbolic_move_delta_version is None:
            symbolic_move_delta_version = 1
        return cls(
            sample_id=str(payload["sample_id"]),
            split=split,
            feature_vector=[float(value) for value in list(payload["feature_vector"])],
            action_index=int(payload["action_index"]),
            action_candidate_context_version=action_candidate_context_version,
            action_features=action_features,
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
            transition_context_version=transition_context_version,
            transition_features=transition_features,
            symbolic_move_delta_version=symbolic_move_delta_version,
            symbolic_move_delta_features=symbolic_move_delta_features,
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


def load_split_examples_range(
    dataset_path: Path,
    split: str,
    *,
    start_index: int = 0,
    max_examples: int | None = None,
) -> list[DatasetExample]:
    """Load a contiguous slice of examples for one split without loading the full split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples must be non-negative when provided")
    if max_examples == 0:
        return []

    if dataset_path.is_dir():
        split_path = dataset_path / f"{split}.jsonl"
        if split_path.exists():
            return _load_examples_from_jsonl_range(
                split_path,
                start_index=start_index,
                max_examples=max_examples,
            )

    split_examples = load_split_examples(dataset_path, split)
    end_index = None if max_examples is None else start_index + max_examples
    return split_examples[start_index:end_index]


def load_proposer_examples(
    dataset_path: Path,
    split: str,
    *,
    variant: str = "standard",
) -> list[ProposerTrainingExample] | list[SymbolicProposerTrainingExample]:
    """Load split examples and convert them into proposer-ready features."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")

    if variant == "symbolic":
        if dataset_path.is_dir():
            symbolic_path = dataset_path / symbolic_proposer_artifact_name(split)
            if symbolic_path.exists():
                return _load_symbolic_proposer_examples_from_jsonl(symbolic_path)
        return [to_symbolic_proposer_example(example) for example in load_split_examples(dataset_path, split)]

    if variant != "standard":
        raise ValueError(f"unsupported proposer artifact variant: {variant}")

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

    loaded_examples: list[DynamicsTrainingExample] | None = None
    if dataset_path.is_dir():
        dynamics_path = dataset_path / dynamics_artifact_name(split)
        if dynamics_path.exists():
            loaded_examples = _load_dynamics_examples_from_jsonl(dynamics_path)
            if all(example.action_features is not None for example in loaded_examples):
                return loaded_examples

    split_examples = load_split_examples(dataset_path, split)
    if split_examples:
        return _build_dynamics_examples(
            split_examples,
            repo_root=repo_root,
            oracle_command=oracle_command,
        )

    return [] if loaded_examples is None else loaded_examples


def materialize_proposer_artifacts(dataset_path: Path) -> dict[str, int]:
    """Write proposer_<split>.jsonl files next to full dataset split artifacts."""
    if not dataset_path.is_dir():
        raise ValueError("dataset_path must be a directory")

    written_counts: dict[str, int] = {}
    for split in sorted(SUPPORTED_SPLITS):
        split_path = dataset_path / f"{split}.jsonl"
        output_path = dataset_path / proposer_artifact_name(split)
        if split_path.exists():
            written_counts[split] = _stream_write_jsonl(
                output_path,
                (
                    to_proposer_example(example)
                    for example in _iter_dataset_examples_from_jsonl(split_path)
                ),
            )
            continue

        split_examples = load_split_examples(dataset_path, split)
        proposer_examples = [to_proposer_example(example) for example in split_examples]
        _write_jsonl(path=output_path, records=proposer_examples)
        written_counts[split] = len(proposer_examples)
    return written_counts


def materialize_symbolic_proposer_artifacts(
    dataset_path: Path,
    *,
    candidate_context_version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION,
    global_context_version: int = DEFAULT_GLOBAL_CONTEXT_VERSION,
) -> dict[str, int]:
    """Write proposer_symbolic_<split>.jsonl files next to full dataset split artifacts."""
    if not dataset_path.is_dir():
        raise ValueError("dataset_path must be a directory")

    written_counts: dict[str, int] = {}
    for split in sorted(SUPPORTED_SPLITS):
        split_path = dataset_path / f"{split}.jsonl"
        output_path = dataset_path / symbolic_proposer_artifact_name(split)
        if split_path.exists():
            written_counts[split] = _stream_write_jsonl(
                output_path,
                (
                    build_symbolic_proposer_example(
                        example,
                        candidate_context_version=candidate_context_version,
                        global_context_version=global_context_version,
                    )
                    for example in _iter_dataset_examples_from_jsonl(split_path)
                ),
            )
            continue

        split_examples = load_split_examples(dataset_path, split)
        symbolic_examples = [
            build_symbolic_proposer_example(
                example,
                candidate_context_version=candidate_context_version,
                global_context_version=global_context_version,
            )
            for example in split_examples
        ]
        _write_jsonl(path=output_path, records=symbolic_examples)
        written_counts[split] = len(symbolic_examples)
    return written_counts


def materialize_dynamics_artifacts(
    dataset_path: Path,
    *,
    repo_root: Path | None = None,
    oracle_command: Sequence[str] | None = None,
    chunk_size: int = 5000,
) -> dict[str, int]:
    """Write dynamics_<split>.jsonl files next to full dataset split artifacts."""
    if not dataset_path.is_dir():
        raise ValueError("dataset_path must be a directory")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    written_counts: dict[str, int] = {}
    for split in sorted(SUPPORTED_SPLITS):
        split_path = dataset_path / f"{split}.jsonl"
        output_path = dataset_path / dynamics_artifact_name(split)
        if split_path.exists():
            written_counts[split] = _stream_write_dynamics_jsonl(
                split_path,
                output_path=output_path,
                repo_root=repo_root,
                oracle_command=oracle_command,
                chunk_size=chunk_size,
            )
            continue

        split_examples = load_split_examples(dataset_path, split)
        dynamics_examples = _build_dynamics_examples(
            split_examples,
            repo_root=repo_root,
            oracle_command=oracle_command,
        )
        _write_jsonl(path=output_path, records=dynamics_examples)
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


def to_symbolic_proposer_example(example: DatasetExample) -> SymbolicProposerTrainingExample:
    """Convert a dataset example into symbolic candidate-scoring supervision."""
    return build_symbolic_proposer_example(
        example,
        candidate_context_version=DEFAULT_CANDIDATE_CONTEXT_VERSION,
        global_context_version=DEFAULT_GLOBAL_CONTEXT_VERSION,
    )


def build_symbolic_proposer_example(
    example: DatasetExample,
    *,
    candidate_context_version: int,
    global_context_version: int,
) -> SymbolicProposerTrainingExample:
    """Convert a dataset example into a versioned symbolic candidate-scoring example."""
    if chess is None:  # pragma: no cover - exercised when chess is absent
        raise RuntimeError(
            "python-chess is required for symbolic proposer artifacts. Install the 'train' extra."
        )

    board = chess.Board(example.fen)
    own_attacks = _attack_map(board, board.turn)
    opp_attacks = _attack_map(board, not board.turn)
    legal_action_indices = flatten_legal_actions(example.legal_action_encodings)
    legal_moves_by_action = {
        flatten_action(action): move
        for move, action in zip(example.legal_moves, example.legal_action_encodings, strict=True)
    }
    candidate_action_indices = list(legal_action_indices)
    candidate_features = [
        _candidate_feature_row(
            board,
            chess.Move.from_uci(legal_moves_by_action[action_index]),
            own_attacks,
            opp_attacks,
            version=candidate_context_version,
        )
        for action_index in candidate_action_indices
    ]
    selected_action_index = (
        None
        if example.selected_action_encoding is None
        else flatten_action(example.selected_action_encoding)
    )
    if selected_action_index is not None and selected_action_index not in legal_action_indices:
        raise ValueError(
            f"{example.sample_id}: selected action is not part of the legal action set"
        )

    return SymbolicProposerTrainingExample(
        sample_id=example.sample_id,
        split=example.split,
        feature_vector=pack_position_features(example.position_encoding),
        candidate_context_version=candidate_context_version,
        global_context_version=global_context_version,
        global_features=_symbolic_global_features(
            example,
            own_attacks,
            opp_attacks,
            version=global_context_version,
        ),
        legal_action_indices=legal_action_indices,
        candidate_action_indices=candidate_action_indices,
        candidate_features=candidate_features,
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


def symbolic_proposer_artifact_name(split: str) -> str:
    """Return the canonical symbolic proposer artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{SYMBOLIC_PROPOSER_ARTIFACT_PREFIX}{split}.jsonl"


def dynamics_artifact_name(split: str) -> str:
    """Return the canonical dynamics artifact filename for one split."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return f"{DYNAMICS_ARTIFACT_PREFIX}{split}.jsonl"


def symbolic_proposer_feature_spec() -> dict[str, object]:
    """Describe the symbolic proposer side inputs."""
    return symbolic_candidate_context_spec()


def symbolic_candidate_context_v2_feature_spec() -> dict[str, object]:
    """Describe the next candidate-context contract without changing the current runtime default."""
    return symbolic_candidate_context_spec(candidate_context_version=2)


def dynamics_symbolic_action_feature_spec() -> dict[str, object]:
    """Describe the symbolic action feature vector shared with the proposer."""
    proposer_spec = symbolic_proposer_feature_spec()
    return {
        "candidate_context_version": proposer_spec["candidate_context_version"],
        "feature_dim": proposer_spec["candidate_feature_dim"],
        "feature_order": proposer_spec["candidate_feature_order"],
    }


def dynamics_symbolic_move_delta_feature_spec() -> dict[str, object]:
    """Describe the optional symbolic move-delta feature vector for hybrid dynamics."""
    spec = symbolic_move_delta_spec()
    return {
        "symbolic_move_delta_version": spec.version,
        "feature_dim": spec.feature_dim,
        "feature_order": list(spec.feature_order),
    }


def transition_context_feature_spec() -> dict[str, object]:
    """Describe the richer selected-action transition contract for future dynamics/opponent work."""
    return transition_context_spec()


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


def _load_examples_from_jsonl_range(
    path: Path,
    *,
    start_index: int,
    max_examples: int | None,
) -> list[DatasetExample]:
    if not path.exists():
        raise FileNotFoundError(f"dataset artifact not found: {path}")

    examples: list[DatasetExample] = []
    end_index = None if max_examples is None else start_index + max_examples
    current_index = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            if current_index < start_index:
                current_index += 1
                continue
            if end_index is not None and current_index >= end_index:
                break
            examples.append(DatasetExample.from_json(line, source=f"{path}:{line_number}"))
            current_index += 1
    return examples


def _iter_dataset_examples_from_jsonl(path: Path) -> Iterator[DatasetExample]:
    if not path.exists():
        raise FileNotFoundError(f"dataset artifact not found: {path}")

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            yield DatasetExample.from_json(line, source=f"{path}:{line_number}")


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


def _load_symbolic_proposer_examples_from_jsonl(path: Path) -> list[SymbolicProposerTrainingExample]:
    if not path.exists():
        raise FileNotFoundError(f"dataset artifact not found: {path}")

    examples: list[SymbolicProposerTrainingExample] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        examples.append(
            SymbolicProposerTrainingExample.from_json(line, source=f"{path}:{line_number}")
        )
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


def _stream_write_jsonl(path: Path, records: Any) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def _stream_write_dynamics_jsonl(
    split_path: Path,
    *,
    output_path: Path,
    repo_root: Path | None,
    oracle_command: Sequence[str] | None,
    chunk_size: int,
) -> int:
    written = 0
    buffered_examples: list[DatasetExample] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for example in _iter_dataset_examples_from_jsonl(split_path):
            if example.selected_action_encoding is None or example.next_fen is None:
                continue
            buffered_examples.append(example)
            if len(buffered_examples) < chunk_size:
                continue
            written += _flush_dynamics_chunk(
                buffered_examples,
                handle=handle,
                repo_root=repo_root,
                oracle_command=oracle_command,
            )
            buffered_examples = []
        if buffered_examples:
            written += _flush_dynamics_chunk(
                buffered_examples,
                handle=handle,
                repo_root=repo_root,
                oracle_command=oracle_command,
            )
    return written


def _flush_dynamics_chunk(
    examples: Sequence[DatasetExample],
    *,
    handle: Any,
    repo_root: Path | None,
    oracle_command: Sequence[str] | None,
) -> int:
    dynamics_examples = _build_dynamics_examples(
        examples,
        repo_root=repo_root,
        oracle_command=oracle_command,
    )
    for record in dynamics_examples:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    return len(dynamics_examples)


def _symbolic_global_features(
    example: DatasetExample,
    own_attacks: list[bool],
    opp_attacks: list[bool],
    *,
    version: int = DEFAULT_GLOBAL_CONTEXT_VERSION,
) -> list[float]:
    if version != 1:
        raise ValueError(f"unsupported GlobalContext version: {version}")
    annotations = example.annotations
    return [
        float(annotations.in_check),
        float(annotations.has_legal_castle),
        float(annotations.has_legal_en_passant),
        float(annotations.has_legal_promotion),
        float(annotations.is_low_material_endgame),
        float(annotations.legal_move_count) / 256.0,
        float(annotations.piece_count) / 32.0,
        sum(1.0 for attacked in own_attacks if attacked) / 64.0,
        sum(1.0 for attacked in opp_attacks if attacked) / 64.0,
    ]


def _candidate_feature_row(
    board: "chess.Board",
    move: "chess.Move",
    own_attacks: list[bool],
    opp_attacks: list[bool],
    *,
    version: int = DEFAULT_CANDIDATE_CONTEXT_VERSION,
) -> list[float]:
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError(f"missing moving piece on {move.uci()}")
    captured_piece = (
        board.piece_at(move.to_square)
        if not board.is_en_passant(move)
        else board.piece_at(move.to_square + (-8 if board.turn else 8))
    )
    piece_one_hot = [0.0] * 6
    piece_one_hot[moving_piece.piece_type - 1] = 1.0
    captured_minor_or_major = (
        1.0
        if captured_piece is not None
        and captured_piece.piece_type
        in {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}
        else 0.0
    )
    if version == 1:
        return [
            float(board.is_capture(move)),
            float(move.promotion is not None),
            float(board.is_castling(move)),
            float(board.is_en_passant(move)),
            float(board.gives_check(move)),
            float(opp_attacks[move.from_square]),
            float(opp_attacks[move.to_square]),
            float(own_attacks[move.from_square]),
            float(own_attacks[move.to_square]),
            *piece_one_hot,
            float(captured_piece is not None),
            float(captured_piece is not None and captured_piece.piece_type == chess.PAWN),
            captured_minor_or_major,
        ]

    if version != 2:
        raise ValueError(f"unsupported CandidateContext version: {version}")

    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    delta_file = (to_file - from_file) / 7.0
    delta_rank = (to_rank - from_rank) / 7.0
    captured_piece_types = {
        chess.PAWN: 0.0,
        chess.KNIGHT: 0.0,
        chess.BISHOP: 0.0,
        chess.ROOK: 0.0,
        chess.QUEEN: 0.0,
    }
    if captured_piece is not None and captured_piece.piece_type in captured_piece_types:
        captured_piece_types[captured_piece.piece_type] = 1.0
    return [
        float(board.is_capture(move)),
        float(move.promotion is not None),
        float(board.is_castling(move)),
        float(board.is_en_passant(move)),
        float(board.gives_check(move)),
        float(opp_attacks[move.from_square]),
        float(opp_attacks[move.to_square]),
        float(own_attacks[move.from_square]),
        float(own_attacks[move.to_square]),
        *piece_one_hot,
        float(captured_piece is not None),
        captured_piece_types[chess.PAWN],
        captured_piece_types[chess.KNIGHT],
        captured_piece_types[chess.BISHOP],
        captured_piece_types[chess.ROOK],
        captured_piece_types[chess.QUEEN],
        float(move.promotion == chess.KNIGHT),
        float(move.promotion == chess.BISHOP),
        float(move.promotion == chess.ROOK),
        float(move.promotion == chess.QUEEN),
        float(board.is_castling(move) and to_file > from_file),
        float(board.is_castling(move) and to_file < from_file),
        from_file / 7.0,
        from_rank / 7.0,
        to_file / 7.0,
        to_rank / 7.0,
        delta_file,
        delta_rank,
        abs(delta_file),
        abs(delta_rank),
    ]


def _attack_map(board: "chess.Board", color: "chess.Color") -> list[bool]:
    attacked = [False] * 64
    for square, piece in board.piece_map().items():
        if piece.color != color:
            continue
        for target in board.attacks(square):
            attacked[target] = True
    return attacked


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


def _optional_float_list(value: object) -> list[float] | None:
    if value is None:
        return None
    return [float(element) for element in list(value)]


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
        action_features = _selected_move_action_features(example)
        dynamics_examples.append(
            DynamicsTrainingExample(
                sample_id=example.sample_id,
                split=example.split,
                feature_vector=pack_position_features(example.position_encoding),
                action_index=flatten_action(example.selected_action_encoding),
                action_candidate_context_version=DEFAULT_CANDIDATE_CONTEXT_VERSION,
                action_features=action_features,
                next_feature_vector=pack_position_features(next_position),
                is_capture=bool(example.annotations.selected_move_is_capture),
                is_promotion=bool(example.annotations.selected_move_is_promotion),
                is_castle=bool(example.annotations.selected_move_is_castle),
                is_en_passant=bool(example.annotations.selected_move_is_en_passant),
                gives_check=bool(example.annotations.selected_move_gives_check),
                trajectory_id=_trajectory_id(example),
                ply_index=_trajectory_ply(example),
                transition_context_version=1,
                transition_features=build_transition_context_features(example, version=1),
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


def _selected_move_action_features(example: DatasetExample) -> list[float]:
    return build_selected_move_action_features(
        example,
        candidate_context_version=DEFAULT_CANDIDATE_CONTEXT_VERSION,
    )


def build_selected_move_action_features(
    example: DatasetExample,
    *,
    candidate_context_version: int,
) -> list[float]:
    """Build versioned selected-move candidate features for dynamics artifacts."""
    if chess is None:  # pragma: no cover - exercised when chess is absent
        raise RuntimeError(
            "python-chess is required for symbolic dynamics artifacts. Install the 'train' extra."
        )
    if example.selected_move_uci is None:
        raise ValueError(f"{example.sample_id}: selected_move_uci is required for dynamics")

    board = chess.Board(example.fen)
    move = chess.Move.from_uci(example.selected_move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"{example.sample_id}: selected move {example.selected_move_uci} is illegal")
    own_attacks = _attack_map(board, board.turn)
    opp_attacks = _attack_map(board, not board.turn)
    return _candidate_feature_row(
        board,
        move,
        own_attacks,
        opp_attacks,
        version=candidate_context_version,
    )


def build_transition_context_features(
    example: DatasetExample,
    *,
    version: int,
) -> list[float]:
    """Build a selected-action transition feature vector from exact pre/post-move state."""
    if version != 1:
        raise ValueError(f"unsupported TransitionContext version: {version}")
    if chess is None:  # pragma: no cover - exercised when chess is absent
        raise RuntimeError(
            "python-chess is required for transition context features. Install the 'train' extra."
        )
    if example.selected_move_uci is None:
        raise ValueError(f"{example.sample_id}: selected_move_uci is required for dynamics")

    board = chess.Board(example.fen)
    move = chess.Move.from_uci(example.selected_move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"{example.sample_id}: selected move {example.selected_move_uci} is illegal")

    own_attacks = _attack_map(board, board.turn)
    opp_attacks = _attack_map(board, not board.turn)
    candidate_features = _candidate_feature_row(
        board,
        move,
        own_attacks,
        opp_attacks,
        version=2,
    )

    next_board = board.copy(stack=False)
    next_board.push(move)
    mover_color = not next_board.turn
    return candidate_features + [
        float(next_board.is_check()),
        float(next_board.is_attacked_by(next_board.turn, move.to_square)),
        float(next_board.is_attacked_by(mover_color, move.to_square)),
        float(next_board.halfmove_clock == 0),
        float(
            board.has_kingside_castling_rights(chess.WHITE)
            and not next_board.has_kingside_castling_rights(chess.WHITE)
        ),
        float(
            board.has_queenside_castling_rights(chess.WHITE)
            and not next_board.has_queenside_castling_rights(chess.WHITE)
        ),
        float(
            board.has_kingside_castling_rights(chess.BLACK)
            and not next_board.has_kingside_castling_rights(chess.BLACK)
        ),
        float(
            board.has_queenside_castling_rights(chess.BLACK)
            and not next_board.has_queenside_castling_rights(chess.BLACK)
        ),
        float(next_board.ep_square is not None),
        float(board.ep_square is not None and next_board.ep_square != board.ep_square),
    ]
