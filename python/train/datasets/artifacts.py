"""Dataset artifact loaders and feature packing for proposer training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from train.action_space import flatten_action, flatten_legal_actions
from train.datasets.schema import DatasetExample, PositionEncoding, SUPPORTED_SPLITS

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


@dataclass(frozen=True)
class ProposerTrainingExample:
    """Training-ready proposer example with packed numeric features."""

    sample_id: str
    split: str
    feature_vector: list[float]
    legal_action_indices: list[int]
    selected_action_index: int | None


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
    return [to_proposer_example(example) for example in load_split_examples(dataset_path, split)]


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
