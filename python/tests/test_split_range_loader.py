from __future__ import annotations

import json
from pathlib import Path

from train.datasets import load_split_examples_range
from train.datasets.schema import DatasetExample, PositionEncoding, TacticalAnnotations


def _make_example(*, sample_id: str, split: str) -> DatasetExample:
    return DatasetExample(
        sample_id=sample_id,
        split=split,
        source="split_range_test",
        fen="8/8/8/8/8/8/8/8 w - - 0 1",
        side_to_move="w",
        selected_move_uci=None,
        selected_action_encoding=None,
        next_fen=None,
        legal_moves=["a2a3"],
        legal_action_encodings=[[0, 0, 0]],
        position_encoding=PositionEncoding(
            piece_tokens=[],
            square_tokens=[[index, 0] for index in range(64)],
            rule_token=[0, 0, -1, 0, 1, 0],
        ),
        wdl_target=None,
        annotations=TacticalAnnotations(
            in_check=False,
            is_checkmate=False,
            is_stalemate=False,
            has_legal_en_passant=False,
            has_legal_castle=False,
            has_legal_promotion=False,
            is_low_material_endgame=False,
            legal_move_count=1,
            piece_count=2,
            selected_move_is_capture=None,
            selected_move_is_promotion=None,
            selected_move_is_castle=None,
            selected_move_is_en_passant=None,
            selected_move_gives_check=None,
        ),
        result=None,
        metadata={},
    )


def test_load_split_examples_range_uses_split_jsonl_slice(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    train_examples = [
        _make_example(sample_id=f"train_{index}", split="train")
        for index in range(6)
    ]
    (dataset_dir / "train.jsonl").write_text(
        "".join(json.dumps(example.to_dict(), sort_keys=True) + "\n" for example in train_examples),
        encoding="utf-8",
    )

    selected = load_split_examples_range(dataset_dir, "train", start_index=2, max_examples=3)

    assert [example.sample_id for example in selected] == ["train_2", "train_3", "train_4"]


def test_load_split_examples_range_accepts_open_ended_slice(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    test_examples = [
        _make_example(sample_id=f"test_{index}", split="test")
        for index in range(4)
    ]
    (dataset_dir / "test.jsonl").write_text(
        "".join(json.dumps(example.to_dict(), sort_keys=True) + "\n" for example in test_examples),
        encoding="utf-8",
    )

    selected = load_split_examples_range(dataset_dir, "test", start_index=1)

    assert [example.sample_id for example in selected] == ["test_1", "test_2", "test_3"]
