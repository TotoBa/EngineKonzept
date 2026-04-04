"""Golden tests for the fixed-width encoder feature vector."""

from __future__ import annotations

import json
from pathlib import Path

from train.datasets.artifacts import POSITION_FEATURE_SIZE, pack_position_features
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import PositionEncoding, RawPositionRecord


def test_python_encoder_matches_golden_vectors_exactly() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    golden_path = repo_root / "artifacts" / "golden" / "encoder_golden_v1.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    assert golden["version"] == 1
    examples = list(golden["examples"])
    assert len(examples) == 10

    records = [
        RawPositionRecord(
            sample_id=f"golden_{index}",
            fen=str(example["fen"]),
            source="golden",
        )
        for index, example in enumerate(examples)
    ]
    payloads = label_records_with_oracle(records, repo_root=repo_root)
    for example, payload in zip(examples, payloads, strict=True):
        encoding = PositionEncoding.from_oracle_dict(dict(payload["position_encoding"]))
        actual = pack_position_features(encoding)
        expected = [float(value) for value in list(example["features"])]
        assert len(expected) == POSITION_FEATURE_SIZE
        assert actual == expected


def test_encoder_golden_file_is_well_formed() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    golden_path = repo_root / "artifacts" / "golden" / "encoder_golden_v1.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    assert golden["version"] == 1
    examples = list(golden["examples"])
    assert len(examples) == 10
    fens = set()
    for example in examples:
        fen = str(example["fen"])
        features = list(example["features"])
        assert fen not in fens
        fens.add(fen)
        assert len(features) == POSITION_FEATURE_SIZE
