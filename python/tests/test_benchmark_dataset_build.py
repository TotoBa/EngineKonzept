"""Tests for the dataset-build benchmark helper."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from train.datasets.schema import RawPositionRecord


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_dataset_build.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_dataset_build", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_parse_config = _MODULE._parse_config
_select_records = _MODULE._select_records


def test_parse_config_splits_name_workers_and_batch_size() -> None:
    assert _parse_config("auto_w4:4:0") == ("auto_w4", 4, 0)


def test_parse_config_keeps_full_name_prefix() -> None:
    assert _parse_config("phase5-auto:2:1000") == ("phase5-auto", 2, 1000)


def test_select_records_expands_by_cycling_templates() -> None:
    records = [
        RawPositionRecord(sample_id="a", fen="fen-a", source="src"),
        RawPositionRecord(sample_id="b", fen="fen-b", source="src"),
    ]

    selected = _select_records(records, target_count=5)

    assert len(selected) == 5
    assert selected[0].fen == "fen-a"
    assert selected[1].fen == "fen-b"
    assert selected[2].sample_id == "a:bench:2"
    assert selected[3].sample_id == "b:bench:3"
