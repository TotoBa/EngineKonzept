"""Tests for the dataset-build benchmark helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from train.datasets.schema import RawPositionRecord


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_dataset_build.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_dataset_build", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_parse_config = _MODULE._parse_config
_select_records = _MODULE._select_records
main = _MODULE.main


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


def test_main_creates_artifact_parent_directory(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "fen": "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
                "source": "synthetic",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "out"
    artifact_out = tmp_path / "nested" / "artifact.json"

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--source-format",
            "jsonl",
            "--output-root",
            str(output_root),
            "--artifact-out",
            str(artifact_out),
            "--config",
            "single:1:0",
        ]
    )

    assert exit_code == 0
    assert artifact_out.exists()
    rendered = json.loads(artifact_out.read_text(encoding="utf-8"))
    assert rendered["runtime"]["hostname"]
    assert rendered["runtime"]["python_version"]
