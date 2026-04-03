"""Tests for the unique-corpus to current-artifact pipeline runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_unique_stockfish_dataset_pipeline.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "build_unique_stockfish_dataset_pipeline",
    _SCRIPT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

main = _MODULE.main


def test_snapshot_only_pipeline_exports_and_materializes(
    tmp_path: Path,
    capsys,
) -> None:
    work_dir = tmp_path / "work"
    train_output_dir = tmp_path / "current_train"
    verify_output_dir = tmp_path / "current_verify"

    with (
        patch.object(
            _MODULE,
            "export_unique_corpus_snapshot",
            return_value={
                "train_raw_path": str(work_dir / "train_raw.jsonl"),
                "verify_raw_path": str(work_dir / "verify_raw.jsonl"),
                "train_records": 12,
                "verify_records": 3,
                "counts": {"train": 12, "verify": 3},
                "labeled_counts": {"train": 12, "verify": 3},
            },
        ) as export_mock,
        patch.object(
            _MODULE,
            "_materialize_current_datasets",
            return_value={
                "train_dataset": {"output_dir": str(train_output_dir)},
                "verify_dataset": {"output_dir": str(verify_output_dir)},
            },
        ) as materialize_mock,
    ):
        exit_code = main(
            [
                "--work-dir",
                str(work_dir),
                "--train-output-dir",
                str(train_output_dir),
                "--verify-output-dir",
                str(verify_output_dir),
                "--snapshot-only",
            ]
        )

    assert exit_code == 0
    export_mock.assert_called_once_with(work_dir)
    materialize_mock.assert_called_once()
    payload = json.loads(capsys.readouterr().out)
    assert payload["snapshot_only"] is True
    assert payload["export_summary"]["train_records"] == 12
    assert payload["artifact_summary"]["verify_dataset"]["output_dir"] == str(verify_output_dir)
