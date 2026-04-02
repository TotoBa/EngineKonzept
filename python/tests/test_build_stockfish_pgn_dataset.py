"""Tests for the PGN/Stockfish dataset builder CLI defaults."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_stockfish_pgn_dataset.py"
)
_SPEC = importlib.util.spec_from_file_location("build_stockfish_pgn_dataset", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
main = _MODULE.main


def test_main_writes_proposer_artifacts_by_default(tmp_path: Path) -> None:
    fake_selection = SimpleNamespace(
        train_records=[],
        verify_records=[],
        summary={"ok": True},
    )
    fake_dataset = SimpleNamespace(summary={"ok": True})
    seen_flags: list[bool] = []

    def fake_write_dataset_artifacts(*_: object, write_proposer_artifacts: bool, **__: object) -> None:
        seen_flags.append(write_proposer_artifacts)

    with (
        patch.object(_MODULE, "sample_policy_records_from_pgns", return_value=fake_selection),
        patch.object(_MODULE, "build_dataset", return_value=fake_dataset),
        patch.object(_MODULE, "write_dataset_artifacts", side_effect=fake_write_dataset_artifacts),
    ):
        exit_code = main(
            [
                "--pgn",
                str(tmp_path / "sample.pgn"),
                "--train-output-dir",
                str(tmp_path / "train"),
                "--verify-output-dir",
                str(tmp_path / "verify"),
            ]
        )

    assert exit_code == 0
    assert seen_flags == [True, True]


def test_main_can_disable_proposer_artifacts_explicitly(tmp_path: Path) -> None:
    fake_selection = SimpleNamespace(
        train_records=[],
        verify_records=[],
        summary={"ok": True},
    )
    fake_dataset = SimpleNamespace(summary={"ok": True})
    seen_flags: list[bool] = []

    def fake_write_dataset_artifacts(*_: object, write_proposer_artifacts: bool, **__: object) -> None:
        seen_flags.append(write_proposer_artifacts)

    with (
        patch.object(_MODULE, "sample_policy_records_from_pgns", return_value=fake_selection),
        patch.object(_MODULE, "build_dataset", return_value=fake_dataset),
        patch.object(_MODULE, "write_dataset_artifacts", side_effect=fake_write_dataset_artifacts),
    ):
        exit_code = main(
            [
                "--pgn",
                str(tmp_path / "sample.pgn"),
                "--train-output-dir",
                str(tmp_path / "train"),
                "--verify-output-dir",
                str(tmp_path / "verify"),
                "--no-proposer-artifacts",
            ]
        )

    assert exit_code == 0
    assert seen_flags == [False, False]
