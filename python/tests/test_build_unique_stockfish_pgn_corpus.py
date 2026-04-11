"""Tests for the resumable unique PGN/Stockfish corpus builder."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path
import sys
import tempfile
from typing import Any
from unittest.mock import patch

import chess


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "build_unique_stockfish_pgn_corpus.py"
)
_SPEC = importlib.util.spec_from_file_location("build_unique_stockfish_pgn_corpus", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

UniqueCorpusConfig = _MODULE.UniqueCorpusConfig
build_unique_corpus_from_pgns = _MODULE.build_unique_corpus_from_pgns
export_unique_corpus_snapshot = _MODULE.export_unique_corpus_snapshot
main = _MODULE.main


class _FakePlayResult:
    def __init__(self, move: chess.Move) -> None:
        self.move = move


class _FakeEngine:
    def configure(self, _: dict[str, Any]) -> None:
        return None

    def play(self, board: chess.Board, _: chess.engine.Limit) -> _FakePlayResult:
        legal_move = next(iter(board.legal_moves))
        return _FakePlayResult(legal_move)

    def quit(self) -> None:
        return None


def test_build_unique_corpus_deduplicates_and_exports_disjoint_splits(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(
        """
[Event "Game 1"]
[Site "?"]
[Date "2026.04.02"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Game 2"]
[Site "?"]
[Date "2026.04.02"]
[Round "2"]
[White "C"]
[Black "D"]
[Result "0-1"]

1. e4 e5 2. Nf3 Nc6 0-1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = UniqueCorpusConfig(
        engine_path=Path("/usr/games/stockfish18"),
        work_dir=tmp_path / "work",
        target_train_records=2,
        target_verify_records=1,
        min_ply=1,
        max_ply=4,
        ply_stride=1,
        verify_divisor=1,
        progress_every=1,
    )

    real_connect = sqlite3.connect
    with tempfile.TemporaryDirectory() as database_tmpdir:
        database_path = Path(database_tmpdir) / "corpus.sqlite3"
        with (
            patch.object(_MODULE.chess.engine.SimpleEngine, "popen_uci", return_value=_FakeEngine()),
            patch.object(_MODULE.sqlite3, "connect", side_effect=lambda _: real_connect(database_path)),
        ):
            summary = build_unique_corpus_from_pgns([pgn_path], config=config)

    assert summary["completed"] is True
    assert summary["counts"] == {"train": 2, "verify": 1}
    assert summary["labeled_counts"] == {"train": 2, "verify": 1}

    train_rows = [
        json.loads(line)
        for line in (config.work_dir / "train_raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    verify_rows = [
        json.loads(line)
        for line in (config.work_dir / "verify_raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    train_fens = {row["fen"] for row in train_rows}
    verify_fens = {row["fen"] for row in verify_rows}
    assert len(train_rows) == 2
    assert len(verify_rows) == 1
    assert train_fens.isdisjoint(verify_fens)
    assert len(train_fens | verify_fens) == 3
    assert all("stockfish_bestmove_uci" in row["metadata"] for row in train_rows + verify_rows)


def test_resume_reserved_rows_labels_existing_entries(tmp_path: Path) -> None:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    _MODULE._initialize_database(connection)
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    reserved = _MODULE._reserve_sample(
        connection,
        sample_id="resume:start",
        fen=fen,
        split="train",
        source="test",
        result="1-0",
        metadata={"played_move_uci": "e2e4"},
    )
    assert reserved is True
    connection.commit()

    config = UniqueCorpusConfig(
        engine_path=Path("/usr/games/stockfish18"),
        work_dir=tmp_path,
        target_train_records=1,
        target_verify_records=0,
    )
    _MODULE._resume_reserved_rows(connection, engine=_FakeEngine(), config=config)

    row = connection.execute(
        """
        SELECT selected_move_uci, status, metadata_json
        FROM corpus_samples
        WHERE sample_id = 'resume:start'
        """
    ).fetchone()
    connection.close()

    assert row is not None
    assert row["status"] == "labeled"
    assert row["selected_move_uci"] is not None
    metadata = json.loads(str(row["metadata_json"]))
    assert metadata["label_source"] == "stockfish18"
    assert metadata["stockfish_nodes"] == 1500


def test_export_unique_corpus_snapshot_writes_raw_jsonl(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    connection = sqlite3.connect(work_dir / "corpus.sqlite3")
    connection.row_factory = sqlite3.Row
    _MODULE._initialize_database(connection)
    _MODULE._reserve_sample(
        connection,
        sample_id="train:1",
        fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        split="train",
        source="test",
        result="1-0",
        metadata={"played_move_uci": "e1e2"},
    )
    _MODULE._reserve_sample(
        connection,
        sample_id="verify:1",
        fen="4k3/8/8/8/8/8/8/3K4 w - - 0 1",
        split="verify",
        source="test",
        result="0-1",
        metadata={"played_move_uci": "d1d2"},
    )
    connection.execute(
        """
        UPDATE corpus_samples
        SET selected_move_uci = 'e1e2', status = 'labeled'
        WHERE sample_id = 'train:1'
        """
    )
    connection.execute(
        """
        UPDATE corpus_samples
        SET selected_move_uci = 'd1d2', status = 'labeled'
        WHERE sample_id = 'verify:1'
        """
    )
    connection.commit()
    connection.close()

    summary = export_unique_corpus_snapshot(work_dir)

    assert summary["counts"] == {"train": 1, "verify": 1}
    assert summary["labeled_counts"] == {"train": 1, "verify": 1}
    train_rows = (work_dir / "train_raw.jsonl").read_text(encoding="utf-8").splitlines()
    verify_rows = (work_dir / "verify_raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train_rows) == 1
    assert len(verify_rows) == 1


def test_main_scans_nested_pgns_by_default(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    pgn_path = nested / "sample.pgn"
    pgn_path.write_text(
        """
[Event "Nested"]
[Site "?"]
[Date "2026.04.02"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 1-0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    seen_paths: list[Path] = []

    def fake_build(pgn_paths: list[Path], *, config: Any) -> dict[str, Any]:
        seen_paths.extend(pgn_paths)
        return {"ok": True, "work_dir": str(config.work_dir)}

    with patch.object(_MODULE, "build_unique_corpus_from_pgns", side_effect=fake_build):
        exit_code = main(["--pgn-root", str(tmp_path), "--work-dir", str(tmp_path / "work")])

    assert exit_code == 0
    assert seen_paths == [pgn_path]


def test_build_unique_corpus_can_complete_at_eof(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(
        """
[Event "Short"]
[Site "?"]
[Date "2026.04.10"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 1-0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = UniqueCorpusConfig(
        engine_path=Path("/usr/games/stockfish18"),
        work_dir=tmp_path / "work",
        target_train_records=100,
        target_verify_records=10,
        min_ply=1,
        max_ply=2,
        ply_stride=1,
        progress_every=1,
        complete_at_eof=True,
    )

    real_connect = sqlite3.connect
    with tempfile.TemporaryDirectory() as database_tmpdir:
        database_path = Path(database_tmpdir) / "corpus.sqlite3"
        with (
            patch.object(_MODULE.chess.engine.SimpleEngine, "popen_uci", return_value=_FakeEngine()),
            patch.object(_MODULE.sqlite3, "connect", side_effect=lambda _: real_connect(database_path)),
        ):
            summary = build_unique_corpus_from_pgns([pgn_path], config=config)

    assert summary["completed"] is True
    assert summary["completion_reason"] == "eof"
    assert summary["counts"]["train"] > 0
    assert (config.work_dir / "train_raw.jsonl").exists()
