"""Tests for the resumable unique PGN/Stockfish corpus builder."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
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

MySQLConfig = _MODULE.MySQLConfig
UniqueCorpusConfig = _MODULE.UniqueCorpusConfig
build_unique_corpus_from_pgns = _MODULE.build_unique_corpus_from_pgns
export_unique_corpus_snapshot = _MODULE.export_unique_corpus_snapshot
main = _MODULE.main
select_pgn_file_shard = _MODULE.select_pgn_file_shard


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


@dataclass(frozen=True)
class _MemoryReservedSample:
    fen_hash: str
    fen: str
    split: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _MemoryLabeledSample:
    sample_id: str
    fen: str
    split: str
    source: str
    result: str | None
    metadata: dict[str, Any]
    selected_move_uci: str


class _MemoryLedger:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, dict[str, Any]]] = {}
        self.schema_ensured = 0

    def ensure_schema(self) -> None:
        self.schema_ensured += 1

    def split_counts(self, namespace: str, *, labeled_only: bool = False) -> dict[str, int]:
        counts = {"train": 0, "verify": 0}
        for row in self._rows_for(namespace).values():
            if labeled_only and row["status"] != "labeled":
                continue
            if not labeled_only and row["status"] not in {"reserved", "labeled"}:
                continue
            counts[str(row["split"])] += 1
        return counts

    def reserve_sample(
        self,
        namespace: str,
        *,
        fen_hash: str,
        sample_id: str,
        fen: str,
        split: str,
        source: str,
        result: str | None,
        metadata: dict[str, Any],
    ) -> bool:
        rows = self._rows_for(namespace)
        if fen_hash in rows:
            return False
        rows[fen_hash] = {
            "sample_id": sample_id,
            "fen": fen,
            "split": split,
            "source": source,
            "result": result,
            "metadata": dict(metadata),
            "selected_move_uci": None,
            "status": "reserved",
        }
        return True

    def iter_reserved_samples(self, namespace: str):
        for fen_hash in sorted(self._rows_for(namespace)):
            row = self._rows_for(namespace)[fen_hash]
            if row["status"] != "reserved":
                continue
            yield _MemoryReservedSample(
                fen_hash=fen_hash,
                fen=str(row["fen"]),
                split=str(row["split"]),
                metadata=dict(row["metadata"]),
            )

    def load_reserved_sample(self, namespace: str, *, fen_hash: str) -> _MemoryReservedSample | None:
        row = self._rows_for(namespace).get(fen_hash)
        if row is None or row["status"] != "reserved":
            return None
        return _MemoryReservedSample(
            fen_hash=fen_hash,
            fen=str(row["fen"]),
            split=str(row["split"]),
            metadata=dict(row["metadata"]),
        )

    def delete_reserved_sample(self, namespace: str, *, fen_hash: str) -> None:
        row = self._rows_for(namespace).get(fen_hash)
        if row is None or row["status"] != "reserved":
            return
        self._rows_for(namespace).pop(fen_hash, None)

    def mark_sample_labeled(
        self,
        namespace: str,
        *,
        fen_hash: str,
        selected_move_uci: str,
        metadata: dict[str, Any],
    ) -> bool:
        row = self._rows_for(namespace).get(fen_hash)
        if row is None or row["status"] != "reserved":
            return False
        row["selected_move_uci"] = selected_move_uci
        row["metadata"] = dict(metadata)
        row["status"] = "labeled"
        return True

    def iter_labeled_samples(self, namespace: str, *, split: str):
        for fen_hash in sorted(self._rows_for(namespace)):
            row = self._rows_for(namespace)[fen_hash]
            if row["status"] != "labeled" or row["split"] != split:
                continue
            selected_move_uci = row["selected_move_uci"]
            assert selected_move_uci is not None
            yield _MemoryLabeledSample(
                sample_id=str(row["sample_id"]),
                fen=str(row["fen"]),
                split=str(row["split"]),
                source=str(row["source"]),
                result=(str(row["result"]) if row["result"] is not None else None),
                metadata=dict(row["metadata"]),
                selected_move_uci=str(selected_move_uci),
            )

    def close(self) -> None:
        return None

    def _rows_for(self, namespace: str) -> dict[str, dict[str, Any]]:
        return self._rows.setdefault(namespace, {})


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
        db_config=MySQLConfig(
            host="localhost",
            user="user",
            password="password",
            database="chesstrainer",
        ),
    )
    ledger = _MemoryLedger()
    with patch.object(_MODULE.chess.engine.SimpleEngine, "popen_uci", return_value=_FakeEngine()):
        summary = build_unique_corpus_from_pgns([pgn_path], config=config, ledger=ledger)

    assert ledger.schema_ensured == 1
    assert summary["completed"] is True
    assert summary["counts"] == {"train": 2, "verify": 1}
    assert summary["labeled_counts"] == {"train": 2, "verify": 1}
    assert summary["ledger_backend"] == "mysql"
    assert summary["ledger_namespace"] == str(config.work_dir)

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
    assert not (config.work_dir / "corpus.sqlite3").exists()


def test_resume_reserved_rows_labels_existing_entries(tmp_path: Path) -> None:
    ledger = _MemoryLedger()
    namespace = "resume-namespace"
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    reserved = ledger.reserve_sample(
        namespace,
        fen_hash=_MODULE._fen_hash_hex(fen),
        sample_id="resume:start",
        fen=fen,
        split="train",
        source="test",
        result="1-0",
        metadata={"played_move_uci": "e2e4"},
    )
    assert reserved is True

    config = UniqueCorpusConfig(
        engine_path=Path("/usr/games/stockfish18"),
        work_dir=tmp_path / "work",
        target_train_records=1,
        target_verify_records=0,
    )
    _MODULE._resume_reserved_rows(ledger, namespace=namespace, engine=_FakeEngine(), config=config)

    labeled_rows = list(ledger.iter_labeled_samples(namespace, split="train"))
    assert len(labeled_rows) == 1
    assert labeled_rows[0].selected_move_uci is not None
    assert labeled_rows[0].metadata["label_source"] == "stockfish18"
    assert labeled_rows[0].metadata["stockfish_nodes"] == 1500


def test_export_unique_corpus_snapshot_writes_raw_jsonl(tmp_path: Path) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    ledger = _MemoryLedger()
    namespace = str(work_dir)
    ledger.reserve_sample(
        namespace,
        fen_hash="train-hash",
        sample_id="train:1",
        fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        split="train",
        source="test",
        result="1-0",
        metadata={"played_move_uci": "e1e2"},
    )
    ledger.mark_sample_labeled(
        namespace,
        fen_hash="train-hash",
        selected_move_uci="e1e2",
        metadata={"played_move_uci": "e1e2"},
    )
    ledger.reserve_sample(
        namespace,
        fen_hash="verify-hash",
        sample_id="verify:1",
        fen="4k3/8/8/8/8/8/8/3K4 w - - 0 1",
        split="verify",
        source="test",
        result="0-1",
        metadata={"played_move_uci": "d1d2"},
    )
    ledger.mark_sample_labeled(
        namespace,
        fen_hash="verify-hash",
        selected_move_uci="d1d2",
        metadata={"played_move_uci": "d1d2"},
    )

    summary = export_unique_corpus_snapshot(work_dir, ledger=ledger)

    assert summary["counts"] == {"train": 1, "verify": 1}
    assert summary["labeled_counts"] == {"train": 1, "verify": 1}
    train_rows = (work_dir / "train_raw.jsonl").read_text(encoding="utf-8").splitlines()
    verify_rows = (work_dir / "verify_raw.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train_rows) == 1
    assert len(verify_rows) == 1
    assert not (work_dir / "corpus.sqlite3").exists()


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

    def fake_build(pgn_paths: list[Path], *, config: Any, ledger: Any | None = None) -> dict[str, Any]:
        del config, ledger
        seen_paths.extend(pgn_paths)
        return {"ok": True, "work_dir": str(tmp_path / "work")}

    with (
        patch.object(_MODULE.MySQLConfig, "from_env", return_value=MySQLConfig("h", "d", "u", "p")),
        patch.object(_MODULE, "build_unique_corpus_from_pgns", side_effect=fake_build),
    ):
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
    ledger = _MemoryLedger()
    with patch.object(_MODULE.chess.engine.SimpleEngine, "popen_uci", return_value=_FakeEngine()):
        summary = build_unique_corpus_from_pgns([pgn_path], config=config, ledger=ledger)

    assert summary["completed"] is True
    assert summary["completion_reason"] == "eof"
    assert summary["counts"]["train"] > 0
    assert (config.work_dir / "train_raw.jsonl").exists()


def test_select_pgn_file_shard_returns_deterministic_subset(tmp_path: Path) -> None:
    pgn_paths = [tmp_path / f"sample_{index}.pgn" for index in range(1, 7)]

    selected = select_pgn_file_shard(pgn_paths, shard_index=2, shard_count=3)

    assert selected == [pgn_paths[1], pgn_paths[4]]


def test_build_unique_corpus_run_max_games_stops_one_invocation(tmp_path: Path) -> None:
    pgn_path = tmp_path / "sample.pgn"
    pgn_path.write_text(
        """
[Event "Game 1"]
[Site "?"]
[Date "2026.04.11"]
[Round "1"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 1-0

[Event "Game 2"]
[Site "?"]
[Date "2026.04.11"]
[Round "2"]
[White "C"]
[Black "D"]
[Result "0-1"]

1. d4 d5 2. c4 e6 0-1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = UniqueCorpusConfig(
        engine_path=Path("/usr/games/stockfish18"),
        work_dir=tmp_path / "work",
        target_train_records=10,
        target_verify_records=5,
        min_ply=1,
        max_ply=4,
        ply_stride=1,
        progress_every=1,
        run_max_games=1,
    )
    ledger = _MemoryLedger()
    with patch.object(_MODULE.chess.engine.SimpleEngine, "popen_uci", return_value=_FakeEngine()):
        summary = build_unique_corpus_from_pgns([pgn_path], config=config, ledger=ledger)

    assert summary["completed"] is False
    assert summary["completion_reason"] == "targets_not_reached"
    assert summary["games_seen"] == 1
