"""Build a resumable unique Stockfish-labeled PGN corpus for large Phase-5 runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Sequence

import chess
import chess.engine
import chess.pgn

from train.datasets.schema import RawPositionRecord


@dataclass(frozen=True)
class UniqueCorpusConfig:
    """Controls large-scale unique PGN sampling and Stockfish labeling."""

    engine_path: Path
    work_dir: Path
    target_train_records: int
    target_verify_records: int
    min_ply: int = 8
    max_ply: int = 80
    ply_stride: int = 2
    engine_nodes: int = 1500
    hash_mb: int = 32
    threads: int = 1
    split_seed: str = "phase5-stockfish-unique-v1"
    verify_divisor: int = 1000
    progress_every: int = 1000
    max_games: int = 0
    export_jsonl_on_complete: bool = True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn-root", type=Path, required=True)
    parser.add_argument("--glob", default="**/*.pgn")
    parser.add_argument("--engine-path", type=Path, default=Path("/usr/games/stockfish18"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--target-train-records", type=int, default=10_000_000)
    parser.add_argument("--target-verify-records", type=int, default=10_000)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=80)
    parser.add_argument("--ply-stride", type=int, default=2)
    parser.add_argument("--engine-nodes", type=int, default=1500)
    parser.add_argument("--hash-mb", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--split-seed", default="phase5-stockfish-unique-v1")
    parser.add_argument("--verify-divisor", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument(
        "--no-export-jsonl-on-complete",
        action="store_false",
        dest="export_jsonl_on_complete",
    )
    args = parser.parse_args(argv)

    config = UniqueCorpusConfig(
        engine_path=args.engine_path,
        work_dir=args.work_dir,
        target_train_records=args.target_train_records,
        target_verify_records=args.target_verify_records,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        ply_stride=args.ply_stride,
        engine_nodes=args.engine_nodes,
        hash_mb=args.hash_mb,
        threads=args.threads,
        split_seed=args.split_seed,
        verify_divisor=args.verify_divisor,
        progress_every=args.progress_every,
        max_games=args.max_games,
        export_jsonl_on_complete=args.export_jsonl_on_complete,
    )

    pgn_paths = sorted(path for path in args.pgn_root.glob(args.glob) if path.is_file())
    if not pgn_paths:
        raise ValueError(f"no PGNs matched {args.glob!r} under {args.pgn_root}")

    summary = build_unique_corpus_from_pgns(pgn_paths, config=config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_unique_corpus_from_pgns(
    pgn_paths: Sequence[Path],
    *,
    config: UniqueCorpusConfig,
) -> dict[str, Any]:
    """Stream PGNs into a resumable unique sqlite corpus and optional JSONL export."""
    config.work_dir.mkdir(parents=True, exist_ok=True)
    database_path = config.work_dir / "corpus.sqlite3"
    progress_path = config.work_dir / "progress.json"

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    _initialize_database(connection)

    engine = chess.engine.SimpleEngine.popen_uci(str(config.engine_path))
    try:
        engine.configure({"Hash": config.hash_mb, "Threads": config.threads})
        _resume_reserved_rows(connection, engine=engine, config=config)

        progress = _current_progress(
            connection,
            config=config,
            pgn_paths=pgn_paths,
            current_pgn=None,
            games_seen=0,
        )
        _write_progress(progress_path, progress)

        games_seen = int(progress["games_seen"])
        accepted_since_progress = 0
        for pgn_path in pgn_paths:
            if _targets_reached(connection, config=config):
                break
            with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
                while True:
                    if config.max_games > 0 and games_seen >= config.max_games:
                        break
                    if _targets_reached(connection, config=config):
                        break
                    game = chess.pgn.read_game(handle)
                    if game is None:
                        break
                    games_seen += 1
                    board = game.board()
                    result = _normalize_result(game.headers.get("Result"))
                    for ply_index, move in enumerate(game.mainline_moves(), start=1):
                        if config.min_ply <= ply_index <= config.max_ply and (
                            (ply_index - config.min_ply) % config.ply_stride == 0
                        ):
                            if board.is_valid() and not board.is_game_over(claim_draw=True):
                                fen = board.fen()
                                split = _choose_split(
                                    fen,
                                    connection=connection,
                                    config=config,
                                )
                                if split is not None:
                                    sample_id = (
                                        f"stockfish-unique:{pgn_path.stem}:{games_seen}:{ply_index}:"
                                        f"{_fen_hash_hex(fen)[:12]}"
                                    )
                                    reserved = _reserve_sample(
                                        connection,
                                        sample_id=sample_id,
                                        fen=fen,
                                        split=split,
                                        source="stockfish-unique-pgn",
                                        result=result,
                                        metadata={
                                            "source_pgn": pgn_path.stem,
                                            "game_index": games_seen,
                                            "ply": ply_index,
                                            "event": game.headers.get("Event"),
                                            "white": game.headers.get("White"),
                                            "black": game.headers.get("Black"),
                                            "played_move_uci": move.uci(),
                                        },
                                    )
                                    if reserved:
                                        _label_reserved_sample(
                                            connection,
                                            engine=engine,
                                            fen_hash_hex=_fen_hash_hex(fen),
                                            config=config,
                                        )
                                        accepted_since_progress += 1
                                        if accepted_since_progress >= config.progress_every:
                                            accepted_since_progress = 0
                                            _write_progress(
                                                progress_path,
                                                _current_progress(
                                                    connection,
                                                    config=config,
                                                    pgn_paths=pgn_paths,
                                                    current_pgn=str(pgn_path),
                                                    games_seen=games_seen,
                                                ),
                                            )
                        board.push(move)
        final_summary = _current_progress(
            connection,
            config=config,
            pgn_paths=pgn_paths,
            current_pgn=None,
            games_seen=games_seen,
        )
        final_summary["completed"] = _targets_reached(connection, config=config)
        if final_summary["completed"] and config.export_jsonl_on_complete:
            export_summary = _export_jsonl(connection, work_dir=config.work_dir)
            final_summary["export"] = export_summary
        _write_progress(progress_path, final_summary)
        return final_summary
    finally:
        engine.quit()
        connection.close()


def export_unique_corpus_snapshot(work_dir: Path) -> dict[str, Any]:
    """Export the currently labeled unique-corpus rows as raw JSONL artifacts."""
    database_path = work_dir / "corpus.sqlite3"
    if not database_path.exists():
        raise FileNotFoundError(f"unique corpus database not found: {database_path}")

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        export_summary = _export_jsonl(connection, work_dir=work_dir)
        export_summary["counts"] = _split_counts(connection)
        export_summary["labeled_counts"] = _labeled_counts(connection)
        return export_summary
    finally:
        connection.close()


def _initialize_database(connection: sqlite3.Connection) -> None:
    for pragma in (
        "PRAGMA journal_mode = WAL",
        "PRAGMA synchronous = NORMAL",
        "PRAGMA temp_store = MEMORY",
    ):
        try:
            connection.execute(pragma)
        except sqlite3.OperationalError:
            continue
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS corpus_samples (
            fen_hash TEXT PRIMARY KEY,
            fen TEXT NOT NULL,
            split TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            source TEXT NOT NULL,
            result TEXT,
            metadata_json TEXT NOT NULL,
            selected_move_uci TEXT,
            status TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS corpus_samples_split_status
        ON corpus_samples (split, status);
        """
    )
    connection.commit()


def _targets_reached(connection: sqlite3.Connection, *, config: UniqueCorpusConfig) -> bool:
    counts = _split_counts(connection)
    return (
        counts["train"] >= config.target_train_records
        and counts["verify"] >= config.target_verify_records
    )


def _split_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT split, COUNT(*) AS count
        FROM corpus_samples
        WHERE status IN ('reserved', 'labeled')
        GROUP BY split
        """
    ).fetchall()
    counts = {"train": 0, "verify": 0}
    for row in rows:
        counts[str(row["split"])] = int(row["count"])
    return counts


def _labeled_counts(connection: sqlite3.Connection) -> dict[str, int]:
    rows = connection.execute(
        """
        SELECT split, COUNT(*) AS count
        FROM corpus_samples
        WHERE status = 'labeled'
        GROUP BY split
        """
    ).fetchall()
    counts = {"train": 0, "verify": 0}
    for row in rows:
        counts[str(row["split"])] = int(row["count"])
    return counts


def _choose_split(
    fen: str,
    *,
    connection: sqlite3.Connection,
    config: UniqueCorpusConfig,
) -> str | None:
    counts = _split_counts(connection)
    if counts["train"] >= config.target_train_records and counts["verify"] >= config.target_verify_records:
        return None

    prefer_verify = _stable_hash(f"{config.split_seed}:{fen}") % config.verify_divisor == 0
    if prefer_verify and counts["verify"] < config.target_verify_records:
        return "verify"
    if counts["train"] < config.target_train_records:
        return "train"
    if counts["verify"] < config.target_verify_records:
        return "verify"
    return None


def _reserve_sample(
    connection: sqlite3.Connection,
    *,
    sample_id: str,
    fen: str,
    split: str,
    source: str,
    result: str | None,
    metadata: dict[str, Any],
) -> bool:
    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO corpus_samples (
            fen_hash,
            fen,
            split,
            sample_id,
            source,
            result,
            metadata_json,
            selected_move_uci,
            status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'reserved')
        """,
        (
            _fen_hash_hex(fen),
            fen,
            split,
            sample_id,
            source,
            result,
            json.dumps(metadata, sort_keys=True),
        ),
    )
    return cursor.rowcount > 0


def _resume_reserved_rows(
    connection: sqlite3.Connection,
    *,
    engine: chess.engine.SimpleEngine,
    config: UniqueCorpusConfig,
) -> None:
    rows = connection.execute(
        """
        SELECT fen_hash
        FROM corpus_samples
        WHERE status = 'reserved'
        ORDER BY rowid
        """
    ).fetchall()
    for row in rows:
        _label_reserved_sample(
            connection,
            engine=engine,
            fen_hash_hex=str(row["fen_hash"]),
            config=config,
        )


def _label_reserved_sample(
    connection: sqlite3.Connection,
    *,
    engine: chess.engine.SimpleEngine,
    fen_hash_hex: str,
    config: UniqueCorpusConfig,
) -> None:
    row = connection.execute(
        """
        SELECT fen, metadata_json
        FROM corpus_samples
        WHERE fen_hash = ? AND status = 'reserved'
        """,
        (fen_hash_hex,),
    ).fetchone()
    if row is None:
        return

    board = chess.Board(str(row["fen"]))
    result = engine.play(board, chess.engine.Limit(nodes=config.engine_nodes))
    if result.move is None:
        connection.execute(
            "DELETE FROM corpus_samples WHERE fen_hash = ? AND status = 'reserved'",
            (fen_hash_hex,),
        )
        connection.commit()
        return

    metadata = json.loads(str(row["metadata_json"]))
    metadata["label_source"] = "stockfish18"
    metadata["stockfish_nodes"] = config.engine_nodes
    metadata["stockfish_bestmove_uci"] = result.move.uci()
    metadata["stockfish_matches_played"] = metadata.get("played_move_uci") == result.move.uci()

    connection.execute(
        """
        UPDATE corpus_samples
        SET selected_move_uci = ?, metadata_json = ?, status = 'labeled'
        WHERE fen_hash = ? AND status = 'reserved'
        """,
        (
            result.move.uci(),
            json.dumps(metadata, sort_keys=True),
            fen_hash_hex,
        ),
    )
    connection.commit()


def _current_progress(
    connection: sqlite3.Connection,
    *,
    config: UniqueCorpusConfig,
    pgn_paths: Sequence[Path],
    current_pgn: str | None,
    games_seen: int,
) -> dict[str, Any]:
    counts = _split_counts(connection)
    labeled_counts = _labeled_counts(connection)
    return {
        "engine_path": str(config.engine_path),
        "engine_nodes": config.engine_nodes,
        "hash_mb": config.hash_mb,
        "threads": config.threads,
        "split_seed": config.split_seed,
        "verify_divisor": config.verify_divisor,
        "pgn_files": [str(path) for path in pgn_paths],
        "current_pgn": current_pgn,
        "games_seen": games_seen,
        "targets": {
            "train": config.target_train_records,
            "verify": config.target_verify_records,
        },
        "counts": counts,
        "labeled_counts": labeled_counts,
        "sampling": {
            "min_ply": config.min_ply,
            "max_ply": config.max_ply,
            "ply_stride": config.ply_stride,
        },
        "timestamp": int(time.time()),
    }


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _export_jsonl(connection: sqlite3.Connection, *, work_dir: Path) -> dict[str, Any]:
    train_path = work_dir / "train_raw.jsonl"
    verify_path = work_dir / "verify_raw.jsonl"
    train_count = _write_split_jsonl(connection, split="train", output_path=train_path)
    verify_count = _write_split_jsonl(connection, split="verify", output_path=verify_path)
    return {
        "train_raw_path": str(train_path),
        "verify_raw_path": str(verify_path),
        "train_records": train_count,
        "verify_records": verify_count,
    }


def _write_split_jsonl(
    connection: sqlite3.Connection,
    *,
    split: str,
    output_path: Path,
) -> int:
    rows = connection.execute(
        """
        SELECT sample_id, fen, source, selected_move_uci, result, metadata_json
        FROM corpus_samples
        WHERE split = ? AND status = 'labeled'
        ORDER BY rowid
        """,
        (split,),
    ).fetchall()
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = RawPositionRecord(
                sample_id=str(row["sample_id"]),
                fen=str(row["fen"]),
                source=str(row["source"]),
                selected_move_uci=str(row["selected_move_uci"]),
                result=None if row["result"] is None else str(row["result"]),
                metadata=json.loads(str(row["metadata_json"])),
            )
            handle.write(
                json.dumps(
                    {
                        "sample_id": record.sample_id,
                        "fen": record.fen,
                        "source": record.source,
                        "selected_move_uci": record.selected_move_uci,
                        "result": record.result,
                        "metadata": record.metadata,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
    return count


def _fen_hash_hex(fen: str) -> str:
    return hashlib.sha256(fen.encode("utf-8")).hexdigest()


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _normalize_result(value: str | None) -> str | None:
    if value in {"1-0", "0-1", "1/2-1/2"}:
        return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
