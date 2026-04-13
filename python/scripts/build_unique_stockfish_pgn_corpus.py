"""Build a resumable unique Stockfish-labeled PGN corpus for large Phase-5 runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import chess
import chess.engine
import chess.pgn

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from train.datasets.schema import RawPositionRecord  # noqa: E402
from train.orchestrator.label_corpus_ledger import (  # noqa: E402
    LabelCorpusLedger,
    MySQLLabelCorpusLedger,
)
from train.orchestrator.models import MySQLConfig  # noqa: E402


@dataclass(frozen=True)
class UniqueCorpusConfig:
    """Controls large-scale unique PGN sampling and Stockfish labeling."""

    engine_path: Path
    work_dir: Path
    target_train_records: int
    target_verify_records: int
    db_config: MySQLConfig | None = None
    ledger_namespace: str | None = None
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
    file_shard_index: int | None = None
    file_shard_count: int | None = None
    run_max_games: int = 0
    export_jsonl_on_complete: bool = True
    complete_at_eof: bool = False


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn-root", type=Path, required=True)
    parser.add_argument("--glob", default="**/*.pgn")
    parser.add_argument("--engine-path", type=Path, default=Path("/usr/games/stockfish18"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ledger-namespace", default=None)
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
    parser.add_argument("--file-shard-index", type=int, default=None)
    parser.add_argument("--file-shard-count", type=int, default=None)
    parser.add_argument("--run-max-games", type=int, default=0)
    parser.add_argument(
        "--no-export-jsonl-on-complete",
        action="store_false",
        dest="export_jsonl_on_complete",
    )
    parser.add_argument("--complete-at-eof", action="store_true")
    args = parser.parse_args(argv)

    config = UniqueCorpusConfig(
        engine_path=args.engine_path,
        work_dir=args.work_dir,
        target_train_records=args.target_train_records,
        target_verify_records=args.target_verify_records,
        db_config=MySQLConfig.from_env(),
        ledger_namespace=(
            str(args.ledger_namespace) if args.ledger_namespace is not None else None
        ),
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
        file_shard_index=args.file_shard_index,
        file_shard_count=args.file_shard_count,
        run_max_games=args.run_max_games,
        export_jsonl_on_complete=args.export_jsonl_on_complete,
        complete_at_eof=args.complete_at_eof,
    )

    pgn_paths = sorted(path for path in args.pgn_root.glob(args.glob) if path.is_file())
    pgn_paths = select_pgn_file_shard(
        pgn_paths,
        shard_index=args.file_shard_index,
        shard_count=args.file_shard_count,
    )
    if not pgn_paths:
        raise ValueError(f"no PGNs matched {args.glob!r} under {args.pgn_root}")

    summary = build_unique_corpus_from_pgns(pgn_paths, config=config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_unique_corpus_from_pgns(
    pgn_paths: Sequence[Path],
    *,
    config: UniqueCorpusConfig,
    ledger: LabelCorpusLedger | None = None,
) -> dict[str, Any]:
    """Stream PGNs into a resumable MySQL-backed unique corpus and optional JSONL export."""
    config.work_dir.mkdir(parents=True, exist_ok=True)
    progress_path = config.work_dir / "progress.json"
    ledger_namespace = _resolve_ledger_namespace(config)
    owned_ledger = ledger is None
    resolved_ledger = ledger or MySQLLabelCorpusLedger(_resolve_db_config(config))
    resolved_ledger.ensure_schema()

    engine = chess.engine.SimpleEngine.popen_uci(str(config.engine_path))
    try:
        engine.configure({"Hash": config.hash_mb, "Threads": config.threads})
        _resume_reserved_rows(
            resolved_ledger,
            namespace=ledger_namespace,
            engine=engine,
            config=config,
        )
        counts = resolved_ledger.split_counts(ledger_namespace)
        labeled_counts = resolved_ledger.split_counts(ledger_namespace, labeled_only=True)

        progress = _current_progress(
            config=config,
            pgn_paths=pgn_paths,
            current_pgn=None,
            games_seen=0,
            skipped_games=0,
            counts=counts,
            labeled_counts=labeled_counts,
            ledger_namespace=ledger_namespace,
        )
        _write_progress(progress_path, progress)

        games_seen = int(progress["games_seen"])
        invocation_games_seen = 0
        skipped_games = int(progress.get("skipped_games", 0))
        accepted_since_progress = 0
        for pgn_path in pgn_paths:
            if _targets_reached(counts, config=config):
                break
            with pgn_path.open("r", encoding="utf-8", errors="replace") as handle:
                while True:
                    if config.max_games > 0 and games_seen >= config.max_games:
                        break
                    if config.run_max_games > 0 and invocation_games_seen >= config.run_max_games:
                        break
                    if _targets_reached(counts, config=config):
                        break
                    try:
                        game = chess.pgn.read_game(handle)
                    except Exception:
                        skipped_games += 1
                        break
                    if game is None:
                        break
                    games_seen += 1
                    invocation_games_seen += 1
                    try:
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
                                        counts=counts,
                                        config=config,
                                    )
                                    if split is not None:
                                        fen_hash = _fen_hash_hex(fen)
                                        sample_id = (
                                            f"stockfish-unique:{pgn_path.stem}:{games_seen}:{ply_index}:"
                                            f"{fen_hash[:12]}"
                                        )
                                        reserved = resolved_ledger.reserve_sample(
                                            ledger_namespace,
                                            fen_hash=fen_hash,
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
                                            counts[split] += 1
                                            label_outcome = _label_reserved_sample(
                                                resolved_ledger,
                                                namespace=ledger_namespace,
                                                engine=engine,
                                                fen_hash_hex=fen_hash,
                                                config=config,
                                            )
                                            if label_outcome["deleted"]:
                                                counts[split] = max(0, counts[split] - 1)
                                            elif label_outcome["labeled"]:
                                                labeled_counts[split] += 1
                                            accepted_since_progress += 1
                                            if accepted_since_progress >= config.progress_every:
                                                accepted_since_progress = 0
                                                _write_progress(
                                                    progress_path,
                                                    _current_progress(
                                                        config=config,
                                                        pgn_paths=pgn_paths,
                                                        current_pgn=str(pgn_path),
                                                        games_seen=games_seen,
                                                        skipped_games=skipped_games,
                                                        counts=counts,
                                                        labeled_counts=labeled_counts,
                                                        ledger_namespace=ledger_namespace,
                                                    ),
                                                )
                            board.push(move)
                    except Exception:
                        skipped_games += 1
                        continue
            if config.run_max_games > 0 and invocation_games_seen >= config.run_max_games:
                break
        final_summary = _current_progress(
            config=config,
            pgn_paths=pgn_paths,
            current_pgn=None,
            games_seen=games_seen,
            skipped_games=skipped_games,
            counts=counts,
            labeled_counts=labeled_counts,
            ledger_namespace=ledger_namespace,
        )
        final_summary["completed"] = bool(
            _targets_reached(counts, config=config) or config.complete_at_eof
        )
        final_summary["completion_reason"] = (
            "targets_reached"
            if _targets_reached(counts, config=config)
            else "eof"
            if config.complete_at_eof
            else "targets_not_reached"
        )
        if final_summary["completed"] and config.export_jsonl_on_complete:
            export_summary = _export_jsonl(
                resolved_ledger,
                namespace=ledger_namespace,
                output_dir=config.work_dir,
            )
            final_summary["export"] = export_summary
        _write_progress(progress_path, final_summary)
        return final_summary
    finally:
        engine.quit()
        if owned_ledger:
            resolved_ledger.close()


def export_unique_corpus_snapshot(
    work_dir: Path,
    *,
    db_config: MySQLConfig | None = None,
    ledger_namespace: str | None = None,
    ledger: LabelCorpusLedger | None = None,
) -> dict[str, Any]:
    """Export the currently labeled unique-corpus rows as raw JSONL artifacts."""
    namespace = ledger_namespace or str(work_dir)
    owned_ledger = ledger is None
    resolved_ledger = ledger or MySQLLabelCorpusLedger(db_config or MySQLConfig.from_env())
    resolved_ledger.ensure_schema()
    try:
        export_summary = _export_jsonl(resolved_ledger, namespace=namespace, output_dir=work_dir)
        export_summary["counts"] = resolved_ledger.split_counts(namespace)
        export_summary["labeled_counts"] = resolved_ledger.split_counts(
            namespace,
            labeled_only=True,
        )
        return export_summary
    finally:
        if owned_ledger:
            resolved_ledger.close()


def _targets_reached(counts: dict[str, int], *, config: UniqueCorpusConfig) -> bool:
    return (
        counts["train"] >= config.target_train_records
        and counts["verify"] >= config.target_verify_records
    )


def _choose_split(
    fen: str,
    *,
    counts: dict[str, int],
    config: UniqueCorpusConfig,
) -> str | None:
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


def _resume_reserved_rows(
    ledger: LabelCorpusLedger,
    *,
    namespace: str,
    engine: chess.engine.SimpleEngine,
    config: UniqueCorpusConfig,
) -> None:
    for row in ledger.iter_reserved_samples(namespace):
        _label_reserved_sample(
            ledger,
            namespace=namespace,
            engine=engine,
            fen_hash_hex=row.fen_hash,
            config=config,
        )


def _label_reserved_sample(
    ledger: LabelCorpusLedger,
    *,
    namespace: str,
    engine: chess.engine.SimpleEngine,
    fen_hash_hex: str,
    config: UniqueCorpusConfig,
) -> dict[str, bool]:
    row = ledger.load_reserved_sample(namespace, fen_hash=fen_hash_hex)
    if row is None:
        return {"labeled": False, "deleted": False}

    board = chess.Board(str(row.fen))
    result = engine.play(board, chess.engine.Limit(nodes=config.engine_nodes))
    if result.move is None:
        ledger.delete_reserved_sample(namespace, fen_hash=fen_hash_hex)
        return {"labeled": False, "deleted": True}

    metadata = dict(row.metadata)
    metadata["label_source"] = "stockfish18"
    metadata["stockfish_nodes"] = config.engine_nodes
    metadata["stockfish_bestmove_uci"] = result.move.uci()
    metadata["stockfish_matches_played"] = metadata.get("played_move_uci") == result.move.uci()

    labeled = ledger.mark_sample_labeled(
        namespace,
        fen_hash=fen_hash_hex,
        selected_move_uci=result.move.uci(),
        metadata=metadata,
    )
    return {"labeled": labeled, "deleted": False}


def _current_progress(
    *,
    config: UniqueCorpusConfig,
    pgn_paths: Sequence[Path],
    current_pgn: str | None,
    games_seen: int,
    skipped_games: int,
    counts: dict[str, int],
    labeled_counts: dict[str, int],
    ledger_namespace: str,
) -> dict[str, Any]:
    return {
        "engine_path": str(config.engine_path),
        "engine_nodes": config.engine_nodes,
        "hash_mb": config.hash_mb,
        "threads": config.threads,
        "split_seed": config.split_seed,
        "verify_divisor": config.verify_divisor,
        "ledger_backend": "mysql",
        "ledger_namespace": ledger_namespace,
        "file_shard_index": config.file_shard_index,
        "file_shard_count": config.file_shard_count,
        "run_max_games": config.run_max_games,
        "pgn_files": [str(path) for path in pgn_paths],
        "current_pgn": current_pgn,
        "games_seen": games_seen,
        "skipped_games": skipped_games,
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
    _write_atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _export_jsonl(
    ledger: LabelCorpusLedger,
    *,
    namespace: str,
    output_dir: Path,
) -> dict[str, Any]:
    train_path = output_dir / "train_raw.jsonl"
    verify_path = output_dir / "verify_raw.jsonl"
    train_count = _write_split_jsonl(
        ledger,
        namespace=namespace,
        split="train",
        output_path=train_path,
    )
    verify_count = _write_split_jsonl(
        ledger,
        namespace=namespace,
        split="verify",
        output_path=verify_path,
    )
    return {
        "train_raw_path": str(train_path),
        "verify_raw_path": str(verify_path),
        "train_records": train_count,
        "verify_records": verify_count,
    }


def _write_split_jsonl(
    ledger: LabelCorpusLedger,
    *,
    namespace: str,
    split: str,
    output_path: Path,
) -> int:
    count = 0
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in ledger.iter_labeled_samples(namespace, split=split):
            record = RawPositionRecord(
                sample_id=row.sample_id,
                fen=row.fen,
                source=row.source,
                selected_move_uci=row.selected_move_uci,
                result=row.result,
                metadata=row.metadata,
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
    temp_path.replace(output_path)
    return count


def _write_atomic_text(path: Path, content: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _fen_hash_hex(fen: str) -> str:
    return hashlib.sha256(fen.encode("utf-8")).hexdigest()


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def select_pgn_file_shard(
    pgn_paths: Sequence[Path],
    *,
    shard_index: int | None,
    shard_count: int | None,
) -> list[Path]:
    """Return the deterministic file subset for one 1-based shard selection."""
    if shard_index is None and shard_count is None:
        return list(pgn_paths)
    if shard_index is None or shard_count is None:
        raise ValueError("file sharding requires both shard_index and shard_count")
    if shard_count <= 0:
        raise ValueError("file shard_count must be positive")
    if not 1 <= shard_index <= shard_count:
        raise ValueError("file shard_index must be within [1, shard_count]")
    return [
        path
        for index, path in enumerate(pgn_paths)
        if index % shard_count == shard_index - 1
    ]


def _normalize_result(value: str | None) -> str | None:
    if value in {"1-0", "0-1", "1/2-1/2"}:
        return value
    return None


def _resolve_db_config(config: UniqueCorpusConfig) -> MySQLConfig:
    return config.db_config if config.db_config is not None else MySQLConfig.from_env()


def _resolve_ledger_namespace(config: UniqueCorpusConfig) -> str:
    return config.ledger_namespace if config.ledger_namespace is not None else str(config.work_dir)


if __name__ == "__main__":
    raise SystemExit(main())
