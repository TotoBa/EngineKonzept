"""Streaming PGN sampling and Stockfish labeling for Phase-5 policy datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import chess
import chess.engine
import chess.pgn

from train.datasets.schema import RawPositionRecord, SplitRatios


@dataclass(frozen=True)
class PgnPolicySamplingConfig:
    """Bounded sampling and engine settings for PGN policy extraction."""

    engine_path: Path
    max_train_records: int
    max_verify_records: int
    min_ply: int = 8
    max_ply: int = 80
    samples_per_game: int = 2
    engine_nodes: int = 2_000
    hash_mb: int = 32
    threads: int = 1
    split_seed: str = "phase5-stockfish-pgn-v1"


@dataclass(frozen=True)
class PgnPolicySample:
    """Candidate PGN position before exact-rule enrichment."""

    sample_id: str
    fen: str
    source: str
    result: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PgnPolicySelection:
    """Selected train and verification raw records plus summary metadata."""

    train_records: list[RawPositionRecord]
    verify_records: list[RawPositionRecord]
    summary: dict[str, Any]


def sample_policy_records_from_pgns(
    pgn_paths: Sequence[Path],
    *,
    config: PgnPolicySamplingConfig,
) -> PgnPolicySelection:
    """Stream PGNs, sample bounded positions, and label them with Stockfish."""
    if config.max_train_records <= 0:
        raise ValueError("max_train_records must be positive")
    if config.max_verify_records <= 0:
        raise ValueError("max_verify_records must be positive")
    if config.samples_per_game <= 0:
        raise ValueError("samples_per_game must be positive")

    resolved_paths = [path for path in sorted(pgn_paths) if path.is_file()]
    if not resolved_paths:
        raise ValueError("no PGN files found for policy sampling")

    train_candidates: list[PgnPolicySample] = []
    verify_candidates: list[PgnPolicySample] = []
    games_seen = 0

    for path in resolved_paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            while len(train_candidates) < config.max_train_records or len(
                verify_candidates
            ) < config.max_verify_records:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                games_seen += 1
                for candidate in _select_game_candidates(
                    game,
                    source_name=path.stem,
                    game_index=games_seen,
                    config=config,
                ):
                    route_value = _stable_hash(f"{config.split_seed}:{candidate.sample_id}")
                    target = (
                        verify_candidates
                        if route_value % 5 == 0
                        else train_candidates
                    )
                    target_limit = (
                        config.max_verify_records
                        if target is verify_candidates
                        else config.max_train_records
                    )
                    if len(target) < target_limit:
                        target.append(candidate)
                    elif target is verify_candidates and len(train_candidates) < config.max_train_records:
                        train_candidates.append(candidate)

                if len(train_candidates) >= config.max_train_records and len(
                    verify_candidates
                ) >= config.max_verify_records:
                    break
        if len(train_candidates) >= config.max_train_records and len(
            verify_candidates
        ) >= config.max_verify_records:
            break

    if not train_candidates:
        raise ValueError("no training candidates selected from PGNs")
    if not verify_candidates:
        raise ValueError("no verification candidates selected from PGNs")

    engine = chess.engine.SimpleEngine.popen_uci(str(config.engine_path))
    try:
        engine.configure({"Hash": config.hash_mb, "Threads": config.threads})
        train_records = _label_candidates(engine, train_candidates, config=config)
        verify_records = _label_candidates(engine, verify_candidates, config=config)
    finally:
        engine.quit()

    return PgnPolicySelection(
        train_records=train_records,
        verify_records=verify_records,
        summary={
            "engine_path": str(config.engine_path),
            "engine_nodes": config.engine_nodes,
            "games_seen": games_seen,
            "pgn_files": [str(path) for path in resolved_paths],
            "train_records": len(train_records),
            "verify_records": len(verify_records),
            "sampling": {
                "min_ply": config.min_ply,
                "max_ply": config.max_ply,
                "samples_per_game": config.samples_per_game,
                "split_seed": config.split_seed,
            },
        },
    )


def training_split_ratios() -> SplitRatios:
    """Return deterministic train/validation ratios for PGN-labeled training data."""
    return SplitRatios(train=0.9, validation=0.1, test=0.0)


def verification_split_ratios() -> SplitRatios:
    """Return a split assignment that keeps the verification set in `test`."""
    return SplitRatios(train=0.0, validation=0.0, test=1.0)


def _select_game_candidates(
    game: chess.pgn.Game,
    *,
    source_name: str,
    game_index: int,
    config: PgnPolicySamplingConfig,
) -> list[PgnPolicySample]:
    board = game.board()
    result = _normalize_result(game.headers.get("Result"))
    ranked: list[tuple[int, PgnPolicySample]] = []

    for ply_index, move in enumerate(game.mainline_moves(), start=1):
        if config.min_ply <= ply_index <= config.max_ply:
            if board.is_valid() and not board.is_game_over(claim_draw=True):
                sample_id = f"stockfish-pgn:{source_name}:{game_index}:{ply_index}"
                played_move_uci = move.uci()
                candidate = PgnPolicySample(
                    sample_id=sample_id,
                    fen=board.fen(),
                    source="stockfish-pgn",
                    result=result,
                    metadata={
                        "source_pgn": source_name,
                        "game_index": game_index,
                        "ply": ply_index,
                        "event": game.headers.get("Event"),
                        "white": game.headers.get("White"),
                        "black": game.headers.get("Black"),
                        "played_move_uci": played_move_uci,
                    },
                )
                rank = _stable_hash(f"{config.split_seed}:{sample_id}")
                ranked.append((rank, candidate))
        board.push(move)

    ranked.sort(key=lambda item: item[0])
    return [candidate for _, candidate in ranked[: config.samples_per_game]]


def _label_candidates(
    engine: chess.engine.SimpleEngine,
    candidates: Sequence[PgnPolicySample],
    *,
    config: PgnPolicySamplingConfig,
) -> list[RawPositionRecord]:
    records: list[RawPositionRecord] = []
    limit = chess.engine.Limit(nodes=config.engine_nodes)

    for candidate in candidates:
        board = chess.Board(candidate.fen)
        result = engine.play(board, limit)
        if result.move is None:
            continue
        metadata = dict(candidate.metadata)
        metadata["label_source"] = "stockfish18"
        metadata["stockfish_nodes"] = config.engine_nodes
        metadata["stockfish_bestmove_uci"] = result.move.uci()
        metadata["stockfish_matches_played"] = (
            metadata.get("played_move_uci") == result.move.uci()
        )
        records.append(
            RawPositionRecord(
                sample_id=candidate.sample_id,
                fen=candidate.fen,
                source=candidate.source,
                selected_move_uci=result.move.uci(),
                result=candidate.result,
                metadata=metadata,
            )
        )

    return records


def _normalize_result(value: str | None) -> str | None:
    if value in {"1-0", "0-1", "1/2-1/2"}:
        return value
    return None


def _stable_hash(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
