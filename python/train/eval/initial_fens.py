"""Versioned selfplay initial-position suites for larger arena stages."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import io
import json
from pathlib import Path
from typing import Any


SELFPLAY_INITIAL_FEN_SUITE_VERSION = 1


@dataclass(frozen=True)
class SelfplayInitialFenEntry:
    """One curated nonterminal arena start position."""

    fen: str
    tier: str
    sample_id: str
    source_path: str
    result: str
    selection_score: float
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fen:
            raise ValueError("selfplay initial FEN entry fen must be non-empty")
        if not self.tier:
            raise ValueError("selfplay initial FEN entry tier must be non-empty")
        if not self.sample_id:
            raise ValueError("selfplay initial FEN entry sample_id must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "fen": self.fen,
            "tier": self.tier,
            "sample_id": self.sample_id,
            "source_path": self.source_path,
            "result": self.result,
            "selection_score": round(self.selection_score, 6),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayInitialFenEntry":
        return cls(
            fen=str(payload["fen"]),
            tier=str(payload["tier"]),
            sample_id=str(payload["sample_id"]),
            source_path=str(payload["source_path"]),
            result=str(payload["result"]),
            selection_score=float(payload["selection_score"]),
            tags=[str(value) for value in list(payload.get("tags") or [])],
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class SelfplayInitialFenSuite:
    """Versioned set of curated arena start positions."""

    name: str
    entries: list[SelfplayInitialFenEntry]
    metadata: dict[str, Any] = field(default_factory=dict)
    spec_version: int = SELFPLAY_INITIAL_FEN_SUITE_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("selfplay initial FEN suite name must be non-empty")
        if self.spec_version != SELFPLAY_INITIAL_FEN_SUITE_VERSION:
            raise ValueError(f"unsupported initial FEN suite version: {self.spec_version}")
        if not self.entries:
            raise ValueError("selfplay initial FEN suite must include at least one entry")

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "entries": [entry.to_dict() for entry in self.entries],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayInitialFenSuite":
        return cls(
            spec_version=int(payload.get("spec_version", SELFPLAY_INITIAL_FEN_SUITE_VERSION)),
            name=str(payload["name"]),
            entries=[
                SelfplayInitialFenEntry.from_dict(dict(entry))
                for entry in list(payload["entries"])
            ],
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "SelfplayInitialFenSuite":
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("selfplay initial FEN suite must be a JSON object")
        return cls.from_dict(payload)

    def fen_list(self) -> list[str]:
        return [entry.fen for entry in self.entries]


def load_selfplay_initial_fen_suite(path: Path) -> SelfplayInitialFenSuite:
    """Load a versioned initial-position suite from JSON."""
    return SelfplayInitialFenSuite.from_json(path.read_text(encoding="utf-8"))


def write_selfplay_initial_fen_suite(path: Path, suite: SelfplayInitialFenSuite) -> None:
    """Write a versioned initial-position suite as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(suite.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_opening_initial_fen_suite(
    *,
    name: str,
    tsv_paths: list[Path],
    entries_per_file: int = 2,
    min_ply_count: int = 8,
) -> SelfplayInitialFenSuite:
    """Build a deterministic opening-position suite from TSV opening books."""
    if not tsv_paths:
        raise ValueError("opening suite requires at least one TSV path")
    if entries_per_file <= 0:
        raise ValueError("opening suite entries_per_file must be positive")
    if min_ply_count <= 0:
        raise ValueError("opening suite min_ply_count must be positive")

    entries: list[SelfplayInitialFenEntry] = []
    file_summaries: list[dict[str, object]] = []
    for tsv_path in tsv_paths:
        candidates = _load_opening_candidates(tsv_path=tsv_path, min_ply_count=min_ply_count)
        selected_candidates = _select_evenly_spaced_candidates(
            candidates=candidates,
            target_count=min(entries_per_file, len(candidates)),
        )
        entries.extend(selected_candidates)
        file_summaries.append(
            {
                "path": str(tsv_path),
                "candidate_count": len(candidates),
                "selected_count": len(selected_candidates),
                "sample_ids": [entry.sample_id for entry in selected_candidates],
            }
        )
    if not entries:
        raise ValueError("opening suite selection produced no entries")
    return SelfplayInitialFenSuite(
        name=name,
        entries=entries,
        metadata={
            "source": "thor_openings_tsv",
            "entries_per_file": entries_per_file,
            "min_ply_count": min_ply_count,
            "file_summaries": file_summaries,
        },
    )


def merge_selfplay_initial_fen_suites(
    *,
    name: str,
    suites: list[SelfplayInitialFenSuite],
    metadata: dict[str, Any] | None = None,
) -> SelfplayInitialFenSuite:
    """Merge one or more suites while preserving first-seen order and deduping by FEN."""
    if not suites:
        raise ValueError("merged selfplay initial FEN suite requires at least one source suite")
    merged_entries: list[SelfplayInitialFenEntry] = []
    seen_fens: set[str] = set()
    for suite in suites:
        for entry in suite.entries:
            if entry.fen in seen_fens:
                continue
            merged_entries.append(entry)
            seen_fens.add(entry.fen)
    if not merged_entries:
        raise ValueError("merged selfplay initial FEN suite produced no entries")
    return SelfplayInitialFenSuite(
        name=name,
        entries=merged_entries,
        metadata={
            "source_suite_names": [suite.name for suite in suites],
            **(metadata or {}),
        },
    )


def _load_opening_candidates(
    *,
    tsv_path: Path,
    min_ply_count: int,
) -> list[SelfplayInitialFenEntry]:
    try:
        import chess.pgn
    except ImportError as exc:  # pragma: no cover - exercised through CLI/tests with optional deps
        raise RuntimeError("python-chess is required for opening FEN suite generation") from exc

    candidates: list[SelfplayInitialFenEntry] = []
    seen_fens: set[str] = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_index, row in enumerate(reader, start=1):
            pgn_moves = str(row.get("pgn") or "").strip()
            if not pgn_moves:
                continue
            board = _parse_opening_board(chess_pgn=chess.pgn, pgn_moves=pgn_moves)
            if board is None:
                continue
            ply_count = board.ply()
            if ply_count < min_ply_count or board.is_game_over():
                continue
            fen = board.fen()
            if fen in seen_fens:
                continue
            eco = str(row.get("eco") or tsv_path.stem.upper())
            name = str(row.get("name") or f"{tsv_path.stem}:{row_index}")
            seen_fens.add(fen)
            candidates.append(
                SelfplayInitialFenEntry(
                    fen=fen,
                    tier="thor_openings",
                    sample_id=f"{tsv_path.stem}:{row_index}:{eco}",
                    source_path=str(tsv_path),
                    result="*",
                    selection_score=float(ply_count),
                    tags=["opening", "thor_openings", eco],
                    metadata={
                        "eco": eco,
                        "opening_name": name,
                        "ply_count": ply_count,
                        "side_to_move": "w" if board.turn else "b",
                    },
                )
            )
    return candidates


def _parse_opening_board(*, chess_pgn: Any, pgn_moves: str) -> Any | None:
    pgn_text = '[Event "opening"]\n\n' + pgn_moves.strip() + " *\n"
    game = chess_pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return None
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


def _select_evenly_spaced_candidates(
    *,
    candidates: list[SelfplayInitialFenEntry],
    target_count: int,
) -> list[SelfplayInitialFenEntry]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(candidates) <= target_count:
        return list(candidates)
    selected_indices = {
        round((len(candidates) - 1) * index / (target_count - 1))
        for index in range(target_count)
    }
    return [candidate for index, candidate in enumerate(candidates) if index in selected_indices]
