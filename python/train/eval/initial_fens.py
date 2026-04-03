"""Versioned selfplay initial-position suites for larger arena stages."""

from __future__ import annotations

from dataclasses import dataclass, field
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
