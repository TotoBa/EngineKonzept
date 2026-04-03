"""Versioned selfplay-agent specs for runtime-style proposer/planner stacks."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


SUPPORTED_OPPONENT_MODES = {"none", "symbolic", "learned"}
SELFPLAY_AGENT_SPEC_VERSION = 1


@dataclass(frozen=True)
class SelfplayAgentSpec:
    """Serializable runtime spec for one selfplay agent arm."""

    name: str
    proposer_checkpoint: str
    planner_checkpoint: str | None = None
    opponent_checkpoint: str | None = None
    dynamics_checkpoint: str | None = None
    opponent_mode: str = "symbolic"
    root_top_k: int = 4
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    spec_version: int = SELFPLAY_AGENT_SPEC_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("agent spec name must be non-empty")
        if not self.proposer_checkpoint:
            raise ValueError("agent spec proposer_checkpoint must be non-empty")
        if self.opponent_mode not in SUPPORTED_OPPONENT_MODES:
            raise ValueError(f"unsupported opponent_mode: {self.opponent_mode}")
        if self.root_top_k <= 0:
            raise ValueError("root_top_k must be positive")
        if self.spec_version != SELFPLAY_AGENT_SPEC_VERSION:
            raise ValueError(
                f"unsupported selfplay agent spec version: {self.spec_version}"
            )
        if self.opponent_mode == "learned" and self.opponent_checkpoint is None:
            raise ValueError("learned opponent_mode requires opponent_checkpoint")
        if self.planner_checkpoint is None:
            if self.opponent_checkpoint is not None or self.dynamics_checkpoint is not None:
                raise ValueError(
                    "opponent_checkpoint and dynamics_checkpoint require planner_checkpoint"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "proposer_checkpoint": self.proposer_checkpoint,
            "planner_checkpoint": self.planner_checkpoint,
            "opponent_checkpoint": self.opponent_checkpoint,
            "dynamics_checkpoint": self.dynamics_checkpoint,
            "opponent_mode": self.opponent_mode,
            "root_top_k": self.root_top_k,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayAgentSpec":
        return cls(
            spec_version=int(payload.get("spec_version", SELFPLAY_AGENT_SPEC_VERSION)),
            name=str(payload["name"]),
            proposer_checkpoint=str(payload["proposer_checkpoint"]),
            planner_checkpoint=_optional_str(payload.get("planner_checkpoint")),
            opponent_checkpoint=_optional_str(payload.get("opponent_checkpoint")),
            dynamics_checkpoint=_optional_str(payload.get("dynamics_checkpoint")),
            opponent_mode=str(payload.get("opponent_mode", "symbolic")),
            root_top_k=int(payload.get("root_top_k", 4)),
            tags=[str(value) for value in list(payload.get("tags") or [])],
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "SelfplayAgentSpec":
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("selfplay agent spec must be a JSON object")
        return cls.from_dict(payload)


def load_selfplay_agent_spec(path: Path) -> SelfplayAgentSpec:
    """Load a versioned selfplay-agent spec from JSON."""
    return SelfplayAgentSpec.from_json(path.read_text(encoding="utf-8"))


def write_selfplay_agent_spec(path: Path, spec: SelfplayAgentSpec) -> None:
    """Write a versioned selfplay-agent spec to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
