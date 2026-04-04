"""Versioned selfplay-agent specs for runtime-style and offline arena agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


SUPPORTED_AGENT_KINDS = {"planner", "uci_engine"}
SUPPORTED_OPPONENT_MODES = {"none", "symbolic", "learned"}
SELFPLAY_AGENT_SPEC_VERSION = 1


@dataclass(frozen=True)
class SelfplayAgentSpec:
    """Serializable runtime spec for one selfplay agent arm."""

    name: str
    proposer_checkpoint: str | None = None
    planner_checkpoint: str | None = None
    opponent_checkpoint: str | None = None
    dynamics_checkpoint: str | None = None
    opponent_mode: str = "symbolic"
    root_top_k: int = 4
    agent_kind: str = "planner"
    external_engine_path: str | None = None
    external_engine_nodes: int | None = None
    external_engine_depth: int | None = None
    external_engine_movetime_ms: int | None = None
    external_engine_threads: int = 1
    external_engine_hash_mb: int | None = 16
    external_engine_options: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    spec_version: int = SELFPLAY_AGENT_SPEC_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("agent spec name must be non-empty")
        if self.agent_kind not in SUPPORTED_AGENT_KINDS:
            raise ValueError(f"unsupported agent_kind: {self.agent_kind}")
        if self.opponent_mode not in SUPPORTED_OPPONENT_MODES:
            raise ValueError(f"unsupported opponent_mode: {self.opponent_mode}")
        if self.root_top_k <= 0:
            raise ValueError("root_top_k must be positive")
        if self.external_engine_threads <= 0:
            raise ValueError("external_engine_threads must be positive")
        if self.external_engine_hash_mb is not None and self.external_engine_hash_mb <= 0:
            raise ValueError("external_engine_hash_mb must be positive when provided")
        if self.spec_version != SELFPLAY_AGENT_SPEC_VERSION:
            raise ValueError(
                f"unsupported selfplay agent spec version: {self.spec_version}"
            )
        if self.agent_kind == "planner":
            if not self.proposer_checkpoint:
                raise ValueError("planner agent proposer_checkpoint must be non-empty")
            if self.opponent_mode == "learned" and self.opponent_checkpoint is None:
                raise ValueError("learned opponent_mode requires opponent_checkpoint")
            if self.planner_checkpoint is None:
                if self.opponent_checkpoint is not None or self.dynamics_checkpoint is not None:
                    raise ValueError(
                        "opponent_checkpoint and dynamics_checkpoint require planner_checkpoint"
                    )
            if self.external_engine_path is not None:
                raise ValueError(
                    "external engine fields are not allowed for planner agents"
                )
            return

        if not self.external_engine_path:
            raise ValueError("uci_engine agent requires external_engine_path")
        if (
            self.external_engine_nodes is None
            and self.external_engine_depth is None
            and self.external_engine_movetime_ms is None
        ):
            raise ValueError(
                "uci_engine agent requires one of external_engine_nodes, external_engine_depth, or external_engine_movetime_ms"
            )
        if self.external_engine_nodes is not None and self.external_engine_nodes <= 0:
            raise ValueError("external_engine_nodes must be positive when provided")
        if self.external_engine_depth is not None and self.external_engine_depth <= 0:
            raise ValueError("external_engine_depth must be positive when provided")
        if self.external_engine_movetime_ms is not None and self.external_engine_movetime_ms <= 0:
            raise ValueError("external_engine_movetime_ms must be positive when provided")
        if (
            self.proposer_checkpoint is not None
            or self.planner_checkpoint is not None
            or self.opponent_checkpoint is not None
            or self.dynamics_checkpoint is not None
        ):
            raise ValueError("uci_engine agent must not include planner checkpoints")

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
            "agent_kind": self.agent_kind,
            "external_engine_path": self.external_engine_path,
            "external_engine_nodes": self.external_engine_nodes,
            "external_engine_depth": self.external_engine_depth,
            "external_engine_movetime_ms": self.external_engine_movetime_ms,
            "external_engine_threads": self.external_engine_threads,
            "external_engine_hash_mb": self.external_engine_hash_mb,
            "external_engine_options": dict(self.external_engine_options),
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayAgentSpec":
        return cls(
            spec_version=int(payload.get("spec_version", SELFPLAY_AGENT_SPEC_VERSION)),
            name=str(payload["name"]),
            proposer_checkpoint=_optional_str(payload.get("proposer_checkpoint")),
            planner_checkpoint=_optional_str(payload.get("planner_checkpoint")),
            opponent_checkpoint=_optional_str(payload.get("opponent_checkpoint")),
            dynamics_checkpoint=_optional_str(payload.get("dynamics_checkpoint")),
            opponent_mode=str(payload.get("opponent_mode", "symbolic")),
            root_top_k=int(payload.get("root_top_k", 4)),
            agent_kind=str(payload.get("agent_kind", "planner")),
            external_engine_path=_optional_str(payload.get("external_engine_path")),
            external_engine_nodes=_optional_int(payload.get("external_engine_nodes")),
            external_engine_depth=_optional_int(payload.get("external_engine_depth")),
            external_engine_movetime_ms=_optional_int(payload.get("external_engine_movetime_ms")),
            external_engine_threads=int(payload.get("external_engine_threads", 1)),
            external_engine_hash_mb=_optional_int(payload.get("external_engine_hash_mb", 16)),
            external_engine_options={
                str(name): str(value)
                for name, value in dict(payload.get("external_engine_options") or {}).items()
            },
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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
