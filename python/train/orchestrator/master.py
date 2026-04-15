"""Long-running master loop that submits and promotes MySQL-backed campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import pymysql

from train.datasets import RawPositionRecord, load_raw_records
from train.eval.agent_spec import SelfplayAgentSpec, load_selfplay_agent_spec, write_selfplay_agent_spec
from train.eval.matrix import build_selfplay_arena_matrix
from train.eval.phase10_campaign import (
    Phase10Lapv1ArenaCampaignSpec,
    load_phase10_lapv1_arena_campaign_spec,
    resolve_repo_path,
)
from scripts.merge_phase5_raw_corpora import RawCorpusSourceSpec, merge_phase5_raw_corpora
from train.orchestrator.controller import OrchestratorController
from train.orchestrator.db import OrchestratorDB
from train.orchestrator.models import CampaignRow, ModelRow
from train.orchestrator.training_data_usage_ledger import (
    InMemoryLineageTrainingUsageLedger,
    LineageSampleUsageState,
    LineageTrainingUsageLedger,
    MySQLLineageTrainingUsageLedger,
)


MASTER_SPEC_VERSION = 1
TERMINAL_CAMPAIGN_STATES = {"succeeded", "failed"}
ACTIVE_CAMPAIGN_STATES = {"queued", "running", "training", "verifying", "finalizing", "selfplay_completed"}
ACCEPT_POLICIES = {"continue_training", "stop"}
REJECT_POLICIES = {"stop", "restart_from_seed"}
ARENA_PROGRESS_POLICY_VERSION = 1
_TRANSIENT_MYSQL_ERROR_CODES = {2006, 2013, 2055}


@dataclass(frozen=True)
class LabelPgnCorpusJobSpec:
    """One resumable PGN-labeling job managed by the master loop."""

    name: str
    pgn_root: str
    work_dir: str
    enabled: bool = True
    glob: str = "**/*.pgn"
    engine_path: str = "/usr/games/stockfish18"
    target_train_records: int = 10_000_000
    target_verify_records: int = 10_000
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
    complete_at_eof: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("label job name must be non-empty")
        if not self.pgn_root:
            raise ValueError("label job pgn_root must be non-empty")
        if not self.work_dir:
            raise ValueError("label job work_dir must be non-empty")
        if self.target_train_records <= 0:
            raise ValueError("label job target_train_records must be positive")
        if self.target_verify_records < 0:
            raise ValueError("label job target_verify_records must be non-negative")
        if self.threads <= 0:
            raise ValueError("label job threads must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pgn_root": self.pgn_root,
            "work_dir": self.work_dir,
            "enabled": self.enabled,
            "glob": self.glob,
            "engine_path": self.engine_path,
            "target_train_records": self.target_train_records,
            "target_verify_records": self.target_verify_records,
            "min_ply": self.min_ply,
            "max_ply": self.max_ply,
            "ply_stride": self.ply_stride,
            "engine_nodes": self.engine_nodes,
            "hash_mb": self.hash_mb,
            "threads": self.threads,
            "split_seed": self.split_seed,
            "verify_divisor": self.verify_divisor,
            "progress_every": self.progress_every,
            "max_games": self.max_games,
            "export_jsonl_on_complete": self.export_jsonl_on_complete,
            "complete_at_eof": self.complete_at_eof,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LabelPgnCorpusJobSpec":
        return cls(
            name=str(payload["name"]),
            pgn_root=str(payload["pgn_root"]),
            work_dir=str(payload["work_dir"]),
            enabled=bool(payload.get("enabled", True)),
            glob=str(payload.get("glob", "**/*.pgn")),
            engine_path=str(payload.get("engine_path", "/usr/games/stockfish18")),
            target_train_records=int(payload.get("target_train_records", 10_000_000)),
            target_verify_records=int(payload.get("target_verify_records", 10_000)),
            min_ply=int(payload.get("min_ply", 8)),
            max_ply=int(payload.get("max_ply", 80)),
            ply_stride=int(payload.get("ply_stride", 2)),
            engine_nodes=int(payload.get("engine_nodes", 1500)),
            hash_mb=int(payload.get("hash_mb", 32)),
            threads=int(payload.get("threads", 1)),
            split_seed=str(payload.get("split_seed", "phase5-stockfish-unique-v1")),
            verify_divisor=int(payload.get("verify_divisor", 1000)),
            progress_every=int(payload.get("progress_every", 1000)),
            max_games=int(payload.get("max_games", 0)),
            export_jsonl_on_complete=bool(payload.get("export_jsonl_on_complete", True)),
            complete_at_eof=bool(payload.get("complete_at_eof", False)),
        )


@dataclass(frozen=True)
class IdlePhase10ArtifactJobSpec:
    """One low-priority PGN -> LAPv2 Phase-10 artifact build managed by the master."""

    name: str
    phase10_config_path: str
    pgn_root: str
    work_root: str
    enabled: bool = True
    glob: str = "**/*.pgn"
    engine_path: str = "/usr/games/stockfish18"
    target_train_records: int = 10_000_000
    target_verify_records: int = 10_000
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
    shard_count: int = 1
    run_max_games: int = 0
    complete_at_eof: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("idle phase10 job name must be non-empty")
        if not self.phase10_config_path:
            raise ValueError("idle phase10 job phase10_config_path must be non-empty")
        if not self.pgn_root:
            raise ValueError("idle phase10 job pgn_root must be non-empty")
        if not self.work_root:
            raise ValueError("idle phase10 job work_root must be non-empty")
        if self.target_train_records <= 0:
            raise ValueError("idle phase10 job target_train_records must be positive")
        if self.target_verify_records < 0:
            raise ValueError("idle phase10 job target_verify_records must be non-negative")
        if self.threads <= 0:
            raise ValueError("idle phase10 job threads must be positive")
        if self.shard_count <= 0:
            raise ValueError("idle phase10 job shard_count must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "phase10_config_path": self.phase10_config_path,
            "pgn_root": self.pgn_root,
            "work_root": self.work_root,
            "enabled": self.enabled,
            "glob": self.glob,
            "engine_path": self.engine_path,
            "target_train_records": self.target_train_records,
            "target_verify_records": self.target_verify_records,
            "min_ply": self.min_ply,
            "max_ply": self.max_ply,
            "ply_stride": self.ply_stride,
            "engine_nodes": self.engine_nodes,
            "hash_mb": self.hash_mb,
            "threads": self.threads,
            "split_seed": self.split_seed,
            "verify_divisor": self.verify_divisor,
            "progress_every": self.progress_every,
            "max_games": self.max_games,
            "shard_count": self.shard_count,
            "run_max_games": self.run_max_games,
            "complete_at_eof": self.complete_at_eof,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "IdlePhase10ArtifactJobSpec":
        return cls(
            name=str(payload["name"]),
            phase10_config_path=str(payload["phase10_config_path"]),
            pgn_root=str(payload["pgn_root"]),
            work_root=str(payload["work_root"]),
            enabled=bool(payload.get("enabled", True)),
            glob=str(payload.get("glob", "**/*.pgn")),
            engine_path=str(payload.get("engine_path", "/usr/games/stockfish18")),
            target_train_records=int(payload.get("target_train_records", 10_000_000)),
            target_verify_records=int(payload.get("target_verify_records", 10_000)),
            min_ply=int(payload.get("min_ply", 8)),
            max_ply=int(payload.get("max_ply", 80)),
            ply_stride=int(payload.get("ply_stride", 2)),
            engine_nodes=int(payload.get("engine_nodes", 1500)),
            hash_mb=int(payload.get("hash_mb", 32)),
            threads=int(payload.get("threads", 1)),
            split_seed=str(payload.get("split_seed", "phase5-stockfish-unique-v1")),
            verify_divisor=int(payload.get("verify_divisor", 1000)),
            progress_every=int(payload.get("progress_every", 1000)),
            max_games=int(payload.get("max_games", 0)),
            shard_count=int(payload.get("shard_count", 1)),
            run_max_games=int(payload.get("run_max_games", 0)),
            complete_at_eof=bool(payload.get("complete_at_eof", True)),
        )


@dataclass(frozen=True)
class PromotionThresholds:
    """Minimum accept thresholds for one completed Phase-10 generation."""

    min_verify_top1_accuracy: float | None = None
    min_verify_top3_accuracy: float | None = None
    min_arena_score_rate: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "min_verify_top1_accuracy": self.min_verify_top1_accuracy,
            "min_verify_top3_accuracy": self.min_verify_top3_accuracy,
            "min_arena_score_rate": self.min_arena_score_rate,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PromotionThresholds":
        return cls(
            min_verify_top1_accuracy=_optional_float(payload.get("min_verify_top1_accuracy")),
            min_verify_top3_accuracy=_optional_float(payload.get("min_verify_top3_accuracy")),
            min_arena_score_rate=_optional_float(payload.get("min_arena_score_rate")),
        )


@dataclass(frozen=True)
class ArenaProgressionSpec:
    """Arena-opponent registry and external-engine progression policy."""

    safe_score_rate: float = 0.55
    min_games: int = 2
    stockfish_engine_path: str = "/usr/games/stockfish18"
    stockfish_initial_skill_level: int = 0
    stockfish_max_skill_level: int = 20
    stockfish_skill_step: int = 1
    stockfish_depth: int | None = 4
    stockfish_nodes: int | None = None
    stockfish_movetime_ms: int | None = None
    stockfish_threads: int = 1
    stockfish_hash_mb: int | None = 16
    stockfish_engine_options: dict[str, str] = field(default_factory=dict)
    spec_version: int = ARENA_PROGRESS_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.spec_version != ARENA_PROGRESS_POLICY_VERSION:
            raise ValueError(
                f"unsupported arena progression spec version: {self.spec_version}"
            )
        if not 0.0 <= self.safe_score_rate <= 1.0:
            raise ValueError("arena progression safe_score_rate must be in [0.0, 1.0]")
        if self.min_games <= 0:
            raise ValueError("arena progression min_games must be positive")
        if not self.stockfish_engine_path:
            raise ValueError("arena progression stockfish_engine_path must be non-empty")
        if not 0 <= self.stockfish_initial_skill_level <= 20:
            raise ValueError(
                "arena progression stockfish_initial_skill_level must be in [0, 20]"
            )
        if not 0 <= self.stockfish_max_skill_level <= 20:
            raise ValueError("arena progression stockfish_max_skill_level must be in [0, 20]")
        if self.stockfish_initial_skill_level > self.stockfish_max_skill_level:
            raise ValueError(
                "arena progression stockfish_initial_skill_level must not exceed stockfish_max_skill_level"
            )
        if self.stockfish_skill_step <= 0:
            raise ValueError("arena progression stockfish_skill_step must be positive")
        if (
            self.stockfish_depth is None
            and self.stockfish_nodes is None
            and self.stockfish_movetime_ms is None
        ):
            raise ValueError(
                "arena progression requires one of stockfish_depth, stockfish_nodes, or stockfish_movetime_ms"
            )
        if self.stockfish_threads <= 0:
            raise ValueError("arena progression stockfish_threads must be positive")
        if self.stockfish_hash_mb is not None and self.stockfish_hash_mb <= 0:
            raise ValueError(
                "arena progression stockfish_hash_mb must be positive when provided"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "safe_score_rate": self.safe_score_rate,
            "min_games": self.min_games,
            "stockfish_engine_path": self.stockfish_engine_path,
            "stockfish_initial_skill_level": self.stockfish_initial_skill_level,
            "stockfish_max_skill_level": self.stockfish_max_skill_level,
            "stockfish_skill_step": self.stockfish_skill_step,
            "stockfish_depth": self.stockfish_depth,
            "stockfish_nodes": self.stockfish_nodes,
            "stockfish_movetime_ms": self.stockfish_movetime_ms,
            "stockfish_threads": self.stockfish_threads,
            "stockfish_hash_mb": self.stockfish_hash_mb,
            "stockfish_engine_options": dict(self.stockfish_engine_options),
            "spec_version": self.spec_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ArenaProgressionSpec":
        return cls(
            safe_score_rate=float(payload.get("safe_score_rate", 0.55)),
            min_games=int(payload.get("min_games", 2)),
            stockfish_engine_path=str(
                payload.get("stockfish_engine_path", "/usr/games/stockfish18")
            ),
            stockfish_initial_skill_level=int(payload.get("stockfish_initial_skill_level", 0)),
            stockfish_max_skill_level=int(payload.get("stockfish_max_skill_level", 20)),
            stockfish_skill_step=int(payload.get("stockfish_skill_step", 1)),
            stockfish_depth=_optional_int(payload.get("stockfish_depth", 4)),
            stockfish_nodes=_optional_int(payload.get("stockfish_nodes")),
            stockfish_movetime_ms=_optional_int(payload.get("stockfish_movetime_ms")),
            stockfish_threads=int(payload.get("stockfish_threads", 1)),
            stockfish_hash_mb=_optional_int(payload.get("stockfish_hash_mb", 16)),
            stockfish_engine_options={
                str(name): str(value)
                for name, value in dict(payload.get("stockfish_engine_options") or {}).items()
            },
            spec_version=int(payload.get("spec_version", ARENA_PROGRESS_POLICY_VERSION)),
        )


@dataclass(frozen=True)
class Phase10LineageSpec:
    """One sequential Phase-10 lineage controlled by the master loop."""

    name: str
    seed_phase10_config_path: str
    output_root: str
    enabled: bool = True
    label_job_name: str | None = None
    idle_phase10_job_names: tuple[str, ...] | None = None
    seed_raw_dirs: tuple[str, ...] = ()
    seed_warm_start_checkpoint: str | None = None
    use_all_available_labeled_positions: bool = False
    bootstrap_generation_from_seed_artifacts: bool = False
    bootstrap_generation1_skip_training: bool = False
    max_generations: int = 1
    on_accept: str = "continue_training"
    on_reject: str = "stop"
    promotion_thresholds: PromotionThresholds = field(default_factory=PromotionThresholds)
    arena_progression: ArenaProgressionSpec = field(default_factory=ArenaProgressionSpec)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("lineage name must be non-empty")
        if not self.seed_phase10_config_path:
            raise ValueError("lineage seed_phase10_config_path must be non-empty")
        if not self.output_root:
            raise ValueError("lineage output_root must be non-empty")
        if self.max_generations <= 0:
            raise ValueError("lineage max_generations must be positive")
        if self.on_accept not in ACCEPT_POLICIES:
            raise ValueError(f"unsupported on_accept policy: {self.on_accept}")
        if self.on_reject not in REJECT_POLICIES:
            raise ValueError(f"unsupported on_reject policy: {self.on_reject}")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed_phase10_config_path": self.seed_phase10_config_path,
            "output_root": self.output_root,
            "enabled": self.enabled,
            "label_job_name": self.label_job_name,
            "idle_phase10_job_names": (
                None
                if self.idle_phase10_job_names is None
                else list(self.idle_phase10_job_names)
            ),
            "seed_raw_dirs": list(self.seed_raw_dirs),
            "seed_warm_start_checkpoint": self.seed_warm_start_checkpoint,
            "use_all_available_labeled_positions": self.use_all_available_labeled_positions,
            "bootstrap_generation_from_seed_artifacts": self.bootstrap_generation_from_seed_artifacts,
            "bootstrap_generation1_skip_training": self.bootstrap_generation1_skip_training,
            "max_generations": self.max_generations,
            "on_accept": self.on_accept,
            "on_reject": self.on_reject,
            "promotion_thresholds": self.promotion_thresholds.to_dict(),
            "arena_progression": self.arena_progression.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Phase10LineageSpec":
        return cls(
            name=str(payload["name"]),
            seed_phase10_config_path=str(payload["seed_phase10_config_path"]),
            output_root=str(payload["output_root"]),
            enabled=bool(payload.get("enabled", True)),
            label_job_name=(
                str(payload["label_job_name"]) if payload.get("label_job_name") is not None else None
            ),
            idle_phase10_job_names=(
                None
                if "idle_phase10_job_names" not in payload
                or payload.get("idle_phase10_job_names") is None
                else tuple(str(entry) for entry in list(payload.get("idle_phase10_job_names") or []))
            ),
            seed_raw_dirs=tuple(str(entry) for entry in list(payload.get("seed_raw_dirs") or [])),
            seed_warm_start_checkpoint=(
                str(payload["seed_warm_start_checkpoint"])
                if payload.get("seed_warm_start_checkpoint") is not None
                else None
            ),
            use_all_available_labeled_positions=bool(
                payload.get("use_all_available_labeled_positions", False)
            ),
            bootstrap_generation_from_seed_artifacts=bool(
                payload.get("bootstrap_generation_from_seed_artifacts", False)
            ),
            bootstrap_generation1_skip_training=bool(
                payload.get("bootstrap_generation1_skip_training", False)
            ),
            max_generations=int(payload.get("max_generations", 1)),
            on_accept=str(payload.get("on_accept", "continue_training")),
            on_reject=str(payload.get("on_reject", "stop")),
            promotion_thresholds=PromotionThresholds.from_dict(
                dict(payload.get("promotion_thresholds") or {})
            ),
            arena_progression=ArenaProgressionSpec.from_dict(
                dict(payload.get("arena_progression") or {})
            ),
        )


@dataclass(frozen=True)
class MasterSpec:
    """Top-level configuration for the MySQL orchestration master."""

    name: str
    output_root: str
    label_jobs: tuple[LabelPgnCorpusJobSpec, ...] = ()
    idle_phase10_jobs: tuple[IdlePhase10ArtifactJobSpec, ...] = ()
    lineages: tuple[Phase10LineageSpec, ...] = ()
    poll_interval_seconds: float = 30.0
    spec_version: int = MASTER_SPEC_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("master name must be non-empty")
        if not self.output_root:
            raise ValueError("master output_root must be non-empty")
        if self.poll_interval_seconds <= 0.0:
            raise ValueError("master poll_interval_seconds must be positive")
        if self.spec_version != MASTER_SPEC_VERSION:
            raise ValueError(f"unsupported master spec version: {self.spec_version}")

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "output_root": self.output_root,
            "poll_interval_seconds": self.poll_interval_seconds,
            "label_jobs": [job.to_dict() for job in self.label_jobs],
            "idle_phase10_jobs": [job.to_dict() for job in self.idle_phase10_jobs],
            "lineages": [lineage.to_dict() for lineage in self.lineages],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MasterSpec":
        return cls(
            spec_version=int(payload.get("spec_version", MASTER_SPEC_VERSION)),
            name=str(payload["name"]),
            output_root=str(payload["output_root"]),
            poll_interval_seconds=float(payload.get("poll_interval_seconds", 30.0)),
            label_jobs=tuple(
                LabelPgnCorpusJobSpec.from_dict(dict(entry))
                for entry in list(payload.get("label_jobs") or [])
            ),
            idle_phase10_jobs=tuple(
                IdlePhase10ArtifactJobSpec.from_dict(dict(entry))
                for entry in list(payload.get("idle_phase10_jobs") or [])
            ),
            lineages=tuple(
                Phase10LineageSpec.from_dict(dict(entry))
                for entry in list(payload.get("lineages") or [])
            ),
        )


def load_master_spec(path: Path) -> MasterSpec:
    """Load one orchestrator master spec from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("master spec must be a JSON object")
    return MasterSpec.from_dict(payload)


def write_master_spec(path: Path, spec: MasterSpec) -> None:
    """Write one orchestrator master spec to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class OrchestratorMaster:
    """Reconcile label jobs and Phase-10 lineages against the MySQL control plane."""

    def __init__(
        self,
        *,
        db: OrchestratorDB,
        controller: OrchestratorController,
        repo_root: Path,
        spec: MasterSpec,
        spec_path: Path,
        usage_ledger: LineageTrainingUsageLedger | None = None,
    ) -> None:
        self._db = db
        self._controller = controller
        self._repo_root = repo_root
        self._spec = spec
        self._spec_path = spec_path
        self._output_root = resolve_repo_path(repo_root, Path(spec.output_root))
        self._usage_ledger = usage_ledger

    @property
    def spec(self) -> MasterSpec:
        return self._spec

    @property
    def db(self) -> OrchestratorDB:
        return self._db

    @property
    def controller(self) -> OrchestratorController:
        return self._controller

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def spec_path(self) -> Path:
        return self._spec_path

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def usage_ledger(self) -> LineageTrainingUsageLedger:
        if self._usage_ledger is None:
            db_config = getattr(self._db, "config", None)
            self._usage_ledger = (
                MySQLLineageTrainingUsageLedger(db_config)
                if db_config is not None
                else InMemoryLineageTrainingUsageLedger()
            )
        return self._usage_ledger

    def replace_spec(self, spec: MasterSpec, *, spec_path: Path | None = None) -> None:
        """Replace the active master spec without rebuilding the process."""
        self._spec = spec
        if spec_path is not None:
            self._spec_path = spec_path
        self._output_root = resolve_repo_path(self._repo_root, Path(self._spec.output_root))

    def reconcile_once(self) -> dict[str, Any]:
        """Submit or evaluate whatever the master can advance in one pass."""
        self._output_root.mkdir(parents=True, exist_ok=True)
        self.usage_ledger.ensure_schema()

        label_states = {
            job.name: self._reconcile_label_job(job)
            for job in self._spec.label_jobs
        }
        idle_phase10_states = {
            job.name: self._reconcile_idle_phase10_job(job)
            for job in self._spec.idle_phase10_jobs
        }
        lineage_states = {
            lineage.name: self._reconcile_lineage(lineage, label_states=label_states)
            for lineage in self._spec.lineages
        }
        summary = {
            "master_name": self._spec.name,
            "spec_path": str(self._spec_path),
            "output_root": str(self._output_root),
            "label_jobs": label_states,
            "idle_phase10_jobs": idle_phase10_states,
            "lineages": lineage_states,
            "all_terminal": all(
                state.get("terminal", False)
                for state in (
                    *label_states.values(),
                    *idle_phase10_states.values(),
                    *lineage_states.values(),
                )
            ),
            "timestamp": int(time.time()),
        }
        summary_path = self._output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    def run_forever(self, *, poll_interval_seconds: float | None = None) -> None:
        """Poll forever, reconciling all configured jobs and lineages."""
        sleep_seconds = float(
            poll_interval_seconds
            if poll_interval_seconds is not None
            else self._spec.poll_interval_seconds
        )
        while True:
            summary = self._reconcile_with_transient_db_guard()
            if summary is None:
                time.sleep(min(sleep_seconds, 5.0))
                continue
            time.sleep(sleep_seconds)

    def run_until_terminal(
        self,
        *,
        poll_interval_seconds: float | None = None,
        max_cycles: int = 1000,
    ) -> dict[str, Any]:
        """Poll until all configured jobs/lineages are terminal or the cycle budget is hit."""
        sleep_seconds = float(
            poll_interval_seconds
            if poll_interval_seconds is not None
            else self._spec.poll_interval_seconds
        )
        summary: dict[str, Any] = {}
        for _cycle_index in range(max_cycles):
            guarded_summary = self._reconcile_with_transient_db_guard()
            if guarded_summary is None:
                time.sleep(min(sleep_seconds, 5.0))
                continue
            summary = guarded_summary
            if bool(summary.get("all_terminal")):
                return summary
            time.sleep(sleep_seconds)
        raise TimeoutError(f"master did not reach a terminal state after {max_cycles} cycles")

    def _reconcile_with_transient_db_guard(self) -> dict[str, Any] | None:
        try:
            return self.reconcile_once()
        except (
            pymysql.err.InterfaceError,
            pymysql.err.InternalError,
            pymysql.err.OperationalError,
        ) as exc:
            error_code = int(exc.args[0]) if exc.args else 0
            if error_code not in _TRANSIENT_MYSQL_ERROR_CODES:
                raise
            print(
                f"[ek-master][warn] transient MySQL error during reconcile: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

    def _reconcile_label_job(self, job: LabelPgnCorpusJobSpec) -> dict[str, Any]:
        if not job.enabled:
            return {
                "status": "disabled",
                "terminal": True,
                "work_dir": str(job.work_dir),
            }
        campaigns = self._matching_campaigns(master_job_type="label_job", job_name=job.name)
        latest = campaigns[-1] if campaigns else None
        completed_snapshot = _completed_label_snapshot(Path(job.work_dir))
        if latest is None and completed_snapshot is not None:
            return {
                "status": "completed_external",
                "terminal": True,
                "work_dir": str(job.work_dir),
                "summary_path": str(completed_snapshot["summary_path"]),
            }
        if latest is None:
            config_path = self._write_label_job_config(job)
            submission = self._controller.submit_label_pgn_corpus_campaign(
                config_path=config_path,
                campaign_metadata={
                    "master_name": self._spec.name,
                    "master_job_type": "label_job",
                    "job_name": job.name,
                },
            )
            return {
                "status": "submitted",
                "terminal": False,
                "campaign_id": int(submission["campaign_id"]),
                "config_path": str(config_path),
            }
        if latest.status in TERMINAL_CAMPAIGN_STATES:
            if latest.status == "succeeded":
                summary_path = Path(job.work_dir) / "summary.json"
                return {
                    "status": "completed",
                    "terminal": True,
                    "campaign_id": latest.id,
                    "summary_path": str(summary_path),
                    "work_dir": str(job.work_dir),
                }
            return {
                "status": "failed",
                "terminal": True,
                "campaign_id": latest.id,
                "config_path": latest.config_path,
            }
        return {
            "status": latest.status,
            "terminal": False,
            "campaign_id": latest.id,
            "config_path": latest.config_path,
        }

    def _reconcile_idle_phase10_job(self, job: IdlePhase10ArtifactJobSpec) -> dict[str, Any]:
        if not job.enabled:
            return {
                "status": "disabled",
                "terminal": True,
                "work_root": str(job.work_root),
            }
        campaigns = self._matching_campaigns(master_job_type="idle_phase10_job", job_name=job.name)
        latest = campaigns[-1] if campaigns else None
        summary_path = resolve_repo_path(self._repo_root, Path(job.work_root)) / "summary.json"
        if latest is None and summary_path.exists():
            return {
                "status": "completed_external",
                "terminal": True,
                "work_root": job.work_root,
                "summary_path": str(summary_path),
            }
        if latest is None:
            config_path = self._write_idle_phase10_job_config(job)
            submission = self._controller.submit_idle_phase10_artifact_campaign(
                config_path=config_path,
                campaign_metadata={
                    "master_name": self._spec.name,
                    "master_job_type": "idle_phase10_job",
                    "job_name": job.name,
                },
            )
            return {
                "status": "submitted",
                "terminal": False,
                "campaign_id": int(submission["campaign_id"]),
                "config_path": str(config_path),
            }
        if latest.status in TERMINAL_CAMPAIGN_STATES:
            if latest.status == "succeeded":
                return {
                    "status": "completed",
                    "terminal": True,
                    "campaign_id": latest.id,
                    "summary_path": str(summary_path),
                    "work_root": job.work_root,
                }
            return {
                "status": "failed",
                "terminal": True,
                "campaign_id": latest.id,
                "config_path": latest.config_path,
            }
        return {
            "status": latest.status,
            "terminal": False,
            "campaign_id": latest.id,
            "config_path": latest.config_path,
        }

    def _reconcile_lineage(
        self,
        lineage: Phase10LineageSpec,
        *,
        label_states: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not lineage.enabled:
            return {
                "status": "disabled",
                "terminal": True,
                "active_arena_opponents": [],
            }
        label_work_dir: Path | None = None
        if lineage.label_job_name is not None:
            label_state = dict(label_states.get(lineage.label_job_name) or {})
            if not label_state:
                raise ValueError(f"unknown label job referenced by lineage: {lineage.label_job_name}")
            if not bool(label_state.get("terminal")) or str(label_state.get("status")) == "failed":
                return {
                    "status": "waiting_for_label",
                    "terminal": False,
                    "label_job_name": lineage.label_job_name,
                }
            label_work_dir = resolve_repo_path(
                self._repo_root,
                Path(str(label_state.get("work_dir") or "")),
            )
        registry = self._load_or_create_lineage_arena_registry(lineage)

        campaigns = self._matching_campaigns(master_job_type="phase10_lineage", job_name=lineage.name)
        models = self._matching_models(lineage_name=lineage.name)
        usage_state = self._backfill_lineage_generation_usage(
            lineage=lineage,
            campaigns=campaigns,
            models=models,
        )
        latest_campaign = campaigns[-1] if campaigns else None
        next_generation = 1 if latest_campaign is None else int(latest_generation(campaigns)) + 1

        if latest_campaign is None:
            submission = self._submit_generation(
                lineage=lineage,
                generation=1,
                parent_model=None,
                warm_start_checkpoint=lineage.seed_warm_start_checkpoint,
                label_work_dir=label_work_dir,
                arena_registry=registry,
            )
            return {
                "status": "submitted_generation",
                "terminal": False,
                "generation": 1,
                "campaign_id": int(submission["campaign_id"]),
                "config_path": str(submission["config_path"]),
                "active_arena_opponents": list(_active_registry_opponent_names(registry)),
                "training_data_usage": usage_state,
            }

        latest_generation_value = int(latest_campaign.metadata.get("generation", 0) or 0)
        if latest_campaign.status not in TERMINAL_CAMPAIGN_STATES:
            return {
                "status": latest_campaign.status,
                "terminal": False,
                "generation": latest_generation_value,
                "campaign_id": latest_campaign.id,
                "active_arena_opponents": list(_active_registry_opponent_names(registry)),
                "training_data_usage": usage_state,
            }

        if latest_campaign.status == "failed":
            return {
                "status": "failed",
                "terminal": True,
                "generation": latest_generation_value,
                "campaign_id": latest_campaign.id,
                "training_data_usage": usage_state,
            }

        current_model = _latest_model_for_campaign(models=models, campaign_id=latest_campaign.id)
        evaluation = self._evaluate_completed_generation(
            lineage=lineage,
            campaign=latest_campaign,
            model=current_model,
            arena_registry=registry,
        )
        if current_model is not None:
            self._db.update_model_record(
                current_model.id,
                promotion_score=float(evaluation["promotion_score"]),
                status=str(evaluation["model_status"]),
            )

        feedback_state = self._reconcile_generation_feedback(
            lineage=lineage,
            generation=latest_generation_value,
            campaign=latest_campaign,
        )
        lineage_state = {
            **evaluation,
            "generation": latest_generation_value,
            "campaign_id": latest_campaign.id,
            "arena_feedback": feedback_state,
            "training_data_usage": usage_state,
        }
        if str(feedback_state.get("status")) == "failed":
            return {
                **lineage_state,
                "status": "feedback_failed",
                "terminal": True,
            }
        if not bool(feedback_state.get("terminal")):
            return {
                **lineage_state,
                "status": f"waiting_for_feedback_{feedback_state['status']}",
                "terminal": False,
            }

        if latest_generation_value >= lineage.max_generations:
            return {
                **lineage_state,
                "status": "completed",
                "terminal": True,
            }

        existing_next = [
            campaign
            for campaign in campaigns
            if int(campaign.metadata.get("generation", 0) or 0) == next_generation
        ]
        if existing_next:
            next_campaign = existing_next[-1]
            return {
                **lineage_state,
                "status": f"next_generation_{next_campaign.status}",
                "terminal": False,
                "next_generation": next_generation,
                "next_campaign_id": next_campaign.id,
            }

        if bool(evaluation["accepted"]):
            if lineage.on_accept == "stop":
                return {
                    **lineage_state,
                    "status": "accepted_stop",
                    "terminal": True,
                }
            submission = self._submit_generation(
                lineage=lineage,
                generation=next_generation,
                parent_model=current_model,
                warm_start_checkpoint=(
                    current_model.checkpoint_path if current_model is not None else None
                ),
                label_work_dir=label_work_dir,
                arena_registry=registry,
            )
            return {
                **lineage_state,
                "status": "accepted_submitted_next",
                "terminal": False,
                "next_generation": next_generation,
                "next_campaign_id": int(submission["campaign_id"]),
            }

        if lineage.on_reject == "restart_from_seed":
            submission = self._submit_generation(
                lineage=lineage,
                generation=next_generation,
                parent_model=None,
                warm_start_checkpoint=None,
                label_work_dir=label_work_dir,
                arena_registry=registry,
            )
            return {
                **lineage_state,
                "status": "rejected_restarted_from_seed",
                "terminal": False,
                "next_generation": next_generation,
                "next_campaign_id": int(submission["campaign_id"]),
            }

        return {
            **lineage_state,
            "status": "rejected_stop",
            "terminal": True,
        }

    def _submit_generation(
        self,
        *,
        lineage: Phase10LineageSpec,
        generation: int,
        parent_model: ModelRow | None,
        warm_start_checkpoint: str | None,
        label_work_dir: Path | None,
        arena_registry: Mapping[str, Any],
    ) -> dict[str, Any]:
        generation_paths = self._materialize_generation_configs(
            lineage=lineage,
            generation=generation,
            parent_model=parent_model,
            warm_start_checkpoint=warm_start_checkpoint,
            label_work_dir=label_work_dir,
            arena_registry=arena_registry,
        )
        return self._controller.submit_phase10_campaign(
            config_path=generation_paths["phase10_config_path"],
            kind="phase10_master",
            campaign_metadata={
                "master_name": self._spec.name,
                "master_job_type": "phase10_lineage",
                "job_name": lineage.name,
                "generation": generation,
                "generated_root": str(generation_paths["generation_root"]),
                "label_job_name": lineage.label_job_name,
            },
            model_metadata={
                "master_name": self._spec.name,
                "lineage_name": lineage.name,
                "generation": generation,
                "generated_root": str(generation_paths["generation_root"]),
                "warm_start_checkpoint": warm_start_checkpoint,
            },
            generation=generation,
            parent_model_id=(parent_model.id if parent_model is not None else None),
        )

    def _materialize_generation_configs(
        self,
        *,
        lineage: Phase10LineageSpec,
        generation: int,
        parent_model: ModelRow | None,
        warm_start_checkpoint: str | None,
        label_work_dir: Path | None,
        arena_registry: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        generation_root = resolve_repo_path(
            self._repo_root,
            Path(lineage.output_root) / f"generation_{generation:04d}",
        )
        config_root = generation_root / "configs"
        outputs_root = generation_root / "outputs"
        config_root.mkdir(parents=True, exist_ok=True)
        outputs_root.mkdir(parents=True, exist_ok=True)

        seed_phase10_path = resolve_repo_path(
            self._repo_root, Path(lineage.seed_phase10_config_path)
        )
        seed_phase10_spec = load_phase10_lapv1_arena_campaign_spec(seed_phase10_path)
        seed_phase10_payload = _load_json_object(seed_phase10_path)
        seed_train_path = resolve_repo_path(
            self._repo_root, Path(seed_phase10_spec.lapv1_config_path)
        )
        seed_train_payload = _load_json_object(seed_train_path)
        seed_agent_path = resolve_repo_path(
            self._repo_root, Path(seed_phase10_spec.lapv1_agent_spec_path)
        )
        seed_agent_spec = load_selfplay_agent_spec(seed_agent_path)

        phase10_name = f"{self._spec.name}_{lineage.name}_g{generation:04d}"
        phase10_output_root = outputs_root / "campaign"
        reuse_seed_artifacts = generation == 1 and lineage.bootstrap_generation_from_seed_artifacts
        workflow_output_root = (
            resolve_repo_path(self._repo_root, Path(seed_phase10_spec.workflow_output_root))
            if reuse_seed_artifacts
            else outputs_root / "workflow"
        )
        train_dataset_dir = (
            resolve_repo_path(self._repo_root, Path(seed_phase10_spec.train_dataset_dir))
            if reuse_seed_artifacts
            else outputs_root / "dataset_train"
        )
        verify_dataset_dir = (
            resolve_repo_path(self._repo_root, Path(seed_phase10_spec.verify_dataset_dir))
            if reuse_seed_artifacts
            else outputs_root / "dataset_verify"
        )
        model_output_dir = outputs_root / "model"
        bundle_dir = model_output_dir / "bundle"
        train_config_path = config_root / "train_config.json"
        agent_spec_path = config_root / "agent_spec.json"
        phase10_config_path = config_root / "campaign.json"
        verify_output_path = workflow_output_root / "all_unique_verify_v1" / "lapv1_test.jsonl"
        opponents_root = config_root / "arena_opponents"
        merged_raw_dir = (
            resolve_repo_path(self._repo_root, Path(seed_phase10_spec.merged_raw_dir))
            if reuse_seed_artifacts
            else self._prepare_generation_raw_corpus(
                lineage=lineage,
                generation=generation,
                generation_root=generation_root,
                base_label_work_dir=label_work_dir,
            )
        )

        train_replacements: dict[str, str] = {
            str(seed_train_payload["output_dir"]): str(model_output_dir),
            str(seed_train_payload["export"]["bundle_dir"]): str(bundle_dir),
        }
        if not reuse_seed_artifacts:
            train_replacements[str(seed_phase10_spec.workflow_output_root)] = str(workflow_output_root)
            for workflow_root in _infer_train_workflow_roots(seed_train_payload):
                train_replacements.setdefault(str(workflow_root), str(workflow_output_root))
        train_payload = _replace_path_prefixes(
            seed_train_payload,
            replacements=train_replacements,
        )
        train_payload["output_dir"] = str(model_output_dir)
        train_payload["export"]["bundle_dir"] = str(bundle_dir)
        if warm_start_checkpoint is not None:
            train_payload["initial_checkpoint"] = str(warm_start_checkpoint)
        else:
            initial_checkpoint = train_payload.get("initial_checkpoint")
            if initial_checkpoint is not None:
                initial_checkpoint_path = resolve_repo_path(
                    self._repo_root,
                    Path(str(initial_checkpoint)),
                )
                if not initial_checkpoint_path.exists():
                    train_payload.pop("initial_checkpoint", None)

        agent_spec = SelfplayAgentSpec(
            **{
                **seed_agent_spec.to_dict(),
                "name": phase10_name,
                "lapv1_checkpoint": str(bundle_dir / str(train_payload["export"]["checkpoint_name"])),
                "metadata": {
                    **dict(seed_agent_spec.metadata),
                    "master_name": self._spec.name,
                    "lineage_name": lineage.name,
                    "generation": generation,
                    "previous_checkpoint": (
                        parent_model.checkpoint_path if parent_model is not None else None
                    ),
                },
            }
        )
        benchmark_agent_specs = self._materialize_generation_arena_opponents(
            lineage=lineage,
            generation=generation,
            arena_registry=(
                arena_registry
                if arena_registry is not None
                else self._load_or_create_lineage_arena_registry(lineage)
            ),
            output_root=opponents_root,
        )

        phase10_payload = dict(seed_phase10_payload)
        phase10_payload["name"] = phase10_name
        phase10_payload["model_label"] = f"{seed_phase10_spec.model_label}-g{generation:04d}"
        phase10_payload["output_root"] = str(phase10_output_root)
        phase10_payload["merged_raw_dir"] = str(
            merged_raw_dir if merged_raw_dir is not None else phase10_payload["merged_raw_dir"]
        )
        phase10_payload["train_dataset_dir"] = str(train_dataset_dir)
        phase10_payload["verify_dataset_dir"] = str(verify_dataset_dir)
        phase10_payload["workflow_output_root"] = str(workflow_output_root)
        phase10_payload["reuse_existing_artifacts"] = reuse_seed_artifacts
        phase10_payload["skip_training"] = bool(
            generation == 1 and lineage.bootstrap_generation1_skip_training
        )
        phase10_payload["lapv1_config_path"] = str(train_config_path)
        phase10_payload["lapv1_agent_spec_path"] = str(agent_spec_path)
        phase10_payload["lapv1_verify_output_path"] = str(verify_output_path)
        phase10_payload["warm_start_source_checkpoint"] = warm_start_checkpoint
        phase10_payload["reference_agents"] = []
        phase10_payload["reference_active_agent_specs_dir"] = None
        phase10_payload["reference_excluded_agents"] = []
        phase10_payload["top_reference_agents_count"] = 0
        phase10_payload["benchmark_agent_specs"] = benchmark_agent_specs

        Phase10Lapv1ArenaCampaignSpec.from_dict(phase10_payload)
        _assert_lapv1_train_payload(train_payload)

        train_config_path.write_text(
            json.dumps(train_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_selfplay_agent_spec(agent_spec_path, agent_spec)
        phase10_config_path.write_text(
            json.dumps(phase10_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        generation_summary_path = generation_root / "summary.json"
        generation_summary_path.write_text(
            json.dumps(
                {
                    "master_name": self._spec.name,
                    "lineage_name": lineage.name,
                    "generation": generation,
                    "parent_model_id": parent_model.id if parent_model is not None else None,
                    "warm_start_checkpoint": warm_start_checkpoint,
                    "label_work_dir": str(label_work_dir) if label_work_dir is not None else None,
                    "merged_raw_dir": str(phase10_payload["merged_raw_dir"]),
                    "merged_raw_selection_summary_path": str(
                        Path(str(phase10_payload["merged_raw_dir"])) / "selection_summary.json"
                    ),
                    "reuse_existing_artifacts": reuse_seed_artifacts,
                    "feedback_raw_dirs": [
                        str(path) for path in self._lineage_feedback_raw_dirs(lineage, generation=generation)
                    ],
                    "seed_raw_dirs": [str(path) for path in self._lineage_seed_raw_dirs(lineage)],
                    "idle_raw_dirs": [
                        str(path) for _name, path in self._lineage_idle_raw_dirs(lineage)
                    ],
                    "use_all_available_labeled_positions": (
                        lineage.use_all_available_labeled_positions
                    ),
                    "seed_phase10_config_path": str(seed_phase10_path),
                    "train_config_path": str(train_config_path),
                    "agent_spec_path": str(agent_spec_path),
                    "phase10_config_path": str(phase10_config_path),
                    "active_arena_opponents": sorted(benchmark_agent_specs),
                    "arena_registry_path": str(self._lineage_arena_registry_path(lineage)),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "generation_root": generation_root,
            "train_config_path": train_config_path,
            "agent_spec_path": agent_spec_path,
            "phase10_config_path": phase10_config_path,
        }

    def _lineage_output_root(self, lineage: Phase10LineageSpec) -> Path:
        return resolve_repo_path(self._repo_root, Path(lineage.output_root))

    def _lineage_feedback_root(self, lineage: Phase10LineageSpec) -> Path:
        return self._lineage_output_root(lineage) / "feedback"

    def _generation_feedback_root(self, lineage: Phase10LineageSpec, generation: int) -> Path:
        return self._lineage_feedback_root(lineage) / f"generation_{generation:04d}"

    def _generation_feedback_pgn_root(self, lineage: Phase10LineageSpec, generation: int) -> Path:
        return self._generation_feedback_root(lineage, generation) / "pgns"

    def _generation_feedback_label_work_dir(
        self,
        lineage: Phase10LineageSpec,
        generation: int,
    ) -> Path:
        return self._generation_feedback_root(lineage, generation) / "label_work"

    def _generation_feedback_label_config_path(
        self,
        lineage: Phase10LineageSpec,
        generation: int,
    ) -> Path:
        return self._generation_feedback_root(lineage, generation) / "label_config.json"

    def _generation_feedback_export_summary_path(
        self,
        lineage: Phase10LineageSpec,
        generation: int,
    ) -> Path:
        return self._generation_feedback_root(lineage, generation) / "pgn_export_summary.json"

    def _base_label_job_for_lineage(
        self,
        lineage: Phase10LineageSpec,
    ) -> LabelPgnCorpusJobSpec | None:
        if lineage.label_job_name is None:
            return None
        for job in self._spec.label_jobs:
            if job.name == lineage.label_job_name:
                return job
        return None

    def _reconcile_generation_feedback(
        self,
        *,
        lineage: Phase10LineageSpec,
        generation: int,
        campaign: CampaignRow,
    ) -> dict[str, Any]:
        export_summary = self._materialize_generation_feedback_pgns(
            lineage=lineage,
            generation=generation,
            campaign=campaign,
        )
        if int(export_summary["pgn_file_count"]) <= 0:
            return {
                "status": "no_games",
                "terminal": True,
                "generation": generation,
                "pgn_file_count": 0,
                "export_summary_path": str(
                    self._generation_feedback_export_summary_path(lineage, generation)
                ),
            }

        work_dir = self._generation_feedback_label_work_dir(lineage, generation)
        feedback_campaigns = self._matching_generation_feedback_campaigns(
            lineage_name=lineage.name,
            generation=generation,
        )
        latest = feedback_campaigns[-1] if feedback_campaigns else None
        completed_snapshot = _completed_label_snapshot(work_dir)
        if latest is None and completed_snapshot is not None:
            return {
                "status": "completed_external",
                "terminal": True,
                "generation": generation,
                "work_dir": str(work_dir),
                "summary_path": str(completed_snapshot["summary_path"]),
                "export_summary_path": str(
                    self._generation_feedback_export_summary_path(lineage, generation)
                ),
            }
        if latest is None:
            job = self._build_generation_feedback_label_job(
                lineage=lineage,
                generation=generation,
            )
            config_path = self._generation_feedback_label_config_path(lineage, generation)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(job.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            submission = self._controller.submit_label_pgn_corpus_campaign(
                config_path=config_path,
                campaign_metadata={
                    "master_name": self._spec.name,
                    "master_job_type": "generation_feedback_label",
                    "job_name": lineage.name,
                    "generation": generation,
                },
            )
            return {
                "status": "submitted",
                "terminal": False,
                "generation": generation,
                "campaign_id": int(submission["campaign_id"]),
                "config_path": str(config_path),
                "work_dir": str(work_dir),
                "export_summary_path": str(
                    self._generation_feedback_export_summary_path(lineage, generation)
                ),
            }
        if latest.status in TERMINAL_CAMPAIGN_STATES:
            if latest.status == "succeeded":
                return {
                    "status": "completed",
                    "terminal": True,
                    "generation": generation,
                    "campaign_id": latest.id,
                    "work_dir": str(work_dir),
                    "summary_path": str(work_dir / "summary.json"),
                    "export_summary_path": str(
                        self._generation_feedback_export_summary_path(lineage, generation)
                    ),
                }
            return {
                "status": "failed",
                "terminal": True,
                "generation": generation,
                "campaign_id": latest.id,
                "config_path": latest.config_path,
                "work_dir": str(work_dir),
            }
        return {
            "status": latest.status,
            "terminal": False,
            "generation": generation,
            "campaign_id": latest.id,
            "config_path": latest.config_path,
            "work_dir": str(work_dir),
        }

    def _materialize_generation_feedback_pgns(
        self,
        *,
        lineage: Phase10LineageSpec,
        generation: int,
        campaign: CampaignRow,
    ) -> dict[str, Any]:
        campaign_spec = load_phase10_lapv1_arena_campaign_spec(Path(campaign.config_path))
        campaign_output_root = resolve_repo_path(self._repo_root, Path(campaign_spec.output_root))
        pgn_root = self._generation_feedback_pgn_root(lineage, generation)
        arena_source_root = campaign_output_root / "arena" / "sessions"
        selfplay_source_root = campaign_output_root / "pre_verify_selfplay" / "sessions"
        arena_pgn_count = _export_session_tree_to_pgn_dir(
            source_root=arena_source_root,
            output_root=pgn_root / "arena",
            event_name=f"{lineage.name}_arena_feedback",
            phase_name="arena",
            generation=generation,
        )
        selfplay_pgn_count = _export_session_tree_to_pgn_dir(
            source_root=selfplay_source_root,
            output_root=pgn_root / "selfplay",
            event_name=f"{lineage.name}_selfplay_feedback",
            phase_name="pre_verify_selfplay",
            generation=generation,
        )
        summary = {
            "generation": generation,
            "campaign_id": campaign.id,
            "campaign_name": campaign.name,
            "pgn_root": str(pgn_root),
            "arena_sessions_root": str(arena_source_root),
            "selfplay_sessions_root": str(selfplay_source_root),
            "arena_pgn_count": arena_pgn_count,
            "selfplay_pgn_count": selfplay_pgn_count,
            "pgn_file_count": arena_pgn_count + selfplay_pgn_count,
        }
        summary_path = self._generation_feedback_export_summary_path(lineage, generation)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _build_generation_feedback_label_job(
        self,
        *,
        lineage: Phase10LineageSpec,
        generation: int,
    ) -> LabelPgnCorpusJobSpec:
        base_job = self._base_label_job_for_lineage(lineage)
        split_seed_base = (
            base_job.split_seed
            if base_job is not None
            else f"{self._spec.name}:{lineage.name}:feedback"
        )
        return LabelPgnCorpusJobSpec(
            name=f"{self._spec.name}_{lineage.name}_g{generation:04d}_feedback_label",
            pgn_root=str(self._generation_feedback_pgn_root(lineage, generation)),
            work_dir=str(self._generation_feedback_label_work_dir(lineage, generation)),
            glob="**/*.pgn",
            engine_path=(
                base_job.engine_path if base_job is not None else "/usr/games/stockfish18"
            ),
            target_train_records=1_000_000_000,
            target_verify_records=1_000_000,
            min_ply=base_job.min_ply if base_job is not None else 8,
            max_ply=base_job.max_ply if base_job is not None else 80,
            ply_stride=base_job.ply_stride if base_job is not None else 2,
            engine_nodes=base_job.engine_nodes if base_job is not None else 1500,
            hash_mb=base_job.hash_mb if base_job is not None else 32,
            threads=base_job.threads if base_job is not None else 1,
            split_seed=f"{split_seed_base}:feedback:g{generation:04d}",
            verify_divisor=base_job.verify_divisor if base_job is not None else 1000,
            progress_every=base_job.progress_every if base_job is not None else 1000,
            max_games=0,
            export_jsonl_on_complete=True,
            complete_at_eof=True,
        )

    def _matching_generation_feedback_campaigns(
        self,
        *,
        lineage_name: str,
        generation: int,
    ) -> list[CampaignRow]:
        campaigns = self._db.list_campaign_records(limit=5000)
        matching = [
            campaign
            for campaign in campaigns
            if campaign.metadata.get("master_name") == self._spec.name
            and campaign.metadata.get("master_job_type") == "generation_feedback_label"
            and campaign.metadata.get("job_name") == lineage_name
            and int(campaign.metadata.get("generation", 0) or 0) == generation
        ]
        matching.sort(key=lambda campaign: campaign.id)
        return matching

    def _prepare_generation_raw_corpus(
        self,
        *,
        lineage: Phase10LineageSpec,
        generation: int,
        generation_root: Path,
        base_label_work_dir: Path | None,
    ) -> Path | None:
        source_dirs: list[tuple[str, Path]] = []
        if base_label_work_dir is not None and _available_raw_snapshot(
            base_label_work_dir,
            require_completed=True,
        ) is not None:
            source_dirs.append(("base_label", base_label_work_dir))
        source_dirs.extend(
            (
                f"seed_raw_{index:02d}",
                path,
            )
            for index, path in enumerate(self._lineage_seed_raw_dirs(lineage), start=1)
        )
        source_dirs.extend(
            (
                f"feedback_generation_{index:04d}",
                path,
            )
            for index, path in enumerate(
                self._lineage_feedback_raw_dirs(lineage, generation=generation),
                start=1,
            )
        )
        source_dirs.extend(self._lineage_idle_raw_dirs(lineage))
        unique_source_dirs: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for name, path in source_dirs:
            normalized = str(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_source_dirs.append((name, path))
        if (
            generation == 1
            and base_label_work_dir is not None
            and not lineage.seed_raw_dirs
            and not lineage.use_all_available_labeled_positions
            and len(unique_source_dirs) == 1
            and unique_source_dirs[0][1] == base_label_work_dir
        ):
            return base_label_work_dir
        if not unique_source_dirs:
            return self._previous_generation_merged_raw_dir(lineage, generation=generation)

        candidate_dir = generation_root / "inputs" / "raw_corpus_candidates"
        merge_phase5_raw_corpora(
            source_specs=[
                RawCorpusSourceSpec(
                    name=f"source_{index:02d}_{_safe_filename(name)}",
                    raw_dir=path,
                )
                for index, (name, path) in enumerate(unique_source_dirs, start=1)
            ],
            output_dir=candidate_dir,
        )
        target_counts = (
            _raw_corpus_counts(candidate_dir)
            if lineage.use_all_available_labeled_positions
            else self._desired_generation_raw_counts(
                lineage=lineage,
                generation=generation,
                base_label_work_dir=base_label_work_dir,
            )
        )
        output_dir = generation_root / "inputs" / "raw_corpus_merged"
        _materialize_usage_balanced_raw_corpus(
            candidate_dir=candidate_dir,
            output_dir=output_dir,
            master_name=self._spec.name,
            lineage_name=lineage.name,
            generation=generation,
            desired_train_records=int(target_counts["train"]),
            desired_verify_records=int(target_counts["verify"]),
            usage_ledger=self.usage_ledger,
            source_dirs=[path for _name, path in unique_source_dirs],
        )
        return output_dir

    def _lineage_seed_raw_dirs(self, lineage: Phase10LineageSpec) -> list[Path]:
        raw_dirs: list[Path] = []
        for raw_dir in lineage.seed_raw_dirs:
            resolved = resolve_repo_path(self._repo_root, Path(raw_dir))
            if not _raw_corpus_exists(resolved):
                continue
            raw_dirs.append(resolved)
        return raw_dirs

    def _desired_generation_raw_counts(
        self,
        *,
        lineage: Phase10LineageSpec,
        generation: int,
        base_label_work_dir: Path | None,
    ) -> dict[str, int]:
        previous_merged_raw_dir = self._previous_generation_merged_raw_dir(
            lineage,
            generation=generation,
        )
        if previous_merged_raw_dir is not None:
            return _raw_corpus_counts(previous_merged_raw_dir)
        if base_label_work_dir is not None:
            return _raw_corpus_counts(base_label_work_dir)
        return {"train": 0, "verify": 0}

    def _previous_generation_merged_raw_dir(
        self,
        lineage: Phase10LineageSpec,
        *,
        generation: int,
    ) -> Path | None:
        if generation <= 1:
            return None
        summary_path = self._lineage_output_root(lineage) / f"generation_{generation - 1:04d}" / "summary.json"
        if not summary_path.exists():
            return None
        summary = _load_json_object(summary_path)
        merged_raw_dir = summary.get("merged_raw_dir")
        if merged_raw_dir is None:
            return None
        resolved = resolve_repo_path(self._repo_root, Path(str(merged_raw_dir)))
        if not _raw_corpus_exists(resolved):
            return None
        return resolved

    def _selected_idle_phase10_jobs(
        self,
        lineage: Phase10LineageSpec,
    ) -> list[IdlePhase10ArtifactJobSpec]:
        jobs_by_name = {job.name: job for job in self._spec.idle_phase10_jobs if job.enabled}
        if lineage.idle_phase10_job_names is None:
            return list(jobs_by_name.values())
        selected: list[IdlePhase10ArtifactJobSpec] = []
        for job_name in lineage.idle_phase10_job_names:
            job = jobs_by_name.get(job_name)
            if job is None:
                raise ValueError(f"unknown or disabled idle phase10 job referenced by lineage: {job_name}")
            selected.append(job)
        return selected

    def _lineage_idle_raw_dirs(
        self,
        lineage: Phase10LineageSpec,
    ) -> list[tuple[str, Path]]:
        raw_dirs: list[tuple[str, Path]] = []
        for job in self._selected_idle_phase10_jobs(lineage):
            work_root = resolve_repo_path(self._repo_root, Path(job.work_root))
            label_shards_root = work_root / "label_shards"
            shard_dirs: list[Path] = []
            if label_shards_root.exists():
                shard_dirs = [
                    path
                    for path in sorted(label_shards_root.glob("shard_*"))
                    if _available_raw_snapshot(path, require_completed=False) is not None
                ]
            if shard_dirs:
                raw_dirs.extend(
                    (
                        f"idle_{job.name}_{path.name}",
                        path,
                    )
                    for path in shard_dirs
                )
                continue
            summary_path = work_root / "summary.json"
            if not summary_path.exists():
                continue
            summary = _load_json_object(summary_path)
            merged_raw_dir = summary.get("merged_raw_dir")
            if not merged_raw_dir:
                continue
            resolved = resolve_repo_path(self._repo_root, Path(str(merged_raw_dir)))
            if not _raw_corpus_exists(resolved):
                continue
            raw_dirs.append((f"idle_{job.name}_merged_raw", resolved))
        return raw_dirs

    def _lineage_feedback_raw_dirs(
        self,
        lineage: Phase10LineageSpec,
        *,
        generation: int,
    ) -> list[Path]:
        feedback_dirs: list[Path] = []
        for previous_generation in range(1, generation):
            work_dir = self._generation_feedback_label_work_dir(lineage, previous_generation)
            if (work_dir / "train_raw.jsonl").exists() and (work_dir / "verify_raw.jsonl").exists():
                feedback_dirs.append(work_dir)
        return feedback_dirs

    def _lineage_arena_registry_path(self, lineage: Phase10LineageSpec) -> Path:
        return self._lineage_output_root(lineage) / "arena_progression.json"

    def _load_or_create_lineage_arena_registry(
        self,
        lineage: Phase10LineageSpec,
    ) -> dict[str, Any]:
        registry_path = self._lineage_arena_registry_path(lineage)
        raw_registry = (
            _load_json_object(registry_path)
            if registry_path.exists()
            else self._build_initial_lineage_arena_registry(lineage)
        )
        registry = self._normalize_lineage_arena_registry(lineage, raw_registry)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return registry

    def _build_initial_lineage_arena_registry(
        self,
        lineage: Phase10LineageSpec,
    ) -> dict[str, Any]:
        seed_path = resolve_repo_path(self._repo_root, Path(lineage.seed_phase10_config_path))
        seed_spec = load_phase10_lapv1_arena_campaign_spec(seed_path)
        opponents: dict[str, dict[str, Any]] = {}
        active_specs_dir = (
            resolve_repo_path(self._repo_root, Path(seed_spec.reference_active_agent_specs_dir))
            if seed_spec.reference_active_agent_specs_dir is not None
            else None
        )
        if active_specs_dir is not None and active_specs_dir.exists():
            for spec_path in sorted(active_specs_dir.glob("*.json")):
                spec = load_selfplay_agent_spec(spec_path)
                if spec.agent_kind == "lapv1":
                    continue
                opponents[spec.name] = _build_registry_opponent_record(
                    name=spec.name,
                    spec=spec,
                    source_spec_path=spec_path,
                    category="historical",
                )
        for entry in seed_spec.reference_agents:
            candidate_path = None
            if active_specs_dir is not None:
                active_path = active_specs_dir / f"{entry.name}.json"
                if active_path.exists():
                    candidate_path = active_path
            if candidate_path is None:
                configured_path = resolve_repo_path(self._repo_root, Path(entry.spec_path))
                if configured_path.exists():
                    candidate_path = configured_path
            if candidate_path is None:
                continue
            spec = load_selfplay_agent_spec(candidate_path)
            if spec.agent_kind == "lapv1":
                continue
            opponents.setdefault(
                spec.name,
                _build_registry_opponent_record(
                    name=spec.name,
                    spec=spec,
                    source_spec_path=candidate_path,
                    category="historical",
                ),
            )
        for benchmark_name, raw_path in sorted(seed_spec.benchmark_agent_specs.items()):
            spec_path = resolve_repo_path(self._repo_root, Path(raw_path))
            spec = load_selfplay_agent_spec(spec_path)
            if spec.agent_kind == "lapv1":
                continue
            opponents[benchmark_name] = _build_registry_opponent_record(
                name=benchmark_name,
                spec=spec,
                source_spec_path=spec_path,
                category=("vice" if _is_vice_spec(name=benchmark_name, spec=spec) else "benchmark"),
            )
        return self._normalize_lineage_arena_registry(
            lineage,
            {
                "spec_version": ARENA_PROGRESS_POLICY_VERSION,
                "master_name": self._spec.name,
                "lineage_name": lineage.name,
                "seed_phase10_config_path": str(seed_path),
                "opponents": opponents,
                "stockfish": {
                    "active": True,
                    "current_skill_level": lineage.arena_progression.stockfish_initial_skill_level,
                    "cleared_levels": [],
                    "metadata": {
                        "engine_path": lineage.arena_progression.stockfish_engine_path,
                    },
                },
            },
        )

    def _normalize_lineage_arena_registry(
        self,
        lineage: Phase10LineageSpec,
        raw_registry: Mapping[str, Any],
    ) -> dict[str, Any]:
        stockfish_raw = dict(raw_registry.get("stockfish") or {})
        current_skill_level = int(
            stockfish_raw.get(
                "current_skill_level",
                lineage.arena_progression.stockfish_initial_skill_level,
            )
        )
        current_skill_level = max(
            lineage.arena_progression.stockfish_initial_skill_level,
            min(lineage.arena_progression.stockfish_max_skill_level, current_skill_level),
        )
        opponents: dict[str, dict[str, Any]] = {}
        for name, raw_record in sorted(dict(raw_registry.get("opponents") or {}).items()):
            record = dict(raw_record or {})
            normalized_name = str(record.get("name") or name)
            opponents[normalized_name] = {
                "name": normalized_name,
                "source_spec_path": str(record["source_spec_path"]),
                "agent_kind": str(record.get("agent_kind") or "planner"),
                "category": str(record.get("category") or "historical"),
                "active": bool(record.get("active", True)),
                "cleared_generation": _optional_int(record.get("cleared_generation")),
                "last_result": dict(record.get("last_result") or {}),
                "metadata": dict(record.get("metadata") or {}),
            }
        return {
            "spec_version": ARENA_PROGRESS_POLICY_VERSION,
            "master_name": self._spec.name,
            "lineage_name": lineage.name,
            "seed_phase10_config_path": str(lineage.seed_phase10_config_path),
            "safe_score_rate": lineage.arena_progression.safe_score_rate,
            "min_games": lineage.arena_progression.min_games,
            "opponents": opponents,
            "stockfish": {
                "active": bool(stockfish_raw.get("active", True)),
                "current_skill_level": current_skill_level,
                "cleared_levels": [
                    dict(entry) for entry in list(stockfish_raw.get("cleared_levels") or [])
                ],
                "metadata": {
                    **dict(stockfish_raw.get("metadata") or {}),
                    "engine_path": lineage.arena_progression.stockfish_engine_path,
                },
            },
            "last_applied_generation": _optional_int(raw_registry.get("last_applied_generation")),
            "last_applied_campaign_id": _optional_int(raw_registry.get("last_applied_campaign_id")),
        }

    def _materialize_generation_arena_opponents(
        self,
        *,
        lineage: Phase10LineageSpec,
        generation: int,
        arena_registry: Mapping[str, Any],
        output_root: Path,
    ) -> dict[str, str]:
        output_root.mkdir(parents=True, exist_ok=True)
        registry = self._normalize_lineage_arena_registry(lineage, dict(arena_registry))
        resolved: dict[str, str] = {}
        for opponent_name, record in sorted(dict(registry.get("opponents") or {}).items()):
            if not bool(record.get("active", True)):
                continue
            source_path = resolve_repo_path(self._repo_root, Path(str(record["source_spec_path"])))
            spec = load_selfplay_agent_spec(source_path)
            if spec.name != opponent_name:
                spec = SelfplayAgentSpec(
                    **{
                        **spec.to_dict(),
                        "name": opponent_name,
                        "metadata": {
                            **dict(spec.metadata),
                            "lineage_name": lineage.name,
                            "arena_progression_category": str(record.get("category") or ""),
                        },
                    }
                )
            resolved_path = output_root / f"{_safe_filename(opponent_name)}.json"
            write_selfplay_agent_spec(resolved_path, spec)
            resolved[opponent_name] = str(resolved_path)

        stockfish_record = dict(registry.get("stockfish") or {})
        if bool(stockfish_record.get("active", True)):
            stockfish_level = int(stockfish_record.get("current_skill_level", 0))
            stockfish_name = _stockfish_agent_name(stockfish_level)
            stockfish_spec = SelfplayAgentSpec(
                name=stockfish_name,
                agent_kind="uci_engine",
                opponent_mode="none",
                root_top_k=1,
                external_engine_path=lineage.arena_progression.stockfish_engine_path,
                external_engine_nodes=lineage.arena_progression.stockfish_nodes,
                external_engine_depth=lineage.arena_progression.stockfish_depth,
                external_engine_movetime_ms=lineage.arena_progression.stockfish_movetime_ms,
                external_engine_threads=lineage.arena_progression.stockfish_threads,
                external_engine_hash_mb=lineage.arena_progression.stockfish_hash_mb,
                external_engine_options={
                    "Skill Level": str(stockfish_level),
                    **dict(lineage.arena_progression.stockfish_engine_options),
                },
                tags=["external", "stockfish18", "arena_progression"],
                metadata={
                    "engine_family": "stockfish18",
                    "lineage_name": lineage.name,
                    "generation": generation,
                    "skill_level": stockfish_level,
                },
            )
            stockfish_path = output_root / f"{stockfish_name}.json"
            write_selfplay_agent_spec(stockfish_path, stockfish_spec)
            resolved[stockfish_name] = str(stockfish_path)
        return resolved

    def _evaluate_completed_generation(
        self,
        *,
        lineage: Phase10LineageSpec,
        campaign: CampaignRow,
        model: ModelRow | None,
        arena_registry: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        campaign_spec = load_phase10_lapv1_arena_campaign_spec(Path(campaign.config_path))
        summary_path = resolve_repo_path(
            self._repo_root,
            Path(campaign_spec.output_root) / "summary.json",
        )
        summary = _load_json_object(summary_path)
        verify_metrics = dict(summary.get("lapv1_verify_metrics") or {})
        arena_summary_path = Path(str(summary["arena_summary_path"]))
        arena_summary = _load_json_object(arena_summary_path)
        tracked_records = _tracked_arena_records(arena_summary)
        tracked_agent_names = [str(record["agent"]) for record in tracked_records]
        registry = self._normalize_lineage_arena_registry(
            lineage,
            dict(
                arena_registry
                if arena_registry is not None
                else self._load_or_create_lineage_arena_registry(lineage)
            ),
        )
        generation = int(campaign.metadata.get("generation", 0) or 0)
        arena_matrix = build_selfplay_arena_matrix(arena_summary)
        opponent_results = _collect_registry_opponent_results(
            registry=registry,
            arena_matrix=arena_matrix,
            tracked_agent_names=tracked_agent_names,
            progression=lineage.arena_progression,
        )
        already_applied = (
            int(registry.get("last_applied_generation", 0) or 0) == generation
            and int(registry.get("last_applied_campaign_id", 0) or 0) == campaign.id
        )
        if not already_applied:
            _apply_registry_progression_results(
                registry=registry,
                opponent_results=opponent_results,
                generation=generation,
                campaign_id=campaign.id,
                progression=lineage.arena_progression,
            )
            registry_path = self._lineage_arena_registry_path(lineage)
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry_path.write_text(
                json.dumps(registry, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if isinstance(arena_registry, dict):
            arena_registry.clear()
            arena_registry.update(registry)

        verify_top1 = float(verify_metrics.get("root_top1_accuracy", 0.0))
        verify_top3 = float(verify_metrics.get("root_top3_accuracy", 0.0))
        best_arena_score = max(
            (float(record["score_rate"]) for record in tracked_records),
            default=0.0,
        )
        accepted = True
        threshold_checks: dict[str, dict[str, Any]] = {}
        accepted &= _record_threshold(
            threshold_checks,
            key="verify_top1",
            actual=verify_top1,
            required=lineage.promotion_thresholds.min_verify_top1_accuracy,
        )
        accepted &= _record_threshold(
            threshold_checks,
            key="verify_top3",
            actual=verify_top3,
            required=lineage.promotion_thresholds.min_verify_top3_accuracy,
        )
        accepted &= _record_threshold(
            threshold_checks,
            key="arena_score_rate",
            actual=best_arena_score,
            required=lineage.promotion_thresholds.min_arena_score_rate,
        )

        evaluation = {
            "accepted": accepted,
            "model_status": "accepted" if accepted else "rejected",
            "promotion_score": best_arena_score if tracked_records else verify_top1,
            "summary_path": str(summary_path),
            "verify_top1_accuracy": verify_top1,
            "verify_top3_accuracy": verify_top3,
            "tracked_arena_records": tracked_records,
            "best_tracked_arena_score_rate": best_arena_score,
            "threshold_checks": threshold_checks,
            "opponent_results": opponent_results,
            "active_arena_opponents": list(_active_registry_opponent_names(registry)),
            "arena_registry_path": str(self._lineage_arena_registry_path(lineage)),
            "arena_progression_applied": not already_applied,
            "stockfish_skill_level": int(
                dict(registry.get("stockfish") or {}).get("current_skill_level", 0)
            ),
        }
        decision_path = resolve_repo_path(
            self._repo_root,
            Path(lineage.output_root)
            / f"generation_{int(campaign.metadata.get('generation', 0) or 0):04d}"
            / "decision.json",
        )
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if model is not None:
            self._db.update_campaign_record(campaign.id, active_model_id=model.id)
        return evaluation

    def _write_label_job_config(self, job: LabelPgnCorpusJobSpec) -> Path:
        config_path = self._output_root / "label_jobs" / job.name / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return config_path

    def _write_idle_phase10_job_config(self, job: IdlePhase10ArtifactJobSpec) -> Path:
        config_path = self._output_root / "idle_phase10_jobs" / job.name / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return config_path

    def _backfill_lineage_generation_usage(
        self,
        *,
        lineage: Phase10LineageSpec,
        campaigns: Sequence[CampaignRow],
        models: Sequence[ModelRow],
    ) -> dict[str, Any]:
        recorded_generations: list[int] = []
        skipped_generations: list[int] = []
        latest_recorded_generation = 0
        for campaign in campaigns:
            generation = int(campaign.metadata.get("generation", 0) or 0)
            if generation <= 0 or campaign.status != "succeeded":
                continue
            existing = self.usage_ledger.generation_usage(
                master_name=self._spec.name,
                lineage_name=lineage.name,
                generation=generation,
            )
            if existing is not None:
                recorded_generations.append(int(existing["generation"]))
                latest_recorded_generation = max(
                    latest_recorded_generation,
                    int(existing["generation"]),
                )
                continue
            merged_raw_dir = self._campaign_merged_raw_dir(campaign)
            if merged_raw_dir is None:
                skipped_generations.append(generation)
                continue
            train_records = load_raw_records(
                merged_raw_dir / "train_raw.jsonl",
                "jsonl",
                source_name=f"{lineage.name}:generation:{generation}:train",
            )
            verify_records = load_raw_records(
                merged_raw_dir / "verify_raw.jsonl",
                "jsonl",
                source_name=f"{lineage.name}:generation:{generation}:verify",
            )
            model = _latest_model_for_campaign(models=models, campaign_id=campaign.id)
            self.usage_ledger.record_generation_usage(
                master_name=self._spec.name,
                lineage_name=lineage.name,
                generation=generation,
                campaign_id=campaign.id,
                model_id=(model.id if model is not None else None),
                merged_raw_dir=str(merged_raw_dir),
                train_records=train_records,
                verify_records=verify_records,
            )
            recorded_generations.append(generation)
            latest_recorded_generation = max(latest_recorded_generation, generation)
        return {
            "recorded_generations": recorded_generations,
            "skipped_generations": skipped_generations,
            "latest_recorded_generation": latest_recorded_generation,
        }

    def _campaign_merged_raw_dir(self, campaign: CampaignRow) -> Path | None:
        config_path = resolve_repo_path(self._repo_root, Path(campaign.config_path))
        if not config_path.exists():
            return None
        config_payload = _load_json_object(config_path)
        merged_raw_dir = config_payload.get("merged_raw_dir")
        if merged_raw_dir is None:
            return None
        resolved = resolve_repo_path(self._repo_root, Path(str(merged_raw_dir)))
        if not _raw_corpus_exists(resolved):
            return None
        return resolved

    def _matching_campaigns(self, *, master_job_type: str, job_name: str) -> list[CampaignRow]:
        campaigns = self._db.list_campaign_records(limit=5000)
        matching = [
            campaign
            for campaign in campaigns
            if campaign.metadata.get("master_name") == self._spec.name
            and campaign.metadata.get("master_job_type") == master_job_type
            and campaign.metadata.get("job_name") == job_name
        ]
        matching.sort(
            key=lambda campaign: (
                int(campaign.metadata.get("generation", 0) or 0),
                campaign.id,
            )
        )
        return matching

    def _matching_models(self, *, lineage_name: str) -> list[ModelRow]:
        models = self._db.list_model_records(limit=5000)
        matching = [
            model
            for model in models
            if model.metadata.get("master_name") == self._spec.name
            and model.metadata.get("lineage_name") == lineage_name
        ]
        matching.sort(key=lambda model: (model.generation, model.id))
        return matching


def _build_registry_opponent_record(
    *,
    name: str,
    spec: SelfplayAgentSpec,
    source_spec_path: Path,
    category: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "source_spec_path": str(source_spec_path),
        "agent_kind": spec.agent_kind,
        "category": category,
        "active": True,
        "cleared_generation": None,
        "last_result": {},
        "metadata": dict(spec.metadata),
    }


def _is_vice_spec(*, name: str, spec: SelfplayAgentSpec) -> bool:
    return "vice" in name.lower() or str(spec.metadata.get("engine_family", "")).lower() == "vice"


def _stockfish_agent_name(skill_level: int) -> str:
    return f"stockfish18_skill_{skill_level:02d}"


def _safe_filename(value: str) -> str:
    cleaned = [
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in value
    ]
    return "".join(cleaned).strip("_") or "opponent"


def _active_registry_opponent_names(registry: Mapping[str, Any]) -> list[str]:
    names = [
        str(name)
        for name, record in sorted(dict(registry.get("opponents") or {}).items())
        if bool(dict(record).get("active", True))
    ]
    stockfish = dict(registry.get("stockfish") or {})
    if bool(stockfish.get("active", True)):
        names.append(_stockfish_agent_name(int(stockfish.get("current_skill_level", 0) or 0)))
    return names


def _collect_registry_opponent_results(
    *,
    registry: Mapping[str, Any],
    arena_matrix: Mapping[str, Any],
    tracked_agent_names: Sequence[str],
    progression: ArenaProgressionSpec,
) -> dict[str, dict[str, Any]]:
    matrix = dict(arena_matrix.get("matrix") or {})
    results: dict[str, dict[str, Any]] = {}
    for opponent_name, record in sorted(dict(registry.get("opponents") or {}).items()):
        if not bool(dict(record).get("active", True)):
            continue
        results[opponent_name] = _best_tracked_result_vs_opponent(
            tracked_agent_names=tracked_agent_names,
            matrix=matrix,
            opponent_name=opponent_name,
            safe_score_rate=progression.safe_score_rate,
            min_games=progression.min_games,
            category=str(dict(record).get("category") or "historical"),
        )
    stockfish = dict(registry.get("stockfish") or {})
    if bool(stockfish.get("active", True)):
        stockfish_name = _stockfish_agent_name(int(stockfish.get("current_skill_level", 0) or 0))
        results[stockfish_name] = _best_tracked_result_vs_opponent(
            tracked_agent_names=tracked_agent_names,
            matrix=matrix,
            opponent_name=stockfish_name,
            safe_score_rate=progression.safe_score_rate,
            min_games=progression.min_games,
            category="stockfish",
        )
    return results


def _best_tracked_result_vs_opponent(
    *,
    tracked_agent_names: Sequence[str],
    matrix: Mapping[str, Any],
    opponent_name: str,
    safe_score_rate: float,
    min_games: int,
    category: str,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for tracked_agent_name in tracked_agent_names:
        row = dict(matrix.get(tracked_agent_name) or {})
        cell = dict(row.get(opponent_name) or {})
        candidate = {
            "opponent": opponent_name,
            "category": category,
            "best_tracked_agent": tracked_agent_name,
            "game_count": int(cell.get("game_count", 0)),
            "score": round(float(cell.get("score", 0.0)), 6),
            "score_rate": round(float(cell.get("score_rate", 0.0)), 6),
            "wins": int(cell.get("wins", 0)),
            "losses": int(cell.get("losses", 0)),
            "draws": int(cell.get("draws", 0)),
            "unfinished": int(cell.get("unfinished", 0)),
        }
        if best is None or (
            float(candidate["score_rate"]),
            float(candidate["score"]),
            str(candidate["best_tracked_agent"]),
        ) > (
            float(best["score_rate"]),
            float(best["score"]),
            str(best["best_tracked_agent"]),
        ):
            best = candidate
    if best is None:
        best = {
            "opponent": opponent_name,
            "category": category,
            "best_tracked_agent": None,
            "game_count": 0,
            "score": 0.0,
            "score_rate": 0.0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "unfinished": 0,
        }
    best["safe_score_rate"] = safe_score_rate
    best["min_games"] = min_games
    best["safely_beaten"] = bool(
        int(best["game_count"]) >= min_games and float(best["score_rate"]) >= safe_score_rate
    )
    return best


def _apply_registry_progression_results(
    *,
    registry: dict[str, Any],
    opponent_results: Mapping[str, Mapping[str, Any]],
    generation: int,
    campaign_id: int,
    progression: ArenaProgressionSpec,
) -> None:
    for opponent_name, record in dict(registry.get("opponents") or {}).items():
        result = dict(opponent_results.get(opponent_name) or {})
        record["last_result"] = result
        if bool(record.get("active", True)) and bool(result.get("safely_beaten")):
            record["active"] = False
            record["cleared_generation"] = generation

    stockfish = dict(registry.get("stockfish") or {})
    current_skill_level = int(stockfish.get("current_skill_level", 0) or 0)
    stockfish_name = _stockfish_agent_name(current_skill_level)
    stockfish_result = dict(opponent_results.get(stockfish_name) or {})
    stockfish["last_result"] = stockfish_result
    if bool(stockfish_result.get("safely_beaten")):
        next_skill_level = min(
            progression.stockfish_max_skill_level,
            current_skill_level + progression.stockfish_skill_step,
        )
        if next_skill_level != current_skill_level:
            cleared_levels = list(stockfish.get("cleared_levels") or [])
            cleared_levels.append(
                {
                    "generation": generation,
                    "campaign_id": campaign_id,
                    "skill_level": current_skill_level,
                    "score_rate": float(stockfish_result.get("score_rate", 0.0)),
                    "game_count": int(stockfish_result.get("game_count", 0)),
                }
            )
            stockfish["cleared_levels"] = cleared_levels
            stockfish["current_skill_level"] = next_skill_level
    registry["stockfish"] = stockfish
    registry["last_applied_generation"] = generation
    registry["last_applied_campaign_id"] = campaign_id


def _export_session_tree_to_pgn_dir(
    *,
    source_root: Path,
    output_root: Path,
    event_name: str,
    phase_name: str,
    generation: int,
) -> int:
    if not source_root.exists():
        return 0
    session_paths = sorted(source_root.glob("*.json"))
    if not session_paths:
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    written = 0
    for session_path in session_paths:
        target_path = output_root / f"{session_path.stem}.pgn"
        _write_session_record_as_pgn(
            session_path=session_path,
            output_path=target_path,
            event_name=event_name,
            phase_name=phase_name,
            generation=generation,
        )
        written += 1
    return written


def _write_session_record_as_pgn(
    *,
    session_path: Path,
    output_path: Path,
    event_name: str,
    phase_name: str,
    generation: int,
) -> None:
    import chess
    import chess.pgn

    from train.eval.selfplay import SelfplaySessionRecord

    payload = json.loads(session_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{session_path}: selfplay session must be a JSON object")
    session = SelfplaySessionRecord.from_dict(payload)
    games: list[str] = []
    for game_index, game_record in enumerate(session.games, start=1):
        board = chess.Board(game_record.initial_fen)
        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = event_name
        pgn_game.headers["Site"] = phase_name
        pgn_game.headers["Round"] = str(generation)
        pgn_game.headers["White"] = game_record.white_agent
        pgn_game.headers["Black"] = game_record.black_agent
        pgn_game.headers["Result"] = game_record.result
        pgn_game.headers["GameId"] = game_record.game_id
        pgn_game.headers["Termination"] = game_record.termination_reason
        pgn_game.headers["SessionPath"] = str(session_path)
        if game_record.initial_fen != "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
            pgn_game.headers["SetUp"] = "1"
            pgn_game.headers["FEN"] = game_record.initial_fen
        node = pgn_game
        for move_record in game_record.moves:
            move = chess.Move.from_uci(move_record.move_uci)
            if move not in board.legal_moves:
                raise ValueError(
                    f"{session_path}: illegal move {move_record.move_uci} in {game_record.game_id}"
                )
            node = node.add_variation(move)
            board.push(move)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        rendered = pgn_game.accept(exporter).strip()
        if rendered:
            games.append(rendered)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(games) + ("\n" if games else ""), encoding="utf-8")


def latest_generation(campaigns: Sequence[CampaignRow]) -> int:
    """Return the highest master generation present in the campaign list."""
    return max(int(campaign.metadata.get("generation", 0) or 0) for campaign in campaigns)


def _latest_model_for_campaign(*, models: Sequence[ModelRow], campaign_id: int) -> ModelRow | None:
    candidates = [model for model in models if model.campaign_id == campaign_id]
    if not candidates:
        return None
    candidates.sort(key=lambda model: (model.generation, model.id))
    return candidates[-1]


def _record_threshold(
    checks: dict[str, dict[str, Any]],
    *,
    key: str,
    actual: float,
    required: float | None,
) -> bool:
    passed = required is None or actual >= required
    checks[key] = {
        "actual": round(actual, 6),
        "required": required,
        "passed": passed,
    }
    return passed


def _tracked_arena_records(arena_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    tracked_agent_names = [
        str(name)
        for name in list(dict(arena_summary.get("metadata") or {}).get("lapv1_agent_names") or [])
    ]
    standings = dict(arena_summary.get("standings") or {})
    records: list[dict[str, Any]] = []
    for agent_name in tracked_agent_names:
        row = dict(standings.get(agent_name) or {})
        games = int(row.get("games", 0))
        score = float(row.get("score", 0.0))
        records.append(
            {
                "agent": agent_name,
                "games": games,
                "score": score,
                "score_rate": round(score / games, 6) if games > 0 else 0.0,
            }
        )
    return records


def _available_raw_snapshot(
    work_dir: Path,
    *,
    require_completed: bool,
) -> dict[str, Path] | None:
    progress_path = work_dir / "progress.json"
    train_raw_path = work_dir / "train_raw.jsonl"
    verify_raw_path = work_dir / "verify_raw.jsonl"
    summary_path = work_dir / "summary.json"
    if not progress_path.exists() or not train_raw_path.exists() or not verify_raw_path.exists():
        return None
    progress = _load_json_object(progress_path)
    if require_completed and not bool(progress.get("completed")):
        return None
    if not summary_path.exists():
        summary_path.write_text(
            json.dumps(
                {
                    "progress_path": str(progress_path),
                    "train_raw_path": str(train_raw_path),
                    "verify_raw_path": str(verify_raw_path),
                    "progress": progress,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "progress_path": progress_path,
        "train_raw_path": train_raw_path,
        "verify_raw_path": verify_raw_path,
        "summary_path": summary_path,
    }


def _completed_label_snapshot(work_dir: Path) -> dict[str, Path] | None:
    snapshot = _available_raw_snapshot(work_dir, require_completed=True)
    if snapshot is None:
        return None
    return {
        "progress_path": snapshot["progress_path"],
        "summary_path": snapshot["summary_path"],
    }


def _raw_corpus_counts(raw_dir: Path) -> dict[str, int]:
    snapshot = _available_raw_snapshot(raw_dir, require_completed=False)
    if snapshot is not None:
        progress = _load_json_object(snapshot["progress_path"])
        counts = dict(progress.get("counts") or {})
        return {
            "train": int(counts.get("train", 0)),
            "verify": int(counts.get("verify", 0)),
        }
    selection_summary_path = raw_dir / "selection_summary.json"
    if selection_summary_path.exists():
        summary = _load_json_object(selection_summary_path)
        return {
            "train": int(summary.get("train_records", 0)),
            "verify": int(summary.get("verify_records", 0)),
        }
    return {
        "train": _count_nonempty_lines(raw_dir / "train_raw.jsonl"),
        "verify": _count_nonempty_lines(raw_dir / "verify_raw.jsonl"),
    }


def _raw_corpus_exists(raw_dir: Path) -> bool:
    return (raw_dir / "train_raw.jsonl").exists() and (raw_dir / "verify_raw.jsonl").exists()


def _count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                count += 1
    return count


def _materialize_usage_balanced_raw_corpus(
    *,
    candidate_dir: Path,
    output_dir: Path,
    master_name: str,
    lineage_name: str,
    generation: int,
    desired_train_records: int,
    desired_verify_records: int,
    usage_ledger: LineageTrainingUsageLedger,
    source_dirs: Sequence[Path],
) -> dict[str, Any]:
    train_records = load_raw_records(
        candidate_dir / "train_raw.jsonl",
        "jsonl",
        source_name=f"{lineage_name}:generation:{generation}:candidate_train",
    )
    verify_records = load_raw_records(
        candidate_dir / "verify_raw.jsonl",
        "jsonl",
        source_name=f"{lineage_name}:generation:{generation}:candidate_verify",
    )
    selected_train_records, train_summary = _select_usage_balanced_records(
        records=train_records,
        split_name="train",
        desired_count=desired_train_records,
        generation=generation,
        master_name=master_name,
        lineage_name=lineage_name,
        usage_ledger=usage_ledger,
    )
    selected_verify_records, verify_summary = _select_usage_balanced_records(
        records=verify_records,
        split_name="verify",
        desired_count=desired_verify_records,
        generation=generation,
        master_name=master_name,
        lineage_name=lineage_name,
        usage_ledger=usage_ledger,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_raw_records(output_dir / "train_raw.jsonl", selected_train_records)
    _write_raw_records(output_dir / "verify_raw.jsonl", selected_verify_records)
    summary = {
        "selection_policy": "usage_balanced_generation_delta",
        "master_name": master_name,
        "lineage_name": lineage_name,
        "generation": generation,
        "candidate_dir": str(candidate_dir),
        "source_dirs": [str(path) for path in source_dirs],
        "desired_train_records": desired_train_records,
        "desired_verify_records": desired_verify_records,
        "train_records": len(selected_train_records),
        "verify_records": len(selected_verify_records),
        "train_summary": train_summary,
        "verify_summary": verify_summary,
        "train_raw_path": str(output_dir / "train_raw.jsonl"),
        "verify_raw_path": str(output_dir / "verify_raw.jsonl"),
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _select_usage_balanced_records(
    *,
    records: Sequence[RawPositionRecord],
    split_name: str,
    desired_count: int,
    generation: int,
    master_name: str,
    lineage_name: str,
    usage_ledger: LineageTrainingUsageLedger,
) -> tuple[list[RawPositionRecord], dict[str, Any]]:
    if not records:
        return [], {
            "candidate_records": 0,
            "selected_records": 0,
            "fresh_records": 0,
            "reused_records": 0,
            "max_usage_before_selection": 0,
            "usage_histogram_before_selection": {},
        }
    desired = desired_count if desired_count > 0 else len(records)
    by_hash = {hashlib.sha256(record.fen.encode("utf-8")).hexdigest(): record for record in records}
    usage_state = usage_ledger.usage_state(
        master_name=master_name,
        lineage_name=lineage_name,
        split_name=split_name,
        fen_hashes=list(by_hash),
    )
    scored: list[tuple[int, int, int, str]] = []
    usage_histogram: dict[str, int] = {}
    for fen_hash, record in by_hash.items():
        state = usage_state.get(fen_hash, LineageSampleUsageState())
        usage_histogram[str(int(state.usage_count))] = (
            usage_histogram.get(str(int(state.usage_count)), 0) + 1
        )
        scored.append(
            (
                int(state.usage_count),
                int(state.last_generation),
                _stable_generation_tiebreak(
                    generation=generation,
                    split_name=split_name,
                    sample_id=record.sample_id,
                    fen_hash=fen_hash,
                ),
                fen_hash,
            )
        )
    scored.sort()
    selected_hashes = [fen_hash for _usage_count, _last_generation, _tie, fen_hash in scored[:desired]]
    selected_records = [by_hash[fen_hash] for fen_hash in selected_hashes]
    fresh_records = sum(
        1
        for fen_hash in selected_hashes
        if int(usage_state.get(fen_hash, LineageSampleUsageState()).usage_count) == 0
    )
    return selected_records, {
        "candidate_records": len(records),
        "selected_records": len(selected_records),
        "fresh_records": fresh_records,
        "reused_records": len(selected_records) - fresh_records,
        "max_usage_before_selection": max(
            [int(state.usage_count) for state in usage_state.values()],
            default=0,
        ),
        "usage_histogram_before_selection": usage_histogram,
    }


def _stable_generation_tiebreak(
    *,
    generation: int,
    split_name: str,
    sample_id: str,
    fen_hash: str,
) -> int:
    digest = hashlib.sha256(
        f"{generation}:{split_name}:{sample_id}:{fen_hash}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _write_raw_records(path: Path, records: Sequence[RawPositionRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
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


def _replace_path_prefixes(payload: Any, *, replacements: Mapping[str, str]) -> Any:
    if isinstance(payload, dict):
        return {
            str(key): _replace_path_prefixes(value, replacements=replacements)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_replace_path_prefixes(value, replacements=replacements) for value in payload]
    if isinstance(payload, str):
        updated = payload
        for source, target in replacements.items():
            normalized_source = str(source)
            normalized_target = str(target)
            if updated == normalized_source:
                updated = normalized_target
                continue
            source_prefix = normalized_source.rstrip("/") + "/"
            if updated.startswith(source_prefix):
                updated = normalized_target.rstrip("/") + updated[len(normalized_source) :]
        return updated
    return payload


def _infer_train_workflow_roots(payload: Mapping[str, Any]) -> tuple[Path, ...]:
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in (
        payload.get("data", {}).get("train_path"),
        payload.get("data", {}).get("validation_path"),
    ):
        if not isinstance(candidate, str):
            continue
        path = Path(candidate)
        if len(path.parts) < 3:
            continue
        root = path.parents[1]
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return tuple(roots)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _assert_lapv1_train_payload(payload: Mapping[str, Any]) -> None:
    required_top_level = ("output_dir", "data", "model", "optimization", "evaluation", "runtime", "export")
    for key in required_top_level:
        if key not in payload:
            raise ValueError(f"lapv1 train payload missing required key: {key}")
    data_payload = payload.get("data")
    export_payload = payload.get("export")
    if not isinstance(data_payload, Mapping):
        raise ValueError("lapv1 train payload data must be a JSON object")
    if not isinstance(export_payload, Mapping):
        raise ValueError("lapv1 train payload export must be a JSON object")
    for key in ("train_path", "validation_path"):
        if key not in data_payload:
            raise ValueError(f"lapv1 train payload data missing required key: {key}")
    for key in ("bundle_dir", "checkpoint_name"):
        if key not in export_payload:
            raise ValueError(f"lapv1 train payload export missing required key: {key}")
