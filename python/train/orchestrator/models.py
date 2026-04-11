"""Shared data models for the MySQL-backed training orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from typing import Any, Mapping


ORCHESTRATOR_SCHEMA_VERSION = 1
DEFAULT_MYSQL_PORT = 3306

TASK_STATE_QUEUED = "queued"
TASK_STATE_LEASED = "leased"
TASK_STATE_SUCCEEDED = "succeeded"
TASK_STATE_FAILED = "failed"


@dataclass(frozen=True)
class MySQLConfig:
    """Connection parameters for the MySQL control plane."""

    host: str
    database: str
    user: str
    password: str
    port: int = DEFAULT_MYSQL_PORT
    connect_timeout_seconds: int = 10
    read_timeout_seconds: int = 60
    write_timeout_seconds: int = 60

    @classmethod
    def from_env(cls, prefix: str = "EK_MYSQL_") -> "MySQLConfig":
        """Build a config from environment variables."""
        host = os.environ.get(f"{prefix}HOST")
        database = os.environ.get(f"{prefix}DATABASE")
        user = os.environ.get(f"{prefix}USER")
        password = os.environ.get(f"{prefix}PASSWORD")
        missing = [
            name
            for name, value in (
                ("HOST", host),
                ("DATABASE", database),
                ("USER", user),
                ("PASSWORD", password),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "missing MySQL environment variables: "
                + ", ".join(f"{prefix}{name}" for name in missing)
            )
        return cls(
            host=str(host),
            database=str(database),
            user=str(user),
            password=str(password),
            port=int(os.environ.get(f"{prefix}PORT", DEFAULT_MYSQL_PORT)),
            connect_timeout_seconds=int(os.environ.get(f"{prefix}CONNECT_TIMEOUT", 10)),
            read_timeout_seconds=int(os.environ.get(f"{prefix}READ_TIMEOUT", 60)),
            write_timeout_seconds=int(os.environ.get(f"{prefix}WRITE_TIMEOUT", 60)),
        )

    def safe_dict(self) -> dict[str, object]:
        """Return a log-safe summary without the password."""
        return {
            "host": self.host,
            "database": self.database,
            "user": self.user,
            "port": self.port,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "write_timeout_seconds": self.write_timeout_seconds,
        }


@dataclass(frozen=True)
class ArtifactRef:
    """Small artifact record stored in task results and the artifact registry."""

    kind: str
    path: str
    state: str = "ready"
    summary_path: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "path": self.path,
            "state": self.state,
            "summary_path": self.summary_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TaskResult:
    """Serializable success payload for one completed orchestrator task."""

    summary_path: str
    artifacts: tuple[ArtifactRef, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_task_keys: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_path": self.summary_path,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "created_task_keys": list(self.created_task_keys),
        }


@dataclass(frozen=True)
class PlannedTask:
    """One task planned by the controller before it is inserted into MySQL."""

    key: str
    task_type: str
    capability: str
    payload: dict[str, Any]
    priority: int = 0
    max_attempts: int = 1
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkerDescriptor:
    """Stable worker identity and capability metadata."""

    worker_id: str
    hostname: str
    capabilities: tuple[str, ...]
    scratch_root: str
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_db_payload(self) -> dict[str, Any]:
        return {
            "id": self.worker_id,
            "hostname": self.hostname,
            "capabilities_json": list(self.capabilities),
            "scratch_root": self.scratch_root,
            "status": "idle",
            "version": self.version,
            "metadata_json": dict(self.metadata),
        }


@dataclass(frozen=True)
class TaskRow:
    """Normalized task row returned from the MySQL layer."""

    id: int
    campaign_id: int
    model_id: int | None
    task_type: str
    capability: str
    priority: int
    state: str
    payload: dict[str, Any]
    result: dict[str, Any] | None
    worker_id: str | None
    lease_until: str | None
    attempt_count: int
    max_attempts: int
    depends_on_count: int
    not_before: str | None
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_db_row(cls, row: Mapping[str, Any]) -> "TaskRow":
        return cls(
            id=int(row["id"]),
            campaign_id=int(row["campaign_id"]),
            model_id=(int(row["model_id"]) if row.get("model_id") is not None else None),
            task_type=str(row["task_type"]),
            capability=str(row["capability"]),
            priority=int(row["priority"]),
            state=str(row["state"]),
            payload=dict(row["payload_json"]),
            result=(dict(row["result_json"]) if row.get("result_json") is not None else None),
            worker_id=(str(row["worker_id"]) if row.get("worker_id") is not None else None),
            lease_until=(str(row["lease_until"]) if row.get("lease_until") is not None else None),
            attempt_count=int(row["attempt_count"]),
            max_attempts=int(row["max_attempts"]),
            depends_on_count=int(row["depends_on_count"]),
            not_before=(str(row["not_before"]) if row.get("not_before") is not None else None),
            created_at=(str(row["created_at"]) if row.get("created_at") is not None else None),
            updated_at=(str(row["updated_at"]) if row.get("updated_at") is not None else None),
        )


@dataclass(frozen=True)
class CampaignRow:
    """Normalized campaign row returned from the MySQL layer."""

    id: int
    name: str
    kind: str
    status: str
    config_path: str
    active_model_id: int | None
    metadata: dict[str, Any]
    created_at: str | None
    updated_at: str | None

    @classmethod
    def from_db_row(cls, row: Mapping[str, Any]) -> "CampaignRow":
        return cls(
            id=int(row["id"]),
            name=str(row["name"]),
            kind=str(row["kind"]),
            status=str(row["status"]),
            config_path=str(row["config_path"]),
            active_model_id=(
                int(row["active_model_id"]) if row.get("active_model_id") is not None else None
            ),
            metadata=dict(row.get("metadata_json") or {}),
            created_at=(str(row["created_at"]) if row.get("created_at") is not None else None),
            updated_at=(str(row["updated_at"]) if row.get("updated_at") is not None else None),
        )


@dataclass(frozen=True)
class ModelRow:
    """Normalized model row returned from the MySQL layer."""

    id: int
    campaign_id: int
    parent_model_id: int | None
    generation: int
    train_config_path: str | None
    agent_spec_path: str | None
    checkpoint_path: str | None
    bundle_path: str | None
    verify_json_path: str | None
    arena_summary_path: str | None
    status: str
    promotion_score: float | None
    metadata: dict[str, Any]
    created_at: str | None

    @classmethod
    def from_db_row(cls, row: Mapping[str, Any]) -> "ModelRow":
        return cls(
            id=int(row["id"]),
            campaign_id=int(row["campaign_id"]),
            parent_model_id=(
                int(row["parent_model_id"]) if row.get("parent_model_id") is not None else None
            ),
            generation=int(row["generation"]),
            train_config_path=(
                str(row["train_config_path"])
                if row.get("train_config_path") is not None
                else None
            ),
            agent_spec_path=(
                str(row["agent_spec_path"]) if row.get("agent_spec_path") is not None else None
            ),
            checkpoint_path=(
                str(row["checkpoint_path"]) if row.get("checkpoint_path") is not None else None
            ),
            bundle_path=(str(row["bundle_path"]) if row.get("bundle_path") is not None else None),
            verify_json_path=(
                str(row["verify_json_path"]) if row.get("verify_json_path") is not None else None
            ),
            arena_summary_path=(
                str(row["arena_summary_path"])
                if row.get("arena_summary_path") is not None
                else None
            ),
            status=str(row["status"]),
            promotion_score=(
                float(row["promotion_score"]) if row.get("promotion_score") is not None else None
            ),
            metadata=dict(row.get("metadata_json") or {}),
            created_at=(str(row["created_at"]) if row.get("created_at") is not None else None),
        )


@dataclass(frozen=True)
class LabelPgnCorpusPayload:
    """Task payload for one resumable PGN -> Stockfish raw corpus labeling run."""

    config_path: str
    pgn_root: str
    pgn_glob: str
    engine_path: str
    work_dir: str
    target_train_records: int
    target_verify_records: int
    min_ply: int
    max_ply: int
    ply_stride: int
    engine_nodes: int
    hash_mb: int
    threads: int
    split_seed: str
    verify_divisor: int
    progress_every: int
    max_games: int
    export_jsonl_on_complete: bool = True
    complete_at_eof: bool = False
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "label_pgn_corpus"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LabelPgnCorpusIdleSlicePayload:
    """Task payload for one low-priority PGN labeling slice used during train idle time."""

    config_path: str
    pgn_root: str
    pgn_glob: str
    engine_path: str
    work_dir: str
    target_train_records: int
    target_verify_records: int
    min_ply: int
    max_ply: int
    ply_stride: int
    engine_nodes: int
    hash_mb: int
    threads: int
    split_seed: str
    verify_divisor: int
    progress_every: int
    max_games: int
    file_shard_index: int
    file_shard_count: int
    run_max_games: int
    export_jsonl_on_complete: bool = True
    complete_at_eof: bool = False
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "label_pgn_corpus_idle_slice"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase5RawMergePayload:
    """Task payload for merging multiple labeled raw shard corpora."""

    config_path: str
    output_dir: str
    source_dirs: tuple[str, ...]
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase5_raw_merge"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10MaterializePayload:
    """Task payload for the exact Phase-5 materialization stage."""

    config_path: str
    output_root: str
    raw_dir: str
    train_output_dir: str
    verify_output_dir: str
    source_name: str
    seed: str
    oracle_workers: int
    oracle_batch_size: int
    chunk_size: int
    log_every_chunks: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_materialize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10WorkflowPreparePayload:
    """Task payload for the workflow-expansion control step."""

    config_path: str
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_workflow_prepare"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10ArtifactWorkflowPreparePayload:
    """Task payload for expanding the idle artifact-only workflow DAG."""

    config_path: str
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_artifact_workflow_prepare"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10WorkflowChunkPayload:
    """Task payload for one single workflow chunk."""

    config_path: str
    split: str
    canonical_split: str
    chunk_index: int
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_workflow_chunk"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10WorkflowFinalizePayload:
    """Task payload for the deterministic workflow finalize pass."""

    config_path: str
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_workflow_finalize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10ArtifactFinalizePayload:
    """Task payload for writing the final idle artifact-build summary."""

    config_path: str
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_artifact_finalize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainLapv1Payload:
    """Task payload for one tracked LAP training job."""

    config_path: str
    model_id: int
    model_label: str
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "train_lapv1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10SelfplayPreparePayload:
    """Task payload for resolving the tracked LAP agent and expanding selfplay shards."""

    config_path: str
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_selfplay_prepare"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10SelfplayShardPayload:
    """Task payload for one distributed pre-verify LAP selfplay shard."""

    config_path: str
    agent_spec_path: str
    agent_name: str
    output_root: str
    shard_index: int
    starting_game_index: int
    games: int
    max_plies: int
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_selfplay_shard"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10SelfplayFinalizePayload:
    """Task payload for the pre-verify selfplay aggregate summary."""

    config_path: str
    agent_spec_path: str
    agent_name: str
    output_root: str
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_selfplay_finalize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerifyLapv1Payload:
    """Task payload for one held-out LAP verify job."""

    config_path: str
    model_id: int
    checkpoint_path: str
    dataset_path: str
    output_dir: str
    output_path: str
    top_k: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "verify_lapv1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10ArenaPreparePayload:
    """Task payload for resolving the arena spec and expanding matchup tasks."""

    config_path: str
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_arena_prepare"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArenaMatchPayload:
    """Task payload for one single ordered arena matchup."""

    config_path: str
    resolved_arena_spec_path: str
    output_root: str
    matchup_index: int
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "arena_match"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArenaFinalizePayload:
    """Task payload for the arena summary/matrix finalizer."""

    config_path: str
    resolved_arena_spec_path: str
    output_root: str
    matrix_path: str
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "arena_finalize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase10FinalizePayload:
    """Task payload for the final top-level Phase-10 summary writer."""

    config_path: str
    resolved_arena_spec_path: str
    model_id: int
    schema_version: int = ORCHESTRATOR_SCHEMA_VERSION
    task_kind: str = "phase10_finalize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
