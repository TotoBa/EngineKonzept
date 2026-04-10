"""High-level controller logic for the MySQL-backed training orchestrator."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from train.eval.phase10_campaign import (
    Phase10Lapv1ArenaCampaignSpec,
    build_resolved_arena_spec,
    load_phase10_lapv1_arena_campaign_spec,
    materialize_resolved_lapv1_agent_specs,
    resolve_repo_path,
    select_reference_agents,
)
from train.orchestrator.db import OrchestratorDB
from train.orchestrator.models import (
    ArenaFinalizePayload,
    ArenaMatchPayload,
    Phase10ArenaPreparePayload,
    Phase10FinalizePayload,
    Phase10MaterializePayload,
    Phase10WorkflowChunkPayload,
    Phase10WorkflowFinalizePayload,
    Phase10WorkflowPreparePayload,
    PlannedTask,
    TrainLapv1Payload,
    VerifyLapv1Payload,
)

if TYPE_CHECKING:
    from train.eval.arena import SelfplayArenaSpec


class OrchestratorController:
    """Repo-aware DAG controller for distributed Phase-10 campaigns."""

    def __init__(self, *, db: OrchestratorDB, repo_root: Path) -> None:
        self._db = db
        self._repo_root = repo_root

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def db(self) -> OrchestratorDB:
        return self._db

    def submit_phase10_campaign(
        self,
        *,
        config_path: Path,
        kind: str = "phase10_native",
    ) -> dict[str, Any]:
        """Insert the initial campaign plus bootstrap tasks into MySQL."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        spec = load_phase10_lapv1_arena_campaign_spec(resolved_config_path)
        train_metadata = _load_lap_train_metadata(
            resolve_repo_path(self._repo_root, Path(spec.lapv1_config_path))
        )
        campaign_id = self._db.insert_campaign(
            name=spec.name,
            kind=kind,
            status="queued",
            config_path=str(resolved_config_path),
            metadata={
                "model_label": spec.model_label,
                "output_root": spec.output_root,
            },
        )
        model_id = self._db.insert_model(
            campaign_id=campaign_id,
            status="queued",
            train_config_path=str(resolve_repo_path(self._repo_root, Path(spec.lapv1_config_path))),
            agent_spec_path=str(resolve_repo_path(self._repo_root, Path(spec.lapv1_agent_spec_path))),
            metadata={
                "model_label": spec.model_label,
                "expected_checkpoint_path": train_metadata["checkpoint_path"],
            },
        )
        planned_tasks = build_phase10_bootstrap_tasks(
            spec_path=resolved_config_path,
            spec=spec,
            model_id=model_id,
        )
        task_ids = self._db.insert_planned_tasks(
            campaign_id=campaign_id,
            model_id=model_id,
            planned_tasks=planned_tasks,
        )
        self._db.update_campaign_record(campaign_id, status="queued")
        return {
            "campaign_id": campaign_id,
            "model_id": model_id,
            "task_ids": task_ids,
            "config_path": str(resolved_config_path),
        }

    def expand_phase10_workflow(
        self,
        *,
        parent_task_id: int,
        campaign_id: int,
        model_id: int,
        config_path: Path,
    ) -> dict[str, Any]:
        """Expand the workflow/build/train/verify DAG after materialization."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        spec = load_phase10_lapv1_arena_campaign_spec(resolved_config_path)
        train_summary = _load_json(
            resolve_repo_path(self._repo_root, Path(spec.train_dataset_dir)) / "summary.json"
        )
        verify_summary = _load_json(
            resolve_repo_path(self._repo_root, Path(spec.verify_dataset_dir)) / "summary.json"
        )
        planned_tasks = build_phase10_workflow_tasks(
            spec_path=resolved_config_path,
            spec=spec,
            model_id=model_id,
            train_summary=train_summary,
            verify_summary=verify_summary,
            repo_root=self._repo_root,
        )
        task_ids = self._db.insert_planned_tasks(
            campaign_id=campaign_id,
            model_id=model_id,
            planned_tasks=planned_tasks,
            extra_dependency_task_ids=(parent_task_id,),
        )
        summary_path = _write_json_summary(
            resolve_repo_path(self._repo_root, Path(spec.output_root))
            / "orchestrator"
            / "workflow_prepare"
            / "summary.json",
            {
                "campaign_name": spec.name,
                "train_split_counts": dict(train_summary.get("split_counts") or {}),
                "verify_split_counts": dict(verify_summary.get("split_counts") or {}),
                "created_task_ids": task_ids,
            },
        )
        self._db.update_campaign_record(campaign_id, status="running")
        return {
            "summary_path": str(summary_path),
            "created_task_ids": task_ids,
        }

    def expand_phase10_arena(
        self,
        *,
        parent_task_id: int,
        campaign_id: int,
        model_id: int,
        config_path: Path,
    ) -> dict[str, Any]:
        """Resolve the arena spec and expand distributed arena match tasks."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        spec = load_phase10_lapv1_arena_campaign_spec(resolved_config_path)
        campaign_output_root = resolve_repo_path(self._repo_root, Path(spec.output_root))
        tracked_agent_spec_path = resolve_repo_path(
            self._repo_root, Path(spec.lapv1_agent_spec_path)
        )
        selected_reference_agents = select_reference_agents(spec, repo_root=self._repo_root)
        lap_agent_paths = materialize_resolved_lapv1_agent_specs(
            spec=spec,
            tracked_lapv1_agent_path=tracked_agent_spec_path,
            output_root=campaign_output_root,
        )
        resolved_arena_spec = build_resolved_arena_spec(
            spec,
            selected_reference_agents,
            repo_root=self._repo_root,
            lapv1_agent_paths=lap_agent_paths,
        )
        resolved_arena_spec_path = campaign_output_root / "arena_spec.resolved.json"
        from train.eval.arena import write_selfplay_arena_spec

        write_selfplay_arena_spec(resolved_arena_spec_path, resolved_arena_spec)
        self._db.update_model_record(
            model_id,
            agent_spec_path=str(resolved_arena_spec_path),
            status="evaluating",
        )
        planned_tasks = build_phase10_arena_tasks(
            spec_path=resolved_config_path,
            spec=spec,
            resolved_arena_spec=resolved_arena_spec,
            resolved_arena_spec_path=resolved_arena_spec_path,
            model_id=model_id,
            repo_root=self._repo_root,
        )
        task_ids = self._db.insert_planned_tasks(
            campaign_id=campaign_id,
            model_id=model_id,
            planned_tasks=planned_tasks,
            extra_dependency_task_ids=(parent_task_id,),
        )
        summary_path = _write_json_summary(
            campaign_output_root / "orchestrator" / "arena_prepare" / "summary.json",
            {
                "campaign_name": spec.name,
                "resolved_arena_spec_path": str(resolved_arena_spec_path),
                "selected_reference_agents": selected_reference_agents,
                "lapv1_agent_paths": {name: str(path) for name, path in lap_agent_paths.items()},
                "created_task_ids": task_ids,
            },
        )
        return {
            "summary_path": str(summary_path),
            "resolved_arena_spec_path": str(resolved_arena_spec_path),
            "created_task_ids": task_ids,
        }

    def write_phase10_summary(
        self,
        *,
        config_path: Path,
        resolved_arena_spec_path: Path,
    ) -> Path:
        """Write the final top-level Phase-10 campaign summary."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        spec = load_phase10_lapv1_arena_campaign_spec(resolved_config_path)
        campaign_output_root = resolve_repo_path(self._repo_root, Path(spec.output_root))
        train_metadata = _load_lap_train_metadata(
            resolve_repo_path(self._repo_root, Path(spec.lapv1_config_path))
        )
        verify_summary_path = campaign_output_root / "verify" / "summary.json"
        verify_metrics_path = campaign_output_root / "verify" / "metrics.json"
        arena_summary_path = campaign_output_root / "arena" / "summary.json"
        arena_matrix_path = campaign_output_root / "arena_matrix.json"
        from train.eval.arena import SelfplayArenaSpec

        resolved_arena_spec = SelfplayArenaSpec.from_json(
            resolved_arena_spec_path.read_text(encoding="utf-8")
        )
        summary = {
            "campaign_name": spec.name,
            "model_label": spec.model_label,
            "output_root": str(campaign_output_root),
            "lapv1_config_path": str(resolve_repo_path(self._repo_root, Path(spec.lapv1_config_path))),
            "lapv1_checkpoint": train_metadata["checkpoint_path"],
            "lapv1_summary_path": train_metadata["summary_path"],
            "workflow_summary_path": str(
                resolve_repo_path(self._repo_root, Path(spec.workflow_output_root)) / "summary.json"
            ),
            "train_dataset_summary_path": str(
                resolve_repo_path(self._repo_root, Path(spec.train_dataset_dir)) / "summary.json"
            ),
            "verify_dataset_summary_path": str(
                resolve_repo_path(self._repo_root, Path(spec.verify_dataset_dir)) / "summary.json"
            ),
            "lapv1_verify_path": str(verify_metrics_path),
            "lapv1_verify_summary_path": str(verify_summary_path),
            "lapv1_verify_metrics": _load_json(verify_metrics_path),
            "resolved_arena_spec_path": str(resolved_arena_spec_path),
            "resolved_arena_agents": list(resolved_arena_spec.agent_specs),
            "arena_summary_path": str(arena_summary_path),
            "arena_matrix_path": str(arena_matrix_path),
        }
        return _write_json_summary(campaign_output_root / "summary.json", summary)


def build_phase10_bootstrap_tasks(
    *,
    spec_path: Path,
    spec: Phase10Lapv1ArenaCampaignSpec,
    model_id: int,
) -> list[PlannedTask]:
    """Return the initial materialize-plus-prepare task pair for one campaign."""
    return [
        PlannedTask(
            key="materialize",
            task_type="phase10_materialize",
            capability="materialize",
            priority=80,
            max_attempts=2,
            payload=Phase10MaterializePayload(
                config_path=str(spec_path),
                output_root=spec.output_root,
                raw_dir=spec.merged_raw_dir,
                train_output_dir=spec.train_dataset_dir,
                verify_output_dir=spec.verify_dataset_dir,
                source_name=spec.phase5_source_name,
                seed=spec.phase5_seed,
                oracle_workers=spec.phase5_oracle_workers,
                oracle_batch_size=spec.phase5_oracle_batch_size,
                chunk_size=spec.phase5_chunk_size,
                log_every_chunks=spec.phase5_log_every_chunks,
            ).to_dict(),
        ),
        PlannedTask(
            key="workflow_prepare",
            task_type="phase10_workflow_prepare",
            capability="aggregate",
            priority=70,
            max_attempts=2,
            depends_on=("materialize",),
            payload=Phase10WorkflowPreparePayload(
                config_path=str(spec_path),
                model_id=model_id,
            ).to_dict(),
        ),
    ]


def build_phase10_workflow_tasks(
    *,
    spec_path: Path,
    spec: Phase10Lapv1ArenaCampaignSpec,
    model_id: int,
    train_summary: Mapping[str, Any],
    verify_summary: Mapping[str, Any],
    repo_root: Path,
) -> list[PlannedTask]:
    """Return the workflow/build/train/verify DAG after dataset summaries exist."""
    train_config_path = resolve_repo_path(repo_root, Path(spec.lapv1_config_path))
    train_metadata = _load_lap_train_metadata(train_config_path)
    train_split_counts = dict(train_summary.get("split_counts") or {})
    verify_split_counts = dict(verify_summary.get("split_counts") or {})
    tasks: list[PlannedTask] = []
    chunk_keys: list[str] = []
    for split_name, canonical_split, count in (
        ("train", "train", int(train_split_counts.get("train", 0))),
        ("validation", "validation", int(train_split_counts.get("validation", 0))),
        ("verify", "test", int(verify_split_counts.get("test", 0))),
    ):
        total_chunks = _chunk_count(count=count, chunk_size=spec.workflow_chunk_size)
        for chunk_index in range(1, total_chunks + 1):
            key = f"workflow_{split_name}_chunk_{chunk_index:04d}"
            chunk_keys.append(key)
            tasks.append(
                PlannedTask(
                    key=key,
                    task_type="phase10_workflow_chunk",
                    capability="workflow",
                    priority=60,
                    max_attempts=2,
                    payload=Phase10WorkflowChunkPayload(
                        config_path=str(spec_path),
                        split=split_name,
                        canonical_split=canonical_split,
                        chunk_index=chunk_index,
                        model_id=model_id,
                    ).to_dict(),
                )
            )
    tasks.append(
        PlannedTask(
            key="workflow_finalize",
            task_type="phase10_workflow_finalize",
            capability="aggregate",
            priority=50,
            max_attempts=2,
            depends_on=tuple(chunk_keys),
            payload=Phase10WorkflowFinalizePayload(
                config_path=str(spec_path),
                model_id=model_id,
            ).to_dict(),
        )
    )
    tasks.append(
        PlannedTask(
            key="train",
            task_type="train_lapv1",
            capability="train",
            priority=40,
            max_attempts=2,
            depends_on=("workflow_finalize",),
            payload=TrainLapv1Payload(
                config_path=str(train_config_path),
                model_id=model_id,
                model_label=spec.model_label,
            ).to_dict(),
        )
    )
    verify_output_dir = resolve_repo_path(repo_root, Path(spec.output_root)) / "verify"
    verify_output_path = verify_output_dir / "metrics.json"
    tasks.append(
        PlannedTask(
            key="verify",
            task_type="verify_lapv1",
            capability="verify",
            priority=30,
            max_attempts=2,
            depends_on=("train",),
            payload=VerifyLapv1Payload(
                config_path=str(train_config_path),
                model_id=model_id,
                checkpoint_path=train_metadata["checkpoint_path"],
                dataset_path=str(resolve_repo_path(repo_root, Path(spec.lapv1_verify_output_path))),
                output_dir=str(verify_output_dir),
                output_path=str(verify_output_path),
                top_k=int(train_metadata["top_k"]),
            ).to_dict(),
        )
    )
    tasks.append(
        PlannedTask(
            key="arena_prepare",
            task_type="phase10_arena_prepare",
            capability="aggregate",
            priority=20,
            max_attempts=2,
            depends_on=("verify",),
            payload=Phase10ArenaPreparePayload(
                config_path=str(spec_path),
                model_id=model_id,
            ).to_dict(),
        )
    )
    return tasks


def build_phase10_arena_tasks(
    *,
    spec_path: Path,
    spec: Phase10Lapv1ArenaCampaignSpec,
    resolved_arena_spec: "SelfplayArenaSpec",
    resolved_arena_spec_path: Path,
    model_id: int,
    repo_root: Path,
) -> list[PlannedTask]:
    """Return the distributed arena DAG once the resolved arena spec exists."""
    arena_output_root = resolve_repo_path(repo_root, Path(spec.output_root)) / "arena"
    matrix_path = resolve_repo_path(repo_root, Path(spec.output_root)) / "arena_matrix.json"
    matchups = resolved_arena_spec.expanded_matchups()
    tasks: list[PlannedTask] = []
    match_keys: list[str] = []
    for matchup_index in range(1, len(matchups) + 1):
        key = f"arena_match_{matchup_index:04d}"
        match_keys.append(key)
        tasks.append(
            PlannedTask(
                key=key,
                task_type="arena_match",
                capability="arena",
                priority=10,
                max_attempts=2,
                payload=ArenaMatchPayload(
                    config_path=str(spec_path),
                    resolved_arena_spec_path=str(resolved_arena_spec_path),
                    output_root=str(arena_output_root),
                    matchup_index=matchup_index,
                    model_id=model_id,
                ).to_dict(),
            )
        )
    tasks.append(
        PlannedTask(
            key="arena_finalize",
            task_type="arena_finalize",
            capability="aggregate",
            priority=5,
            max_attempts=2,
            depends_on=tuple(match_keys),
            payload=ArenaFinalizePayload(
                config_path=str(spec_path),
                resolved_arena_spec_path=str(resolved_arena_spec_path),
                output_root=str(arena_output_root),
                matrix_path=str(matrix_path),
                model_id=model_id,
            ).to_dict(),
        )
    )
    tasks.append(
        PlannedTask(
            key="phase10_finalize",
            task_type="phase10_finalize",
            capability="aggregate",
            priority=0,
            max_attempts=1,
            depends_on=("arena_finalize",),
            payload=Phase10FinalizePayload(
                config_path=str(spec_path),
                resolved_arena_spec_path=str(resolved_arena_spec_path),
                model_id=model_id,
            ).to_dict(),
        )
    )
    return tasks


def _chunk_count(*, count: int, chunk_size: int) -> int:
    return math.ceil(count / chunk_size) if count > 0 else 0


def _load_lap_train_metadata(config_path: Path) -> dict[str, Any]:
    payload = _load_json(config_path)
    output_dir = resolve_repo_path(config_path.parents[2], Path(str(payload["output_dir"])))
    bundle_dir = resolve_repo_path(
        config_path.parents[2],
        Path(str(payload["export"]["bundle_dir"])),
    )
    checkpoint_name = str(payload["export"]["checkpoint_name"])
    return {
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "summary.json"),
        "checkpoint_path": str(bundle_dir / checkpoint_name),
        "bundle_dir": str(bundle_dir),
        "top_k": int(payload.get("evaluation", {}).get("top_k", 3)),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _write_json_summary(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
