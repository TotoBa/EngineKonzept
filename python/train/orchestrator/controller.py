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
    select_pre_verify_lapv1_agent,
    select_reference_agents,
)
from train.orchestrator.db import OrchestratorDB
from train.orchestrator.models import (
    ArenaFinalizePayload,
    ArenaMatchPayload,
    LabelPgnCorpusPayload,
    LabelPgnCorpusIdleSlicePayload,
    Phase10ArtifactFinalizePayload,
    Phase10ArtifactWorkflowPreparePayload,
    Phase10ArenaPreparePayload,
    Phase10FinalizePayload,
    Phase5RawMergePayload,
    Phase10MaterializePayload,
    Phase10SeedCheckpointPayload,
    Phase10SelfplayFinalizePayload,
    Phase10SelfplayPreparePayload,
    Phase10SelfplayShardPayload,
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
        campaign_metadata: Mapping[str, Any] | None = None,
        model_metadata: Mapping[str, Any] | None = None,
        generation: int = 0,
        parent_model_id: int | None = None,
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
                **dict(campaign_metadata or {}),
            },
        )
        model_id = self._db.insert_model(
            campaign_id=campaign_id,
            status="queued",
            generation=generation,
            parent_model_id=parent_model_id,
            train_config_path=str(resolve_repo_path(self._repo_root, Path(spec.lapv1_config_path))),
            agent_spec_path=str(resolve_repo_path(self._repo_root, Path(spec.lapv1_agent_spec_path))),
            metadata={
                "model_label": spec.model_label,
                "expected_checkpoint_path": train_metadata["checkpoint_path"],
                **dict(model_metadata or {}),
            },
        )
        if spec.reuse_existing_artifacts:
            _assert_existing_phase10_artifacts(spec=spec, repo_root=self._repo_root)
            planned_tasks = build_phase10_reuse_existing_artifact_tasks(
                spec_path=resolved_config_path,
                spec=spec,
                model_id=model_id,
                repo_root=self._repo_root,
            )
        else:
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

    def submit_label_pgn_corpus_campaign(
        self,
        *,
        config_path: Path,
        kind: str = "label_pgn_corpus",
        campaign_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert one PGN-labeling campaign plus its single label task."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        label_config = _load_json(resolved_config_path)
        campaign_id = self._db.insert_campaign(
            name=str(label_config["name"]),
            kind=kind,
            status="queued",
            config_path=str(resolved_config_path),
            metadata={
                "work_dir": str(label_config["work_dir"]),
                "pgn_root": str(label_config["pgn_root"]),
                **dict(campaign_metadata or {}),
            },
        )
        task_ids = self._db.insert_planned_tasks(
            campaign_id=campaign_id,
            model_id=None,
            planned_tasks=build_label_pgn_corpus_tasks(
                config_path=resolved_config_path,
                config_payload=label_config,
            ),
        )
        return {
            "campaign_id": campaign_id,
            "task_ids": task_ids,
            "config_path": str(resolved_config_path),
        }

    def submit_idle_label_pgn_corpus_campaign(
        self,
        *,
        config_path: Path,
        kind: str = "label_pgn_corpus_idle",
        campaign_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert one low-priority, shardable PGN-labeling campaign for idle workers."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        label_config = _load_json(resolved_config_path)
        campaign_id = self._db.insert_campaign(
            name=str(label_config["name"]),
            kind=kind,
            status="queued",
            config_path=str(resolved_config_path),
            metadata={
                "work_root": str(label_config["work_root"]),
                "pgn_root": str(label_config["pgn_root"]),
                "shard_count": int(label_config.get("shard_count", 1)),
                **dict(campaign_metadata or {}),
            },
        )
        task_ids = self._db.insert_planned_tasks(
            campaign_id=campaign_id,
            model_id=None,
            planned_tasks=build_label_pgn_corpus_idle_tasks(
                config_path=resolved_config_path,
                config_payload=label_config,
            ),
        )
        return {
            "campaign_id": campaign_id,
            "task_ids": task_ids,
            "config_path": str(resolved_config_path),
        }

    def submit_idle_phase10_artifact_campaign(
        self,
        *,
        config_path: Path,
        kind: str = "phase10_idle_artifacts",
        campaign_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Insert one low-priority PGN -> LAPv2 Phase-10 artifact build campaign."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        idle_config = _load_idle_phase10_artifact_config(resolved_config_path)
        phase10_config_path = resolve_repo_path(
            self._repo_root,
            Path(str(idle_config["phase10_config_path"])),
        )
        phase10_spec = load_phase10_lapv1_arena_campaign_spec(phase10_config_path)
        campaign_id = self._db.insert_campaign(
            name=str(idle_config["name"]),
            kind=kind,
            status="queued",
            config_path=str(resolved_config_path),
            metadata={
                "phase10_config_path": str(phase10_config_path),
                "work_root": str(idle_config["work_root"]),
                "merged_raw_dir": phase10_spec.merged_raw_dir,
                "train_dataset_dir": phase10_spec.train_dataset_dir,
                "verify_dataset_dir": phase10_spec.verify_dataset_dir,
                "workflow_output_root": phase10_spec.workflow_output_root,
                "shard_count": int(idle_config.get("shard_count", 1)),
                **dict(campaign_metadata or {}),
            },
        )
        task_ids = self._db.insert_planned_tasks(
            campaign_id=campaign_id,
            model_id=None,
            planned_tasks=build_phase10_idle_artifact_tasks(
                config_path=resolved_config_path,
                config_payload=idle_config,
                phase10_config_path=phase10_config_path,
                phase10_spec=phase10_spec,
            ),
        )
        return {
            "campaign_id": campaign_id,
            "task_ids": task_ids,
            "config_path": str(resolved_config_path),
            "phase10_config_path": str(phase10_config_path),
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

    def expand_phase10_artifact_workflow(
        self,
        *,
        parent_task_id: int,
        campaign_id: int,
        config_path: Path,
    ) -> dict[str, Any]:
        """Expand the workflow-only DAG for a low-priority Phase-10 artifact build."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        idle_config = _load_idle_phase10_artifact_config(resolved_config_path)
        phase10_config_path = resolve_repo_path(
            self._repo_root,
            Path(str(idle_config["phase10_config_path"])),
        )
        spec = load_phase10_lapv1_arena_campaign_spec(phase10_config_path)
        train_summary = _load_json(
            resolve_repo_path(self._repo_root, Path(spec.train_dataset_dir)) / "summary.json"
        )
        verify_summary = _load_json(
            resolve_repo_path(self._repo_root, Path(spec.verify_dataset_dir)) / "summary.json"
        )
        planned_tasks = build_phase10_idle_artifact_workflow_tasks(
            config_path=resolved_config_path,
            config_payload=idle_config,
            phase10_config_path=phase10_config_path,
            spec=spec,
            train_summary=train_summary,
            verify_summary=verify_summary,
        )
        task_ids = self._db.insert_planned_tasks(
            campaign_id=campaign_id,
            model_id=None,
            planned_tasks=planned_tasks,
            extra_dependency_task_ids=(parent_task_id,),
        )
        work_root = resolve_repo_path(self._repo_root, Path(str(idle_config["work_root"])))
        summary_path = _write_json_summary(
            work_root / "orchestrator" / "workflow_prepare" / "summary.json",
            {
                "campaign_name": str(idle_config["name"]),
                "phase10_config_path": str(phase10_config_path),
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

    def expand_phase10_selfplay(
        self,
        *,
        parent_task_id: int,
        campaign_id: int,
        model_id: int,
        config_path: Path,
    ) -> dict[str, Any]:
        """Resolve the tracked LAP runtime and expand distributed pre-verify selfplay."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        spec = load_phase10_lapv1_arena_campaign_spec(resolved_config_path)
        campaign_output_root = resolve_repo_path(self._repo_root, Path(spec.output_root))
        resolved_agent_paths = materialize_resolved_lapv1_agent_specs(
            spec=spec,
            tracked_lapv1_agent_path=resolve_repo_path(
                self._repo_root, Path(spec.lapv1_agent_spec_path)
            ),
            output_root=campaign_output_root / "pre_verify_selfplay",
        )
        agent_name, agent_spec_path = select_pre_verify_lapv1_agent(
            spec,
            resolved_agent_paths=resolved_agent_paths,
            repo_root=self._repo_root,
        )
        planned_tasks = build_phase10_selfplay_tasks(
            spec_path=resolved_config_path,
            spec=spec,
            model_id=model_id,
            repo_root=self._repo_root,
            agent_name=agent_name,
            agent_spec_path=agent_spec_path,
        )
        task_ids = self._db.insert_planned_tasks(
            campaign_id=campaign_id,
            model_id=model_id,
            planned_tasks=planned_tasks,
            extra_dependency_task_ids=(parent_task_id,),
        )
        summary_path = _write_json_summary(
            campaign_output_root / "orchestrator" / "selfplay_prepare" / "summary.json",
            {
                "campaign_name": spec.name,
                "agent_name": agent_name,
                "agent_spec_path": str(agent_spec_path),
                "games": spec.pre_verify_selfplay_games,
                "games_per_task": spec.pre_verify_selfplay_games_per_task,
                "created_task_ids": task_ids,
            },
        )
        return {
            "summary_path": str(summary_path),
            "agent_name": agent_name,
            "agent_spec_path": str(agent_spec_path),
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
        selfplay_summary_path = campaign_output_root / "pre_verify_selfplay" / "summary.json"
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
            "pre_verify_selfplay_summary_path": (
                str(selfplay_summary_path) if selfplay_summary_path.exists() else None
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

    def write_phase10_artifact_summary(
        self,
        *,
        config_path: Path,
    ) -> Path:
        """Write the final top-level summary for one idle artifact-build campaign."""
        resolved_config_path = resolve_repo_path(self._repo_root, config_path)
        idle_config = _load_idle_phase10_artifact_config(resolved_config_path)
        phase10_config_path = resolve_repo_path(
            self._repo_root,
            Path(str(idle_config["phase10_config_path"])),
        )
        spec = load_phase10_lapv1_arena_campaign_spec(phase10_config_path)
        work_root = resolve_repo_path(self._repo_root, Path(str(idle_config["work_root"])))
        workflow_root = resolve_repo_path(self._repo_root, Path(spec.workflow_output_root))
        merged_raw_dir = resolve_repo_path(self._repo_root, Path(spec.merged_raw_dir))
        train_dataset_dir = resolve_repo_path(self._repo_root, Path(spec.train_dataset_dir))
        verify_dataset_dir = resolve_repo_path(self._repo_root, Path(spec.verify_dataset_dir))
        summary = {
            "campaign_name": str(idle_config["name"]),
            "work_root": str(work_root),
            "phase10_config_path": str(phase10_config_path),
            "merged_raw_dir": str(merged_raw_dir),
            "merged_raw_summary_path": str(merged_raw_dir / "selection_summary.json"),
            "train_dataset_summary_path": str(train_dataset_dir / "summary.json"),
            "verify_dataset_summary_path": str(verify_dataset_dir / "summary.json"),
            "workflow_output_root": str(workflow_root),
            "workflow_summary_path": str(workflow_root / "summary.json"),
            "hard_train_path": str(workflow_root / "all_unique_train_hard_v1" / "lapv1_train_hard.jsonl"),
            "hard_validation_path": str(
                workflow_root / "all_unique_validation_hard_v1" / "lapv1_validation_hard.jsonl"
            ),
        }
        return _write_json_summary(work_root / "summary.json", summary)


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


def build_phase10_reuse_existing_artifact_tasks(
    *,
    spec_path: Path,
    spec: Phase10Lapv1ArenaCampaignSpec,
    model_id: int,
    repo_root: Path,
) -> list[PlannedTask]:
    """Return the train/eval DAG when dataset and workflow artifacts already exist."""
    train_config_path = resolve_repo_path(repo_root, Path(spec.lapv1_config_path))
    train_metadata = _load_lap_train_metadata(train_config_path)
    if spec.skip_training:
        if spec.warm_start_source_checkpoint is None:
            raise ValueError(
                "phase10 config requested skip_training without warm_start_source_checkpoint"
            )
        entrypoint_key = "seed_checkpoint"
        tasks = [
            PlannedTask(
                key=entrypoint_key,
                task_type="phase10_seed_checkpoint",
                capability="aggregate",
                priority=45,
                max_attempts=2,
                payload=Phase10SeedCheckpointPayload(
                    config_path=str(spec_path),
                    model_id=model_id,
                    source_checkpoint_path=str(
                        resolve_repo_path(repo_root, Path(spec.warm_start_source_checkpoint))
                    ),
                    target_checkpoint_path=str(train_metadata["checkpoint_path"]),
                    output_dir=str(train_metadata["output_dir"]),
                ).to_dict(),
            )
        ]
    else:
        entrypoint_key = "train"
        tasks = [
            PlannedTask(
                key=entrypoint_key,
                task_type="train_lapv1",
                capability="train",
                priority=40,
                max_attempts=2,
                payload=TrainLapv1Payload(
                    config_path=str(train_config_path),
                    model_id=model_id,
                    model_label=spec.model_label,
                ).to_dict(),
            )
        ]
    if spec.pre_verify_selfplay_games > 0:
        tasks.append(
            PlannedTask(
                key="selfplay_prepare",
                task_type="phase10_selfplay_prepare",
                capability="aggregate",
                priority=30,
                max_attempts=2,
                depends_on=(entrypoint_key,),
                payload=Phase10SelfplayPreparePayload(
                    config_path=str(spec_path),
                    model_id=model_id,
                ).to_dict(),
            )
        )
    else:
        tasks.extend(
            _build_phase10_verify_and_arena_prepare_tasks(
                spec=spec,
                model_id=model_id,
                repo_root=repo_root,
                train_config_path=train_config_path,
                train_metadata=train_metadata,
                verify_depends_on=(entrypoint_key,),
                spec_path=spec_path,
            )
        )
    return tasks


def build_label_pgn_corpus_tasks(
    *,
    config_path: Path,
    config_payload: Mapping[str, Any],
) -> list[PlannedTask]:
    """Return the single-task DAG for one resumable PGN-labeling campaign."""
    return [
        PlannedTask(
            key="label",
            task_type="label_pgn_corpus",
            capability="label",
            priority=90,
            max_attempts=2,
            payload=LabelPgnCorpusPayload(
                config_path=str(config_path),
                pgn_root=str(config_payload["pgn_root"]),
                pgn_glob=str(config_payload.get("glob", "**/*.pgn")),
                engine_path=str(config_payload.get("engine_path", "/usr/games/stockfish18")),
                work_dir=str(config_payload["work_dir"]),
                target_train_records=int(config_payload["target_train_records"]),
                target_verify_records=int(config_payload["target_verify_records"]),
                min_ply=int(config_payload.get("min_ply", 8)),
                max_ply=int(config_payload.get("max_ply", 80)),
                ply_stride=int(config_payload.get("ply_stride", 2)),
                engine_nodes=int(config_payload.get("engine_nodes", 1500)),
                hash_mb=int(config_payload.get("hash_mb", 32)),
                threads=int(config_payload.get("threads", 1)),
                split_seed=str(config_payload.get("split_seed", "phase5-stockfish-unique-v1")),
                verify_divisor=int(config_payload.get("verify_divisor", 1000)),
                progress_every=int(config_payload.get("progress_every", 1000)),
                max_games=int(config_payload.get("max_games", 0)),
                export_jsonl_on_complete=bool(
                    config_payload.get("export_jsonl_on_complete", True)
                ),
                complete_at_eof=bool(config_payload.get("complete_at_eof", False)),
            ).to_dict(),
        )
    ]


def build_label_pgn_corpus_idle_tasks(
    *,
    config_path: Path,
    config_payload: Mapping[str, Any],
) -> list[PlannedTask]:
    """Return one low-priority idle labeling task per configured shard."""
    shard_count = int(config_payload.get("shard_count", 1))
    if shard_count <= 0:
        raise ValueError("idle label shard_count must be positive")
    work_root = Path(str(config_payload["work_root"]))
    tasks: list[PlannedTask] = []
    for shard_index in range(1, shard_count + 1):
        shard_name = f"shard_{shard_index:02d}"
        tasks.append(
            PlannedTask(
                key=f"idle_label_{shard_name}",
                task_type="label_pgn_corpus_idle_slice",
                capability=str(config_payload.get("capability", "label_idle")),
                priority=int(config_payload.get("priority", -100)),
                max_attempts=int(config_payload.get("max_attempts", 1000)),
                payload=LabelPgnCorpusIdleSlicePayload(
                    config_path=str(config_path),
                    pgn_root=str(config_payload["pgn_root"]),
                    pgn_glob=str(config_payload.get("glob", "**/*.pgn")),
                    engine_path=str(config_payload.get("engine_path", "/usr/games/stockfish18")),
                    work_dir=str(work_root / shard_name),
                    target_train_records=_split_target_across_shards(
                        total=int(config_payload["target_train_records"]),
                        shard_count=shard_count,
                        shard_index=shard_index,
                    ),
                    target_verify_records=_split_target_across_shards(
                        total=int(config_payload["target_verify_records"]),
                        shard_count=shard_count,
                        shard_index=shard_index,
                    ),
                    min_ply=int(config_payload.get("min_ply", 8)),
                    max_ply=int(config_payload.get("max_ply", 80)),
                    ply_stride=int(config_payload.get("ply_stride", 2)),
                    engine_nodes=int(config_payload.get("engine_nodes", 1500)),
                    hash_mb=int(config_payload.get("hash_mb", 32)),
                    threads=int(config_payload.get("threads", 1)),
                    split_seed=str(config_payload.get("split_seed", "phase5-stockfish-unique-v1")),
                    verify_divisor=int(config_payload.get("verify_divisor", 1000)),
                    progress_every=int(config_payload.get("progress_every", 1000)),
                    max_games=int(config_payload.get("max_games", 0)),
                    file_shard_index=shard_index,
                    file_shard_count=shard_count,
                    run_max_games=int(config_payload.get("run_max_games", 0)),
                    export_jsonl_on_complete=bool(
                        config_payload.get("export_jsonl_on_complete", True)
                    ),
                    complete_at_eof=bool(config_payload.get("complete_at_eof", False)),
                ).to_dict(),
            )
        )
    return tasks


def build_phase10_idle_artifact_tasks(
    *,
    config_path: Path,
    config_payload: Mapping[str, Any],
    phase10_config_path: Path,
    phase10_spec: Phase10Lapv1ArenaCampaignSpec,
) -> list[PlannedTask]:
    """Return the low-priority DAG for PGN -> Phase-10 artifact construction."""
    work_root = Path(str(config_payload["work_root"]))
    label_tasks = build_label_pgn_corpus_idle_tasks(
        config_path=config_path,
        config_payload={
            **dict(config_payload),
            "work_root": str(work_root / "label_shards"),
            "capability": str(config_payload.get("label_capability", "label_idle")),
            "priority": int(config_payload.get("label_priority", -100)),
            "complete_at_eof": bool(config_payload.get("complete_at_eof", True)),
        },
    )
    merge_task = PlannedTask(
        key="merge_raw",
        task_type="phase5_raw_merge",
        capability=str(config_payload.get("merge_capability", "aggregate_idle")),
        priority=int(config_payload.get("merge_priority", -90)),
        max_attempts=2,
        depends_on=tuple(task.key for task in label_tasks),
        payload=Phase5RawMergePayload(
            config_path=str(config_path),
            output_dir=phase10_spec.merged_raw_dir,
            source_dirs=tuple(
                str(work_root / "label_shards" / f"shard_{shard_index:02d}")
                for shard_index in range(1, int(config_payload.get("shard_count", 1)) + 1)
            ),
        ).to_dict(),
    )
    return [
        *label_tasks,
        merge_task,
        PlannedTask(
            key="materialize",
            task_type="phase10_materialize",
            capability=str(config_payload.get("materialize_capability", "materialize_idle")),
            priority=int(config_payload.get("materialize_priority", -80)),
            max_attempts=2,
            depends_on=("merge_raw",),
            payload=Phase10MaterializePayload(
                config_path=str(phase10_config_path),
                output_root=phase10_spec.output_root,
                raw_dir=phase10_spec.merged_raw_dir,
                train_output_dir=phase10_spec.train_dataset_dir,
                verify_output_dir=phase10_spec.verify_dataset_dir,
                source_name=phase10_spec.phase5_source_name,
                seed=phase10_spec.phase5_seed,
                oracle_workers=phase10_spec.phase5_oracle_workers,
                oracle_batch_size=phase10_spec.phase5_oracle_batch_size,
                chunk_size=phase10_spec.phase5_chunk_size,
                log_every_chunks=phase10_spec.phase5_log_every_chunks,
            ).to_dict(),
        ),
        PlannedTask(
            key="workflow_prepare",
            task_type="phase10_artifact_workflow_prepare",
            capability=str(config_payload.get("aggregate_capability", "aggregate_idle")),
            priority=int(config_payload.get("aggregate_priority", -70)),
            max_attempts=2,
            depends_on=("materialize",),
            payload=Phase10ArtifactWorkflowPreparePayload(
                config_path=str(config_path),
            ).to_dict(),
        ),
    ]


def build_phase10_idle_artifact_workflow_tasks(
    *,
    config_path: Path,
    config_payload: Mapping[str, Any],
    phase10_config_path: Path,
    spec: Phase10Lapv1ArenaCampaignSpec,
    train_summary: Mapping[str, Any],
    verify_summary: Mapping[str, Any],
) -> list[PlannedTask]:
    """Return the workflow-only Phase-10 DAG for one idle artifact-build campaign."""
    train_split_counts = dict(train_summary.get("split_counts") or {})
    verify_split_counts = dict(verify_summary.get("split_counts") or {})
    tasks: list[PlannedTask] = []
    chunk_keys: list[str] = []
    workflow_capability = str(config_payload.get("workflow_capability", "workflow_idle"))
    workflow_priority = int(config_payload.get("workflow_priority", -60))
    aggregate_capability = str(config_payload.get("aggregate_capability", "aggregate_idle"))
    aggregate_priority = int(config_payload.get("aggregate_priority", -70))
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
                    capability=workflow_capability,
                    priority=workflow_priority,
                    max_attempts=2,
                    payload=Phase10WorkflowChunkPayload(
                        config_path=str(phase10_config_path),
                        split=split_name,
                        canonical_split=canonical_split,
                        chunk_index=chunk_index,
                        model_id=0,
                    ).to_dict(),
                )
            )
    tasks.append(
        PlannedTask(
            key="workflow_finalize",
            task_type="phase10_workflow_finalize",
            capability=aggregate_capability,
            priority=aggregate_priority,
            max_attempts=2,
            depends_on=tuple(chunk_keys),
            payload=Phase10WorkflowFinalizePayload(
                config_path=str(phase10_config_path),
                model_id=0,
            ).to_dict(),
        )
    )
    tasks.append(
        PlannedTask(
            key="artifact_finalize",
            task_type="phase10_artifact_finalize",
            capability=aggregate_capability,
            priority=int(config_payload.get("finalize_priority", aggregate_priority + 10)),
            max_attempts=1,
            depends_on=("workflow_finalize",),
            payload=Phase10ArtifactFinalizePayload(
                config_path=str(config_path),
            ).to_dict(),
        )
    )
    return tasks


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
    if spec.pre_verify_selfplay_games > 0:
        tasks.append(
            PlannedTask(
                key="selfplay_prepare",
                task_type="phase10_selfplay_prepare",
                capability="aggregate",
                priority=30,
                max_attempts=2,
                depends_on=("train",),
                payload=Phase10SelfplayPreparePayload(
                    config_path=str(spec_path),
                    model_id=model_id,
                ).to_dict(),
            )
        )
    else:
        tasks.extend(
            _build_phase10_verify_and_arena_prepare_tasks(
                spec=spec,
                model_id=model_id,
                repo_root=repo_root,
                train_config_path=train_config_path,
                train_metadata=train_metadata,
                verify_depends_on=("train",),
                spec_path=spec_path,
            )
        )
    return tasks


def build_phase10_selfplay_tasks(
    *,
    spec_path: Path,
    spec: Phase10Lapv1ArenaCampaignSpec,
    model_id: int,
    repo_root: Path,
    agent_name: str,
    agent_spec_path: Path,
) -> list[PlannedTask]:
    """Return the distributed pre-verify selfplay DAG for one tracked LAP runtime."""
    if spec.pre_verify_selfplay_games <= 0:
        raise ValueError("pre_verify_selfplay_games must be positive to build selfplay tasks")
    games_per_task = max(1, spec.pre_verify_selfplay_games_per_task)
    total_shards = _chunk_count(count=spec.pre_verify_selfplay_games, chunk_size=games_per_task)
    output_root = resolve_repo_path(repo_root, Path(spec.output_root)) / "pre_verify_selfplay"
    max_plies = spec.pre_verify_selfplay_max_plies or spec.arena_default_max_plies
    tasks: list[PlannedTask] = []
    shard_keys: list[str] = []
    for shard_index in range(1, total_shards + 1):
        starting_game_index = (shard_index - 1) * games_per_task
        games = min(games_per_task, spec.pre_verify_selfplay_games - starting_game_index)
        key = f"selfplay_shard_{shard_index:04d}"
        shard_keys.append(key)
        tasks.append(
            PlannedTask(
                key=key,
                task_type="phase10_selfplay_shard",
                capability="selfplay",
                priority=25,
                max_attempts=2,
                payload=Phase10SelfplayShardPayload(
                    config_path=str(spec_path),
                    agent_spec_path=str(agent_spec_path),
                    agent_name=agent_name,
                    output_root=str(output_root),
                    shard_index=shard_index,
                    starting_game_index=starting_game_index,
                    games=games,
                    max_plies=max_plies,
                    model_id=model_id,
                ).to_dict(),
            )
        )
    tasks.append(
        PlannedTask(
            key="selfplay_finalize",
            task_type="phase10_selfplay_finalize",
            capability="aggregate",
            priority=24,
            max_attempts=2,
            depends_on=tuple(shard_keys),
            payload=Phase10SelfplayFinalizePayload(
                config_path=str(spec_path),
                agent_spec_path=str(agent_spec_path),
                agent_name=agent_name,
                output_root=str(output_root),
                model_id=model_id,
            ).to_dict(),
        )
    )
    tasks.extend(
        _build_phase10_verify_and_arena_prepare_tasks(
            spec=spec,
            model_id=model_id,
            repo_root=repo_root,
            train_config_path=resolve_repo_path(repo_root, Path(spec.lapv1_config_path)),
            train_metadata=_load_lap_train_metadata(
                resolve_repo_path(repo_root, Path(spec.lapv1_config_path))
            ),
            verify_depends_on=("selfplay_finalize",),
            spec_path=spec_path,
        )
    )
    return tasks


def _build_phase10_verify_and_arena_prepare_tasks(
    *,
    spec: Phase10Lapv1ArenaCampaignSpec,
    model_id: int,
    repo_root: Path,
    train_config_path: Path,
    train_metadata: Mapping[str, Any],
    verify_depends_on: tuple[str, ...],
    spec_path: Path,
) -> list[PlannedTask]:
    verify_output_dir = resolve_repo_path(repo_root, Path(spec.output_root)) / "verify"
    verify_output_path = verify_output_dir / "metrics.json"
    return [
        PlannedTask(
            key="verify",
            task_type="verify_lapv1",
            capability="verify",
            priority=20,
            max_attempts=2,
            depends_on=verify_depends_on,
            payload=VerifyLapv1Payload(
                config_path=str(train_config_path),
                model_id=model_id,
                checkpoint_path=train_metadata["checkpoint_path"],
                dataset_path=str(resolve_repo_path(repo_root, Path(spec.lapv1_verify_output_path))),
                output_dir=str(verify_output_dir),
                output_path=str(verify_output_path),
                top_k=int(train_metadata["top_k"]),
            ).to_dict(),
        ),
        PlannedTask(
            key="arena_prepare",
            task_type="phase10_arena_prepare",
            capability="aggregate",
            priority=10,
            max_attempts=2,
            depends_on=("verify",),
            payload=Phase10ArenaPreparePayload(
                config_path=str(spec_path),
                model_id=model_id,
            ).to_dict(),
        ),
    ]


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


def _split_target_across_shards(*, total: int, shard_count: int, shard_index: int) -> int:
    if total < 0:
        raise ValueError("split target total must be non-negative")
    if shard_count <= 0:
        raise ValueError("split target shard_count must be positive")
    if not 1 <= shard_index <= shard_count:
        raise ValueError("split target shard_index out of range")
    if total == 0:
        return 0
    base = total // shard_count
    remainder = total % shard_count
    return base + (1 if shard_index <= remainder else 0)


def _assert_existing_phase10_artifacts(
    *,
    spec: Phase10Lapv1ArenaCampaignSpec,
    repo_root: Path,
) -> None:
    required_paths = (
        resolve_repo_path(repo_root, Path(spec.train_dataset_dir)) / "summary.json",
        resolve_repo_path(repo_root, Path(spec.verify_dataset_dir)) / "summary.json",
        resolve_repo_path(repo_root, Path(spec.workflow_output_root)) / "summary.json",
        resolve_repo_path(repo_root, Path(spec.lapv1_verify_output_path)),
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "phase10 config requested reuse_existing_artifacts, but required files are missing: "
            + ", ".join(missing)
        )


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


def _load_idle_phase10_artifact_config(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    required = (
        "name",
        "phase10_config_path",
        "pgn_root",
        "work_root",
        "target_train_records",
        "target_verify_records",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(
            f"{path}: missing required idle phase10 artifact keys: {', '.join(missing)}"
        )
    if int(payload.get("shard_count", 1)) <= 0:
        raise ValueError(f"{path}: shard_count must be positive")
    return payload


def _write_json_summary(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
