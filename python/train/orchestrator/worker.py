"""Worker loop and task handlers for the MySQL-backed control plane."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, IO, Sequence

from train.eval.phase10_campaign import (
    load_phase10_lapv1_arena_campaign_spec,
    resolve_repo_path,
)
from train.orchestrator.controller import OrchestratorController
from train.orchestrator.db import OrchestratorDB
from train.orchestrator.lease import LeaseHeartbeat
from train.orchestrator.models import ArtifactRef, TaskResult, TaskRow, WorkerDescriptor


class OrchestratorWorker:
    """Single-process worker that claims, executes, and records task attempts."""

    _HARD_TRAIN_MAX_EXAMPLES = 150_000
    _HARD_VALIDATION_MAX_EXAMPLES = 15_000
    _HARD_LOG_EVERY = 10_000

    def __init__(
        self,
        *,
        db: OrchestratorDB,
        controller: OrchestratorController,
        descriptor: WorkerDescriptor,
        repo_root: Path,
        log_root: Path,
        lease_seconds: int = 300,
        heartbeat_interval_seconds: float = 30.0,
        distributed_task_threads: int = 1,
    ) -> None:
        self._db = db
        self._controller = controller
        self._descriptor = descriptor
        self._repo_root = repo_root
        self._log_root = log_root
        self._lease_seconds = lease_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._distributed_task_threads = max(1, distributed_task_threads)

    def register(self) -> None:
        """Register or refresh the worker row in MySQL."""
        self._db.register_worker(
            worker_id=self._descriptor.worker_id,
            hostname=self._descriptor.hostname,
            capabilities=self._descriptor.capabilities,
            scratch_root=self._descriptor.scratch_root,
            version=self._descriptor.version,
            metadata=self._descriptor.metadata,
        )
        self._db.heartbeat_worker(
            worker_id=self._descriptor.worker_id,
            status="idle",
            current_task_id=None,
        )

    def run_once(self) -> bool:
        """Claim and execute at most one task."""
        self.register()
        self._db.requeue_expired_tasks()
        capabilities = self._claim_capabilities()
        if not capabilities:
            self._db.heartbeat_worker(
                worker_id=self._descriptor.worker_id,
                status="idle",
                current_task_id=None,
            )
            return False
        claimed = self._db.claim_tasks(
            worker_id=self._descriptor.worker_id,
            capabilities=capabilities,
            lease_seconds=self._lease_seconds,
            limit=1,
        )
        if not claimed:
            self._db.heartbeat_worker(
                worker_id=self._descriptor.worker_id,
                status="idle",
                current_task_id=None,
            )
            return False
        task = claimed[0]
        stdout_path, stderr_path = self._attempt_log_paths(task)
        attempt_id = self._db.record_task_attempt_start(
            task_id=task.id,
            worker_id=self._descriptor.worker_id,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )
        self._db.heartbeat_worker(
            worker_id=self._descriptor.worker_id,
            status="busy",
            current_task_id=task.id,
            lease_seconds=self._lease_seconds,
        )
        result_summary_path: str | None = None
        details: dict[str, Any] = {"task_type": task.task_type}
        exit_code = 0
        try:
            self._before_execute(task)
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                stdout_path.open("w", encoding="utf-8") as stdout_handle,
                stderr_path.open("w", encoding="utf-8") as stderr_handle,
                redirect_stdout(stdout_handle),
                redirect_stderr(stderr_handle),
                LeaseHeartbeat(
                    client=self._db,
                    worker_id=self._descriptor.worker_id,
                    task_id=task.id,
                    lease_seconds=self._lease_seconds,
                    interval_seconds=self._heartbeat_interval_seconds,
                ),
            ):
                result = self._execute_task(
                    task=task,
                    stdout_handle=stdout_handle,
                    stderr_handle=stderr_handle,
                )
            result_summary_path = result.summary_path
            if bool(result.metadata.get("requeue_requested", False)):
                self._db.requeue_task(task_id=task.id, result=result)
            else:
                self._db.mark_task_succeeded(
                    task_id=task.id,
                    result=result,
                    artifacts=result.artifacts,
                )
                self._after_success(task=task, result=result)
        except Exception as exc:
            exit_code = 1
            details.update(
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            self._db.mark_task_failed(
                task=task,
                error_message=str(exc),
                retry_allowed=task.attempt_count < task.max_attempts,
                details=details,
            )
        finally:
            self._db.finish_task_attempt(
                attempt_id=attempt_id,
                exit_code=exit_code,
                result_summary_path=result_summary_path,
                details=details,
            )
            self._db.heartbeat_worker(
                worker_id=self._descriptor.worker_id,
                status="idle",
                current_task_id=None,
            )
        return True

    def run_forever(self, *, poll_interval_seconds: float) -> None:
        """Poll forever, executing one task per iteration when available."""
        while True:
            claimed = self.run_once()
            if not claimed:
                time.sleep(poll_interval_seconds)

    def _claim_capabilities(self) -> tuple[str, ...]:
        if self._db.has_active_training_tasks():
            return self._descriptor.capabilities
        return tuple(
            capability for capability in self._descriptor.capabilities if not capability.endswith("_idle")
        )

    def _execute_task(
        self,
        *,
        task: TaskRow,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> TaskResult:
        task_type = task.task_type
        if task_type == "label_pgn_corpus":
            return self._handle_label_pgn_corpus(task, stdout_handle, stderr_handle)
        if task_type == "label_pgn_corpus_idle_slice":
            return self._handle_label_pgn_corpus_idle_slice(task, stdout_handle, stderr_handle)
        if task_type == "phase5_raw_merge":
            return self._handle_phase5_raw_merge(task)
        if task_type == "phase10_materialize":
            return self._handle_phase10_materialize(task, stdout_handle, stderr_handle)
        if task_type == "phase10_workflow_prepare":
            return self._handle_phase10_workflow_prepare(task)
        if task_type == "phase10_artifact_workflow_prepare":
            return self._handle_phase10_artifact_workflow_prepare(task)
        if task_type == "phase10_workflow_chunk":
            return self._handle_phase10_workflow_chunk(task, stdout_handle, stderr_handle)
        if task_type == "phase10_workflow_finalize":
            return self._handle_phase10_workflow_finalize(task, stdout_handle, stderr_handle)
        if task_type == "train_lapv1":
            return self._handle_train_lapv1(task, stdout_handle, stderr_handle)
        if task_type == "phase10_selfplay_prepare":
            return self._handle_phase10_selfplay_prepare(task)
        if task_type == "phase10_selfplay_shard":
            return self._handle_phase10_selfplay_shard(task)
        if task_type == "phase10_selfplay_finalize":
            return self._handle_phase10_selfplay_finalize(task)
        if task_type == "verify_lapv1":
            return self._handle_verify_lapv1(task, stdout_handle, stderr_handle)
        if task_type == "phase10_arena_prepare":
            return self._handle_phase10_arena_prepare(task)
        if task_type == "arena_match":
            return self._handle_arena_match(task)
        if task_type == "arena_finalize":
            return self._handle_arena_finalize(task)
        if task_type == "phase10_artifact_finalize":
            return self._handle_phase10_artifact_finalize(task)
        if task_type == "phase10_finalize":
            return self._handle_phase10_finalize(task)
        raise ValueError(f"unsupported task type: {task_type}")

    def _handle_label_pgn_corpus(
        self,
        task: TaskRow,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> TaskResult:
        return self._run_label_pgn_corpus_task(
            task=task,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            require_completed=True,
            allow_requeue=False,
        )

    def _handle_label_pgn_corpus_idle_slice(
        self,
        task: TaskRow,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> TaskResult:
        return self._run_label_pgn_corpus_task(
            task=task,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            require_completed=False,
            allow_requeue=True,
        )

    def _run_label_pgn_corpus_task(
        self,
        *,
        task: TaskRow,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
        require_completed: bool,
        allow_requeue: bool,
    ) -> TaskResult:
        from scripts.build_unique_stockfish_pgn_corpus import export_unique_corpus_snapshot

        payload = dict(task.payload)
        work_dir = resolve_repo_path(self._repo_root, Path(str(payload["work_dir"])))
        command = [
            sys.executable,
            "-u",
            str(self._repo_root / "python" / "scripts" / "build_unique_stockfish_pgn_corpus.py"),
            "--pgn-root",
            str(payload["pgn_root"]),
            "--glob",
            str(payload["pgn_glob"]),
            "--engine-path",
            str(payload["engine_path"]),
            "--work-dir",
            str(payload["work_dir"]),
            "--target-train-records",
            str(payload["target_train_records"]),
            "--target-verify-records",
            str(payload["target_verify_records"]),
            "--min-ply",
            str(payload["min_ply"]),
            "--max-ply",
            str(payload["max_ply"]),
            "--ply-stride",
            str(payload["ply_stride"]),
            "--engine-nodes",
            str(payload["engine_nodes"]),
            "--hash-mb",
            str(payload["hash_mb"]),
            "--threads",
            str(payload["threads"]),
            "--split-seed",
            str(payload["split_seed"]),
            "--verify-divisor",
            str(payload["verify_divisor"]),
            "--progress-every",
            str(payload["progress_every"]),
            "--max-games",
            str(payload["max_games"]),
        ]
        if not bool(payload.get("export_jsonl_on_complete", True)):
            command.append("--no-export-jsonl-on-complete")
        if bool(payload.get("complete_at_eof", False)):
            command.append("--complete-at-eof")
        file_shard_index = payload.get("file_shard_index")
        file_shard_count = payload.get("file_shard_count")
        if file_shard_index is not None:
            command.extend(["--file-shard-index", str(file_shard_index)])
        if file_shard_count is not None:
            command.extend(["--file-shard-count", str(file_shard_count)])
        run_max_games = int(payload.get("run_max_games", 0))
        if run_max_games > 0:
            command.extend(["--run-max-games", str(run_max_games)])
        self._run_command(
            command,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            thread_budget=self._distributed_task_threads,
        )
        progress_path = work_dir / "progress.json"
        if not progress_path.exists():
            raise FileNotFoundError(f"label progress file not found: {progress_path}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        database_path = work_dir / "corpus.sqlite3"
        if database_path.exists():
            export_summary = export_unique_corpus_snapshot(work_dir)
        else:
            export_summary = {
                "train_raw_path": str(work_dir / "train_raw.jsonl"),
                "verify_raw_path": str(work_dir / "verify_raw.jsonl"),
            }
        if require_completed and not bool(progress.get("completed")):
            raise RuntimeError(
                f"label campaign did not reach requested targets: {progress_path}"
            )
        train_raw_path = work_dir / "train_raw.jsonl"
        verify_raw_path = work_dir / "verify_raw.jsonl"
        if not train_raw_path.exists() or not verify_raw_path.exists():
            raise FileNotFoundError(
                "label campaign completed without exported raw corpora: "
                f"{train_raw_path} / {verify_raw_path}"
            )
        summary_path = self._write_json(
            work_dir / "summary.json",
            {
                "config_path": str(payload["config_path"]),
                "pgn_root": str(payload["pgn_root"]),
                "work_dir": str(work_dir),
                "progress_path": str(progress_path),
                "train_raw_path": str(train_raw_path),
                "verify_raw_path": str(verify_raw_path),
                "progress": progress,
                "export": export_summary,
            },
        )
        counts = dict(progress.get("counts") or {})
        metadata: dict[str, Any] = {}
        if allow_requeue and not bool(progress.get("completed")):
            metadata["requeue_requested"] = True
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(
                self._artifact_ref(
                    kind="raw_corpus_dir",
                    path=work_dir,
                    summary_path=summary_path,
                ),
                self._artifact_ref(
                    kind="raw_train_jsonl",
                    path=train_raw_path,
                    summary_path=summary_path,
                ),
                self._artifact_ref(
                    kind="raw_verify_jsonl",
                    path=verify_raw_path,
                    summary_path=summary_path,
                ),
            ),
            metrics={
                "train_records": int(counts.get("train", 0)),
                "verify_records": int(counts.get("verify", 0)),
                "games_seen": int(progress.get("games_seen", 0)),
            },
            metadata=metadata,
        )

    def _handle_phase5_raw_merge(self, task: TaskRow) -> TaskResult:
        from scripts.merge_phase5_raw_corpora import RawCorpusSourceSpec, merge_phase5_raw_corpora

        payload = dict(task.payload)
        output_dir = resolve_repo_path(self._repo_root, Path(str(payload["output_dir"])))
        source_specs = [
            RawCorpusSourceSpec(
                name=resolve_repo_path(self._repo_root, Path(str(source_dir))).name,
                raw_dir=resolve_repo_path(self._repo_root, Path(str(source_dir))),
            )
            for source_dir in list(payload.get("source_dirs") or [])
        ]
        merge_phase5_raw_corpora(
            source_specs=source_specs,
            output_dir=output_dir,
        )
        summary_path = output_dir / "selection_summary.json"
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(
                self._artifact_ref(
                    kind="raw_corpus_dir",
                    path=output_dir,
                    summary_path=summary_path,
                ),
                self._artifact_ref(
                    kind="raw_train_jsonl",
                    path=output_dir / "train_raw.jsonl",
                    summary_path=summary_path,
                ),
                self._artifact_ref(
                    kind="raw_verify_jsonl",
                    path=output_dir / "verify_raw.jsonl",
                    summary_path=summary_path,
                ),
            ),
        )

    def _handle_phase10_materialize(
        self,
        task: TaskRow,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> TaskResult:
        payload = dict(task.payload)
        output_root = resolve_repo_path(self._repo_root, Path(str(payload["output_root"])))
        train_output_dir = resolve_repo_path(self._repo_root, Path(str(payload["train_output_dir"])))
        verify_output_dir = resolve_repo_path(self._repo_root, Path(str(payload["verify_output_dir"])))
        command = [
            sys.executable,
            "-u",
            str(self._repo_root / "python" / "scripts" / "materialize_phase5_raw_tier.py"),
            "--raw-dir",
            str(payload["raw_dir"]),
            "--train-output-dir",
            str(payload["train_output_dir"]),
            "--verify-output-dir",
            str(payload["verify_output_dir"]),
            "--source-name",
            str(payload["source_name"]),
            "--seed",
            str(payload["seed"]),
            "--oracle-workers",
            str(payload["oracle_workers"]),
            "--oracle-batch-size",
            str(payload["oracle_batch_size"]),
            "--chunk-size",
            str(payload["chunk_size"]),
            "--log-every-chunks",
            str(payload["log_every_chunks"]),
        ]
        self._run_command(
            command,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            thread_budget=self._distributed_task_threads,
        )
        summary_path = self._write_json(
            output_root / "orchestrator" / "materialize" / "summary.json",
            {
                "train_summary_path": str(train_output_dir / "summary.json"),
                "verify_summary_path": str(verify_output_dir / "summary.json"),
            },
        )
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(
                self._artifact_ref(
                    kind="dataset_dir",
                    path=train_output_dir,
                    summary_path=train_output_dir / "summary.json",
                ),
                self._artifact_ref(
                    kind="dataset_dir",
                    path=verify_output_dir,
                    summary_path=verify_output_dir / "summary.json",
                ),
            ),
        )

    def _handle_phase10_workflow_prepare(self, task: TaskRow) -> TaskResult:
        payload = dict(task.payload)
        expanded = self._controller.expand_phase10_workflow(
            parent_task_id=task.id,
            campaign_id=task.campaign_id,
            model_id=int(payload["model_id"]),
            config_path=Path(str(payload["config_path"])),
        )
        return TaskResult(
            summary_path=str(expanded["summary_path"]),
            created_task_keys=tuple(sorted(dict(expanded["created_task_ids"]).keys())),
            metadata={"created_task_ids": dict(expanded["created_task_ids"])},
        )

    def _handle_phase10_artifact_workflow_prepare(self, task: TaskRow) -> TaskResult:
        payload = dict(task.payload)
        expanded = self._controller.expand_phase10_artifact_workflow(
            parent_task_id=task.id,
            campaign_id=task.campaign_id,
            config_path=Path(str(payload["config_path"])),
        )
        return TaskResult(
            summary_path=str(expanded["summary_path"]),
            created_task_keys=tuple(sorted(dict(expanded["created_task_ids"]).keys())),
            metadata={"created_task_ids": dict(expanded["created_task_ids"])},
        )

    def _handle_phase10_workflow_chunk(
        self,
        task: TaskRow,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> TaskResult:
        payload = dict(task.payload)
        spec = load_phase10_lapv1_arena_campaign_spec(Path(str(payload["config_path"])))
        split_name = str(payload["split"])
        chunk_index = int(payload["chunk_index"])
        command = [
            sys.executable,
            "-u",
            str(self._repo_root / "python" / "scripts" / "build_phase10_lapv1_workflow.py"),
            "--train-dataset-dir",
            spec.train_dataset_dir,
            "--verify-dataset-dir",
            spec.verify_dataset_dir,
            "--checkpoint",
            spec.proposer_checkpoint,
            "--teacher-engine",
            spec.teacher_engine_path,
            "--output-root",
            spec.workflow_output_root,
            "--multipv",
            str(spec.teacher_multipv),
            "--policy-temperature-cp",
            str(spec.teacher_policy_temperature_cp),
            "--top-k",
            str(spec.teacher_top_k),
            "--root-top-k",
            "4",
            "--chunk-size",
            str(spec.workflow_chunk_size),
            "--parallel-workers",
            "1",
            "--log-every",
            str(spec.workflow_log_every),
            "--skip-existing",
            "--skip-finalize",
            "--only-split",
            split_name,
        ]
        if split_name == "train":
            command.extend(["--train-chunk-start", str(chunk_index), "--train-chunk-end", str(chunk_index)])
        elif split_name == "validation":
            command.extend(
                [
                    "--validation-chunk-start",
                    str(chunk_index),
                    "--validation-chunk-end",
                    str(chunk_index),
                ]
            )
        else:
            command.extend(
                [
                    "--verify-chunk-start",
                    str(chunk_index),
                    "--verify-chunk-end",
                    str(chunk_index),
                ]
            )
        if spec.teacher_depth is not None:
            command.extend(["--depth", str(spec.teacher_depth)])
        elif spec.teacher_nodes is not None:
            command.extend(["--nodes", str(spec.teacher_nodes)])
        if spec.train_teacher_depth is not None:
            command.extend(["--train-depth", str(spec.train_teacher_depth)])
        if spec.validation_teacher_depth is not None:
            command.extend(["--validation-depth", str(spec.validation_teacher_depth)])
        if spec.verify_teacher_depth is not None:
            command.extend(["--verify-depth", str(spec.verify_teacher_depth)])
        self._run_command(
            command,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            thread_budget=self._distributed_task_threads,
        )
        workflow_root = resolve_repo_path(self._repo_root, Path(spec.workflow_output_root))
        split_dir = workflow_root / {
            "train": "all_unique_train_v1",
            "validation": "all_unique_validation_v1",
            "verify": "all_unique_verify_v1",
        }[split_name]
        chunk_dir = split_dir / "chunks" / f"chunk_{chunk_index:04d}_{(chunk_index - 1) * spec.workflow_chunk_size:08d}"
        summary_path = self._write_json(
            chunk_dir / "orchestrator" / "summary.json",
            {
                "split": split_name,
                "chunk_index": chunk_index,
                "workflow_summary_path": str(chunk_dir / "workflow.summary.json"),
                "planner_head_summary_path": str(chunk_dir / "planner_head.summary.json"),
                "lapv1_summary_path": str(chunk_dir / "lapv1.summary.json"),
            },
        )
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(self._artifact_ref(kind="workflow_chunk_dir", path=chunk_dir, summary_path=summary_path),),
        )

    def _handle_phase10_workflow_finalize(
        self,
        task: TaskRow,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> TaskResult:
        payload = dict(task.payload)
        spec = load_phase10_lapv1_arena_campaign_spec(Path(str(payload["config_path"])))
        command = [
            sys.executable,
            "-u",
            str(self._repo_root / "python" / "scripts" / "build_phase10_lapv1_workflow.py"),
            "--train-dataset-dir",
            spec.train_dataset_dir,
            "--verify-dataset-dir",
            spec.verify_dataset_dir,
            "--checkpoint",
            spec.proposer_checkpoint,
            "--teacher-engine",
            spec.teacher_engine_path,
            "--output-root",
            spec.workflow_output_root,
            "--multipv",
            str(spec.teacher_multipv),
            "--policy-temperature-cp",
            str(spec.teacher_policy_temperature_cp),
            "--top-k",
            str(spec.teacher_top_k),
            "--root-top-k",
            "4",
            "--chunk-size",
            str(spec.workflow_chunk_size),
            "--parallel-workers",
            "1",
            "--log-every",
            str(spec.workflow_log_every),
            "--finalize-only",
        ]
        if spec.teacher_depth is not None:
            command.extend(["--depth", str(spec.teacher_depth)])
        elif spec.teacher_nodes is not None:
            command.extend(["--nodes", str(spec.teacher_nodes)])
        if spec.train_teacher_depth is not None:
            command.extend(["--train-depth", str(spec.train_teacher_depth)])
        if spec.validation_teacher_depth is not None:
            command.extend(["--validation-depth", str(spec.validation_teacher_depth)])
        if spec.verify_teacher_depth is not None:
            command.extend(["--verify-depth", str(spec.verify_teacher_depth)])
        self._run_command(
            command,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            thread_budget=self._distributed_task_threads,
        )
        workflow_root = resolve_repo_path(self._repo_root, Path(spec.workflow_output_root))
        hard_paths = self._build_phase10_hard_subsets(
            workflow_root=workflow_root,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )
        summary_path = self._write_json(
            workflow_root / "orchestrator" / "workflow_finalize" / "summary.json",
            {
                "workflow_summary_path": str(workflow_root / "summary.json"),
                "hard_train_path": str(hard_paths["train_path"]),
                "hard_train_summary_path": str(hard_paths["train_summary_path"]),
                "hard_validation_path": str(hard_paths["validation_path"]),
                "hard_validation_summary_path": str(hard_paths["validation_summary_path"]),
            },
        )
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(
                self._artifact_ref(
                    kind="workflow_root",
                    path=workflow_root,
                    summary_path=workflow_root / "summary.json",
                ),
            ),
        )

    def _build_phase10_hard_subsets(
        self,
        *,
        workflow_root: Path,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> dict[str, Path]:
        full_train_path = workflow_root / "all_unique_train_v1" / "lapv1_train.jsonl"
        full_validation_path = workflow_root / "all_unique_validation_v1" / "lapv1_validation.jsonl"
        hard_train_path = workflow_root / "all_unique_train_hard_v1" / "lapv1_train_hard.jsonl"
        hard_validation_path = (
            workflow_root / "all_unique_validation_hard_v1" / "lapv1_validation_hard.jsonl"
        )
        self._build_lapv1_hard_positions_dataset(
            input_path=full_train_path,
            output_path=hard_train_path,
            max_examples=self._HARD_TRAIN_MAX_EXAMPLES,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )
        self._build_lapv1_hard_positions_dataset(
            input_path=full_validation_path,
            output_path=hard_validation_path,
            max_examples=self._HARD_VALIDATION_MAX_EXAMPLES,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )
        return {
            "train_path": hard_train_path,
            "train_summary_path": hard_train_path.parent / "lapv1_train_hard.summary.json",
            "validation_path": hard_validation_path,
            "validation_summary_path": (
                hard_validation_path.parent / "lapv1_validation_hard.summary.json"
            ),
        }

    def _build_lapv1_hard_positions_dataset(
        self,
        *,
        input_path: Path,
        output_path: Path,
        max_examples: int,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> None:
        command = [
            sys.executable,
            "-u",
            str(self._repo_root / "python" / "scripts" / "build_lapv1_hard_positions_dataset.py"),
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--max-examples",
            str(max_examples),
            "--log-every",
            str(self._HARD_LOG_EVERY),
        ]
        self._run_command(
            command,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            thread_budget=self._distributed_task_threads,
        )

    def _handle_train_lapv1(
        self,
        task: TaskRow,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> TaskResult:
        payload = dict(task.payload)
        config_path = Path(str(payload["config_path"]))
        train_metadata = self._load_train_metadata(config_path)
        command = [
            sys.executable,
            "-u",
            str(self._repo_root / "python" / "scripts" / "train_lapv1.py"),
            "--config",
            str(config_path),
        ]
        self._run_command(command, stdout_handle=stdout_handle, stderr_handle=stderr_handle)
        self._db.update_model_record(
            int(payload["model_id"]),
            checkpoint_path=train_metadata["checkpoint_path"],
            bundle_path=train_metadata["bundle_dir"],
            status="trained",
        )
        return TaskResult(
            summary_path=train_metadata["summary_path"],
            artifacts=(
                self._artifact_ref(
                    kind="checkpoint",
                    path=Path(train_metadata["checkpoint_path"]),
                    summary_path=Path(train_metadata["summary_path"]),
                ),
                self._artifact_ref(
                    kind="bundle_dir",
                    path=Path(train_metadata["bundle_dir"]),
                    summary_path=Path(train_metadata["summary_path"]),
                ),
            ),
        )

    def _handle_phase10_selfplay_prepare(self, task: TaskRow) -> TaskResult:
        payload = dict(task.payload)
        expanded = self._controller.expand_phase10_selfplay(
            parent_task_id=task.id,
            campaign_id=task.campaign_id,
            model_id=int(payload["model_id"]),
            config_path=Path(str(payload["config_path"])),
        )
        return TaskResult(
            summary_path=str(expanded["summary_path"]),
            created_task_keys=tuple(sorted(dict(expanded["created_task_ids"]).keys())),
            metadata={
                "created_task_ids": dict(expanded["created_task_ids"]),
                "agent_name": expanded["agent_name"],
                "agent_spec_path": expanded["agent_spec_path"],
            },
            artifacts=(
                self._artifact_ref(
                    kind="selfplay_agent_spec",
                    path=Path(str(expanded["agent_spec_path"])),
                    summary_path=Path(str(expanded["summary_path"])),
                ),
            ),
        )

    def _handle_phase10_selfplay_shard(self, task: TaskRow) -> TaskResult:
        from train.eval.distributed_selfplay import run_phase10_pre_verify_selfplay_shard

        payload = dict(task.payload)
        spec = load_phase10_lapv1_arena_campaign_spec(Path(str(payload["config_path"])))
        output_root = Path(str(payload["output_root"]))
        with self._thread_budget(self._distributed_task_threads):
            summary = run_phase10_pre_verify_selfplay_shard(
                spec=spec,
                repo_root=self._repo_root,
                agent_spec_path=Path(str(payload["agent_spec_path"])),
                agent_name=str(payload["agent_name"]),
                output_root=output_root,
                shard_index=int(payload["shard_index"]),
                starting_game_index=int(payload["starting_game_index"]),
                games=int(payload["games"]),
                max_plies=int(payload["max_plies"]),
            )
        summary_path = output_root / "shards" / f"selfplay_shard_{int(payload['shard_index']):04d}.summary.json"
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(
                self._artifact_ref(
                    kind="selfplay_session",
                    path=Path(str(summary["session_path"])),
                    summary_path=summary_path,
                ),
            ),
            metrics=dict(summary["aggregate"]),
        )

    def _handle_phase10_selfplay_finalize(self, task: TaskRow) -> TaskResult:
        from train.eval.distributed_selfplay import rebuild_phase10_pre_verify_selfplay_summary

        payload = dict(task.payload)
        output_root = Path(str(payload["output_root"]))
        summary = rebuild_phase10_pre_verify_selfplay_summary(
            output_root=output_root,
            agent_name=str(payload["agent_name"]),
            agent_spec_path=Path(str(payload["agent_spec_path"])),
        )
        self._db.update_model_record(
            int(payload["model_id"]),
            status="selfplay_completed",
        )
        summary_path = output_root / "summary.json"
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(
                self._artifact_ref(
                    kind="selfplay_summary",
                    path=summary_path,
                    summary_path=summary_path,
                ),
            ),
            metrics=dict(summary["aggregate"]),
        )

    def _handle_verify_lapv1(
        self,
        task: TaskRow,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
    ) -> TaskResult:
        payload = dict(task.payload)
        output_dir = Path(str(payload["output_dir"]))
        output_path = Path(str(payload["output_path"]))
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            str(self._repo_root / "python" / "scripts" / "eval_lapv1.py"),
            "--checkpoint",
            str(payload["checkpoint_path"]),
            "--dataset-path",
            str(payload["dataset_path"]),
            "--top-k",
            str(payload["top_k"]),
        ]
        self._run_command(
            command,
            stdout_handle=output_path.open("w", encoding="utf-8"),
            stderr_handle=stderr_handle,
            close_stdout=True,
            thread_budget=self._distributed_task_threads,
        )
        metrics = json.loads(output_path.read_text(encoding="utf-8"))
        summary_path = self._write_json(
            output_dir / "summary.json",
            {
                "checkpoint_path": str(payload["checkpoint_path"]),
                "dataset_path": str(payload["dataset_path"]),
                "metrics_path": str(output_path),
                "metrics": metrics,
            },
        )
        self._db.update_model_record(
            int(payload["model_id"]),
            verify_json_path=str(output_path),
            status="verified",
        )
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(
                self._artifact_ref(
                    kind="verify_metrics",
                    path=output_path,
                    summary_path=summary_path,
                ),
            ),
            metrics=metrics,
        )

    def _handle_phase10_arena_prepare(self, task: TaskRow) -> TaskResult:
        payload = dict(task.payload)
        expanded = self._controller.expand_phase10_arena(
            parent_task_id=task.id,
            campaign_id=task.campaign_id,
            model_id=int(payload["model_id"]),
            config_path=Path(str(payload["config_path"])),
        )
        return TaskResult(
            summary_path=str(expanded["summary_path"]),
            created_task_keys=tuple(sorted(dict(expanded["created_task_ids"]).keys())),
            metadata={
                "created_task_ids": dict(expanded["created_task_ids"]),
                "resolved_arena_spec_path": expanded["resolved_arena_spec_path"],
            },
            artifacts=(
                self._artifact_ref(
                    kind="arena_spec",
                    path=Path(str(expanded["resolved_arena_spec_path"])),
                    summary_path=Path(str(expanded["summary_path"])),
                ),
            ),
        )

    def _handle_arena_match(self, task: TaskRow) -> TaskResult:
        from train.eval.arena import load_selfplay_arena_spec, run_selfplay_arena_matchup

        payload = dict(task.payload)
        resolved_spec_path = Path(str(payload["resolved_arena_spec_path"]))
        spec = load_selfplay_arena_spec(resolved_spec_path)
        output_root = Path(str(payload["output_root"]))
        with self._thread_budget(self._distributed_task_threads):
            summary = run_selfplay_arena_matchup(
                spec=spec,
                matchup_index=int(payload["matchup_index"]),
                repo_root=self._repo_root,
                output_root=output_root,
            )
        summary_path = self._write_json(
            output_root
            / "matchup_tasks"
            / f"matchup_{int(payload['matchup_index']):04d}"
            / "summary.json",
            summary,
        )
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(
                self._artifact_ref(
                    kind="arena_session",
                    path=Path(str(summary["session_path"])),
                    summary_path=summary_path,
                ),
            ),
        )

    def _handle_arena_finalize(self, task: TaskRow) -> TaskResult:
        from train.eval.arena import load_selfplay_arena_spec, rebuild_selfplay_arena_summary
        from train.eval.matrix import build_selfplay_arena_matrix, write_selfplay_arena_matrix

        payload = dict(task.payload)
        spec = load_selfplay_arena_spec(Path(str(payload["resolved_arena_spec_path"])))
        output_root = Path(str(payload["output_root"]))
        summary = rebuild_selfplay_arena_summary(spec=spec, output_root=output_root)
        matrix = build_selfplay_arena_matrix(summary)
        matrix_path = Path(str(payload["matrix_path"]))
        write_selfplay_arena_matrix(matrix_path, matrix)
        self._db.update_model_record(
            int(payload["model_id"]),
            arena_summary_path=str(output_root / "summary.json"),
            status="evaluated",
        )
        return TaskResult(
            summary_path=str(output_root / "summary.json"),
            artifacts=(
                self._artifact_ref(
                    kind="arena_summary",
                    path=output_root / "summary.json",
                    summary_path=output_root / "summary.json",
                ),
                self._artifact_ref(
                    kind="arena_matrix",
                    path=matrix_path,
                    summary_path=output_root / "summary.json",
                ),
            ),
            metrics={"matchup_count": summary["aggregate"]["matchup_count"]},
        )

    def _handle_phase10_finalize(self, task: TaskRow) -> TaskResult:
        payload = dict(task.payload)
        summary_path = self._controller.write_phase10_summary(
            config_path=Path(str(payload["config_path"])),
            resolved_arena_spec_path=Path(str(payload["resolved_arena_spec_path"])),
        )
        self._db.update_model_record(int(payload["model_id"]), status="completed")
        self._db.update_campaign_record(
            task.campaign_id,
            status="succeeded",
            active_model_id=int(payload["model_id"]),
        )
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(self._artifact_ref(kind="campaign_summary", path=summary_path, summary_path=summary_path),),
        )

    def _handle_phase10_artifact_finalize(self, task: TaskRow) -> TaskResult:
        payload = dict(task.payload)
        summary_path = self._controller.write_phase10_artifact_summary(
            config_path=Path(str(payload["config_path"])),
        )
        self._db.update_campaign_record(task.campaign_id, status="succeeded")
        return TaskResult(
            summary_path=str(summary_path),
            artifacts=(
                self._artifact_ref(
                    kind="artifact_build_summary",
                    path=summary_path,
                    summary_path=summary_path,
                ),
            ),
        )

    def _before_execute(self, task: TaskRow) -> None:
        campaign_status = self._campaign_status_for_task_start(task.task_type)
        if campaign_status is not None:
            self._db.update_campaign_record(task.campaign_id, status=campaign_status)

    def _campaign_status_for_task_start(self, task_type: str) -> str | None:
        if task_type in {
            "label_pgn_corpus",
            "label_pgn_corpus_idle_slice",
            "phase5_raw_merge",
            "phase10_materialize",
            "phase10_workflow_prepare",
            "phase10_artifact_workflow_prepare",
            "phase10_workflow_chunk",
            "phase10_workflow_finalize",
            "phase10_selfplay_prepare",
            "phase10_selfplay_shard",
            "phase10_selfplay_finalize",
        }:
            return "running"
        if task_type == "train_lapv1":
            return "training"
        if task_type == "verify_lapv1":
            return "verifying"
        if task_type in {
            "phase10_arena_prepare",
            "arena_match",
            "arena_finalize",
            "phase10_finalize",
            "phase10_artifact_finalize",
        }:
            return "finalizing"
        return None

    def _after_success(self, *, task: TaskRow, result: TaskResult) -> None:
        if task.task_type in {"phase10_finalize", "phase10_artifact_finalize"}:
            return
        if task.task_type == "label_pgn_corpus":
            self._db.update_campaign_record(task.campaign_id, status="succeeded")
            return
        if task.task_type == "phase10_workflow_prepare":
            self._db.update_campaign_record(task.campaign_id, status="running")
        if task.task_type == "train_lapv1":
            self._db.update_campaign_record(task.campaign_id, status="training")
        if task.task_type == "phase10_selfplay_finalize":
            self._db.update_campaign_record(task.campaign_id, status="selfplay_completed")
        if task.task_type == "verify_lapv1":
            self._db.update_campaign_record(task.campaign_id, status="verifying")
        if task.task_type == "arena_finalize":
            self._db.update_campaign_record(task.campaign_id, status="finalizing")

    def _run_command(
        self,
        command: list[str],
        *,
        stdout_handle: IO[str],
        stderr_handle: IO[str],
        close_stdout: bool = False,
        thread_budget: int | None = None,
    ) -> None:
        env = os.environ.copy()
        python_root = str(self._repo_root / "python")
        current_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            python_root
            if not current_pythonpath
            else os.pathsep.join((python_root, current_pythonpath))
        )
        if thread_budget is not None:
            for variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ):
                env[variable] = str(thread_budget)
        process = subprocess.Popen(
            command,
            cwd=self._repo_root,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        return_code = process.wait()
        if close_stdout:
            stdout_handle.close()
        if return_code != 0:
            raise RuntimeError(f"command failed with exit code {return_code}: {' '.join(command)}")

    @contextmanager
    def _thread_budget(self, thread_budget: int | None):
        if thread_budget is None:
            yield
            return
        variables = (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
        previous_env = {variable: os.environ.get(variable) for variable in variables}
        for variable in variables:
            os.environ[variable] = str(thread_budget)

        torch_module: Any | None = None
        previous_threads: int | None = None
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ModuleNotFoundError:
            torch_module = None
        if torch_module is not None:
            previous_threads = int(torch_module.get_num_threads())
            torch_module.set_num_threads(thread_budget)
        try:
            yield
        finally:
            if torch_module is not None and previous_threads is not None:
                torch_module.set_num_threads(previous_threads)
            for variable, value in previous_env.items():
                if value is None:
                    os.environ.pop(variable, None)
                else:
                    os.environ[variable] = value

    def _attempt_log_paths(self, task: TaskRow) -> tuple[Path, Path]:
        task_root = self._log_root / self._descriptor.worker_id / f"task_{task.id:08d}"
        attempt_label = f"attempt_{task.attempt_count + 1:02d}"
        return task_root / f"{attempt_label}.stdout.log", task_root / f"{attempt_label}.stderr.log"

    def _artifact_ref(
        self,
        *,
        kind: str,
        path: Path,
        summary_path: Path,
    ) -> ArtifactRef:
        path_exists = path.exists()
        sha256 = _file_sha256(path) if path_exists and path.is_file() else None
        size_bytes = path.stat().st_size if path_exists and path.is_file() else None
        return ArtifactRef(
            kind=kind,
            path=str(path),
            summary_path=str(summary_path),
            sha256=sha256,
            size_bytes=size_bytes,
            metadata={"path_exists": path_exists},
        )

    def _write_json(self, path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def _load_train_metadata(self, config_path: Path) -> dict[str, Any]:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        output_dir = resolve_repo_path(self._repo_root, Path(str(payload["output_dir"])))
        bundle_dir = resolve_repo_path(self._repo_root, Path(str(payload["export"]["bundle_dir"])))
        checkpoint_path = bundle_dir / str(payload["export"]["checkpoint_name"])
        return {
            "summary_path": str(output_dir / "summary.json"),
            "checkpoint_path": str(checkpoint_path),
            "bundle_dir": str(bundle_dir),
        }


def build_default_worker_descriptor(
    *,
    capabilities: Sequence[str],
    scratch_root: Path,
    version: str,
    worker_id: str | None = None,
) -> WorkerDescriptor:
    """Create one default worker descriptor from local host metadata."""
    hostname = socket.gethostname()
    resolved_worker_id = worker_id or f"{hostname}-{os.getpid()}"
    return WorkerDescriptor(
        worker_id=resolved_worker_id,
        hostname=hostname,
        capabilities=tuple(capabilities),
        scratch_root=str(scratch_root),
        version=version,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
