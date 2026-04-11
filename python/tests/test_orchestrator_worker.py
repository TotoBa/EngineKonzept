from __future__ import annotations

import json
from pathlib import Path
import os

import pytest

from train.orchestrator.models import TaskRow, WorkerDescriptor
from train.orchestrator.worker import OrchestratorWorker


class _StubDB:
    pass


class _StubController:
    pass


def test_run_command_inherits_repo_pythonpath(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    worker = OrchestratorWorker(
        db=_StubDB(),
        controller=_StubController(),
        descriptor=WorkerDescriptor(
            worker_id="worker-test",
            hostname="localhost",
            capabilities=("materialize",),
            scratch_root=str(tmp_path / "scratch"),
            version="test",
        ),
        repo_root=repo_root,
        log_root=tmp_path / "logs",
    )
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        worker._run_command(
            ["python3", "-c", "import train; print('ok')"],
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
        )

    assert stdout_path.read_text(encoding="utf-8").strip() == "ok"
    assert stderr_path.read_text(encoding="utf-8") == ""


def test_run_command_applies_thread_budget_env(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    worker = OrchestratorWorker(
        db=_StubDB(),
        controller=_StubController(),
        descriptor=WorkerDescriptor(
            worker_id="worker-test",
            hostname="localhost",
            capabilities=("selfplay",),
            scratch_root=str(tmp_path / "scratch"),
            version="test",
        ),
        repo_root=repo_root,
        log_root=tmp_path / "logs",
    )
    stdout_path = tmp_path / "stdout-env.log"
    stderr_path = tmp_path / "stderr-env.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        worker._run_command(
            [
                "python3",
                "-c",
                "import os; print(os.environ['OMP_NUM_THREADS']); print(os.environ['MKL_NUM_THREADS'])",
            ],
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            thread_budget=1,
        )

    assert stdout_path.read_text(encoding="utf-8").splitlines() == ["1", "1"]
    assert stderr_path.read_text(encoding="utf-8") == ""


def test_thread_budget_context_restores_env_and_torch_threads(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    repo_root = Path(__file__).resolve().parents[2]
    worker = OrchestratorWorker(
        db=_StubDB(),
        controller=_StubController(),
        descriptor=WorkerDescriptor(
            worker_id="worker-test",
            hostname="localhost",
            capabilities=("selfplay",),
            scratch_root=str(tmp_path / "scratch"),
            version="test",
        ),
        repo_root=repo_root,
        log_root=tmp_path / "logs",
    )
    previous_threads = int(torch.get_num_threads())
    previous_omp = os.environ.get("OMP_NUM_THREADS")

    with worker._thread_budget(1):
        assert torch.get_num_threads() == 1
        assert os.environ["OMP_NUM_THREADS"] == "1"

    assert torch.get_num_threads() == previous_threads
    assert os.environ.get("OMP_NUM_THREADS") == previous_omp


def test_workflow_finalize_builds_hard_subsets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    worker = OrchestratorWorker(
        db=_StubDB(),
        controller=_StubController(),
        descriptor=WorkerDescriptor(
            worker_id="worker-test",
            hostname="localhost",
            capabilities=("aggregate",),
            scratch_root=str(tmp_path / "scratch"),
            version="test",
        ),
        repo_root=repo_root,
        log_root=tmp_path / "logs",
    )
    phase10_config = json.loads(
        (
            repo_root / "python" / "configs" / "phase10_lapv2_stage2_native_arena_all_sources_v1.json"
        ).read_text(encoding="utf-8")
    )
    workflow_root = tmp_path / "workflow"
    phase10_config["name"] = "workflow-finalize-test"
    phase10_config["output_root"] = str(tmp_path / "campaign")
    phase10_config["merged_raw_dir"] = str(tmp_path / "raw")
    phase10_config["train_dataset_dir"] = str(tmp_path / "dataset_train")
    phase10_config["verify_dataset_dir"] = str(tmp_path / "dataset_verify")
    phase10_config["workflow_output_root"] = str(workflow_root)
    config_path = tmp_path / "campaign.json"
    config_path.write_text(json.dumps(phase10_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task = TaskRow(
        id=1,
        campaign_id=1,
        model_id=None,
        task_type="phase10_workflow_finalize",
        capability="aggregate",
        priority=0,
        state="leased",
        payload={"config_path": str(config_path)},
        result=None,
        worker_id="worker-test",
        lease_until=None,
        attempt_count=1,
        max_attempts=1,
        depends_on_count=0,
        not_before=None,
        created_at=None,
        updated_at=None,
    )
    commands: list[list[str]] = []

    def fake_run_command(
        command: list[str],
        *,
        stdout_handle: object,
        stderr_handle: object,
        thread_budget: int | None = None,
    ) -> None:
        del stdout_handle, stderr_handle, thread_budget
        commands.append(list(command))
        if command[2].endswith("build_phase10_lapv1_workflow.py"):
            workflow_root.mkdir(parents=True, exist_ok=True)
            (workflow_root / "summary.json").write_text("{}\n", encoding="utf-8")
            (workflow_root / "all_unique_train_v1").mkdir(parents=True, exist_ok=True)
            (workflow_root / "all_unique_validation_v1").mkdir(parents=True, exist_ok=True)
            (workflow_root / "all_unique_train_v1" / "lapv1_train.jsonl").write_text(
                "{}\n",
                encoding="utf-8",
            )
            (
                workflow_root / "all_unique_validation_v1" / "lapv1_validation.jsonl"
            ).write_text("{}\n", encoding="utf-8")
            return
        if command[2].endswith("build_lapv1_hard_positions_dataset.py"):
            output_path = Path(command[command.index("--output-path") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("{}\n", encoding="utf-8")
            (output_path.parent / f"{output_path.stem}.summary.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            return
        raise AssertionError(command)

    worker._run_command = fake_run_command  # type: ignore[method-assign]
    stdout_path = tmp_path / "stdout-workflow.log"
    stderr_path = tmp_path / "stderr-workflow.log"
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        result = worker._handle_phase10_workflow_finalize(task, stdout_handle, stderr_handle)

    assert len(commands) == 3
    assert commands[0][2].endswith("build_phase10_lapv1_workflow.py")
    assert commands[1][2].endswith("build_lapv1_hard_positions_dataset.py")
    assert commands[2][2].endswith("build_lapv1_hard_positions_dataset.py")
    summary_payload = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    assert summary_payload["hard_train_path"] == str(
        workflow_root / "all_unique_train_hard_v1" / "lapv1_train_hard.jsonl"
    )
    assert summary_payload["hard_validation_path"] == str(
        workflow_root / "all_unique_validation_hard_v1" / "lapv1_validation_hard.jsonl"
    )


def test_handle_label_pgn_corpus_reads_progress_and_exports(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    worker = OrchestratorWorker(
        db=_StubDB(),
        controller=_StubController(),
        descriptor=WorkerDescriptor(
            worker_id="worker-test",
            hostname="localhost",
            capabilities=("label",),
            scratch_root=str(tmp_path / "scratch"),
            version="test",
        ),
        repo_root=repo_root,
        log_root=tmp_path / "logs",
    )
    work_dir = tmp_path / "label_work"
    stdout_path = tmp_path / "stdout-label.log"
    stderr_path = tmp_path / "stderr-label.log"
    task = TaskRow(
        id=1,
        campaign_id=1,
        model_id=None,
        task_type="label_pgn_corpus",
        capability="label",
        priority=0,
        state="leased",
        payload={
            "config_path": str(tmp_path / "label.json"),
            "pgn_root": "/srv/schach/PGN_DATA/pgn",
            "pgn_glob": "**/*.pgn",
            "engine_path": "/usr/games/stockfish18",
            "work_dir": str(work_dir),
            "target_train_records": 4,
            "target_verify_records": 2,
            "min_ply": 1,
            "max_ply": 8,
            "ply_stride": 1,
            "engine_nodes": 64,
            "hash_mb": 32,
            "threads": 1,
            "split_seed": "seed",
            "verify_divisor": 2,
            "progress_every": 1,
            "max_games": 4,
            "export_jsonl_on_complete": True,
        },
        result=None,
        worker_id="worker-test",
        lease_until=None,
        attempt_count=1,
        max_attempts=1,
        depends_on_count=0,
        not_before=None,
        created_at=None,
        updated_at=None,
    )

    def fake_run_command(*args: object, **kwargs: object) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "progress.json").write_text(
            """
{
  "completed": true,
  "counts": {"train": 4, "verify": 2},
  "games_seen": 3
}
""".strip()
            + "\n",
            encoding="utf-8",
        )
        (work_dir / "train_raw.jsonl").write_text("{}\n", encoding="utf-8")
        (work_dir / "verify_raw.jsonl").write_text("{}\n", encoding="utf-8")

    worker._run_command = fake_run_command  # type: ignore[method-assign]
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        result = worker._handle_label_pgn_corpus(task, stdout_handle, stderr_handle)

    assert Path(result.summary_path) == work_dir / "summary.json"
    assert result.metrics == {"train_records": 4, "verify_records": 2, "games_seen": 3}
    assert len(result.artifacts) == 3
