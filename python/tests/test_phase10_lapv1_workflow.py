from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_phase10_lapv1_workflow.py"
)
_SPEC = importlib.util.spec_from_file_location("build_phase10_lapv1_workflow", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

_build_expected_chunk_records = _MODULE._build_expected_chunk_records
_normalize_chunk_range = _MODULE._normalize_chunk_range
_missing_workflow_chunk_artifacts = _MODULE._missing_workflow_chunk_artifacts
_process_chunk_job = _MODULE._process_chunk_job
_wait_for_workflow_chunk_artifacts_complete = _MODULE._wait_for_workflow_chunk_artifacts_complete
_workflow_chunk_artifacts_complete = _MODULE._workflow_chunk_artifacts_complete


def test_normalize_chunk_range_defaults_to_full_span() -> None:
    assert _normalize_chunk_range(total_chunks=7, chunk_start=None, chunk_end=None) == (1, 7)


def test_build_expected_chunk_records_can_slice_middle_range(tmp_path: Path) -> None:
    records = _build_expected_chunk_records(
        output_dir=tmp_path / "workflow",
        canonical_split="train",
        max_examples=10,
        chunk_size=2,
        chunk_start=2,
        chunk_end=4,
    )

    assert [record["index"] for record in records] == [2, 3, 4]
    assert [record["start_index"] for record in records] == [2, 4, 6]
    assert all(record["total_chunks"] == 5 for record in records)


def test_workflow_chunk_artifacts_complete_requires_all_workflow_outputs(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunk"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (chunk_dir / "workflow.summary.json").write_text("{}\n", encoding="utf-8")
    (chunk_dir / "search_teacher_train.jsonl").write_text("{}\n", encoding="utf-8")
    (chunk_dir / "search_traces_train.jsonl").write_text("{}\n", encoding="utf-8")
    (chunk_dir / "search_disagreements_train.jsonl").write_text("{}\n", encoding="utf-8")

    assert not _workflow_chunk_artifacts_complete(chunk_dir=chunk_dir, canonical_split="train")

    (chunk_dir / "search_curriculum_train.jsonl").write_text("{}\n", encoding="utf-8")
    assert _workflow_chunk_artifacts_complete(chunk_dir=chunk_dir, canonical_split="train")


def test_wait_for_workflow_chunk_artifacts_complete_reports_missing_paths(
    tmp_path: Path,
) -> None:
    chunk_dir = tmp_path / "chunk"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (chunk_dir / "workflow.summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="search_teacher_train.jsonl"):
        _wait_for_workflow_chunk_artifacts_complete(
            chunk_dir=chunk_dir,
            canonical_split="train",
            timeout_seconds=0.0,
            poll_interval_seconds=0.0,
        )

    missing = _missing_workflow_chunk_artifacts(chunk_dir=chunk_dir, canonical_split="train")
    assert any(path.name == "search_teacher_train.jsonl" for path in missing)


def test_process_chunk_job_rebuilds_workflow_when_resume_state_is_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_dir = tmp_path / "chunk"
    chunk_dir.mkdir(parents=True)
    (chunk_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    (chunk_dir / "workflow.summary.json").write_text("{}\n", encoding="utf-8")

    calls: list[str] = []

    def fake_run_workflow_build(**kwargs: object) -> None:
        del kwargs
        calls.append("workflow")
        (chunk_dir / "summary.json").write_text("{}\n", encoding="utf-8")
        (chunk_dir / "workflow.summary.json").write_text("{}\n", encoding="utf-8")
        (chunk_dir / "search_teacher_train.jsonl").write_text("{}\n", encoding="utf-8")
        (chunk_dir / "search_traces_train.jsonl").write_text("{}\n", encoding="utf-8")
        (chunk_dir / "search_disagreements_train.jsonl").write_text("{}\n", encoding="utf-8")
        (chunk_dir / "search_curriculum_train.jsonl").write_text("{}\n", encoding="utf-8")

    def fake_run_planner_head_build(**kwargs: object) -> None:
        calls.append("planner")
        output_path = Path(str(kwargs["output_path"]))
        output_path.write_text("{}\n", encoding="utf-8")
        (output_path.parent / "planner_head.summary.json").write_text("{}\n", encoding="utf-8")

    def fake_run_lapv1_training_artifact_build(**kwargs: object) -> None:
        calls.append("lapv1")
        output_path = Path(str(kwargs["output_path"]))
        output_path.write_text("{}\n", encoding="utf-8")
        (output_path.parent / "lapv1.summary.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(_MODULE, "_run_workflow_build", fake_run_workflow_build)
    monkeypatch.setattr(_MODULE, "_run_planner_head_build", fake_run_planner_head_build)
    monkeypatch.setattr(
        _MODULE,
        "_run_lapv1_training_artifact_build",
        fake_run_lapv1_training_artifact_build,
    )

    record = _process_chunk_job(
        job={
            "chunk_index": 1,
            "total_chunks": 1,
            "start_index": 0,
            "example_count": 2,
            "chunk_dir": str(chunk_dir),
        },
        dataset_dir=tmp_path / "dataset",
        split="train",
        canonical_split="train",
        checkpoint_path=tmp_path / "checkpoint.pt",
        teacher_engine=tmp_path / "teacher",
        nodes=64,
        depth=None,
        multipv=4,
        policy_temperature_cp=100.0,
        top_k=4,
        root_top_k=4,
        log_every=10,
        skip_existing=True,
    )

    assert calls == ["workflow", "planner", "lapv1"]
    assert record["planner_head_path"].endswith("planner_head_train.jsonl")
    assert record["lapv1_path"].endswith("lapv1_train.jsonl")


def test_process_chunk_job_stops_before_planner_when_workflow_outputs_stay_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_dir = tmp_path / "chunk"
    chunk_dir.mkdir(parents=True)

    calls: list[str] = []

    def fake_run_workflow_build(**kwargs: object) -> None:
        del kwargs
        calls.append("workflow")
        (chunk_dir / "summary.json").write_text("{}\n", encoding="utf-8")
        (chunk_dir / "workflow.summary.json").write_text("{}\n", encoding="utf-8")

    def fake_run_planner_head_build(**kwargs: object) -> None:
        del kwargs
        calls.append("planner")

    def fake_run_lapv1_training_artifact_build(**kwargs: object) -> None:
        del kwargs
        calls.append("lapv1")

    monkeypatch.setattr(_MODULE, "_run_workflow_build", fake_run_workflow_build)
    monkeypatch.setattr(_MODULE, "_run_planner_head_build", fake_run_planner_head_build)
    monkeypatch.setattr(
        _MODULE,
        "_run_lapv1_training_artifact_build",
        fake_run_lapv1_training_artifact_build,
    )

    with pytest.raises(FileNotFoundError, match="search_teacher_train.jsonl"):
        _process_chunk_job(
            job={
                "chunk_index": 1,
                "total_chunks": 1,
                "start_index": 0,
                "example_count": 2,
                "chunk_dir": str(chunk_dir),
            },
            dataset_dir=tmp_path / "dataset",
            split="train",
            canonical_split="train",
            checkpoint_path=tmp_path / "checkpoint.pt",
            teacher_engine=tmp_path / "teacher",
            nodes=64,
            depth=None,
            multipv=4,
            policy_temperature_cp=100.0,
            top_k=4,
            root_top_k=4,
            log_every=10,
            skip_existing=False,
        )

    assert calls == ["workflow"]
