from __future__ import annotations

from pathlib import Path

from train.eval.phase10_campaign import Phase10Lapv1ArenaCampaignSpec
from train.orchestrator.controller import (
    build_phase10_arena_tasks,
    build_phase10_bootstrap_tasks,
    build_phase10_workflow_tasks,
)


class _StubArenaSpec:
    def __init__(self, matchup_count: int) -> None:
        self._matchup_count = matchup_count

    def expanded_matchups(self) -> list[object]:
        return [object() for _index in range(self._matchup_count)]


def _phase10_spec(tmp_path: Path) -> Phase10Lapv1ArenaCampaignSpec:
    return Phase10Lapv1ArenaCampaignSpec(
        name="phase10_distributed_test",
        output_root=str(tmp_path / "campaign"),
        merged_raw_dir=str(tmp_path / "raw"),
        train_dataset_dir=str(tmp_path / "datasets" / "train"),
        verify_dataset_dir=str(tmp_path / "datasets" / "verify"),
        phase5_source_name="test_source",
        phase5_seed="seed",
        workflow_output_root=str(tmp_path / "workflow"),
        proposer_checkpoint="models/proposer/checkpoint.pt",
        lapv1_config_path="python/configs/test_lapv1.json",
        lapv1_agent_spec_path="python/configs/test_agent.json",
        lapv1_verify_output_path=str(
            tmp_path / "workflow" / "all_unique_verify_v1" / "lapv1_test.jsonl"
        ),
        reference_arena_summary_path=str(tmp_path / "refs" / "arena_summary.json"),
        initial_fen_suite_path="artifacts/fens.json",
        teacher_nodes=64,
        workflow_chunk_size=4,
    )


def test_build_phase10_bootstrap_tasks_has_materialize_then_prepare(tmp_path: Path) -> None:
    spec = _phase10_spec(tmp_path)

    tasks = build_phase10_bootstrap_tasks(
        spec_path=tmp_path / "phase10.json",
        spec=spec,
        model_id=11,
    )

    assert [task.key for task in tasks] == ["materialize", "workflow_prepare"]
    assert tasks[1].depends_on == ("materialize",)
    assert tasks[0].payload["task_kind"] == "phase10_materialize"


def test_build_phase10_workflow_tasks_expands_chunks_and_downstream_steps(tmp_path: Path) -> None:
    spec = _phase10_spec(tmp_path)
    config_path = tmp_path / "python" / "configs" / "test_lapv1.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        """
{
  "output_dir": "models/lapv1/test_run",
  "evaluation": {"top_k": 5},
  "export": {
    "bundle_dir": "models/lapv1/test_run/bundle",
    "checkpoint_name": "checkpoint.pt"
  }
}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    spec = Phase10Lapv1ArenaCampaignSpec(
        **{**spec.__dict__, "lapv1_config_path": str(config_path.relative_to(tmp_path))}
    )

    tasks = build_phase10_workflow_tasks(
        spec_path=tmp_path / "phase10.json",
        spec=spec,
        model_id=13,
        train_summary={"split_counts": {"train": 9, "validation": 5}},
        verify_summary={"split_counts": {"test": 3}},
        repo_root=tmp_path,
    )

    task_keys = [task.key for task in tasks]
    assert "workflow_train_chunk_0001" in task_keys
    assert "workflow_train_chunk_0003" in task_keys
    assert "workflow_validation_chunk_0002" in task_keys
    assert "workflow_verify_chunk_0001" in task_keys
    assert "workflow_finalize" in task_keys
    assert "train" in task_keys
    assert "verify" in task_keys
    assert "arena_prepare" in task_keys
    workflow_finalize = next(task for task in tasks if task.key == "workflow_finalize")
    assert "workflow_train_chunk_0001" in workflow_finalize.depends_on
    train_task = next(task for task in tasks if task.key == "train")
    assert train_task.depends_on == ("workflow_finalize",)
    verify_task = next(task for task in tasks if task.key == "verify")
    assert verify_task.payload["task_kind"] == "verify_lapv1"
    assert verify_task.payload["top_k"] == 5


def test_build_phase10_arena_tasks_expands_matchups_and_finalizers(tmp_path: Path) -> None:
    spec = _phase10_spec(tmp_path)
    resolved_arena_spec = _StubArenaSpec(matchup_count=6)

    tasks = build_phase10_arena_tasks(
        spec_path=tmp_path / "phase10.json",
        spec=spec,
        resolved_arena_spec=resolved_arena_spec,
        resolved_arena_spec_path=tmp_path / "arena_spec.json",
        model_id=19,
        repo_root=tmp_path,
    )

    match_tasks = [task for task in tasks if task.task_type == "arena_match"]
    assert len(match_tasks) == 6
    arena_finalize = next(task for task in tasks if task.key == "arena_finalize")
    assert len(arena_finalize.depends_on) == 6
    phase10_finalize = next(task for task in tasks if task.key == "phase10_finalize")
    assert phase10_finalize.depends_on == ("arena_finalize",)
