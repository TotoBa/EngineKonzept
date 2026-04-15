from __future__ import annotations

from pathlib import Path

from train.eval.phase10_campaign import Phase10Lapv1ArenaCampaignSpec
from train.orchestrator.controller import (
    build_label_pgn_corpus_tasks,
    build_label_pgn_corpus_idle_tasks,
    build_phase10_idle_artifact_tasks,
    build_phase10_idle_artifact_workflow_tasks,
    build_phase10_arena_tasks,
    build_phase10_bootstrap_tasks,
    build_phase10_reuse_existing_artifact_tasks,
    build_phase10_selfplay_tasks,
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


def test_build_phase10_reuse_existing_artifact_tasks_starts_with_train(tmp_path: Path) -> None:
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
        **{
            **_phase10_spec(tmp_path).__dict__,
            "lapv1_config_path": str(config_path.relative_to(tmp_path)),
            "reuse_existing_artifacts": True,
            "pre_verify_selfplay_games": 12,
            "pre_verify_selfplay_games_per_task": 4,
        }
    )

    tasks = build_phase10_reuse_existing_artifact_tasks(
        spec_path=tmp_path / "phase10.json",
        spec=spec,
        model_id=11,
        repo_root=tmp_path,
    )

    assert [task.key for task in tasks] == ["train", "selfplay_prepare"]
    assert tasks[0].depends_on == ()
    assert tasks[1].depends_on == ("train",)


def test_build_phase10_reuse_existing_artifact_tasks_can_seed_checkpoint_and_skip_train(
    tmp_path: Path,
) -> None:
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
    warm_start_checkpoint = tmp_path / "seed" / "checkpoint.pt"
    warm_start_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    warm_start_checkpoint.write_bytes(b"seed")
    spec = Phase10Lapv1ArenaCampaignSpec(
        **{
            **_phase10_spec(tmp_path).__dict__,
            "lapv1_config_path": str(config_path.relative_to(tmp_path)),
            "reuse_existing_artifacts": True,
            "skip_training": True,
            "warm_start_source_checkpoint": str(warm_start_checkpoint),
            "pre_verify_selfplay_games": 12,
            "pre_verify_selfplay_games_per_task": 4,
        }
    )

    tasks = build_phase10_reuse_existing_artifact_tasks(
        spec_path=tmp_path / "phase10.json",
        spec=spec,
        model_id=11,
        repo_root=tmp_path,
    )

    assert [task.key for task in tasks] == ["seed_checkpoint", "selfplay_prepare"]
    assert tasks[0].task_type == "phase10_seed_checkpoint"
    assert tasks[0].payload["source_checkpoint_path"] == str(warm_start_checkpoint)
    assert tasks[1].depends_on == ("seed_checkpoint",)


def test_build_label_pgn_corpus_tasks_has_single_label_task(tmp_path: Path) -> None:
    tasks = build_label_pgn_corpus_tasks(
        config_path=tmp_path / "label.json",
        config_payload={
            "name": "label_smoke",
            "pgn_root": "/srv/schach/PGN_DATA/pgn",
            "work_dir": str(tmp_path / "label_work"),
            "target_train_records": 32,
            "target_verify_records": 8,
            "engine_nodes": 64,
        },
    )

    assert [task.key for task in tasks] == ["label"]
    assert tasks[0].task_type == "label_pgn_corpus"
    assert tasks[0].capability == "label"
    assert tasks[0].payload["work_dir"] == str(tmp_path / "label_work")


def test_build_label_pgn_corpus_idle_tasks_splits_targets_across_shards(tmp_path: Path) -> None:
    tasks = build_label_pgn_corpus_idle_tasks(
        config_path=tmp_path / "idle_label.json",
        config_payload={
            "name": "label_idle",
            "pgn_root": "/srv/schach/PGN_DATA/pgn",
            "work_root": str(tmp_path / "label_work"),
            "target_train_records": 10,
            "target_verify_records": 4,
            "shard_count": 3,
            "run_max_games": 25,
        },
    )

    assert [task.task_type for task in tasks] == ["label_pgn_corpus_idle_slice"] * 3
    assert [task.payload["target_train_records"] for task in tasks] == [4, 3, 3]
    assert [task.payload["target_verify_records"] for task in tasks] == [2, 1, 1]
    assert tasks[0].payload["run_max_games"] == 25
    assert tasks[0].capability == "label_idle"


def test_build_phase10_idle_artifact_tasks_builds_merge_materialize_prepare(tmp_path: Path) -> None:
    spec = _phase10_spec(tmp_path)

    tasks = build_phase10_idle_artifact_tasks(
        config_path=tmp_path / "idle_phase10.json",
        config_payload={
            "name": "idle_phase10",
            "phase10_config_path": str(tmp_path / "phase10.json"),
            "pgn_root": "/srv/schach/PGN_DATA/pgn",
            "work_root": str(tmp_path / "idle_phase10"),
            "target_train_records": 12,
            "target_verify_records": 3,
            "shard_count": 2,
        },
        phase10_config_path=tmp_path / "phase10.json",
        phase10_spec=spec,
    )

    assert [task.key for task in tasks[:4]] == [
        "idle_label_shard_01",
        "idle_label_shard_02",
        "merge_raw",
        "materialize",
    ]
    assert tasks[2].task_type == "phase5_raw_merge"
    assert tasks[2].depends_on == ("idle_label_shard_01", "idle_label_shard_02")
    assert tasks[3].capability == "materialize_idle"
    assert tasks[4].task_type == "phase10_artifact_workflow_prepare"


def test_build_phase10_idle_artifact_workflow_tasks_excludes_train_and_verify(tmp_path: Path) -> None:
    spec = _phase10_spec(tmp_path)

    tasks = build_phase10_idle_artifact_workflow_tasks(
        config_path=tmp_path / "idle_phase10.json",
        config_payload={
            "name": "idle_phase10",
            "phase10_config_path": str(tmp_path / "phase10.json"),
            "pgn_root": "/srv/schach/PGN_DATA/pgn",
            "work_root": str(tmp_path / "idle_phase10"),
            "target_train_records": 12,
            "target_verify_records": 3,
        },
        phase10_config_path=tmp_path / "phase10.json",
        spec=spec,
        train_summary={"split_counts": {"train": 9, "validation": 5}},
        verify_summary={"split_counts": {"test": 3}},
    )

    task_keys = [task.key for task in tasks]
    assert "workflow_train_chunk_0001" in task_keys
    assert "workflow_finalize" in task_keys
    assert "artifact_finalize" in task_keys
    assert "train" not in task_keys
    assert "verify" not in task_keys


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


def test_build_phase10_workflow_tasks_inserts_selfplay_prepare_when_enabled(tmp_path: Path) -> None:
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
        **{
            **spec.__dict__,
            "lapv1_config_path": str(config_path.relative_to(tmp_path)),
            "pre_verify_selfplay_games": 12,
            "pre_verify_selfplay_games_per_task": 4,
        }
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
    assert "selfplay_prepare" in task_keys
    assert "verify" not in task_keys
    selfplay_prepare = next(task for task in tasks if task.key == "selfplay_prepare")
    assert selfplay_prepare.depends_on == ("train",)


def test_build_phase10_selfplay_tasks_expands_shards_and_verify(tmp_path: Path) -> None:
    spec = Phase10Lapv1ArenaCampaignSpec(
        **{
            **_phase10_spec(tmp_path).__dict__,
            "lapv1_config_path": "python/configs/test_lapv1.json",
            "pre_verify_selfplay_games": 10,
            "pre_verify_selfplay_games_per_task": 4,
            "pre_verify_selfplay_max_plies": 24,
        }
    )
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

    tasks = build_phase10_selfplay_tasks(
        spec_path=tmp_path / "phase10.json",
        spec=spec,
        model_id=21,
        repo_root=tmp_path,
        agent_name="lapv2_primary",
        agent_spec_path=tmp_path / "lapv2_primary.json",
    )

    shard_tasks = [task for task in tasks if task.task_type == "phase10_selfplay_shard"]
    assert len(shard_tasks) == 3
    assert shard_tasks[0].payload["starting_game_index"] == 0
    assert shard_tasks[1].payload["starting_game_index"] == 4
    assert shard_tasks[2].payload["games"] == 2
    finalize_task = next(task for task in tasks if task.key == "selfplay_finalize")
    assert finalize_task.depends_on == (
        "selfplay_shard_0001",
        "selfplay_shard_0002",
        "selfplay_shard_0003",
    )
    verify_task = next(task for task in tasks if task.key == "verify")
    assert verify_task.depends_on == ("selfplay_finalize",)
    arena_prepare = next(task for task in tasks if task.key == "arena_prepare")
    assert arena_prepare.depends_on == ("verify",)


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
