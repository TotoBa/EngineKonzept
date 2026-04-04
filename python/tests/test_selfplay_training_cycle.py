from __future__ import annotations

import json
from pathlib import Path

from train.config import PlannerTrainConfig
from train.eval.agent_spec import SelfplayAgentSpec, load_selfplay_agent_spec, write_selfplay_agent_spec
from train.eval.arena import SelfplayArenaSpec
from train.eval.selfplay_training_cycle import (
    SelfplayTeacherRetrainAgentSpec,
    SelfplayTeacherRetrainCycleSpec,
    _build_matchup_batches,
    _materialize_cycle_planner_config,
    run_selfplay_teacher_retrain_cycle,
)


def test_build_matchup_batches_pairs_reciprocals() -> None:
    spec = SelfplayArenaSpec(
        name="round_robin",
        agent_specs={"a": "a.json", "b": "b.json"},
        schedule_mode="round_robin",
        default_games=2,
        default_max_plies=12,
        default_initial_fens=["startpos"],
    )

    batches = _build_matchup_batches(spec, mode="reciprocal_pair")

    assert len(batches) == 1
    assert [(matchup.white_agent, matchup.black_agent) for matchup in batches[0]] == [
        ("a", "b"),
        ("b", "a"),
    ]


def test_materialize_cycle_planner_config_uses_current_checkpoint(tmp_path: Path) -> None:
    base_config = PlannerTrainConfig.from_dict(
        {
            "seed": 7,
            "output_dir": str(tmp_path / "base_run"),
            "initial_checkpoint": None,
            "data": {
                "train_path": str(tmp_path / "base_train.jsonl"),
                "validation_path": str(tmp_path / "base_validation.jsonl"),
                "additional_train_paths": [str(tmp_path / "extra_train.jsonl")],
            },
            "model": {"architecture": "set_v2", "hidden_dim": 64, "hidden_layers": 2, "action_embedding_dim": 32},
            "optimization": {
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 0.001,
                "teacher_policy_loss_weight": 1.0,
            },
            "evaluation": {"top_k": 3},
            "runtime": {"torch_threads": 1, "dataloader_workers": 0},
            "export": {"bundle_dir": str(tmp_path / "base_bundle")},
        }
    )
    agent_spec = SelfplayAgentSpec(
        name="planner_a",
        proposer_checkpoint=str(tmp_path / "proposer.pt"),
        planner_checkpoint=str(tmp_path / "current_checkpoint.pt"),
        opponent_mode="none",
        root_top_k=4,
    )
    retrain_spec = SelfplayTeacherRetrainAgentSpec(
        agent_name="planner_a",
        planner_train_config_path=str(tmp_path / "planner.json"),
        epochs_override=1,
        learning_rate_override=0.0002,
        batch_size_override=8,
    )

    resolved = _materialize_cycle_planner_config(
        base_config=base_config,
        current_agent_spec=agent_spec,
        train_path=tmp_path / "teacher" / "planner_head_train.jsonl",
        output_root=tmp_path / "cycle_batch",
        agent_name="planner_a",
        retrain_spec=retrain_spec,
    )

    assert resolved.initial_checkpoint == str(tmp_path / "current_checkpoint.pt")
    assert resolved.data.train_path == str(tmp_path / "teacher" / "planner_head_train.jsonl")
    assert resolved.data.additional_train_paths == ()
    assert resolved.optimization.epochs == 1
    assert resolved.optimization.learning_rate == 0.0002
    assert resolved.optimization.batch_size == 8


def test_run_selfplay_teacher_retrain_cycle_updates_active_agent_spec(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path
    proposer_checkpoint = repo_root / "models" / "proposer.pt"
    planner_checkpoint = repo_root / "models" / "planner.pt"
    proposer_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    proposer_checkpoint.write_text("stub", encoding="utf-8")
    planner_checkpoint.write_text("stub", encoding="utf-8")

    planner_agent_path = repo_root / "planner_agent.json"
    external_agent_path = repo_root / "vice_agent.json"
    write_selfplay_agent_spec(
        planner_agent_path,
        SelfplayAgentSpec(
            name="planner_a",
            proposer_checkpoint=str(proposer_checkpoint),
            planner_checkpoint=str(planner_checkpoint),
            opponent_mode="none",
            root_top_k=4,
        ),
    )
    write_selfplay_agent_spec(
        external_agent_path,
        SelfplayAgentSpec(
            name="vice_v1",
            agent_kind="uci_engine",
            external_engine_path="/usr/games/vice",
            external_engine_depth=3,
        ),
    )

    arena_spec = SelfplayArenaSpec.from_dict(
        {
            "name": "cycle_arena",
            "agent_specs": {
                "planner_a": str(planner_agent_path),
                "vice_v1": str(external_agent_path),
            },
            "schedule_mode": "round_robin",
            "default_games": 1,
            "default_max_plies": 8,
            "default_initial_fens": ["startpos"],
        }
    )
    arena_spec_path = repo_root / "arena_spec.json"
    arena_spec_path.write_text(json.dumps(arena_spec.to_dict(), indent=2) + "\n", encoding="utf-8")

    base_config_path = repo_root / "planner_base_config.json"
    base_config_path.write_text(
        json.dumps(
            {
                "seed": 7,
                "output_dir": str(repo_root / "planner_run"),
                "initial_checkpoint": None,
                "data": {
                    "train_path": str(repo_root / "base_train.jsonl"),
                    "validation_path": str(repo_root / "base_validation.jsonl"),
                },
                "model": {
                    "architecture": "set_v2",
                    "hidden_dim": 64,
                    "hidden_layers": 2,
                    "action_embedding_dim": 32,
                },
                "optimization": {
                    "epochs": 1,
                    "batch_size": 8,
                    "learning_rate": 0.001,
                    "teacher_policy_loss_weight": 1.0,
                },
                "evaluation": {"top_k": 3},
                "runtime": {"torch_threads": 1, "dataloader_workers": 0},
                "export": {"bundle_dir": str(repo_root / "bundle")},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cycle_spec = SelfplayTeacherRetrainCycleSpec(
        name="cycle_probe",
        arena_spec_path=str(arena_spec_path),
        output_root=str(repo_root / "cycle_output"),
        retrain_agents=(
            SelfplayTeacherRetrainAgentSpec(
                agent_name="planner_a",
                planner_train_config_path=str(base_config_path),
            ),
        ),
    )

    def fake_arena_runner(*, spec, repo_root, output_root):
        output_root.mkdir(parents=True, exist_ok=True)
        summary_path = output_root / "summary.json"
        summary_payload = {
            "aggregate": {"game_count": 1, "matchup_count": 1, "mean_games_per_matchup": 1.0},
            "arena_name": spec.name,
            "arena_spec_version": 1,
            "matchups": [],
            "standings": {},
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
        return summary_payload

    def fake_materialize_teacher_training_sets(*, arena_summary_path, active_agent_spec_paths, batch_root, spec, repo_root):
        agent_root = batch_root / "planner_a"
        agent_root.mkdir(parents=True, exist_ok=True)
        planner_head_path = agent_root / "planner_head_train.jsonl"
        planner_head_path.write_text("{}", encoding="utf-8")
        summary_payload = {
            "agent_spec_path": str(active_agent_spec_paths["planner_a"]),
            "planner_head_path": str(planner_head_path),
            "planner_head_example_count": 3,
            "review_path": str(agent_root / "selfplay_teacher_review_train.jsonl"),
            "review_summary": {"example_count": 3, "mistake_count": 3},
        }
        (agent_root / "summary.json").write_text(
            json.dumps(summary_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"planner_a": summary_payload}

    class _FakeRun:
        def to_dict(self):
            return {"best_epoch": 1}

    def fake_planner_trainer(config, *, repo_root):
        checkpoint_path = Path(config.export.bundle_dir) / config.export.checkpoint_name
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("trained", encoding="utf-8")
        return _FakeRun()

    monkeypatch.setattr(
        "train.eval.selfplay_training_cycle._materialize_teacher_training_sets",
        fake_materialize_teacher_training_sets,
    )

    summary = run_selfplay_teacher_retrain_cycle(
        spec=cycle_spec,
        repo_root=repo_root,
        arena_runner=fake_arena_runner,
        planner_trainer=fake_planner_trainer,
    )

    final_spec_path = Path(summary["final_agent_spec_paths"]["planner_a"])
    final_spec = load_selfplay_agent_spec(final_spec_path)
    assert final_spec.planner_checkpoint is not None
    assert final_spec.planner_checkpoint.endswith("/planner_models/planner_a/checkpoint.pt")
    assert final_spec.metadata["selfplay_teacher_cycle"] == "cycle_probe"
