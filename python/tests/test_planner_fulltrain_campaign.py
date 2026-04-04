from __future__ import annotations

from pathlib import Path

from train.config import PlannerTrainConfig
from train.eval.agent_spec import SelfplayAgentSpec
from train.eval.arena import SelfplayArenaSpec
from train.eval.fulltrain_campaign import (
    PlannerFulltrainArenaCampaignSpec,
    PlannerFulltrainRunSpec,
    load_planner_fulltrain_arena_campaign_spec,
    materialize_fulltrain_arena_spec,
    materialize_fulltrain_planner_config,
    resolve_trained_agent_spec,
    write_planner_fulltrain_arena_campaign_spec,
)


def test_planner_fulltrain_campaign_spec_round_trip(tmp_path: Path) -> None:
    spec = PlannerFulltrainArenaCampaignSpec(
        name="campaign",
        output_root="/srv/campaign",
        workflow_summary="/srv/workflow/summary.json",
        training_tiers=("pgn_10k", "merged_unique_122k", "unique_pi_400k"),
        verify_tiers=("pgn_10k", "merged_unique_122k", "unique_pi_400k"),
        training_epochs=12,
        arena_template_spec_path="python/configs/arena.json",
        initial_fen_suite_path="artifacts/phase9/openings.json",
        arena_default_games=1,
        arena_parallel_workers=6,
        static_agent_specs={"symbolic_root_v1": "python/configs/phase9_agent_symbolic_root_v1.json"},
        arena_agent_order=("symbolic_root_v1", "planner_set_v2_expanded_v1"),
        baseline_metrics={"planner_active_expanded_v2": "artifacts/phase8/active.json"},
        reference_run_name="planner_active_expanded_v2",
        planner_runs=(
            PlannerFulltrainRunSpec(
                name="planner_set_v2_expanded_v1",
                base_config_path="python/configs/planner_set_v2.json",
                agent_template_spec_path="python/configs/phase9_agent_planner_set_v2_expanded_v1.json",
            ),
        ),
    )
    path = tmp_path / "campaign.json"
    write_planner_fulltrain_arena_campaign_spec(path, spec)
    loaded = load_planner_fulltrain_arena_campaign_spec(path)
    assert loaded.name == spec.name
    assert loaded.training_epochs == 12
    assert loaded.arena_parallel_workers == 6
    assert loaded.planner_runs[0].agent_template_spec_path.endswith(
        "phase9_agent_planner_set_v2_expanded_v1.json"
    )


def test_materialize_fulltrain_planner_config_repoints_workflow_and_epochs() -> None:
    base_config = PlannerTrainConfig.from_dict(
        {
            "seed": 1,
            "output_dir": "/srv/base/output",
            "data": {
                "train_path": "/srv/base/train.jsonl",
                "validation_path": "/srv/base/validation.jsonl",
            },
            "model": {
                "architecture": "set_v2",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 32,
                "dropout": 0.0,
            },
            "optimization": {
                "epochs": 4,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "teacher_policy_loss_weight": 1.0,
            },
            "evaluation": {"top_k": 3},
            "runtime": {"torch_threads": 0, "dataloader_workers": 0},
            "export": {"bundle_dir": "/srv/base/export", "checkpoint_name": "checkpoint.pt"},
        }
    )
    workflow_summary = {
        "tiers": {
            "pgn_10k": {
                "train": {"planner_head_path": "/srv/workflow/pgn_10k_train.jsonl"},
                "validation": {"planner_head_path": "/srv/workflow/pgn_10k_validation.jsonl"},
                "verify": {"planner_head_path": "/srv/workflow/pgn_10k_test.jsonl"},
            },
            "merged_unique_122k": {
                "train": {"planner_head_path": "/srv/workflow/122k_train.jsonl"},
                "validation": {"planner_head_path": "/srv/workflow/122k_validation.jsonl"},
                "verify": {"planner_head_path": "/srv/workflow/122k_test.jsonl"},
            },
            "unique_pi_400k": {
                "train": {"planner_head_path": "/srv/workflow/400k_train.jsonl"},
                "validation": {"planner_head_path": "/srv/workflow/400k_validation.jsonl"},
                "verify": {"planner_head_path": "/srv/workflow/400k_test.jsonl"},
            },
        }
    }
    payload = materialize_fulltrain_planner_config(
        base_config=base_config,
        workflow_summary=workflow_summary,
        training_tiers=("pgn_10k", "merged_unique_122k", "unique_pi_400k"),
        output_root=Path("/tmp/phase9_campaign"),
        run_name="planner_set_v2_expanded_v1",
        training_epochs=12,
    )
    assert payload["data"]["train_path"] == "/srv/workflow/pgn_10k_train.jsonl"
    assert payload["data"]["additional_train_paths"] == [
        "/srv/workflow/122k_train.jsonl",
        "/srv/workflow/400k_train.jsonl",
    ]
    assert payload["data"]["validation_path"] == "/srv/workflow/pgn_10k_validation.jsonl"
    assert payload["optimization"]["epochs"] == 12
    assert payload["output_dir"] == "/tmp/phase9_campaign/planner_runs/planner_set_v2_expanded_v1"
    assert payload["export"]["bundle_dir"] == "/tmp/phase9_campaign/planner_models/planner_set_v2_expanded_v1"


def test_resolve_trained_agent_and_arena_spec_preserve_runtime_contract() -> None:
    agent_template = SelfplayAgentSpec(
        name="planner_set_v2_expanded_v1",
        proposer_checkpoint="models/proposer/checkpoint.pt",
        planner_checkpoint="/srv/old/checkpoint.pt",
        opponent_checkpoint="models/opponent/checkpoint.pt",
        opponent_mode="learned",
        root_top_k=4,
        tags=["experimental"],
        metadata={"role": "experimental"},
    )
    resolved_agent = resolve_trained_agent_spec(
        template_spec=agent_template,
        agent_name="planner_set_v2_expanded_v1",
        planner_checkpoint=Path("/srv/new/checkpoint.pt"),
        run_name="planner_set_v2_expanded_v1",
    )
    assert resolved_agent.planner_checkpoint == "/srv/new/checkpoint.pt"
    assert "fulltrain_campaign" in resolved_agent.tags
    assert resolved_agent.metadata["fulltrain_campaign_run"] == "planner_set_v2_expanded_v1"

    arena_template = SelfplayArenaSpec.from_dict(
        {
            "spec_version": 1,
            "name": "template",
            "agent_specs": {
                "a": "python/configs/a.json",
                "b": "python/configs/b.json",
            },
            "schedule_mode": "round_robin",
            "matchups": [],
            "default_games": 2,
            "default_max_plies": 80,
            "default_initial_fens": ["startpos"],
            "parallel_workers": 2,
            "opening_selection_seed": 7,
            "round_robin_swap_colors": True,
            "include_self_matches": False,
            "metadata": {"purpose": "template"},
        }
    )
    resolved_arena = materialize_fulltrain_arena_spec(
        template_spec=arena_template,
        resolved_agent_specs={
            "symbolic_root_v1": "artifacts/phase9/symbolic_root.json",
            "planner_set_v2_expanded_v1": "artifacts/phase9/planner_set_v2_expanded_v1.json",
        },
        default_initial_fens=["fen_a", "fen_b"],
        campaign_name="phase9_fulltrain_then_arena_expanded_v1",
        default_games=1,
        parallel_workers=6,
        default_max_plies=96,
        opening_selection_seed=20260404,
    )
    assert resolved_arena.name == "phase9_fulltrain_then_arena_expanded_v1_arena"
    assert resolved_arena.default_games == 1
    assert resolved_arena.parallel_workers == 6
    assert resolved_arena.default_max_plies == 96
    assert resolved_arena.default_initial_fens == ["fen_a", "fen_b"]
    assert resolved_arena.agent_specs["planner_set_v2_expanded_v1"].endswith(
        "planner_set_v2_expanded_v1.json"
    )
