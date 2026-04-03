from __future__ import annotations

from pathlib import Path

from train.config import PlannerTrainConfig
from train.eval.campaign import (
    PlannerReplayCampaignRunSpec,
    SelfplayReplayCampaignSpec,
    build_planner_verify_matrix,
    materialize_replay_campaign_planner_config,
    write_selfplay_replay_campaign_spec,
    load_selfplay_replay_campaign_spec,
)


def test_replay_campaign_spec_round_trip(tmp_path: Path) -> None:
    spec = SelfplayReplayCampaignSpec(
        name="campaign",
        output_root="artifacts/phase9/campaign",
        curriculum_plan="artifacts/phase9/curriculum.json",
        stage_name="active_experimental",
        proposer_checkpoint="models/proposer/checkpoint.pt",
        opponent_mode="learned",
        opponent_checkpoint="models/opponent/checkpoint.pt",
        include_unfinished_replay=True,
        verify_dataset_paths=("artifacts/phase8/verify_a.jsonl",),
        baseline_metrics={"active": "artifacts/phase8/active.json"},
        reference_run_name="active",
        planner_runs=(
            PlannerReplayCampaignRunSpec(
                name="run_a",
                base_config_path="python/configs/base.json",
                compare_name="run_a_compare",
            ),
        ),
    )
    path = tmp_path / "campaign.json"
    write_selfplay_replay_campaign_spec(path, spec)
    loaded = load_selfplay_replay_campaign_spec(path)
    assert loaded.name == spec.name
    assert loaded.reference_run_name == "active"
    assert loaded.include_unfinished_replay is True
    assert loaded.planner_runs[0].compare_name == "run_a_compare"


def test_build_planner_verify_matrix_ranks_and_deltas() -> None:
    payload = build_planner_verify_matrix(
        campaign_name="campaign",
        run_metrics={
            "active": {
                "root_top1_accuracy": 0.80,
                "root_top3_accuracy": 0.95,
                "teacher_root_mean_reciprocal_rank": 0.88,
                "teacher_root_mean_probability": 0.70,
            },
            "candidate": {
                "root_top1_accuracy": 0.82,
                "root_top3_accuracy": 0.96,
                "teacher_root_mean_reciprocal_rank": 0.89,
                "teacher_root_mean_probability": 0.72,
            },
        },
        reference_run_name="active",
    )
    assert payload["ranking_by_top1"][0]["name"] == "candidate"
    assert payload["ranking_by_mrr"][0]["name"] == "candidate"
    assert payload["deltas_vs_reference"]["candidate"]["root_top1_accuracy"] == 0.02


def test_materialize_replay_campaign_planner_config_repoints_train_and_outputs() -> None:
    base_config = PlannerTrainConfig.from_dict(
        {
            "seed": 1,
            "output_dir": "/srv/base/output",
            "initial_checkpoint": "/srv/base/model/checkpoint.pt",
            "data": {
                "train_path": "artifacts/base/train.jsonl",
                "validation_path": "artifacts/base/validation.jsonl",
            },
            "model": {
                "architecture": "set_v6",
                "hidden_dim": 64,
                "hidden_layers": 1,
                "action_embedding_dim": 32,
                "latent_feature_dim": 0,
                "deliberation_steps": 1,
                "memory_slots": 0,
                "dropout": 0.0,
            },
            "optimization": {
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "teacher_policy_loss_weight": 1.0,
            },
            "evaluation": {"top_k": 3},
            "runtime": {"torch_threads": 0, "dataloader_workers": 0},
            "export": {"bundle_dir": "/srv/base/export", "checkpoint_name": "checkpoint.pt"},
        }
    )
    payload = materialize_replay_campaign_planner_config(
        base_config=base_config,
        replay_head_train_path=Path("artifacts/replay/planner_head_train.jsonl"),
        output_root=Path("/tmp/campaign"),
        run_name="run_a",
    )
    assert payload["data"]["train_path"] == "artifacts/replay/planner_head_train.jsonl"
    assert payload["output_dir"] == "/tmp/campaign/planner_runs/run_a"
    assert payload["export"]["bundle_dir"] == "/tmp/campaign/planner_models/run_a"
