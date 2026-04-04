from __future__ import annotations

import json
from pathlib import Path

from train.datasets.planner_head import PlannerHeadExample, write_planner_head_artifact
from train.eval.evolution_campaign import (
    PlannerEvolutionCampaignSpec,
    PlannerEvolutionRunSpec,
    load_planner_evolution_campaign_spec,
    materialize_filtered_planner_workflow_summary,
    write_planner_evolution_campaign_spec,
)


def _planner_head_example(*, sample_id: str, root_value_cp: float, candidate_scores: list[float]) -> PlannerHeadExample:
    candidate_count = len(candidate_scores)
    return PlannerHeadExample(
        sample_id=sample_id,
        split="train",
        fen="startpos",
        feature_vector=[0.0, 1.0],
        candidate_context_version=2,
        global_context_version=1,
        global_features=[0.0, 0.0],
        candidate_action_indices=list(range(candidate_count)),
        candidate_features=[[0.1, 0.2] for _ in range(candidate_count)],
        proposer_scores=[0.0 for _ in range(candidate_count)],
        transition_context_version=1,
        transition_features=[[0.0, 0.0] for _ in range(candidate_count)],
        reply_peak_probabilities=[0.5 for _ in range(candidate_count)],
        pressures=[0.0 for _ in range(candidate_count)],
        uncertainties=[0.0 for _ in range(candidate_count)],
        curriculum_bucket_labels=["medium"],
        curriculum_priority=1.0,
        teacher_top1_action_index=0,
        teacher_top1_candidate_index=0,
        teacher_policy=[1.0 if index == 0 else 0.0 for index in range(candidate_count)],
        teacher_root_value_cp=root_value_cp,
        teacher_top1_minus_top2_cp=20.0,
        teacher_candidate_scores_cp=candidate_scores,
    )


def test_planner_evolution_campaign_spec_round_trip(tmp_path: Path) -> None:
    spec = PlannerEvolutionCampaignSpec(
        name="campaign",
        output_root="/srv/evolution",
        source_workflow_summary="/srv/workflow/summary.json",
        filtered_workflow_root="/srv/workflow_filtered",
        bootstrap_summary_path="/srv/evolution_seed/round_03/summary.json",
        training_tiers=("pgn_10k", "merged_unique_122k", "unique_pi_400k"),
        verify_tiers=("pgn_10k", "merged_unique_122k", "unique_pi_400k"),
        filtered_training_tiers=("unique_pi_400k",),
        iterations=20,
        arena_template_spec_path="python/configs/arena.json",
        benchmark_agent_specs={"symbolic_root_v1": "python/configs/symbolic.json"},
        planner_runs=(
            PlannerEvolutionRunSpec(
                name="planner_set_v2_expanded_v1",
                base_config_path="python/configs/planner.json",
                agent_template_spec_path="python/configs/agent.json",
            ),
        ),
    )
    path = tmp_path / "campaign.json"
    write_planner_evolution_campaign_spec(path, spec)
    loaded = load_planner_evolution_campaign_spec(path)
    assert loaded.name == spec.name
    assert loaded.iterations == 20
    assert loaded.bootstrap_summary_path == "/srv/evolution_seed/round_03/summary.json"
    assert loaded.filtered_training_tiers == ("unique_pi_400k",)
    assert loaded.planner_runs[0].name == "planner_set_v2_expanded_v1"


def test_materialize_filtered_planner_workflow_summary_filters_only_selected_tier(tmp_path: Path) -> None:
    train_input_dir = tmp_path / "source" / "unique_pi_400k_train_v1"
    validation_input_dir = tmp_path / "source" / "unique_pi_400k_validation_v1"
    verify_input_dir = tmp_path / "source" / "unique_pi_400k_verify_v1"
    train_input_dir.mkdir(parents=True)
    validation_input_dir.mkdir(parents=True)
    verify_input_dir.mkdir(parents=True)

    kept = _planner_head_example(
        sample_id="kept",
        root_value_cp=120.0,
        candidate_scores=[120.0, 60.0],
    )
    dropped_ambiguous = _planner_head_example(
        sample_id="drop_ambiguous",
        root_value_cp=80.0,
        candidate_scores=[11.0, 8.0],
    )
    dropped_extreme = _planner_head_example(
        sample_id="drop_extreme",
        root_value_cp=5000.0,
        candidate_scores=[5000.0, 4500.0],
    )
    train_path = train_input_dir / "planner_head_train.jsonl"
    validation_path = validation_input_dir / "planner_head_validation.jsonl"
    verify_path = verify_input_dir / "planner_head_test.jsonl"
    write_planner_head_artifact(train_path, [kept, dropped_ambiguous, dropped_extreme])
    write_planner_head_artifact(validation_path, [kept, dropped_ambiguous])
    write_planner_head_artifact(verify_path, [kept, dropped_ambiguous])

    workflow_summary = {
        "tiers": {
            "pgn_10k": {
                "train": {"planner_head_path": str(train_path)},
                "validation": {"planner_head_path": str(validation_path)},
                "verify": {"planner_head_path": str(verify_path)},
            },
            "unique_pi_400k": {
                "train": {"planner_head_path": str(train_path)},
                "validation": {"planner_head_path": str(validation_path)},
                "verify": {"planner_head_path": str(verify_path)},
            },
        }
    }
    source_summary_path = tmp_path / "workflow_summary.json"
    source_summary_path.write_text(json.dumps(workflow_summary) + "\n", encoding="utf-8")

    filtered = materialize_filtered_planner_workflow_summary(
        source_summary_path=source_summary_path,
        output_root=tmp_path / "filtered",
        filtered_tiers=("unique_pi_400k",),
        max_abs_root_value_cp=2000.0,
        ambiguous_score_span_cp=5.0,
        min_candidate_count=2,
    )

    filtered_train_path = Path(filtered["tiers"]["unique_pi_400k"]["train"]["planner_head_path"])
    filtered_validation_path = Path(filtered["tiers"]["unique_pi_400k"]["validation"]["planner_head_path"])
    untouched_verify_path = Path(filtered["tiers"]["unique_pi_400k"]["verify"]["planner_head_path"])
    assert filtered_train_path != train_path
    assert filtered_validation_path != validation_path
    assert untouched_verify_path == verify_path
    assert len(filtered_train_path.read_text(encoding="utf-8").splitlines()) == 1
    assert len(filtered_validation_path.read_text(encoding="utf-8").splitlines()) == 1
    assert Path(filtered["tiers"]["unique_pi_400k"]["train"]["filter_summary_path"]).exists()
