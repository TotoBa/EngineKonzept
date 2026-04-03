from __future__ import annotations

import json
from pathlib import Path

from train.eval.curriculum import (
    build_phase9_expanded_curriculum_plan,
    write_selfplay_curriculum_plan,
)


def _write_agent_spec(path: Path, *, name: str, tags: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": name,
                "proposer_checkpoint": "models/proposer/example/checkpoint.pt",
                "planner_checkpoint": None,
                "opponent_checkpoint": None,
                "dynamics_checkpoint": None,
                "opponent_mode": "none",
                "root_top_k": 1,
                "tags": tags,
                "metadata": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_phase9_expanded_curriculum_plan_uses_required_tiers(tmp_path: Path) -> None:
    repo_root = tmp_path
    config_root = repo_root / "python" / "configs"
    config_root.mkdir(parents=True)

    _write_agent_spec(config_root / "phase9_agent_symbolic_root_v1.json", name="symbolic_root_v1", tags=["baseline"])
    _write_agent_spec(config_root / "phase9_agent_planner_set_v2_expanded_v1.json", name="planner_set_v2_expanded_v1", tags=["active"])
    _write_agent_spec(config_root / "phase9_agent_planner_set_v6_expanded_v1.json", name="planner_set_v6_expanded_v1", tags=["experimental"])
    _write_agent_spec(config_root / "phase9_agent_planner_set_v6_margin_expanded_v1.json", name="planner_set_v6_margin_expanded_v1", tags=["experimental"])
    _write_agent_spec(config_root / "phase9_agent_planner_set_v6_rank_expanded_v1.json", name="planner_set_v6_rank_expanded_v1", tags=["experimental"])
    _write_agent_spec(config_root / "phase9_agent_planner_recurrent_expanded_v1.json", name="planner_recurrent_expanded_v1", tags=["experimental"])

    arena_summary_path = repo_root / "arena_summary.json"
    arena_summary_path.write_text(
        json.dumps(
            {
                "standings": {
                    "symbolic_root_v1": {"games": 2, "score": 1.0},
                    "planner_set_v2_expanded_v1": {"games": 2, "score": 1.5},
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    corpus_manifest_path = repo_root / "corpus_suite.json"
    corpus_manifest_path.write_text(
        json.dumps(
            {
                "tiers": {
                    "pgn_10k": {},
                    "merged_unique_122k": {},
                    "unique_pi_400k": {},
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    plan = build_phase9_expanded_curriculum_plan(
        repo_root=repo_root,
        source_arena_summary_path=arena_summary_path,
        corpus_suite_manifest_path=corpus_manifest_path,
    )
    assert plan.metadata["required_tiers"] == ["pgn_10k", "merged_unique_122k", "unique_pi_400k"]
    assert len(plan.planner_runs) == 5
    assert len(plan.stages) == 2
    probe_stage = plan.stages[0]
    assert probe_stage.games_per_matchup == 2
    assert probe_stage.agent_sampling_weights["python/configs/phase9_agent_planner_set_v2_expanded_v1.json"] > probe_stage.agent_sampling_weights["python/configs/phase9_agent_symbolic_root_v1.json"]


def test_write_selfplay_curriculum_plan_round_trip_payload(tmp_path: Path) -> None:
    repo_root = tmp_path
    config_root = repo_root / "python" / "configs"
    config_root.mkdir(parents=True)
    for name, tags in (
        ("phase9_agent_symbolic_root_v1.json", ["baseline"]),
        ("phase9_agent_planner_set_v2_expanded_v1.json", ["active"]),
        ("phase9_agent_planner_set_v6_expanded_v1.json", ["experimental"]),
        ("phase9_agent_planner_set_v6_margin_expanded_v1.json", ["experimental"]),
        ("phase9_agent_planner_set_v6_rank_expanded_v1.json", ["experimental"]),
        ("phase9_agent_planner_recurrent_expanded_v1.json", ["experimental"]),
    ):
        _write_agent_spec(config_root / name, name=name.removesuffix(".json"), tags=tags)

    arena_summary_path = repo_root / "arena_summary.json"
    arena_summary_path.write_text('{"standings": {}}\n', encoding="utf-8")
    corpus_manifest_path = repo_root / "corpus_suite.json"
    corpus_manifest_path.write_text(
        '{"tiers": {"pgn_10k": {}, "merged_unique_122k": {}, "unique_pi_400k": {}}}\n',
        encoding="utf-8",
    )
    plan = build_phase9_expanded_curriculum_plan(
        repo_root=repo_root,
        source_arena_summary_path=arena_summary_path,
        corpus_suite_manifest_path=corpus_manifest_path,
    )
    output_path = tmp_path / "curriculum.json"
    write_selfplay_curriculum_plan(output_path, plan)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["name"] == "phase9_active_experimental_expanded_v1"
    assert payload["spec_version"] == 1
    assert len(payload["stages"]) == 2
