from __future__ import annotations

import json
from pathlib import Path

from train.eval.curriculum import (
    build_curriculum_stage_arena_spec,
    build_phase9_expanded_curriculum_plan,
    load_selfplay_curriculum_plan,
    write_selfplay_curriculum_plan,
)
from train.eval.initial_fens import SelfplayInitialFenEntry, SelfplayInitialFenSuite, write_selfplay_initial_fen_suite


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
        plan_name="phase9_active_experimental_expanded_v2",
        expanded_replay_buffer_output_root="artifacts/phase9/replay_buffer_active_experimental_expanded_v2",
        expanded_initial_fen_suite="artifacts/phase9/initial_fens_v1.json",
        expanded_games_per_matchup=6,
        expanded_max_plies=96,
    )
    assert plan.name == "phase9_active_experimental_expanded_v2"
    assert plan.metadata["required_tiers"] == ["pgn_10k", "merged_unique_122k", "unique_pi_400k"]
    assert len(plan.planner_runs) == 5
    assert len(plan.stages) == 2
    probe_stage = plan.stages[0]
    assert probe_stage.games_per_matchup == 2
    assert probe_stage.agent_sampling_weights["python/configs/phase9_agent_planner_set_v2_expanded_v1.json"] > probe_stage.agent_sampling_weights["python/configs/phase9_agent_symbolic_root_v1.json"]
    expanded_stage = plan.stages[1]
    assert expanded_stage.initial_fen_suite == "artifacts/phase9/initial_fens_v1.json"
    assert expanded_stage.games_per_matchup == 6
    assert expanded_stage.max_plies == 96
    assert expanded_stage.replay_buffer_output_root == "artifacts/phase9/replay_buffer_active_experimental_expanded_v2"


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
    loaded = load_selfplay_curriculum_plan(output_path)
    assert loaded.name == plan.name
    assert len(loaded.stages) == 2


def test_build_curriculum_stage_arena_spec_applies_stage_overrides(tmp_path: Path) -> None:
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

    (config_root / "phase9_arena_active_probe_v1.json").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": "probe",
                "agent_specs": {
                    "symbolic_root_v1": "python/configs/phase9_agent_symbolic_root_v1.json",
                    "planner_set_v2_expanded_v1": "python/configs/phase9_agent_planner_set_v2_expanded_v1.json",
                },
                "schedule_mode": "round_robin",
                "matchups": [],
                "default_games": 1,
                "default_max_plies": 8,
                "default_initial_fens": ["startpos"],
                "parallel_workers": 6,
                "opening_selection_seed": 99,
                "round_robin_swap_colors": True,
                "include_self_matches": False,
                "metadata": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (config_root / "phase9_arena_active_experimental_expanded_v1.json").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": "expanded",
                "agent_specs": {
                    "symbolic_root_v1": "python/configs/phase9_agent_symbolic_root_v1.json",
                    "planner_set_v2_expanded_v1": "python/configs/phase9_agent_planner_set_v2_expanded_v1.json",
                    "planner_set_v6_expanded_v1": "python/configs/phase9_agent_planner_set_v6_expanded_v1.json",
                    "planner_set_v6_margin_expanded_v1": "python/configs/phase9_agent_planner_set_v6_margin_expanded_v1.json",
                    "planner_set_v6_rank_expanded_v1": "python/configs/phase9_agent_planner_set_v6_rank_expanded_v1.json",
                    "planner_recurrent_expanded_v1": "python/configs/phase9_agent_planner_recurrent_expanded_v1.json",
                },
                "schedule_mode": "round_robin",
                "matchups": [],
                "default_games": 1,
                "default_max_plies": 8,
                "default_initial_fens": ["startpos"],
                "parallel_workers": 6,
                "opening_selection_seed": 99,
                "round_robin_swap_colors": True,
                "include_self_matches": False,
                "metadata": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    arena_summary_path = repo_root / "arena_summary.json"
    arena_summary_path.write_text('{"standings": {}}\n', encoding="utf-8")
    corpus_manifest_path = repo_root / "corpus_suite.json"
    corpus_manifest_path.write_text(
        '{"tiers": {"pgn_10k": {}, "merged_unique_122k": {}, "unique_pi_400k": {}}}\n',
        encoding="utf-8",
    )
    initial_fen_suite_path = repo_root / "artifacts" / "phase9" / "initial_fens_v1.json"
    write_selfplay_initial_fen_suite(
        initial_fen_suite_path,
        SelfplayInitialFenSuite(
            name="initial_fens_v1",
            entries=[
                SelfplayInitialFenEntry(
                    fen="8/8/8/8/8/8/8/K6k w - - 0 1",
                    tier="pgn_10k",
                    sample_id="sample_1",
                    source_path="dataset.jsonl",
                    result="1-0",
                    selection_score=3.0,
                    tags=["decisive"],
                )
            ],
        ),
    )
    plan = build_phase9_expanded_curriculum_plan(
        repo_root=repo_root,
        source_arena_summary_path=arena_summary_path,
        corpus_suite_manifest_path=corpus_manifest_path,
        expanded_initial_fen_suite="artifacts/phase9/initial_fens_v1.json",
        expanded_games_per_matchup=6,
        expanded_max_plies=96,
    )
    arena_spec = build_curriculum_stage_arena_spec(
        repo_root=repo_root,
        plan=plan,
        stage_name="active_experimental_expanded_round_robin",
    )
    assert arena_spec.default_games == 6
    assert arena_spec.default_max_plies == 96
    assert arena_spec.default_initial_fens == ["8/8/8/8/8/8/8/K6k w - - 0 1"]
    assert arena_spec.parallel_workers == 6
    assert arena_spec.opening_selection_seed == 99
    assert set(arena_spec.agent_specs.values()) == set(plan.stages[1].agent_specs)
    assert arena_spec.metadata["curriculum_stage"] == "active_experimental_expanded_round_robin"
    assert arena_spec.metadata["initial_fen_count"] == 1
