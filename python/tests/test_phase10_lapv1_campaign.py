from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_phase10_lapv1_stage1_arena_campaign.py"
)
_SPEC = importlib.util.spec_from_file_location("run_phase10_lapv1_stage1_arena_campaign", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Phase10Lapv1ArenaCampaignSpec = _MODULE.Phase10Lapv1ArenaCampaignSpec
_build_resolved_arena_spec = _MODULE._build_resolved_arena_spec
_select_reference_agents = _MODULE._select_reference_agents
run_phase10_lapv1_stage1_arena_campaign = _MODULE.run_phase10_lapv1_stage1_arena_campaign


def test_select_reference_agents_uses_arena_then_verify_tiebreak(tmp_path: Path) -> None:
    arena_summary = {
        "standings": {
            "arm_a": {"score": 10.0, "games": 20},
            "arm_b": {"score": 10.0, "games": 20},
            "arm_c": {"score": 9.0, "games": 20},
            "vice_v2": {"score": 19.0, "games": 20},
            "symbolic_root_v1": {"score": 2.0, "games": 20},
        }
    }
    verify_matrix = {
        "runs": {
            "arm_a": {"root_top1_accuracy": 0.60, "teacher_root_mean_reciprocal_rank": 0.70},
            "arm_b": {"root_top1_accuracy": 0.61, "teacher_root_mean_reciprocal_rank": 0.69},
            "arm_c": {"root_top1_accuracy": 0.55, "teacher_root_mean_reciprocal_rank": 0.65},
        }
    }
    arena_path = tmp_path / "arena.json"
    verify_path = tmp_path / "verify.json"
    arena_path.write_text(json.dumps(arena_summary), encoding="utf-8")
    verify_path.write_text(json.dumps(verify_matrix), encoding="utf-8")

    spec = Phase10Lapv1ArenaCampaignSpec.from_dict(
        {
            "name": "phase10_test",
            "output_root": str(tmp_path / "out"),
            "merged_raw_dir": "raw",
            "train_dataset_dir": "train",
            "verify_dataset_dir": "verify",
            "phase5_source_name": "stockfish-unique-pgn",
            "phase5_seed": "seed",
            "workflow_output_root": "workflow",
            "proposer_checkpoint": "models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
            "lapv1_config_path": "python/configs/phase10_lapv1_stage1_all_unique_v1.json",
            "lapv1_agent_spec_path": "python/configs/phase10_agent_lapv1_stage1_all_unique_v1.json",
            "lapv1_verify_output_path": "verify_head.jsonl",
            "reference_arena_summary_path": str(arena_path),
            "reference_verify_matrix_path": str(verify_path),
            "reference_agents": [
                {"name": "arm_a", "spec_path": "a.json"},
                {"name": "arm_b", "spec_path": "b.json"},
                {"name": "arm_c", "spec_path": "c.json"},
            ],
            "reference_excluded_agents": ["symbolic_root_v1", "vice_v2"],
            "top_reference_agents_count": 2,
            "benchmark_agent_specs": {"vice_v2": "python/configs/phase9_agent_uci_vice_v2.json"},
            "initial_fen_suite_path": "artifacts/phase9/initial_fens_active_replay_campaign_adjudicated_v2.json",
        }
    )

    assert _select_reference_agents(spec) == ["arm_b", "arm_a"]


def test_campaign_spec_parses_workflow_chunk_size(tmp_path: Path) -> None:
    arena_path = tmp_path / "arena.json"
    verify_path = tmp_path / "verify.json"
    arena_path.write_text(json.dumps({"standings": {}}), encoding="utf-8")
    verify_path.write_text(json.dumps({"runs": {}}), encoding="utf-8")

    spec = Phase10Lapv1ArenaCampaignSpec.from_dict(
        {
            "name": "phase10_test",
            "output_root": str(tmp_path / "out"),
            "merged_raw_dir": "raw",
            "train_dataset_dir": "train",
            "verify_dataset_dir": "verify",
            "phase5_source_name": "stockfish-unique-pgn",
            "phase5_seed": "seed",
            "workflow_output_root": "workflow",
            "model_label": "LAPv2",
            "proposer_checkpoint": "models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
            "lapv1_config_path": "python/configs/phase10_lapv1_stage1_all_unique_v1.json",
            "lapv1_agent_spec_path": "python/configs/phase10_agent_lapv1_stage1_all_unique_v1.json",
            "lapv1_agent_variants": [
                {"name": "lapv1_inner0", "deliberation_max_inner_steps": 0},
                {"name": "lapv1_inner1", "deliberation_max_inner_steps": 1},
            ],
            "lapv1_verify_output_path": "verify_head.jsonl",
            "warm_start_source_checkpoint": "models/lapv1/baseline_v5/freeze_inner_hard_best.pt",
            "reference_arena_summary_path": str(arena_path),
            "reference_verify_matrix_path": str(verify_path),
            "reference_agents": [],
            "benchmark_agent_specs": {"vice_v2": "python/configs/phase9_agent_uci_vice_v2.json"},
            "initial_fen_suite_path": "artifacts/phase9/initial_fens_active_replay_campaign_adjudicated_v2.json",
            "workflow_chunk_size": 1024,
        }
    )

    assert spec.workflow_chunk_size == 1024
    assert spec.model_label == "LAPv2"
    assert (
        spec.warm_start_source_checkpoint
        == "models/lapv1/baseline_v5/freeze_inner_hard_best.pt"
    )
    assert [variant.name for variant in spec.lapv1_agent_variants] == [
        "lapv1_inner0",
        "lapv1_inner1",
    ]


def test_campaign_spec_parses_teacher_depth_and_parallel_workers(tmp_path: Path) -> None:
    arena_path = tmp_path / "arena.json"
    verify_path = tmp_path / "verify.json"
    arena_path.write_text(json.dumps({"standings": {}}), encoding="utf-8")
    verify_path.write_text(json.dumps({"runs": {}}), encoding="utf-8")

    spec = Phase10Lapv1ArenaCampaignSpec.from_dict(
        {
            "name": "phase10_test",
            "output_root": str(tmp_path / "out"),
            "merged_raw_dir": "raw",
            "train_dataset_dir": "train",
            "verify_dataset_dir": "verify",
            "phase5_source_name": "stockfish-unique-pgn",
            "phase5_seed": "seed",
            "workflow_output_root": "workflow",
            "proposer_checkpoint": "models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
            "teacher_nodes": None,
            "teacher_depth": 10,
            "teacher_multipv": 8,
            "workflow_parallel_workers": 3,
            "lapv1_config_path": "python/configs/phase10_lapv1_stage1_all_unique_v1.json",
            "lapv1_agent_spec_path": "python/configs/phase10_agent_lapv1_stage1_all_unique_v1.json",
            "lapv1_verify_output_path": "verify_head.jsonl",
            "reference_arena_summary_path": str(arena_path),
            "reference_verify_matrix_path": str(verify_path),
            "reference_agents": [],
            "benchmark_agent_specs": {"vice_v2": "python/configs/phase9_agent_uci_vice_v2.json"},
            "initial_fen_suite_path": "artifacts/phase9/initial_fens_active_replay_campaign_adjudicated_v2.json",
        }
    )

    assert spec.teacher_nodes is None
    assert spec.teacher_depth == 10
    assert spec.workflow_parallel_workers == 3


def test_campaign_dry_run_exposes_model_label_and_warm_start(tmp_path: Path) -> None:
    arena_path = tmp_path / "arena.json"
    verify_path = tmp_path / "verify.json"
    arena_path.write_text(json.dumps({"standings": {}}), encoding="utf-8")
    verify_path.write_text(json.dumps({"runs": {}}), encoding="utf-8")

    spec = Phase10Lapv1ArenaCampaignSpec.from_dict(
        {
            "name": "phase10_test",
            "output_root": str(tmp_path / "out"),
            "model_label": "LAPv2",
            "merged_raw_dir": "raw",
            "train_dataset_dir": "train",
            "verify_dataset_dir": "verify",
            "phase5_source_name": "stockfish-unique-pgn",
            "phase5_seed": "seed",
            "workflow_output_root": "workflow",
            "proposer_checkpoint": "models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
            "lapv1_config_path": "python/configs/phase10_lapv1_stage1_all_unique_v1.json",
            "lapv1_agent_spec_path": "python/configs/phase10_agent_lapv1_stage1_all_unique_v1.json",
            "lapv1_verify_output_path": "verify_head.jsonl",
            "warm_start_source_checkpoint": "models/lapv1/stage2_fast_all_unique_v4/bundle/checkpoint.pt",
            "reference_arena_summary_path": str(arena_path),
            "reference_verify_matrix_path": str(verify_path),
            "reference_agents": [],
            "top_reference_agents_count": 0,
            "benchmark_agent_specs": {"vice_v2": "python/configs/phase9_agent_uci_vice_v2.json"},
            "initial_fen_suite_path": "artifacts/phase9/initial_fens_active_replay_campaign_adjudicated_v2.json",
        }
    )

    summary = run_phase10_lapv1_stage1_arena_campaign(
        spec=spec,
        skip_existing=True,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["model_label"] == "LAPv2"
    assert (
        summary["warm_start_source_checkpoint"]
        == "models/lapv1/stage2_fast_all_unique_v4/bundle/checkpoint.pt"
    )


def test_build_resolved_arena_spec_uses_initial_fen_suite_entries(tmp_path: Path) -> None:
    arena_path = tmp_path / "arena.json"
    verify_path = tmp_path / "verify.json"
    suite_path = tmp_path / "initial_fens.json"
    reference_spec_path = tmp_path / "a.json"
    arena_path.write_text(json.dumps({"standings": {}}), encoding="utf-8")
    verify_path.write_text(json.dumps({"runs": {}}), encoding="utf-8")
    reference_spec_path.write_text("{}", encoding="utf-8")
    suite_path.write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": "suite",
                "entries": [
                    {
                        "fen": "startpos",
                        "tier": "test",
                        "sample_id": "s1",
                        "source_path": "x",
                        "result": "*",
                        "selection_score": 1.0,
                    },
                    {
                        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                        "tier": "test",
                        "sample_id": "s2",
                        "source_path": "x",
                        "result": "*",
                        "selection_score": 2.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    spec = Phase10Lapv1ArenaCampaignSpec.from_dict(
        {
            "name": "phase10_test",
            "output_root": str(tmp_path / "out"),
            "merged_raw_dir": "raw",
            "train_dataset_dir": "train",
            "verify_dataset_dir": "verify",
            "phase5_source_name": "stockfish-unique-pgn",
            "phase5_seed": "seed",
            "workflow_output_root": "workflow",
            "proposer_checkpoint": "models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
            "lapv1_config_path": "python/configs/phase10_lapv1_stage1_fast_all_unique_v1.json",
            "lapv1_agent_spec_path": "python/configs/phase10_agent_lapv1_stage1_fast_all_unique_v1.json",
            "lapv1_verify_output_path": "verify_head.jsonl",
            "reference_arena_summary_path": str(arena_path),
            "reference_verify_matrix_path": str(verify_path),
            "reference_agents": [
                {"name": "arm_a", "spec_path": str(reference_spec_path)},
            ],
            "top_reference_agents_count": 1,
            "benchmark_agent_specs": {"vice_v2": "python/configs/phase9_agent_uci_vice_v2.json"},
            "initial_fen_suite_path": str(suite_path),
        }
    )

    resolved = _build_resolved_arena_spec(spec, ["arm_a"])
    assert resolved.default_initial_fens == [
        "startpos",
        "8/8/8/8/8/8/8/K6k w - - 0 1",
    ]


def test_build_resolved_arena_spec_can_use_resolved_lapv1_agent_paths(tmp_path: Path) -> None:
    arena_path = tmp_path / "arena.json"
    verify_path = tmp_path / "verify.json"
    suite_path = tmp_path / "initial_fens.json"
    resolved_agent_path = tmp_path / "lapv1_agent_spec.resolved.json"
    resolved_agent_path_step1 = tmp_path / "lapv1_agent_spec_step1.resolved.json"
    reference_spec_path = tmp_path / "a.json"
    arena_path.write_text(json.dumps({"standings": {}}), encoding="utf-8")
    verify_path.write_text(json.dumps({"runs": {}}), encoding="utf-8")
    reference_spec_path.write_text("{}", encoding="utf-8")
    suite_path.write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": "suite",
                "entries": [
                    {
                        "fen": "startpos",
                        "tier": "test",
                        "sample_id": "s1",
                        "source_path": "x",
                        "result": "*",
                        "selection_score": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    spec = Phase10Lapv1ArenaCampaignSpec.from_dict(
        {
            "name": "phase10_test",
            "output_root": str(tmp_path / "out"),
            "merged_raw_dir": "raw",
            "train_dataset_dir": "train",
            "verify_dataset_dir": "verify",
            "phase5_source_name": "stockfish-unique-pgn",
            "phase5_seed": "seed",
            "workflow_output_root": "workflow",
            "proposer_checkpoint": "models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
            "lapv1_config_path": "python/configs/phase10_lapv1_stage1_fast_all_unique_v1.json",
            "lapv1_agent_spec_path": "python/configs/phase10_agent_lapv1_stage1_fast_all_unique_v1.json",
            "lapv1_verify_output_path": "verify_head.jsonl",
            "reference_arena_summary_path": str(arena_path),
            "reference_verify_matrix_path": str(verify_path),
            "reference_agents": [
                {"name": "arm_a", "spec_path": str(reference_spec_path)},
            ],
            "top_reference_agents_count": 1,
            "benchmark_agent_specs": {"vice_v2": "python/configs/phase9_agent_uci_vice_v2.json"},
            "initial_fen_suite_path": str(suite_path),
        }
    )

    resolved = _build_resolved_arena_spec(
        spec,
        ["arm_a"],
        lapv1_agent_paths={
            "lapv1_stage1_all_unique_v1_inner0": resolved_agent_path,
            "lapv1_stage1_all_unique_v1_inner1": resolved_agent_path_step1,
        },
    )
    assert resolved.agent_specs["lapv1_stage1_all_unique_v1_inner0"] == str(resolved_agent_path)
    assert resolved.agent_specs["lapv1_stage1_all_unique_v1_inner1"] == str(
        resolved_agent_path_step1
    )


def test_build_resolved_arena_spec_prefers_materialized_reference_agent_specs(tmp_path: Path) -> None:
    arena_path = tmp_path / "arena.json"
    verify_path = tmp_path / "verify.json"
    suite_path = tmp_path / "initial_fens.json"
    active_specs_dir = tmp_path / "active_agent_specs"
    active_specs_dir.mkdir(parents=True, exist_ok=True)
    materialized_spec_path = active_specs_dir / "arm_a.json"
    materialized_spec_path.write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": "arm_a",
                "agent_kind": "planner",
                "proposer_checkpoint": "models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
                "planner_checkpoint": "/tmp/materialized/checkpoint.pt",
                "opponent_mode": "none",
                "root_top_k": 4,
            }
        ),
        encoding="utf-8",
    )
    arena_path.write_text(json.dumps({"standings": {"arm_a": {"score": 1.0, "games": 1}}}), encoding="utf-8")
    verify_path.write_text(json.dumps({"runs": {"arm_a": {"root_top1_accuracy": 0.5}}}), encoding="utf-8")
    suite_path.write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": "suite",
                "entries": [
                    {
                        "fen": "startpos",
                        "tier": "test",
                        "sample_id": "s1",
                        "source_path": "x",
                        "result": "*",
                        "selection_score": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    spec = Phase10Lapv1ArenaCampaignSpec.from_dict(
        {
            "name": "phase10_test",
            "output_root": str(tmp_path / "out"),
            "merged_raw_dir": "raw",
            "train_dataset_dir": "train",
            "verify_dataset_dir": "verify",
            "phase5_source_name": "stockfish-unique-pgn",
            "phase5_seed": "seed",
            "workflow_output_root": "workflow",
            "proposer_checkpoint": "models/proposer/stockfish_pgn_symbolic_v1_v1/checkpoint.pt",
            "lapv1_config_path": "python/configs/phase10_lapv1_stage1_fast_all_unique_v1.json",
            "lapv1_agent_spec_path": "python/configs/phase10_agent_lapv1_stage1_fast_all_unique_v1.json",
            "lapv1_verify_output_path": "verify_head.jsonl",
            "reference_arena_summary_path": str(arena_path),
            "reference_verify_matrix_path": str(verify_path),
            "reference_agents": [
                {"name": "arm_a", "spec_path": "fallback_a.json"},
            ],
            "reference_active_agent_specs_dir": str(active_specs_dir),
            "top_reference_agents_count": 1,
            "benchmark_agent_specs": {"vice_v2": "python/configs/phase9_agent_uci_vice_v2.json"},
            "initial_fen_suite_path": str(suite_path),
        }
    )

    resolved = _build_resolved_arena_spec(spec, ["arm_a"])
    assert resolved.agent_specs["arm_a"] == str(materialized_spec_path)
