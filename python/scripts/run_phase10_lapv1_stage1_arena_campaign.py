"""Run the Phase-10 all-unique LAPv1 Stage1 bootstrap followed by an 8-agent arena."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from train.eval.agent_spec import write_selfplay_agent_spec
from train.eval.arena import SelfplayArenaSpec, run_selfplay_arena
from train.eval.initial_fens import load_selfplay_initial_fen_suite
from train.eval.matrix import build_selfplay_arena_matrix, write_selfplay_arena_matrix
from train.eval.selfplay import SelfplayMaxPliesAdjudicationSpec
from train.trainers import evaluate_lapv1_checkpoint, load_lapv1_train_config


REPO_ROOT = Path(__file__).resolve().parents[2]
_MATERIALIZE_SCRIPT = REPO_ROOT / "python" / "scripts" / "materialize_phase5_raw_tier.py"
_WORKFLOW_SCRIPT = REPO_ROOT / "python" / "scripts" / "build_phase10_lapv1_workflow.py"
_TRAIN_SCRIPT = REPO_ROOT / "python" / "scripts" / "train_lapv1.py"


@dataclass(frozen=True)
class Phase10ReferenceAgentSpec:
    name: str
    spec_path: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Phase10ReferenceAgentSpec":
        return cls(name=str(payload["name"]), spec_path=str(payload["spec_path"]))


@dataclass(frozen=True)
class Phase10Lapv1ArenaCampaignSpec:
    name: str
    output_root: str
    merged_raw_dir: str
    train_dataset_dir: str
    verify_dataset_dir: str
    phase5_source_name: str
    phase5_seed: str
    phase5_oracle_workers: int = 6
    phase5_oracle_batch_size: int = 0
    phase5_chunk_size: int = 5000
    phase5_log_every_chunks: int = 1
    workflow_output_root: str = ""
    proposer_checkpoint: str = ""
    teacher_engine_path: str = "/usr/games/stockfish18"
    teacher_nodes: int = 64
    teacher_multipv: int = 8
    teacher_policy_temperature_cp: float = 100.0
    teacher_top_k: int = 8
    workflow_chunk_size: int = 2048
    workflow_log_every: int = 1000
    lapv1_config_path: str = ""
    lapv1_agent_spec_path: str = ""
    lapv1_verify_output_path: str = ""
    reference_arena_summary_path: str = ""
    reference_verify_matrix_path: str | None = None
    reference_agents: tuple[Phase10ReferenceAgentSpec, ...] = ()
    reference_excluded_agents: tuple[str, ...] = ("symbolic_root_v1", "vice_v2")
    top_reference_agents_count: int = 6
    benchmark_agent_specs: dict[str, str] = field(default_factory=dict)
    initial_fen_suite_path: str = ""
    arena_default_games: int = 1
    arena_parallel_workers: int = 6
    arena_default_max_plies: int = 96
    arena_opening_selection_seed: int | None = None
    max_plies_adjudication: dict[str, object] | None = None
    spec_version: int = 1

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Phase10Lapv1ArenaCampaignSpec":
        return cls(
            name=str(payload["name"]),
            output_root=str(payload["output_root"]),
            merged_raw_dir=str(payload["merged_raw_dir"]),
            train_dataset_dir=str(payload["train_dataset_dir"]),
            verify_dataset_dir=str(payload["verify_dataset_dir"]),
            phase5_source_name=str(payload["phase5_source_name"]),
            phase5_seed=str(payload["phase5_seed"]),
            phase5_oracle_workers=int(payload.get("phase5_oracle_workers", 6)),
            phase5_oracle_batch_size=int(payload.get("phase5_oracle_batch_size", 0)),
            phase5_chunk_size=int(payload.get("phase5_chunk_size", 5000)),
            phase5_log_every_chunks=int(payload.get("phase5_log_every_chunks", 1)),
            workflow_output_root=str(payload["workflow_output_root"]),
            proposer_checkpoint=str(payload["proposer_checkpoint"]),
            teacher_engine_path=str(payload.get("teacher_engine_path", "/usr/games/stockfish18")),
            teacher_nodes=int(payload.get("teacher_nodes", 64)),
            teacher_multipv=int(payload.get("teacher_multipv", 8)),
            teacher_policy_temperature_cp=float(payload.get("teacher_policy_temperature_cp", 100.0)),
            teacher_top_k=int(payload.get("teacher_top_k", 8)),
            workflow_chunk_size=int(payload.get("workflow_chunk_size", 2048)),
            workflow_log_every=int(payload.get("workflow_log_every", 1000)),
            lapv1_config_path=str(payload["lapv1_config_path"]),
            lapv1_agent_spec_path=str(payload["lapv1_agent_spec_path"]),
            lapv1_verify_output_path=str(payload["lapv1_verify_output_path"]),
            reference_arena_summary_path=str(payload["reference_arena_summary_path"]),
            reference_verify_matrix_path=(
                str(payload["reference_verify_matrix_path"])
                if payload.get("reference_verify_matrix_path") is not None
                else None
            ),
            reference_agents=tuple(
                Phase10ReferenceAgentSpec.from_dict(dict(entry))
                for entry in list(payload.get("reference_agents") or [])
            ),
            reference_excluded_agents=tuple(
                str(value) for value in list(payload.get("reference_excluded_agents") or [])
            ),
            top_reference_agents_count=int(payload.get("top_reference_agents_count", 6)),
            benchmark_agent_specs={
                str(name): str(path)
                for name, path in dict(payload.get("benchmark_agent_specs") or {}).items()
            },
            initial_fen_suite_path=str(payload["initial_fen_suite_path"]),
            arena_default_games=int(payload.get("arena_default_games", 1)),
            arena_parallel_workers=int(payload.get("arena_parallel_workers", 6)),
            arena_default_max_plies=int(payload.get("arena_default_max_plies", 96)),
            arena_opening_selection_seed=(
                int(payload["arena_opening_selection_seed"])
                if payload.get("arena_opening_selection_seed") is not None
                else None
            ),
            max_plies_adjudication=(
                dict(payload["max_plies_adjudication"])
                if isinstance(payload.get("max_plies_adjudication"), dict)
                else None
            ),
            spec_version=int(payload.get("spec_version", 1)),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = _load_spec(_resolve_repo_path(args.config))
    summary = run_phase10_lapv1_stage1_arena_campaign(
        spec=spec,
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_phase10_lapv1_stage1_arena_campaign(
    *,
    spec: Phase10Lapv1ArenaCampaignSpec,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    output_root = _resolve_repo_path(Path(spec.output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    lapv1_config_path = _resolve_repo_path(Path(spec.lapv1_config_path))
    lapv1_config = load_lapv1_train_config(lapv1_config_path)
    lapv1_train_paths = [
        _resolve_repo_path(Path(path))
        for path in lapv1_config.data.resolved_train_paths()
    ]
    lapv1_validation_paths = [
        _resolve_repo_path(Path(path))
        for path in lapv1_config.data.resolved_validation_paths()
    ]
    lapv1_verify_path = _resolve_repo_path(Path(spec.lapv1_verify_output_path))

    selected_reference_agents = _select_reference_agents(spec)
    plan = {
        "campaign_name": spec.name,
        "output_root": str(output_root),
        "selected_reference_agents": selected_reference_agents,
        "benchmark_agents": spec.benchmark_agent_specs,
        "lapv1_config_path": str(lapv1_config_path),
        "lapv1_agent_spec_path": str(_resolve_repo_path(Path(spec.lapv1_agent_spec_path))),
    }
    if dry_run:
        return {"dry_run": True, **plan}

    _log("[phase10] materializing all-unique Phase-5 dataset tier")
    train_summary_path = _resolve_repo_path(Path(spec.train_dataset_dir)) / "summary.json"
    verify_summary_path = _resolve_repo_path(Path(spec.verify_dataset_dir)) / "summary.json"
    if not skip_existing or not train_summary_path.exists() or not verify_summary_path.exists():
        _run_materialization(spec)
    else:
        _log("[phase10] reusing existing Phase-5 all-unique dataset artifacts")

    _log("[phase10] building full LAPv1 workflow")
    workflow_summary_path = _resolve_repo_path(Path(spec.workflow_output_root)) / "summary.json"
    if (
        not skip_existing
        or not workflow_summary_path.exists()
        or any(not path.exists() for path in lapv1_train_paths)
        or any(not path.exists() for path in lapv1_validation_paths)
        or not lapv1_verify_path.exists()
    ):
        _run_workflow_build(spec)
    else:
        _log("[phase10] reusing existing LAPv1 workflow artifacts")

    _log("[phase10] training LAPv1 Stage1")
    lapv1_checkpoint = _resolve_repo_path(Path(lapv1_config.export.bundle_dir)) / lapv1_config.export.checkpoint_name
    lapv1_summary_path = _resolve_repo_path(Path(lapv1_config.output_dir)) / "summary.json"
    if not skip_existing or not lapv1_checkpoint.exists() or not lapv1_summary_path.exists():
        _run_lapv1_training(lapv1_config_path)
    else:
        _log("[phase10] reusing existing LAPv1 Stage1 checkpoint")

    _log("[phase10] evaluating LAPv1 verify holdout")
    lapv1_verify_metrics = evaluate_lapv1_checkpoint(
        lapv1_checkpoint,
        dataset_path=lapv1_verify_path,
        top_k=lapv1_config.evaluation.top_k,
    ).to_dict()
    lapv1_verify_path = output_root / "lapv1_verify.json"
    lapv1_verify_path.write_text(
        json.dumps(lapv1_verify_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _log("[phase10] writing resolved LAPv1 agent spec")
    lapv1_agent_path = _resolve_repo_path(Path(spec.lapv1_agent_spec_path))
    lapv1_agent_path.parent.mkdir(parents=True, exist_ok=True)
    # The tracked agent spec already points at the configured checkpoint path; re-write to ensure sync.
    write_selfplay_agent_spec(lapv1_agent_path, _load_agent_spec(lapv1_agent_path))

    _log("[phase10] materializing resolved 8-agent arena spec")
    resolved_arena_spec = _build_resolved_arena_spec(spec, selected_reference_agents)
    resolved_arena_spec_path = output_root / "arena_spec.resolved.json"
    resolved_arena_spec_path.write_text(
        json.dumps(resolved_arena_spec.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _log("[phase10] running arena")
    arena_output_root = output_root / "arena"
    arena_summary = run_selfplay_arena(
        resolved_arena_spec,
        output_root=arena_output_root,
        repo_root=REPO_ROOT,
    )
    arena_matrix = build_selfplay_arena_matrix(arena_summary)
    arena_matrix_path = output_root / "arena_matrix.json"
    write_selfplay_arena_matrix(arena_matrix_path, arena_matrix)

    summary = {
        **plan,
        "train_dataset_summary_path": str(train_summary_path),
        "verify_dataset_summary_path": str(verify_summary_path),
        "workflow_summary_path": str(workflow_summary_path),
        "lapv1_summary_path": str(lapv1_summary_path),
        "lapv1_checkpoint": str(lapv1_checkpoint),
        "lapv1_verify_path": str(lapv1_verify_path),
        "lapv1_verify_metrics": lapv1_verify_metrics,
        "resolved_arena_spec_path": str(resolved_arena_spec_path),
        "arena_summary_path": str(arena_output_root / "summary.json"),
        "arena_matrix_path": str(arena_matrix_path),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_materialization(spec: Phase10Lapv1ArenaCampaignSpec) -> None:
    command = [
        sys.executable,
        "-u",
        str(_MATERIALIZE_SCRIPT),
        "--raw-dir",
        str(_resolve_repo_path(Path(spec.merged_raw_dir))),
        "--train-output-dir",
        str(_resolve_repo_path(Path(spec.train_dataset_dir))),
        "--verify-output-dir",
        str(_resolve_repo_path(Path(spec.verify_dataset_dir))),
        "--source-name",
        spec.phase5_source_name,
        "--seed",
        spec.phase5_seed,
        "--oracle-workers",
        str(spec.phase5_oracle_workers),
        "--oracle-batch-size",
        str(spec.phase5_oracle_batch_size),
        "--chunk-size",
        str(spec.phase5_chunk_size),
        "--log-every-chunks",
        str(spec.phase5_log_every_chunks),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _run_workflow_build(spec: Phase10Lapv1ArenaCampaignSpec) -> None:
    command = [
        sys.executable,
        "-u",
        str(_WORKFLOW_SCRIPT),
        "--train-dataset-dir",
        str(_resolve_repo_path(Path(spec.train_dataset_dir))),
        "--verify-dataset-dir",
        str(_resolve_repo_path(Path(spec.verify_dataset_dir))),
        "--checkpoint",
        str(_resolve_repo_path(Path(spec.proposer_checkpoint))),
        "--teacher-engine",
        str(_resolve_repo_path(Path(spec.teacher_engine_path))),
        "--output-root",
        str(_resolve_repo_path(Path(spec.workflow_output_root))),
        "--nodes",
        str(spec.teacher_nodes),
        "--multipv",
        str(spec.teacher_multipv),
        "--policy-temperature-cp",
        str(spec.teacher_policy_temperature_cp),
        "--top-k",
        str(spec.teacher_top_k),
        "--root-top-k",
        "4",
        "--chunk-size",
        str(spec.workflow_chunk_size),
        "--log-every",
        str(spec.workflow_log_every),
        "--skip-existing",
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _run_lapv1_training(config_path: Path) -> None:
    command = [
        sys.executable,
        "-u",
        str(_TRAIN_SCRIPT),
        "--config",
        str(config_path),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _select_reference_agents(spec: Phase10Lapv1ArenaCampaignSpec) -> list[str]:
    arena_summary = json.loads(
        _resolve_repo_path(Path(spec.reference_arena_summary_path)).read_text(encoding="utf-8")
    )
    verify_payload = (
        json.loads(_resolve_repo_path(Path(spec.reference_verify_matrix_path)).read_text(encoding="utf-8"))
        if spec.reference_verify_matrix_path is not None
        else None
    )
    verify_runs = dict(verify_payload.get("runs") or {}) if isinstance(verify_payload, dict) else {}
    agent_paths = {entry.name: entry.spec_path for entry in spec.reference_agents}
    excluded = set(spec.reference_excluded_agents)

    ranking: list[tuple[float, float, float, float, str]] = []
    for name, stats in dict(arena_summary["standings"]).items():
        if name in excluded or name not in agent_paths:
            continue
        games = max(int(stats.get("games", 0)), 1)
        score = float(stats.get("score", 0.0))
        verify_metrics = dict(verify_runs.get(name) or {})
        ranking.append(
            (
                score / games,
                score,
                float(verify_metrics.get("root_top1_accuracy", 0.0)),
                float(verify_metrics.get("teacher_root_mean_reciprocal_rank", 0.0)),
                name,
            )
        )
    ranking.sort(reverse=True)
    selected = [name for *_metrics, name in ranking[: spec.top_reference_agents_count]]
    if len(selected) != spec.top_reference_agents_count:
        raise ValueError(
            f"expected {spec.top_reference_agents_count} reference agents, found {len(selected)}"
        )
    return selected


def _build_resolved_arena_spec(
    spec: Phase10Lapv1ArenaCampaignSpec,
    selected_reference_agents: Sequence[str],
) -> SelfplayArenaSpec:
    initial_fens = load_selfplay_initial_fen_suite(
        _resolve_repo_path(Path(spec.initial_fen_suite_path))
    ).fen_list()
    agent_specs = {
        "lapv1_stage1_all_unique_v1": str(_resolve_repo_path(Path(spec.lapv1_agent_spec_path))),
        **{
            name: str(_resolve_repo_path(Path(_reference_agent_paths(spec)[name])))
            for name in selected_reference_agents
        },
        **{
            name: str(_resolve_repo_path(Path(path)))
            for name, path in spec.benchmark_agent_specs.items()
        },
    }
    adjudication = (
        SelfplayMaxPliesAdjudicationSpec.from_dict(dict(spec.max_plies_adjudication))
        if spec.max_plies_adjudication is not None
        else None
    )
    return SelfplayArenaSpec(
        name=f"{spec.name}_arena",
        agent_specs=agent_specs,
        schedule_mode="round_robin",
        matchups=[],
        default_games=spec.arena_default_games,
        default_max_plies=spec.arena_default_max_plies,
        default_initial_fens=list(initial_fens),
        parallel_workers=spec.arena_parallel_workers,
        opening_selection_seed=spec.arena_opening_selection_seed,
        round_robin_swap_colors=True,
        include_self_matches=False,
        max_plies_adjudication=adjudication,
        metadata={
            "campaign_name": spec.name,
            "purpose": "lapv1_stage1_bootstrap_vs_top6_plus_vice",
            "reference_arena_summary_path": spec.reference_arena_summary_path,
            "selected_reference_agents": list(selected_reference_agents),
        },
    )


def _reference_agent_paths(spec: Phase10Lapv1ArenaCampaignSpec) -> dict[str, str]:
    return {entry.name: entry.spec_path for entry in spec.reference_agents}


def _load_agent_spec(path: Path) -> Any:
    from train.eval.agent_spec import load_selfplay_agent_spec

    return load_selfplay_agent_spec(path)


def _load_spec(path: Path) -> Phase10Lapv1ArenaCampaignSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("phase10 LAPv1 arena campaign spec must be a JSON object")
    return Phase10Lapv1ArenaCampaignSpec.from_dict(payload)


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
