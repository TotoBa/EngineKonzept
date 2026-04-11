"""Reusable Phase-10 campaign spec and arena-resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

from train.eval.agent_spec import load_selfplay_agent_spec, write_selfplay_agent_spec
from train.eval.initial_fens import load_selfplay_initial_fen_suite

if TYPE_CHECKING:
    from train.eval.arena import SelfplayArenaSpec


@dataclass(frozen=True)
class Phase10ReferenceAgentSpec:
    """Configured reference-arm pointer for one Phase-10 comparison campaign."""

    name: str
    spec_path: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Phase10ReferenceAgentSpec":
        return cls(name=str(payload["name"]), spec_path=str(payload["spec_path"]))


@dataclass(frozen=True)
class Phase10Lapv1AgentVariantSpec:
    """Resolved runtime variant for one tracked LAP checkpoint."""

    name: str
    deliberation_max_inner_steps: int = 0
    deliberation_q_threshold: float | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Phase10Lapv1AgentVariantSpec":
        return cls(
            name=str(payload["name"]),
            deliberation_max_inner_steps=int(payload.get("deliberation_max_inner_steps", 0)),
            deliberation_q_threshold=(
                float(payload["deliberation_q_threshold"])
                if payload.get("deliberation_q_threshold") is not None
                else None
            ),
            tags=tuple(str(value) for value in list(payload.get("tags") or [])),
            metadata={str(key): value for key, value in dict(payload.get("metadata") or {}).items()},
        )


@dataclass(frozen=True)
class Phase10Lapv1ArenaCampaignSpec:
    """Versioned Phase-10 training-plus-arena campaign contract."""

    name: str
    output_root: str
    merged_raw_dir: str
    train_dataset_dir: str
    verify_dataset_dir: str
    phase5_source_name: str
    phase5_seed: str
    model_label: str = "LAPv1"
    phase5_oracle_workers: int = 6
    phase5_oracle_batch_size: int = 0
    phase5_chunk_size: int = 5000
    phase5_log_every_chunks: int = 1
    reuse_existing_artifacts: bool = False
    workflow_output_root: str = ""
    proposer_checkpoint: str = ""
    teacher_engine_path: str = "/usr/games/stockfish18"
    teacher_nodes: int | None = 64
    teacher_depth: int | None = None
    train_teacher_depth: int | None = None
    validation_teacher_depth: int | None = None
    verify_teacher_depth: int | None = None
    teacher_multipv: int = 8
    teacher_policy_temperature_cp: float = 100.0
    teacher_top_k: int = 8
    workflow_chunk_size: int = 2048
    workflow_parallel_workers: int = 1
    workflow_log_every: int = 1000
    lapv1_config_path: str = ""
    lapv1_agent_spec_path: str = ""
    lapv1_agent_variants: tuple[Phase10Lapv1AgentVariantSpec, ...] = ()
    pre_verify_selfplay_games: int = 0
    pre_verify_selfplay_games_per_task: int = 8
    pre_verify_selfplay_max_plies: int | None = None
    pre_verify_selfplay_opening_selection_seed: int | None = None
    pre_verify_selfplay_agent_variant_name: str | None = None
    lapv1_verify_output_path: str = ""
    warm_start_source_checkpoint: str | None = None
    reference_arena_summary_path: str = ""
    reference_verify_matrix_path: str | None = None
    reference_agents: tuple[Phase10ReferenceAgentSpec, ...] = ()
    reference_active_agent_specs_dir: str | None = None
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
            model_label=str(payload.get("model_label", "LAPv1")),
            merged_raw_dir=str(payload["merged_raw_dir"]),
            train_dataset_dir=str(payload["train_dataset_dir"]),
            verify_dataset_dir=str(payload["verify_dataset_dir"]),
            phase5_source_name=str(payload["phase5_source_name"]),
            phase5_seed=str(payload["phase5_seed"]),
            phase5_oracle_workers=int(payload.get("phase5_oracle_workers", 6)),
            phase5_oracle_batch_size=int(payload.get("phase5_oracle_batch_size", 0)),
            phase5_chunk_size=int(payload.get("phase5_chunk_size", 5000)),
            phase5_log_every_chunks=int(payload.get("phase5_log_every_chunks", 1)),
            reuse_existing_artifacts=bool(payload.get("reuse_existing_artifacts", False)),
            workflow_output_root=str(payload["workflow_output_root"]),
            proposer_checkpoint=str(payload["proposer_checkpoint"]),
            teacher_engine_path=str(payload.get("teacher_engine_path", "/usr/games/stockfish18")),
            teacher_nodes=(
                int(payload["teacher_nodes"])
                if payload.get("teacher_nodes") is not None
                else None
            ),
            teacher_depth=(
                int(payload["teacher_depth"])
                if payload.get("teacher_depth") is not None
                else None
            ),
            train_teacher_depth=(
                int(payload["train_teacher_depth"])
                if payload.get("train_teacher_depth") is not None
                else None
            ),
            validation_teacher_depth=(
                int(payload["validation_teacher_depth"])
                if payload.get("validation_teacher_depth") is not None
                else None
            ),
            verify_teacher_depth=(
                int(payload["verify_teacher_depth"])
                if payload.get("verify_teacher_depth") is not None
                else None
            ),
            teacher_multipv=int(payload.get("teacher_multipv", 8)),
            teacher_policy_temperature_cp=float(payload.get("teacher_policy_temperature_cp", 100.0)),
            teacher_top_k=int(payload.get("teacher_top_k", 8)),
            workflow_chunk_size=int(payload.get("workflow_chunk_size", 2048)),
            workflow_parallel_workers=int(payload.get("workflow_parallel_workers", 1)),
            workflow_log_every=int(payload.get("workflow_log_every", 1000)),
            lapv1_config_path=str(payload["lapv1_config_path"]),
            lapv1_agent_spec_path=str(payload["lapv1_agent_spec_path"]),
            lapv1_agent_variants=tuple(
                Phase10Lapv1AgentVariantSpec.from_dict(dict(entry))
                for entry in list(payload.get("lapv1_agent_variants") or [])
            ),
            pre_verify_selfplay_games=int(payload.get("pre_verify_selfplay_games", 0)),
            pre_verify_selfplay_games_per_task=int(
                payload.get("pre_verify_selfplay_games_per_task", 8)
            ),
            pre_verify_selfplay_max_plies=(
                int(payload["pre_verify_selfplay_max_plies"])
                if payload.get("pre_verify_selfplay_max_plies") is not None
                else None
            ),
            pre_verify_selfplay_opening_selection_seed=(
                int(payload["pre_verify_selfplay_opening_selection_seed"])
                if payload.get("pre_verify_selfplay_opening_selection_seed") is not None
                else None
            ),
            pre_verify_selfplay_agent_variant_name=(
                str(payload["pre_verify_selfplay_agent_variant_name"])
                if payload.get("pre_verify_selfplay_agent_variant_name") is not None
                else None
            ),
            lapv1_verify_output_path=str(payload["lapv1_verify_output_path"]),
            warm_start_source_checkpoint=(
                str(payload["warm_start_source_checkpoint"])
                if payload.get("warm_start_source_checkpoint") is not None
                else None
            ),
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
            reference_active_agent_specs_dir=(
                str(payload["reference_active_agent_specs_dir"])
                if payload.get("reference_active_agent_specs_dir") is not None
                else None
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


def resolve_repo_path(repo_root: Path, path: Path) -> Path:
    """Resolve one potentially repo-relative path against the repository root."""
    return path if path.is_absolute() else repo_root / path


def load_phase10_lapv1_arena_campaign_spec(path: Path) -> Phase10Lapv1ArenaCampaignSpec:
    """Load one versioned Phase-10 campaign spec from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("phase10 LAPv1 arena campaign spec must be a JSON object")
    return Phase10Lapv1ArenaCampaignSpec.from_dict(payload)


def select_reference_agents(
    spec: Phase10Lapv1ArenaCampaignSpec,
    *,
    repo_root: Path,
) -> list[str]:
    """Select the top configured reference agents from one tracked arena summary."""
    arena_summary = json.loads(
        resolve_repo_path(repo_root, Path(spec.reference_arena_summary_path)).read_text(
            encoding="utf-8"
        )
    )
    verify_payload = (
        json.loads(
            resolve_repo_path(repo_root, Path(spec.reference_verify_matrix_path)).read_text(
                encoding="utf-8"
            )
        )
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


def materialize_resolved_lapv1_agent_specs(
    *,
    spec: Phase10Lapv1ArenaCampaignSpec,
    tracked_lapv1_agent_path: Path,
    output_root: Path,
) -> dict[str, Path]:
    """Resolve the tracked agent spec into one or more runtime-variant files."""
    base_spec = load_selfplay_agent_spec(tracked_lapv1_agent_path)
    if not spec.lapv1_agent_variants:
        resolved_path = output_root / "lapv1_agent_spec.resolved.json"
        write_selfplay_agent_spec(resolved_path, base_spec)
        return {base_spec.name: resolved_path}

    resolved_root = output_root / "lapv1_agent_specs"
    resolved_root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for variant in spec.lapv1_agent_variants:
        variant_spec = replace(
            base_spec,
            name=variant.name,
            deliberation_max_inner_steps=variant.deliberation_max_inner_steps,
            deliberation_q_threshold=(
                variant.deliberation_q_threshold
                if variant.deliberation_q_threshold is not None
                else base_spec.deliberation_q_threshold
            ),
            tags=list(dict.fromkeys([*base_spec.tags, *variant.tags])),
            metadata={
                **dict(base_spec.metadata),
                **dict(variant.metadata),
                "lapv1_variant": variant.name,
                "deliberation_max_inner_steps": variant.deliberation_max_inner_steps,
            },
        )
        resolved_path = resolved_root / f"{variant.name}.json"
        write_selfplay_agent_spec(resolved_path, variant_spec)
        resolved[variant.name] = resolved_path
    return resolved


def select_pre_verify_lapv1_agent(
    spec: Phase10Lapv1ArenaCampaignSpec,
    *,
    resolved_agent_paths: Mapping[str, Path] | None,
    repo_root: Path,
) -> tuple[str, Path]:
    """Select the tracked LAP runtime used for pre-verify selfplay."""
    if resolved_agent_paths:
        if spec.pre_verify_selfplay_agent_variant_name is not None:
            selected_name = spec.pre_verify_selfplay_agent_variant_name
            selected_path = resolved_agent_paths.get(selected_name)
            if selected_path is None:
                raise ValueError(
                    "unknown pre_verify_selfplay_agent_variant_name: "
                    f"{spec.pre_verify_selfplay_agent_variant_name}"
                )
            return selected_name, selected_path

        for variant in spec.lapv1_agent_variants:
            if str(variant.metadata.get("variant_role", "")) == "primary_runtime_candidate":
                selected_path = resolved_agent_paths.get(variant.name)
                if selected_path is not None:
                    return variant.name, selected_path
        for variant in spec.lapv1_agent_variants:
            if str(variant.metadata.get("variant_role", "")) == "budgeted_deeper_runtime_candidate":
                selected_path = resolved_agent_paths.get(variant.name)
                if selected_path is not None:
                    return variant.name, selected_path
        first_name = next(iter(resolved_agent_paths))
        return first_name, resolved_agent_paths[first_name]

    base_path = resolve_repo_path(repo_root, Path(spec.lapv1_agent_spec_path))
    base_spec = load_selfplay_agent_spec(base_path)
    if spec.pre_verify_selfplay_agent_variant_name not in (None, base_spec.name):
        raise ValueError(
            "pre_verify_selfplay_agent_variant_name requires configured lapv1_agent_variants "
            f"or must match the base agent name {base_spec.name!r}"
        )
    return base_spec.name, base_path


def build_resolved_arena_spec(
    spec: Phase10Lapv1ArenaCampaignSpec,
    selected_reference_agents: Sequence[str],
    *,
    repo_root: Path,
    lapv1_agent_paths: Mapping[str, Path] | None = None,
) -> "SelfplayArenaSpec":
    """Materialize the resolved Phase-10 comparison arena for one campaign run."""
    from train.eval.arena import SelfplayArenaSpec
    from train.eval.selfplay import SelfplayMaxPliesAdjudicationSpec

    initial_fens = load_selfplay_initial_fen_suite(
        resolve_repo_path(repo_root, Path(spec.initial_fen_suite_path))
    ).fen_list()
    reference_agent_paths = resolve_reference_agent_paths(
        spec,
        selected_reference_agents,
        repo_root=repo_root,
    )
    lapv1_specs = (
        {name: str(path) for name, path in lapv1_agent_paths.items()}
        if lapv1_agent_paths is not None
        else {
            "lapv1_stage1_all_unique_v1": str(
                resolve_repo_path(repo_root, Path(spec.lapv1_agent_spec_path))
            )
        }
    )
    agent_specs = {
        **lapv1_specs,
        **reference_agent_paths,
        **{
            name: str(resolve_repo_path(repo_root, Path(path)))
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
            "purpose": "lap_model_compare_vs_selected_refs_plus_benchmarks",
            "model_label": spec.model_label,
            "lapv1_agent_names": list(lapv1_specs),
            "reference_arena_summary_path": spec.reference_arena_summary_path,
            "selected_reference_agents": list(selected_reference_agents),
        },
    )


def resolve_reference_agent_paths(
    spec: Phase10Lapv1ArenaCampaignSpec,
    selected_reference_agents: Sequence[str],
    *,
    repo_root: Path,
) -> dict[str, str]:
    """Resolve the final runtime agent-spec file for each chosen reference agent."""
    configured_paths = {entry.name: entry.spec_path for entry in spec.reference_agents}
    active_specs_dir = (
        resolve_repo_path(repo_root, Path(spec.reference_active_agent_specs_dir))
        if spec.reference_active_agent_specs_dir is not None
        else None
    )
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in selected_reference_agents:
        candidate_path: Path | None = None
        if active_specs_dir is not None:
            active_spec_path = active_specs_dir / f"{name}.json"
            if active_spec_path.exists():
                candidate_path = active_spec_path
        if candidate_path is None and name in configured_paths:
            candidate_path = resolve_repo_path(repo_root, Path(configured_paths[name]))
        if candidate_path is None or not candidate_path.exists():
            missing.append(name)
            continue
        resolved[name] = str(candidate_path)
    if missing:
        raise ValueError(
            "missing resolved reference agent specs for: " + ", ".join(sorted(missing))
        )
    return resolved
