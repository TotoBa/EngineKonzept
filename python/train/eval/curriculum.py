"""Versioned selfplay-curriculum plans over planner runs, arena suites, and replay buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from train.eval.agent_spec import load_selfplay_agent_spec
from train.eval.arena import (
    SelfplayArenaMatchupSpec,
    SelfplayArenaSpec,
    load_selfplay_arena_spec,
)
from train.eval.initial_fens import load_selfplay_initial_fen_suite


SELFPLAY_CURRICULUM_PLAN_VERSION = 1
_BASE_TAG_WEIGHTS = {
    "active": 1.0,
    "experimental": 0.6,
    "baseline": 0.25,
}


@dataclass(frozen=True)
class PlannerRunSpec:
    """One planned planner-training run to materialize before a larger selfplay sweep."""

    name: str
    config_path: str
    expected_agent_spec: str
    required_tiers: list[str]
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "config_path": self.config_path,
            "expected_agent_spec": self.expected_agent_spec,
            "required_tiers": list(self.required_tiers),
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PlannerRunSpec":
        return cls(
            name=str(payload["name"]),
            config_path=str(payload["config_path"]),
            expected_agent_spec=str(payload["expected_agent_spec"]),
            required_tiers=[str(value) for value in list(payload["required_tiers"])],
            tags=[str(value) for value in list(payload.get("tags") or [])],
        )


@dataclass(frozen=True)
class SelfplayCurriculumStage:
    """One staged selfplay sweep over a versioned arena suite."""

    name: str
    arena_spec: str
    agent_specs: list[str]
    games_per_matchup: int
    max_plies: int
    replay_buffer_output_root: str
    agent_sampling_weights: dict[str, float]
    initial_fen_suite: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "arena_spec": self.arena_spec,
            "agent_specs": list(self.agent_specs),
            "games_per_matchup": self.games_per_matchup,
            "max_plies": self.max_plies,
            "replay_buffer_output_root": self.replay_buffer_output_root,
            "agent_sampling_weights": dict(sorted(self.agent_sampling_weights.items())),
            "initial_fen_suite": self.initial_fen_suite,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayCurriculumStage":
        return cls(
            name=str(payload["name"]),
            arena_spec=str(payload["arena_spec"]),
            agent_specs=[str(value) for value in list(payload["agent_specs"])],
            games_per_matchup=int(payload["games_per_matchup"]),
            max_plies=int(payload["max_plies"]),
            replay_buffer_output_root=str(payload["replay_buffer_output_root"]),
            agent_sampling_weights={
                str(key): float(value)
                for key, value in dict(payload.get("agent_sampling_weights") or {}).items()
            },
            initial_fen_suite=(
                str(payload["initial_fen_suite"])
                if payload.get("initial_fen_suite") is not None
                else None
            ),
            tags=[str(value) for value in list(payload.get("tags") or [])],
        )


@dataclass(frozen=True)
class SelfplayCurriculumPlan:
    """Versioned launch plan for large-corpus planner reruns plus selfplay."""

    name: str
    corpus_suite_manifest: str
    source_arena_summary: str
    planner_runs: list[PlannerRunSpec]
    stages: list[SelfplayCurriculumStage]
    metadata: dict[str, Any] = field(default_factory=dict)
    spec_version: int = SELFPLAY_CURRICULUM_PLAN_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("curriculum plan name must be non-empty")
        if self.spec_version != SELFPLAY_CURRICULUM_PLAN_VERSION:
            raise ValueError(f"unsupported curriculum plan version: {self.spec_version}")
        if not self.planner_runs:
            raise ValueError("curriculum plan must include planner runs")
        if not self.stages:
            raise ValueError("curriculum plan must include at least one stage")

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "corpus_suite_manifest": self.corpus_suite_manifest,
            "source_arena_summary": self.source_arena_summary,
            "planner_runs": [run.to_dict() for run in self.planner_runs],
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SelfplayCurriculumPlan":
        return cls(
            spec_version=int(payload.get("spec_version", SELFPLAY_CURRICULUM_PLAN_VERSION)),
            name=str(payload["name"]),
            corpus_suite_manifest=str(payload["corpus_suite_manifest"]),
            source_arena_summary=str(payload["source_arena_summary"]),
            planner_runs=[
                PlannerRunSpec.from_dict(dict(run))
                for run in list(payload["planner_runs"])
            ],
            stages=[
                SelfplayCurriculumStage.from_dict(dict(stage))
                for stage in list(payload["stages"])
            ],
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "SelfplayCurriculumPlan":
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("selfplay curriculum plan must be a JSON object")
        return cls.from_dict(payload)


def write_selfplay_curriculum_plan(path: Path, plan: SelfplayCurriculumPlan) -> None:
    """Write a curriculum plan as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_selfplay_curriculum_plan(path: Path) -> SelfplayCurriculumPlan:
    """Load a curriculum plan from JSON."""
    return SelfplayCurriculumPlan.from_json(path.read_text(encoding="utf-8"))


def resolve_curriculum_stage(
    plan: SelfplayCurriculumPlan,
    *,
    stage_name: str,
) -> SelfplayCurriculumStage:
    """Look up one stage by name."""
    for stage in plan.stages:
        if stage.name == stage_name:
            return stage
    raise ValueError(f"unknown curriculum stage: {stage_name}")


def build_curriculum_stage_arena_spec(
    *,
    repo_root: Path,
    plan: SelfplayCurriculumPlan,
    stage_name: str,
) -> SelfplayArenaSpec:
    """Resolve one curriculum stage into a concrete arena spec."""
    stage = resolve_curriculum_stage(plan, stage_name=stage_name)
    arena_spec = load_selfplay_arena_spec(_resolve_repo_path(repo_root, stage.arena_spec))
    initial_fen_suite = None
    if stage.initial_fen_suite is not None:
        initial_fen_suite = load_selfplay_initial_fen_suite(
            _resolve_repo_path(repo_root, stage.initial_fen_suite)
        )
    requested_agent_paths = set(stage.agent_specs)
    filtered_agent_specs = {
        agent_name: spec_path
        for agent_name, spec_path in arena_spec.agent_specs.items()
        if spec_path in requested_agent_paths
    }
    if len(filtered_agent_specs) != len(requested_agent_paths):
        missing_paths = sorted(requested_agent_paths - set(filtered_agent_specs.values()))
        raise ValueError(
            f"{stage_name}: stage agent specs missing from arena spec: {', '.join(missing_paths)}"
        )
    filtered_matchups = [
        matchup
        for matchup in arena_spec.matchups
        if matchup.white_agent in filtered_agent_specs and matchup.black_agent in filtered_agent_specs
    ]
    return SelfplayArenaSpec(
        name=f"{arena_spec.name}:{stage.name}",
        agent_specs=filtered_agent_specs,
        schedule_mode=arena_spec.schedule_mode,
        matchups=[
            SelfplayArenaMatchupSpec(
                white_agent=matchup.white_agent,
                black_agent=matchup.black_agent,
                games=stage.games_per_matchup,
                max_plies=stage.max_plies,
                initial_fens=(
                    initial_fen_suite.fen_list()
                    if initial_fen_suite is not None
                    else list(matchup.initial_fens)
                ),
                tags=list(matchup.tags),
            )
            for matchup in filtered_matchups
        ],
        default_games=stage.games_per_matchup,
        default_max_plies=stage.max_plies,
        default_initial_fens=(
            initial_fen_suite.fen_list()
            if initial_fen_suite is not None
            else list(arena_spec.default_initial_fens)
        ),
        round_robin_swap_colors=arena_spec.round_robin_swap_colors,
        include_self_matches=arena_spec.include_self_matches,
        metadata={
            **arena_spec.metadata,
            "curriculum_plan": plan.name,
            "curriculum_stage": stage.name,
            "stage_tags": list(stage.tags),
            "initial_fen_suite": stage.initial_fen_suite,
            "initial_fen_count": len(initial_fen_suite.entries) if initial_fen_suite is not None else len(arena_spec.default_initial_fens),
        },
    )


def build_phase9_expanded_curriculum_plan(
    *,
    repo_root: Path,
    source_arena_summary_path: Path,
    corpus_suite_manifest_path: Path,
    plan_name: str = "phase9_active_experimental_expanded_v1",
    probe_replay_buffer_output_root: str = "artifacts/phase9/replay_buffer_active_expanded_probe_v1",
    expanded_replay_buffer_output_root: str = "artifacts/phase9/replay_buffer_active_experimental_expanded_v1",
    expanded_initial_fen_suite: str | None = None,
    expanded_games_per_matchup: int = 2,
    expanded_max_plies: int = 64,
) -> SelfplayCurriculumPlan:
    """Build the repo-preferred 400k-ready curriculum plan for active and experimental arms."""
    arena_summary = json.loads(source_arena_summary_path.read_text(encoding="utf-8"))
    corpus_manifest = json.loads(corpus_suite_manifest_path.read_text(encoding="utf-8"))
    available_tiers = sorted(dict(corpus_manifest["tiers"]).keys())
    required_tiers = ["pgn_10k", "merged_unique_122k", "unique_pi_400k"]
    missing_tiers = [tier for tier in required_tiers if tier not in available_tiers]
    if missing_tiers:
        raise ValueError(
            f"curriculum plan requires missing corpus tiers: {', '.join(missing_tiers)}"
        )

    planner_runs = [
        PlannerRunSpec(
            name="planner_set_v2_expanded_v1",
            config_path="python/configs/phase8_planner_corpus_suite_set_v2_expanded_v1.json",
            expected_agent_spec="python/configs/phase9_agent_planner_set_v2_expanded_v1.json",
            required_tiers=required_tiers,
            tags=["active", "expanded_400k"],
        ),
        PlannerRunSpec(
            name="planner_set_v6_expanded_v1",
            config_path="python/configs/phase8_planner_corpus_suite_set_v6_expanded_v1.json",
            expected_agent_spec="python/configs/phase9_agent_planner_set_v6_expanded_v1.json",
            required_tiers=required_tiers,
            tags=["experimental", "expanded_400k"],
        ),
        PlannerRunSpec(
            name="planner_set_v6_margin_expanded_v1",
            config_path="python/configs/phase8_planner_corpus_suite_set_v6_margin_expanded_v1.json",
            expected_agent_spec="python/configs/phase9_agent_planner_set_v6_margin_expanded_v1.json",
            required_tiers=required_tiers,
            tags=["experimental", "expanded_400k"],
        ),
        PlannerRunSpec(
            name="planner_set_v6_rank_expanded_v1",
            config_path="python/configs/phase8_planner_corpus_suite_set_v6_rank_expanded_v1.json",
            expected_agent_spec="python/configs/phase9_agent_planner_set_v6_rank_expanded_v1.json",
            required_tiers=required_tiers,
            tags=["experimental", "expanded_400k"],
        ),
        PlannerRunSpec(
            name="planner_recurrent_expanded_v1",
            config_path="python/configs/phase8_planner_corpus_suite_recurrent_v1_expanded_v1.json",
            expected_agent_spec="python/configs/phase9_agent_planner_recurrent_expanded_v1.json",
            required_tiers=required_tiers,
            tags=["experimental", "expanded_400k"],
        ),
    ]

    active_probe_agents = [
        "python/configs/phase9_agent_symbolic_root_v1.json",
        "python/configs/phase9_agent_planner_set_v2_expanded_v1.json",
    ]
    expanded_agents = [
        "python/configs/phase9_agent_symbolic_root_v1.json",
        "python/configs/phase9_agent_planner_set_v2_expanded_v1.json",
        "python/configs/phase9_agent_planner_set_v6_expanded_v1.json",
        "python/configs/phase9_agent_planner_set_v6_margin_expanded_v1.json",
        "python/configs/phase9_agent_planner_set_v6_rank_expanded_v1.json",
        "python/configs/phase9_agent_planner_recurrent_expanded_v1.json",
    ]

    stages = [
        SelfplayCurriculumStage(
            name="active_expanded_probe",
            arena_spec="python/configs/phase9_arena_active_probe_v1.json",
            agent_specs=active_probe_agents,
            games_per_matchup=2,
            max_plies=24,
            replay_buffer_output_root=probe_replay_buffer_output_root,
            agent_sampling_weights=_agent_sampling_weights(
                repo_root=repo_root,
                source_arena_summary=arena_summary,
                agent_spec_paths=active_probe_agents,
            ),
            tags=["probe", "active_only", "expanded_400k"],
        ),
        SelfplayCurriculumStage(
            name="active_experimental_expanded_round_robin",
            arena_spec="python/configs/phase9_arena_active_experimental_expanded_v1.json",
            agent_specs=expanded_agents,
            games_per_matchup=expanded_games_per_matchup,
            max_plies=expanded_max_plies,
            replay_buffer_output_root=expanded_replay_buffer_output_root,
            agent_sampling_weights=_agent_sampling_weights(
                repo_root=repo_root,
                source_arena_summary=arena_summary,
                agent_spec_paths=expanded_agents,
            ),
            initial_fen_suite=expanded_initial_fen_suite,
            tags=["full_suite", "active_plus_experimental", "expanded_400k"],
        ),
    ]

    return SelfplayCurriculumPlan(
        name=plan_name,
        corpus_suite_manifest=str(corpus_suite_manifest_path),
        source_arena_summary=str(source_arena_summary_path),
        planner_runs=planner_runs,
        stages=stages,
        metadata={
            "required_tiers": required_tiers,
            "available_tiers": available_tiers,
        },
    )


def _agent_sampling_weights(
    *,
    repo_root: Path,
    source_arena_summary: dict[str, object],
    agent_spec_paths: Sequence[str],
) -> dict[str, float]:
    standings = dict(source_arena_summary.get("standings") or {})
    weights: dict[str, float] = {}
    for raw_path in agent_spec_paths:
        spec = load_selfplay_agent_spec(_resolve_repo_path(repo_root, raw_path))
        base_weight = _base_weight(spec.tags)
        standing = standings.get(spec.name)
        if isinstance(standing, dict):
            games = max(int(standing.get("games", 0)), 1)
            score_rate = float(standing.get("score", 0.0)) / games
            weight = base_weight * (0.75 + score_rate)
        else:
            weight = base_weight
        weights[raw_path] = round(weight, 3)
    return weights


def _base_weight(tags: Sequence[str]) -> float:
    for tag_name, weight in _BASE_TAG_WEIGHTS.items():
        if tag_name in tags:
            return weight
    return 0.5


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path
