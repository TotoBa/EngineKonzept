"""Versioned selfplay-curriculum plans over planner runs, arena suites, and replay buffers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence

from train.eval.agent_spec import load_selfplay_agent_spec


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
            "tags": list(self.tags),
        }


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


def write_selfplay_curriculum_plan(path: Path, plan: SelfplayCurriculumPlan) -> None:
    """Write a curriculum plan as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_phase9_expanded_curriculum_plan(
    *,
    repo_root: Path,
    source_arena_summary_path: Path,
    corpus_suite_manifest_path: Path,
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
            replay_buffer_output_root="artifacts/phase9/replay_buffer_active_expanded_probe_v1",
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
            games_per_matchup=2,
            max_plies=64,
            replay_buffer_output_root="artifacts/phase9/replay_buffer_active_experimental_expanded_v1",
            agent_sampling_weights=_agent_sampling_weights(
                repo_root=repo_root,
                source_arena_summary=arena_summary,
                agent_spec_paths=expanded_agents,
            ),
            tags=["full_suite", "active_plus_experimental", "expanded_400k"],
        ),
    ]

    return SelfplayCurriculumPlan(
        name="phase9_active_experimental_expanded_v1",
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
