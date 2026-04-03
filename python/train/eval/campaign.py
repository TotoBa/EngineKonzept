"""Versioned long-run selfplay replay campaigns over arena, replay, and planner reruns."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from train.config import PlannerTrainConfig
from train.datasets import (
    build_planner_head_examples_from_replay,
    build_planner_replay_examples,
    build_replay_buffer_entries_from_sessions,
    load_arena_session_paths,
    load_replay_buffer_entries,
    planner_head_artifact_name,
    planner_replay_artifact_name,
    planner_replay_summary,
    replay_buffer_summary,
    write_planner_head_artifact,
    write_planner_replay_artifact,
    write_replay_buffer_artifact,
)
from train.eval.arena import SelfplayArenaSpec, run_selfplay_arena
from train.eval.curriculum import (
    build_curriculum_stage_arena_spec,
    load_selfplay_curriculum_plan,
)
from train.eval.matrix import build_selfplay_arena_matrix
from train.eval.selfplay import SelfplaySessionRecord
from train.trainers import evaluate_planner_checkpoint, train_planner


SELFPLAY_REPLAY_CAMPAIGN_VERSION = 1


@dataclass(frozen=True)
class PlannerReplayCampaignRunSpec:
    """One replay retrain arm inside a long selfplay campaign."""

    name: str
    base_config_path: str
    compare_name: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("campaign run name must be non-empty")
        if not self.base_config_path:
            raise ValueError("campaign run base_config_path must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "base_config_path": self.base_config_path,
            "compare_name": self.compare_name,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PlannerReplayCampaignRunSpec":
        return cls(
            name=str(payload["name"]),
            base_config_path=str(payload["base_config_path"]),
            compare_name=(
                str(payload["compare_name"])
                if payload.get("compare_name") is not None
                else None
            ),
            tags=[str(value) for value in list(payload.get("tags") or [])],
        )


@dataclass(frozen=True)
class SelfplayReplayCampaignSpec:
    """Versioned long-run contract for replay-aware selfplay reruns."""

    name: str
    output_root: str
    curriculum_plan: str
    stage_name: str
    proposer_checkpoint: str
    dynamics_checkpoint: str | None = None
    opponent_mode: str = "none"
    opponent_checkpoint: str | None = None
    root_top_k: int = 4
    replay_split: str = "train"
    verify_dataset_paths: tuple[str, ...] = ()
    baseline_metrics: dict[str, str] = field(default_factory=dict)
    reference_run_name: str | None = None
    planner_runs: tuple[PlannerReplayCampaignRunSpec, ...] = ()
    spec_version: int = SELFPLAY_REPLAY_CAMPAIGN_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("campaign name must be non-empty")
        if self.spec_version != SELFPLAY_REPLAY_CAMPAIGN_VERSION:
            raise ValueError(f"unsupported campaign version: {self.spec_version}")
        if not self.output_root:
            raise ValueError("campaign output_root must be non-empty")
        if not self.curriculum_plan:
            raise ValueError("campaign curriculum_plan must be non-empty")
        if not self.stage_name:
            raise ValueError("campaign stage_name must be non-empty")
        if not self.proposer_checkpoint:
            raise ValueError("campaign proposer_checkpoint must be non-empty")
        if self.opponent_mode not in {"none", "symbolic", "learned"}:
            raise ValueError("campaign opponent_mode must be 'none', 'symbolic', or 'learned'")
        if self.root_top_k <= 0:
            raise ValueError("campaign root_top_k must be positive")
        if not self.planner_runs:
            raise ValueError("campaign must include at least one planner run")
        if not self.verify_dataset_paths:
            raise ValueError("campaign must include at least one verify dataset path")

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_version": self.spec_version,
            **asdict(self),
            "planner_runs": [run.to_dict() for run in self.planner_runs],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SelfplayReplayCampaignSpec":
        return cls(
            spec_version=int(payload.get("spec_version", SELFPLAY_REPLAY_CAMPAIGN_VERSION)),
            name=str(payload["name"]),
            output_root=str(payload["output_root"]),
            curriculum_plan=str(payload["curriculum_plan"]),
            stage_name=str(payload["stage_name"]),
            proposer_checkpoint=str(payload["proposer_checkpoint"]),
            dynamics_checkpoint=(
                str(payload["dynamics_checkpoint"])
                if payload.get("dynamics_checkpoint") is not None
                else None
            ),
            opponent_mode=str(payload.get("opponent_mode", "none")),
            opponent_checkpoint=(
                str(payload["opponent_checkpoint"])
                if payload.get("opponent_checkpoint") is not None
                else None
            ),
            root_top_k=int(payload.get("root_top_k", 4)),
            replay_split=str(payload.get("replay_split", "train")),
            verify_dataset_paths=tuple(
                str(path) for path in list(payload.get("verify_dataset_paths") or [])
            ),
            baseline_metrics={
                str(name): str(path)
                for name, path in dict(payload.get("baseline_metrics") or {}).items()
            },
            reference_run_name=(
                str(payload["reference_run_name"])
                if payload.get("reference_run_name") is not None
                else None
            ),
            planner_runs=tuple(
                PlannerReplayCampaignRunSpec.from_dict(dict(run))
                for run in list(payload["planner_runs"])
            ),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "SelfplayReplayCampaignSpec":
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("campaign spec must be a JSON object")
        return cls.from_dict(payload)


def load_selfplay_replay_campaign_spec(path: Path) -> SelfplayReplayCampaignSpec:
    """Load a replay campaign spec from JSON."""
    return SelfplayReplayCampaignSpec.from_json(path.read_text(encoding="utf-8"))


def write_selfplay_replay_campaign_spec(
    path: Path,
    spec: SelfplayReplayCampaignSpec,
) -> None:
    """Write a replay campaign spec to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_planner_verify_matrix(
    *,
    campaign_name: str,
    run_metrics: Mapping[str, Mapping[str, float | int]],
    reference_run_name: str | None = None,
) -> dict[str, Any]:
    """Build a ranking/delta payload from multiple planner verify metrics."""
    runs = {
        str(name): dict(metrics)
        for name, metrics in sorted(run_metrics.items())
    }
    ranking_by_top1 = _ranking(
        runs,
        metric_name="root_top1_accuracy",
        secondary_metric="teacher_root_mean_reciprocal_rank",
    )
    ranking_by_mrr = _ranking(
        runs,
        metric_name="teacher_root_mean_reciprocal_rank",
        secondary_metric="root_top1_accuracy",
    )
    ranking_by_probability = _ranking(
        runs,
        metric_name="teacher_root_mean_probability",
        secondary_metric="root_top1_accuracy",
    )
    deltas_vs_reference = {}
    if reference_run_name is not None and reference_run_name in runs:
        reference = runs[reference_run_name]
        for name, metrics in runs.items():
            if name == reference_run_name:
                continue
            deltas_vs_reference[name] = {
                metric_name: round(
                    float(metrics.get(metric_name, 0.0)) - float(reference.get(metric_name, 0.0)),
                    6,
                )
                for metric_name in (
                    "root_top1_accuracy",
                    "root_top3_accuracy",
                    "teacher_root_mean_reciprocal_rank",
                    "teacher_root_mean_probability",
                )
            }
    return {
        "campaign_name": campaign_name,
        "reference_run": reference_run_name,
        "runs": runs,
        "ranking_by_top1": ranking_by_top1,
        "ranking_by_mrr": ranking_by_mrr,
        "ranking_by_probability": ranking_by_probability,
        "deltas_vs_reference": deltas_vs_reference,
    }


def materialize_replay_campaign_planner_config(
    *,
    base_config: PlannerTrainConfig,
    replay_head_train_path: Path,
    output_root: Path,
    run_name: str,
) -> dict[str, Any]:
    """Return a reproducible planner config payload for one campaign run."""
    payload = base_config.to_dict()
    payload["output_dir"] = str(output_root / "planner_runs" / run_name)
    payload["export"]["bundle_dir"] = str(output_root / "planner_models" / run_name)
    payload["data"]["train_path"] = str(replay_head_train_path)
    return payload


def run_selfplay_replay_campaign(
    *,
    spec: SelfplayReplayCampaignSpec,
    repo_root: Path,
    games_per_matchup_override: int | None = None,
    max_plies_override: int | None = None,
    max_replay_examples: int | None = None,
    max_replay_head_examples: int | None = None,
    selected_runs: Sequence[str] | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Run the full replay campaign and return the campaign summary."""
    plan = load_selfplay_curriculum_plan(_resolve_repo_path(repo_root, spec.curriculum_plan))
    arena_spec = build_curriculum_stage_arena_spec(
        repo_root=repo_root,
        plan=plan,
        stage_name=spec.stage_name,
    )
    if games_per_matchup_override is not None or max_plies_override is not None:
        arena_spec = _override_arena_spec(
            arena_spec,
            games_per_matchup=games_per_matchup_override,
            max_plies=max_plies_override,
        )

    output_root = _resolve_repo_path(repo_root, spec.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    arena_root = output_root / "arena"
    arena_summary_path = arena_root / "summary.json"
    if not skip_existing or not arena_summary_path.exists():
        arena_root.mkdir(parents=True, exist_ok=True)
        resolved_path = arena_root / "arena_spec.resolved.json"
        resolved_path.write_text(
            json.dumps(arena_spec.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        arena_summary = run_selfplay_arena(
            spec=arena_spec,
            repo_root=repo_root,
            output_root=arena_root,
        )
        arena_summary_payload = {
            "curriculum_plan": str(_resolve_repo_path(repo_root, spec.curriculum_plan)),
            "stage_name": spec.stage_name,
            "resolved_arena_spec": str(resolved_path),
            **arena_summary,
        }
        arena_summary_path.write_text(
            json.dumps(arena_summary_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    arena_summary_payload = json.loads(arena_summary_path.read_text(encoding="utf-8"))

    arena_matrix = build_selfplay_arena_matrix(arena_summary_payload)
    arena_matrix_path = output_root / "arena_matrix.json"
    arena_matrix_path.write_text(json.dumps(arena_matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    replay_root = output_root / "replay_buffer"
    replay_root.mkdir(parents=True, exist_ok=True)
    session_paths = load_arena_session_paths(arena_summary_path)
    sessions = [
        SelfplaySessionRecord.from_json(path.read_text(encoding="utf-8"))
        for path in session_paths
    ]
    replay_entries = build_replay_buffer_entries_from_sessions(
        sessions,
        session_labels=[path.stem for path in session_paths],
    )
    replay_path = replay_root / "replay_buffer.jsonl"
    write_replay_buffer_artifact(replay_path, replay_entries)
    replay_summary_payload = {
        "arena_summary": str(arena_summary_path),
        "session_count": len(session_paths),
        "replay_path": str(replay_path),
        "summary": replay_buffer_summary(replay_entries),
    }
    replay_summary_path = replay_root / "summary.json"
    replay_summary_path.write_text(
        json.dumps(replay_summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    replay_supervision_root = output_root / "planner_replay"
    replay_supervision_root.mkdir(parents=True, exist_ok=True)
    replay_examples = build_planner_replay_examples(
        load_replay_buffer_entries(replay_path),
        split=spec.replay_split,
        include_unfinished=False,
        max_examples=max_replay_examples,
    )
    replay_supervision_path = replay_supervision_root / planner_replay_artifact_name(spec.replay_split)
    write_planner_replay_artifact(replay_supervision_path, replay_examples)
    replay_supervision_summary = {
        "replay_path": str(replay_path),
        "artifact_path": str(replay_supervision_path),
        "summary": planner_replay_summary(replay_examples),
    }
    replay_supervision_summary_path = replay_supervision_root / "summary.json"
    replay_supervision_summary_path.write_text(
        json.dumps(replay_supervision_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    replay_head_root = output_root / "planner_replay_head"
    replay_head_root.mkdir(parents=True, exist_ok=True)
    replay_head_examples = build_planner_head_examples_from_replay(
        planner_replay_path=replay_supervision_path,
        proposer_checkpoint=_resolve_repo_path(repo_root, Path(spec.proposer_checkpoint)),
        dynamics_checkpoint=(
            _resolve_repo_path(repo_root, Path(spec.dynamics_checkpoint))
            if spec.dynamics_checkpoint is not None
            else None
        ),
        opponent_mode=spec.opponent_mode,
        opponent_checkpoint=(
            _resolve_repo_path(repo_root, Path(spec.opponent_checkpoint))
            if spec.opponent_checkpoint is not None
            else None
        ),
        root_top_k=spec.root_top_k,
        max_examples=max_replay_head_examples,
        repo_root=repo_root,
    )
    replay_head_train_path = replay_head_root / planner_head_artifact_name("train")
    write_planner_head_artifact(replay_head_train_path, replay_head_examples)
    replay_head_summary_path = replay_head_root / "summary.json"
    replay_head_summary_payload = {
        "planner_replay_path": str(replay_supervision_path),
        "artifact_path": str(replay_head_train_path),
        "example_count": len(replay_head_examples),
        "root_top_k": spec.root_top_k,
        "opponent_mode": spec.opponent_mode,
        "mean_candidate_count": round(
            sum(len(example.candidate_action_indices) for example in replay_head_examples)
            / len(replay_head_examples),
            6,
        )
        if replay_head_examples
        else 0.0,
    }
    replay_head_summary_path.write_text(
        json.dumps(replay_head_summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected_run_names = set(selected_runs) if selected_runs is not None else None
    run_results: dict[str, dict[str, Any]] = {}
    for run_spec in spec.planner_runs:
        if selected_run_names is not None and run_spec.name not in selected_run_names:
            continue
        base_config_path = _resolve_repo_path(repo_root, Path(run_spec.base_config_path))
        base_config = PlannerTrainConfig.from_dict(
            json.loads(base_config_path.read_text(encoding="utf-8"))
        )
        resolved_config_payload = materialize_replay_campaign_planner_config(
            base_config=base_config,
            replay_head_train_path=replay_head_train_path,
            output_root=output_root,
            run_name=run_spec.name,
        )
        resolved_config_path = output_root / "resolved_configs" / f"{run_spec.name}.json"
        resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_config_path.write_text(
            json.dumps(resolved_config_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resolved_config = PlannerTrainConfig.from_dict(resolved_config_payload)
        checkpoint_path = Path(resolved_config.export.bundle_dir) / resolved_config.export.checkpoint_name
        if not skip_existing or not checkpoint_path.exists():
            training_run = train_planner(resolved_config, repo_root=repo_root)
            training_summary = training_run.to_dict()
        else:
            training_summary_path = Path(resolved_config.output_dir) / "summary.json"
            training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        verify_metrics = evaluate_planner_checkpoint(
            checkpoint_path,
            dataset_paths=[
                _resolve_repo_path(repo_root, Path(path))
                for path in spec.verify_dataset_paths
            ],
            top_k=resolved_config.evaluation.top_k,
        ).to_dict()
        verify_path = output_root / "planner_verify" / f"{run_spec.name}.json"
        verify_path.parent.mkdir(parents=True, exist_ok=True)
        verify_path.write_text(json.dumps(verify_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run_results[run_spec.compare_name or run_spec.name] = {
            "run_name": run_spec.name,
            "resolved_config_path": str(resolved_config_path),
            "training_summary": training_summary,
            "verify_metrics_path": str(verify_path),
            "verify_metrics": verify_metrics,
            "tags": list(run_spec.tags),
        }

    compare_runs = {
        name: json.loads(_resolve_repo_path(repo_root, Path(path)).read_text(encoding="utf-8"))
        for name, path in spec.baseline_metrics.items()
    }
    for compare_name, payload in run_results.items():
        compare_runs[compare_name] = dict(payload["verify_metrics"])
    verify_matrix = build_planner_verify_matrix(
        campaign_name=spec.name,
        run_metrics=compare_runs,
        reference_run_name=spec.reference_run_name,
    )
    verify_matrix_path = output_root / "planner_verify_matrix.json"
    verify_matrix_path.write_text(
        json.dumps(verify_matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "campaign_name": spec.name,
        "campaign_version": spec.spec_version,
        "curriculum_plan": str(_resolve_repo_path(repo_root, Path(spec.curriculum_plan))),
        "stage_name": spec.stage_name,
        "output_root": str(output_root),
        "arena_summary_path": str(arena_summary_path),
        "arena_matrix_path": str(arena_matrix_path),
        "replay_summary_path": str(replay_summary_path),
        "planner_replay_summary_path": str(replay_supervision_summary_path),
        "planner_replay_head_summary_path": str(replay_head_summary_path),
        "verify_matrix_path": str(verify_matrix_path),
        "run_results": run_results,
        "baseline_metrics": spec.baseline_metrics,
        "reference_run_name": spec.reference_run_name,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _ranking(
    runs: Mapping[str, Mapping[str, float | int]],
    *,
    metric_name: str,
    secondary_metric: str,
) -> list[dict[str, float | str]]:
    ranking = [
        {
            "name": name,
            metric_name: round(float(metrics.get(metric_name, 0.0)), 6),
            secondary_metric: round(float(metrics.get(secondary_metric, 0.0)), 6),
        }
        for name, metrics in runs.items()
    ]
    ranking.sort(
        key=lambda record: (
            -float(record[metric_name]),
            -float(record[secondary_metric]),
            str(record["name"]),
        )
    )
    return ranking


def _override_arena_spec(
    spec: SelfplayArenaSpec,
    *,
    games_per_matchup: int | None,
    max_plies: int | None,
) -> SelfplayArenaSpec:
    matchups = []
    for matchup in spec.matchups:
        matchups.append(
            type(matchup)(
                white_agent=matchup.white_agent,
                black_agent=matchup.black_agent,
                games=games_per_matchup or matchup.games,
                max_plies=max_plies or matchup.max_plies,
                initial_fens=list(matchup.initial_fens),
                tags=list(matchup.tags),
            )
        )
    return SelfplayArenaSpec(
        name=spec.name,
        agent_specs=dict(spec.agent_specs),
        schedule_mode=spec.schedule_mode,
        matchups=matchups,
        default_games=games_per_matchup or spec.default_games,
        default_max_plies=max_plies or spec.default_max_plies,
        default_initial_fens=list(spec.default_initial_fens),
        round_robin_swap_colors=spec.round_robin_swap_colors,
        include_self_matches=spec.include_self_matches,
        metadata=dict(spec.metadata),
        spec_version=spec.spec_version,
    )


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path
