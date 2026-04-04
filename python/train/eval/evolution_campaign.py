"""Versioned long-run planner evolution campaign over fulltrain, arena, and selfplay retraining."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from train.config import PlannerTrainConfig
from train.datasets import (
    build_planner_head_examples_from_selfplay_teacher_reviews,
    build_selfplay_teacher_review_examples,
    filter_planner_head_examples,
    load_planner_head_examples,
    planner_head_artifact_name,
    selfplay_teacher_review_artifact_name,
    selfplay_teacher_review_summary,
    write_planner_head_artifact,
    write_selfplay_teacher_review_artifact,
)
from train.eval.agent_spec import (
    SelfplayAgentSpec,
    load_selfplay_agent_spec,
    write_selfplay_agent_spec,
)
from train.eval.arena import load_selfplay_arena_spec, run_selfplay_arena
from train.eval.campaign import build_planner_verify_matrix
from train.eval.fulltrain_campaign import (
    materialize_fulltrain_arena_spec,
    materialize_fulltrain_planner_config,
    resolve_trained_agent_spec,
)
from train.eval.initial_fens import load_selfplay_initial_fen_suite
from train.eval.matrix import build_selfplay_arena_matrix, write_selfplay_arena_matrix
from train.trainers import evaluate_planner_checkpoint, train_planner


PLANNER_EVOLUTION_CAMPAIGN_VERSION = 1


@dataclass(frozen=True)
class PlannerEvolutionRunSpec:
    """One trainable planner arm inside the evolution campaign."""

    name: str
    base_config_path: str
    agent_template_spec_path: str
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evolution run name must be non-empty")
        if not self.base_config_path:
            raise ValueError("evolution run base_config_path must be non-empty")
        if not self.agent_template_spec_path:
            raise ValueError("evolution run agent_template_spec_path must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "base_config_path": self.base_config_path,
            "agent_template_spec_path": self.agent_template_spec_path,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PlannerEvolutionRunSpec":
        return cls(
            name=str(payload["name"]),
            base_config_path=str(payload["base_config_path"]),
            agent_template_spec_path=str(payload["agent_template_spec_path"]),
            tags=[str(value) for value in list(payload.get("tags") or [])],
        )


@dataclass(frozen=True)
class PlannerEvolutionCampaignSpec:
    """Versioned spec for start -> fulltrain -> iterative selfplay planner evolution."""

    name: str
    output_root: str
    source_workflow_summary: str
    filtered_workflow_root: str
    training_tiers: tuple[str, ...]
    verify_tiers: tuple[str, ...]
    filtered_training_tiers: tuple[str, ...] = ()
    filter_max_abs_root_value_cp: float = 2000.0
    filter_ambiguous_score_span_cp: float = 5.0
    filter_min_candidate_count: int = 2
    training_epochs: int = 12
    iterations: int = 20
    arena_template_spec_path: str = ""
    initial_fen_suite_path: str | None = None
    arena_default_games: int = 1
    arena_parallel_workers: int = 6
    arena_default_max_plies: int | None = None
    arena_opening_selection_seed: int | None = None
    benchmark_agent_specs: dict[str, str] = field(default_factory=dict)
    arena_agent_order: tuple[str, ...] = ()
    reference_run_name: str | None = None
    planner_runs: tuple[PlannerEvolutionRunSpec, ...] = ()
    teacher_engine_path: str = "/usr/games/stockfish18"
    teacher_depth: int | None = 5
    teacher_nodes: int | None = None
    teacher_movetime_ms: int | None = None
    policy_temperature_cp: float = 64.0
    mistake_deadzone_cp: float = 8.0
    mistake_priority_scale_cp: float = 64.0
    max_mistake_priority: float = 4.0
    max_review_examples_per_agent: int | None = None
    max_head_examples_per_agent: int | None = None
    retrain_epochs: int = 1
    retrain_learning_rate: float | None = 2e-4
    retrain_batch_size: int | None = 128
    retrain_include_non_mistakes: bool = False
    retrain_retain_base_train_paths: bool = True
    skip_missing_start_agents: bool = True
    spec_version: int = PLANNER_EVOLUTION_CAMPAIGN_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("campaign name must be non-empty")
        if self.spec_version != PLANNER_EVOLUTION_CAMPAIGN_VERSION:
            raise ValueError(f"unsupported evolution campaign version: {self.spec_version}")
        if not self.output_root:
            raise ValueError("campaign output_root must be non-empty")
        if not self.source_workflow_summary:
            raise ValueError("campaign source_workflow_summary must be non-empty")
        if not self.filtered_workflow_root:
            raise ValueError("campaign filtered_workflow_root must be non-empty")
        if not self.training_tiers:
            raise ValueError("campaign training_tiers must be non-empty")
        if not self.verify_tiers:
            raise ValueError("campaign verify_tiers must be non-empty")
        if self.training_epochs <= 0:
            raise ValueError("campaign training_epochs must be positive")
        if self.iterations <= 0:
            raise ValueError("campaign iterations must be positive")
        if not self.arena_template_spec_path:
            raise ValueError("campaign arena_template_spec_path must be non-empty")
        if self.arena_default_games <= 0:
            raise ValueError("campaign arena_default_games must be positive")
        if self.arena_parallel_workers <= 0:
            raise ValueError("campaign arena_parallel_workers must be positive")
        if self.arena_default_max_plies is not None and self.arena_default_max_plies <= 0:
            raise ValueError("campaign arena_default_max_plies must be positive when provided")
        if self.filter_max_abs_root_value_cp <= 0.0:
            raise ValueError("campaign filter_max_abs_root_value_cp must be positive")
        if self.filter_ambiguous_score_span_cp < 0.0:
            raise ValueError("campaign filter_ambiguous_score_span_cp must be non-negative")
        if self.filter_min_candidate_count <= 0:
            raise ValueError("campaign filter_min_candidate_count must be positive")
        if not self.planner_runs:
            raise ValueError("campaign must include at least one planner run")
        if not self.teacher_engine_path:
            raise ValueError("campaign teacher_engine_path must be non-empty")
        if (
            self.teacher_depth is None
            and self.teacher_nodes is None
            and self.teacher_movetime_ms is None
        ):
            raise ValueError("campaign requires one of teacher_depth, teacher_nodes, or teacher_movetime_ms")
        if self.teacher_depth is not None and self.teacher_depth <= 0:
            raise ValueError("campaign teacher_depth must be positive when provided")
        if self.teacher_nodes is not None and self.teacher_nodes <= 0:
            raise ValueError("campaign teacher_nodes must be positive when provided")
        if self.teacher_movetime_ms is not None and self.teacher_movetime_ms <= 0:
            raise ValueError("campaign teacher_movetime_ms must be positive when provided")
        if self.policy_temperature_cp <= 0.0:
            raise ValueError("campaign policy_temperature_cp must be positive")
        if self.mistake_deadzone_cp < 0.0:
            raise ValueError("campaign mistake_deadzone_cp must be non-negative")
        if self.mistake_priority_scale_cp <= 0.0:
            raise ValueError("campaign mistake_priority_scale_cp must be positive")
        if self.max_mistake_priority <= 0.0:
            raise ValueError("campaign max_mistake_priority must be positive")
        if self.max_review_examples_per_agent is not None and self.max_review_examples_per_agent <= 0:
            raise ValueError("campaign max_review_examples_per_agent must be positive when provided")
        if self.max_head_examples_per_agent is not None and self.max_head_examples_per_agent <= 0:
            raise ValueError("campaign max_head_examples_per_agent must be positive when provided")
        if self.retrain_epochs <= 0:
            raise ValueError("campaign retrain_epochs must be positive")
        if self.retrain_learning_rate is not None and self.retrain_learning_rate <= 0.0:
            raise ValueError("campaign retrain_learning_rate must be positive when provided")
        if self.retrain_batch_size is not None and self.retrain_batch_size <= 0:
            raise ValueError("campaign retrain_batch_size must be positive when provided")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["planner_runs"] = [run.to_dict() for run in self.planner_runs]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PlannerEvolutionCampaignSpec":
        return cls(
            spec_version=int(payload.get("spec_version", PLANNER_EVOLUTION_CAMPAIGN_VERSION)),
            name=str(payload["name"]),
            output_root=str(payload["output_root"]),
            source_workflow_summary=str(payload["source_workflow_summary"]),
            filtered_workflow_root=str(payload["filtered_workflow_root"]),
            training_tiers=tuple(str(value) for value in list(payload["training_tiers"])),
            verify_tiers=tuple(str(value) for value in list(payload["verify_tiers"])),
            filtered_training_tiers=tuple(
                str(value) for value in list(payload.get("filtered_training_tiers") or [])
            ),
            filter_max_abs_root_value_cp=float(payload.get("filter_max_abs_root_value_cp", 2000.0)),
            filter_ambiguous_score_span_cp=float(payload.get("filter_ambiguous_score_span_cp", 5.0)),
            filter_min_candidate_count=int(payload.get("filter_min_candidate_count", 2)),
            training_epochs=int(payload.get("training_epochs", 12)),
            iterations=int(payload.get("iterations", 20)),
            arena_template_spec_path=str(payload["arena_template_spec_path"]),
            initial_fen_suite_path=(
                str(payload["initial_fen_suite_path"])
                if payload.get("initial_fen_suite_path") is not None
                else None
            ),
            arena_default_games=int(payload.get("arena_default_games", 1)),
            arena_parallel_workers=int(payload.get("arena_parallel_workers", 6)),
            arena_default_max_plies=(
                int(payload["arena_default_max_plies"])
                if payload.get("arena_default_max_plies") is not None
                else None
            ),
            arena_opening_selection_seed=(
                int(payload["arena_opening_selection_seed"])
                if payload.get("arena_opening_selection_seed") is not None
                else None
            ),
            benchmark_agent_specs={
                str(name): str(path)
                for name, path in dict(payload.get("benchmark_agent_specs") or {}).items()
            },
            arena_agent_order=tuple(
                str(value) for value in list(payload.get("arena_agent_order") or [])
            ),
            reference_run_name=(
                str(payload["reference_run_name"])
                if payload.get("reference_run_name") is not None
                else None
            ),
            planner_runs=tuple(
                PlannerEvolutionRunSpec.from_dict(dict(run))
                for run in list(payload["planner_runs"])
            ),
            teacher_engine_path=str(payload.get("teacher_engine_path", "/usr/games/stockfish18")),
            teacher_depth=_optional_int(payload.get("teacher_depth", 5)),
            teacher_nodes=_optional_int(payload.get("teacher_nodes")),
            teacher_movetime_ms=_optional_int(payload.get("teacher_movetime_ms")),
            policy_temperature_cp=float(payload.get("policy_temperature_cp", 64.0)),
            mistake_deadzone_cp=float(payload.get("mistake_deadzone_cp", 8.0)),
            mistake_priority_scale_cp=float(payload.get("mistake_priority_scale_cp", 64.0)),
            max_mistake_priority=float(payload.get("max_mistake_priority", 4.0)),
            max_review_examples_per_agent=_optional_int(
                payload.get("max_review_examples_per_agent")
            ),
            max_head_examples_per_agent=_optional_int(
                payload.get("max_head_examples_per_agent")
            ),
            retrain_epochs=int(payload.get("retrain_epochs", 1)),
            retrain_learning_rate=_optional_float(payload.get("retrain_learning_rate", 2e-4)),
            retrain_batch_size=_optional_int(payload.get("retrain_batch_size", 128)),
            retrain_include_non_mistakes=bool(payload.get("retrain_include_non_mistakes", False)),
            retrain_retain_base_train_paths=bool(payload.get("retrain_retain_base_train_paths", True)),
            skip_missing_start_agents=bool(payload.get("skip_missing_start_agents", True)),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "PlannerEvolutionCampaignSpec":
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("evolution campaign spec must be a JSON object")
        return cls.from_dict(payload)


def load_planner_evolution_campaign_spec(path: Path) -> PlannerEvolutionCampaignSpec:
    """Load a planner evolution campaign spec from JSON."""
    return PlannerEvolutionCampaignSpec.from_json(path.read_text(encoding="utf-8"))


def write_planner_evolution_campaign_spec(
    path: Path,
    spec: PlannerEvolutionCampaignSpec,
) -> None:
    """Write a planner evolution campaign spec to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_filtered_planner_workflow_summary(
    *,
    source_summary_path: Path,
    output_root: Path,
    filtered_tiers: Sequence[str],
    max_abs_root_value_cp: float,
    ambiguous_score_span_cp: float,
    min_candidate_count: int,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Filter selected workflow tiers for train/validation while keeping verify intact."""
    summary_path = output_root / "summary.json"
    if skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    filtered_summary = json.loads(json.dumps(source_summary))
    tier_payloads = dict(filtered_summary["tiers"])
    filter_report: dict[str, Any] = {}
    filtered_tier_set = set(filtered_tiers)
    for tier_name in filtered_tier_set:
        if tier_name not in tier_payloads:
            raise ValueError(f"unknown workflow tier for filtering: {tier_name}")
        filter_report[tier_name] = {}
        for split_name in ("train", "validation"):
            split_payload = dict(tier_payloads[tier_name][split_name])
            source_path = Path(split_payload["planner_head_path"])
            examples = load_planner_head_examples(source_path)
            kept_examples, filter_summary = filter_planner_head_examples(
                examples,
                max_abs_root_value_cp=max_abs_root_value_cp,
                ambiguous_score_span_cp=ambiguous_score_span_cp,
                min_candidate_count=min_candidate_count,
            )
            output_dir = output_root / f"{tier_name}_{split_name}_v1"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / source_path.name
            write_planner_head_artifact(output_path, kept_examples)
            filter_summary_path = output_dir / "filter_summary.json"
            filter_summary_payload = {
                "source_path": str(source_path),
                "output_path": str(output_path),
                "tier_name": tier_name,
                "split_name": split_name,
                "max_abs_root_value_cp": max_abs_root_value_cp,
                "ambiguous_score_span_cp": ambiguous_score_span_cp,
                "min_candidate_count": min_candidate_count,
                **filter_summary.to_dict(),
            }
            filter_summary_path.write_text(
                json.dumps(filter_summary_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            split_payload["planner_head_path"] = str(output_path)
            split_payload["filter_summary_path"] = str(filter_summary_path)
            split_payload["source_planner_head_path"] = str(source_path)
            if isinstance(split_payload.get("summary"), dict):
                split_payload["summary"] = {
                    **split_payload["summary"],
                    "output_path": str(output_path),
                    "source_planner_head_path": str(source_path),
                    "teacher_quality_filter": {
                        "max_abs_root_value_cp": max_abs_root_value_cp,
                        "ambiguous_score_span_cp": ambiguous_score_span_cp,
                        "min_candidate_count": min_candidate_count,
                        **filter_summary.to_dict(),
                    },
                }
            tier_payloads[tier_name][split_name] = split_payload
            filter_report[tier_name][split_name] = filter_summary_payload

    filtered_summary["tiers"] = tier_payloads
    filtered_summary["source_summary"] = str(source_summary_path)
    filtered_summary["filter_report"] = filter_report
    filtered_summary["train_paths"] = [
        str(path)
        for path in _collect_workflow_split_paths(
            workflow_summary=filtered_summary,
            tiers=tuple(filtered_summary["tiers"].keys()),
            split_name="train",
        )
    ]
    filtered_summary["validation_paths"] = [
        str(path)
        for path in _collect_workflow_split_paths(
            workflow_summary=filtered_summary,
            tiers=tuple(filtered_summary["tiers"].keys()),
            split_name="validation",
        )
    ]
    filtered_summary["verify_paths"] = [
        str(path)
        for path in _collect_workflow_split_paths(
            workflow_summary=filtered_summary,
            tiers=tuple(filtered_summary["tiers"].keys()),
            split_name="verify",
        )
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(filtered_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return filtered_summary


def run_planner_evolution_campaign(
    *,
    spec: PlannerEvolutionCampaignSpec,
    repo_root: Path,
    skip_existing: bool = False,
    iterations_override: int | None = None,
    selected_runs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the full planner-evolution campaign and return the summary payload."""
    output_root = _resolve_repo_path(repo_root, Path(spec.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    source_workflow_summary_path = _resolve_repo_path(repo_root, Path(spec.source_workflow_summary))
    source_workflow_summary = json.loads(source_workflow_summary_path.read_text(encoding="utf-8"))
    filtered_workflow_root = _resolve_repo_path(repo_root, Path(spec.filtered_workflow_root))
    training_workflow_summary = materialize_filtered_planner_workflow_summary(
        source_summary_path=source_workflow_summary_path,
        output_root=filtered_workflow_root,
        filtered_tiers=spec.filtered_training_tiers,
        max_abs_root_value_cp=spec.filter_max_abs_root_value_cp,
        ambiguous_score_span_cp=spec.filter_ambiguous_score_span_cp,
        min_candidate_count=spec.filter_min_candidate_count,
        skip_existing=skip_existing,
    )
    verify_paths = _collect_workflow_split_paths(
        workflow_summary=source_workflow_summary,
        tiers=spec.verify_tiers,
        split_name="verify",
    )
    selected_run_names = set(selected_runs) if selected_runs is not None else None
    iterations = iterations_override or spec.iterations

    start_summary = _run_start_stage(
        spec=spec,
        repo_root=repo_root,
        output_root=output_root,
        verify_paths=verify_paths,
        skip_existing=skip_existing,
        selected_run_names=selected_run_names,
    )

    fulltrain_summary = _run_fulltrain_stage(
        spec=spec,
        repo_root=repo_root,
        output_root=output_root,
        training_workflow_summary=training_workflow_summary,
        verify_paths=verify_paths,
        skip_existing=skip_existing,
        selected_run_names=selected_run_names,
    )

    current_agent_specs = {
        str(name): Path(path)
        for name, path in dict(fulltrain_summary["active_agent_specs"]).items()
    }
    iteration_summaries: list[dict[str, Any]] = []
    for iteration_index in range(1, iterations + 1):
        iteration_summary = _run_iteration_stage(
            spec=spec,
            repo_root=repo_root,
            output_root=output_root,
            training_workflow_summary=training_workflow_summary,
            verify_paths=verify_paths,
            current_agent_specs=current_agent_specs,
            iteration_index=iteration_index,
            skip_existing=skip_existing,
            selected_run_names=selected_run_names,
        )
        current_agent_specs = {
            str(name): Path(path)
            for name, path in dict(iteration_summary["active_agent_specs"]).items()
        }
        iteration_summaries.append(iteration_summary)

    final_summary = _run_final_stage(
        spec=spec,
        repo_root=repo_root,
        output_root=output_root,
        verify_paths=verify_paths,
        current_agent_specs=current_agent_specs,
        skip_existing=skip_existing,
    )

    final_report = build_planner_evolution_report(
        campaign_name=spec.name,
        start_summary=start_summary,
        fulltrain_summary=fulltrain_summary,
        iteration_summaries=iteration_summaries,
        final_summary=final_summary,
    )
    final_report_path = output_root / "final_report.json"
    final_report_path.write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "campaign_name": spec.name,
        "campaign_version": spec.spec_version,
        "source_workflow_summary": str(source_workflow_summary_path),
        "filtered_workflow_summary": str(filtered_workflow_root / "summary.json"),
        "training_tiers": list(spec.training_tiers),
        "verify_tiers": list(spec.verify_tiers),
        "iterations": iterations,
        "start_summary_path": str(output_root / "start" / "summary.json"),
        "fulltrain_summary_path": str(output_root / "after_fulltrain" / "summary.json"),
        "iteration_summary_paths": [
            str(output_root / "iterations" / f"round_{index:02d}" / "summary.json")
            for index in range(1, iterations + 1)
        ],
        "final_summary_path": str(output_root / "final" / "summary.json"),
        "final_report_path": str(final_report_path),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_planner_evolution_report(
    *,
    campaign_name: str,
    start_summary: Mapping[str, object],
    fulltrain_summary: Mapping[str, object],
    iteration_summaries: Sequence[Mapping[str, object]],
    final_summary: Mapping[str, object],
) -> dict[str, Any]:
    """Aggregate the per-stage verify/arena/review signals into one final report."""
    stages: list[tuple[str, Mapping[str, object]]] = [
        ("start", start_summary),
        ("after_fulltrain", fulltrain_summary),
    ]
    stages.extend(
        (f"round_{index:02d}", summary)
        for index, summary in enumerate(iteration_summaries, 1)
    )
    stages.append(("final", final_summary))

    verify_history: dict[str, list[dict[str, Any]]] = {}
    arena_history: dict[str, list[dict[str, Any]]] = {}
    review_history: dict[str, list[dict[str, Any]]] = {}
    best_verify_by_stage: list[dict[str, Any]] = []
    best_arena_by_stage: list[dict[str, Any]] = []

    for stage_name, summary in stages:
        verify_path = _optional_path(summary.get("planner_verify_matrix_path"))
        if verify_path is not None and verify_path.exists():
            verify_payload = json.loads(verify_path.read_text(encoding="utf-8"))
            for agent_name, metrics in dict(verify_payload.get("runs") or {}).items():
                verify_history.setdefault(agent_name, []).append(
                    {
                        "stage": stage_name,
                        "root_top1_accuracy": float(metrics.get("root_top1_accuracy", 0.0)),
                        "root_top3_accuracy": float(metrics.get("root_top3_accuracy", 0.0)),
                        "teacher_root_mean_probability": float(
                            metrics.get("teacher_root_mean_probability", 0.0)
                        ),
                        "teacher_root_mean_reciprocal_rank": float(
                            metrics.get("teacher_root_mean_reciprocal_rank", 0.0)
                        ),
                    }
                )
            ranking = list(verify_payload.get("ranking_by_top1") or [])
            if ranking:
                best_verify_by_stage.append(
                    {
                        "stage": stage_name,
                        "best_agent": str(ranking[0]["name"]),
                        "root_top1_accuracy": float(ranking[0]["metric"]),
                        "teacher_root_mean_reciprocal_rank": float(
                            ranking[0].get("secondary_metric", 0.0)
                        ),
                    }
                )

        arena_path = _optional_path(summary.get("arena_summary_path"))
        if arena_path is not None and arena_path.exists():
            arena_payload = json.loads(arena_path.read_text(encoding="utf-8"))
            standings = dict(arena_payload.get("standings") or {})
            best_name: str | None = None
            best_score_rate = -1.0
            for agent_name, row in standings.items():
                games = int(row.get("games", 0))
                score = float(row.get("score", 0.0))
                score_rate = score / games if games else 0.0
                arena_history.setdefault(agent_name, []).append(
                    {
                        "stage": stage_name,
                        "games": games,
                        "wins": int(row.get("wins", 0)),
                        "losses": int(row.get("losses", 0)),
                        "draws": int(row.get("draws", 0)),
                        "unfinished": int(row.get("unfinished", 0)),
                        "score": score,
                        "score_rate": round(score_rate, 6),
                    }
                )
                if games > 0 and score_rate > best_score_rate:
                    best_name = str(agent_name)
                    best_score_rate = score_rate
            if best_name is not None:
                best_arena_by_stage.append(
                    {
                        "stage": stage_name,
                        "best_agent": best_name,
                        "score_rate": round(best_score_rate, 6),
                    }
                )

        teacher_training = dict(summary.get("teacher_training") or {})
        for agent_name, payload in teacher_training.items():
            review_payload = dict(payload.get("review_summary") or {})
            review_history.setdefault(agent_name, []).append(
                {
                    "stage": stage_name,
                    "example_count": int(review_payload.get("example_count", 0)),
                    "mistake_count": int(review_payload.get("mistake_count", 0)),
                    "mean_mistake_cp": float(review_payload.get("mean_mistake_cp", 0.0)),
                    "mean_mistake_priority": float(
                        review_payload.get("mean_mistake_priority", 0.0)
                    ),
                    "planner_head_example_count": int(payload.get("planner_head_example_count", 0)),
                }
            )

    return {
        "campaign_name": campaign_name,
        "stage_order": [stage_name for stage_name, _ in stages],
        "best_verify_by_stage": best_verify_by_stage,
        "best_arena_by_stage": best_arena_by_stage,
        "verify_history": verify_history,
        "arena_history": arena_history,
        "review_history": review_history,
    }


def _run_start_stage(
    *,
    spec: PlannerEvolutionCampaignSpec,
    repo_root: Path,
    output_root: Path,
    verify_paths: Sequence[Path],
    skip_existing: bool,
    selected_run_names: set[str] | None,
) -> dict[str, Any]:
    stage_root = output_root / "start"
    summary_path = stage_root / "summary.json"
    if skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    available_specs, unavailable_specs = _materialize_start_agent_specs(
        spec=spec,
        repo_root=repo_root,
        output_root=stage_root / "active_agent_specs",
        selected_run_names=selected_run_names,
    )
    verify_metrics = _evaluate_planner_agents(
        agent_spec_paths=available_specs,
        repo_root=repo_root,
        verify_paths=verify_paths,
        top_k=3,
    )
    verify_matrix = build_planner_verify_matrix(
        campaign_name=f"{spec.name}:start",
        run_metrics=verify_metrics,
        reference_run_name=spec.reference_run_name,
    )
    verify_matrix_path = stage_root / "planner_verify_matrix.json"
    verify_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    verify_matrix_path.write_text(
        json.dumps(verify_matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    arena_summary_path, arena_matrix_path = _run_arena_stage(
        stage_name="start",
        spec=spec,
        repo_root=repo_root,
        stage_root=stage_root,
        agent_spec_paths=available_specs,
    )
    summary = {
        "stage_name": "start",
        "active_agent_specs": {name: str(path) for name, path in available_specs.items()},
        "unavailable_agent_specs": unavailable_specs,
        "planner_verify_matrix_path": str(verify_matrix_path),
        "arena_summary_path": str(arena_summary_path),
        "arena_matrix_path": str(arena_matrix_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_fulltrain_stage(
    *,
    spec: PlannerEvolutionCampaignSpec,
    repo_root: Path,
    output_root: Path,
    training_workflow_summary: Mapping[str, object],
    verify_paths: Sequence[Path],
    skip_existing: bool,
    selected_run_names: set[str] | None,
) -> dict[str, Any]:
    stage_root = output_root / "after_fulltrain"
    summary_path = stage_root / "summary.json"
    if skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_results: dict[str, Any] = {}
    active_agent_specs: dict[str, Path] = {}

    benchmark_specs, unavailable_benchmarks = _materialize_agent_specs(
        repo_root=repo_root,
        output_root=stage_root / "active_agent_specs",
        agent_specs=spec.benchmark_agent_specs,
        allow_missing=spec.skip_missing_start_agents,
    )
    active_agent_specs.update(benchmark_specs)

    for run_spec in spec.planner_runs:
        if selected_run_names is not None and run_spec.name not in selected_run_names:
            continue
        base_config_path = _resolve_repo_path(repo_root, Path(run_spec.base_config_path))
        base_config = PlannerTrainConfig.from_dict(
            json.loads(base_config_path.read_text(encoding="utf-8"))
        )
        resolved_payload = materialize_fulltrain_planner_config(
            base_config=base_config,
            workflow_summary=training_workflow_summary,
            training_tiers=spec.training_tiers,
            output_root=stage_root,
            run_name=run_spec.name,
            training_epochs=spec.training_epochs,
        )
        template_spec = load_selfplay_agent_spec(
            _resolve_repo_path(repo_root, Path(run_spec.agent_template_spec_path))
        )
        warm_start_checkpoint = _existing_agent_checkpoint(template_spec, repo_root=repo_root)
        if warm_start_checkpoint is not None:
            resolved_payload["initial_checkpoint"] = str(warm_start_checkpoint)
        resolved_config_path = stage_root / "resolved_configs" / f"{run_spec.name}.json"
        resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_config_path.write_text(
            json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resolved_config = PlannerTrainConfig.from_dict(resolved_payload)
        checkpoint_path = Path(resolved_config.export.bundle_dir) / resolved_config.export.checkpoint_name
        training_summary_path = Path(resolved_config.output_dir) / "summary.json"
        if not skip_existing or not checkpoint_path.exists() or not training_summary_path.exists():
            training_run = train_planner(resolved_config, repo_root=repo_root)
            training_summary = training_run.to_dict()
        else:
            training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        verify_metrics = evaluate_planner_checkpoint(
            checkpoint_path,
            dataset_paths=verify_paths,
            top_k=resolved_config.evaluation.top_k,
        ).to_dict()
        verify_path = stage_root / "planner_verify" / f"{run_spec.name}.json"
        verify_path.parent.mkdir(parents=True, exist_ok=True)
        verify_path.write_text(
            json.dumps(verify_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resolved_agent_spec = resolve_trained_agent_spec(
            template_spec=template_spec,
            agent_name=run_spec.name,
            planner_checkpoint=checkpoint_path,
            run_name=run_spec.name,
        )
        resolved_agent_path = stage_root / "active_agent_specs" / f"{run_spec.name}.json"
        write_selfplay_agent_spec(resolved_agent_path, resolved_agent_spec)
        active_agent_specs[run_spec.name] = resolved_agent_path
        run_results[run_spec.name] = {
            "base_config_path": str(base_config_path),
            "resolved_config_path": str(resolved_config_path),
            "checkpoint_path": str(checkpoint_path),
            "training_summary": training_summary,
            "verify_metrics_path": str(verify_path),
            "verify_metrics": verify_metrics,
            "agent_spec_path": str(resolved_agent_path),
            "warm_start_checkpoint": (
                str(warm_start_checkpoint) if warm_start_checkpoint is not None else None
            ),
            "tags": list(run_spec.tags),
        }

    compare_runs = dict(_evaluate_planner_agents(
        agent_spec_paths=benchmark_specs,
        repo_root=repo_root,
        verify_paths=verify_paths,
        top_k=3,
    ))
    for run_name, payload in run_results.items():
        compare_runs[run_name] = dict(payload["verify_metrics"])
    verify_matrix = build_planner_verify_matrix(
        campaign_name=f"{spec.name}:after_fulltrain",
        run_metrics=compare_runs,
        reference_run_name=spec.reference_run_name,
    )
    verify_matrix_path = stage_root / "planner_verify_matrix.json"
    verify_matrix_path.write_text(
        json.dumps(verify_matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    ordered_specs = _ordered_agent_specs(active_agent_specs, order=spec.arena_agent_order)
    arena_summary_path, arena_matrix_path = _run_arena_stage(
        stage_name="after_fulltrain",
        spec=spec,
        repo_root=repo_root,
        stage_root=stage_root,
        agent_spec_paths=ordered_specs,
    )
    summary = {
        "stage_name": "after_fulltrain",
        "active_agent_specs": {name: str(path) for name, path in ordered_specs.items()},
        "run_results": run_results,
        "unavailable_benchmark_agent_specs": unavailable_benchmarks,
        "planner_verify_matrix_path": str(verify_matrix_path),
        "arena_summary_path": str(arena_summary_path),
        "arena_matrix_path": str(arena_matrix_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_iteration_stage(
    *,
    spec: PlannerEvolutionCampaignSpec,
    repo_root: Path,
    output_root: Path,
    training_workflow_summary: Mapping[str, object],
    verify_paths: Sequence[Path],
    current_agent_specs: Mapping[str, Path],
    iteration_index: int,
    skip_existing: bool,
    selected_run_names: set[str] | None,
) -> dict[str, Any]:
    stage_name = f"round_{iteration_index:02d}"
    stage_root = output_root / "iterations" / stage_name
    summary_path = stage_root / "summary.json"
    if skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    pre_round_specs = _copy_agent_specs(
        agent_spec_paths=current_agent_specs,
        output_root=stage_root / "pre_round_agent_specs",
    )
    ordered_specs = _ordered_agent_specs(pre_round_specs, order=spec.arena_agent_order)
    arena_summary_path, arena_matrix_path = _run_arena_stage(
        stage_name=stage_name,
        spec=spec,
        repo_root=repo_root,
        stage_root=stage_root,
        agent_spec_paths=ordered_specs,
    )

    trainable_agent_names = [
        run_spec.name
        for run_spec in spec.planner_runs
        if (selected_run_names is None or run_spec.name in selected_run_names)
        and run_spec.name in ordered_specs
    ]
    reviews_by_agent = build_selfplay_teacher_review_examples(
        arena_summary_path=arena_summary_path,
        trainable_agent_names=tuple(trainable_agent_names),
        repo_root=repo_root,
        teacher_engine_path=_resolve_repo_path(repo_root, Path(spec.teacher_engine_path)),
        split="train",
        nodes=spec.teacher_nodes,
        depth=spec.teacher_depth,
        movetime_ms=spec.teacher_movetime_ms,
        multipv=0,
        policy_temperature_cp=spec.policy_temperature_cp,
        mistake_deadzone_cp=spec.mistake_deadzone_cp,
        mistake_priority_scale_cp=spec.mistake_priority_scale_cp,
        max_mistake_priority=spec.max_mistake_priority,
        max_examples_per_agent=spec.max_review_examples_per_agent,
    )

    next_active_specs = _copy_agent_specs(
        agent_spec_paths=ordered_specs,
        output_root=stage_root / "active_agent_specs",
    )
    teacher_training: dict[str, Any] = {}
    retrain_results: dict[str, Any] = {}
    for run_spec in spec.planner_runs:
        if selected_run_names is not None and run_spec.name not in selected_run_names:
            continue
        if run_spec.name not in ordered_specs:
            retrain_results[run_spec.name] = {"status": "skipped_unavailable"}
            continue
        current_spec = load_selfplay_agent_spec(ordered_specs[run_spec.name])
        if current_spec.planner_checkpoint is None:
            retrain_results[run_spec.name] = {"status": "skipped_non_planner"}
            continue
        current_checkpoint = _resolve_repo_path(repo_root, Path(current_spec.planner_checkpoint))
        if not current_checkpoint.exists():
            retrain_results[run_spec.name] = {"status": "skipped_missing_checkpoint"}
            continue

        review_examples = reviews_by_agent.get(run_spec.name, [])
        agent_training_root = stage_root / "teacher_training" / run_spec.name
        agent_training_root.mkdir(parents=True, exist_ok=True)
        review_path = agent_training_root / selfplay_teacher_review_artifact_name("train")
        write_selfplay_teacher_review_artifact(review_path, review_examples)

        planner_head_examples = build_planner_head_examples_from_selfplay_teacher_reviews(
            review_examples=review_examples,
            proposer_checkpoint=_resolve_repo_path(repo_root, Path(current_spec.proposer_checkpoint or "")),
            dynamics_checkpoint=(
                _resolve_repo_path(repo_root, Path(current_spec.dynamics_checkpoint))
                if current_spec.dynamics_checkpoint is not None
                else None
            ),
            opponent_mode=current_spec.opponent_mode,
            opponent_checkpoint=(
                _resolve_repo_path(repo_root, Path(current_spec.opponent_checkpoint))
                if current_spec.opponent_checkpoint is not None
                else None
            ),
            root_top_k=current_spec.root_top_k,
            max_examples=spec.max_head_examples_per_agent,
            include_non_mistakes=spec.retrain_include_non_mistakes,
            repo_root=repo_root,
        )
        planner_head_path = agent_training_root / planner_head_artifact_name("train")
        write_planner_head_artifact(planner_head_path, planner_head_examples)
        agent_training_summary = {
            "review_path": str(review_path),
            "review_summary": selfplay_teacher_review_summary(review_examples),
            "planner_head_path": str(planner_head_path),
            "planner_head_example_count": len(planner_head_examples),
        }
        (agent_training_root / "summary.json").write_text(
            json.dumps(agent_training_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        teacher_training[run_spec.name] = agent_training_summary

        if not planner_head_examples:
            retrain_results[run_spec.name] = {
                "status": "skipped_empty_training_set",
                "planner_head_example_count": 0,
            }
            continue

        base_config_path = _resolve_repo_path(repo_root, Path(run_spec.base_config_path))
        base_config = PlannerTrainConfig.from_dict(
            json.loads(base_config_path.read_text(encoding="utf-8"))
        )
        resolved_payload = materialize_fulltrain_planner_config(
            base_config=base_config,
            workflow_summary=training_workflow_summary,
            training_tiers=spec.training_tiers,
            output_root=stage_root,
            run_name=run_spec.name,
            training_epochs=spec.retrain_epochs,
        )
        existing_train_paths = [
            str(resolved_payload["data"]["train_path"]),
            *[str(path) for path in resolved_payload["data"].get("additional_train_paths", [])],
        ]
        resolved_payload["initial_checkpoint"] = str(current_checkpoint)
        resolved_payload["output_dir"] = str(stage_root / "planner_runs" / run_spec.name)
        resolved_payload["export"]["bundle_dir"] = str(stage_root / "planner_models" / run_spec.name)
        resolved_payload["data"]["train_path"] = str(planner_head_path)
        resolved_payload["data"]["additional_train_paths"] = (
            existing_train_paths if spec.retrain_retain_base_train_paths else []
        )
        resolved_payload["optimization"]["epochs"] = spec.retrain_epochs
        if spec.retrain_learning_rate is not None:
            resolved_payload["optimization"]["learning_rate"] = spec.retrain_learning_rate
        if spec.retrain_batch_size is not None:
            resolved_payload["optimization"]["batch_size"] = spec.retrain_batch_size
        resolved_config_path = stage_root / "resolved_configs" / f"{run_spec.name}.json"
        resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_config_path.write_text(
            json.dumps(resolved_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resolved_config = PlannerTrainConfig.from_dict(resolved_payload)
        training_run = train_planner(resolved_config, repo_root=repo_root)
        checkpoint_path = Path(resolved_config.export.bundle_dir) / resolved_config.export.checkpoint_name
        updated_spec = replace(
            current_spec,
            planner_checkpoint=str(checkpoint_path),
            metadata={
                **current_spec.metadata,
                "evolution_campaign": spec.name,
                "evolution_round": stage_name,
                "previous_planner_checkpoint": current_spec.planner_checkpoint,
            },
        )
        updated_spec_path = stage_root / "active_agent_specs" / f"{run_spec.name}.json"
        write_selfplay_agent_spec(updated_spec_path, updated_spec)
        next_active_specs[run_spec.name] = updated_spec_path
        retrain_results[run_spec.name] = {
            "status": "trained",
            "resolved_config_path": str(resolved_config_path),
            "checkpoint_path": str(checkpoint_path),
            "planner_head_example_count": len(planner_head_examples),
            "training_summary": training_run.to_dict(),
        }

    verify_metrics = _evaluate_planner_agents(
        agent_spec_paths=next_active_specs,
        repo_root=repo_root,
        verify_paths=verify_paths,
        top_k=3,
    )
    verify_matrix = build_planner_verify_matrix(
        campaign_name=f"{spec.name}:{stage_name}",
        run_metrics=verify_metrics,
        reference_run_name=spec.reference_run_name,
    )
    verify_matrix_path = stage_root / "planner_verify_matrix.json"
    verify_matrix_path.write_text(
        json.dumps(verify_matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "stage_name": stage_name,
        "pre_round_agent_specs": {name: str(path) for name, path in ordered_specs.items()},
        "active_agent_specs": {
            name: str(path) for name, path in _ordered_agent_specs(next_active_specs, order=spec.arena_agent_order).items()
        },
        "arena_summary_path": str(arena_summary_path),
        "arena_matrix_path": str(arena_matrix_path),
        "teacher_training": teacher_training,
        "retrain_results": retrain_results,
        "planner_verify_matrix_path": str(verify_matrix_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_final_stage(
    *,
    spec: PlannerEvolutionCampaignSpec,
    repo_root: Path,
    output_root: Path,
    verify_paths: Sequence[Path],
    current_agent_specs: Mapping[str, Path],
    skip_existing: bool,
) -> dict[str, Any]:
    stage_root = output_root / "final"
    summary_path = stage_root / "summary.json"
    if skip_existing and summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    ordered_specs = _ordered_agent_specs(current_agent_specs, order=spec.arena_agent_order)
    verify_metrics = _evaluate_planner_agents(
        agent_spec_paths=ordered_specs,
        repo_root=repo_root,
        verify_paths=verify_paths,
        top_k=3,
    )
    verify_matrix = build_planner_verify_matrix(
        campaign_name=f"{spec.name}:final",
        run_metrics=verify_metrics,
        reference_run_name=spec.reference_run_name,
    )
    verify_matrix_path = stage_root / "planner_verify_matrix.json"
    verify_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    verify_matrix_path.write_text(
        json.dumps(verify_matrix, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arena_summary_path, arena_matrix_path = _run_arena_stage(
        stage_name="final",
        spec=spec,
        repo_root=repo_root,
        stage_root=stage_root,
        agent_spec_paths=ordered_specs,
    )
    summary = {
        "stage_name": "final",
        "active_agent_specs": {name: str(path) for name, path in ordered_specs.items()},
        "planner_verify_matrix_path": str(verify_matrix_path),
        "arena_summary_path": str(arena_summary_path),
        "arena_matrix_path": str(arena_matrix_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _run_arena_stage(
    *,
    stage_name: str,
    spec: PlannerEvolutionCampaignSpec,
    repo_root: Path,
    stage_root: Path,
    agent_spec_paths: Mapping[str, Path],
) -> tuple[Path, Path]:
    arena_template = load_selfplay_arena_spec(
        _resolve_repo_path(repo_root, Path(spec.arena_template_spec_path))
    )
    initial_fens = list(arena_template.default_initial_fens)
    if spec.initial_fen_suite_path is not None:
        suite = load_selfplay_initial_fen_suite(
            _resolve_repo_path(repo_root, Path(spec.initial_fen_suite_path))
        )
        initial_fens = suite.fen_list()
    resolved_arena_spec = materialize_fulltrain_arena_spec(
        template_spec=arena_template,
        resolved_agent_specs={name: str(path) for name, path in agent_spec_paths.items()},
        default_initial_fens=initial_fens,
        campaign_name=f"{spec.name}:{stage_name}",
        default_games=spec.arena_default_games,
        parallel_workers=spec.arena_parallel_workers,
        default_max_plies=spec.arena_default_max_plies,
        opening_selection_seed=spec.arena_opening_selection_seed,
    )
    arena_root = stage_root / "arena"
    arena_root.mkdir(parents=True, exist_ok=True)
    resolved_arena_spec_path = arena_root / "arena_spec.resolved.json"
    resolved_arena_spec_path.write_text(
        json.dumps(resolved_arena_spec.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arena_summary = run_selfplay_arena(
        spec=resolved_arena_spec,
        repo_root=repo_root,
        output_root=arena_root,
    )
    arena_summary_payload = {
        "campaign_name": spec.name,
        "stage_name": stage_name,
        "resolved_arena_spec": str(resolved_arena_spec_path),
        **arena_summary,
    }
    arena_summary_path = arena_root / "summary.json"
    arena_summary_path.write_text(
        json.dumps(arena_summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    arena_matrix = build_selfplay_arena_matrix(arena_summary_payload)
    arena_matrix_path = stage_root / "arena_matrix.json"
    write_selfplay_arena_matrix(arena_matrix_path, arena_matrix)
    return arena_summary_path, arena_matrix_path


def _materialize_start_agent_specs(
    *,
    spec: PlannerEvolutionCampaignSpec,
    repo_root: Path,
    output_root: Path,
    selected_run_names: set[str] | None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    requested_specs: dict[str, str] = dict(spec.benchmark_agent_specs)
    for run_spec in spec.planner_runs:
        if selected_run_names is not None and run_spec.name not in selected_run_names:
            continue
        requested_specs[run_spec.name] = run_spec.agent_template_spec_path
    return _materialize_agent_specs(
        repo_root=repo_root,
        output_root=output_root,
        agent_specs=requested_specs,
        allow_missing=spec.skip_missing_start_agents,
    )


def _materialize_agent_specs(
    *,
    repo_root: Path,
    output_root: Path,
    agent_specs: Mapping[str, str],
    allow_missing: bool,
) -> tuple[dict[str, Path], dict[str, Any]]:
    available: dict[str, Path] = {}
    unavailable: dict[str, Any] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for agent_name, spec_path in sorted(agent_specs.items()):
        source_path = _resolve_repo_path(repo_root, Path(spec_path))
        agent_spec = load_selfplay_agent_spec(source_path)
        missing_paths = _missing_agent_runtime_paths(agent_spec, repo_root=repo_root)
        if missing_paths:
            if not allow_missing:
                raise FileNotFoundError(f"{agent_name}: missing runtime paths: {missing_paths}")
            unavailable[agent_name] = {
                "spec_path": str(source_path),
                "missing_paths": missing_paths,
            }
            continue
        materialized_path = output_root / f"{agent_name}.json"
        write_selfplay_agent_spec(materialized_path, agent_spec)
        available[agent_name] = materialized_path
    return available, unavailable


def _copy_agent_specs(
    *,
    agent_spec_paths: Mapping[str, Path],
    output_root: Path,
) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for agent_name, path in sorted(agent_spec_paths.items()):
        agent_spec = load_selfplay_agent_spec(path)
        copied_path = output_root / f"{agent_name}.json"
        write_selfplay_agent_spec(copied_path, agent_spec)
        copied[agent_name] = copied_path
    return copied


def _evaluate_planner_agents(
    *,
    agent_spec_paths: Mapping[str, Path],
    repo_root: Path,
    verify_paths: Sequence[Path],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for agent_name, spec_path in sorted(agent_spec_paths.items()):
        agent_spec = load_selfplay_agent_spec(spec_path)
        if agent_spec.agent_kind != "planner" or agent_spec.planner_checkpoint is None:
            continue
        checkpoint_path = _resolve_repo_path(repo_root, Path(agent_spec.planner_checkpoint))
        if not checkpoint_path.exists():
            continue
        metrics[agent_name] = evaluate_planner_checkpoint(
            checkpoint_path,
            dataset_paths=verify_paths,
            top_k=top_k,
        ).to_dict()
    return metrics


def _existing_agent_checkpoint(
    agent_spec: SelfplayAgentSpec,
    *,
    repo_root: Path,
) -> Path | None:
    if agent_spec.planner_checkpoint is None:
        return None
    checkpoint_path = _resolve_repo_path(repo_root, Path(agent_spec.planner_checkpoint))
    if checkpoint_path.exists():
        return checkpoint_path
    return None


def _missing_agent_runtime_paths(
    agent_spec: SelfplayAgentSpec,
    *,
    repo_root: Path,
) -> list[str]:
    missing: list[str] = []
    if agent_spec.agent_kind == "uci_engine":
        engine_path = _resolve_repo_path(repo_root, Path(agent_spec.external_engine_path or ""))
        if not engine_path.exists():
            missing.append(str(engine_path))
        return missing
    if agent_spec.proposer_checkpoint is not None:
        proposer_path = _resolve_repo_path(repo_root, Path(agent_spec.proposer_checkpoint))
        if not proposer_path.exists():
            missing.append(str(proposer_path))
    if agent_spec.planner_checkpoint is not None:
        planner_path = _resolve_repo_path(repo_root, Path(agent_spec.planner_checkpoint))
        if not planner_path.exists():
            missing.append(str(planner_path))
    if agent_spec.opponent_checkpoint is not None:
        opponent_path = _resolve_repo_path(repo_root, Path(agent_spec.opponent_checkpoint))
        if not opponent_path.exists():
            missing.append(str(opponent_path))
    if agent_spec.dynamics_checkpoint is not None:
        dynamics_path = _resolve_repo_path(repo_root, Path(agent_spec.dynamics_checkpoint))
        if not dynamics_path.exists():
            missing.append(str(dynamics_path))
    return missing


def _collect_workflow_split_paths(
    *,
    workflow_summary: Mapping[str, object],
    tiers: Sequence[str],
    split_name: str,
) -> list[Path]:
    split_key = "verify" if split_name == "verify" else split_name
    workflow_tiers = dict(workflow_summary["tiers"])
    paths: list[Path] = []
    for tier_name in tiers:
        if tier_name not in workflow_tiers:
            raise ValueError(f"unknown workflow tier: {tier_name}")
        paths.append(Path(workflow_tiers[tier_name][split_key]["planner_head_path"]))
    if not paths:
        raise ValueError(f"no workflow paths selected for split {split_name!r}")
    return paths


def _ordered_agent_specs(
    agent_specs: Mapping[str, Path],
    *,
    order: Sequence[str],
) -> dict[str, Path]:
    if not order:
        return dict(sorted(agent_specs.items()))
    ordered: dict[str, Path] = {}
    seen: set[str] = set()
    for agent_name in order:
        if agent_name not in agent_specs:
            continue
        ordered[agent_name] = agent_specs[agent_name]
        seen.add(agent_name)
    for agent_name, spec_path in sorted(agent_specs.items()):
        if agent_name in seen:
            continue
        ordered[agent_name] = spec_path
    return ordered


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _optional_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_path(value: object | None) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value))
