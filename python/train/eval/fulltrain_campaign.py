"""Versioned Phase-9 long-run campaign for full-data planner training and arena evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from train.config import PlannerTrainConfig
from train.eval.agent_spec import (
    SelfplayAgentSpec,
    load_selfplay_agent_spec,
    write_selfplay_agent_spec,
)
from train.eval.arena import SelfplayArenaSpec, load_selfplay_arena_spec, run_selfplay_arena
from train.eval.campaign import build_planner_verify_matrix
from train.eval.initial_fens import load_selfplay_initial_fen_suite
from train.eval.matrix import build_selfplay_arena_matrix, write_selfplay_arena_matrix
from train.trainers import evaluate_planner_checkpoint, train_planner


PLANNER_FULLTRAIN_CAMPAIGN_VERSION = 1


@dataclass(frozen=True)
class PlannerFulltrainRunSpec:
    """One planner arm inside the train-then-arena campaign."""

    name: str
    base_config_path: str
    agent_template_spec_path: str
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fulltrain run name must be non-empty")
        if not self.base_config_path:
            raise ValueError("fulltrain run base_config_path must be non-empty")
        if not self.agent_template_spec_path:
            raise ValueError("fulltrain run agent_template_spec_path must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "base_config_path": self.base_config_path,
            "agent_template_spec_path": self.agent_template_spec_path,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PlannerFulltrainRunSpec":
        return cls(
            name=str(payload["name"]),
            base_config_path=str(payload["base_config_path"]),
            agent_template_spec_path=str(payload["agent_template_spec_path"]),
            tags=[str(value) for value in list(payload.get("tags") or [])],
        )


@dataclass(frozen=True)
class PlannerFulltrainArenaCampaignSpec:
    """Versioned contract for a large planner full-train followed by arena evaluation."""

    name: str
    output_root: str
    workflow_summary: str
    training_tiers: tuple[str, ...]
    verify_tiers: tuple[str, ...]
    training_epochs: int = 12
    arena_template_spec_path: str = ""
    initial_fen_suite_path: str | None = None
    arena_default_games: int = 1
    arena_parallel_workers: int = 6
    arena_default_max_plies: int | None = None
    arena_opening_selection_seed: int | None = None
    static_agent_specs: dict[str, str] = field(default_factory=dict)
    arena_agent_order: tuple[str, ...] = ()
    baseline_metrics: dict[str, str] = field(default_factory=dict)
    reference_run_name: str | None = None
    planner_runs: tuple[PlannerFulltrainRunSpec, ...] = ()
    spec_version: int = PLANNER_FULLTRAIN_CAMPAIGN_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("campaign name must be non-empty")
        if self.spec_version != PLANNER_FULLTRAIN_CAMPAIGN_VERSION:
            raise ValueError(f"unsupported fulltrain campaign version: {self.spec_version}")
        if not self.output_root:
            raise ValueError("campaign output_root must be non-empty")
        if not self.workflow_summary:
            raise ValueError("campaign workflow_summary must be non-empty")
        if not self.training_tiers:
            raise ValueError("campaign training_tiers must be non-empty")
        if not self.verify_tiers:
            raise ValueError("campaign verify_tiers must be non-empty")
        if self.training_epochs <= 0:
            raise ValueError("campaign training_epochs must be positive")
        if not self.arena_template_spec_path:
            raise ValueError("campaign arena_template_spec_path must be non-empty")
        if self.arena_default_games <= 0:
            raise ValueError("campaign arena_default_games must be positive")
        if self.arena_parallel_workers <= 0:
            raise ValueError("campaign arena_parallel_workers must be positive")
        if self.arena_default_max_plies is not None and self.arena_default_max_plies <= 0:
            raise ValueError("campaign arena_default_max_plies must be positive when provided")
        if self.arena_opening_selection_seed is not None and self.arena_opening_selection_seed < 0:
            raise ValueError("campaign arena_opening_selection_seed must be non-negative when provided")
        if not self.planner_runs:
            raise ValueError("campaign must include at least one planner run")

    def to_dict(self) -> dict[str, object]:
        return {
            "spec_version": self.spec_version,
            **asdict(self),
            "planner_runs": [run.to_dict() for run in self.planner_runs],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PlannerFulltrainArenaCampaignSpec":
        return cls(
            spec_version=int(payload.get("spec_version", PLANNER_FULLTRAIN_CAMPAIGN_VERSION)),
            name=str(payload["name"]),
            output_root=str(payload["output_root"]),
            workflow_summary=str(payload["workflow_summary"]),
            training_tiers=tuple(str(value) for value in list(payload["training_tiers"])),
            verify_tiers=tuple(
                str(value)
                for value in list(payload.get("verify_tiers") or payload["training_tiers"])
            ),
            training_epochs=int(payload.get("training_epochs", 12)),
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
            static_agent_specs={
                str(name): str(path)
                for name, path in dict(payload.get("static_agent_specs") or {}).items()
            },
            arena_agent_order=tuple(
                str(value) for value in list(payload.get("arena_agent_order") or [])
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
                PlannerFulltrainRunSpec.from_dict(dict(run))
                for run in list(payload["planner_runs"])
            ),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "PlannerFulltrainArenaCampaignSpec":
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("fulltrain campaign spec must be a JSON object")
        return cls.from_dict(payload)


def load_planner_fulltrain_arena_campaign_spec(
    path: Path,
) -> PlannerFulltrainArenaCampaignSpec:
    """Load a fulltrain-arena campaign spec from JSON."""
    return PlannerFulltrainArenaCampaignSpec.from_json(path.read_text(encoding="utf-8"))


def write_planner_fulltrain_arena_campaign_spec(
    path: Path,
    spec: PlannerFulltrainArenaCampaignSpec,
) -> None:
    """Write a fulltrain-arena campaign spec to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize_fulltrain_planner_config(
    *,
    base_config: PlannerTrainConfig,
    workflow_summary: Mapping[str, object],
    training_tiers: Sequence[str],
    output_root: Path,
    run_name: str,
    training_epochs: int,
) -> dict[str, Any]:
    """Return a reproducible planner config payload for one full-data training run."""
    train_paths = _collect_workflow_split_paths(
        workflow_summary=workflow_summary,
        tiers=training_tiers,
        split_name="train",
    )
    validation_paths = _collect_workflow_split_paths(
        workflow_summary=workflow_summary,
        tiers=training_tiers,
        split_name="validation",
    )
    payload = base_config.to_dict()
    payload["output_dir"] = str(output_root / "planner_runs" / run_name)
    payload["export"]["bundle_dir"] = str(output_root / "planner_models" / run_name)
    payload["data"]["train_path"] = str(train_paths[0])
    payload["data"]["additional_train_paths"] = [str(path) for path in train_paths[1:]]
    payload["data"]["validation_path"] = str(validation_paths[0])
    payload["data"]["additional_validation_paths"] = [
        str(path) for path in validation_paths[1:]
    ]
    payload["optimization"]["epochs"] = training_epochs
    return payload


def resolve_trained_agent_spec(
    *,
    template_spec: SelfplayAgentSpec,
    agent_name: str,
    planner_checkpoint: Path,
    run_name: str,
) -> SelfplayAgentSpec:
    """Repoint one planner agent spec at the newly trained checkpoint."""
    if template_spec.agent_kind != "planner":
        raise ValueError("only planner templates can be repointed to trained checkpoints")
    metadata = dict(template_spec.metadata)
    metadata["fulltrain_campaign_run"] = run_name
    metadata["planner_checkpoint_source"] = str(planner_checkpoint)
    return replace(
        template_spec,
        name=agent_name,
        planner_checkpoint=str(planner_checkpoint),
        metadata=metadata,
        tags=[*template_spec.tags, "fulltrain_campaign"],
    )


def materialize_fulltrain_arena_spec(
    *,
    template_spec: SelfplayArenaSpec,
    resolved_agent_specs: Mapping[str, str],
    default_initial_fens: Sequence[str],
    campaign_name: str,
    default_games: int,
    parallel_workers: int,
    default_max_plies: int | None = None,
    opening_selection_seed: int | None = None,
) -> SelfplayArenaSpec:
    """Build the resolved arena spec for the newly trained planner family."""
    metadata = dict(template_spec.metadata)
    metadata["source_template"] = template_spec.name
    metadata["purpose"] = "full_data_planner_train_then_arena"
    metadata["campaign_name"] = campaign_name
    return replace(
        template_spec,
        name=f"{campaign_name}_arena",
        agent_specs=dict(resolved_agent_specs),
        schedule_mode="round_robin",
        matchups=[],
        default_games=default_games,
        default_max_plies=(
            default_max_plies
            if default_max_plies is not None
            else template_spec.default_max_plies
        ),
        default_initial_fens=list(default_initial_fens),
        parallel_workers=parallel_workers,
        opening_selection_seed=(
            opening_selection_seed
            if opening_selection_seed is not None
            else template_spec.opening_selection_seed
        ),
        metadata=metadata,
    )


def run_planner_fulltrain_arena_campaign(
    *,
    spec: PlannerFulltrainArenaCampaignSpec,
    repo_root: Path,
    selected_runs: Sequence[str] | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    """Train the configured planner family on full data and then run the arena."""
    workflow_summary_path = _resolve_repo_path(repo_root, Path(spec.workflow_summary))
    workflow_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))
    verify_paths = _collect_workflow_split_paths(
        workflow_summary=workflow_summary,
        tiers=spec.verify_tiers,
        split_name="verify",
    )

    output_root = _resolve_repo_path(repo_root, Path(spec.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    run_name_filter = set(selected_runs) if selected_runs is not None else None
    run_results: dict[str, dict[str, Any]] = {}
    resolved_agent_spec_paths: dict[str, str] = {}

    for run_spec in spec.planner_runs:
        if run_name_filter is not None and run_spec.name not in run_name_filter:
            continue
        base_config_path = _resolve_repo_path(repo_root, Path(run_spec.base_config_path))
        base_config = PlannerTrainConfig.from_dict(
            json.loads(base_config_path.read_text(encoding="utf-8"))
        )
        resolved_config_payload = materialize_fulltrain_planner_config(
            base_config=base_config,
            workflow_summary=workflow_summary,
            training_tiers=spec.training_tiers,
            output_root=output_root,
            run_name=run_spec.name,
            training_epochs=spec.training_epochs,
        )
        resolved_config_path = output_root / "resolved_configs" / f"{run_spec.name}.json"
        resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_config_path.write_text(
            json.dumps(resolved_config_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        resolved_config = PlannerTrainConfig.from_dict(resolved_config_payload)
        checkpoint_path = (
            Path(resolved_config.export.bundle_dir) / resolved_config.export.checkpoint_name
        )
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
        verify_path = output_root / "planner_verify" / f"{run_spec.name}.json"
        verify_path.parent.mkdir(parents=True, exist_ok=True)
        verify_path.write_text(
            json.dumps(verify_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        template_spec = load_selfplay_agent_spec(
            _resolve_repo_path(repo_root, Path(run_spec.agent_template_spec_path))
        )
        resolved_agent_spec = resolve_trained_agent_spec(
            template_spec=template_spec,
            agent_name=run_spec.name,
            planner_checkpoint=checkpoint_path,
            run_name=run_spec.name,
        )
        resolved_agent_spec_path = output_root / "resolved_agent_specs" / f"{run_spec.name}.json"
        write_selfplay_agent_spec(resolved_agent_spec_path, resolved_agent_spec)
        resolved_agent_spec_paths[run_spec.name] = str(resolved_agent_spec_path)

        run_results[run_spec.name] = {
            "base_config_path": str(base_config_path),
            "resolved_config_path": str(resolved_config_path),
            "checkpoint": str(checkpoint_path),
            "training_summary": training_summary,
            "verify_metrics_path": str(verify_path),
            "verify_metrics": verify_metrics,
            "agent_spec_path": str(resolved_agent_spec_path),
            "tags": list(run_spec.tags),
        }

    for agent_name, spec_path in sorted(spec.static_agent_specs.items()):
        static_spec = load_selfplay_agent_spec(_resolve_repo_path(repo_root, Path(spec_path)))
        static_path = output_root / "resolved_agent_specs" / f"{agent_name}.json"
        write_selfplay_agent_spec(static_path, static_spec)
        resolved_agent_spec_paths[agent_name] = str(static_path)

    compare_runs = {
        name: json.loads(_resolve_repo_path(repo_root, Path(path)).read_text(encoding="utf-8"))
        for name, path in spec.baseline_metrics.items()
    }
    for run_name, payload in run_results.items():
        compare_runs[run_name] = dict(payload["verify_metrics"])
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

    arena_template = load_selfplay_arena_spec(
        _resolve_repo_path(repo_root, Path(spec.arena_template_spec_path))
    )
    initial_fens = list(arena_template.default_initial_fens)
    if spec.initial_fen_suite_path is not None:
        suite = load_selfplay_initial_fen_suite(
            _resolve_repo_path(repo_root, Path(spec.initial_fen_suite_path))
        )
        initial_fens = suite.fen_list()
    ordered_agent_specs = _ordered_agent_specs(
        resolved_agent_spec_paths,
        order=spec.arena_agent_order,
    )
    resolved_arena_spec = materialize_fulltrain_arena_spec(
        template_spec=arena_template,
        resolved_agent_specs=ordered_agent_specs,
        default_initial_fens=initial_fens,
        campaign_name=spec.name,
        default_games=spec.arena_default_games,
        parallel_workers=spec.arena_parallel_workers,
        default_max_plies=spec.arena_default_max_plies,
        opening_selection_seed=spec.arena_opening_selection_seed,
    )
    arena_root = output_root / "arena"
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
        "resolved_arena_spec": str(resolved_arena_spec_path),
        **arena_summary,
    }
    arena_summary_path = arena_root / "summary.json"
    arena_summary_path.write_text(
        json.dumps(arena_summary_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    arena_matrix = build_selfplay_arena_matrix(arena_summary_payload)
    arena_matrix_path = output_root / "arena_matrix.json"
    write_selfplay_arena_matrix(arena_matrix_path, arena_matrix)

    summary = {
        "campaign_name": spec.name,
        "campaign_version": spec.spec_version,
        "workflow_summary": str(workflow_summary_path),
        "training_tiers": list(spec.training_tiers),
        "verify_tiers": list(spec.verify_tiers),
        "training_epochs": spec.training_epochs,
        "output_root": str(output_root),
        "run_results": run_results,
        "resolved_agent_specs": ordered_agent_specs,
        "planner_verify_matrix_path": str(verify_matrix_path),
        "arena_summary_path": str(arena_summary_path),
        "arena_matrix_path": str(arena_matrix_path),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


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
    agent_specs: Mapping[str, str],
    *,
    order: Sequence[str],
) -> dict[str, str]:
    if not order:
        return dict(sorted(agent_specs.items()))
    ordered: dict[str, str] = {}
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
