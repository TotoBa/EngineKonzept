"""Round-batched selfplay -> teacher review -> planner retraining cycles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from train.config import PlannerTrainConfig
from train.datasets import (
    build_planner_head_examples_from_selfplay_teacher_reviews,
    build_selfplay_teacher_review_examples,
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
from train.eval.arena import (
    SelfplayArenaMatchupSpec,
    SelfplayArenaSpec,
    run_selfplay_arena,
)
from train.trainers import train_planner


SELFPLAY_TEACHER_RETRAIN_CYCLE_VERSION = 1


@dataclass(frozen=True)
class SelfplayTeacherRetrainAgentSpec:
    """One planner agent that can be warm-start retrained inside the cycle."""

    agent_name: str
    planner_train_config_path: str
    max_head_examples: int | None = None
    include_non_mistakes: bool = False
    retain_base_train_paths: bool = False
    epochs_override: int | None = None
    learning_rate_override: float | None = None
    batch_size_override: int | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.agent_name:
            raise ValueError("retrain agent_name must be non-empty")
        if not self.planner_train_config_path:
            raise ValueError("retrain planner_train_config_path must be non-empty")
        if self.max_head_examples is not None and self.max_head_examples <= 0:
            raise ValueError("retrain max_head_examples must be positive when provided")
        if self.epochs_override is not None and self.epochs_override <= 0:
            raise ValueError("retrain epochs_override must be positive when provided")
        if self.learning_rate_override is not None and self.learning_rate_override <= 0.0:
            raise ValueError(
                "retrain learning_rate_override must be positive when provided"
            )
        if self.batch_size_override is not None and self.batch_size_override <= 0:
            raise ValueError("retrain batch_size_override must be positive when provided")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SelfplayTeacherRetrainAgentSpec":
        return cls(
            agent_name=str(payload["agent_name"]),
            planner_train_config_path=str(payload["planner_train_config_path"]),
            max_head_examples=_optional_int(payload.get("max_head_examples")),
            include_non_mistakes=bool(payload.get("include_non_mistakes", False)),
            retain_base_train_paths=bool(payload.get("retain_base_train_paths", False)),
            epochs_override=_optional_int(payload.get("epochs_override")),
            learning_rate_override=_optional_float(payload.get("learning_rate_override")),
            batch_size_override=_optional_int(payload.get("batch_size_override")),
            tags=[str(value) for value in list(payload.get("tags") or [])],
        )


@dataclass(frozen=True)
class SelfplayTeacherRetrainCycleSpec:
    """Versioned spec for batched selfplay teacher-review retraining."""

    name: str
    arena_spec_path: str
    output_root: str
    retrain_agents: tuple[SelfplayTeacherRetrainAgentSpec, ...]
    batch_mode: str = "reciprocal_pair"
    teacher_engine_path: str = "/usr/games/stockfish18"
    teacher_depth: int | None = 5
    teacher_nodes: int | None = None
    teacher_movetime_ms: int | None = None
    policy_temperature_cp: float = 64.0
    mistake_deadzone_cp: float = 8.0
    mistake_priority_scale_cp: float = 64.0
    max_mistake_priority: float = 4.0
    max_review_examples_per_agent: int | None = None
    spec_version: int = SELFPLAY_TEACHER_RETRAIN_CYCLE_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cycle name must be non-empty")
        if self.spec_version != SELFPLAY_TEACHER_RETRAIN_CYCLE_VERSION:
            raise ValueError(f"unsupported selfplay teacher cycle version: {self.spec_version}")
        if not self.arena_spec_path:
            raise ValueError("cycle arena_spec_path must be non-empty")
        if not self.output_root:
            raise ValueError("cycle output_root must be non-empty")
        if self.batch_mode not in {"session", "reciprocal_pair", "stage"}:
            raise ValueError(
                "cycle batch_mode must be 'session', 'reciprocal_pair', or 'stage'"
            )
        if not self.retrain_agents:
            raise ValueError("cycle must list at least one retrainable agent")
        if not self.teacher_engine_path:
            raise ValueError("cycle teacher_engine_path must be non-empty")
        if (
            self.teacher_depth is None
            and self.teacher_nodes is None
            and self.teacher_movetime_ms is None
        ):
            raise ValueError("cycle requires one of teacher_depth, teacher_nodes, or teacher_movetime_ms")
        if self.teacher_depth is not None and self.teacher_depth <= 0:
            raise ValueError("cycle teacher_depth must be positive when provided")
        if self.teacher_nodes is not None and self.teacher_nodes <= 0:
            raise ValueError("cycle teacher_nodes must be positive when provided")
        if self.teacher_movetime_ms is not None and self.teacher_movetime_ms <= 0:
            raise ValueError("cycle teacher_movetime_ms must be positive when provided")
        if self.policy_temperature_cp <= 0.0:
            raise ValueError("cycle policy_temperature_cp must be positive")
        if self.mistake_deadzone_cp < 0.0:
            raise ValueError("cycle mistake_deadzone_cp must be non-negative")
        if self.mistake_priority_scale_cp <= 0.0:
            raise ValueError("cycle mistake_priority_scale_cp must be positive")
        if self.max_mistake_priority <= 0.0:
            raise ValueError("cycle max_mistake_priority must be positive")
        if self.max_review_examples_per_agent is not None and self.max_review_examples_per_agent <= 0:
            raise ValueError("cycle max_review_examples_per_agent must be positive when provided")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["retrain_agents"] = [agent.to_dict() for agent in self.retrain_agents]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SelfplayTeacherRetrainCycleSpec":
        return cls(
            name=str(payload["name"]),
            arena_spec_path=str(payload["arena_spec_path"]),
            output_root=str(payload["output_root"]),
            retrain_agents=tuple(
                SelfplayTeacherRetrainAgentSpec.from_dict(dict(agent))
                for agent in list(payload["retrain_agents"])
            ),
            batch_mode=str(payload.get("batch_mode", "reciprocal_pair")),
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
            spec_version=int(payload.get("spec_version", SELFPLAY_TEACHER_RETRAIN_CYCLE_VERSION)),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "SelfplayTeacherRetrainCycleSpec":
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            raise ValueError("selfplay teacher retrain cycle spec must be a JSON object")
        return cls.from_dict(payload)


def load_selfplay_teacher_retrain_cycle_spec(path: Path) -> SelfplayTeacherRetrainCycleSpec:
    """Load a selfplay teacher retrain cycle spec from JSON."""
    return SelfplayTeacherRetrainCycleSpec.from_json(path.read_text(encoding="utf-8"))


def write_selfplay_teacher_retrain_cycle_spec(
    path: Path,
    spec: SelfplayTeacherRetrainCycleSpec,
) -> None:
    """Write a selfplay teacher retrain cycle spec to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_selfplay_teacher_retrain_cycle(
    *,
    spec: SelfplayTeacherRetrainCycleSpec,
    repo_root: Path,
    arena_runner: Callable[..., dict[str, Any]] = run_selfplay_arena,
    planner_trainer: Callable[..., Any] = train_planner,
) -> dict[str, Any]:
    """Run one batched selfplay teacher-review retraining cycle."""
    output_root = _resolve_repo_path(repo_root, Path(spec.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    arena_spec = SelfplayArenaSpec.from_json(
        _resolve_repo_path(repo_root, Path(spec.arena_spec_path)).read_text(encoding="utf-8")
    )
    retrain_by_agent = {entry.agent_name: entry for entry in spec.retrain_agents}
    for agent_name in retrain_by_agent:
        if agent_name not in arena_spec.agent_specs:
            raise ValueError(f"cycle retrain agent is not in arena spec: {agent_name}")

    active_agent_specs_root = output_root / "active_agent_specs"
    active_agent_specs_root.mkdir(parents=True, exist_ok=True)
    active_agent_spec_paths = _materialize_initial_agent_specs(
        arena_spec=arena_spec,
        repo_root=repo_root,
        output_root=active_agent_specs_root,
    )

    batch_groups = _build_matchup_batches(arena_spec, mode=spec.batch_mode)
    batch_summaries: list[dict[str, Any]] = []

    for batch_index, batch_matchups in enumerate(batch_groups, 1):
        batch_name = _batch_name(batch_index, batch_matchups)
        batch_root = output_root / "batches" / batch_name
        batch_root.mkdir(parents=True, exist_ok=True)

        batch_arena_spec = _build_batch_arena_spec(
            arena_spec=arena_spec,
            active_agent_spec_paths=active_agent_spec_paths,
            batch_name=batch_name,
            matchups=batch_matchups,
        )
        (batch_root / "arena_spec.resolved.json").write_text(
            json.dumps(batch_arena_spec.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        arena_root = batch_root / "arena"
        arena_root.mkdir(parents=True, exist_ok=True)
        arena_summary = arena_runner(
            spec=batch_arena_spec,
            repo_root=repo_root,
            output_root=arena_root,
        )
        arena_summary_path = arena_root / "summary.json"
        if not arena_summary_path.exists():
            arena_summary_path.write_text(
                json.dumps(arena_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        teacher_training_summary = _materialize_teacher_training_sets(
            arena_summary_path=arena_summary_path,
            active_agent_spec_paths=active_agent_spec_paths,
            batch_root=batch_root / "teacher_training",
            spec=spec,
            repo_root=repo_root,
        )

        retrain_results: dict[str, Any] = {}
        for agent_name, retrain_spec in sorted(retrain_by_agent.items()):
            current_spec_path = active_agent_spec_paths[agent_name]
            current_agent_spec = load_selfplay_agent_spec(current_spec_path)
            if current_agent_spec.agent_kind != "planner":
                retrain_results[agent_name] = {"status": "skipped_non_planner"}
                continue
            if current_agent_spec.planner_checkpoint is None:
                raise ValueError(f"{agent_name}: planner agent is missing planner_checkpoint")

            agent_training_root = batch_root / "teacher_training" / agent_name
            planner_head_path = agent_training_root / planner_head_artifact_name("train")
            agent_summary_path = agent_training_root / "summary.json"
            agent_summary_payload = (
                json.loads(agent_summary_path.read_text(encoding="utf-8"))
                if agent_summary_path.exists()
                else {}
            )
            planner_head_example_count = int(
                agent_summary_payload.get("planner_head_example_count", 0)
            )
            if planner_head_example_count <= 0 or not planner_head_path.exists():
                retrain_results[agent_name] = {
                    "status": "skipped_empty_training_set",
                    "planner_head_example_count": planner_head_example_count,
                }
                continue

            base_config_path = _resolve_repo_path(
                repo_root,
                Path(retrain_spec.planner_train_config_path),
            )
            base_config = PlannerTrainConfig.from_dict(
                json.loads(base_config_path.read_text(encoding="utf-8"))
            )
            resolved_config = _materialize_cycle_planner_config(
                base_config=base_config,
                current_agent_spec=current_agent_spec,
                train_path=planner_head_path,
                output_root=batch_root,
                agent_name=agent_name,
                retrain_spec=retrain_spec,
            )
            resolved_config_path = batch_root / "resolved_configs" / f"{agent_name}.json"
            resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_config_path.write_text(
                json.dumps(resolved_config.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            training_run = planner_trainer(resolved_config, repo_root=repo_root)
            checkpoint_path = (
                Path(resolved_config.export.bundle_dir) / resolved_config.export.checkpoint_name
            )
            updated_agent_spec = replace(
                current_agent_spec,
                planner_checkpoint=str(checkpoint_path),
                metadata={
                    **current_agent_spec.metadata,
                    "selfplay_teacher_cycle": spec.name,
                    "selfplay_teacher_cycle_batch": batch_name,
                    "previous_planner_checkpoint": current_agent_spec.planner_checkpoint,
                },
            )
            updated_spec_path = active_agent_specs_root / f"{agent_name}.json"
            write_selfplay_agent_spec(updated_spec_path, updated_agent_spec)
            active_agent_spec_paths[agent_name] = updated_spec_path
            retrain_results[agent_name] = {
                "status": "trained",
                "resolved_config_path": str(resolved_config_path),
                "checkpoint_path": str(checkpoint_path),
                "planner_head_example_count": planner_head_example_count,
                "training_summary": training_run.to_dict(),
            }

        batch_summary = {
            "batch_name": batch_name,
            "matchups": [matchup.to_dict() for matchup in batch_matchups],
            "arena_summary_path": str(arena_summary_path),
            "teacher_training_summary": teacher_training_summary,
            "retrain_results": retrain_results,
        }
        batch_summary_path = batch_root / "summary.json"
        batch_summary_path.write_text(
            json.dumps(batch_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        batch_summaries.append(batch_summary)

    summary = {
        "cycle_name": spec.name,
        "spec_version": spec.spec_version,
        "output_root": str(output_root),
        "batch_mode": spec.batch_mode,
        "batch_count": len(batch_summaries),
        "final_agent_spec_paths": {
            agent_name: str(path) for agent_name, path in sorted(active_agent_spec_paths.items())
        },
        "batches": batch_summaries,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _materialize_initial_agent_specs(
    *,
    arena_spec: SelfplayArenaSpec,
    repo_root: Path,
    output_root: Path,
) -> dict[str, Path]:
    active_paths: dict[str, Path] = {}
    for agent_name, spec_path in sorted(arena_spec.agent_specs.items()):
        agent_spec = load_selfplay_agent_spec(_resolve_repo_path(repo_root, Path(spec_path)))
        active_path = output_root / f"{agent_name}.json"
        write_selfplay_agent_spec(active_path, agent_spec)
        active_paths[agent_name] = active_path
    return active_paths


def _build_matchup_batches(
    arena_spec: SelfplayArenaSpec,
    *,
    mode: str,
) -> list[list[SelfplayArenaMatchupSpec]]:
    expanded = arena_spec.expanded_matchups()
    if mode == "stage":
        return [expanded]
    if mode == "session":
        return [[matchup] for matchup in expanded]
    used: set[int] = set()
    batches: list[list[SelfplayArenaMatchupSpec]] = []
    for index, matchup in enumerate(expanded):
        if index in used:
            continue
        reciprocal_index = _find_reciprocal_matchup(expanded, index=index, used=used)
        if reciprocal_index is None:
            used.add(index)
            batches.append([matchup])
            continue
        used.update({index, reciprocal_index})
        batches.append([matchup, expanded[reciprocal_index]])
    return batches


def _find_reciprocal_matchup(
    matchups: Sequence[SelfplayArenaMatchupSpec],
    *,
    index: int,
    used: set[int],
) -> int | None:
    matchup = matchups[index]
    for candidate_index in range(index + 1, len(matchups)):
        if candidate_index in used:
            continue
        candidate = matchups[candidate_index]
        if (
            candidate.white_agent == matchup.black_agent
            and candidate.black_agent == matchup.white_agent
            and candidate.games == matchup.games
            and candidate.max_plies == matchup.max_plies
            and candidate.initial_fens == matchup.initial_fens
        ):
            return candidate_index
    return None


def _build_batch_arena_spec(
    *,
    arena_spec: SelfplayArenaSpec,
    active_agent_spec_paths: Mapping[str, Path],
    batch_name: str,
    matchups: Sequence[SelfplayArenaMatchupSpec],
) -> SelfplayArenaSpec:
    return SelfplayArenaSpec(
        name=f"{arena_spec.name}:{batch_name}",
        agent_specs={name: str(path) for name, path in active_agent_spec_paths.items()},
        schedule_mode="explicit",
        matchups=[
            SelfplayArenaMatchupSpec(
                white_agent=matchup.white_agent,
                black_agent=matchup.black_agent,
                games=matchup.games,
                max_plies=matchup.max_plies,
                initial_fens=list(matchup.initial_fens),
                tags=list(matchup.tags),
            )
            for matchup in matchups
        ],
        default_games=arena_spec.default_games,
        default_max_plies=arena_spec.default_max_plies,
        default_initial_fens=list(arena_spec.default_initial_fens),
        parallel_workers=arena_spec.parallel_workers,
        opening_selection_seed=arena_spec.opening_selection_seed,
        round_robin_swap_colors=arena_spec.round_robin_swap_colors,
        include_self_matches=arena_spec.include_self_matches,
        max_plies_adjudication=arena_spec.max_plies_adjudication,
        metadata={
            **arena_spec.metadata,
            "selfplay_teacher_cycle_batch": batch_name,
        },
    )


def _materialize_teacher_training_sets(
    *,
    arena_summary_path: Path,
    active_agent_spec_paths: Mapping[str, Path],
    batch_root: Path,
    spec: SelfplayTeacherRetrainCycleSpec,
    repo_root: Path,
) -> dict[str, Any]:
    batch_root.mkdir(parents=True, exist_ok=True)
    loaded_agent_specs = {
        agent_name: load_selfplay_agent_spec(path)
        for agent_name, path in active_agent_spec_paths.items()
    }
    planner_agent_specs = {
        agent_name: agent_spec
        for agent_name, agent_spec in loaded_agent_specs.items()
        if agent_spec.agent_kind == "planner"
    }
    reviews_by_agent = build_selfplay_teacher_review_examples(
        arena_summary_path=arena_summary_path,
        trainable_agent_names=tuple(planner_agent_specs),
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
    summaries: dict[str, Any] = {}
    for agent_name, agent_spec in sorted(planner_agent_specs.items()):
        agent_root = batch_root / agent_name
        agent_root.mkdir(parents=True, exist_ok=True)
        review_examples = reviews_by_agent.get(agent_name, [])
        review_path = agent_root / selfplay_teacher_review_artifact_name("train")
        write_selfplay_teacher_review_artifact(review_path, review_examples)
        retrain_spec = next(entry for entry in spec.retrain_agents if entry.agent_name == agent_name)
        proposer_checkpoint = agent_spec.proposer_checkpoint
        if proposer_checkpoint is None:
            raise ValueError(f"{agent_name}: planner agent is missing proposer_checkpoint")
        planner_head_examples = build_planner_head_examples_from_selfplay_teacher_reviews(
            review_examples=review_examples,
            proposer_checkpoint=_resolve_repo_path(repo_root, Path(proposer_checkpoint)),
            dynamics_checkpoint=(
                _resolve_repo_path(repo_root, Path(agent_spec.dynamics_checkpoint))
                if agent_spec.dynamics_checkpoint is not None
                else None
            ),
            opponent_mode=agent_spec.opponent_mode,
            opponent_checkpoint=(
                _resolve_repo_path(repo_root, Path(agent_spec.opponent_checkpoint))
                if agent_spec.opponent_checkpoint is not None
                else None
            ),
            root_top_k=agent_spec.root_top_k,
            max_examples=retrain_spec.max_head_examples,
            include_non_mistakes=retrain_spec.include_non_mistakes,
            repo_root=repo_root,
        )
        planner_head_path = agent_root / planner_head_artifact_name("train")
        write_planner_head_artifact(planner_head_path, planner_head_examples)
        agent_summary = {
            "agent_spec_path": str(active_agent_spec_paths[agent_name]),
            "review_path": str(review_path),
            "review_summary": selfplay_teacher_review_summary(review_examples),
            "planner_head_path": str(planner_head_path),
            "planner_head_example_count": len(planner_head_examples),
        }
        (agent_root / "summary.json").write_text(
            json.dumps(agent_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries[agent_name] = agent_summary
    return summaries


def _materialize_cycle_planner_config(
    *,
    base_config: PlannerTrainConfig,
    current_agent_spec: SelfplayAgentSpec,
    train_path: Path,
    output_root: Path,
    agent_name: str,
    retrain_spec: SelfplayTeacherRetrainAgentSpec,
) -> PlannerTrainConfig:
    payload = base_config.to_dict()
    payload["initial_checkpoint"] = str(current_agent_spec.planner_checkpoint)
    payload["output_dir"] = str(output_root / "planner_runs" / agent_name)
    payload["export"]["bundle_dir"] = str(output_root / "planner_models" / agent_name)
    if retrain_spec.retain_base_train_paths:
        existing_train_paths = [
            str(payload["data"]["train_path"]),
            *[str(path) for path in payload["data"].get("additional_train_paths", [])],
        ]
        payload["data"]["train_path"] = str(train_path)
        payload["data"]["additional_train_paths"] = existing_train_paths
    else:
        payload["data"]["train_path"] = str(train_path)
        payload["data"]["additional_train_paths"] = []
    if retrain_spec.epochs_override is not None:
        payload["optimization"]["epochs"] = retrain_spec.epochs_override
    if retrain_spec.learning_rate_override is not None:
        payload["optimization"]["learning_rate"] = retrain_spec.learning_rate_override
    if retrain_spec.batch_size_override is not None:
        payload["optimization"]["batch_size"] = retrain_spec.batch_size_override
    return PlannerTrainConfig.from_dict(payload)


def _batch_name(
    batch_index: int,
    matchups: Sequence[SelfplayArenaMatchupSpec],
) -> str:
    labels = [
        f"{matchup.white_agent}_vs_{matchup.black_agent}"
        for matchup in matchups
    ]
    compact = "__".join(labels)
    return f"{batch_index:02d}_{compact}"


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
