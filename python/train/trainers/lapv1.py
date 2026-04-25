"""Stage-T1/T2 trainer and evaluation helpers for the model-only LAPv1 wrapper."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, field
import json
import random
from pathlib import Path
import time
from typing import Any, BinaryIO, Mapping, Sequence

from train.config import (
    PlannerDataConfig,
    PlannerEvaluationConfig,
    PlannerExportConfig,
    PlannerRuntimeConfig,
    resolve_repo_path,
)
from train.datasets.lapv1_training import (
    LAPv1TrainingExample,
    lapv1_training_example_from_planner_head,
)
from train.datasets.planner_head import PlannerHeadExample
from train.models.intention_encoder import torch
from train.models.dual_accumulator import pack_sparse_feature_lists
from train.models.lapv1 import (
    LAPV1_MODEL_NAME,
    LAPV2_MODEL_VERSION,
    LAPv1Config,
    LAPv1Model,
)
from train.models.policy_head_large import MASKED_CANDIDATE_LOGIT_VALUE
from train.models.proposer import torch_is_available


LAPV1_STAGE1_NAME = "lapv1_stage1"
_PIECE_ROLE_CLASS_COUNT = 7
_CP_TARGET_SCALE = 256.0
_GAP_TARGET_SCALE = 128.0
_ROOT_VALUE_TARGET_CLIP_CP = 1024.0
_ROOT_GAP_TARGET_CLIP_CP = 512.0
_PHASE_NAMES = (
    "opening",
    "early_middlegame",
    "late_middlegame",
    "endgame",
)


@dataclass(frozen=True)
class LAPv1OptimizationConfig:
    """Optimizer and loss weighting settings for LAPv1 stage T1/T2."""

    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    log_interval_batches: int = 128
    value_wdl_weight: float = 1.0
    value_cp_weight: float = 0.25
    sharpness_weight: float = 0.1
    policy_ce_weight: float = 1.0
    policy_kl_weight: float = 0.25
    policy_margin_weight: float = 0.0
    policy_rank_weight: float = 0.0
    curriculum_priority_weight: float = 0.1
    intention_aux_weight: float = 0.0
    sharpness_target_loss_weight: float = 0.0
    deliberation_monotonicity_weight: float = 0.0
    deliberation_step_policy_weight: float = 0.0
    deliberation_improvement_weight: float = 0.0
    deliberation_rank_progress_weight: float = 0.1
    deliberation_step_utility_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("optimization.epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("optimization.batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("optimization.learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("optimization.weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("optimization.max_grad_norm must be positive when set")
        if self.log_interval_batches <= 0:
            raise ValueError("optimization.log_interval_batches must be positive")
        for name in (
            "value_wdl_weight",
            "value_cp_weight",
            "sharpness_weight",
            "policy_ce_weight",
            "policy_kl_weight",
            "policy_margin_weight",
            "policy_rank_weight",
            "curriculum_priority_weight",
            "intention_aux_weight",
            "sharpness_target_loss_weight",
            "deliberation_monotonicity_weight",
            "deliberation_step_policy_weight",
            "deliberation_improvement_weight",
            "deliberation_rank_progress_weight",
            "deliberation_step_utility_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"optimization.{name} must be non-negative")


@dataclass(frozen=True)
class LAPv1Stage2PhaseConfig:
    """One explicit Stage-T2 training phase with its own trainable groups."""

    name: str
    epochs: int
    trainable_parameter_groups: tuple[str, ...] = ("all",)
    max_inner_steps_schedule: tuple[int, ...] = (2, 4, 8)
    min_inner_steps_schedule: tuple[int, ...] = ()
    train_paths: tuple[str, ...] = ()
    train_path_weights: tuple[float, ...] = ()
    train_epoch_examples: int | None = None
    validation_paths: tuple[str, ...] = ()
    learning_rate_scale_by_group: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage2.phases[].name must be non-empty")
        if self.epochs <= 0:
            raise ValueError("stage2.phases[].epochs must be positive")
        if not self.trainable_parameter_groups:
            raise ValueError(
                "stage2.phases[].trainable_parameter_groups must be non-empty"
            )
        allowed_groups = {
            "all",
            "root_backbone",
            "root_heads",
            "inner_loop",
            "inner_loop_core",
            "inner_delta_network",
            "aux_probe",
        }
        invalid = sorted(set(self.trainable_parameter_groups) - allowed_groups)
        if invalid:
            raise ValueError(
                "stage2.phases[].trainable_parameter_groups contains unsupported "
                f"entries: {', '.join(invalid)}"
            )
        if "all" in self.trainable_parameter_groups and len(self.trainable_parameter_groups) > 1:
            raise ValueError(
                "stage2.phases[].trainable_parameter_groups may not mix 'all' with other groups"
            )
        if not self.max_inner_steps_schedule:
            raise ValueError("stage2.phases[].max_inner_steps_schedule must be non-empty")
        if any(step <= 0 for step in self.max_inner_steps_schedule):
            raise ValueError(
                "stage2.phases[].max_inner_steps_schedule entries must be positive"
            )
        if self.min_inner_steps_schedule and any(
            step < 0 for step in self.min_inner_steps_schedule
        ):
            raise ValueError(
                "stage2.phases[].min_inner_steps_schedule entries must be non-negative"
            )
        if self.min_inner_steps_schedule and len(self.min_inner_steps_schedule) != len(
            self.max_inner_steps_schedule
        ):
            raise ValueError(
                "stage2.phases[].min_inner_steps_schedule must align with "
                "stage2.phases[].max_inner_steps_schedule"
            )
        if any(not path for path in self.train_paths):
            raise ValueError("stage2.phases[].train_paths entries must be non-empty")
        if self.train_path_weights:
            if not self.train_paths:
                raise ValueError(
                    "stage2.phases[].train_path_weights requires explicit train_paths"
                )
            if len(self.train_path_weights) != len(self.train_paths):
                raise ValueError(
                    "stage2.phases[].train_path_weights must align with train_paths"
                )
            if any(weight <= 0.0 for weight in self.train_path_weights):
                raise ValueError(
                    "stage2.phases[].train_path_weights entries must be positive"
                )
        if self.train_epoch_examples is not None and self.train_epoch_examples <= 0:
            raise ValueError("stage2.phases[].train_epoch_examples must be positive")
        if any(not path for path in self.validation_paths):
            raise ValueError("stage2.phases[].validation_paths entries must be non-empty")
        invalid_lr_groups = sorted(
            set(self.learning_rate_scale_by_group) - set(self.trainable_parameter_groups)
        )
        if invalid_lr_groups:
            raise ValueError(
                "stage2.phases[].learning_rate_scale_by_group may only reference active "
                f"trainable groups: {', '.join(invalid_lr_groups)}"
            )
        if any(scale <= 0.0 for scale in self.learning_rate_scale_by_group.values()):
            raise ValueError(
                "stage2.phases[].learning_rate_scale_by_group values must be positive"
            )


@dataclass(frozen=True)
class LAPv1Stage2Config:
    """Trainer-only curriculum and auxiliary-loss settings for LAPv1 stage T2."""

    max_inner_steps_schedule: tuple[int, ...] = (2, 4, 8)
    phases: tuple[LAPv1Stage2PhaseConfig, ...] = ()
    phase_load_balance: bool = False
    gate_stage_a_steps: int = 0
    gate_stage_b_steps: int = 0
    selection_validation_paths: tuple[str, ...] = ()
    selection_min_inner_steps: int | None = None
    selection_max_inner_steps: int | None = None

    def __post_init__(self) -> None:
        if not self.max_inner_steps_schedule:
            raise ValueError("stage2.max_inner_steps_schedule must be non-empty")
        if any(step <= 0 for step in self.max_inner_steps_schedule):
            raise ValueError("stage2.max_inner_steps_schedule entries must be positive")
        if self.phases and self.max_inner_steps_schedule != (2, 4, 8):
            raise ValueError(
                "stage2.max_inner_steps_schedule may only be overridden when stage2.phases is empty"
            )
        if self.gate_stage_a_steps < 0:
            raise ValueError("stage2.gate_stage_a_steps must be non-negative")
        if self.gate_stage_b_steps < 0:
            raise ValueError("stage2.gate_stage_b_steps must be non-negative")
        if (
            self.gate_stage_b_steps > 0
            and self.gate_stage_b_steps < self.gate_stage_a_steps
        ):
            raise ValueError(
                "stage2.gate_stage_b_steps must be >= stage2.gate_stage_a_steps"
            )
        if any(not path for path in self.selection_validation_paths):
            raise ValueError("stage2.selection_validation_paths entries must be non-empty")
        if self.selection_min_inner_steps is not None and self.selection_min_inner_steps < 0:
            raise ValueError("stage2.selection_min_inner_steps must be non-negative")
        if self.selection_max_inner_steps is not None and self.selection_max_inner_steps <= 0:
            raise ValueError("stage2.selection_max_inner_steps must be positive")
        if (
            self.selection_min_inner_steps is not None
            and self.selection_max_inner_steps is not None
            and self.selection_min_inner_steps > self.selection_max_inner_steps
        ):
            raise ValueError(
                "stage2.selection_min_inner_steps must not exceed "
                "stage2.selection_max_inner_steps"
            )


@dataclass(frozen=True)
class LAPv1TrainConfig:
    """Full configuration for LAPv1 stage-T1/T2 training."""

    seed: int
    output_dir: str
    stage: str
    data: PlannerDataConfig
    model: LAPv1Config
    optimization: LAPv1OptimizationConfig
    evaluation: PlannerEvaluationConfig
    runtime: PlannerRuntimeConfig
    export: PlannerExportConfig
    initial_checkpoint: str | None = None
    stage2: LAPv1Stage2Config | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")
        if self.stage not in {"T1", "T2"}:
            raise ValueError("stage must be 'T1' or 'T2'")
        if self.stage == "T1":
            if self.stage2 is not None:
                raise ValueError("stage2 settings are only valid when stage='T2'")
            if self.model.deliberation.max_inner_steps != 0:
                raise ValueError("stage T1 requires model.deliberation.max_inner_steps == 0")
        else:
            if self.stage2 is None:
                raise ValueError("stage2 settings are required when stage='T2'")
            if self.model.deliberation.max_inner_steps <= 0:
                raise ValueError("stage T2 requires model.deliberation.max_inner_steps > 0")
            max_schedule_step = (
                max(self.stage2.max_inner_steps_schedule)
                if not self.stage2.phases
                else max(
                    max(phase.max_inner_steps_schedule)
                    for phase in self.stage2.phases
                )
            )
            if max_schedule_step > self.model.deliberation.max_inner_steps:
                raise ValueError(
                    "stage2.max_inner_steps_schedule must not exceed model.deliberation.max_inner_steps"
                )
            if self.stage2.phases:
                total_phase_epochs = sum(phase.epochs for phase in self.stage2.phases)
                if total_phase_epochs != self.optimization.epochs:
                    raise ValueError(
                        "sum(stage2.phases[].epochs) must equal optimization.epochs"
                    )
                for phase in self.stage2.phases:
                    if phase.min_inner_steps_schedule and any(
                        min_step > max_step
                        for min_step, max_step in zip(
                            phase.min_inner_steps_schedule,
                            phase.max_inner_steps_schedule,
                            strict=True,
                        )
                    ):
                        raise ValueError(
                            "stage2.phases[].min_inner_steps_schedule must not exceed "
                            "stage2.phases[].max_inner_steps_schedule"
                        )
            if (
                self.stage2.selection_max_inner_steps is not None
                and self.stage2.selection_max_inner_steps
                > self.model.deliberation.max_inner_steps
            ):
                raise ValueError(
                    "stage2.selection_max_inner_steps must not exceed "
                    "model.deliberation.max_inner_steps"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LAPv1TrainConfig":
        """Parse one JSON-like LAPv1 training config."""
        data_payload = payload.get("data")
        if not isinstance(data_payload, Mapping):
            raise ValueError("data must be a JSON object")
        model_payload = payload.get("model")
        if not isinstance(model_payload, Mapping):
            raise ValueError("model must be a JSON object")
        architecture = str(model_payload.get("architecture", "lapv1"))
        if architecture != "lapv1":
            raise ValueError("model.architecture must be 'lapv1'")
        optimization_payload = payload.get("optimization")
        if not isinstance(optimization_payload, Mapping):
            raise ValueError("optimization must be a JSON object")
        evaluation_payload = payload.get("evaluation")
        if not isinstance(evaluation_payload, Mapping):
            raise ValueError("evaluation must be a JSON object")
        export_payload = payload.get("export")
        if not isinstance(export_payload, Mapping):
            raise ValueError("export must be a JSON object")
        return cls(
            seed=int(payload.get("seed", 0)),
            output_dir=str(payload["output_dir"]),
            stage=str(payload["stage"]),
            initial_checkpoint=(
                None
                if payload.get("initial_checkpoint") in (None, "")
                else str(payload["initial_checkpoint"])
            ),
            stage2=(
                None
                if payload.get("stage2") is None
                else LAPv1Stage2Config(
                    max_inner_steps_schedule=tuple(
                        int(step)
                        for step in list(
                            dict(payload["stage2"]).get("max_inner_steps_schedule", (2, 4, 8))
                        )
                    ),
                    phase_load_balance=bool(
                        dict(payload["stage2"]).get("phase_load_balance", False)
                    ),
                    gate_stage_a_steps=int(
                        dict(payload["stage2"]).get("gate_stage_a_steps", 0)
                    ),
                    gate_stage_b_steps=int(
                        dict(payload["stage2"]).get("gate_stage_b_steps", 0)
                    ),
                    phases=tuple(
                        LAPv1Stage2PhaseConfig(
                            name=str(entry["name"]),
                            epochs=int(entry["epochs"]),
                            trainable_parameter_groups=tuple(
                                str(value)
                                for value in list(
                                    dict(entry).get(
                                        "trainable_parameter_groups",
                                        ("all",),
                                    )
                                )
                            ),
                            max_inner_steps_schedule=tuple(
                                int(step)
                                for step in list(
                                    dict(entry).get("max_inner_steps_schedule", (2, 4, 8))
                                )
                            ),
                            min_inner_steps_schedule=tuple(
                                int(step)
                                for step in list(
                                    dict(entry).get("min_inner_steps_schedule", ())
                                )
                            ),
                            train_paths=tuple(
                                str(path)
                                for path in list(dict(entry).get("train_paths") or [])
                            ),
                            train_path_weights=tuple(
                                float(weight)
                                for weight in list(
                                    dict(entry).get("train_path_weights") or []
                                )
                            ),
                            train_epoch_examples=(
                                int(dict(entry)["train_epoch_examples"])
                                if dict(entry).get("train_epoch_examples") is not None
                                else None
                            ),
                            validation_paths=tuple(
                                str(path)
                                for path in list(
                                    dict(entry).get("validation_paths") or []
                                )
                            ),
                            learning_rate_scale_by_group={
                                str(key): float(value)
                                for key, value in dict(
                                    dict(entry).get("learning_rate_scale_by_group") or {}
                                ).items()
                            },
                        )
                        for entry in list(dict(payload["stage2"]).get("phases") or [])
                    ),
                    selection_validation_paths=tuple(
                        str(path)
                        for path in list(
                            dict(payload["stage2"]).get("selection_validation_paths") or []
                        )
                    ),
                    selection_min_inner_steps=(
                        int(dict(payload["stage2"])["selection_min_inner_steps"])
                        if dict(payload["stage2"]).get("selection_min_inner_steps") is not None
                        else None
                    ),
                    selection_max_inner_steps=(
                        int(dict(payload["stage2"])["selection_max_inner_steps"])
                        if dict(payload["stage2"]).get("selection_max_inner_steps") is not None
                        else None
                    ),
                )
            ),
            data=PlannerDataConfig(
                train_path=str(data_payload["train_path"]),
                validation_path=str(data_payload["validation_path"]),
                additional_train_paths=tuple(
                    str(path) for path in data_payload.get("additional_train_paths", [])
                ),
                additional_validation_paths=tuple(
                    str(path) for path in data_payload.get("additional_validation_paths", [])
                ),
            ),
            model=LAPv1Config.from_mapping(dict(model_payload)),
            optimization=LAPv1OptimizationConfig(**dict(optimization_payload)),
            evaluation=PlannerEvaluationConfig(**dict(evaluation_payload)),
            runtime=PlannerRuntimeConfig(**dict(payload.get("runtime", {}))),
            export=PlannerExportConfig(**dict(export_payload)),
        )


@dataclass(frozen=True)
class _ResolvedStage2Phase:
    name: str
    epoch_start: int
    epoch_end: int
    trainable_parameter_groups: tuple[str, ...]
    max_inner_steps_schedule: tuple[int, ...]
    min_inner_steps_schedule: tuple[int, ...]
    train_paths: tuple[str, ...]
    train_path_weights: tuple[float, ...]
    train_epoch_examples: int | None
    validation_paths: tuple[str, ...]
    learning_rate_scale_by_group: dict[str, float]

    def contains_epoch(self, epoch: int) -> bool:
        return self.epoch_start <= epoch <= self.epoch_end


@dataclass(frozen=True)
class LAPv1Metrics:
    """Aggregated static-head metrics for one LAPv1 train/eval pass."""

    total_examples: int
    supervised_examples: int
    total_loss: float
    value_wdl_loss: float
    value_cp_loss: float
    sharpness_loss: float
    sharpness_target_loss: float
    policy_ce_loss: float
    policy_kl_loss: float
    policy_margin_loss: float
    policy_rank_loss: float
    intention_aux_loss: float
    deliberation_monotonicity_loss: float
    deliberation_step_policy_loss: float
    deliberation_improvement_loss: float
    deliberation_rank_progress_loss: float
    deliberation_step_utility_loss: float
    opponent_distill_loss: float
    root_top1_accuracy: float
    root_top3_accuracy: float
    teacher_root_mean_reciprocal_rank: float
    teacher_root_mean_probability: float
    rollbacks: int
    rollback_examples: int
    mean_rollback_step: float
    rollback_hit_rate: float
    rollback_example_rate: float
    initial_root_top1_accuracy: float
    initial_root_top3_accuracy: float
    initial_teacher_root_mean_reciprocal_rank: float
    top1_changed_rate: float
    teacher_rank_improved_rate: float
    teacher_rank_degraded_rate: float
    step_rank_improved_rate: float
    step_rank_degraded_rate: float
    step_utility_continue_rate: float
    step_utility_predicted_continue_rate: float
    root_incorrect_improvement_rate: float
    root_correct_degraded_rate: float
    mean_teacher_rank_delta: float
    mean_step_rank_delta: float
    mean_inner_steps_executed: float
    step_histogram: dict[str, int]
    phase_usage: dict[str, int]
    phase_value_loss: dict[str, float]
    phase_policy_loss: dict[str, float]
    ft_drift: float | None
    adapter_cosine_distance: float | None
    reply_consistency: float | None
    frontier_revisit_rate: float
    frontier_turnover_rate: float
    frontier_stable_rate: float
    frontier_unique_coverage: float
    final_top1_frontier_coverage: float
    frontier_state_drift: float
    frontier_memory_norm: float
    frontier_update_gate_mean: float
    frontier_reply_pressure_mean: float
    frontier_reply_uncertainty_mean: float
    frontier_interaction_norm_mean: float
    examples_per_second: float

    def to_dict(self) -> dict[str, object]:
        return {
            "total_examples": self.total_examples,
            "supervised_examples": self.supervised_examples,
            "total_loss": round(self.total_loss, 6),
            "value_wdl_loss": round(self.value_wdl_loss, 6),
            "value_cp_loss": round(self.value_cp_loss, 6),
            "sharpness_loss": round(self.sharpness_loss, 6),
            "sharpness_target_loss": round(self.sharpness_target_loss, 6),
            "policy_ce_loss": round(self.policy_ce_loss, 6),
            "policy_kl_loss": round(self.policy_kl_loss, 6),
            "policy_margin_loss": round(self.policy_margin_loss, 6),
            "policy_rank_loss": round(self.policy_rank_loss, 6),
            "intention_aux_loss": round(self.intention_aux_loss, 6),
            "deliberation_monotonicity_loss": round(
                self.deliberation_monotonicity_loss,
                6,
            ),
            "deliberation_step_policy_loss": round(
                self.deliberation_step_policy_loss,
                6,
            ),
            "deliberation_improvement_loss": round(
                self.deliberation_improvement_loss,
                6,
            ),
            "deliberation_rank_progress_loss": round(
                self.deliberation_rank_progress_loss,
                6,
            ),
            "deliberation_step_utility_loss": round(
                self.deliberation_step_utility_loss,
                6,
            ),
            "opponent_distill_loss": round(self.opponent_distill_loss, 6),
            "root_top1_accuracy": round(self.root_top1_accuracy, 6),
            "root_top3_accuracy": round(self.root_top3_accuracy, 6),
            "teacher_root_mean_reciprocal_rank": round(
                self.teacher_root_mean_reciprocal_rank,
                6,
            ),
            "teacher_root_mean_probability": round(
                self.teacher_root_mean_probability,
                6,
            ),
            "rollbacks": self.rollbacks,
            "rollback_examples": self.rollback_examples,
            "mean_rollback_step": round(self.mean_rollback_step, 6),
            "rollback_hit_rate": round(self.rollback_hit_rate, 6),
            "rollback_example_rate": round(self.rollback_example_rate, 6),
            "initial_root_top1_accuracy": round(self.initial_root_top1_accuracy, 6),
            "initial_root_top3_accuracy": round(self.initial_root_top3_accuracy, 6),
            "initial_teacher_root_mean_reciprocal_rank": round(
                self.initial_teacher_root_mean_reciprocal_rank,
                6,
            ),
            "top1_changed_rate": round(self.top1_changed_rate, 6),
            "teacher_rank_improved_rate": round(self.teacher_rank_improved_rate, 6),
            "teacher_rank_degraded_rate": round(self.teacher_rank_degraded_rate, 6),
            "step_rank_improved_rate": round(self.step_rank_improved_rate, 6),
            "step_rank_degraded_rate": round(self.step_rank_degraded_rate, 6),
            "step_utility_continue_rate": round(
                self.step_utility_continue_rate,
                6,
            ),
            "step_utility_predicted_continue_rate": round(
                self.step_utility_predicted_continue_rate,
                6,
            ),
            "root_incorrect_improvement_rate": round(
                self.root_incorrect_improvement_rate,
                6,
            ),
            "root_correct_degraded_rate": round(
                self.root_correct_degraded_rate,
                6,
            ),
            "mean_teacher_rank_delta": round(self.mean_teacher_rank_delta, 6),
            "mean_step_rank_delta": round(self.mean_step_rank_delta, 6),
            "mean_inner_steps_executed": round(self.mean_inner_steps_executed, 6),
            "step_histogram": dict(self.step_histogram),
            "phase_usage": dict(self.phase_usage),
            "phase_value_loss": {
                key: round(value, 6)
                for key, value in self.phase_value_loss.items()
            },
            "phase_policy_loss": {
                key: round(value, 6)
                for key, value in self.phase_policy_loss.items()
            },
            "ft_drift": (
                None if self.ft_drift is None else round(self.ft_drift, 6)
            ),
            "adapter_cosine_distance": (
                None
                if self.adapter_cosine_distance is None
                else round(self.adapter_cosine_distance, 6)
            ),
            "reply_consistency": (
                None
                if self.reply_consistency is None
                else round(self.reply_consistency, 6)
            ),
            "frontier_revisit_rate": round(self.frontier_revisit_rate, 6),
            "frontier_turnover_rate": round(self.frontier_turnover_rate, 6),
            "frontier_stable_rate": round(self.frontier_stable_rate, 6),
            "frontier_unique_coverage": round(self.frontier_unique_coverage, 6),
            "final_top1_frontier_coverage": round(
                self.final_top1_frontier_coverage,
                6,
            ),
            "frontier_state_drift": round(self.frontier_state_drift, 6),
            "frontier_memory_norm": round(self.frontier_memory_norm, 6),
            "frontier_update_gate_mean": round(self.frontier_update_gate_mean, 6),
            "frontier_reply_pressure_mean": round(
                self.frontier_reply_pressure_mean,
                6,
            ),
            "frontier_reply_uncertainty_mean": round(
                self.frontier_reply_uncertainty_mean,
                6,
            ),
            "frontier_interaction_norm_mean": round(
                self.frontier_interaction_norm_mean,
                6,
            ),
            "examples_per_second": round(self.examples_per_second, 3),
        }


@dataclass(frozen=True)
class LAPv1TrainingRun:
    """Serializable result summary for one static-head LAPv1 run."""

    history: list[dict[str, Any]]
    best_epoch: int
    best_validation: dict[str, float | int]
    best_validation_source: str
    best_validation_paths: list[str]
    export_paths: dict[str, str]
    summary_path: str
    model_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_validation": self.best_validation,
            "best_validation_source": self.best_validation_source,
            "best_validation_paths": self.best_validation_paths,
            "export_paths": self.export_paths,
            "summary_path": self.summary_path,
            "model_parameter_count": self.model_parameter_count,
        }


@dataclass(frozen=True)
class LAPv1WarmStartResult:
    """Serializable result summary for one LAPv2 warm-start checkpoint export."""

    output_checkpoint: str
    source_checkpoint: str
    target_stage: str
    lapv2_version: int
    fresh_init_prefixes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_checkpoint": self.output_checkpoint,
            "source_checkpoint": self.source_checkpoint,
            "target_stage": self.target_stage,
            "lapv2_version": self.lapv2_version,
            "fresh_init_prefixes": list(self.fresh_init_prefixes),
        }


_PreparedLAPv1Example = LAPv1TrainingExample


@dataclass(frozen=True)
class _ResolvedPhaseDatasets:
    train_examples: Sequence[_PreparedLAPv1Example]
    validation_examples: Sequence[_PreparedLAPv1Example]
    train_paths: tuple[str, ...]
    validation_paths: tuple[str, ...]
    train_path_weights: tuple[float, ...]
    train_epoch_examples: int | None


class _LazyPreparedLAPv1Dataset(Sequence[_PreparedLAPv1Example]):
    """Disk-backed planner-head dataset that prepares LAPv1 examples on demand."""

    def __init__(
        self,
        paths: Sequence[Path],
        *,
        log_label: str | None = None,
        log_every_examples: int = 0,
    ) -> None:
        self._paths = tuple(paths)
        self._offsets_per_path: list[list[int]] = []
        self._line_numbers_per_path: list[list[int]] = []
        self._source_kinds: list[str] = []
        self._cumulative_sizes: list[int] = []
        self._handles: list[BinaryIO | None] = [None] * len(self._paths)
        self._log_label = log_label
        self._log_every_examples = max(int(log_every_examples), 0)
        running_total = 0
        if self._log_label is not None:
            print(
                "[lapv1-train] "
                f"dataset_index_start label={self._log_label} paths={len(self._paths)}",
                flush=True,
            )
        for path in self._paths:
            offsets: list[int] = []
            line_numbers: list[int] = []
            source_kind = "lapv1"
            next_log_count = self._log_every_examples
            if self._log_label is not None:
                print(
                    "[lapv1-train] "
                    f"dataset_index_path_start label={self._log_label} path={path}",
                    flush=True,
                )
            with path.open("rb") as handle:
                line_number = 0
                while True:
                    offset = handle.tell()
                    raw_line = handle.readline()
                    if not raw_line:
                        break
                    line_number += 1
                    if raw_line.strip():
                        if not offsets:
                            source_kind = _detect_lapv1_source_kind(raw_line, source=str(path))
                        offsets.append(offset)
                        line_numbers.append(line_number)
                        if (
                            self._log_label is not None
                            and self._log_every_examples > 0
                            and len(offsets) >= next_log_count
                        ):
                            print(
                                "[lapv1-train] "
                                f"dataset_index_progress label={self._log_label} "
                                f"path={path} examples={len(offsets)} line={line_number}",
                                flush=True,
                            )
                            next_log_count += self._log_every_examples
            self._offsets_per_path.append(offsets)
            self._line_numbers_per_path.append(line_numbers)
            self._source_kinds.append(source_kind)
            running_total += len(offsets)
            self._cumulative_sizes.append(running_total)
            if self._log_label is not None:
                print(
                    "[lapv1-train] "
                    f"dataset_index_path_done label={self._log_label} "
                    f"path={path} source_kind={source_kind} examples={len(offsets)}",
                    flush=True,
                )
        if self._log_label is not None:
            print(
                "[lapv1-train] "
                f"dataset_index_done label={self._log_label} total_examples={running_total}",
                flush=True,
            )

    def __len__(self) -> int:
        if not self._cumulative_sizes:
            return 0
        return self._cumulative_sizes[-1]

    def __getitem__(
        self,
        index: int | slice,
    ) -> _PreparedLAPv1Example | list[_PreparedLAPv1Example]:
        if isinstance(index, slice):
            return [
                self[position]
                for position in range(*index.indices(len(self)))
            ]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("dataset index out of range")
        path_index = bisect_right(self._cumulative_sizes, index)
        previous_total = 0 if path_index == 0 else self._cumulative_sizes[path_index - 1]
        local_index = index - previous_total
        handle = self._file_handle(path_index)
        handle.seek(self._offsets_per_path[path_index][local_index])
        raw_line = handle.readline()
        if not raw_line:
            raise IndexError("planner head line missing at indexed offset")
        line = raw_line.decode("utf-8").strip()
        if not line:
            raise ValueError("indexed planner head line was unexpectedly blank")
        path = self._paths[path_index]
        line_number = self._line_numbers_per_path[path_index][local_index]
        if self._source_kinds[path_index] == "planner_head":
            return _prepare_example(
                PlannerHeadExample.from_json(line, source=f"{path}:{line_number}")
            )
        return LAPv1TrainingExample.from_json(line, source=f"{path}:{line_number}")

    def close(self) -> None:
        for index, handle in enumerate(self._handles):
            if handle is not None:
                handle.close()
                self._handles[index] = None

    def _file_handle(self, path_index: int) -> BinaryIO:
        handle = self._handles[path_index]
        if handle is None or handle.closed:
            handle = self._paths[path_index].open("rb")
            self._handles[path_index] = handle
        return handle

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        self.close()


class _ResampledPreparedLAPv1Dataset(Sequence[_PreparedLAPv1Example]):
    """Epoch-local weighted mixture over multiple lazy LAPv1 datasets."""

    def __init__(
        self,
        datasets: Sequence[Sequence[_PreparedLAPv1Example]],
        mapping: Sequence[tuple[int, int]],
    ) -> None:
        if not datasets:
            raise ValueError("resampled dataset requires at least one source dataset")
        self._datasets = tuple(datasets)
        self._mapping = tuple(mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __getitem__(
        self,
        index: int | slice,
    ) -> _PreparedLAPv1Example | list[_PreparedLAPv1Example]:
        if isinstance(index, slice):
            return [
                self[position]
                for position in range(*index.indices(len(self)))
            ]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("resampled dataset index out of range")
        dataset_index, local_index = self._mapping[index]
        return self._datasets[dataset_index][local_index]


if torch is not None:

    class _PieceRoleAuxProbe(torch.nn.Module):
        def __init__(self, *, intention_dim: int) -> None:
            super().__init__()
            self.network = torch.nn.Linear(intention_dim, _PIECE_ROLE_CLASS_COUNT)

        def forward(self, piece_intentions: torch.Tensor) -> torch.Tensor:
            return self.network(piece_intentions)


def train_lapv1(config: LAPv1TrainConfig, *, repo_root: Path) -> LAPv1TrainingRun:
    """Train LAPv1 heads on planner-head artifacts for stage T1 or T2."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for LAPv1 training. Install the 'train' extra or torch."
        )

    output_dir = resolve_repo_path(repo_root, config.output_dir)
    bundle_dir = resolve_repo_path(repo_root, config.export.bundle_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(config.seed)
    _configure_torch_runtime(config.runtime.torch_threads)

    dataset_cache: dict[tuple[str, tuple[str, ...]], _LazyPreparedLAPv1Dataset] = {}

    def get_dataset(
        *,
        label: str,
        paths: Sequence[str],
        log_every_examples: int,
    ) -> _LazyPreparedLAPv1Dataset:
        resolved_paths = tuple(
            str(resolve_repo_path(repo_root, path))
            for path in paths
        )
        cache_key = (label, resolved_paths)
        if cache_key not in dataset_cache:
            dataset_cache[cache_key] = _build_lazy_dataset(
                [Path(path) for path in resolved_paths],
                log_label=label,
                log_every_examples=log_every_examples,
            )
        return dataset_cache[cache_key]

    model = LAPv1Model(config.model)
    aux_probe = _PieceRoleAuxProbe(
        intention_dim=config.model.intention_encoder.intention_dim
    )
    if config.initial_checkpoint is not None:
        initial_checkpoint_path = resolve_repo_path(repo_root, config.initial_checkpoint)
        initial_payload = torch.load(initial_checkpoint_path, map_location="cpu")
        if initial_payload.get("model_name") != LAPV1_MODEL_NAME:
            raise ValueError(
                f"{initial_checkpoint_path}: unsupported initial LAPv1 model name "
                f"{initial_payload.get('model_name')!r}"
            )
        _load_lapv1_model_state(
            model,
            dict(initial_payload["model_state_dict"]),
            checkpoint_path=initial_checkpoint_path,
        )
        aux_state_dict = initial_payload.get("aux_state_dict")
        if isinstance(aux_state_dict, dict):
            aux_probe.load_state_dict(dict(aux_state_dict))
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    stage2_phases = _resolve_stage2_phases(config)
    initial_phase = None if config.stage == "T1" else _stage2_phase_for_epoch(stage2_phases, epoch=1)
    initial_datasets = _datasets_for_phase(
        config=config,
        repo_root=repo_root,
        phase=initial_phase,
        get_dataset=get_dataset,
        epoch=1,
        seed=config.seed + 1,
    )
    train_examples = initial_datasets.train_examples
    validation_examples = initial_datasets.validation_examples
    active_train_paths = initial_datasets.train_paths
    active_validation_paths = initial_datasets.validation_paths
    if len(train_examples) == 0:
        raise ValueError("training artifact is empty")
    if len(validation_examples) == 0:
        raise ValueError("validation artifact is empty")
    selection_validation_examples: Sequence[_PreparedLAPv1Example] | None = None
    selection_validation_paths: tuple[str, ...] = ()
    selection_min_inner_steps: int | None = None
    selection_max_inner_steps: int | None = None
    if config.stage == "T2" and config.stage2 is not None and config.stage2.selection_validation_paths:
        selection_validation_paths = tuple(config.stage2.selection_validation_paths)
        selection_validation_examples = get_dataset(
            label="selection:common",
            paths=selection_validation_paths,
            log_every_examples=25_000,
        )
        selection_min_inner_steps = config.stage2.selection_min_inner_steps
        selection_max_inner_steps = config.stage2.selection_max_inner_steps
    print(
        "[lapv1-train] "
        f"stage={config.stage} output_dir={output_dir} bundle_dir={bundle_dir} "
        f"epochs={config.optimization.epochs} batch_size={config.optimization.batch_size} "
        f"train_examples={len(train_examples)} validation_examples={len(validation_examples)} "
        f"initial_checkpoint={config.initial_checkpoint} data_access=lazy_jsonl",
        flush=True,
    )
    active_phase_name = "joint" if config.stage == "T1" else None
    active_phase_groups: tuple[str, ...] = ("all",)
    active_phase_lr_scales: dict[str, float] = {}
    if config.stage == "T1":
        optimizer, trainable_parameter_count = _build_optimizer(
            model=model,
            aux_probe=aux_probe,
            groups=("all",),
            learning_rate_scale_by_group={},
            learning_rate=config.optimization.learning_rate,
            weight_decay=config.optimization.weight_decay,
        )
    else:
        optimizer = None

    history: list[dict[str, Any]] = []
    best_epoch = 1
    best_validation: LAPv1Metrics | None = None
    best_validation_source = "validation"
    best_validation_paths = list(active_validation_paths)
    best_model_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    best_aux_state = {
        name: tensor.detach().clone()
        for name, tensor in aux_probe.state_dict().items()
    }
    optimizer_step_counter = [0]
    loss_balance_state: dict[str, float] = {}

    try:
        for epoch in range(1, config.optimization.epochs + 1):
            if config.stage == "T2":
                current_phase = _stage2_phase_for_epoch(stage2_phases, epoch=epoch)
                if current_phase.name != active_phase_name:
                    active_phase_name = current_phase.name
                    active_phase_groups = current_phase.trainable_parameter_groups
                    active_phase_lr_scales = dict(current_phase.learning_rate_scale_by_group)
                    optimizer, trainable_parameter_count = _build_optimizer(
                        model=model,
                        aux_probe=aux_probe,
                        groups=current_phase.trainable_parameter_groups,
                        learning_rate_scale_by_group=active_phase_lr_scales,
                        learning_rate=config.optimization.learning_rate,
                        weight_decay=config.optimization.weight_decay,
                    )
                    lr_groups = _format_lr_group_scales(
                        groups=current_phase.trainable_parameter_groups,
                        learning_rate_scale_by_group=active_phase_lr_scales,
                        base_learning_rate=config.optimization.learning_rate,
                    )
                    print(
                        "[lapv1-train] "
                        f"stage2_phase={current_phase.name} "
                        f"epochs={current_phase.epoch_start}-{current_phase.epoch_end} "
                        f"trainable_groups={','.join(current_phase.trainable_parameter_groups)} "
                        f"trainable_parameters={trainable_parameter_count} "
                        f"optimizer_lrs={lr_groups}",
                        flush=True,
                    )
                phase_datasets = _datasets_for_phase(
                    config=config,
                    repo_root=repo_root,
                    phase=current_phase,
                    get_dataset=get_dataset,
                    epoch=epoch,
                    seed=config.seed + epoch,
                )
                train_examples = phase_datasets.train_examples
                validation_examples = phase_datasets.validation_examples
                active_train_paths = phase_datasets.train_paths
                active_validation_paths = phase_datasets.validation_paths
            else:
                current_phase = None
                phase_datasets = initial_datasets
            assert optimizer is not None
            current_max_inner_steps = (
                0
                if config.stage == "T1"
                else _current_max_inner_steps(
                    phase=current_phase,
                    epoch=epoch,
                )
            )
            current_min_inner_steps = (
                0
                if config.stage == "T1"
                else _current_min_inner_steps(
                    phase=current_phase,
                    epoch=epoch,
                    base_min_inner_steps=config.model.deliberation.min_inner_steps,
                    current_max_inner_steps=current_max_inner_steps,
                )
            )
            model.deliberation_loop.max_inner_steps = current_max_inner_steps
            model.deliberation_loop.min_inner_steps = current_min_inner_steps
            train_metrics = _run_epoch(
                model=model,
                aux_probe=aux_probe,
                examples=train_examples,
                batch_size=config.optimization.batch_size,
                optimizer=optimizer,
                training=True,
                seed=config.seed + epoch,
                optimization=config.optimization,
                top_k=config.evaluation.top_k,
                stage=config.stage,
                stage2=config.stage2,
                epoch=epoch,
                total_epochs=config.optimization.epochs,
                optimizer_step_counter=optimizer_step_counter,
                loss_balance_state=loss_balance_state,
            )
            validation_metrics = _run_epoch(
                model=model,
                aux_probe=aux_probe,
                examples=validation_examples,
                batch_size=config.optimization.batch_size,
                optimizer=None,
                training=False,
                seed=config.seed,
                optimization=config.optimization,
                top_k=config.evaluation.top_k,
                stage=config.stage,
                stage2=config.stage2,
                epoch=epoch,
                total_epochs=config.optimization.epochs,
                optimizer_step_counter=optimizer_step_counter,
                loss_balance_state=loss_balance_state,
            )
            selection_validation_metrics: LAPv1Metrics | None = None
            selection_collapse_alarm: dict[str, float | bool] | None = None
            resolved_selection_max_inner_steps = current_max_inner_steps
            resolved_selection_min_inner_steps = current_min_inner_steps
            if selection_validation_examples is not None:
                previous_max_inner_steps = model.deliberation_loop.max_inner_steps
                previous_min_inner_steps = model.deliberation_loop.min_inner_steps
                resolved_selection_max_inner_steps = (
                    selection_max_inner_steps
                    if selection_max_inner_steps is not None
                    else current_max_inner_steps
                )
                resolved_selection_min_inner_steps = (
                    selection_min_inner_steps
                    if selection_min_inner_steps is not None
                    else current_min_inner_steps
                )
                model.deliberation_loop.max_inner_steps = resolved_selection_max_inner_steps
                model.deliberation_loop.min_inner_steps = resolved_selection_min_inner_steps
                selection_validation_metrics = _run_epoch(
                    model=model,
                    aux_probe=aux_probe,
                    examples=selection_validation_examples,
                    batch_size=config.optimization.batch_size,
                    optimizer=None,
                    training=False,
                    seed=config.seed,
                    optimization=config.optimization,
                    top_k=config.evaluation.top_k,
                    stage=config.stage,
                    stage2=config.stage2,
                    epoch=epoch,
                    total_epochs=config.optimization.epochs,
                    optimizer_step_counter=optimizer_step_counter,
                    loss_balance_state=loss_balance_state,
                )
                model.deliberation_loop.max_inner_steps = previous_max_inner_steps
                model.deliberation_loop.min_inner_steps = previous_min_inner_steps
                selection_collapse_alarm = _collapse_alarm(selection_validation_metrics)
                if selection_collapse_alarm is not None:
                    print(
                        "[lapv1-train][warn] "
                        f"collapse_detected epoch={epoch}/{config.optimization.epochs} "
                        f"scope=selection_validation "
                        f"delta_top1={selection_collapse_alarm['delta_top1']:.6f} "
                        f"delta_mrr={selection_collapse_alarm['delta_mrr']:.6f} "
                        f"top1_changed={selection_collapse_alarm['top1_changed_rate']:.6f} "
                        f"mean_steps={selection_collapse_alarm['mean_inner_steps_executed']:.6f}",
                        flush=True,
                    )
            phase_collapse_alarm = _collapse_alarm(validation_metrics)
            if phase_collapse_alarm is not None:
                print(
                    "[lapv1-train][warn] "
                    f"collapse_detected epoch={epoch}/{config.optimization.epochs} "
                    f"scope=phase_validation "
                    f"delta_top1={phase_collapse_alarm['delta_top1']:.6f} "
                    f"delta_mrr={phase_collapse_alarm['delta_mrr']:.6f} "
                    f"top1_changed={phase_collapse_alarm['top1_changed_rate']:.6f} "
                    f"mean_steps={phase_collapse_alarm['mean_inner_steps_executed']:.6f}",
                    flush=True,
                )
            history_entry = {
                "epoch": epoch,
                "stage2_phase": active_phase_name,
                "trainable_parameter_groups": list(active_phase_groups),
                "learning_rate_scale_by_group": dict(active_phase_lr_scales),
                "train_dataset_paths": list(active_train_paths),
                "train_path_weights": list(phase_datasets.train_path_weights),
                "train_epoch_examples": phase_datasets.train_epoch_examples,
                "validation_dataset_paths": list(active_validation_paths),
                "min_inner_steps": current_min_inner_steps,
                "max_inner_steps": current_max_inner_steps,
                "train": train_metrics.to_dict(),
                "validation": validation_metrics.to_dict(),
                "phase_collapse_alarm": phase_collapse_alarm,
            }
            if selection_validation_metrics is not None:
                history_entry["selection_validation_paths"] = list(selection_validation_paths)
                history_entry["selection_validation"] = selection_validation_metrics.to_dict()
                history_entry["selection_min_inner_steps"] = resolved_selection_min_inner_steps
                history_entry["selection_max_inner_steps"] = resolved_selection_max_inner_steps
                history_entry["selection_collapse_alarm"] = selection_collapse_alarm
            history.append(history_entry)
            selection_reference = (
                selection_validation_metrics
                if selection_validation_metrics is not None
                else validation_metrics
            )
            selection_source = (
                "selection_validation"
                if selection_validation_metrics is not None
                else "validation"
            )
            selection_paths_for_epoch = (
                list(selection_validation_paths)
                if selection_validation_metrics is not None
                else list(active_validation_paths)
            )
            print(
                "[lapv1-train] "
                f"epoch={epoch}/{config.optimization.epochs} "
                f"stage={config.stage} "
                f"stage2_phase={active_phase_name} "
                f"train_paths={len(active_train_paths)} "
                f"validation_paths={len(active_validation_paths)} "
                f"min_inner_steps={current_min_inner_steps} "
                f"max_inner_steps={current_max_inner_steps} "
                f"train_loss={train_metrics.total_loss:.4f} "
                f"val_top1={validation_metrics.root_top1_accuracy:.4f} "
                f"root_top1={validation_metrics.initial_root_top1_accuracy:.4f} "
                f"val_mrr={validation_metrics.teacher_root_mean_reciprocal_rank:.4f} "
                f"root_mrr={validation_metrics.initial_teacher_root_mean_reciprocal_rank:.4f} "
                f"top1_changed={validation_metrics.top1_changed_rate:.4f} "
                f"rank_gain={validation_metrics.mean_teacher_rank_delta:.4f} "
                f"step_rank_gain={validation_metrics.mean_step_rank_delta:.4f} "
                f"step_rank_degrade={validation_metrics.step_rank_degraded_rate:.4f} "
                f"step_utility={validation_metrics.step_utility_continue_rate:.4f} "
                f"sharp_continue={validation_metrics.step_utility_predicted_continue_rate:.4f} "
                f"root_incorrect_gain={validation_metrics.root_incorrect_improvement_rate:.4f} "
                f"root_correct_degrade={validation_metrics.root_correct_degraded_rate:.4f} "
                f"mean_steps={validation_metrics.mean_inner_steps_executed:.4f} "
                f"rollbacks={validation_metrics.rollbacks} "
                f"rollback_hit_rate={validation_metrics.rollback_hit_rate:.4f} "
                f"rollback_example_rate={validation_metrics.rollback_example_rate:.4f} "
                f"frontier_turnover={validation_metrics.frontier_turnover_rate:.4f} "
                f"frontier_unique={validation_metrics.frontier_unique_coverage:.4f} "
                f"frontier_top1_cov={validation_metrics.final_top1_frontier_coverage:.4f} "
                f"frontier_state_drift={validation_metrics.frontier_state_drift:.4f} "
                f"frontier_memory_norm={validation_metrics.frontier_memory_norm:.4f} "
                f"frontier_gate={validation_metrics.frontier_update_gate_mean:.4f} "
                f"frontier_pressure={validation_metrics.frontier_reply_pressure_mean:.4f} "
                f"frontier_reply_uncertainty={validation_metrics.frontier_reply_uncertainty_mean:.4f} "
                f"frontier_interaction={validation_metrics.frontier_interaction_norm_mean:.4f} "
                f"selection_source={selection_source} "
                f"selection_top1={selection_reference.root_top1_accuracy:.4f} "
                f"selection_mrr={selection_reference.teacher_root_mean_reciprocal_rank:.4f} "
                f"phase_usage={_format_phase_metric_map(validation_metrics.phase_usage)} "
                f"phase_value_loss={_format_phase_metric_map(validation_metrics.phase_value_loss, precision=4)} "
                f"phase_policy_loss={_format_phase_metric_map(validation_metrics.phase_policy_loss, precision=4)} "
                f"ft_drift={('n/a' if validation_metrics.ft_drift is None else f'{validation_metrics.ft_drift:.4f}')} "
                f"adapter_cosine={('n/a' if validation_metrics.adapter_cosine_distance is None else f'{validation_metrics.adapter_cosine_distance:.4f}')} "
                f"reply_consistency={('n/a' if validation_metrics.reply_consistency is None else f'{validation_metrics.reply_consistency:.4f}')}",
                flush=True,
            )

            if best_validation is None or _is_better_validation(
                selection_reference,
                best_validation,
            ):
                best_epoch = epoch
                best_validation = selection_reference
                best_validation_source = selection_source
                best_validation_paths = selection_paths_for_epoch
                best_model_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                best_aux_state = {
                    name: tensor.detach().clone()
                    for name, tensor in aux_probe.state_dict().items()
                }
    finally:
        for dataset in dataset_cache.values():
            dataset.close()

    assert best_validation is not None
    model.load_state_dict(best_model_state)
    aux_probe.load_state_dict(best_aux_state)

    checkpoint_path = bundle_dir / config.export.checkpoint_name
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "lapv2_version": LAPV2_MODEL_VERSION if config.model.lapv2.enabled else 0,
            "model_state_dict": model.state_dict(),
            "aux_state_dict": aux_probe.state_dict(),
            "training_config": config.to_dict(),
            "best_validation": best_validation.to_dict(),
        },
        checkpoint_path,
    )
    summary = LAPv1TrainingRun(
        history=history,
        best_epoch=best_epoch,
        best_validation=best_validation.to_dict(),
        best_validation_source=best_validation_source,
        best_validation_paths=best_validation_paths,
        export_paths={"checkpoint": str(checkpoint_path)},
        summary_path=str(output_dir / "summary.json"),
        model_parameter_count=model_parameter_count,
    )
    summary_path = Path(summary.summary_path)
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_lapv1_train_config(path: Path | str) -> LAPv1TrainConfig:
    """Load a LAPv1 training config from JSON."""
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: training config root must be an object")
    return LAPv1TrainConfig.from_dict(payload)


def count_lapv1_model_parameters(config: LAPv1TrainConfig) -> int:
    """Return the instantiated LAPv1 model parameter count for one config."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for LAPv1 model inspection. Install the 'train' extra or torch."
        )
    model = LAPv1Model(config.model)
    return sum(parameter.numel() for parameter in model.parameters())


def build_lapv2_warm_start_checkpoint(
    source_checkpoint: Path | str,
    *,
    target_config: LAPv1TrainConfig,
    output_checkpoint: Path | str,
) -> LAPv1WarmStartResult:
    """Materialize one LAPv2 init checkpoint from an existing LAPv1 T2 checkpoint."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for LAPv2 warm-start export. Install the 'train' extra or torch."
        )
    if not target_config.model.lapv2.enabled:
        raise ValueError("target_config.model.lapv2.enabled must be true for LAPv2 warm starts")
    source_path = Path(source_checkpoint)
    output_path = Path(output_checkpoint)
    payload = torch.load(source_path, map_location="cpu")
    if payload.get("model_name") != LAPV1_MODEL_NAME:
        raise ValueError(
            f"{source_path}: unsupported LAPv1 model name {payload.get('model_name')!r}"
        )
    source_training_config_payload = payload.get("training_config")
    if not isinstance(source_training_config_payload, Mapping):
        raise ValueError(f"{source_path}: checkpoint is missing training_config")
    source_training_config = LAPv1TrainConfig.from_dict(
        dict(source_training_config_payload)
    )
    if source_training_config.stage != "T2":
        raise ValueError(f"{source_path}: warm-start source must be a stage-T2 checkpoint")
    model = LAPv1Model(target_config.model)
    _load_lapv1_model_state(
        model,
        dict(payload["model_state_dict"]),
        checkpoint_path=source_path,
    )
    aux_probe = _PieceRoleAuxProbe(
        intention_dim=target_config.model.intention_encoder.intention_dim
    )
    aux_state_dict = payload.get("aux_state_dict")
    if isinstance(aux_state_dict, Mapping):
        aux_probe.load_state_dict(dict(aux_state_dict))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fresh_init_prefixes = list(_lapv2_fresh_init_prefixes(model))
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
            "lapv2_version": LAPV2_MODEL_VERSION,
            "model_state_dict": model.state_dict(),
            "aux_state_dict": aux_probe.state_dict(),
            "training_config": target_config.to_dict(),
            "warm_start_source_checkpoint": str(source_path),
            "warm_start_source_stage": source_training_config.stage,
            "warm_start_fresh_init_prefixes": fresh_init_prefixes,
            "warm_start_source_best_validation": payload.get("best_validation"),
        },
        output_path,
    )
    return LAPv1WarmStartResult(
        output_checkpoint=str(output_path),
        source_checkpoint=str(source_path),
        target_stage=target_config.stage,
        lapv2_version=LAPV2_MODEL_VERSION,
        fresh_init_prefixes=fresh_init_prefixes,
    )


def _load_lapv1_model_state(
    model: LAPv1Model,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_path: Path,
) -> None:
    effective_state_dict = _expand_legacy_phase_moe_state_dict(model, state_dict)
    compatible_missing_prefixes = [
        "deliberation_loop.cell.candidate_delta_network.",
        "deliberation_loop.cell.frontier_context_projection.",
        "deliberation_loop.cell.candidate_frontier_state_projection.",
        "deliberation_loop.cell.candidate_frontier_memory_projection.",
        "deliberation_loop.cell.depth_condition_projection.",
        "deliberation_loop.cell.candidate_frontier_gate_network.",
        "deliberation_loop.cell.candidate_frontier_delta_network.",
        "deliberation_loop.cell.candidate_interaction_network.",
    ]
    compatible_unexpected_prefixes: list[str] = []
    lapv2_fresh_init_prefixes = list(_lapv2_fresh_init_prefixes(model))
    if model.config.lapv2.enabled and model.config.lapv2.shared_opponent_readout:
        compatible_unexpected_prefixes.extend(
            [
                "deliberation_loop.reply_signal_projector.opponent_head.",
                "deliberation_loop.reply_signal_projector.root_projection.",
                "deliberation_loop.reply_signal_projector.next_projection.",
                "deliberation_loop.reply_signal_projector.transition_projection.",
                "deliberation_loop.reply_signal_projector.reply_global_projection.",
            ]
        )
    if model.config.lapv2.enabled and model.config.lapv2.sharpness_phase_moe:
        compatible_unexpected_prefixes.extend(
            [
                "_sharpness_projector.sharpness_head.network.",
                "deliberation_loop.sharpness_projector.sharpness_head.network.",
            ]
        )
    if lapv2_fresh_init_prefixes:
        compatible_missing_prefixes.extend(lapv2_fresh_init_prefixes)
        print(
            "[lapv1-train][warn] "
            f"{checkpoint_path}: initializing LAPv2 weights from scratch for "
            + ", ".join(sorted(lapv2_fresh_init_prefixes)),
            flush=True,
        )
    compatible_missing_prefixes = tuple(compatible_missing_prefixes)
    effective_state_dict = _drop_compatible_mismatched_model_state_keys(
        model,
        effective_state_dict,
        checkpoint_path=checkpoint_path,
        compatible_missing_prefixes=compatible_missing_prefixes,
    )
    incompatible_missing_keys = [
        key
        for key in model.state_dict().keys()
        if key not in effective_state_dict
        and not key.startswith(compatible_missing_prefixes)
    ]
    if incompatible_missing_keys:
        raise RuntimeError(
            f"{checkpoint_path}: incompatible LAPv1 checkpoint is missing required keys: "
            + ", ".join(sorted(incompatible_missing_keys))
        )
    load_result = model.load_state_dict(dict(effective_state_dict), strict=False)
    unexpected_keys = [
        key
        for key in load_result.unexpected_keys
        if not key.startswith(tuple(compatible_unexpected_prefixes))
    ]
    if unexpected_keys:
        raise RuntimeError(
            f"{checkpoint_path}: incompatible LAPv1 checkpoint has unexpected keys: "
            + ", ".join(sorted(unexpected_keys))
        )
    remaining_missing = [
        key
        for key in load_result.missing_keys
        if not key.startswith(compatible_missing_prefixes)
    ]
    if remaining_missing:
        raise RuntimeError(
            f"{checkpoint_path}: incompatible LAPv1 checkpoint is missing unsupported keys: "
            + ", ".join(sorted(remaining_missing))
        )


def _drop_compatible_mismatched_model_state_keys(
    model: LAPv1Model,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    compatible_missing_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    model_state = model.state_dict()
    filtered_state_dict = dict(state_dict)
    dropped_keys: list[str] = []
    for key, value in list(filtered_state_dict.items()):
        if key not in model_state or not isinstance(value, torch.Tensor):
            continue
        if tuple(value.shape) == tuple(model_state[key].shape):
            continue
        if not key.startswith(compatible_missing_prefixes):
            continue
        filtered_state_dict.pop(key)
        dropped_keys.append(key)
    if dropped_keys:
        print(
            "[lapv1-train][warn] "
            f"{checkpoint_path}: skipping incompatible checkpoint tensors for "
            + ", ".join(sorted(dropped_keys)),
            flush=True,
        )
    return filtered_state_dict


def _lapv2_fresh_init_prefixes(model: LAPv1Model) -> tuple[str, ...]:
    prefixes: list[str] = []
    if model.config.lapv2.nnue_value_enabled:
        prefixes.extend(["ft.", "value_head_nnue."])
    if model.config.lapv2.nnue_policy_enabled:
        prefixes.append("policy_head_nnue.")
    if model.config.lapv2.enabled and model.config.lapv2.sharpness_phase_moe:
        prefixes.extend(
            [
                "_sharpness_projector.sharpness_head.",
                "deliberation_loop.sharpness_projector.sharpness_head.",
            ]
        )
    if model.config.lapv2.enabled and model.config.lapv2.shared_opponent_readout:
        prefixes.append("deliberation_loop.reply_signal_projector.opponent_readout.")
    return tuple(prefixes)


def _expand_legacy_phase_moe_state_dict(
    model: LAPv1Model,
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    expanded = _expand_legacy_sharpness_alias_state_dict(dict(state_dict))
    if model.config.lapv2.phase_moe_enabled:
        expanded = _replicate_phase_moe_module_keys(
            expanded,
            module_prefix="intention_encoder",
            num_phases=4,
        )
        expanded = _replicate_phase_moe_module_keys(
            expanded,
            module_prefix="state_embedder",
            num_phases=4,
        )
    if model.config.lapv2.sharpness_phase_moe_enabled:
        expanded = _replicate_phase_moe_module_keys(
            expanded,
            module_prefix="sharpness_head",
            num_phases=4,
        )
    if model.config.lapv2.nnue_value_phase_moe_enabled:
        expanded = _replicate_phase_moe_module_keys(
            expanded,
            module_prefix="ft",
            num_phases=4,
        )
        expanded = _replicate_phase_moe_module_keys(
            expanded,
            module_prefix="value_head_nnue",
            num_phases=4,
        )
        if model.config.lapv2.nnue_policy_enabled:
            expanded = _replicate_phase_moe_module_keys(
                expanded,
                module_prefix="policy_head_nnue",
                num_phases=4,
            )
    return expanded


def _expand_legacy_sharpness_alias_state_dict(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    expanded = dict(state_dict)
    alias_prefixes = (
        "sharpness_head.",
        "_sharpness_projector.sharpness_head.",
        "deliberation_loop.sharpness_projector.sharpness_head.",
    )
    alias_payloads: dict[str, dict[str, Any]] = {}
    for prefix in alias_prefixes:
        payload = {
            key.removeprefix(prefix): value
            for key, value in expanded.items()
            if key.startswith(prefix)
        }
        if payload:
            alias_payloads[prefix] = payload
    if not alias_payloads:
        return expanded

    source_prefix, source_payload = next(iter(alias_payloads.items()))
    for alias_prefix in alias_prefixes:
        if alias_prefix in alias_payloads:
            continue
        for suffix, value in source_payload.items():
            expanded[f"{alias_prefix}{suffix}"] = value
    return expanded


def _replicate_phase_moe_module_keys(
    state_dict: Mapping[str, Any],
    *,
    module_prefix: str,
    num_phases: int,
) -> dict[str, Any]:
    expert_prefix = f"{module_prefix}.experts."
    legacy_prefix = f"{module_prefix}."
    if any(key.startswith(expert_prefix) for key in state_dict):
        return dict(state_dict)
    legacy_keys = [key for key in state_dict if key.startswith(legacy_prefix)]
    if not legacy_keys:
        return dict(state_dict)
    expanded = dict(state_dict)
    for key in legacy_keys:
        suffix = key[len(legacy_prefix) :]
        value = state_dict[key]
        for phase in range(num_phases):
            expanded[f"{expert_prefix}{phase}.{suffix}"] = value
        del expanded[key]
    return expanded


def _lapv2_phase_gate_stage(
    model: LAPv1Model,
    *,
    stage2: LAPv1Stage2Config | None,
    optimizer_step: int,
) -> str:
    if not model.config.lapv2.nnue_value_phase_moe_enabled:
        return "off"
    if stage2 is not None:
        if stage2.gate_stage_a_steps > 0 and optimizer_step < stage2.gate_stage_a_steps:
            return "stage_a"
        if stage2.gate_stage_b_steps > 0 and optimizer_step < stage2.gate_stage_b_steps:
            return "stage_b"
        if stage2.gate_stage_a_steps > 0 or stage2.gate_stage_b_steps > 0:
            return "released"
    if (
        model.config.lapv2.nnue_phase_gate_steps > 0
        and optimizer_step < model.config.lapv2.nnue_phase_gate_steps
    ):
        return "stage_a"
    return "released"


def _lapv2_phase_gate_should_mean_pull(
    model: LAPv1Model,
    *,
    stage2: LAPv1Stage2Config | None,
    optimizer_step: int,
) -> bool:
    return _lapv2_phase_gate_stage(
        model,
        stage2=stage2,
        optimizer_step=optimizer_step,
    ) == "stage_a"


def _apply_lapv2_phase_gate_mean_pull(model: LAPv1Model) -> None:
    if model.ft is not None:
        _mean_pull_phase_module_parameters(model.ft)
    if model.value_head_nnue is not None:
        _mean_pull_phase_module_parameters(model.value_head_nnue)
    if model.policy_head_nnue is not None:
        _mean_pull_phase_module_parameters(model.policy_head_nnue)


def _mean_pull_phase_module_parameters(module: torch.nn.Module) -> None:
    experts = getattr(module, "experts", None)
    if experts is None:
        return
    expert_modules = list(experts)
    if not expert_modules:
        return
    expert_parameter_lists = [list(expert.parameters()) for expert in expert_modules]
    for parameters in zip(*expert_parameter_lists, strict=True):
        mean_value = torch.stack([parameter.detach() for parameter in parameters], dim=0).mean(dim=0)
        for parameter in parameters:
            parameter.data.copy_(mean_value)


def evaluate_lapv1_checkpoint(
    checkpoint_path: Path | str,
    *,
    dataset_path: Path | str | None = None,
    top_k: int = 3,
) -> LAPv1Metrics:
    """Evaluate a saved LAPv1 stage-T1/T2 checkpoint on one planner-head artifact."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for LAPv1 evaluation. Install the 'train' extra or torch."
        )

    checkpoint = Path(checkpoint_path)
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("model_name") != LAPV1_MODEL_NAME:
        raise ValueError(
            f"{checkpoint}: unsupported LAPv1 model name {payload.get('model_name')!r}"
        )

    training_config = LAPv1TrainConfig.from_dict(dict(payload["training_config"]))
    stage = training_config.stage
    stage2 = training_config.stage2
    lapv1_config = training_config.model
    model = LAPv1Model(lapv1_config)
    if stage == "T2" and stage2 is not None:
        max_inner_steps = (
            max(stage2.max_inner_steps_schedule)
            if not stage2.phases
            else max(
                max(phase.max_inner_steps_schedule)
                for phase in stage2.phases
            )
        )
        model.deliberation_loop.max_inner_steps = max_inner_steps
        model.deliberation_loop.min_inner_steps = min(
            lapv1_config.deliberation.min_inner_steps,
            max_inner_steps,
        )
    _load_lapv1_model_state(
        model,
        dict(payload["model_state_dict"]),
        checkpoint_path=checkpoint,
    )
    aux_probe = _PieceRoleAuxProbe(
        intention_dim=lapv1_config.intention_encoder.intention_dim
    )
    aux_state_dict = payload.get("aux_state_dict")
    if isinstance(aux_state_dict, dict):
        aux_probe.load_state_dict(dict(aux_state_dict))

    effective_dataset_path = (
        Path(dataset_path)
        if dataset_path is not None
        else Path(training_config.data.validation_path)
    )
    examples = _build_lazy_dataset(
        [effective_dataset_path],
        log_label="evaluation",
        log_every_examples=25_000,
    )
    try:
        return _run_epoch(
            model=model,
            aux_probe=aux_probe,
            examples=examples,
            batch_size=training_config.optimization.batch_size,
            optimizer=None,
            training=False,
            seed=training_config.seed,
            optimization=training_config.optimization,
            top_k=top_k,
            stage=stage,
            stage2=stage2,
            epoch=1,
            total_epochs=1,
            optimizer_step_counter=None,
            loss_balance_state={},
        )
    finally:
        examples.close()


def _build_lazy_dataset(
    paths: Sequence[Path],
    *,
    log_label: str | None = None,
    log_every_examples: int = 0,
) -> _LazyPreparedLAPv1Dataset:
    return _LazyPreparedLAPv1Dataset(
        paths,
        log_label=log_label,
        log_every_examples=log_every_examples,
    )


def _prepare_example(example: PlannerHeadExample) -> _PreparedLAPv1Example:
    return lapv1_training_example_from_planner_head(example)


def _detect_lapv1_source_kind(raw_line: bytes, *, source: str) -> str:
    try:
        payload = json.loads(raw_line.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - exercised on corrupt artifacts
        raise ValueError(f"{source}: failed to parse LAPv1 source probe line") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source}: expected JSON object per artifact line")
    if "piece_tokens" in payload and "state_context_global" in payload:
        return "lapv1"
    if "feature_vector" in payload and "fen" in payload:
        return "planner_head"
    raise ValueError(f"{source}: unsupported artifact line schema for LAPv1 training")


def _run_epoch(
    *,
    model: LAPv1Model,
    aux_probe: _PieceRoleAuxProbe,
    examples: Sequence[_PreparedLAPv1Example],
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
    training: bool,
    seed: int,
    optimization: LAPv1OptimizationConfig,
    top_k: int,
    stage: str,
    stage2: LAPv1Stage2Config | None,
    epoch: int,
    total_epochs: int,
    optimizer_step_counter: list[int] | None,
    loss_balance_state: dict[str, float] | None = None,
) -> LAPv1Metrics:
    if training:
        model.train()
        aux_probe.train()
    else:
        model.eval()
        aux_probe.eval()

    start_time = time.perf_counter()
    total_examples = 0
    total_loss = 0.0
    total_value_wdl = 0.0
    total_value_cp = 0.0
    total_sharpness = 0.0
    total_sharpness_target = 0.0
    total_policy_ce = 0.0
    total_policy_kl = 0.0
    total_policy_margin = 0.0
    total_policy_rank = 0.0
    total_intention_aux = 0.0
    total_monotonicity = 0.0
    total_step_policy = 0.0
    total_improvement = 0.0
    total_rank_progress = 0.0
    total_step_utility = 0.0
    total_opponent_distill = 0.0
    correct_top1 = 0
    correct_topk = 0
    initial_correct_top1 = 0
    initial_correct_topk = 0
    reciprocal_rank_sum = 0.0
    initial_reciprocal_rank_sum = 0.0
    teacher_probability_sum = 0.0
    rollback_count = 0
    rollback_example_count = 0
    rollback_step_sum = 0.0
    total_trace_steps = 0
    top1_changed_count = 0
    teacher_rank_improved_count = 0
    teacher_rank_degraded_count = 0
    step_rank_transition_count = 0
    step_rank_improved_count = 0
    step_rank_degraded_count = 0
    step_rank_delta_sum = 0.0
    step_utility_count = 0
    step_utility_continue_count = 0
    step_utility_predicted_continue_sum = 0.0
    root_incorrect_examples = 0
    root_incorrect_improvement_count = 0
    root_correct_examples = 0
    root_correct_degraded_count = 0
    teacher_rank_delta_sum = 0.0
    step_histogram: dict[str, int] = {}
    phase_usage: dict[str, int] = {}
    phase_value_loss_sums: dict[str, float] = {}
    phase_policy_loss_sums: dict[str, float] = {}
    reply_consistency_stats = {
        "count": 0.0,
        "sum_student": 0.0,
        "sum_teacher": 0.0,
        "sum_student_sq": 0.0,
        "sum_teacher_sq": 0.0,
        "sum_product": 0.0,
    }
    frontier_stats = {
        "transition_count": 0.0,
        "turnover_sum": 0.0,
        "revisit_sum": 0.0,
        "stable_count": 0.0,
        "coverage_count": 0.0,
        "unique_coverage_sum": 0.0,
        "top1_hit_count": 0.0,
        "top1_total": 0.0,
    }
    frontier_state_drift_sum = 0.0
    frontier_memory_norm_sum = 0.0
    frontier_update_gate_sum = 0.0
    frontier_reply_pressure_sum = 0.0
    frontier_reply_uncertainty_sum = 0.0
    frontier_interaction_norm_sum = 0.0

    order = list(range(len(examples)))
    if training:
        random.Random(seed).shuffle(order)
    total_batches = max((len(order) + batch_size - 1) // batch_size, 1)
    progress_interval = _progress_log_interval(
        total_batches=total_batches,
        training=training,
        configured_interval=optimization.log_interval_batches,
    )

    print(
        "[lapv1-train] "
        f"epoch={epoch}/{total_epochs} "
        f"phase={'train' if training else 'validation'} "
        f"stage={stage} "
        f"batches={total_batches} "
        f"examples={len(examples)} "
        f"batch_size={batch_size}",
        flush=True,
    )

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch_start in enumerate(range(0, len(order), batch_size), start=1):
            batch_examples = [examples[index] for index in order[batch_start : batch_start + batch_size]]
            batch = _collate_examples(batch_examples)
            outputs = model(
                batch["piece_tokens"],
                batch["square_tokens"],
                batch["state_context_global"],
                batch["reachability_edges"],
                batch["candidate_features"],
                batch["candidate_action_indices"],
                batch["candidate_mask"],
                phase_index=batch["phase_index"],
                side_to_move=batch["side_to_move"],
                nnue_feat_white_indices=batch["nnue_feat_white_indices"],
                nnue_feat_white_offsets=batch["nnue_feat_white_offsets"],
                nnue_feat_black_indices=batch["nnue_feat_black_indices"],
                nnue_feat_black_offsets=batch["nnue_feat_black_offsets"],
                candidate_move_types=batch["candidate_move_types"],
                candidate_delta_white_leave_indices=batch["candidate_delta_white_leave_indices"],
                candidate_delta_white_leave_offsets=batch["candidate_delta_white_leave_offsets"],
                candidate_delta_white_enter_indices=batch["candidate_delta_white_enter_indices"],
                candidate_delta_white_enter_offsets=batch["candidate_delta_white_enter_offsets"],
                candidate_delta_black_leave_indices=batch["candidate_delta_black_leave_indices"],
                candidate_delta_black_leave_offsets=batch["candidate_delta_black_leave_offsets"],
                candidate_delta_black_enter_indices=batch["candidate_delta_black_enter_indices"],
                candidate_delta_black_enter_offsets=batch["candidate_delta_black_enter_offsets"],
                candidate_nnue_feat_white_after_move_indices=batch["candidate_nnue_feat_white_after_move_indices"],
                candidate_nnue_feat_white_after_move_offsets=batch["candidate_nnue_feat_white_after_move_offsets"],
                candidate_nnue_feat_black_after_move_indices=batch["candidate_nnue_feat_black_after_move_indices"],
                candidate_nnue_feat_black_after_move_offsets=batch["candidate_nnue_feat_black_after_move_offsets"],
                candidate_has_king_move=batch["candidate_has_king_move"],
                collect_opponent_distill=(
                    stage == "T2" and model.config.lapv2.enabled and model.config.lapv2.distill_opponent
                ),
            )
            initial_logits = outputs["initial_policy_logits"]
            logits = outputs["final_policy_logits"]
            wdl_logits = outputs["final_value"]["wdl_logits"]
            cp_score = outputs["final_value"]["cp_score"].squeeze(1)
            sigma_value = outputs["final_value"]["sigma_value"].squeeze(1)
            sharpness = outputs["root_sharpness"]
            if not torch.isfinite(sharpness).all():
                raise RuntimeError(
                    "non-finite sharpness probabilities encountered during LAPv1 training"
                )
            sharpness = sharpness.clamp(1e-6, 1.0 - 1e-6)
            piece_role_logits = aux_probe(outputs["piece_intentions"])
            step_sharpness_tensors = tuple(outputs["step_sharpness_tensors"])
            step_value_cp_tensors = tuple(outputs["step_value_cp_tensors"])
            step_candidate_score_tensors = tuple(outputs["step_candidate_score_tensors"])
            step_active_masks = tuple(outputs["step_active_masks"])
            step_rollback_masks = tuple(outputs["step_rollback_masks"])
            step_student_reply_logits_tensors = tuple(
                outputs["step_student_reply_logits_tensors"]
            )
            step_student_pressure_tensors = tuple(
                outputs["step_student_pressure_tensors"]
            )
            step_student_uncertainty_tensors = tuple(
                outputs["step_student_uncertainty_tensors"]
            )
            step_teacher_reply_logits_tensors = tuple(
                outputs["step_teacher_reply_logits_tensors"]
            )
            step_teacher_pressure_tensors = tuple(
                outputs["step_teacher_pressure_tensors"]
            )
            step_teacher_uncertainty_tensors = tuple(
                outputs["step_teacher_uncertainty_tensors"]
            )
            phase_weights = _phase_load_balance_weights(
                batch["phase_index"],
                enabled=(
                    stage == "T2"
                    and stage2 is not None
                    and stage2.phase_load_balance
                    and model.config.lapv2.enabled
                ),
            )
            example_weights = _lapv1_example_weights(
                batch["curriculum_priorities"],
                phase_weights=phase_weights,
                curriculum_priority_weight=optimization.curriculum_priority_weight,
            )

            value_wdl_per_example = torch.nn.functional.cross_entropy(
                wdl_logits,
                batch["teacher_wdl_target"],
                reduction="none",
            )
            value_wdl_loss = _weighted_mean(value_wdl_per_example, example_weights)
            value_cp_per_example = torch.nn.functional.mse_loss(
                cp_score / _CP_TARGET_SCALE,
                batch["teacher_root_value_cp"] / _CP_TARGET_SCALE,
                reduction="none",
            )
            value_cp_loss = _weighted_mean(value_cp_per_example, example_weights)
            sharpness_per_example = torch.nn.functional.binary_cross_entropy(
                sharpness,
                batch["sharpness_target"],
                reduction="none",
            )
            sharpness_loss = _weighted_mean(sharpness_per_example, example_weights)
            sharpness_target_loss = _trace_sharpness_target_loss(
                step_sharpness_tensors,
                batch["sharpness_target"],
                step_active_masks,
                example_weights=example_weights,
            )
            policy_ce_per_example = torch.nn.functional.cross_entropy(
                logits,
                batch["teacher_top1_candidate_index"],
                reduction="none",
            )
            policy_ce_loss = _weighted_mean(policy_ce_per_example, example_weights)
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            policy_kl_per_example = torch.sum(
                batch["teacher_policy"]
                * (
                    torch.log(batch["teacher_policy"].clamp_min(1e-8))
                    - log_probs
                ),
                dim=1,
            )
            policy_kl_loss = _weighted_mean(policy_kl_per_example, example_weights)
            policy_margin_loss = _policy_margin_loss(
                logits,
                batch["candidate_mask"],
                batch["teacher_top1_candidate_index"],
                batch["teacher_top1_minus_top2_cp"],
            )
            policy_rank_loss = _policy_rank_loss(
                logits,
                batch["candidate_mask"],
                batch["teacher_candidate_rank_bucket_targets"],
            )
            intention_aux_loss = torch.nn.functional.cross_entropy(
                piece_role_logits.reshape(-1, _PIECE_ROLE_CLASS_COUNT),
                batch["piece_role_targets"].reshape(-1),
            )
            if step_value_cp_tensors:
                deliberation_monotonicity_loss = _deliberation_monotonicity_loss(
                    step_value_cp_tensors,
                    step_active_masks,
                    step_rollback_masks,
                    example_weights=example_weights,
                )
            else:
                deliberation_monotonicity_loss = torch.zeros(
                    (),
                    dtype=logits.dtype,
                    device=logits.device,
                )
            deliberation_step_policy_loss = _trace_policy_ce_loss(
                step_candidate_score_tensors,
                batch["teacher_top1_candidate_index"],
                step_active_masks,
                example_weights=example_weights,
            )
            if stage == "T2" and stage2 is not None:
                deliberation_improvement_loss = _improvement_over_root_loss(
                    initial_logits=initial_logits,
                    final_logits=logits,
                    teacher_top1_candidate_index=batch["teacher_top1_candidate_index"],
                    candidate_mask=batch["candidate_mask"],
                    step_candidate_score_tensors=step_candidate_score_tensors,
                    step_active_masks=step_active_masks,
                    example_weights=example_weights,
                )
                deliberation_rank_progress_loss = _step_rank_progress_loss(
                    initial_logits=initial_logits,
                    final_logits=logits,
                    teacher_top1_candidate_index=batch["teacher_top1_candidate_index"],
                    candidate_mask=batch["candidate_mask"],
                    step_candidate_score_tensors=step_candidate_score_tensors,
                    step_active_masks=step_active_masks,
                    example_weights=example_weights,
                )
                deliberation_step_utility_loss = _trace_step_utility_loss(
                    step_sharpness_tensors=step_sharpness_tensors,
                    initial_logits=initial_logits,
                    teacher_top1_candidate_index=batch["teacher_top1_candidate_index"],
                    candidate_mask=batch["candidate_mask"],
                    step_candidate_score_tensors=step_candidate_score_tensors,
                    step_active_masks=step_active_masks,
                    example_weights=example_weights,
                )
            else:
                deliberation_improvement_loss = torch.zeros(
                    (),
                    dtype=logits.dtype,
                    device=logits.device,
                )
                deliberation_rank_progress_loss = torch.zeros(
                    (),
                    dtype=logits.dtype,
                    device=logits.device,
                )
                deliberation_step_utility_loss = torch.zeros(
                    (),
                    dtype=logits.dtype,
                    device=logits.device,
                )
            if (
                stage == "T2"
                and model.config.lapv2.enabled
                and model.config.lapv2.distill_opponent
                and _should_apply_opponent_distill(
                    training=training,
                    seed=seed,
                    batch_index=batch_index,
                    fraction=model.config.lapv2.distill_fraction,
                )
            ):
                opponent_distill_loss = _opponent_distill_loss(
                    step_student_reply_logits_tensors=step_student_reply_logits_tensors,
                    step_student_pressure_tensors=step_student_pressure_tensors,
                    step_student_uncertainty_tensors=step_student_uncertainty_tensors,
                    step_teacher_reply_logits_tensors=step_teacher_reply_logits_tensors,
                    step_teacher_pressure_tensors=step_teacher_pressure_tensors,
                    step_teacher_uncertainty_tensors=step_teacher_uncertainty_tensors,
                    step_active_masks=step_active_masks,
                    reply_weight=model.config.lapv2.distill_reply_weight,
                    pressure_weight=model.config.lapv2.distill_pressure_weight,
                    uncertainty_weight=model.config.lapv2.distill_uncertainty_weight,
                    example_weights=example_weights,
                )
            else:
                opponent_distill_loss = torch.zeros(
                    (),
                    dtype=logits.dtype,
                    device=logits.device,
                )

            value_shared_loss = (
                optimization.value_wdl_weight * value_wdl_loss
                + optimization.value_cp_weight * value_cp_loss
            )
            policy_shared_loss = (
                optimization.policy_ce_weight * policy_ce_loss
                + optimization.policy_kl_weight * policy_kl_loss
                + optimization.policy_margin_weight * policy_margin_loss
                + optimization.policy_rank_weight * policy_rank_loss
            )
            per_example_value_branch = (
                optimization.value_wdl_weight * value_wdl_per_example.detach()
                + optimization.value_cp_weight * value_cp_per_example.detach()
            )
            per_example_policy_branch = (
                optimization.policy_ce_weight * policy_ce_per_example.detach()
                + optimization.policy_kl_weight * policy_kl_per_example.detach()
                + (
                    optimization.policy_margin_weight
                    * float(policy_margin_loss.detach().item())
                )
                + (
                    optimization.policy_rank_weight
                    * float(policy_rank_loss.detach().item())
                )
            )
            if model.config.lapv2.nnue_policy_enabled and loss_balance_state is not None:
                value_shared_loss = _normalize_lapv2_shared_loss(
                    raw_loss=value_shared_loss,
                    state=loss_balance_state,
                    key="value_shared",
                    mode=model.config.lapv2.loss_balance.value_loss_norm,
                    training=training,
                )
                policy_shared_loss = _normalize_lapv2_shared_loss(
                    raw_loss=policy_shared_loss,
                    state=loss_balance_state,
                    key="policy_shared",
                    mode=model.config.lapv2.loss_balance.policy_loss_norm,
                    training=training,
                )
            adapter_decoupling_loss = (
                _lapv2_adapter_decoupling_loss(model)
                * model.config.lapv2.loss_balance.adapter_decoupling
            )
            loss = (
                value_shared_loss
                + optimization.sharpness_weight * sharpness_loss
                + optimization.sharpness_target_loss_weight * sharpness_target_loss
                + policy_shared_loss
                + optimization.intention_aux_weight * intention_aux_loss
                + optimization.deliberation_monotonicity_weight * deliberation_monotonicity_loss
                + optimization.deliberation_step_policy_weight * deliberation_step_policy_loss
                + optimization.deliberation_improvement_weight * deliberation_improvement_loss
                + optimization.deliberation_rank_progress_weight * deliberation_rank_progress_loss
                + optimization.deliberation_step_utility_weight * deliberation_step_utility_loss
                + opponent_distill_loss
                + adapter_decoupling_loss
            )

            if training:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                if optimization.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(aux_probe.parameters()),
                        max_norm=optimization.max_grad_norm,
                    )
                optimizer.step()
                if optimizer_step_counter is not None:
                    if _lapv2_phase_gate_should_mean_pull(
                        model,
                        stage2=stage2,
                        optimizer_step=optimizer_step_counter[0],
                    ):
                        _apply_lapv2_phase_gate_mean_pull(model)
                    optimizer_step_counter[0] += 1

            probabilities = torch.softmax(logits, dim=1)
            top1_indices = torch.argmax(logits, dim=1)
            initial_top1_indices = torch.argmax(initial_logits, dim=1)
            topk_indices = torch.topk(logits, k=min(top_k, logits.shape[1]), dim=1).indices
            initial_topk_indices = torch.topk(
                initial_logits,
                k=min(top_k, initial_logits.shape[1]),
                dim=1,
            ).indices
            teacher_ranks = _teacher_ranks(
                logits,
                batch["teacher_top1_candidate_index"],
            )
            initial_teacher_ranks = _teacher_ranks(
                initial_logits,
                batch["teacher_top1_candidate_index"],
            )
            step_rank_progress = _summarize_step_rank_progress(
                initial_logits=initial_logits,
                final_logits=logits,
                teacher_top1_candidate_index=batch["teacher_top1_candidate_index"],
                candidate_mask=batch["candidate_mask"],
                step_candidate_score_tensors=step_candidate_score_tensors,
                step_active_masks=step_active_masks,
            )
            step_utility = _summarize_step_utility_targets(
                step_sharpness_tensors=step_sharpness_tensors,
                initial_logits=initial_logits,
                teacher_top1_candidate_index=batch["teacher_top1_candidate_index"],
                candidate_mask=batch["candidate_mask"],
                step_candidate_score_tensors=step_candidate_score_tensors,
                step_active_masks=step_active_masks,
            )
            for example_index, phase_value in enumerate(batch["phase_index"].tolist()):
                phase_name = _phase_index_name(int(phase_value))
                phase_usage[phase_name] = phase_usage.get(phase_name, 0) + 1
                phase_value_loss_sums[phase_name] = phase_value_loss_sums.get(
                    phase_name,
                    0.0,
                ) + float(per_example_value_branch[example_index].item())
                phase_policy_loss_sums[phase_name] = phase_policy_loss_sums.get(
                    phase_name,
                    0.0,
                ) + float(per_example_policy_branch[example_index].item())
            for student_reply_logits, teacher_reply_logits, step_active_mask in zip(
                step_student_reply_logits_tensors,
                step_teacher_reply_logits_tensors,
                step_active_masks,
            ):
                _update_reply_consistency_stats(
                    reply_consistency_stats,
                    student_reply_logits=student_reply_logits,
                    teacher_reply_logits=teacher_reply_logits,
                    step_active_mask=step_active_mask,
                )
            batch_frontier_stats = _summarize_frontier_activity(
                step_selected_candidate_tensors=outputs["step_selected_candidate_tensors"],
                step_active_masks=step_active_masks,
                final_top1_indices=top1_indices,
                candidate_count=batch["candidate_mask"].shape[1],
            )
            for key, value in batch_frontier_stats.items():
                frontier_stats[key] += value
            frontier_state_drift_sum += float(outputs["frontier_state_drift"].sum().item())
            frontier_memory_norm_sum += float(outputs["frontier_memory_norm"].sum().item())
            frontier_update_gate_sum += float(outputs["frontier_update_gate_mean"].sum().item())
            frontier_reply_pressure_sum += float(
                outputs["frontier_reply_pressure_mean"].sum().item()
            )
            frontier_reply_uncertainty_sum += float(
                outputs["frontier_reply_uncertainty_mean"].sum().item()
            )
            frontier_interaction_norm_sum += float(
                outputs["frontier_interaction_norm_mean"].sum().item()
            )
            total_examples += len(batch_examples)
            total_loss += float(loss.item()) * len(batch_examples)
            total_value_wdl += float(value_wdl_loss.item()) * len(batch_examples)
            total_value_cp += float(value_cp_loss.item()) * len(batch_examples)
            total_sharpness += float(sharpness_loss.item()) * len(batch_examples)
            total_sharpness_target += float(sharpness_target_loss.item()) * len(batch_examples)
            total_policy_ce += float(policy_ce_loss.item()) * len(batch_examples)
            total_policy_kl += float(policy_kl_loss.item()) * len(batch_examples)
            total_policy_margin += float(policy_margin_loss.item()) * len(batch_examples)
            total_policy_rank += float(policy_rank_loss.item()) * len(batch_examples)
            total_intention_aux += float(intention_aux_loss.item()) * len(batch_examples)
            total_monotonicity += float(deliberation_monotonicity_loss.item()) * len(batch_examples)
            total_step_policy += float(deliberation_step_policy_loss.item()) * len(batch_examples)
            total_improvement += float(deliberation_improvement_loss.item()) * len(batch_examples)
            total_rank_progress += float(deliberation_rank_progress_loss.item()) * len(batch_examples)
            total_step_utility += float(deliberation_step_utility_loss.item()) * len(batch_examples)
            total_opponent_distill += float(opponent_distill_loss.item()) * len(batch_examples)
            correct_top1 += int(
                torch.sum(top1_indices == batch["teacher_top1_candidate_index"]).item()
            )
            correct_topk += int(
                torch.sum(
                    topk_indices == batch["teacher_top1_candidate_index"].unsqueeze(1)
                ).item()
            )
            initial_correct_top1 += int(
                torch.sum(initial_top1_indices == batch["teacher_top1_candidate_index"]).item()
            )
            initial_correct_topk += int(
                torch.sum(
                    initial_topk_indices == batch["teacher_top1_candidate_index"].unsqueeze(1)
                ).item()
            )
            reciprocal_rank_sum += float((1.0 / teacher_ranks.float()).sum().item())
            initial_reciprocal_rank_sum += float(
                (1.0 / initial_teacher_ranks.float()).sum().item()
            )
            teacher_probability_sum += float(
                torch.mean(
                    probabilities.gather(
                        1,
                        batch["teacher_top1_candidate_index"].unsqueeze(1),
                    )
                ).item()
            ) * len(batch_examples)
            top1_changed_count += int(torch.sum(top1_indices != initial_top1_indices).item())
            teacher_rank_improved_count += int(
                torch.sum(teacher_ranks < initial_teacher_ranks).item()
            )
            teacher_rank_degraded_count += int(
                torch.sum(teacher_ranks > initial_teacher_ranks).item()
            )
            root_incorrect_mask = initial_top1_indices != batch["teacher_top1_candidate_index"]
            root_correct_mask = ~root_incorrect_mask
            root_incorrect_examples += int(torch.sum(root_incorrect_mask).item())
            root_incorrect_improvement_count += int(
                torch.sum((teacher_ranks < initial_teacher_ranks) & root_incorrect_mask).item()
            )
            root_correct_examples += int(torch.sum(root_correct_mask).item())
            root_correct_degraded_count += int(
                torch.sum((teacher_ranks > initial_teacher_ranks) & root_correct_mask).item()
            )
            teacher_rank_delta_sum += float(
                torch.sum((initial_teacher_ranks - teacher_ranks).float()).item()
            )
            step_rank_transition_count += int(step_rank_progress["transition_count"])
            step_rank_improved_count += int(step_rank_progress["improved_count"])
            step_rank_degraded_count += int(step_rank_progress["degraded_count"])
            step_rank_delta_sum += float(step_rank_progress["rank_delta_sum"])
            step_utility_count += int(step_utility["step_count"])
            step_utility_continue_count += int(step_utility["continue_count"])
            step_utility_predicted_continue_sum += float(
                step_utility["predicted_continue_sum"]
            )
            if stage == "T2" and stage2 is not None:
                batch_rollbacks = sum(
                    int(mask.sum().item())
                    for mask in step_rollback_masks
                )
                rollback_count += batch_rollbacks
                if step_rollback_masks:
                    rollback_example_count += int(
                        torch.stack(step_rollback_masks, dim=0).any(dim=0).sum().item()
                    )
                rollback_step_sum += sum(
                    float(step_index)
                    * float(mask.sum().item())
                    for step_index, mask in enumerate(step_rollback_masks)
                )
                total_trace_steps += sum(
                    int(mask.sum().item())
                    for mask in step_active_masks
                )
                if step_active_masks:
                    batch_step_counts = torch.stack(step_active_masks, dim=0).sum(dim=0)
                else:
                    batch_step_counts = torch.zeros(
                        len(batch_examples),
                        dtype=torch.long,
                    )
            else:
                batch_step_counts = torch.zeros(
                    len(batch_examples),
                    dtype=torch.long,
                )
            for step_count in batch_step_counts.tolist():
                histogram_key = str(int(step_count))
                step_histogram[histogram_key] = step_histogram.get(histogram_key, 0) + 1
            if batch_index % progress_interval == 0 or batch_index == total_batches:
                elapsed = max(time.perf_counter() - start_time, 1e-9)
                print(
                    "[lapv1-train] "
                    f"epoch={epoch}/{total_epochs} "
                    f"batch={batch_index}/{total_batches} "
                    f"phase={'train' if training else 'validation'} "
                    f"stage={stage} "
                    f"examples={total_examples}/{len(examples)} "
                    f"loss={total_loss / total_examples:.4f} "
                    f"top1={correct_top1 / total_examples:.4f} "
                    f"ex_per_s={total_examples / elapsed:.2f}",
                    flush=True,
                )
            del sigma_value

    duration = max(time.perf_counter() - start_time, 1e-9)
    phase_usage_sorted = {
        phase_name: phase_usage[phase_name]
        for phase_name in sorted(
            phase_usage,
            key=lambda name: (
                _PHASE_NAMES.index(name) if name in _PHASE_NAMES else len(_PHASE_NAMES),
                name,
            ),
        )
    }
    phase_value_loss = {
        phase_name: phase_value_loss_sums[phase_name] / phase_usage_sorted[phase_name]
        for phase_name in phase_usage_sorted
    }
    phase_policy_loss = {
        phase_name: phase_policy_loss_sums[phase_name] / phase_usage_sorted[phase_name]
        for phase_name in phase_usage_sorted
    }
    return LAPv1Metrics(
        total_examples=total_examples,
        supervised_examples=total_examples,
        total_loss=total_loss / total_examples,
        value_wdl_loss=total_value_wdl / total_examples,
        value_cp_loss=total_value_cp / total_examples,
        sharpness_loss=total_sharpness / total_examples,
        sharpness_target_loss=total_sharpness_target / total_examples,
        policy_ce_loss=total_policy_ce / total_examples,
        policy_kl_loss=total_policy_kl / total_examples,
        policy_margin_loss=total_policy_margin / total_examples,
        policy_rank_loss=total_policy_rank / total_examples,
        intention_aux_loss=total_intention_aux / total_examples,
        deliberation_monotonicity_loss=total_monotonicity / total_examples,
        deliberation_step_policy_loss=total_step_policy / total_examples,
        deliberation_improvement_loss=total_improvement / total_examples,
        deliberation_rank_progress_loss=total_rank_progress / total_examples,
        deliberation_step_utility_loss=total_step_utility / total_examples,
        opponent_distill_loss=total_opponent_distill / total_examples,
        root_top1_accuracy=correct_top1 / total_examples,
        root_top3_accuracy=correct_topk / total_examples,
        teacher_root_mean_reciprocal_rank=reciprocal_rank_sum / total_examples,
        teacher_root_mean_probability=teacher_probability_sum / total_examples,
        rollbacks=rollback_count,
        rollback_examples=rollback_example_count,
        mean_rollback_step=(
            0.0 if rollback_count == 0 else rollback_step_sum / rollback_count
        ),
        rollback_hit_rate=(
            0.0 if total_trace_steps == 0 else rollback_count / total_trace_steps
        ),
        rollback_example_rate=rollback_example_count / total_examples,
        initial_root_top1_accuracy=initial_correct_top1 / total_examples,
        initial_root_top3_accuracy=initial_correct_topk / total_examples,
        initial_teacher_root_mean_reciprocal_rank=(
            initial_reciprocal_rank_sum / total_examples
        ),
        top1_changed_rate=top1_changed_count / total_examples,
        teacher_rank_improved_rate=teacher_rank_improved_count / total_examples,
        teacher_rank_degraded_rate=teacher_rank_degraded_count / total_examples,
        step_rank_improved_rate=(
            0.0
            if step_rank_transition_count == 0
            else step_rank_improved_count / step_rank_transition_count
        ),
        step_rank_degraded_rate=(
            0.0
            if step_rank_transition_count == 0
            else step_rank_degraded_count / step_rank_transition_count
        ),
        step_utility_continue_rate=(
            0.0
            if step_utility_count == 0
            else step_utility_continue_count / step_utility_count
        ),
        step_utility_predicted_continue_rate=(
            0.0
            if step_utility_count == 0
            else step_utility_predicted_continue_sum / step_utility_count
        ),
        root_incorrect_improvement_rate=(
            0.0
            if root_incorrect_examples == 0
            else root_incorrect_improvement_count / root_incorrect_examples
        ),
        root_correct_degraded_rate=(
            0.0
            if root_correct_examples == 0
            else root_correct_degraded_count / root_correct_examples
        ),
        mean_teacher_rank_delta=teacher_rank_delta_sum / total_examples,
        mean_step_rank_delta=(
            0.0
            if step_rank_transition_count == 0
            else step_rank_delta_sum / step_rank_transition_count
        ),
        mean_inner_steps_executed=total_trace_steps / total_examples,
        step_histogram=dict(sorted(step_histogram.items(), key=lambda item: int(item[0]))),
        phase_usage=phase_usage_sorted,
        phase_value_loss=phase_value_loss,
        phase_policy_loss=phase_policy_loss,
        ft_drift=_lapv2_ft_drift(model),
        adapter_cosine_distance=_lapv2_adapter_cosine_distance(model),
        reply_consistency=_finalize_reply_consistency(reply_consistency_stats),
        frontier_revisit_rate=(
            0.0
            if frontier_stats["transition_count"] == 0.0
            else frontier_stats["revisit_sum"] / frontier_stats["transition_count"]
        ),
        frontier_turnover_rate=(
            0.0
            if frontier_stats["transition_count"] == 0.0
            else frontier_stats["turnover_sum"] / frontier_stats["transition_count"]
        ),
        frontier_stable_rate=(
            0.0
            if frontier_stats["transition_count"] == 0.0
            else frontier_stats["stable_count"] / frontier_stats["transition_count"]
        ),
        frontier_unique_coverage=(
            0.0
            if frontier_stats["coverage_count"] == 0.0
            else frontier_stats["unique_coverage_sum"] / frontier_stats["coverage_count"]
        ),
        final_top1_frontier_coverage=(
            0.0
            if frontier_stats["top1_total"] == 0.0
            else frontier_stats["top1_hit_count"] / frontier_stats["top1_total"]
        ),
        frontier_state_drift=frontier_state_drift_sum / total_examples,
        frontier_memory_norm=frontier_memory_norm_sum / total_examples,
        frontier_update_gate_mean=frontier_update_gate_sum / total_examples,
        frontier_reply_pressure_mean=frontier_reply_pressure_sum / total_examples,
        frontier_reply_uncertainty_mean=(
            frontier_reply_uncertainty_sum / total_examples
        ),
        frontier_interaction_norm_mean=(
            frontier_interaction_norm_sum / total_examples
        ),
        examples_per_second=total_examples / duration,
    )


def _progress_log_interval(
    *,
    total_batches: int,
    training: bool,
    configured_interval: int,
) -> int:
    if training:
        return max(configured_interval, 1)
    return max(1, min(64, total_batches // 8 if total_batches > 8 else 1))


def _pack_candidate_sparse_rows(
    examples: Sequence[_PreparedLAPv1Example],
    *,
    max_candidate_count: int,
    attribute_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows: list[Sequence[int]] = []
    for example in examples:
        candidate_rows = getattr(example, attribute_name)
        candidate_count = len(example.candidate_action_indices)
        for candidate_index in range(max_candidate_count):
            if candidate_index < candidate_count:
                rows.append(candidate_rows[candidate_index])
            else:
                rows.append(())
    return pack_sparse_feature_lists(rows)


def _collate_examples(examples: Sequence[_PreparedLAPv1Example]) -> dict[str, torch.Tensor]:
    max_candidate_count = max(len(example.candidate_action_indices) for example in examples)
    max_edge_count = max(len(example.reachability_edges) for example in examples)

    piece_tokens = torch.tensor(
        [example.piece_tokens for example in examples],
        dtype=torch.long,
    )
    square_tokens = torch.tensor(
        [example.square_tokens for example in examples],
        dtype=torch.float32,
    )
    state_context_global = torch.tensor(
        [example.state_context_global for example in examples],
        dtype=torch.float32,
    )
    reachability_edges = torch.full(
        (len(examples), max_edge_count, 3),
        -1,
        dtype=torch.long,
    )
    candidate_action_indices = torch.zeros(
        (len(examples), max_candidate_count),
        dtype=torch.long,
    )
    candidate_move_types = torch.zeros(
        (len(examples), max_candidate_count),
        dtype=torch.long,
    )
    candidate_features = torch.zeros(
        (
            len(examples),
            max_candidate_count,
            len(examples[0].candidate_features[0]),
        ),
        dtype=torch.float32,
    )
    candidate_mask = torch.zeros(
        (len(examples), max_candidate_count),
        dtype=torch.bool,
    )
    teacher_policy = torch.zeros(
        (len(examples), max_candidate_count),
        dtype=torch.float32,
    )
    teacher_rank_targets = torch.full(
        (len(examples), max_candidate_count),
        -1,
        dtype=torch.long,
    )
    nnue_feat_white_indices, nnue_feat_white_offsets = pack_sparse_feature_lists(
        [example.nnue_feat_white for example in examples]
    )
    nnue_feat_black_indices, nnue_feat_black_offsets = pack_sparse_feature_lists(
        [example.nnue_feat_black for example in examples]
    )
    candidate_delta_white_leave_indices, candidate_delta_white_leave_offsets = (
        _pack_candidate_sparse_rows(
            examples,
            max_candidate_count=max_candidate_count,
            attribute_name="candidate_delta_white_leave",
        )
    )
    candidate_delta_white_enter_indices, candidate_delta_white_enter_offsets = (
        _pack_candidate_sparse_rows(
            examples,
            max_candidate_count=max_candidate_count,
            attribute_name="candidate_delta_white_enter",
        )
    )
    candidate_delta_black_leave_indices, candidate_delta_black_leave_offsets = (
        _pack_candidate_sparse_rows(
            examples,
            max_candidate_count=max_candidate_count,
            attribute_name="candidate_delta_black_leave",
        )
    )
    candidate_delta_black_enter_indices, candidate_delta_black_enter_offsets = (
        _pack_candidate_sparse_rows(
            examples,
            max_candidate_count=max_candidate_count,
            attribute_name="candidate_delta_black_enter",
        )
    )
    candidate_nnue_feat_white_after_move_indices, candidate_nnue_feat_white_after_move_offsets = (
        _pack_candidate_sparse_rows(
            examples,
            max_candidate_count=max_candidate_count,
            attribute_name="candidate_nnue_feat_white_after_move",
        )
    )
    candidate_nnue_feat_black_after_move_indices, candidate_nnue_feat_black_after_move_offsets = (
        _pack_candidate_sparse_rows(
            examples,
            max_candidate_count=max_candidate_count,
            attribute_name="candidate_nnue_feat_black_after_move",
        )
    )
    candidate_has_king_move = torch.zeros(
        (len(examples), max_candidate_count),
        dtype=torch.bool,
    )

    for batch_index, example in enumerate(examples):
        edge_count = len(example.reachability_edges)
        candidate_count = len(example.candidate_action_indices)
        if edge_count > 0:
            reachability_edges[batch_index, :edge_count, :] = torch.tensor(
                example.reachability_edges,
                dtype=torch.long,
            )
        candidate_action_indices[batch_index, :candidate_count] = torch.tensor(
            example.candidate_action_indices,
            dtype=torch.long,
        )
        candidate_move_types[batch_index, :candidate_count] = torch.tensor(
            example.candidate_move_types,
            dtype=torch.long,
        )
        candidate_features[batch_index, :candidate_count, :] = torch.tensor(
            example.candidate_features,
            dtype=torch.float32,
        )
        candidate_mask[batch_index, :candidate_count] = True
        candidate_has_king_move[batch_index, :candidate_count] = torch.tensor(
            [
                white_flag or black_flag
                for white_flag, black_flag in zip(
                    example.candidate_is_white_king_move,
                    example.candidate_is_black_king_move,
                    strict=True,
                )
            ],
            dtype=torch.bool,
        )
        teacher_policy[batch_index, :candidate_count] = torch.tensor(
            example.teacher_policy,
            dtype=torch.float32,
        )
        if example.teacher_candidate_rank_bucket_targets is not None:
            teacher_rank_targets[batch_index, :candidate_count] = torch.tensor(
                example.teacher_candidate_rank_bucket_targets,
                dtype=torch.long,
            )

    piece_role_targets = torch.where(
        piece_tokens[:, :, 2] >= 0,
        piece_tokens[:, :, 2] + 1,
        torch.zeros_like(piece_tokens[:, :, 2]),
    )
    return {
        "piece_tokens": piece_tokens,
        "square_tokens": square_tokens,
        "state_context_global": state_context_global,
        "side_to_move": torch.tensor(
            [example.side_to_move for example in examples],
            dtype=torch.long,
        ),
        "phase_index": torch.tensor(
            [example.phase_index for example in examples],
            dtype=torch.long,
        ),
        "reachability_edges": reachability_edges,
        "nnue_feat_white_indices": nnue_feat_white_indices,
        "nnue_feat_white_offsets": nnue_feat_white_offsets,
        "nnue_feat_black_indices": nnue_feat_black_indices,
        "nnue_feat_black_offsets": nnue_feat_black_offsets,
        "candidate_action_indices": candidate_action_indices,
        "candidate_move_types": candidate_move_types,
        "candidate_delta_white_leave_indices": candidate_delta_white_leave_indices,
        "candidate_delta_white_leave_offsets": candidate_delta_white_leave_offsets,
        "candidate_delta_white_enter_indices": candidate_delta_white_enter_indices,
        "candidate_delta_white_enter_offsets": candidate_delta_white_enter_offsets,
        "candidate_delta_black_leave_indices": candidate_delta_black_leave_indices,
        "candidate_delta_black_leave_offsets": candidate_delta_black_leave_offsets,
        "candidate_delta_black_enter_indices": candidate_delta_black_enter_indices,
        "candidate_delta_black_enter_offsets": candidate_delta_black_enter_offsets,
        "candidate_nnue_feat_white_after_move_indices": (
            candidate_nnue_feat_white_after_move_indices
        ),
        "candidate_nnue_feat_white_after_move_offsets": (
            candidate_nnue_feat_white_after_move_offsets
        ),
        "candidate_nnue_feat_black_after_move_indices": (
            candidate_nnue_feat_black_after_move_indices
        ),
        "candidate_nnue_feat_black_after_move_offsets": (
            candidate_nnue_feat_black_after_move_offsets
        ),
        "candidate_has_king_move": candidate_has_king_move,
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "teacher_top1_candidate_index": torch.tensor(
            [example.teacher_top1_candidate_index for example in examples],
            dtype=torch.long,
        ),
        "teacher_policy": teacher_policy,
        "teacher_root_value_cp": torch.tensor(
            [
                max(
                    -_ROOT_VALUE_TARGET_CLIP_CP,
                    min(_ROOT_VALUE_TARGET_CLIP_CP, example.teacher_root_value_cp),
                )
                for example in examples
            ],
            dtype=torch.float32,
        ),
        "teacher_wdl_target": torch.tensor(
            [example.teacher_wdl_target for example in examples],
            dtype=torch.long,
        ),
        "sharpness_target": torch.tensor(
            [example.sharpness_target for example in examples],
            dtype=torch.float32,
        ),
        "teacher_top1_minus_top2_cp": torch.tensor(
            [
                0.0
                if example.teacher_top1_minus_top2_cp is None
                else max(
                    -_ROOT_GAP_TARGET_CLIP_CP,
                    min(_ROOT_GAP_TARGET_CLIP_CP, example.teacher_top1_minus_top2_cp),
                )
                for example in examples
            ],
            dtype=torch.float32,
        ),
        "teacher_candidate_rank_bucket_targets": teacher_rank_targets,
        "curriculum_priorities": torch.tensor(
            [example.curriculum_priority for example in examples],
            dtype=torch.float32,
        ),
        "piece_role_targets": piece_role_targets,
    }


def _policy_margin_loss(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    teacher_top1_candidate_index: torch.Tensor,
    gap_targets_cp: torch.Tensor,
) -> torch.Tensor:
    candidate_counts = candidate_mask.sum(dim=1)
    margin_mask = candidate_counts > 1
    if not bool(margin_mask.any().item()):
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    other_mask = candidate_mask.clone()
    other_mask.scatter_(1, teacher_top1_candidate_index.unsqueeze(1), False)
    other_logits = logits.masked_fill(~other_mask, MASKED_CANDIDATE_LOGIT_VALUE)
    best_other = other_logits.max(dim=1).values
    teacher_logits = logits.gather(1, teacher_top1_candidate_index.unsqueeze(1)).squeeze(1)
    target_margin = gap_targets_cp.clamp(0.0, _ROOT_GAP_TARGET_CLIP_CP) / _GAP_TARGET_SCALE
    raw_loss = torch.nn.functional.smooth_l1_loss(
        teacher_logits - best_other,
        target_margin,
        reduction="none",
    )
    return raw_loss[margin_mask].mean()


def _normalize_lapv2_shared_loss(
    *,
    raw_loss: torch.Tensor,
    state: dict[str, float],
    key: str,
    mode: str,
    training: bool,
    momentum: float = 0.99,
) -> torch.Tensor:
    if mode == "none":
        return raw_loss
    detached_value = max(float(raw_loss.detach().item()), 1e-6)
    running_mean = state.get(key, detached_value)
    if training:
        running_mean = (momentum * running_mean) + ((1.0 - momentum) * detached_value)
        state[key] = max(running_mean, 1e-6)
    denominator = max(state.get(key, running_mean), 1e-6)
    return raw_loss / denominator


def _phase_load_balance_weights(
    phase_index: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    if phase_index.ndim != 1:
        raise ValueError("phase_index must be rank-1 for phase load balancing")
    weights = torch.ones_like(phase_index, dtype=torch.float32)
    if not enabled or phase_index.numel() == 0:
        return weights
    for phase_value in phase_index.unique(sorted=True).tolist():
        phase_mask = phase_index == int(phase_value)
        frequency = float(phase_mask.float().mean().item())
        raw_weight = max(1.0 / max(frequency, 1e-6), 0.5)
        weights = torch.where(
            phase_mask,
            weights.new_full(weights.shape, raw_weight),
            weights,
        )
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def _phase_weighted_mean(
    per_example_loss: torch.Tensor,
    phase_weights: torch.Tensor,
) -> torch.Tensor:
    if per_example_loss.ndim != 1:
        raise ValueError("per_example_loss must be rank-1 for phase weighting")
    if phase_weights.shape != per_example_loss.shape:
        raise ValueError("phase_weights must align with per_example_loss")
    return torch.sum(per_example_loss * phase_weights) / phase_weights.sum().clamp_min(1e-6)


def _weighted_mean(
    per_example_loss: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if per_example_loss.ndim != 1:
        raise ValueError("per_example_loss must be rank-1 for weighting")
    if weights.shape != per_example_loss.shape:
        raise ValueError("weights must align with per_example_loss")
    return torch.sum(per_example_loss * weights) / weights.sum().clamp_min(1e-6)


def _lapv1_example_weights(
    curriculum_priorities: torch.Tensor,
    *,
    phase_weights: torch.Tensor,
    curriculum_priority_weight: float,
) -> torch.Tensor:
    if curriculum_priorities.shape != phase_weights.shape:
        raise ValueError("curriculum_priorities and phase_weights must align")
    weights = phase_weights.to(dtype=torch.float32)
    if curriculum_priority_weight > 0.0:
        curriculum_weights = 1.0 + curriculum_priority_weight * torch.log1p(
            curriculum_priorities.clamp_min(0.0)
        )
        curriculum_weights = curriculum_weights / curriculum_weights.mean().clamp_min(1e-6)
        weights = weights * curriculum_weights
    return weights / weights.mean().clamp_min(1e-6)


def _masked_weighted_mean(
    per_example: torch.Tensor,
    mask: torch.Tensor,
    *,
    example_weights: torch.Tensor | None,
) -> torch.Tensor:
    if per_example.ndim != 1:
        raise ValueError("per_example must be rank-1")
    if mask.shape != per_example.shape:
        raise ValueError("mask must align with per_example")
    if example_weights is None:
        return per_example[mask].mean()
    if example_weights.shape != per_example.shape:
        raise ValueError("example_weights must align with per_example")
    masked_weights = example_weights[mask]
    return torch.sum(per_example[mask] * masked_weights) / masked_weights.sum().clamp_min(1e-6)


def _phase_index_name(phase_index: int) -> str:
    if 0 <= phase_index < len(_PHASE_NAMES):
        return _PHASE_NAMES[phase_index]
    return f"phase_{phase_index}"


def _format_phase_metric_map(
    phase_metrics: Mapping[str, float | int],
    *,
    precision: int | None = None,
) -> str:
    ordered_keys = list(_PHASE_NAMES) + sorted(
        key for key in phase_metrics if key not in _PHASE_NAMES
    )
    rendered: list[str] = []
    for key in ordered_keys:
        if key not in phase_metrics:
            continue
        value = phase_metrics[key]
        if isinstance(value, float) and precision is not None:
            rendered.append(f"{key}:{value:.{precision}f}")
        else:
            rendered.append(f"{key}:{value}")
    return ",".join(rendered) if rendered else "none"


def _lapv2_ft_drift(model: LAPv1Model) -> float | None:
    if model.ft is None or not hasattr(model.ft, "experts"):
        return None
    experts = list(model.ft.experts)
    if len(experts) <= 1:
        return 0.0
    weights = torch.stack(
        [expert.ft.weight.detach().reshape(-1) for expert in experts],
        dim=0,
    )
    mean_weight = weights.mean(dim=0, keepdim=True)
    drift = torch.linalg.vector_norm(weights - mean_weight, dim=1).mean()
    return float(drift.item())


def _lapv2_adapter_cosine_distance(model: LAPv1Model) -> float | None:
    if (
        not model.config.lapv2.nnue_policy_enabled
        or model.policy_head_nnue is None
        or model.value_head_nnue is None
    ):
        return None
    value_adapters = _lapv2_policy_value_adapters(model.value_head_nnue)
    policy_adapters = _lapv2_policy_value_adapters(model.policy_head_nnue)
    distances: list[float] = []
    for value_adapter, policy_adapter in zip(value_adapters, policy_adapters, strict=True):
        cosine = torch.nn.functional.cosine_similarity(
            value_adapter.detach().reshape(1, -1),
            policy_adapter.detach().reshape(1, -1),
            dim=1,
        ).mean()
        distances.append(float((1.0 - cosine).item()))
    if not distances:
        return None
    return sum(distances) / len(distances)


def _summarize_frontier_activity(
    *,
    step_selected_candidate_tensors: Sequence[torch.Tensor],
    step_active_masks: Sequence[torch.Tensor],
    final_top1_indices: torch.Tensor,
    candidate_count: int,
) -> dict[str, float]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if len(step_selected_candidate_tensors) != len(step_active_masks):
        raise ValueError(
            "step_selected_candidate_tensors and step_active_masks must align"
        )
    if not step_selected_candidate_tensors:
        return {
            "transition_count": 0.0,
            "turnover_sum": 0.0,
            "revisit_sum": 0.0,
            "stable_count": 0.0,
            "coverage_count": 0.0,
            "unique_coverage_sum": 0.0,
            "top1_hit_count": 0.0,
            "top1_total": 0.0,
        }

    batch_size = final_top1_indices.shape[0]
    previous_selected_mask = torch.zeros(
        (batch_size, candidate_count),
        dtype=torch.bool,
        device=final_top1_indices.device,
    )
    visited_mask = torch.zeros_like(previous_selected_mask)
    total_selected = torch.zeros((batch_size,), dtype=torch.float32, device=final_top1_indices.device)
    unique_selected = torch.zeros_like(total_selected)
    examples_with_steps = torch.zeros((batch_size,), dtype=torch.bool, device=final_top1_indices.device)
    transition_count = 0.0
    turnover_sum = 0.0
    revisit_sum = 0.0
    stable_count = 0.0
    top1_hit_count = 0.0
    top1_total = 0.0

    for selected_indices, step_active_mask in zip(
        step_selected_candidate_tensors,
        step_active_masks,
        strict=True,
    ):
        selected_mask = torch.zeros_like(previous_selected_mask)
        selected_mask.scatter_(
            1,
            selected_indices,
            step_active_mask.unsqueeze(1).expand(-1, selected_indices.shape[1]),
        )
        examples_with_steps |= step_active_mask
        selected_counts = selected_mask.sum(dim=1).to(torch.float32)
        total_selected += selected_counts
        new_selected_mask = selected_mask & ~visited_mask
        unique_selected += new_selected_mask.sum(dim=1).to(torch.float32)
        visited_mask = visited_mask | selected_mask
        top1_hit_mask = (
            selected_mask.gather(1, final_top1_indices.unsqueeze(1)).squeeze(1)
            & step_active_mask
        )
        top1_hit_count += float(top1_hit_mask.sum().item())
        top1_total += float(step_active_mask.sum().item())
        comparable_mask = step_active_mask & previous_selected_mask.any(dim=1)
        if bool(comparable_mask.any().item()):
            overlap = (selected_mask & previous_selected_mask).sum(dim=1).to(torch.float32)
            revisit_rate = overlap / selected_counts.clamp(min=1.0)
            turnover_rate = 1.0 - revisit_rate
            stable_mask = comparable_mask & (
                selected_mask == previous_selected_mask
            ).all(dim=1)
            transition_count += float(comparable_mask.sum().item())
            revisit_sum += float(revisit_rate[comparable_mask].sum().item())
            turnover_sum += float(turnover_rate[comparable_mask].sum().item())
            stable_count += float(stable_mask.sum().item())
        previous_selected_mask = torch.where(
            step_active_mask.unsqueeze(1),
            selected_mask,
            previous_selected_mask,
        )

    coverage_mask = examples_with_steps & (total_selected > 0.0)
    if bool(coverage_mask.any().item()):
        unique_coverage = unique_selected / total_selected.clamp(min=1.0)
        coverage_count = float(coverage_mask.sum().item())
        unique_coverage_sum = float(unique_coverage[coverage_mask].sum().item())
    else:
        coverage_count = 0.0
        unique_coverage_sum = 0.0
    return {
        "transition_count": transition_count,
        "turnover_sum": turnover_sum,
        "revisit_sum": revisit_sum,
        "stable_count": stable_count,
        "coverage_count": coverage_count,
        "unique_coverage_sum": unique_coverage_sum,
        "top1_hit_count": top1_hit_count,
        "top1_total": top1_total,
    }


def _update_reply_consistency_stats(
    stats: dict[str, float],
    *,
    student_reply_logits: torch.Tensor,
    teacher_reply_logits: torch.Tensor,
    step_active_mask: torch.Tensor,
) -> None:
    if not bool(step_active_mask.any().item()):
        return
    student_values = student_reply_logits.detach()[step_active_mask].reshape(-1)
    teacher_values = teacher_reply_logits.detach()[step_active_mask].reshape(-1)
    if student_values.numel() == 0:
        return
    stats["count"] += float(student_values.numel())
    stats["sum_student"] += float(student_values.sum().item())
    stats["sum_teacher"] += float(teacher_values.sum().item())
    stats["sum_student_sq"] += float(student_values.square().sum().item())
    stats["sum_teacher_sq"] += float(teacher_values.square().sum().item())
    stats["sum_product"] += float((student_values * teacher_values).sum().item())


def _finalize_reply_consistency(stats: Mapping[str, float]) -> float | None:
    count = stats.get("count", 0.0)
    if count <= 1.0:
        return None
    sum_student = stats.get("sum_student", 0.0)
    sum_teacher = stats.get("sum_teacher", 0.0)
    sum_student_sq = stats.get("sum_student_sq", 0.0)
    sum_teacher_sq = stats.get("sum_teacher_sq", 0.0)
    sum_product = stats.get("sum_product", 0.0)
    numerator = (count * sum_product) - (sum_student * sum_teacher)
    denom_left = (count * sum_student_sq) - (sum_student * sum_student)
    denom_right = (count * sum_teacher_sq) - (sum_teacher * sum_teacher)
    denominator = max(denom_left * denom_right, 0.0) ** 0.5
    if denominator <= 1e-12:
        return None
    return numerator / denominator


def _lapv2_adapter_decoupling_loss(model: LAPv1Model) -> torch.Tensor:
    reference_parameter = next(model.parameters())
    if (
        not model.config.lapv2.nnue_policy_enabled
        or model.policy_head_nnue is None
        or model.value_head_nnue is None
    ):
        return reference_parameter.new_zeros(())
    value_adapters = _lapv2_policy_value_adapters(model.value_head_nnue)
    policy_adapters = _lapv2_policy_value_adapters(model.policy_head_nnue)
    penalties: list[torch.Tensor] = []
    for value_adapter, policy_adapter in zip(value_adapters, policy_adapters, strict=True):
        penalties.append(
            torch.nn.functional.cosine_similarity(
                value_adapter.reshape(1, -1),
                policy_adapter.reshape(1, -1),
                dim=1,
            ).pow(2).mean()
        )
    if not penalties:
        return value_adapters[0].new_zeros(())
    return torch.stack(penalties).mean()


def _lapv2_policy_value_adapters(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    experts = getattr(module, "experts", None)
    if experts is None:
        adapter = getattr(module, "adapter", None)
        if adapter is None or not hasattr(adapter, "weight"):
            raise ValueError("expected NNUE head with an adapter linear layer")
        return (adapter.weight,)
    collected: list[torch.Tensor] = []
    for expert in experts:
        adapter = getattr(expert, "adapter", None)
        if adapter is None or not hasattr(adapter, "weight"):
            raise ValueError("expected phase-routed NNUE head with adapter weights")
        collected.append(adapter.weight)
    return tuple(collected)


def _policy_rank_loss(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    teacher_rank_targets: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for batch_index in range(logits.shape[0]):
        valid = candidate_mask[batch_index]
        buckets = teacher_rank_targets[batch_index]
        for left_index in range(logits.shape[1]):
            if not bool(valid[left_index].item()) or int(buckets[left_index].item()) < 0:
                continue
            for right_index in range(logits.shape[1]):
                if not bool(valid[right_index].item()) or int(buckets[right_index].item()) < 0:
                    continue
                if int(buckets[left_index].item()) < int(buckets[right_index].item()):
                    losses.append(
                        torch.nn.functional.softplus(
                            -(logits[batch_index, left_index] - logits[batch_index, right_index])
                        )
                    )
    if not losses:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return torch.stack(losses).mean()


def _trace_policy_ce_loss(
    step_candidate_score_tensors: Sequence[torch.Tensor],
    teacher_top1_candidate_index: torch.Tensor,
    step_active_masks: Sequence[torch.Tensor],
    *,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not step_candidate_score_tensors:
        return torch.zeros(
            (),
            dtype=teacher_top1_candidate_index.dtype,
            device=teacher_top1_candidate_index.device,
        ).float()
    losses: list[torch.Tensor] = []
    for step_logits, step_active in zip(
        step_candidate_score_tensors,
        step_active_masks,
        strict=True,
    ):
        if not bool(step_active.any().item()):
            continue
        per_example = torch.nn.functional.cross_entropy(
            step_logits,
            teacher_top1_candidate_index,
            reduction="none",
        )
        losses.append(
            _masked_weighted_mean(
                per_example,
                step_active,
                example_weights=example_weights,
            )
        )
    if not losses:
        return torch.zeros(
            (),
            dtype=step_candidate_score_tensors[0].dtype,
            device=step_candidate_score_tensors[0].device,
        )
    return torch.stack(losses).mean()


def _improvement_over_root_loss(
    *,
    initial_logits: torch.Tensor,
    final_logits: torch.Tensor,
    teacher_top1_candidate_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    step_candidate_score_tensors: Sequence[torch.Tensor],
    step_active_masks: Sequence[torch.Tensor],
    target_ce_margin: float = 0.05,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if target_ce_margin < 0.0:
        raise ValueError("target_ce_margin must be non-negative")
    candidate_counts = candidate_mask.sum(dim=1)
    root_incorrect_mask = (
        torch.argmax(initial_logits, dim=1) != teacher_top1_candidate_index
    ) & (candidate_counts > 1)
    if not bool(root_incorrect_mask.any().item()):
        return torch.zeros((), dtype=final_logits.dtype, device=final_logits.device)

    initial_ce = torch.nn.functional.cross_entropy(
        initial_logits,
        teacher_top1_candidate_index,
        reduction="none",
    ).detach()
    required_final_ce = torch.clamp(initial_ce - target_ce_margin, min=0.0)

    losses: list[torch.Tensor] = []
    final_ce = torch.nn.functional.cross_entropy(
        final_logits,
        teacher_top1_candidate_index,
        reduction="none",
    )
    losses.append(
        _masked_weighted_mean(
            torch.nn.functional.relu(final_ce - required_final_ce),
            root_incorrect_mask,
            example_weights=example_weights,
        )
    )
    for step_logits, step_active_mask in zip(
        step_candidate_score_tensors,
        step_active_masks,
        strict=True,
    ):
        active_incorrect_mask = root_incorrect_mask & step_active_mask
        if not bool(active_incorrect_mask.any().item()):
            continue
        step_ce = torch.nn.functional.cross_entropy(
            step_logits,
            teacher_top1_candidate_index,
            reduction="none",
        )
        losses.append(
            _masked_weighted_mean(
                torch.nn.functional.relu(step_ce - required_final_ce),
                active_incorrect_mask,
                example_weights=example_weights,
            )
        )
    return torch.stack(losses).mean()


def _step_rank_progress_loss(
    *,
    initial_logits: torch.Tensor,
    final_logits: torch.Tensor,
    teacher_top1_candidate_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    step_candidate_score_tensors: Sequence[torch.Tensor],
    step_active_masks: Sequence[torch.Tensor],
    target_ce_margin: float = 0.03,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train active inner steps to improve over the best earlier teacher score."""
    if target_ce_margin < 0.0:
        raise ValueError("target_ce_margin must be non-negative")
    valid_mask = candidate_mask.sum(dim=1) > 1
    if not bool(valid_mask.any().item()):
        return torch.zeros((), dtype=final_logits.dtype, device=final_logits.device)

    teacher = teacher_top1_candidate_index
    root_incorrect_mask = (torch.argmax(initial_logits, dim=1) != teacher) & valid_mask
    best_ce_so_far = torch.nn.functional.cross_entropy(
        initial_logits,
        teacher,
        reduction="none",
    ).detach()
    losses: list[torch.Tensor] = []

    def append_progress_loss(current_logits: torch.Tensor, active_mask: torch.Tensor) -> None:
        nonlocal best_ce_so_far
        active = active_mask & valid_mask
        if not bool(active.any().item()):
            return
        current_ce = torch.nn.functional.cross_entropy(
            current_logits,
            teacher,
            reduction="none",
        )
        required_margin = torch.where(
            root_incorrect_mask,
            torch.full_like(best_ce_so_far, target_ce_margin),
            torch.zeros_like(best_ce_so_far),
        )
        target_ce = torch.clamp(best_ce_so_far - required_margin, min=0.0)
        losses.append(
            _masked_weighted_mean(
                torch.nn.functional.relu(current_ce - target_ce),
                active,
                example_weights=example_weights,
            )
        )
        best_ce_so_far = torch.where(
            active,
            torch.minimum(best_ce_so_far, current_ce.detach()),
            best_ce_so_far,
        )

    for step_logits, step_active_mask in zip(
        step_candidate_score_tensors,
        step_active_masks,
        strict=True,
    ):
        append_progress_loss(step_logits, step_active_mask)
    append_progress_loss(final_logits, valid_mask)

    if not losses:
        return torch.zeros((), dtype=final_logits.dtype, device=final_logits.device)
    return torch.stack(losses).mean()


def _summarize_step_rank_progress(
    *,
    initial_logits: torch.Tensor,
    final_logits: torch.Tensor,
    teacher_top1_candidate_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    step_candidate_score_tensors: Sequence[torch.Tensor],
    step_active_masks: Sequence[torch.Tensor],
) -> dict[str, float | int]:
    """Summarize whether deeper steps improve teacher rank over the best prior step."""
    valid_mask = candidate_mask.sum(dim=1) > 1
    if not bool(valid_mask.any().item()):
        return {
            "transition_count": 0,
            "improved_count": 0,
            "degraded_count": 0,
            "rank_delta_sum": 0.0,
        }

    best_rank_so_far = _teacher_ranks(
        initial_logits,
        teacher_top1_candidate_index,
    ).detach()
    transition_count = 0
    improved_count = 0
    degraded_count = 0
    rank_delta_sum = 0.0

    def update(current_logits: torch.Tensor, active_mask: torch.Tensor) -> None:
        nonlocal best_rank_so_far
        nonlocal transition_count
        nonlocal improved_count
        nonlocal degraded_count
        nonlocal rank_delta_sum
        active = active_mask & valid_mask
        if not bool(active.any().item()):
            return
        current_rank = _teacher_ranks(
            current_logits,
            teacher_top1_candidate_index,
        ).detach()
        previous_best = best_rank_so_far
        transition_count += int(active.sum().item())
        improved_count += int(torch.sum((current_rank < previous_best) & active).item())
        degraded_count += int(torch.sum((current_rank > previous_best) & active).item())
        rank_delta_sum += float(
            torch.sum((previous_best - current_rank).float()[active]).item()
        )
        best_rank_so_far = torch.where(
            active,
            torch.minimum(best_rank_so_far, current_rank),
            best_rank_so_far,
        )

    for step_logits, step_active_mask in zip(
        step_candidate_score_tensors,
        step_active_masks,
        strict=True,
    ):
        update(step_logits, step_active_mask)
    update(final_logits, valid_mask)
    return {
        "transition_count": transition_count,
        "improved_count": improved_count,
        "degraded_count": degraded_count,
        "rank_delta_sum": rank_delta_sum,
    }


def _trace_step_utility_loss(
    *,
    step_sharpness_tensors: Sequence[torch.Tensor],
    initial_logits: torch.Tensor,
    teacher_top1_candidate_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    step_candidate_score_tensors: Sequence[torch.Tensor],
    step_active_masks: Sequence[torch.Tensor],
    target_ce_margin: float = 0.01,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train step sharpness as a realized continuation-utility predictor."""
    if not step_sharpness_tensors:
        return torch.zeros((), dtype=initial_logits.dtype, device=initial_logits.device)
    if target_ce_margin < 0.0:
        raise ValueError("target_ce_margin must be non-negative")
    targets = _step_utility_targets(
        initial_logits=initial_logits,
        teacher_top1_candidate_index=teacher_top1_candidate_index,
        candidate_mask=candidate_mask,
        step_candidate_score_tensors=step_candidate_score_tensors,
        step_active_masks=step_active_masks,
        target_ce_margin=target_ce_margin,
    )
    losses: list[torch.Tensor] = []
    for sharpness, target, active in zip(
        step_sharpness_tensors,
        targets,
        step_active_masks,
        strict=True,
    ):
        if not bool(active.any().item()):
            continue
        per_example = torch.nn.functional.binary_cross_entropy(
            sharpness.clamp(1e-6, 1.0 - 1e-6),
            target,
            reduction="none",
        )
        losses.append(
            _masked_weighted_mean(
                per_example,
                active,
                example_weights=example_weights,
            )
        )
    if not losses:
        return torch.zeros((), dtype=initial_logits.dtype, device=initial_logits.device)
    return torch.stack(losses).mean()


def _summarize_step_utility_targets(
    *,
    step_sharpness_tensors: Sequence[torch.Tensor],
    initial_logits: torch.Tensor,
    teacher_top1_candidate_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    step_candidate_score_tensors: Sequence[torch.Tensor],
    step_active_masks: Sequence[torch.Tensor],
    target_ce_margin: float = 0.01,
) -> dict[str, float | int]:
    if not step_sharpness_tensors:
        return {
            "step_count": 0,
            "continue_count": 0,
            "predicted_continue_sum": 0.0,
        }
    targets = _step_utility_targets(
        initial_logits=initial_logits,
        teacher_top1_candidate_index=teacher_top1_candidate_index,
        candidate_mask=candidate_mask,
        step_candidate_score_tensors=step_candidate_score_tensors,
        step_active_masks=step_active_masks,
        target_ce_margin=target_ce_margin,
    )
    step_count = 0
    continue_count = 0
    predicted_continue_sum = 0.0
    for sharpness, target, active in zip(
        step_sharpness_tensors,
        targets,
        step_active_masks,
        strict=True,
    ):
        if not bool(active.any().item()):
            continue
        step_count += int(active.sum().item())
        continue_count += int(target[active].sum().item())
        predicted_continue_sum += float(sharpness.detach()[active].sum().item())
    return {
        "step_count": step_count,
        "continue_count": continue_count,
        "predicted_continue_sum": predicted_continue_sum,
    }


def _step_utility_targets(
    *,
    initial_logits: torch.Tensor,
    teacher_top1_candidate_index: torch.Tensor,
    candidate_mask: torch.Tensor,
    step_candidate_score_tensors: Sequence[torch.Tensor],
    step_active_masks: Sequence[torch.Tensor],
    target_ce_margin: float,
) -> tuple[torch.Tensor, ...]:
    valid_mask = candidate_mask.sum(dim=1) > 1
    best_ce_so_far = torch.nn.functional.cross_entropy(
        initial_logits,
        teacher_top1_candidate_index,
        reduction="none",
    ).detach()
    targets: list[torch.Tensor] = []
    for step_logits, step_active_mask in zip(
        step_candidate_score_tensors,
        step_active_masks,
        strict=True,
    ):
        active = step_active_mask & valid_mask
        step_ce = torch.nn.functional.cross_entropy(
            step_logits,
            teacher_top1_candidate_index,
            reduction="none",
        ).detach()
        target = ((best_ce_so_far - step_ce) > target_ce_margin).to(step_logits.dtype)
        target = torch.where(active, target, torch.zeros_like(target))
        targets.append(target)
        best_ce_so_far = torch.where(
            active,
            torch.minimum(best_ce_so_far, step_ce),
            best_ce_so_far,
        )
    return tuple(targets)


def _should_apply_opponent_distill(
    *,
    training: bool,
    seed: int,
    batch_index: int,
    fraction: float,
) -> bool:
    if fraction <= 0.0:
        return False
    if not training:
        return True
    if fraction >= 1.0:
        return True
    return random.Random((seed * 1_000_003) + batch_index).random() < fraction


def _opponent_distill_loss(
    *,
    step_student_reply_logits_tensors: Sequence[torch.Tensor],
    step_student_pressure_tensors: Sequence[torch.Tensor],
    step_student_uncertainty_tensors: Sequence[torch.Tensor],
    step_teacher_reply_logits_tensors: Sequence[torch.Tensor],
    step_teacher_pressure_tensors: Sequence[torch.Tensor],
    step_teacher_uncertainty_tensors: Sequence[torch.Tensor],
    step_active_masks: Sequence[torch.Tensor],
    reply_weight: float,
    pressure_weight: float,
    uncertainty_weight: float,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not step_student_reply_logits_tensors:
        return torch.zeros((), dtype=torch.float32)
    losses: list[torch.Tensor] = []
    for (
        student_reply_logits,
        student_pressure,
        student_uncertainty,
        teacher_reply_logits,
        teacher_pressure,
        teacher_uncertainty,
        step_active_mask,
    ) in zip(
        step_student_reply_logits_tensors,
        step_student_pressure_tensors,
        step_student_uncertainty_tensors,
        step_teacher_reply_logits_tensors,
        step_teacher_pressure_tensors,
        step_teacher_uncertainty_tensors,
        step_active_masks,
        strict=True,
    ):
        if not bool(step_active_mask.any().item()):
            continue
        step_losses: list[torch.Tensor] = []
        if reply_weight > 0.0:
            teacher_reply = torch.softmax(teacher_reply_logits.detach(), dim=2)
            student_log_reply = torch.log_softmax(student_reply_logits, dim=2)
            per_example_reply = torch.sum(
                teacher_reply * (torch.log(teacher_reply.clamp_min(1e-8)) - student_log_reply),
                dim=2,
            ).mean(dim=1)
            step_losses.append(
                reply_weight
                * _masked_weighted_mean(
                    per_example_reply,
                    step_active_mask,
                    example_weights=example_weights,
                )
            )
        if pressure_weight > 0.0:
            per_example_pressure = torch.nn.functional.mse_loss(
                student_pressure,
                teacher_pressure.detach(),
                reduction="none",
            ).mean(dim=1)
            step_losses.append(
                pressure_weight
                * _masked_weighted_mean(
                    per_example_pressure,
                    step_active_mask,
                    example_weights=example_weights,
                )
            )
        if uncertainty_weight > 0.0:
            per_example_uncertainty = torch.nn.functional.mse_loss(
                student_uncertainty,
                teacher_uncertainty.detach(),
                reduction="none",
            ).mean(dim=1)
            step_losses.append(
                uncertainty_weight
                * _masked_weighted_mean(
                    per_example_uncertainty,
                    step_active_mask,
                    example_weights=example_weights,
                )
            )
        if step_losses:
            losses.append(torch.stack(step_losses).sum())
    if not losses:
        reference = step_student_reply_logits_tensors[0]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    return torch.stack(losses).mean()


def _trace_sharpness_target_loss(
    step_sharpness_tensors: Sequence[torch.Tensor],
    sharpness_target: torch.Tensor,
    step_active_masks: Sequence[torch.Tensor],
    *,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not step_sharpness_tensors:
        return torch.zeros((), dtype=sharpness_target.dtype, device=sharpness_target.device)
    losses: list[torch.Tensor] = []
    for step_sharpness, step_active in zip(
        step_sharpness_tensors,
        step_active_masks,
        strict=True,
    ):
        if not bool(step_active.any().item()):
            continue
        per_example = torch.nn.functional.binary_cross_entropy(
            step_sharpness.clamp(1e-6, 1.0 - 1e-6),
            sharpness_target,
            reduction="none",
        )
        losses.append(
            _masked_weighted_mean(
                per_example,
                step_active,
                example_weights=example_weights,
            )
        )
    if not losses:
        return torch.zeros((), dtype=sharpness_target.dtype, device=sharpness_target.device)
    return torch.stack(losses).mean()


def _deliberation_monotonicity_loss(
    step_value_cp_tensors: Sequence[torch.Tensor],
    step_active_masks: Sequence[torch.Tensor],
    step_rollback_masks: Sequence[torch.Tensor],
    *,
    example_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if not step_value_cp_tensors:
        raise ValueError("deliberation_monotonicity_loss requires at least one trace tensor")
    if len(step_value_cp_tensors) < 2:
        reference = step_value_cp_tensors[0]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    penalties: list[torch.Tensor] = []
    for previous_values, current_values, previous_active, current_active, current_rollback in zip(
        step_value_cp_tensors[:-1],
        step_value_cp_tensors[1:],
        step_active_masks[:-1],
        step_active_masks[1:],
        step_rollback_masks[1:],
        strict=True,
    ):
        valid_mask = previous_active & current_active & ~current_rollback
        if not bool(valid_mask.any().item()):
            continue
        penalties.append(
            _masked_weighted_mean(
                torch.nn.functional.relu(previous_values - current_values)
                / _GAP_TARGET_SCALE,
                valid_mask,
                example_weights=example_weights,
            )
        )
    if not penalties:
        reference = step_value_cp_tensors[0]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    return torch.stack(penalties).mean()


def _resolve_stage2_phases(config: LAPv1TrainConfig) -> tuple[_ResolvedStage2Phase, ...]:
    if config.stage == "T1" or config.stage2 is None:
        return ()
    if not config.stage2.phases:
        return (
            _ResolvedStage2Phase(
                name="joint",
                epoch_start=1,
                epoch_end=config.optimization.epochs,
                trainable_parameter_groups=("all",),
                max_inner_steps_schedule=config.stage2.max_inner_steps_schedule,
                min_inner_steps_schedule=(),
                train_paths=(),
                train_path_weights=(),
                train_epoch_examples=None,
                validation_paths=(),
                learning_rate_scale_by_group={},
            ),
        )
    phases: list[_ResolvedStage2Phase] = []
    next_epoch = 1
    for phase in config.stage2.phases:
        phases.append(
            _ResolvedStage2Phase(
                name=phase.name,
                epoch_start=next_epoch,
                epoch_end=next_epoch + phase.epochs - 1,
                trainable_parameter_groups=phase.trainable_parameter_groups,
                max_inner_steps_schedule=phase.max_inner_steps_schedule,
                min_inner_steps_schedule=phase.min_inner_steps_schedule,
                train_paths=phase.train_paths,
                train_path_weights=phase.train_path_weights,
                train_epoch_examples=phase.train_epoch_examples,
                validation_paths=phase.validation_paths,
                learning_rate_scale_by_group=dict(phase.learning_rate_scale_by_group),
            )
        )
        next_epoch += phase.epochs
    return tuple(phases)


def _stage2_phase_for_epoch(
    phases: Sequence[_ResolvedStage2Phase],
    *,
    epoch: int,
) -> _ResolvedStage2Phase:
    for phase in phases:
        if phase.contains_epoch(epoch):
            return phase
    raise ValueError(f"no LAPv1 stage2 phase configured for epoch {epoch}")


def _named_parameter_groups(
    *,
    model: LAPv1Model,
    aux_probe: _PieceRoleAuxProbe,
) -> dict[str, list[torch.nn.Parameter]]:
    inner_delta_parameters = list(
        model.deliberation_loop.cell.candidate_delta_network.parameters()
    )
    inner_delta_parameter_ids = {id(parameter) for parameter in inner_delta_parameters}
    inner_loop_core_parameters = [
        parameter
        for name, parameter in model.deliberation_loop.named_parameters()
        if not name.startswith("cell.candidate_delta_network.")
    ]
    active_opponent_parameters = (
        list(model.opponent_readout.parameters())
        if model.opponent_readout is not None
        else list(model.opponent_head.parameters())
    )
    inner_loop_core_parameters.extend(active_opponent_parameters)
    return {
        "root_backbone": [
            *model.intention_encoder.parameters(),
            *model.state_embedder.parameters(),
            *(list(model.ft.parameters()) if model.ft is not None else []),
        ],
        "root_heads": [
            *model.value_head.parameters(),
            *(list(model.value_head_nnue.parameters()) if model.value_head_nnue is not None else []),
            *model.sharpness_head.parameters(),
            *(
                list(model.policy_head_nnue.parameters())
                if model.policy_head_nnue is not None
                else list(model.policy_head.parameters())
            ),
        ],
        "inner_loop": [
            *model.deliberation_loop.parameters(),
            *active_opponent_parameters,
        ],
        "inner_loop_core": [
            parameter
            for parameter in inner_loop_core_parameters
            if id(parameter) not in inner_delta_parameter_ids
        ],
        "inner_delta_network": inner_delta_parameters,
        "aux_probe": list(aux_probe.parameters()),
    }


def _build_optimizer(
    *,
    model: LAPv1Model,
    aux_probe: _PieceRoleAuxProbe,
    groups: Sequence[str],
    learning_rate_scale_by_group: Mapping[str, float],
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, int]:
    all_model_parameters = list(model.parameters())
    all_aux_parameters = list(aux_probe.parameters())
    for parameter in [*all_model_parameters, *all_aux_parameters]:
        parameter.requires_grad = False

    if "all" in groups:
        selected_parameters: list[torch.nn.Parameter] = []
        selected_ids: set[int] = set()
        for parameter in [*all_model_parameters, *all_aux_parameters]:
            if id(parameter) in selected_ids:
                continue
            selected_ids.add(id(parameter))
            selected_parameters.append(parameter)
        for parameter in selected_parameters:
            parameter.requires_grad = True
        parameter_count = sum(parameter.numel() for parameter in selected_parameters)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": selected_parameters,
                    "lr": learning_rate * float(learning_rate_scale_by_group.get("all", 1.0)),
                    "weight_decay": weight_decay,
                    "name": "all",
                }
            ]
        )
        return optimizer, parameter_count

    named_groups = _named_parameter_groups(model=model, aux_probe=aux_probe)
    selected_by_id: dict[int, torch.nn.Parameter] = {}
    optimizer_groups: list[dict[str, Any]] = []
    for group_name in groups:
        group_parameters: list[torch.nn.Parameter] = []
        group_seen_ids: set[int] = set()
        for parameter in named_groups[group_name]:
            parameter_id = id(parameter)
            if parameter_id in selected_by_id or parameter_id in group_seen_ids:
                continue
            group_seen_ids.add(parameter_id)
            group_parameters.append(parameter)
        if not group_parameters:
            continue
        for parameter in group_parameters:
            parameter.requires_grad = True
            selected_by_id[id(parameter)] = parameter
        optimizer_groups.append(
            {
                "params": group_parameters,
                "lr": learning_rate * float(learning_rate_scale_by_group.get(group_name, 1.0)),
                "weight_decay": weight_decay,
                "name": group_name,
            }
        )
    if not optimizer_groups:
        raise ValueError("no trainable parameters selected for LAPv1 optimizer")
    parameter_count = sum(parameter.numel() for parameter in selected_by_id.values())
    optimizer = torch.optim.AdamW(optimizer_groups)
    return optimizer, parameter_count


def _format_lr_group_scales(
    *,
    groups: Sequence[str],
    learning_rate_scale_by_group: Mapping[str, float],
    base_learning_rate: float,
) -> str:
    if "all" in groups:
        return (
            "all="
            f"{base_learning_rate * float(learning_rate_scale_by_group.get('all', 1.0)):.6g}"
        )
    return ",".join(
        f"{group}="
        f"{base_learning_rate * float(learning_rate_scale_by_group.get(group, 1.0)):.6g}"
        for group in groups
    )


def _collapse_alarm(metrics: LAPv1Metrics) -> dict[str, float | bool] | None:
    delta_top1 = metrics.root_top1_accuracy - metrics.initial_root_top1_accuracy
    delta_mrr = (
        metrics.teacher_root_mean_reciprocal_rank
        - metrics.initial_teacher_root_mean_reciprocal_rank
    )
    collapsed = (
        abs(delta_top1) <= 0.002
        and abs(delta_mrr) <= 0.002
        and metrics.top1_changed_rate <= 0.01
        and metrics.mean_inner_steps_executed <= 1.25
    )
    if not collapsed:
        return None
    return {
        "collapsed": True,
        "delta_top1": round(delta_top1, 6),
        "delta_mrr": round(delta_mrr, 6),
        "top1_changed_rate": round(metrics.top1_changed_rate, 6),
        "mean_inner_steps_executed": round(metrics.mean_inner_steps_executed, 6),
    }


def _build_resampled_dataset(
    *,
    datasets: Sequence[_LazyPreparedLAPv1Dataset],
    weights: Sequence[float],
    total_examples: int,
    seed: int,
) -> _ResampledPreparedLAPv1Dataset:
    if not datasets:
        raise ValueError("resampled dataset requires at least one source dataset")
    if len(datasets) != len(weights):
        raise ValueError("resampled dataset weights must align with datasets")
    if total_examples <= 0:
        raise ValueError("resampled dataset total_examples must be positive")
    if any(weight <= 0.0 for weight in weights):
        raise ValueError("resampled dataset weights must be positive")

    total_weight = float(sum(weights))
    raw_counts = [(total_examples * weight) / total_weight for weight in weights]
    counts = [int(raw_count) for raw_count in raw_counts]
    remaining = total_examples - sum(counts)
    remainders = sorted(
        range(len(weights)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainders[:remaining]:
        counts[index] += 1

    mapping: list[tuple[int, int]] = []
    for dataset_index, (dataset, count) in enumerate(zip(datasets, counts, strict=True)):
        dataset_length = len(dataset)
        if dataset_length == 0:
            raise ValueError("resampled dataset source is empty")
        local_rng = random.Random(seed + 1009 * (dataset_index + 1))
        remaining_count = count
        while remaining_count > 0:
            local_indices = list(range(dataset_length))
            local_rng.shuffle(local_indices)
            take = min(remaining_count, dataset_length)
            mapping.extend(
                (dataset_index, local_indices[position])
                for position in range(take)
            )
            remaining_count -= take
    random.Random(seed).shuffle(mapping)
    return _ResampledPreparedLAPv1Dataset(datasets, mapping)


def _teacher_ranks(
    logits: torch.Tensor,
    teacher_top1_candidate_index: torch.Tensor,
) -> torch.Tensor:
    ranks: list[int] = []
    for row_logits, teacher_index in zip(logits, teacher_top1_candidate_index, strict=True):
        ranked_indices = torch.argsort(row_logits, descending=True)
        rank = int(
            torch.nonzero(ranked_indices == teacher_index, as_tuple=False)[0, 0].item()
        ) + 1
        ranks.append(rank)
    return torch.tensor(ranks, dtype=torch.long, device=teacher_top1_candidate_index.device)


def _is_better_validation(current: LAPv1Metrics, best: LAPv1Metrics) -> bool:
    current_key = (
        current.root_top1_accuracy,
        current.teacher_root_mean_reciprocal_rank,
        -current.total_loss,
    )
    best_key = (
        best.root_top1_accuracy,
        best.teacher_root_mean_reciprocal_rank,
        -best.total_loss,
    )
    return current_key > best_key


def _current_max_inner_steps(
    *,
    phase: _ResolvedStage2Phase,
    epoch: int,
) -> int:
    return _scheduled_phase_value(
        schedule=phase.max_inner_steps_schedule,
        phase=phase,
        epoch=epoch,
    )


def _current_min_inner_steps(
    *,
    phase: _ResolvedStage2Phase,
    epoch: int,
    base_min_inner_steps: int,
    current_max_inner_steps: int,
) -> int:
    if not phase.min_inner_steps_schedule:
        return min(base_min_inner_steps, current_max_inner_steps)
    return min(
        _scheduled_phase_value(
            schedule=phase.min_inner_steps_schedule,
            phase=phase,
            epoch=epoch,
        ),
        current_max_inner_steps,
    )


def _scheduled_phase_value(
    *,
    schedule: Sequence[int],
    phase: _ResolvedStage2Phase,
    epoch: int,
) -> int:
    if phase.epoch_start == phase.epoch_end or len(schedule) == 1:
        return schedule[-1]
    local_epoch_index = epoch - phase.epoch_start
    local_epoch_count = phase.epoch_end - phase.epoch_start
    schedule_position = round(local_epoch_index * (len(schedule) - 1) / local_epoch_count)
    return schedule[schedule_position]


def _datasets_for_phase(
    *,
    config: LAPv1TrainConfig,
    repo_root: Path,
    phase: _ResolvedStage2Phase | None,
    get_dataset: Any,
    epoch: int,
    seed: int,
) -> _ResolvedPhaseDatasets:
    del repo_root
    train_paths = (
        phase.train_paths
        if phase is not None and phase.train_paths
        else tuple(config.data.resolved_train_paths())
    )
    validation_paths = (
        phase.validation_paths
        if phase is not None and phase.validation_paths
        else tuple(config.data.resolved_validation_paths())
    )
    label_suffix = "base" if phase is None else phase.name
    train_path_weights = (
        phase.train_path_weights
        if phase is not None and phase.train_path_weights
        else ()
    )
    train_epoch_examples = phase.train_epoch_examples if phase is not None else None
    if train_path_weights:
        train_sources = [
            get_dataset(
                label=f"train:{label_suffix}:{path_index}",
                paths=(path,),
                log_every_examples=100_000,
            )
            for path_index, path in enumerate(train_paths)
        ]
        train_examples: Sequence[_PreparedLAPv1Example] = _build_resampled_dataset(
            datasets=train_sources,
            weights=train_path_weights,
            total_examples=(
                train_epoch_examples
                if train_epoch_examples is not None
                else max(len(dataset) for dataset in train_sources)
            ),
            seed=seed + epoch,
        )
    else:
        train_examples = get_dataset(
            label=f"train:{label_suffix}",
            paths=train_paths,
            log_every_examples=100_000,
        )
    validation_examples = get_dataset(
        label=f"validation:{label_suffix}",
        paths=validation_paths,
        log_every_examples=25_000,
    )
    return _ResolvedPhaseDatasets(
        train_examples=train_examples,
        validation_examples=validation_examples,
        train_paths=train_paths,
        validation_paths=validation_paths,
        train_path_weights=train_path_weights,
        train_epoch_examples=train_epoch_examples,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _configure_torch_runtime(torch_threads: int) -> None:
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
