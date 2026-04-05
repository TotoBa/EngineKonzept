"""Stage-T1/T2 trainer and evaluation helpers for the model-only LAPv1 wrapper."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass
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
from train.models.lapv1 import LAPV1_MODEL_NAME, LAPv1Config, LAPv1Model
from train.models.policy_head_large import MASKED_CANDIDATE_LOGIT_VALUE
from train.models.proposer import torch_is_available


LAPV1_STAGE1_NAME = "lapv1_stage1"
_PIECE_ROLE_CLASS_COUNT = 7
_CP_TARGET_SCALE = 256.0
_GAP_TARGET_SCALE = 128.0
_ROOT_VALUE_TARGET_CLIP_CP = 1024.0
_ROOT_GAP_TARGET_CLIP_CP = 512.0


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
    intention_aux_weight: float = 0.0
    sharpness_target_loss_weight: float = 0.0
    deliberation_monotonicity_weight: float = 0.0

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
            "intention_aux_weight",
            "sharpness_target_loss_weight",
            "deliberation_monotonicity_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"optimization.{name} must be non-negative")


@dataclass(frozen=True)
class LAPv1Stage2Config:
    """Trainer-only curriculum and auxiliary-loss settings for LAPv1 stage T2."""

    max_inner_steps_schedule: tuple[int, ...] = (2, 4, 8)

    def __post_init__(self) -> None:
        if not self.max_inner_steps_schedule:
            raise ValueError("stage2.max_inner_steps_schedule must be non-empty")
        if any(step <= 0 for step in self.max_inner_steps_schedule):
            raise ValueError("stage2.max_inner_steps_schedule entries must be positive")


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
            if max(self.stage2.max_inner_steps_schedule) > self.model.deliberation.max_inner_steps:
                raise ValueError(
                    "stage2.max_inner_steps_schedule must not exceed model.deliberation.max_inner_steps"
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
            stage2=(
                None
                if payload.get("stage2") is None
                else LAPv1Stage2Config(**dict(payload["stage2"]))
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
    root_top1_accuracy: float
    root_top3_accuracy: float
    teacher_root_mean_reciprocal_rank: float
    teacher_root_mean_probability: float
    rollbacks: int
    mean_rollback_step: float
    rollback_hit_rate: float
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
            "mean_rollback_step": round(self.mean_rollback_step, 6),
            "rollback_hit_rate": round(self.rollback_hit_rate, 6),
            "examples_per_second": round(self.examples_per_second, 3),
        }


@dataclass(frozen=True)
class LAPv1TrainingRun:
    """Serializable result summary for one static-head LAPv1 run."""

    history: list[dict[str, Any]]
    best_epoch: int
    best_validation: dict[str, float | int]
    export_paths: dict[str, str]
    summary_path: str
    model_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_validation": self.best_validation,
            "export_paths": self.export_paths,
            "summary_path": self.summary_path,
            "model_parameter_count": self.model_parameter_count,
        }


_PreparedLAPv1Example = LAPv1TrainingExample


class _LazyPreparedLAPv1Dataset(Sequence[_PreparedLAPv1Example]):
    """Disk-backed planner-head dataset that prepares LAPv1 examples on demand."""

    def __init__(self, paths: Sequence[Path]) -> None:
        self._paths = tuple(paths)
        self._offsets_per_path: list[list[int]] = []
        self._line_numbers_per_path: list[list[int]] = []
        self._source_kinds: list[str] = []
        self._cumulative_sizes: list[int] = []
        self._handles: list[BinaryIO | None] = [None] * len(self._paths)
        running_total = 0
        for path in self._paths:
            offsets: list[int] = []
            line_numbers: list[int] = []
            source_kind = "lapv1"
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
            self._offsets_per_path.append(offsets)
            self._line_numbers_per_path.append(line_numbers)
            self._source_kinds.append(source_kind)
            running_total += len(offsets)
            self._cumulative_sizes.append(running_total)

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

    train_examples = _build_lazy_dataset(
        [resolve_repo_path(repo_root, path) for path in config.data.resolved_train_paths()]
    )
    validation_examples = _build_lazy_dataset(
        [resolve_repo_path(repo_root, path) for path in config.data.resolved_validation_paths()]
    )
    if len(train_examples) == 0:
        raise ValueError("training artifact is empty")
    if len(validation_examples) == 0:
        raise ValueError("validation artifact is empty")

    print(
        "[lapv1-train] "
        f"stage={config.stage} output_dir={output_dir} bundle_dir={bundle_dir} "
        f"epochs={config.optimization.epochs} batch_size={config.optimization.batch_size} "
        f"train_examples={len(train_examples)} validation_examples={len(validation_examples)} "
        "data_access=lazy_jsonl",
        flush=True,
    )

    model = LAPv1Model(config.model)
    aux_probe = _PieceRoleAuxProbe(
        intention_dim=config.model.intention_encoder.intention_dim
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aux_probe.parameters()),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    history: list[dict[str, Any]] = []
    best_epoch = 1
    best_validation: LAPv1Metrics | None = None
    best_model_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    best_aux_state = {
        name: tensor.detach().clone()
        for name, tensor in aux_probe.state_dict().items()
    }

    try:
        for epoch in range(1, config.optimization.epochs + 1):
            current_max_inner_steps = _current_max_inner_steps(
                config=config,
                epoch=epoch,
            )
            model.deliberation_loop.max_inner_steps = current_max_inner_steps
            model.deliberation_loop.min_inner_steps = min(
                config.model.deliberation.min_inner_steps,
                current_max_inner_steps,
            )
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
            )
            history_entry = {
                "epoch": epoch,
                "max_inner_steps": current_max_inner_steps,
                "train": train_metrics.to_dict(),
                "validation": validation_metrics.to_dict(),
            }
            history.append(history_entry)
            print(
                "[lapv1-train] "
                f"epoch={epoch}/{config.optimization.epochs} "
                f"stage={config.stage} "
                f"max_inner_steps={current_max_inner_steps} "
                f"train_loss={train_metrics.total_loss:.4f} "
                f"val_top1={validation_metrics.root_top1_accuracy:.4f} "
                f"val_mrr={validation_metrics.teacher_root_mean_reciprocal_rank:.4f} "
                f"rollbacks={validation_metrics.rollbacks} "
                f"rollback_hit_rate={validation_metrics.rollback_hit_rate:.4f}",
                flush=True,
            )

            if best_validation is None or _is_better_validation(
                validation_metrics,
                best_validation,
            ):
                best_epoch = epoch
                best_validation = validation_metrics
                best_model_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                best_aux_state = {
                    name: tensor.detach().clone()
                    for name, tensor in aux_probe.state_dict().items()
                }
    finally:
        train_examples.close()
        validation_examples.close()

    assert best_validation is not None
    model.load_state_dict(best_model_state)
    aux_probe.load_state_dict(best_aux_state)

    checkpoint_path = bundle_dir / config.export.checkpoint_name
    torch.save(
        {
            "model_name": LAPV1_MODEL_NAME,
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

    training_config = dict(payload["training_config"])
    stage = str(training_config["stage"])
    stage2_payload = training_config.get("stage2")
    stage2 = (
        None if stage2_payload is None else LAPv1Stage2Config(**dict(stage2_payload))
    )
    lapv1_config = LAPv1Config.from_mapping(dict(training_config["model"]))
    model = LAPv1Model(lapv1_config)
    if stage == "T2" and stage2 is not None:
        model.deliberation_loop.max_inner_steps = max(stage2.max_inner_steps_schedule)
        model.deliberation_loop.min_inner_steps = min(
            lapv1_config.deliberation.min_inner_steps,
            model.deliberation_loop.max_inner_steps,
        )
    model.load_state_dict(dict(payload["model_state_dict"]))
    aux_probe = _PieceRoleAuxProbe(
        intention_dim=lapv1_config.intention_encoder.intention_dim
    )
    aux_state_dict = payload.get("aux_state_dict")
    if isinstance(aux_state_dict, dict):
        aux_probe.load_state_dict(dict(aux_state_dict))

    effective_dataset_path = (
        Path(dataset_path)
        if dataset_path is not None
        else Path(str(training_config["data"]["validation_path"]))
    )
    examples = _build_lazy_dataset([effective_dataset_path])
    optimization = LAPv1OptimizationConfig(
        **dict(training_config["optimization"])
    )
    try:
        return _run_epoch(
            model=model,
            aux_probe=aux_probe,
            examples=examples,
            batch_size=int(training_config["optimization"]["batch_size"]),
            optimizer=None,
            training=False,
            seed=int(training_config["seed"]),
            optimization=optimization,
            top_k=top_k,
            stage=stage,
            stage2=stage2,
            epoch=1,
            total_epochs=1,
        )
    finally:
        examples.close()


def _build_lazy_dataset(paths: Sequence[Path]) -> _LazyPreparedLAPv1Dataset:
    return _LazyPreparedLAPv1Dataset(paths)


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
    correct_top1 = 0
    correct_topk = 0
    reciprocal_rank_sum = 0.0
    teacher_probability_sum = 0.0
    rollback_count = 0
    rollback_step_sum = 0.0
    total_trace_steps = 0

    order = list(range(len(examples)))
    if training:
        random.Random(seed).shuffle(order)
    total_batches = max((len(order) + batch_size - 1) // batch_size, 1)

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
            )
            logits = outputs["final_policy_logits"]
            wdl_logits = outputs["final_value"]["wdl_logits"]
            cp_score = outputs["final_value"]["cp_score"].squeeze(1)
            sigma_value = outputs["final_value"]["sigma_value"].squeeze(1)
            sharpness = model.sharpness_head(outputs["z_root"]).squeeze(1)
            if not torch.isfinite(sharpness).all():
                raise RuntimeError(
                    "non-finite sharpness probabilities encountered during LAPv1 training"
                )
            sharpness = sharpness.clamp(1e-6, 1.0 - 1e-6)
            piece_role_logits = aux_probe(outputs["piece_intentions"])
            step_sharpness_tensors = tuple(outputs["step_sharpness_tensors"])
            step_value_cp_tensors = tuple(outputs["step_value_cp_tensors"])
            step_rollback_flags = tuple(outputs["step_rollback_flags"])

            value_wdl_loss = torch.nn.functional.cross_entropy(
                wdl_logits,
                batch["teacher_wdl_target"],
            )
            value_cp_loss = torch.nn.functional.mse_loss(
                cp_score / _CP_TARGET_SCALE,
                batch["teacher_root_value_cp"] / _CP_TARGET_SCALE,
            )
            sharpness_loss = torch.nn.functional.binary_cross_entropy(
                sharpness,
                batch["sharpness_target"],
            )
            sharpness_target_loss = _trace_sharpness_target_loss(
                step_sharpness_tensors,
                batch["sharpness_target"],
            )
            policy_ce_loss = torch.nn.functional.cross_entropy(
                logits,
                batch["teacher_top1_candidate_index"],
            )
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            policy_kl_loss = torch.sum(
                batch["teacher_policy"]
                * (
                    torch.log(batch["teacher_policy"].clamp_min(1e-8))
                    - log_probs
                ),
                dim=1,
            ).mean()
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
                    step_rollback_flags,
                )
            else:
                deliberation_monotonicity_loss = torch.zeros(
                    (),
                    dtype=logits.dtype,
                    device=logits.device,
                )

            loss = (
                optimization.value_wdl_weight * value_wdl_loss
                + optimization.value_cp_weight * value_cp_loss
                + optimization.sharpness_weight * sharpness_loss
                + optimization.sharpness_target_loss_weight * sharpness_target_loss
                + optimization.policy_ce_weight * policy_ce_loss
                + optimization.policy_kl_weight * policy_kl_loss
                + optimization.policy_margin_weight * policy_margin_loss
                + optimization.policy_rank_weight * policy_rank_loss
                + optimization.intention_aux_weight * intention_aux_loss
                + optimization.deliberation_monotonicity_weight * deliberation_monotonicity_loss
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

            probabilities = torch.softmax(logits, dim=1)
            top1_indices = torch.argmax(logits, dim=1)
            topk_indices = torch.topk(logits, k=min(top_k, logits.shape[1]), dim=1).indices
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
            correct_top1 += int(
                torch.sum(top1_indices == batch["teacher_top1_candidate_index"]).item()
            )
            correct_topk += int(
                torch.sum(
                    topk_indices == batch["teacher_top1_candidate_index"].unsqueeze(1)
                ).item()
            )
            reciprocal_rank_sum += _mean_reciprocal_rank(
                logits,
                batch["teacher_top1_candidate_index"],
            ) * len(batch_examples)
            teacher_probability_sum += float(
                torch.mean(
                    probabilities.gather(
                        1,
                        batch["teacher_top1_candidate_index"].unsqueeze(1),
                    )
                ).item()
            ) * len(batch_examples)
            if stage == "T2" and stage2 is not None:
                batch_rollbacks = sum(1 for flag in step_rollback_flags if flag)
                rollback_count += batch_rollbacks
                rollback_step_sum += sum(
                    float(step_index)
                    for step_index, flag in enumerate(step_rollback_flags)
                    if flag
                )
                total_trace_steps += len(step_rollback_flags)
            if training and (
                batch_index % optimization.log_interval_batches == 0
                or batch_index == total_batches
            ):
                elapsed = max(time.perf_counter() - start_time, 1e-9)
                print(
                    "[lapv1-train] "
                    f"epoch={epoch}/{total_epochs} "
                    f"batch={batch_index}/{total_batches} "
                    f"stage={stage} "
                    f"examples={total_examples}/{len(examples)} "
                    f"loss={total_loss / total_examples:.4f} "
                    f"top1={correct_top1 / total_examples:.4f} "
                    f"ex_per_s={total_examples / elapsed:.2f}",
                    flush=True,
                )
            del sigma_value

    duration = max(time.perf_counter() - start_time, 1e-9)
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
        root_top1_accuracy=correct_top1 / total_examples,
        root_top3_accuracy=correct_topk / total_examples,
        teacher_root_mean_reciprocal_rank=reciprocal_rank_sum / total_examples,
        teacher_root_mean_probability=teacher_probability_sum / total_examples,
        rollbacks=rollback_count,
        mean_rollback_step=(
            0.0 if rollback_count == 0 else rollback_step_sum / rollback_count
        ),
        rollback_hit_rate=(
            0.0 if total_trace_steps == 0 else rollback_count / total_trace_steps
        ),
        examples_per_second=total_examples / duration,
    )


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
        candidate_features[batch_index, :candidate_count, :] = torch.tensor(
            example.candidate_features,
            dtype=torch.float32,
        )
        candidate_mask[batch_index, :candidate_count] = True
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
        "reachability_edges": reachability_edges,
        "candidate_action_indices": candidate_action_indices,
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


def _trace_sharpness_target_loss(
    step_sharpness_tensors: Sequence[torch.Tensor],
    sharpness_target: torch.Tensor,
) -> torch.Tensor:
    if not step_sharpness_tensors:
        return torch.zeros((), dtype=sharpness_target.dtype, device=sharpness_target.device)
    losses = [
        torch.nn.functional.binary_cross_entropy(
            step_sharpness.clamp(1e-6, 1.0 - 1e-6),
            sharpness_target,
        )
        for step_sharpness in step_sharpness_tensors
    ]
    return torch.stack(losses).mean()


def _deliberation_monotonicity_loss(
    step_value_cp_tensors: Sequence[torch.Tensor],
    step_rollback_flags: Sequence[bool],
) -> torch.Tensor:
    if not step_value_cp_tensors:
        raise ValueError("deliberation_monotonicity_loss requires at least one trace tensor")
    if len(step_value_cp_tensors) < 2:
        reference = step_value_cp_tensors[0]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    penalties: list[torch.Tensor] = []
    for previous_values, current_values, rollback_fired in zip(
        step_value_cp_tensors[:-1],
        step_value_cp_tensors[1:],
        step_rollback_flags[1:],
        strict=True,
    ):
        if rollback_fired:
            continue
        penalties.append(
            torch.nn.functional.relu(previous_values - current_values).mean() / _GAP_TARGET_SCALE
        )
    if not penalties:
        reference = step_value_cp_tensors[0]
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    return torch.stack(penalties).mean()


def _mean_reciprocal_rank(
    logits: torch.Tensor,
    teacher_top1_candidate_index: torch.Tensor,
) -> float:
    reciprocal_rank_sum = 0.0
    for row_logits, teacher_index in zip(logits, teacher_top1_candidate_index, strict=True):
        ranked_indices = torch.argsort(row_logits, descending=True)
        rank = int(torch.nonzero(ranked_indices == teacher_index, as_tuple=False)[0, 0].item()) + 1
        reciprocal_rank_sum += 1.0 / rank
    return reciprocal_rank_sum / logits.shape[0]


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
    config: LAPv1TrainConfig,
    epoch: int,
) -> int:
    if config.stage == "T1":
        return 0
    assert config.stage2 is not None
    schedule = config.stage2.max_inner_steps_schedule
    schedule_index = min(
        ((epoch - 1) * len(schedule)) // config.optimization.epochs,
        len(schedule) - 1,
    )
    return schedule[schedule_index]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _configure_torch_runtime(torch_threads: int) -> None:
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)
