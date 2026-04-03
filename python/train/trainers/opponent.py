"""Training loop and evaluation metrics for the Phase-7 opponent head."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
import time
from typing import Any

from train.config import OpponentTrainConfig, resolve_repo_path
from train.datasets.artifacts import (
    SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE,
    TRANSITION_CONTEXT_FEATURE_SIZE,
)
from train.datasets.contracts import candidate_context_feature_dim
from train.datasets.opponent_head import (
    OpponentHeadExample,
    load_opponent_head_examples,
)
from train.losses.opponent import compute_opponent_losses
from train.models.opponent import OpponentHeadModel, torch_is_available

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None


OPPONENT_CANDIDATE_FEATURE_SIZE = candidate_context_feature_dim(2)


@dataclass(frozen=True)
class OpponentMetrics:
    """Aggregated opponent-head losses and held-out reply metrics."""

    total_examples: int
    supervised_examples: int
    total_loss: float
    reply_policy_loss: float
    pressure_loss: float
    uncertainty_loss: float
    reply_top1_accuracy: float
    reply_top3_accuracy: float
    teacher_reply_mean_reciprocal_rank: float
    teacher_reply_mean_probability: float
    pressure_mae: float
    uncertainty_mae: float
    examples_per_second: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Return the JSON-friendly representation."""
        return {
            "total_examples": self.total_examples,
            "supervised_examples": self.supervised_examples,
            "total_loss": round(self.total_loss, 6),
            "reply_policy_loss": round(self.reply_policy_loss, 6),
            "pressure_loss": round(self.pressure_loss, 6),
            "uncertainty_loss": round(self.uncertainty_loss, 6),
            "reply_top1_accuracy": round(self.reply_top1_accuracy, 6),
            "reply_top3_accuracy": round(self.reply_top3_accuracy, 6),
            "teacher_reply_mean_reciprocal_rank": round(
                self.teacher_reply_mean_reciprocal_rank,
                6,
            ),
            "teacher_reply_mean_probability": round(
                self.teacher_reply_mean_probability,
                6,
            ),
            "pressure_mae": round(self.pressure_mae, 6),
            "uncertainty_mae": round(self.uncertainty_mae, 6),
            "examples_per_second": round(self.examples_per_second, 3),
        }


@dataclass(frozen=True)
class OpponentTrainingRun:
    """Serializable result of an opponent-head training run."""

    history: list[dict[str, Any]]
    best_epoch: int
    best_validation: dict[str, float | int]
    export_paths: dict[str, str]
    summary_path: str
    model_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly representation."""
        return {
            "history": self.history,
            "best_epoch": self.best_epoch,
            "best_validation": self.best_validation,
            "export_paths": self.export_paths,
            "summary_path": self.summary_path,
            "model_parameter_count": self.model_parameter_count,
        }


def train_opponent(config: OpponentTrainConfig, *, repo_root: Path) -> OpponentTrainingRun:
    """Train the first explicit opponent head and save the best checkpoint."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for opponent-head training. Install the 'train' extra or torch."
        )

    output_dir = resolve_repo_path(repo_root, config.output_dir)
    bundle_dir = resolve_repo_path(repo_root, config.export.bundle_dir)
    train_paths = [
        resolve_repo_path(repo_root, path)
        for path in config.data.resolved_train_paths()
    ]
    validation_paths = [
        resolve_repo_path(repo_root, path)
        for path in config.data.resolved_validation_paths()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(config.seed)
    _configure_torch_runtime(config.runtime.torch_threads)

    train_examples = _load_examples_from_paths(train_paths)
    validation_examples = _load_examples_from_paths(validation_paths)
    if not train_examples:
        raise ValueError("training artifact is empty")
    if not validation_examples:
        raise ValueError("validation artifact is empty")

    model = OpponentHeadModel(
        architecture=config.model.architecture,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        action_embedding_dim=config.model.action_embedding_dim,
        dropout=config.model.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())

    train_loader = _build_loader(
        train_examples,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        seed=config.seed,
        num_workers=config.runtime.dataloader_workers,
    )
    validation_loader = _build_loader(
        validation_examples,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.runtime.dataloader_workers,
    )

    history: list[dict[str, Any]] = []
    best_epoch = 1
    best_validation: OpponentMetrics | None = None
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    for epoch in range(1, config.optimization.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            training=True,
            reply_policy_weight=config.optimization.reply_policy_loss_weight,
            pressure_weight=config.optimization.pressure_loss_weight,
            uncertainty_weight=config.optimization.uncertainty_loss_weight,
            curriculum_priority_weight=config.optimization.curriculum_priority_weight,
            top_k=config.evaluation.top_k,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            training=False,
            reply_policy_weight=config.optimization.reply_policy_loss_weight,
            pressure_weight=config.optimization.pressure_loss_weight,
            uncertainty_weight=config.optimization.uncertainty_loss_weight,
            curriculum_priority_weight=config.optimization.curriculum_priority_weight,
            top_k=config.evaluation.top_k,
        )
        history.append(
            {
                "epoch": epoch,
                "examples_per_second": round(train_metrics.examples_per_second, 3),
                "train": train_metrics.to_dict(),
                "validation": validation_metrics.to_dict(),
            }
        )
        if best_validation is None or _is_better_validation(validation_metrics, best_validation):
            best_epoch = epoch
            best_validation = validation_metrics
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }

    assert best_validation is not None
    model.load_state_dict(best_state)
    checkpoint_path = bundle_dir / config.export.checkpoint_name
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": config.to_dict(),
            "validation_metrics": best_validation.to_dict(),
        },
        checkpoint_path,
    )
    export_paths = {"checkpoint": str(checkpoint_path)}

    summary = {
        "config": config.to_dict(),
        "history": history,
        "best_epoch": best_epoch,
        "best_validation": best_validation.to_dict(),
        "export_paths": export_paths,
        "model_parameter_count": model_parameter_count,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return OpponentTrainingRun(
        history=history,
        best_epoch=best_epoch,
        best_validation=best_validation.to_dict(),
        export_paths=export_paths,
        summary_path=str(summary_path),
        model_parameter_count=model_parameter_count,
    )


def evaluate_opponent_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_path: Path | None = None,
    dataset_paths: list[Path] | tuple[Path, ...] | None = None,
    top_k: int = 3,
) -> OpponentMetrics:
    """Load a saved opponent checkpoint and evaluate it on one artifact."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for opponent-head evaluation. Install the 'train' extra or torch."
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = OpponentTrainConfig.from_dict(dict(payload["training_config"]))
    model = OpponentHeadModel(
        architecture=config.model.architecture,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        action_embedding_dim=config.model.action_embedding_dim,
        dropout=config.model.dropout,
    )
    model.load_state_dict(dict(payload["model_state_dict"]))
    if dataset_paths is not None:
        resolved_paths = list(dataset_paths)
    elif dataset_path is not None:
        resolved_paths = [dataset_path]
    else:
        raise ValueError("either dataset_path or dataset_paths must be provided")
    examples = _load_examples_from_paths(resolved_paths)
    if not examples:
        raise ValueError(f"evaluation artifact list is empty: {resolved_paths}")
    loader = _build_loader(
        examples,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.runtime.dataloader_workers,
    )
    return _run_epoch(
        model,
        loader,
        optimizer=None,
        training=False,
        reply_policy_weight=config.optimization.reply_policy_loss_weight,
        pressure_weight=config.optimization.pressure_loss_weight,
        uncertainty_weight=config.optimization.uncertainty_loss_weight,
        curriculum_priority_weight=config.optimization.curriculum_priority_weight,
        top_k=top_k,
    )


def _build_loader(
    examples: list[OpponentHeadExample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> Any:
    generator = torch.Generator().manual_seed(seed)
    dataset = _TensorDataset.from_examples(examples)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def _run_epoch(
    model: Any,
    loader: Any,
    *,
    optimizer: Any,
    training: bool,
    reply_policy_weight: float,
    pressure_weight: float,
    uncertainty_weight: float,
    curriculum_priority_weight: float,
    top_k: int,
) -> OpponentMetrics:
    if training:
        model.train()
    else:
        model.eval()

    total_examples = 0
    supervised_examples = 0
    total_loss_sum = 0.0
    reply_loss_sum = 0.0
    pressure_loss_sum = 0.0
    uncertainty_loss_sum = 0.0
    reply_top1_correct = 0
    reply_topk_correct = 0
    reciprocal_rank_total = 0.0
    teacher_reply_probability_total = 0.0
    pressure_error_total = 0.0
    uncertainty_error_total = 0.0

    started_at = time.perf_counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            prediction = model(
                batch["root_features"],
                batch["next_features"],
                batch["chosen_action_indices"],
                batch["transition_features"],
                batch["reply_global_features"],
                batch["reply_candidate_action_indices"],
                batch["reply_candidate_features"],
                batch["reply_candidate_mask"],
            )
            losses = compute_opponent_losses(
                prediction.reply_logits,
                reply_mask=batch["reply_candidate_mask"],
                target_reply_policy=batch["teacher_reply_policy"],
                supervised_mask=batch["supervised_mask"],
                reply_example_weights=_build_reply_example_weights(
                    batch["curriculum_priorities"],
                    supervised_mask=batch["supervised_mask"],
                    curriculum_priority_weight=curriculum_priority_weight,
                ),
                predicted_pressure=prediction.pressure,
                target_pressure=batch["pressure_targets"],
                predicted_uncertainty=prediction.uncertainty,
                target_uncertainty=batch["uncertainty_targets"],
                reply_policy_weight=reply_policy_weight,
                pressure_weight=pressure_weight,
                uncertainty_weight=uncertainty_weight,
            )

            if training:
                optimizer.zero_grad()
                losses.total.backward()
                optimizer.step()

            batch_size = int(batch["root_features"].shape[0])
            total_examples += batch_size
            supervised_count = int(batch["supervised_mask"].sum().item())
            supervised_examples += supervised_count
            total_loss_sum += float(losses.total.item()) * batch_size
            reply_loss_sum += float(losses.reply_policy.item()) * batch_size
            pressure_loss_sum += float(losses.pressure.item()) * batch_size
            uncertainty_loss_sum += float(losses.uncertainty.item()) * batch_size

            masked_logits = prediction.reply_logits.masked_fill(
                ~batch["reply_candidate_mask"],
                -1e9,
            )
            policy = torch.softmax(masked_logits, dim=1)
            ranked = torch.argsort(masked_logits, dim=1, descending=True)

            if supervised_count > 0:
                for row_index in range(batch_size):
                    teacher_index = int(batch["teacher_reply_indices"][row_index].item())
                    if teacher_index < 0:
                        continue
                    reply_top1_correct += int(int(ranked[row_index, 0].item()) == teacher_index)
                    top_k_width = min(top_k, int(batch["reply_candidate_mask"][row_index].sum().item()))
                    reply_topk_correct += int(
                        teacher_index
                        in [int(value) for value in ranked[row_index, :top_k_width].tolist()]
                    )
                    rank_position = [
                        int(value) for value in ranked[row_index].tolist()
                    ].index(teacher_index)
                    reciprocal_rank_total += 1.0 / float(rank_position + 1)
                    teacher_reply_probability_total += float(policy[row_index, teacher_index].item())

            pressure_error_total += float(
                torch.abs(prediction.pressure - batch["pressure_targets"]).sum().item()
            )
            uncertainty_error_total += float(
                torch.abs(prediction.uncertainty - batch["uncertainty_targets"]).sum().item()
            )

    elapsed = time.perf_counter() - started_at
    return OpponentMetrics(
        total_examples=total_examples,
        supervised_examples=supervised_examples,
        total_loss=_ratio(total_loss_sum, total_examples),
        reply_policy_loss=_ratio(reply_loss_sum, total_examples),
        pressure_loss=_ratio(pressure_loss_sum, total_examples),
        uncertainty_loss=_ratio(uncertainty_loss_sum, total_examples),
        reply_top1_accuracy=_ratio(reply_top1_correct, supervised_examples),
        reply_top3_accuracy=_ratio(reply_topk_correct, supervised_examples),
        teacher_reply_mean_reciprocal_rank=_ratio(reciprocal_rank_total, supervised_examples),
        teacher_reply_mean_probability=_ratio(
            teacher_reply_probability_total,
            supervised_examples,
        ),
        pressure_mae=_ratio(pressure_error_total, total_examples),
        uncertainty_mae=_ratio(uncertainty_error_total, total_examples),
        examples_per_second=_ratio(total_examples, elapsed),
    )


class _TensorDataset:
    def __init__(self, tensors: dict[str, Any]) -> None:
        self.tensors = tensors

    @classmethod
    def from_examples(cls, examples: list[OpponentHeadExample]) -> "_TensorDataset":
        if not examples:
            raise ValueError("opponent dataset requires at least one example")
        max_candidates = max(len(example.reply_candidate_action_indices) for example in examples)
        if max_candidates <= 0:
            raise ValueError("opponent dataset requires at least one reply candidate")

        root_features = torch.tensor(
            [example.root_feature_vector for example in examples],
            dtype=torch.float32,
        )
        next_features = torch.tensor(
            [example.next_feature_vector for example in examples],
            dtype=torch.float32,
        )
        chosen_action_indices = torch.tensor(
            [example.chosen_action_index for example in examples],
            dtype=torch.long,
        )
        transition_features = torch.tensor(
            [_validated_transition_features(example) for example in examples],
            dtype=torch.float32,
        )
        reply_global_features = torch.tensor(
            [_validated_reply_global_features(example) for example in examples],
            dtype=torch.float32,
        )
        reply_candidate_action_indices = torch.zeros(
            (len(examples), max_candidates),
            dtype=torch.long,
        )
        reply_candidate_features = torch.zeros(
            (
                len(examples),
                max_candidates,
                OPPONENT_CANDIDATE_FEATURE_SIZE,
            ),
            dtype=torch.float32,
        )
        reply_candidate_mask = torch.zeros(
            (len(examples), max_candidates),
            dtype=torch.bool,
        )
        teacher_reply_policy = torch.zeros(
            (len(examples), max_candidates),
            dtype=torch.float32,
        )
        supervised_mask = torch.zeros(len(examples), dtype=torch.bool)
        teacher_reply_indices = torch.full((len(examples),), -1, dtype=torch.long)
        pressure_targets = torch.tensor(
            [example.pressure_target for example in examples],
            dtype=torch.float32,
        )
        uncertainty_targets = torch.tensor(
            [example.uncertainty_target for example in examples],
            dtype=torch.float32,
        )
        curriculum_priorities = torch.tensor(
            [example.curriculum_priority for example in examples],
            dtype=torch.float32,
        )

        for row_index, example in enumerate(examples):
            candidate_count = len(example.reply_candidate_action_indices)
            reply_candidate_action_indices[row_index, :candidate_count] = torch.tensor(
                example.reply_candidate_action_indices,
                dtype=torch.long,
            )
            reply_candidate_features[row_index, :candidate_count] = torch.tensor(
                _validated_reply_candidate_features(example),
                dtype=torch.float32,
            )
            reply_candidate_mask[row_index, :candidate_count] = True
            if len(example.teacher_reply_policy) != candidate_count:
                raise ValueError(
                    f"{example.sample_id}: teacher_reply_policy length must match candidate count"
                )
            teacher_reply_policy[row_index, :candidate_count] = torch.tensor(
                example.teacher_reply_policy,
                dtype=torch.float32,
            )
            if example.teacher_reply_action_index is not None:
                supervised_mask[row_index] = True
                try:
                    teacher_reply_indices[row_index] = example.reply_candidate_action_indices.index(
                        example.teacher_reply_action_index
                    )
                except ValueError as error:
                    raise ValueError(
                        f"{example.sample_id}: teacher reply action missing from reply candidates"
                    ) from error

        return cls(
            {
                "root_features": root_features,
                "next_features": next_features,
                "chosen_action_indices": chosen_action_indices,
                "transition_features": transition_features,
                "reply_global_features": reply_global_features,
                "reply_candidate_action_indices": reply_candidate_action_indices,
                "reply_candidate_features": reply_candidate_features,
                "reply_candidate_mask": reply_candidate_mask,
                "teacher_reply_policy": teacher_reply_policy,
                "supervised_mask": supervised_mask,
                "teacher_reply_indices": teacher_reply_indices,
                "pressure_targets": pressure_targets,
                "uncertainty_targets": uncertainty_targets,
                "curriculum_priorities": curriculum_priorities,
            }
        )

    def __len__(self) -> int:
        return int(self.tensors["root_features"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            name: tensor[index]
            for name, tensor in self.tensors.items()
        }


def _validated_transition_features(example: OpponentHeadExample) -> list[float]:
    if example.transition_context_version != 1:
        raise ValueError(
            f"{example.sample_id}: opponent training currently requires transition_context_version=1"
        )
    if len(example.transition_features) != TRANSITION_CONTEXT_FEATURE_SIZE:
        raise ValueError(
            f"{example.sample_id}: transition_features must have width {TRANSITION_CONTEXT_FEATURE_SIZE}"
        )
    return example.transition_features


def _validated_reply_global_features(example: OpponentHeadExample) -> list[float]:
    if example.reply_global_context_version != 1:
        raise ValueError(
            f"{example.sample_id}: opponent training currently requires reply_global_context_version=1"
        )
    if len(example.reply_global_features) != SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE:
        raise ValueError(
            f"{example.sample_id}: reply_global_features must have width {SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE}"
        )
    return example.reply_global_features


def _validated_reply_candidate_features(example: OpponentHeadExample) -> list[list[float]]:
    if example.reply_candidate_context_version != 2:
        raise ValueError(
            f"{example.sample_id}: opponent training currently requires reply_candidate_context_version=2"
        )
    for row in example.reply_candidate_features:
        if len(row) != OPPONENT_CANDIDATE_FEATURE_SIZE:
            raise ValueError(
                f"{example.sample_id}: reply_candidate_features rows must have width {OPPONENT_CANDIDATE_FEATURE_SIZE}"
            )
    return example.reply_candidate_features


def _build_reply_example_weights(
    curriculum_priorities: Any,
    *,
    supervised_mask: Any,
    curriculum_priority_weight: float,
) -> Any | None:
    if curriculum_priority_weight <= 0.0:
        return None
    weights = 1.0 + curriculum_priority_weight * torch.log1p(
        curriculum_priorities.clamp_min(0.0)
    )
    if bool(supervised_mask.any()):
        mean_weight = weights[supervised_mask].mean().clamp_min(1e-6)
        weights = weights / mean_weight
    return weights


def _load_examples_from_paths(paths: list[Path] | tuple[Path, ...]) -> list[OpponentHeadExample]:
    examples: list[OpponentHeadExample] = []
    for path in paths:
        examples.extend(load_opponent_head_examples(path))
    return examples


def _is_better_validation(current: OpponentMetrics, best: OpponentMetrics) -> bool:
    current_key = (
        current.reply_top1_accuracy,
        current.reply_top3_accuracy,
        current.teacher_reply_mean_reciprocal_rank,
        current.teacher_reply_mean_probability,
        -current.pressure_mae,
        -current.uncertainty_mae,
    )
    best_key = (
        best.reply_top1_accuracy,
        best.reply_top3_accuracy,
        best.teacher_reply_mean_reciprocal_rank,
        best.teacher_reply_mean_probability,
        -best.pressure_mae,
        -best.uncertainty_mae,
    )
    return current_key > best_key


def _configure_torch_runtime(torch_threads: int) -> None:
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)
