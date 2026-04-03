"""Training loop and evaluation metrics for the first trainable bounded planner arm."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
import time
from typing import Any

from train.config import PlannerTrainConfig, resolve_repo_path
from train.datasets.artifacts import SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE
from train.datasets.contracts import candidate_context_feature_dim, transition_context_feature_dim
from train.datasets.planner_head import PlannerHeadExample, load_planner_head_examples
from train.models.planner import PlannerHeadModel, torch, PLANNER_MODEL_NAME
from train.models.proposer import torch_is_available


PLANNER_CANDIDATE_FEATURE_SIZE = candidate_context_feature_dim(2)
PLANNER_TRANSITION_FEATURE_SIZE = transition_context_feature_dim(1)
PLANNER_ROOT_VALUE_SCALE_CP = 256.0
PLANNER_ROOT_GAP_SCALE_CP = 128.0
PLANNER_CANDIDATE_SCORE_SCALE_CP = 128.0


@dataclass(frozen=True)
class PlannerMetrics:
    """Aggregated planner-head losses and held-out root-ranking metrics."""

    total_examples: int
    supervised_examples: int
    total_loss: float
    teacher_policy_loss: float
    teacher_kl_loss: float
    teacher_score_loss: float
    root_value_loss: float
    root_gap_loss: float
    root_top1_accuracy: float
    root_top3_accuracy: float
    teacher_root_mean_reciprocal_rank: float
    teacher_root_mean_probability: float
    teacher_score_mae_cp: float
    root_value_mae_cp: float
    root_gap_mae_cp: float
    root_gap_examples: int
    examples_per_second: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "total_examples": self.total_examples,
            "supervised_examples": self.supervised_examples,
            "total_loss": round(self.total_loss, 6),
            "teacher_policy_loss": round(self.teacher_policy_loss, 6),
            "teacher_kl_loss": round(self.teacher_kl_loss, 6),
            "teacher_score_loss": round(self.teacher_score_loss, 6),
            "root_value_loss": round(self.root_value_loss, 6),
            "root_gap_loss": round(self.root_gap_loss, 6),
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
            "teacher_score_mae_cp": round(self.teacher_score_mae_cp, 6),
            "root_value_mae_cp": round(self.root_value_mae_cp, 6),
            "root_gap_mae_cp": round(self.root_gap_mae_cp, 6),
            "root_gap_examples": self.root_gap_examples,
            "examples_per_second": round(self.examples_per_second, 3),
        }


@dataclass(frozen=True)
class PlannerTrainingRun:
    """Serializable result of a planner-head training run."""

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


def train_planner(config: PlannerTrainConfig, *, repo_root: Path) -> PlannerTrainingRun:
    """Train the first bounded root planner head and save the best checkpoint."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for planner-head training. Install the 'train' extra or torch."
        )

    output_dir = resolve_repo_path(repo_root, config.output_dir)
    bundle_dir = resolve_repo_path(repo_root, config.export.bundle_dir)
    train_paths = [resolve_repo_path(repo_root, path) for path in config.data.resolved_train_paths()]
    validation_paths = [
        resolve_repo_path(repo_root, path) for path in config.data.resolved_validation_paths()
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
    _validate_latent_feature_dim(
        train_examples,
        expected_dim=config.model.latent_feature_dim,
        context="train",
    )
    _validate_latent_feature_dim(
        validation_examples,
        expected_dim=config.model.latent_feature_dim,
        context="validation",
    )

    model = PlannerHeadModel(
        architecture=config.model.architecture,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        action_embedding_dim=config.model.action_embedding_dim,
        latent_feature_dim=config.model.latent_feature_dim,
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
        latent_feature_dim=config.model.latent_feature_dim,
    )
    validation_loader = _build_loader(
        validation_examples,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=config.runtime.dataloader_workers,
        latent_feature_dim=config.model.latent_feature_dim,
    )

    history: list[dict[str, Any]] = []
    best_epoch = 1
    best_validation: PlannerMetrics | None = None
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    for epoch in range(1, config.optimization.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            training=True,
            teacher_policy_weight=config.optimization.teacher_policy_loss_weight,
            teacher_kl_weight=config.optimization.teacher_kl_loss_weight,
            teacher_score_weight=config.optimization.teacher_score_loss_weight,
            curriculum_priority_weight=config.optimization.curriculum_priority_weight,
            root_value_weight=config.optimization.root_value_loss_weight,
            root_gap_weight=config.optimization.root_gap_loss_weight,
            top_k=config.evaluation.top_k,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            training=False,
            teacher_policy_weight=config.optimization.teacher_policy_loss_weight,
            teacher_kl_weight=config.optimization.teacher_kl_loss_weight,
            teacher_score_weight=config.optimization.teacher_score_loss_weight,
            curriculum_priority_weight=config.optimization.curriculum_priority_weight,
            root_value_weight=config.optimization.root_value_loss_weight,
            root_gap_weight=config.optimization.root_gap_loss_weight,
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
            "model_name": PLANNER_MODEL_NAME,
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
        "model_name": PLANNER_MODEL_NAME,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return PlannerTrainingRun(
        history=history,
        best_epoch=best_epoch,
        best_validation=best_validation.to_dict(),
        export_paths=export_paths,
        summary_path=str(summary_path),
        model_parameter_count=model_parameter_count,
    )


def evaluate_planner_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_path: Path | None = None,
    dataset_paths: list[Path] | tuple[Path, ...] | None = None,
    top_k: int = 3,
) -> PlannerMetrics:
    """Load a saved planner checkpoint and evaluate it on one or more artifacts."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for planner-head evaluation. Install the 'train' extra or torch."
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = PlannerTrainConfig.from_dict(dict(payload["training_config"]))
    model = PlannerHeadModel(
        architecture=config.model.architecture,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        action_embedding_dim=config.model.action_embedding_dim,
        latent_feature_dim=config.model.latent_feature_dim,
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
    _validate_latent_feature_dim(
        examples,
        expected_dim=config.model.latent_feature_dim,
        context="evaluation",
    )
    loader = _build_loader(
        examples,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        seed=config.seed,
        num_workers=0,
        latent_feature_dim=config.model.latent_feature_dim,
    )
    return _run_epoch(
        model,
        loader,
        optimizer=None,
        training=False,
        teacher_policy_weight=config.optimization.teacher_policy_loss_weight,
        teacher_kl_weight=config.optimization.teacher_kl_loss_weight,
        teacher_score_weight=config.optimization.teacher_score_loss_weight,
        curriculum_priority_weight=config.optimization.curriculum_priority_weight,
        root_value_weight=config.optimization.root_value_loss_weight,
        root_gap_weight=config.optimization.root_gap_loss_weight,
        top_k=top_k,
    )


def _load_examples_from_paths(paths: list[Path]) -> list[PlannerHeadExample]:
    examples: list[PlannerHeadExample] = []
    for path in paths:
        if path.is_dir():
            artifact_path = path / "planner_head_train.jsonl"
            if not artifact_path.exists():
                artifact_path = path / "planner_head_validation.jsonl"
            if not artifact_path.exists():
                artifact_path = path / "planner_head_test.jsonl"
            examples.extend(load_planner_head_examples(artifact_path))
        else:
            examples.extend(load_planner_head_examples(path))
    return examples


def _build_loader(
    examples: list[PlannerHeadExample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    latent_feature_dim: int,
) -> Any:
    assert torch is not None
    dataset = _PlannerTensorDataset(examples)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: _collate_planner_batch(
            batch,
            latent_feature_dim=latent_feature_dim,
        ),
        generator=generator,
    )


def _run_epoch(
    model: PlannerHeadModel,
    loader: Any,
    *,
    optimizer: Any | None,
    training: bool,
    teacher_policy_weight: float,
    teacher_kl_weight: float,
    teacher_score_weight: float,
    curriculum_priority_weight: float,
    root_value_weight: float,
    root_gap_weight: float,
    top_k: int,
) -> PlannerMetrics:
    assert torch is not None
    model.train(training)
    started_at = time.perf_counter()
    total_examples = 0
    total_loss = 0.0
    teacher_policy_loss_total = 0.0
    teacher_kl_loss_total = 0.0
    teacher_score_loss_total = 0.0
    root_value_loss_total = 0.0
    root_gap_loss_total = 0.0
    root_top1_correct = 0
    root_top3_correct = 0
    reciprocal_rank_total = 0.0
    teacher_probability_total = 0.0
    teacher_score_abs_error_total = 0.0
    teacher_score_examples = 0
    root_value_abs_error_total = 0.0
    root_gap_abs_error_total = 0.0
    root_gap_examples = 0

    for batch in loader:
        outputs = model(
            batch["root_features"],
            batch["global_features"],
            batch["candidate_action_indices"],
            batch["candidate_features"],
            batch["proposer_scores"],
            batch["transition_features"],
            batch["latent_features"],
            batch["reply_peak_probabilities"],
            batch["pressures"],
            batch["uncertainties"],
            batch["candidate_mask"],
        )
        logits = outputs["logits"]
        log_probs = torch.log_softmax(logits, dim=1)
        ce_loss = torch.nn.functional.nll_loss(
            log_probs,
            batch["teacher_top1_candidate_indices"],
            reduction="none",
        )
        if teacher_kl_weight > 0.0:
            kl_loss = torch.nn.functional.kl_div(
                log_probs,
                batch["teacher_policy"],
                reduction="none",
            ).sum(dim=1)
        else:
            kl_loss = torch.zeros_like(ce_loss)
        if outputs["candidate_score_prediction"] is not None and teacher_score_weight > 0.0:
            raw_score_loss = torch.nn.functional.smooth_l1_loss(
                outputs["candidate_score_prediction"],
                batch["teacher_candidate_score_targets"],
                reduction="none",
            )
            score_mask = batch["teacher_candidate_score_mask"].to(raw_score_loss.dtype)
            score_count = score_mask.sum(dim=1).clamp_min(1.0)
            score_loss = (raw_score_loss * score_mask).sum(dim=1) / score_count
        else:
            raw_score_loss = None
            score_loss = torch.zeros_like(ce_loss)
        if outputs["root_value_prediction"] is not None and root_value_weight > 0.0:
            root_value_loss = torch.nn.functional.smooth_l1_loss(
                outputs["root_value_prediction"],
                batch["teacher_root_value_targets"],
                reduction="none",
            )
        else:
            root_value_loss = torch.zeros_like(ce_loss)
        if outputs["root_gap_prediction"] is not None and root_gap_weight > 0.0:
            raw_gap_loss = torch.nn.functional.smooth_l1_loss(
                outputs["root_gap_prediction"],
                batch["teacher_root_gap_targets"],
                reduction="none",
            )
            root_gap_loss = raw_gap_loss * batch["teacher_root_gap_mask"].to(raw_gap_loss.dtype)
        else:
            root_gap_loss = torch.zeros_like(ce_loss)
        example_weights = 1.0 + (
            curriculum_priority_weight * batch["curriculum_priorities"]
        )
        loss = (
            teacher_policy_weight * ce_loss
            + teacher_kl_weight * kl_loss
            + teacher_score_weight * score_loss
            + root_value_weight * root_value_loss
            + root_gap_weight * root_gap_loss
        ) * example_weights
        mean_loss = loss.mean()

        if training:
            assert optimizer is not None
            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()

        probabilities = torch.softmax(logits, dim=1)
        ranking = torch.argsort(logits, dim=1, descending=True)
        teacher_indices = batch["teacher_top1_candidate_indices"]
        total_examples += int(teacher_indices.shape[0])
        total_loss += float(loss.sum().item())
        teacher_policy_loss_total += float((ce_loss * example_weights).sum().item())
        teacher_kl_loss_total += float((kl_loss * example_weights).sum().item())
        teacher_score_loss_total += float((score_loss * example_weights).sum().item())
        root_value_loss_total += float((root_value_loss * example_weights).sum().item())
        root_gap_loss_total += float((root_gap_loss * example_weights).sum().item())
        if outputs["candidate_score_prediction"] is not None and raw_score_loss is not None:
            score_mask = batch["teacher_candidate_score_mask"].to(
                outputs["candidate_score_prediction"].dtype
            )
            teacher_score_abs_error_total += float(
                (
                    torch.abs(
                        outputs["candidate_score_prediction"]
                        - batch["teacher_candidate_score_targets"]
                    )
                    * score_mask
                ).sum().item()
                * PLANNER_CANDIDATE_SCORE_SCALE_CP
            )
            teacher_score_examples += int(batch["teacher_candidate_score_mask"].sum().item())
        if outputs["root_value_prediction"] is not None:
            root_value_abs_error_total += float(
                torch.abs(
                    outputs["root_value_prediction"] - batch["teacher_root_value_targets"]
                ).sum().item()
                * PLANNER_ROOT_VALUE_SCALE_CP
            )
        if outputs["root_gap_prediction"] is not None:
            gap_mask = batch["teacher_root_gap_mask"].to(outputs["root_gap_prediction"].dtype)
            root_gap_abs_error_total += float(
                (
                    torch.abs(
                        outputs["root_gap_prediction"] - batch["teacher_root_gap_targets"]
                    )
                    * gap_mask
                ).sum().item()
                * PLANNER_ROOT_GAP_SCALE_CP
            )
            root_gap_examples += int(batch["teacher_root_gap_mask"].sum().item())

        for row_index in range(teacher_indices.shape[0]):
            teacher_index = int(teacher_indices[row_index].item())
            ranked = ranking[row_index].tolist()
            root_top1_correct += int(ranked[:1] == [teacher_index])
            root_top3_correct += int(teacher_index in ranked[: min(top_k, 3)])
            reciprocal_rank_total += 1.0 / float(ranked.index(teacher_index) + 1)
            teacher_probability_total += float(probabilities[row_index, teacher_index].item())

    elapsed = time.perf_counter() - started_at
    return PlannerMetrics(
        total_examples=total_examples,
        supervised_examples=total_examples,
        total_loss=_ratio(total_loss, total_examples),
        teacher_policy_loss=_ratio(teacher_policy_loss_total, total_examples),
        teacher_kl_loss=_ratio(teacher_kl_loss_total, total_examples),
        teacher_score_loss=_ratio(teacher_score_loss_total, total_examples),
        root_value_loss=_ratio(root_value_loss_total, total_examples),
        root_gap_loss=_ratio(root_gap_loss_total, total_examples),
        root_top1_accuracy=_ratio(root_top1_correct, total_examples),
        root_top3_accuracy=_ratio(root_top3_correct, total_examples),
        teacher_root_mean_reciprocal_rank=_ratio(reciprocal_rank_total, total_examples),
        teacher_root_mean_probability=_ratio(teacher_probability_total, total_examples),
        teacher_score_mae_cp=_ratio(teacher_score_abs_error_total, teacher_score_examples),
        root_value_mae_cp=_ratio(root_value_abs_error_total, total_examples),
        root_gap_mae_cp=_ratio(root_gap_abs_error_total, root_gap_examples),
        root_gap_examples=root_gap_examples,
        examples_per_second=_ratio(total_examples, elapsed),
    )


def _is_better_validation(candidate: PlannerMetrics, current_best: PlannerMetrics) -> bool:
    candidate_key = (
        candidate.root_top1_accuracy,
        candidate.teacher_root_mean_reciprocal_rank,
        candidate.teacher_root_mean_probability,
        -candidate.total_loss,
    )
    current_key = (
        current_best.root_top1_accuracy,
        current_best.teacher_root_mean_reciprocal_rank,
        current_best.teacher_root_mean_probability,
        -current_best.total_loss,
    )
    return candidate_key > current_key


class _PlannerTensorDataset:
    def __init__(self, examples: list[PlannerHeadExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PlannerHeadExample:
        return self.examples[index]


def _collate_planner_batch(
    examples: list[PlannerHeadExample],
    *,
    latent_feature_dim: int,
) -> dict[str, Any]:
    assert torch is not None
    batch_size = len(examples)
    max_candidates = max(len(example.candidate_action_indices) for example in examples)

    root_features = torch.zeros((batch_size, len(examples[0].feature_vector)), dtype=torch.float32)
    global_features = torch.zeros(
        (batch_size, SYMBOLIC_PROPOSER_GLOBAL_FEATURE_SIZE),
        dtype=torch.float32,
    )
    candidate_action_indices = torch.full((batch_size, max_candidates), -1, dtype=torch.long)
    candidate_features = torch.zeros(
        (batch_size, max_candidates, PLANNER_CANDIDATE_FEATURE_SIZE),
        dtype=torch.float32,
    )
    proposer_scores = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    transition_features = torch.zeros(
        (batch_size, max_candidates, PLANNER_TRANSITION_FEATURE_SIZE),
        dtype=torch.float32,
    )
    latent_features = torch.zeros(
        (batch_size, max_candidates, latent_feature_dim),
        dtype=torch.float32,
    )
    reply_peak_probabilities = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    pressures = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    uncertainties = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    teacher_top1_candidate_indices = torch.zeros((batch_size,), dtype=torch.long)
    teacher_policy = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    teacher_candidate_score_targets = torch.zeros((batch_size, max_candidates), dtype=torch.float32)
    teacher_candidate_score_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    curriculum_priorities = torch.zeros((batch_size,), dtype=torch.float32)
    teacher_root_value_targets = torch.zeros((batch_size,), dtype=torch.float32)
    teacher_root_gap_targets = torch.zeros((batch_size,), dtype=torch.float32)
    teacher_root_gap_mask = torch.zeros((batch_size,), dtype=torch.bool)

    for row_index, example in enumerate(examples):
        candidate_count = len(example.candidate_action_indices)
        root_features[row_index] = torch.tensor(example.feature_vector, dtype=torch.float32)
        global_features[row_index] = torch.tensor(example.global_features, dtype=torch.float32)
        candidate_action_indices[row_index, :candidate_count] = torch.tensor(
            example.candidate_action_indices,
            dtype=torch.long,
        )
        candidate_features[row_index, :candidate_count] = torch.tensor(
            example.candidate_features,
            dtype=torch.float32,
        )
        proposer_scores[row_index, :candidate_count] = torch.tensor(
            example.proposer_scores,
            dtype=torch.float32,
        )
        transition_features[row_index, :candidate_count] = torch.tensor(
            example.transition_features,
            dtype=torch.float32,
        )
        if example.latent_features is not None and latent_feature_dim > 0:
            latent_features[row_index, :candidate_count] = torch.tensor(
                example.latent_features,
                dtype=torch.float32,
            )
        reply_peak_probabilities[row_index, :candidate_count] = torch.tensor(
            example.reply_peak_probabilities,
            dtype=torch.float32,
        )
        pressures[row_index, :candidate_count] = torch.tensor(
            example.pressures,
            dtype=torch.float32,
        )
        uncertainties[row_index, :candidate_count] = torch.tensor(
            example.uncertainties,
            dtype=torch.float32,
        )
        candidate_mask[row_index, :candidate_count] = True
        teacher_top1_candidate_indices[row_index] = example.teacher_top1_candidate_index
        teacher_policy[row_index, :candidate_count] = torch.tensor(
            example.teacher_policy,
            dtype=torch.float32,
        )
        if example.teacher_candidate_scores_cp is not None:
            if example.teacher_candidate_score_delta_targets_cp is not None:
                teacher_candidate_score_targets[row_index, :candidate_count] = torch.tensor(
                    [
                        _normalize_teacher_candidate_score_cp(score_cp)
                        for score_cp in example.teacher_candidate_score_delta_targets_cp
                    ],
                    dtype=torch.float32,
                )
            else:
                teacher_candidate_score_targets[row_index, :candidate_count] = torch.tensor(
                    [
                        _normalize_teacher_candidate_score_cp(
                            score_cp - example.teacher_root_value_cp
                        )
                        for score_cp in example.teacher_candidate_scores_cp
                    ],
                    dtype=torch.float32,
                )
            teacher_candidate_score_mask[row_index, :candidate_count] = True
        curriculum_priorities[row_index] = float(example.curriculum_priority)
        teacher_root_value_targets[row_index] = _normalize_root_value_cp(
            example.teacher_root_value_cp
        )
        if example.teacher_top1_minus_top2_cp is not None:
            teacher_root_gap_targets[row_index] = _normalize_root_gap_cp(
                example.teacher_top1_minus_top2_cp
            )
            teacher_root_gap_mask[row_index] = True

    return {
        "root_features": root_features,
        "global_features": global_features,
        "candidate_action_indices": candidate_action_indices,
        "candidate_features": candidate_features,
        "proposer_scores": proposer_scores,
        "transition_features": transition_features,
        "latent_features": latent_features,
        "reply_peak_probabilities": reply_peak_probabilities,
        "pressures": pressures,
        "uncertainties": uncertainties,
        "candidate_mask": candidate_mask,
        "teacher_top1_candidate_indices": teacher_top1_candidate_indices,
        "teacher_policy": teacher_policy,
        "teacher_candidate_score_targets": teacher_candidate_score_targets,
        "teacher_candidate_score_mask": teacher_candidate_score_mask,
        "curriculum_priorities": curriculum_priorities,
        "teacher_root_value_targets": teacher_root_value_targets,
        "teacher_root_gap_targets": teacher_root_gap_targets,
        "teacher_root_gap_mask": teacher_root_gap_mask,
    }


def _configure_torch_runtime(torch_threads: int) -> None:
    if torch is None:
        return
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _validate_latent_feature_dim(
    examples: list[PlannerHeadExample],
    *,
    expected_dim: int,
    context: str,
) -> None:
    if expected_dim == 0:
        return
    observed_dims = {
        (
            len(example.latent_features[0])
            if example.latent_features is not None and example.latent_features
            else 0
        )
        for example in examples
    }
    if not observed_dims:
        observed_dims = {0}
    if observed_dims != {expected_dim}:
        raise ValueError(
            f"{context}: planner latent feature dimension {sorted(observed_dims)} "
            f"does not match config.model.latent_feature_dim={expected_dim}"
        )


def _normalize_root_value_cp(value_cp: float) -> float:
    return float(value_cp) / PLANNER_ROOT_VALUE_SCALE_CP


def _normalize_root_gap_cp(value_cp: float) -> float:
    return float(value_cp) / PLANNER_ROOT_GAP_SCALE_CP


def _normalize_teacher_candidate_score_cp(value_cp: float) -> float:
    return float(value_cp) / PLANNER_CANDIDATE_SCORE_SCALE_CP
