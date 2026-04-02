"""Training loop and evaluation metrics for the Phase-5 proposer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
import time
from typing import Any

from train.action_space import ACTION_SPACE_SIZE
from train.config import ProposerTrainConfig, resolve_repo_path
from train.datasets.artifacts import ProposerTrainingExample, load_proposer_examples
from train.export.proposer import export_proposer_bundle
from train.losses.proposer import compute_proposer_losses
from train.models.proposer import LegalityPolicyProposer, torch_is_available

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None


@dataclass(frozen=True)
class ProposerMetrics:
    """Aggregated proposer losses and held-out metrics."""

    total_examples: int
    labeled_policy_examples: int
    total_loss: float
    legality_loss: float
    policy_loss: float
    legal_set_precision: float
    legal_set_recall: float
    legal_set_f1: float
    policy_top1_accuracy: float
    examples_per_second: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Return the JSON-friendly representation."""
        return {
            "total_examples": self.total_examples,
            "labeled_policy_examples": self.labeled_policy_examples,
            "total_loss": round(self.total_loss, 6),
            "legality_loss": round(self.legality_loss, 6),
            "policy_loss": round(self.policy_loss, 6),
            "legal_set_precision": round(self.legal_set_precision, 6),
            "legal_set_recall": round(self.legal_set_recall, 6),
            "legal_set_f1": round(self.legal_set_f1, 6),
            "policy_top1_accuracy": round(self.policy_top1_accuracy, 6),
            "examples_per_second": round(self.examples_per_second, 3),
        }


@dataclass(frozen=True)
class ProposerTrainingRun:
    """Serializable result of a proposer training run."""

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


def train_proposer(config: ProposerTrainConfig, *, repo_root: Path) -> ProposerTrainingRun:
    """Train the first legality/policy proposer and export the best checkpoint."""
    if (
        torch is None or not torch_is_available()
    ):  # pragma: no cover - exercised when torch is absent
        raise RuntimeError(
            "PyTorch is required for proposer training. Install the 'train' extra or torch."
        )

    output_dir = resolve_repo_path(repo_root, config.output_dir)
    bundle_dir = resolve_repo_path(repo_root, config.export.bundle_dir)
    dataset_path = resolve_repo_path(repo_root, config.data.dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(config.seed)
    _configure_torch_runtime(config.runtime.torch_threads)

    train_examples = load_proposer_examples(dataset_path, config.data.train_split)
    validation_examples = load_proposer_examples(dataset_path, config.data.validation_split)
    if not train_examples:
        raise ValueError("training split is empty")
    if not validation_examples:
        raise ValueError("validation split is empty")

    model = LegalityPolicyProposer(
        architecture=config.model.architecture,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
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
    best_validation: ProposerMetrics | None = None
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    for epoch in range(1, config.optimization.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            training=True,
            legality_threshold=config.evaluation.legality_threshold,
            legality_loss_weight=config.optimization.legality_loss_weight,
            policy_loss_weight=config.optimization.policy_loss_weight,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            optimizer=None,
            training=False,
            legality_threshold=config.evaluation.legality_threshold,
            legality_loss_weight=config.optimization.legality_loss_weight,
            policy_loss_weight=config.optimization.policy_loss_weight,
        )

        history.append(
            {
                "epoch": epoch,
                "examples_per_second": round(train_metrics.examples_per_second, 3),
                "train": train_metrics.to_dict(),
                "validation": validation_metrics.to_dict(),
            }
        )

        if best_validation is None or _is_better_validation(
            validation_metrics,
            best_validation,
            selection_mode=config.evaluation.checkpoint_selection,
            selection_policy_weight=config.evaluation.selection_policy_weight,
        ):
            best_epoch = epoch
            best_validation = validation_metrics
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }

    assert best_validation is not None
    model.load_state_dict(best_state)
    export_paths = export_proposer_bundle(
        model,
        config=config,
        bundle_dir=bundle_dir,
        validation_metrics=best_validation.to_dict(),
    )

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

    return ProposerTrainingRun(
        history=history,
        best_epoch=best_epoch,
        best_validation=best_validation.to_dict(),
        export_paths=export_paths,
        summary_path=str(summary_path),
        model_parameter_count=model_parameter_count,
    )


def evaluate_proposer_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_path: Path,
    split: str,
    legality_threshold: float,
) -> ProposerMetrics:
    """Load a saved proposer checkpoint and evaluate it on one dataset split."""
    if (
        torch is None or not torch_is_available()
    ):  # pragma: no cover - exercised when torch is absent
        raise RuntimeError(
            "PyTorch is required for proposer evaluation. Install the 'train' extra or torch."
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = ProposerTrainConfig.from_dict(dict(payload["training_config"]))
    model = LegalityPolicyProposer(
        architecture=config.model.architecture,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        dropout=config.model.dropout,
    )
    model.load_state_dict(dict(payload["model_state_dict"]))

    examples = load_proposer_examples(dataset_path, split)
    if not examples:
        raise ValueError(f"evaluation split is empty: {split}")

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
        legality_threshold=legality_threshold,
        legality_loss_weight=config.optimization.legality_loss_weight,
        policy_loss_weight=config.optimization.policy_loss_weight,
    )


def _build_loader(
    examples: list[ProposerTrainingExample],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> Any:
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        _TensorDataset.from_examples(examples),
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
    legality_threshold: float,
    legality_loss_weight: float,
    policy_loss_weight: float,
) -> ProposerMetrics:
    if training:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss = 0.0
    legality_loss_total = 0.0
    policy_loss_total = 0.0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    policy_correct = 0
    policy_examples = 0
    started_at = time.perf_counter()

    for batch in loader:
        features = batch["features"]
        legal_targets = batch["legal_targets"]
        selected_action_indices = batch["selected_action_indices"]

        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            legality_logits, policy_logits = model(features)
            losses = compute_proposer_losses(
                legality_logits,
                policy_logits,
                legal_targets,
                selected_action_indices,
                legality_weight=legality_loss_weight,
                policy_weight=policy_loss_weight,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            optimizer.step()

        batch_size = int(features.shape[0])
        total_examples += batch_size
        total_loss += float(losses.total.item()) * batch_size
        legality_loss_total += float(losses.legality.item()) * batch_size
        policy_loss_total += float(losses.policy.item()) * batch_size

        predicted_legal = torch.sigmoid(legality_logits) >= legality_threshold
        true_legal = legal_targets.bool()
        true_positive += int((predicted_legal & true_legal).sum().item())
        false_positive += int((predicted_legal & ~true_legal).sum().item())
        false_negative += int((~predicted_legal & true_legal).sum().item())

        labeled_policy = selected_action_indices != -100
        if bool(labeled_policy.any()):
            policy_predictions = torch.argmax(policy_logits[labeled_policy], dim=1)
            policy_correct += int(
                (policy_predictions == selected_action_indices[labeled_policy]).sum().item()
            )
            policy_examples += int(labeled_policy.sum().item())

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2.0 * precision * recall, precision + recall) if precision or recall else 0.0

    return ProposerMetrics(
        total_examples=total_examples,
        labeled_policy_examples=policy_examples,
        total_loss=_ratio(total_loss, total_examples),
        legality_loss=_ratio(legality_loss_total, total_examples),
        policy_loss=_ratio(policy_loss_total, total_examples),
        legal_set_precision=precision,
        legal_set_recall=recall,
        legal_set_f1=f1,
        policy_top1_accuracy=_ratio(policy_correct, policy_examples),
        examples_per_second=_ratio(total_examples, time.perf_counter() - started_at),
    )


def _is_better_validation(
    current: ProposerMetrics,
    best: ProposerMetrics,
    *,
    selection_mode: str,
    selection_policy_weight: float,
) -> bool:
    if selection_mode == "legality_first":
        if current.legal_set_f1 != best.legal_set_f1:
            return current.legal_set_f1 > best.legal_set_f1
        if current.legal_set_recall != best.legal_set_recall:
            return current.legal_set_recall > best.legal_set_recall
        return current.policy_top1_accuracy > best.policy_top1_accuracy

    if selection_mode == "policy_first":
        if current.policy_top1_accuracy != best.policy_top1_accuracy:
            return current.policy_top1_accuracy > best.policy_top1_accuracy
        if current.legal_set_f1 != best.legal_set_f1:
            return current.legal_set_f1 > best.legal_set_f1
        return current.legal_set_recall > best.legal_set_recall

    current_score = current.legal_set_f1 + (
        selection_policy_weight * current.policy_top1_accuracy
    )
    best_score = best.legal_set_f1 + (selection_policy_weight * best.policy_top1_accuracy)
    if current_score != best_score:
        return current_score > best_score
    if current.policy_top1_accuracy != best.policy_top1_accuracy:
        return current.policy_top1_accuracy > best.policy_top1_accuracy
    return current.legal_set_recall > best.legal_set_recall


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _configure_torch_runtime(torch_threads: int) -> None:
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


if torch is not None:

    class _TensorDataset(torch.utils.data.Dataset):
        """Tensor-backed dataset that avoids rebuilding dense targets every batch."""

        def __init__(
            self,
            features: Any,
            legal_targets: Any,
            selected_action_indices: Any,
        ) -> None:
            self._features = features
            self._legal_targets = legal_targets
            self._selected_action_indices = selected_action_indices

        @classmethod
        def from_examples(cls, examples: list[ProposerTrainingExample]) -> "_TensorDataset":
            features = torch.tensor(
                [example.feature_vector for example in examples],
                dtype=torch.float32,
            )
            legal_targets = torch.zeros((len(examples), ACTION_SPACE_SIZE), dtype=torch.float32)
            selected_action_indices = torch.full((len(examples),), -100, dtype=torch.long)

            for row_index, example in enumerate(examples):
                if example.legal_action_indices:
                    legal_targets[row_index, example.legal_action_indices] = 1.0
                if example.selected_action_index is not None:
                    selected_action_indices[row_index] = example.selected_action_index

            return cls(features, legal_targets, selected_action_indices)

        def __len__(self) -> int:
            return int(self._features.shape[0])

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "features": self._features[index],
                "legal_targets": self._legal_targets[index],
                "selected_action_indices": self._selected_action_indices[index],
            }

else:

    class _TensorDataset:  # pragma: no cover - exercised when torch is absent
        """Import-safe fallback when PyTorch is not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for proposer training. Install the 'train' extra or torch."
            )
