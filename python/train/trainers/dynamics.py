"""Training loop and evaluation metrics for the Phase-6 latent dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
import time
from typing import Any

from train.config import DynamicsTrainConfig, resolve_repo_path
from train.datasets.artifacts import (
    DynamicsTrainingExample,
    PIECE_FEATURE_SIZE,
    RULE_FEATURE_SIZE,
    SQUARE_FEATURE_SIZE,
    load_dynamics_examples,
)
from train.export.dynamics import export_dynamics_bundle
from train.losses.dynamics import compute_dynamics_losses
from train.models.dynamics import LatentDynamicsModel, torch_is_available

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None


@dataclass(frozen=True)
class DynamicsMetrics:
    """Aggregated dynamics losses and held-out transition metrics."""

    total_examples: int
    total_loss: float
    reconstruction_loss: float
    piece_loss: float
    square_loss: float
    rule_loss: float
    feature_l1_error: float
    piece_feature_l1_error: float
    square_feature_l1_error: float
    rule_feature_l1_error: float
    exact_next_feature_accuracy: float
    capture_examples: int
    capture_exact_next_feature_accuracy: float
    promotion_examples: int
    promotion_exact_next_feature_accuracy: float
    castle_examples: int
    castle_exact_next_feature_accuracy: float
    en_passant_examples: int
    en_passant_exact_next_feature_accuracy: float
    gives_check_examples: int
    gives_check_exact_next_feature_accuracy: float
    drift_examples: int
    drift_feature_l1_error: float
    drift_exact_next_feature_accuracy: float
    examples_per_second: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Return the JSON-friendly representation."""
        return {
            "total_examples": self.total_examples,
            "total_loss": round(self.total_loss, 6),
            "reconstruction_loss": round(self.reconstruction_loss, 6),
            "piece_loss": round(self.piece_loss, 6),
            "square_loss": round(self.square_loss, 6),
            "rule_loss": round(self.rule_loss, 6),
            "feature_l1_error": round(self.feature_l1_error, 6),
            "piece_feature_l1_error": round(self.piece_feature_l1_error, 6),
            "square_feature_l1_error": round(self.square_feature_l1_error, 6),
            "rule_feature_l1_error": round(self.rule_feature_l1_error, 6),
            "exact_next_feature_accuracy": round(self.exact_next_feature_accuracy, 6),
            "capture_examples": self.capture_examples,
            "capture_exact_next_feature_accuracy": round(
                self.capture_exact_next_feature_accuracy,
                6,
            ),
            "promotion_examples": self.promotion_examples,
            "promotion_exact_next_feature_accuracy": round(
                self.promotion_exact_next_feature_accuracy,
                6,
            ),
            "castle_examples": self.castle_examples,
            "castle_exact_next_feature_accuracy": round(
                self.castle_exact_next_feature_accuracy,
                6,
            ),
            "en_passant_examples": self.en_passant_examples,
            "en_passant_exact_next_feature_accuracy": round(
                self.en_passant_exact_next_feature_accuracy,
                6,
            ),
            "gives_check_examples": self.gives_check_examples,
            "gives_check_exact_next_feature_accuracy": round(
                self.gives_check_exact_next_feature_accuracy,
                6,
            ),
            "drift_examples": self.drift_examples,
            "drift_feature_l1_error": round(self.drift_feature_l1_error, 6),
            "drift_exact_next_feature_accuracy": round(self.drift_exact_next_feature_accuracy, 6),
            "examples_per_second": round(self.examples_per_second, 3),
        }


@dataclass(frozen=True)
class DynamicsTrainingRun:
    """Serializable result of a dynamics training run."""

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


def train_dynamics(config: DynamicsTrainConfig, *, repo_root: Path) -> DynamicsTrainingRun:
    """Train the first action-conditioned latent dynamics model and export the best checkpoint."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for dynamics training. Install the 'train' extra or torch."
        )

    output_dir = resolve_repo_path(repo_root, config.output_dir)
    bundle_dir = resolve_repo_path(repo_root, config.export.bundle_dir)
    dataset_path = resolve_repo_path(repo_root, config.data.dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(config.seed)
    _configure_torch_runtime(config.runtime.torch_threads)

    train_examples = load_dynamics_examples(
        dataset_path,
        config.data.train_split,
        repo_root=repo_root,
    )
    validation_examples = load_dynamics_examples(
        dataset_path,
        config.data.validation_split,
        repo_root=repo_root,
    )
    if not train_examples:
        raise ValueError("training split has no action-conditioned transitions")
    if not validation_examples:
        raise ValueError("validation split has no action-conditioned transitions")

    model = LatentDynamicsModel(
        architecture=config.model.architecture,
        latent_dim=config.model.latent_dim,
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
    best_validation: DynamicsMetrics | None = None
    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    for epoch in range(1, config.optimization.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            examples=train_examples,
            optimizer=optimizer,
            training=True,
            reconstruction_weight=config.optimization.reconstruction_loss_weight,
            piece_weight=config.optimization.piece_loss_weight,
            square_weight=config.optimization.square_loss_weight,
            rule_weight=config.optimization.rule_loss_weight,
            drift_horizon=config.evaluation.drift_horizon,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            examples=validation_examples,
            optimizer=None,
            training=False,
            reconstruction_weight=config.optimization.reconstruction_loss_weight,
            piece_weight=config.optimization.piece_loss_weight,
            square_weight=config.optimization.square_loss_weight,
            rule_weight=config.optimization.rule_loss_weight,
            drift_horizon=config.evaluation.drift_horizon,
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
    export_paths = export_dynamics_bundle(
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

    return DynamicsTrainingRun(
        history=history,
        best_epoch=best_epoch,
        best_validation=best_validation.to_dict(),
        export_paths=export_paths,
        summary_path=str(summary_path),
        model_parameter_count=model_parameter_count,
    )


def evaluate_dynamics_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_path: Path,
    split: str,
    drift_horizon: int,
    repo_root: Path,
) -> DynamicsMetrics:
    """Load a saved dynamics checkpoint and evaluate it on one dataset split."""
    if torch is None or not torch_is_available():  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for dynamics evaluation. Install the 'train' extra or torch."
        )

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = DynamicsTrainConfig.from_dict(dict(payload["training_config"]))
    model = LatentDynamicsModel(
        architecture=config.model.architecture,
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        action_embedding_dim=config.model.action_embedding_dim,
        dropout=config.model.dropout,
    )
    model.load_state_dict(dict(payload["model_state_dict"]))

    examples = load_dynamics_examples(dataset_path, split, repo_root=repo_root)
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
        examples=examples,
        optimizer=None,
        training=False,
        reconstruction_weight=config.optimization.reconstruction_loss_weight,
        piece_weight=config.optimization.piece_loss_weight,
        square_weight=config.optimization.square_loss_weight,
        rule_weight=config.optimization.rule_loss_weight,
        drift_horizon=drift_horizon,
    )


def _build_loader(
    examples: list[DynamicsTrainingExample],
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
    examples: list[DynamicsTrainingExample],
    optimizer: Any,
    training: bool,
    reconstruction_weight: float,
    piece_weight: float,
    square_weight: float,
    rule_weight: float,
    drift_horizon: int,
) -> DynamicsMetrics:
    if training:
        model.train()
    else:
        model.eval()

    total_examples = 0
    total_loss = 0.0
    reconstruction_loss_total = 0.0
    piece_loss_total = 0.0
    square_loss_total = 0.0
    rule_loss_total = 0.0
    feature_l1_total = 0.0
    piece_feature_l1_total = 0.0
    square_feature_l1_total = 0.0
    rule_feature_l1_total = 0.0
    exact_next_total = 0
    capture_examples = 0
    capture_exact = 0
    promotion_examples = 0
    promotion_exact = 0
    castle_examples = 0
    castle_exact = 0
    en_passant_examples = 0
    en_passant_exact = 0
    gives_check_examples = 0
    gives_check_exact = 0
    started_at = time.perf_counter()

    for batch in loader:
        features = batch["features"]
        action_indices = batch["action_indices"]
        next_features = batch["next_features"]

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            prediction = model.predict(features, action_indices)
            predicted_next = prediction.next_features
            target_piece, target_square, target_rule = torch.split(
                next_features,
                [PIECE_FEATURE_SIZE, SQUARE_FEATURE_SIZE, RULE_FEATURE_SIZE],
                dim=1,
            )
            losses = compute_dynamics_losses(
                predicted_next,
                next_features,
                predicted_piece_features=prediction.piece_features,
                predicted_square_features=prediction.square_features,
                predicted_rule_features=prediction.rule_features,
                target_piece_features=target_piece,
                target_square_features=target_square,
                target_rule_features=target_rule,
                reconstruction_weight=reconstruction_weight,
                piece_weight=piece_weight,
                square_weight=square_weight,
                rule_weight=rule_weight,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            optimizer.step()

        batch_size = int(features.shape[0])
        total_examples += batch_size
        total_loss += float(losses.total.item()) * batch_size
        reconstruction_loss_total += float(losses.reconstruction.item()) * batch_size
        piece_loss_total += float(losses.piece.item()) * batch_size
        square_loss_total += float(losses.square.item()) * batch_size
        rule_loss_total += float(losses.rule.item()) * batch_size

        exact_mask = _exact_feature_match_mask(predicted_next, next_features)
        feature_l1_total += float(torch.abs(predicted_next - next_features).mean(dim=1).sum().item())
        piece_feature_l1_total += float(
            torch.abs(prediction.piece_features - target_piece).mean(dim=1).sum().item()
        )
        square_feature_l1_total += float(
            torch.abs(prediction.square_features - target_square).mean(dim=1).sum().item()
        )
        rule_feature_l1_total += float(
            torch.abs(prediction.rule_features - target_rule).mean(dim=1).sum().item()
        )
        exact_next_total += int(exact_mask.sum().item())

        capture_examples += int(batch["is_capture"].sum().item())
        capture_exact += int(exact_mask[batch["is_capture"]].sum().item())
        promotion_examples += int(batch["is_promotion"].sum().item())
        promotion_exact += int(exact_mask[batch["is_promotion"]].sum().item())
        castle_examples += int(batch["is_castle"].sum().item())
        castle_exact += int(exact_mask[batch["is_castle"]].sum().item())
        en_passant_examples += int(batch["is_en_passant"].sum().item())
        en_passant_exact += int(exact_mask[batch["is_en_passant"]].sum().item())
        gives_check_examples += int(batch["gives_check"].sum().item())
        gives_check_exact += int(exact_mask[batch["gives_check"]].sum().item())

    drift_metrics = (
        _evaluate_multistep_drift(model, examples, horizon=drift_horizon) if not training else None
    )

    return DynamicsMetrics(
        total_examples=total_examples,
        total_loss=_ratio(total_loss, total_examples),
        reconstruction_loss=_ratio(reconstruction_loss_total, total_examples),
        piece_loss=_ratio(piece_loss_total, total_examples),
        square_loss=_ratio(square_loss_total, total_examples),
        rule_loss=_ratio(rule_loss_total, total_examples),
        feature_l1_error=_ratio(feature_l1_total, total_examples),
        piece_feature_l1_error=_ratio(piece_feature_l1_total, total_examples),
        square_feature_l1_error=_ratio(square_feature_l1_total, total_examples),
        rule_feature_l1_error=_ratio(rule_feature_l1_total, total_examples),
        exact_next_feature_accuracy=_ratio(exact_next_total, total_examples),
        capture_examples=capture_examples,
        capture_exact_next_feature_accuracy=_ratio(capture_exact, capture_examples),
        promotion_examples=promotion_examples,
        promotion_exact_next_feature_accuracy=_ratio(promotion_exact, promotion_examples),
        castle_examples=castle_examples,
        castle_exact_next_feature_accuracy=_ratio(castle_exact, castle_examples),
        en_passant_examples=en_passant_examples,
        en_passant_exact_next_feature_accuracy=_ratio(en_passant_exact, en_passant_examples),
        gives_check_examples=gives_check_examples,
        gives_check_exact_next_feature_accuracy=_ratio(gives_check_exact, gives_check_examples),
        drift_examples=0 if drift_metrics is None else drift_metrics["count"],
        drift_feature_l1_error=0.0 if drift_metrics is None else drift_metrics["feature_l1_error"],
        drift_exact_next_feature_accuracy=(
            0.0 if drift_metrics is None else drift_metrics["exact_next_feature_accuracy"]
        ),
        examples_per_second=_ratio(total_examples, time.perf_counter() - started_at),
    )


def _evaluate_multistep_drift(
    model: Any,
    examples: list[DynamicsTrainingExample],
    *,
    horizon: int,
) -> dict[str, float | int]:
    chains = _build_drift_chains(examples, horizon=horizon)
    if not chains:
        return {"count": 0, "feature_l1_error": 0.0, "exact_next_feature_accuracy": 0.0}

    model.eval()
    total_l1 = 0.0
    exact_total = 0
    with torch.no_grad():
        for chain in chains:
            latent = model.encode(
                torch.tensor([chain[0].feature_vector], dtype=torch.float32)
            )
            for example in chain:
                latent = model.step(
                    latent,
                    torch.tensor([example.action_index], dtype=torch.long),
                )
            predicted = model.decode(latent).next_features
            target = torch.tensor([chain[-1].next_feature_vector], dtype=torch.float32)
            total_l1 += float(torch.abs(predicted - target).mean().item())
            exact_total += int(_exact_feature_match_mask(predicted, target).sum().item())

    return {
        "count": len(chains),
        "feature_l1_error": _ratio(total_l1, len(chains)),
        "exact_next_feature_accuracy": _ratio(exact_total, len(chains)),
    }


def _build_drift_chains(
    examples: list[DynamicsTrainingExample],
    *,
    horizon: int,
) -> list[list[DynamicsTrainingExample]]:
    groups: dict[str, list[DynamicsTrainingExample]] = {}
    for example in examples:
        if example.trajectory_id is None or example.ply_index is None:
            continue
        groups.setdefault(example.trajectory_id, []).append(example)

    chains: list[list[DynamicsTrainingExample]] = []
    for trajectory in groups.values():
        ordered = sorted(trajectory, key=lambda example: int(example.ply_index or 0))
        for start_index in range(0, len(ordered) - horizon + 1):
            chain = ordered[start_index : start_index + horizon]
            if not _is_contiguous_chain(chain):
                continue
            chains.append(chain)
    return chains


def _is_contiguous_chain(chain: list[DynamicsTrainingExample]) -> bool:
    for current, nxt in zip(chain, chain[1:]):
        if current.ply_index is None or nxt.ply_index is None:
            return False
        if nxt.ply_index != current.ply_index + 1:
            return False
        if current.next_feature_vector != nxt.feature_vector:
            return False
    return True


def _is_better_validation(current: DynamicsMetrics, best: DynamicsMetrics) -> bool:
    if current.exact_next_feature_accuracy != best.exact_next_feature_accuracy:
        return current.exact_next_feature_accuracy > best.exact_next_feature_accuracy
    if current.drift_exact_next_feature_accuracy != best.drift_exact_next_feature_accuracy:
        return current.drift_exact_next_feature_accuracy > best.drift_exact_next_feature_accuracy
    if current.feature_l1_error != best.feature_l1_error:
        return current.feature_l1_error < best.feature_l1_error
    return current.drift_feature_l1_error < best.drift_feature_l1_error


def _exact_feature_match_mask(predicted_next: Any, next_features: Any) -> Any:
    discrete_predicted = torch.round(predicted_next)
    return torch.all(discrete_predicted == next_features, dim=1)


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
        """Tensor-backed dataset for one-step dynamics supervision."""

        def __init__(
            self,
            features: Any,
            action_indices: Any,
            next_features: Any,
            is_capture: Any,
            is_promotion: Any,
            is_castle: Any,
            is_en_passant: Any,
            gives_check: Any,
        ) -> None:
            self._features = features
            self._action_indices = action_indices
            self._next_features = next_features
            self._is_capture = is_capture
            self._is_promotion = is_promotion
            self._is_castle = is_castle
            self._is_en_passant = is_en_passant
            self._gives_check = gives_check

        @classmethod
        def from_examples(cls, examples: list[DynamicsTrainingExample]) -> "_TensorDataset":
            return cls(
                features=torch.tensor(
                    [example.feature_vector for example in examples],
                    dtype=torch.float32,
                ),
                action_indices=torch.tensor(
                    [example.action_index for example in examples],
                    dtype=torch.long,
                ),
                next_features=torch.tensor(
                    [example.next_feature_vector for example in examples],
                    dtype=torch.float32,
                ),
                is_capture=torch.tensor(
                    [example.is_capture for example in examples],
                    dtype=torch.bool,
                ),
                is_promotion=torch.tensor(
                    [example.is_promotion for example in examples],
                    dtype=torch.bool,
                ),
                is_castle=torch.tensor(
                    [example.is_castle for example in examples],
                    dtype=torch.bool,
                ),
                is_en_passant=torch.tensor(
                    [example.is_en_passant for example in examples],
                    dtype=torch.bool,
                ),
                gives_check=torch.tensor(
                    [example.gives_check for example in examples],
                    dtype=torch.bool,
                ),
            )

        def __len__(self) -> int:
            return int(self._features.shape[0])

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "features": self._features[index],
                "action_indices": self._action_indices[index],
                "next_features": self._next_features[index],
                "is_capture": self._is_capture[index],
                "is_promotion": self._is_promotion[index],
                "is_castle": self._is_castle[index],
                "is_en_passant": self._is_en_passant[index],
                "gives_check": self._gives_check[index],
            }

else:

    class _TensorDataset:  # pragma: no cover - exercised when torch is absent
        """Import-safe fallback when PyTorch is not installed."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for dynamics training. Install the 'train' extra or torch."
            )
