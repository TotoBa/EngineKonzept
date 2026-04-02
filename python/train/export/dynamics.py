"""Export helpers for the Phase-6 latent dynamics bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from train.action_space import action_space_metadata
from train.config import DynamicsTrainConfig
from train.datasets.artifacts import (
    dynamics_symbolic_action_feature_spec,
    position_feature_spec,
    transition_context_feature_spec,
)
from train.models.dynamics import DYNAMICS_MODEL_NAME, torch_is_available

DYNAMICS_EXPORT_SCHEMA_VERSION = 1

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None


def build_dynamics_export_metadata(
    config: DynamicsTrainConfig,
    *,
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the JSON metadata that Rust will load for the dynamics bundle."""
    return {
        "schema_version": DYNAMICS_EXPORT_SCHEMA_VERSION,
        "model_name": DYNAMICS_MODEL_NAME,
        "artifacts": {
            "checkpoint_file": config.export.checkpoint_name,
            "exported_program_file": config.export.exported_program_name,
        },
        "input": {
            "state": position_feature_spec(),
            "action": {
                **action_space_metadata(),
                "dtype": "int64",
                "shape": {"batch": "dynamic"},
                "symbolic": (
                    dynamics_symbolic_action_feature_spec()
                    if config.model.architecture == "structured_v5"
                    else None
                ),
                "transition": (
                    transition_context_feature_spec()
                    if config.model.architecture == "structured_v6"
                    else None
                ),
            },
        },
        "latent": {
            "latent_dim": config.model.latent_dim,
            "action_embedding_dim": config.model.action_embedding_dim,
        },
        "outputs": {
            "next_state_shape": {
                "batch": "dynamic",
                "features": position_feature_spec()["feature_dim"],
            }
        },
        "training": {
            "seed": config.seed,
            "train_split": config.data.train_split,
            "validation_split": config.data.validation_split,
            "architecture": config.model.architecture,
            "latent_dim": config.model.latent_dim,
            "hidden_dim": config.model.hidden_dim,
            "hidden_layers": config.model.hidden_layers,
            "action_embedding_dim": config.model.action_embedding_dim,
            "dropout": config.model.dropout,
            "epochs": config.optimization.epochs,
            "batch_size": config.optimization.batch_size,
            "learning_rate": config.optimization.learning_rate,
            "weight_decay": config.optimization.weight_decay,
            "reconstruction_loss_weight": config.optimization.reconstruction_loss_weight,
            "piece_loss_weight": config.optimization.piece_loss_weight,
            "square_loss_weight": config.optimization.square_loss_weight,
            "rule_loss_weight": config.optimization.rule_loss_weight,
            "delta_loss_weight": config.optimization.delta_loss_weight,
            "latent_consistency_loss_weight": config.optimization.latent_consistency_loss_weight,
            "drift_supervision_loss_weight": config.optimization.drift_supervision_loss_weight,
            "drift_supervision_horizon": config.optimization.drift_supervision_horizon,
            "drift_horizon": config.evaluation.drift_horizon,
            "drift_dataset_path": config.evaluation.drift_dataset_path,
            "drift_split": config.evaluation.drift_split,
        },
        "validation_metrics": dict(validation_metrics),
    }


def export_dynamics_bundle(
    model: Any,
    *,
    config: DynamicsTrainConfig,
    bundle_dir: Path,
    validation_metrics: Mapping[str, Any],
) -> dict[str, str]:
    """Export checkpoint, torch.export program, and metadata for Rust consumption."""
    if not torch_is_available() or torch is None:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for dynamics export. Install the 'train' extra or torch."
        )

    bundle_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = bundle_dir / config.export.checkpoint_name
    exported_program_path = bundle_dir / config.export.exported_program_name
    metadata_path = bundle_dir / config.export.metadata_name

    model.eval()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "training_config": config.to_dict(),
            "validation_metrics": dict(validation_metrics),
        },
        checkpoint_path,
    )

    export_features = torch.zeros((2, position_feature_spec()["feature_dim"]), dtype=torch.float32)
    export_actions = torch.zeros((2,), dtype=torch.int64)
    if config.model.architecture == "structured_v5":
        symbolic_action_spec = dynamics_symbolic_action_feature_spec()
        export_action_features = torch.zeros(
            (2, int(symbolic_action_spec["feature_dim"])),
            dtype=torch.float32,
        )
        exported_program = torch.export.export(
            model.cpu(),
            (export_features, export_actions, export_action_features),
            dynamic_shapes=(
                {0: torch.export.Dim.DYNAMIC},
                {0: torch.export.Dim.DYNAMIC},
                {0: torch.export.Dim.DYNAMIC},
            ),
        )
    elif config.model.architecture == "structured_v6":
        transition_spec = transition_context_feature_spec()
        export_transition_features = torch.zeros(
            (2, int(transition_spec["feature_dim"])),
            dtype=torch.float32,
        )
        wrapper = _StructuredV6ExportWrapper(model.cpu())
        exported_program = torch.export.export(
            wrapper,
            (export_features, export_actions, export_transition_features),
            dynamic_shapes=(
                {0: torch.export.Dim.DYNAMIC},
                {0: torch.export.Dim.DYNAMIC},
                {0: torch.export.Dim.DYNAMIC},
            ),
        )
    else:
        exported_program = torch.export.export(
            model.cpu(),
            (export_features, export_actions),
            dynamic_shapes=(
                {0: torch.export.Dim.DYNAMIC},
                {0: torch.export.Dim.DYNAMIC},
            ),
        )
    torch.export.save(exported_program, exported_program_path)

    metadata = build_dynamics_export_metadata(config, validation_metrics=validation_metrics)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "checkpoint": str(checkpoint_path),
        "exported_program": str(exported_program_path),
        "metadata": str(metadata_path),
    }


if torch is not None:

    class _StructuredV6ExportWrapper(torch.nn.Module):  # pragma: no cover - simple export adapter
        """Expose a three-input export surface for TransitionContext-powered dynamics."""

        def __init__(self, model: Any) -> None:
            super().__init__()
            self._model = model

        def forward(self, features: Any, actions: Any, transition_features: Any) -> Any:
            return self._model(
                features,
                actions,
                transition_features=transition_features,
            )

else:

    class _StructuredV6ExportWrapper:  # pragma: no cover - import-safe fallback
        """Import-safe fallback when PyTorch is unavailable."""

        def __init__(self, *_: Any, **__: Any) -> None:
            raise RuntimeError(
                "PyTorch is required for dynamics export. Install the 'train' extra or torch."
            )
