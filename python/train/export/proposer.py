"""Export helpers for the Phase-5 proposer bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from train.action_space import ACTION_SPACE_SIZE, action_space_metadata
from train.config import ProposerTrainConfig
from train.datasets.artifacts import position_feature_spec
from train.models.proposer import MODEL_NAME, torch_is_available

PROPOSER_EXPORT_SCHEMA_VERSION = 2


def build_export_metadata(
    config: ProposerTrainConfig,
    *,
    validation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the JSON metadata that Rust will load for the proposer bundle."""
    return {
        "schema_version": PROPOSER_EXPORT_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "artifacts": {
            "checkpoint_file": config.export.checkpoint_name,
            "exported_program_file": config.export.exported_program_name,
        },
        "input": position_feature_spec(),
        "action_space": action_space_metadata(),
        "outputs": {
            "legality_logits_shape": {"batch": "dynamic", "actions": ACTION_SPACE_SIZE},
            "policy_logits_shape": {"batch": "dynamic", "actions": ACTION_SPACE_SIZE},
            "legality_threshold": config.evaluation.legality_threshold,
        },
        "training": {
            "seed": config.seed,
            "train_split": config.data.train_split,
            "validation_split": config.data.validation_split,
            "hidden_dim": config.model.hidden_dim,
            "hidden_layers": config.model.hidden_layers,
            "dropout": config.model.dropout,
            "epochs": config.optimization.epochs,
            "batch_size": config.optimization.batch_size,
            "learning_rate": config.optimization.learning_rate,
            "weight_decay": config.optimization.weight_decay,
            "legality_loss_weight": config.optimization.legality_loss_weight,
            "policy_loss_weight": config.optimization.policy_loss_weight,
        },
        "validation_metrics": dict(validation_metrics),
    }


def export_proposer_bundle(
    model: Any,
    *,
    config: ProposerTrainConfig,
    bundle_dir: Path,
    validation_metrics: Mapping[str, Any],
) -> dict[str, str]:
    """Export checkpoint, torch.export program, and metadata for Rust consumption."""
    if not torch_is_available():  # pragma: no cover - exercised when torch is absent
        raise RuntimeError(
            "PyTorch is required for proposer export. Install the 'train' extra or torch."
        )

    import torch

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

    export_input = torch.zeros((2, position_feature_spec()["feature_dim"]), dtype=torch.float32)
    exported_program = torch.export.export(
        model.cpu(),
        (export_input,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
    )
    torch.export.save(exported_program, exported_program_path)

    metadata = build_export_metadata(config, validation_metrics=validation_metrics)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "checkpoint": str(checkpoint_path),
        "exported_program": str(exported_program_path),
        "metadata": str(metadata_path),
    }
