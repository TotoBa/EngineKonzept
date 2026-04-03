"""Helpers for loading Phase-6 dynamics checkpoints for downstream workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from train.models.dynamics import LatentDynamicsModel, torch_is_available

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None

if TYPE_CHECKING:
    from train.config import DynamicsTrainConfig


def load_dynamics_checkpoint(
    checkpoint_path: Path,
) -> tuple[LatentDynamicsModel, "DynamicsTrainConfig"]:
    """Load a trained dynamics checkpoint for offline workflow use."""
    if torch is None or not torch_is_available():  # pragma: no cover - torch absent
        raise RuntimeError(
            "PyTorch is required for dynamics evaluation. Install the 'train' extra or torch."
        )
    from train.config import DynamicsTrainConfig

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
    model.eval()
    return model, config


def predict_dynamics_latent(
    model: LatentDynamicsModel,
    *,
    feature_vector: Sequence[float],
    action_index: int,
    action_features: Sequence[float] | None,
    transition_features: Sequence[float] | None,
) -> list[float]:
    """Predict the next latent-state vector for one exact transition."""
    if torch is None or not torch_is_available():  # pragma: no cover - torch absent
        raise RuntimeError(
            "PyTorch is required for dynamics evaluation. Install the 'train' extra or torch."
        )
    features = torch.tensor([list(feature_vector)], dtype=torch.float32)
    action_indices = torch.tensor([int(action_index)], dtype=torch.long)
    action_feature_tensor = (
        torch.tensor([list(action_features)], dtype=torch.float32)
        if action_features is not None
        else None
    )
    transition_feature_tensor = (
        torch.tensor([list(transition_features)], dtype=torch.float32)
        if transition_features is not None
        else None
    )
    with torch.inference_mode():
        prediction = model.predict(
            features,
            action_indices,
            action_features=action_feature_tensor,
            transition_features=transition_feature_tensor,
        )
    next_latent = prediction.next_latent
    if next_latent is None:
        raise ValueError("dynamics checkpoint did not produce next_latent")
    return [float(value) for value in next_latent[0].tolist()]
