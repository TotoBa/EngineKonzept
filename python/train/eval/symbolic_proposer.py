"""Shared symbolic proposer checkpoint loading and exact-candidate scoring helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from train.models.proposer import LegalityPolicyProposer, torch_is_available

if TYPE_CHECKING:
    from train.config import ProposerTrainConfig

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised when torch is absent
    torch = None


def load_symbolic_proposer_checkpoint(
    checkpoint_path: Path,
) -> tuple[LegalityPolicyProposer, "ProposerTrainConfig"]:
    """Load a symbolic proposer checkpoint for exact-candidate scoring workflows."""
    if torch is None or not torch_is_available():  # pragma: no cover - torch absent
        raise RuntimeError(
            "PyTorch is required for symbolic proposer evaluation. Install the 'train' extra or torch."
        )
    from train.config import ProposerTrainConfig

    payload = torch.load(checkpoint_path, map_location="cpu")
    config = ProposerTrainConfig.from_dict(dict(payload["training_config"]))
    if config.model.architecture != "symbolic_v1":
        raise ValueError("checkpoint must use proposer architecture 'symbolic_v1'")

    model = LegalityPolicyProposer(
        architecture=config.model.architecture,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        dropout=config.model.dropout,
    )
    model.load_state_dict(dict(payload["model_state_dict"]))
    model.eval()
    return model, config


def score_symbolic_candidates(
    model: LegalityPolicyProposer,
    *,
    feature_vector: Sequence[float],
    candidate_action_indices: Sequence[int],
    candidate_features: Sequence[Sequence[float]],
    global_features: Sequence[float],
    candidate_context_version: int = 1,
) -> tuple[list[float], list[float]]:
    """Score an exact legal candidate set with a loaded symbolic proposer."""
    if torch is None or not torch_is_available():  # pragma: no cover - torch absent
        raise RuntimeError(
            "PyTorch is required for symbolic proposer evaluation. Install the 'train' extra or torch."
        )

    normalized_candidate_features = _normalize_candidate_features(
        candidate_features,
        version=candidate_context_version,
    )
    features = torch.tensor([list(feature_vector)], dtype=torch.float32)
    action_indices = torch.tensor([list(candidate_action_indices)], dtype=torch.long)
    feature_rows = torch.tensor([normalized_candidate_features], dtype=torch.float32)
    candidate_mask = torch.ones(action_indices.shape, dtype=torch.bool)
    global_context = torch.tensor([list(global_features)], dtype=torch.float32)

    with torch.inference_mode():
        _, policy_logits = model(
            features,
            candidate_action_indices=action_indices,
            candidate_features=feature_rows,
            candidate_mask=candidate_mask,
            global_features=global_context,
        )
        candidate_scores = policy_logits[0, action_indices[0]].tolist()
        candidate_policy = torch.softmax(
            torch.tensor(candidate_scores, dtype=torch.float32),
            dim=0,
        ).tolist()
    return (
        [float(value) for value in candidate_scores],
        [float(value) for value in candidate_policy],
    )


def _normalize_candidate_features(
    candidate_features: Sequence[Sequence[float]],
    *,
    version: int,
) -> list[list[float]]:
    if version == 1:
        return [list(row) for row in candidate_features]
    if version != 2:
        raise ValueError(f"unsupported candidate context version for symbolic scoring: {version}")
    normalized: list[list[float]] = []
    for row in candidate_features:
        normalized.append(
            [
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
                float(row[9]),
                float(row[10]),
                float(row[11]),
                float(row[12]),
                float(row[13]),
                float(row[14]),
                float(row[15]),
                float(row[16]),
                float(bool(row[17] or row[18] or row[19] or row[20])),
            ]
        )
    return normalized
