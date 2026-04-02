"""Configuration schema for Phase-5 proposer training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from train.datasets.schema import SUPPORTED_SPLITS

DEFAULT_PROPOSER_METADATA_NAME = "metadata.json"


@dataclass(frozen=True)
class ProposerDataConfig:
    """Dataset and split selection for proposer training."""

    dataset_path: str
    train_split: str = "train"
    validation_split: str = "validation"

    def __post_init__(self) -> None:
        if not self.dataset_path:
            raise ValueError("data.dataset_path must be non-empty")
        if self.train_split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported train split: {self.train_split}")
        if self.validation_split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported validation split: {self.validation_split}")


@dataclass(frozen=True)
class ProposerModelConfig:
    """Model hyperparameters for the first legality/policy proposer."""

    architecture: str = "mlp_v1"
    hidden_dim: int = 256
    hidden_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.architecture not in {"mlp_v1", "multistream_v2", "factorized_v3"}:
            raise ValueError(
                "model.architecture must be 'mlp_v1', 'multistream_v2', or 'factorized_v3'"
            )
        if self.hidden_dim <= 0:
            raise ValueError("model.hidden_dim must be positive")
        if self.hidden_layers <= 0:
            raise ValueError("model.hidden_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0.0, 1.0)")


@dataclass(frozen=True)
class ProposerOptimizationConfig:
    """Optimizer and loss weighting settings."""

    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    legality_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("optimization.epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("optimization.batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("optimization.learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("optimization.weight_decay must be non-negative")
        if self.legality_loss_weight <= 0.0:
            raise ValueError("optimization.legality_loss_weight must be positive")
        if self.policy_loss_weight <= 0.0:
            raise ValueError("optimization.policy_loss_weight must be positive")


@dataclass(frozen=True)
class ProposerEvaluationConfig:
    """Held-out evaluation settings."""

    legality_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.legality_threshold <= 1.0:
            raise ValueError("evaluation.legality_threshold must be in [0.0, 1.0]")


@dataclass(frozen=True)
class ProposerRuntimeConfig:
    """Runtime knobs for CPU-bound proposer training and evaluation."""

    torch_threads: int = 0
    dataloader_workers: int = 0

    def __post_init__(self) -> None:
        if self.torch_threads < 0:
            raise ValueError("runtime.torch_threads must be non-negative")
        if self.dataloader_workers < 0:
            raise ValueError("runtime.dataloader_workers must be non-negative")


@dataclass(frozen=True)
class ProposerExportConfig:
    """Export bundle paths for the trained proposer."""

    bundle_dir: str
    checkpoint_name: str = "checkpoint.pt"
    exported_program_name: str = "proposer.pt2"
    metadata_name: str = DEFAULT_PROPOSER_METADATA_NAME

    def __post_init__(self) -> None:
        if not self.bundle_dir:
            raise ValueError("export.bundle_dir must be non-empty")
        if not self.checkpoint_name:
            raise ValueError("export.checkpoint_name must be non-empty")
        if not self.exported_program_name:
            raise ValueError("export.exported_program_name must be non-empty")
        if not self.metadata_name:
            raise ValueError("export.metadata_name must be non-empty")
        if self.metadata_name != DEFAULT_PROPOSER_METADATA_NAME:
            raise ValueError(
                f"export.metadata_name must be {DEFAULT_PROPOSER_METADATA_NAME!r}"
            )


@dataclass(frozen=True)
class ProposerTrainConfig:
    """Full training configuration for the Phase-5 proposer."""

    seed: int
    output_dir: str
    data: ProposerDataConfig
    model: ProposerModelConfig
    optimization: ProposerOptimizationConfig
    evaluation: ProposerEvaluationConfig
    runtime: ProposerRuntimeConfig
    export: ProposerExportConfig

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProposerTrainConfig":
        """Parse the training config from a JSON object."""
        return cls(
            seed=int(payload.get("seed", 0)),
            output_dir=str(payload["output_dir"]),
            data=ProposerDataConfig(**_mapping(payload, "data")),
            model=ProposerModelConfig(**_mapping(payload, "model")),
            optimization=ProposerOptimizationConfig(**_mapping(payload, "optimization")),
            evaluation=ProposerEvaluationConfig(**_mapping(payload, "evaluation")),
            runtime=ProposerRuntimeConfig(**dict(payload.get("runtime", {}))),
            export=ProposerExportConfig(**_mapping(payload, "export")),
        )


def load_proposer_train_config(path: Path) -> ProposerTrainConfig:
    """Load a proposer training config from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: training config root must be an object")
    return ProposerTrainConfig.from_dict(payload)


def resolve_repo_path(repo_root: Path, configured_path: str) -> Path:
    """Resolve a repo-relative or absolute config path."""
    path = Path(configured_path)
    return path if path.is_absolute() else repo_root / path


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return dict(value)
