"""Configuration schemas for proposer and dynamics training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from train.datasets.schema import SUPPORTED_SPLITS

DEFAULT_PROPOSER_METADATA_NAME = "metadata.json"
DEFAULT_DYNAMICS_METADATA_NAME = "metadata.json"


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
        if self.architecture not in {
            "mlp_v1",
            "multistream_v2",
            "factorized_v3",
            "factorized_v4",
            "factorized_v5",
            "factorized_v6",
            "relational_v1",
            "symbolic_v1",
        }:
            raise ValueError(
                "model.architecture must be 'mlp_v1', 'multistream_v2', 'factorized_v3', 'factorized_v4', 'factorized_v5', 'factorized_v6', 'relational_v1', or 'symbolic_v1'"
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
    checkpoint_selection: str = "legality_first"
    selection_policy_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.legality_threshold <= 1.0:
            raise ValueError("evaluation.legality_threshold must be in [0.0, 1.0]")
        if self.checkpoint_selection not in {"legality_first", "policy_first", "balanced"}:
            raise ValueError(
                "evaluation.checkpoint_selection must be 'legality_first', 'policy_first', or 'balanced'"
            )
        if self.selection_policy_weight <= 0.0:
            raise ValueError("evaluation.selection_policy_weight must be positive")


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
    enabled: bool = True
    checkpoint_name: str = "checkpoint.pt"
    exported_program_name: str = "proposer.pt2"
    runtime_weights_name: str = "symbolic_runtime.bin"
    metadata_name: str = DEFAULT_PROPOSER_METADATA_NAME

    def __post_init__(self) -> None:
        if not self.bundle_dir:
            raise ValueError("export.bundle_dir must be non-empty")
        if not self.checkpoint_name:
            raise ValueError("export.checkpoint_name must be non-empty")
        if not self.exported_program_name:
            raise ValueError("export.exported_program_name must be non-empty")
        if not self.runtime_weights_name:
            raise ValueError("export.runtime_weights_name must be non-empty")
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


@dataclass(frozen=True)
class DynamicsDataConfig:
    """Dataset and split selection for dynamics training."""

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
class DynamicsModelConfig:
    """Model hyperparameters for the first latent dynamics model."""

    architecture: str = "mlp_v1"
    latent_dim: int = 128
    hidden_dim: int = 256
    hidden_layers: int = 2
    action_embedding_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.architecture not in {
            "mlp_v1",
            "structured_v2",
            "structured_v3",
            "structured_v4",
            "structured_v5",
            "edit_v1",
        }:
            raise ValueError(
                "model.architecture must be 'mlp_v1', 'structured_v2', 'structured_v3', 'structured_v4', 'structured_v5', or 'edit_v1'"
            )
        if self.latent_dim <= 0:
            raise ValueError("model.latent_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("model.hidden_dim must be positive")
        if self.hidden_layers <= 0:
            raise ValueError("model.hidden_layers must be positive")
        if self.action_embedding_dim <= 0:
            raise ValueError("model.action_embedding_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0.0, 1.0)")


@dataclass(frozen=True)
class DynamicsOptimizationConfig:
    """Optimizer and reconstruction loss settings."""

    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    reconstruction_loss_weight: float = 1.0
    piece_loss_weight: float = 1.0
    square_loss_weight: float = 1.0
    rule_loss_weight: float = 1.0
    delta_loss_weight: float = 0.0
    latent_consistency_loss_weight: float = 0.0
    drift_supervision_loss_weight: float = 0.0
    drift_supervision_horizon: int = 2
    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("optimization.epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("optimization.batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("optimization.learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("optimization.weight_decay must be non-negative")
        if self.reconstruction_loss_weight <= 0.0:
            raise ValueError("optimization.reconstruction_loss_weight must be positive")
        if self.piece_loss_weight <= 0.0:
            raise ValueError("optimization.piece_loss_weight must be positive")
        if self.square_loss_weight <= 0.0:
            raise ValueError("optimization.square_loss_weight must be positive")
        if self.rule_loss_weight <= 0.0:
            raise ValueError("optimization.rule_loss_weight must be positive")
        if self.delta_loss_weight < 0.0:
            raise ValueError("optimization.delta_loss_weight must be non-negative")
        if self.latent_consistency_loss_weight < 0.0:
            raise ValueError("optimization.latent_consistency_loss_weight must be non-negative")
        if self.drift_supervision_loss_weight < 0.0:
            raise ValueError("optimization.drift_supervision_loss_weight must be non-negative")
        if self.drift_supervision_horizon < 2:
            raise ValueError("optimization.drift_supervision_horizon must be at least 2")


@dataclass(frozen=True)
class DynamicsEvaluationConfig:
    """Held-out evaluation settings for dynamics training."""

    drift_horizon: int = 2
    drift_dataset_path: str | None = None
    drift_split: str = "test"

    def __post_init__(self) -> None:
        if self.drift_horizon < 2:
            raise ValueError("evaluation.drift_horizon must be at least 2")
        if self.drift_split not in SUPPORTED_SPLITS:
            raise ValueError(f"unsupported drift split: {self.drift_split}")


@dataclass(frozen=True)
class DynamicsRuntimeConfig:
    """Runtime knobs for CPU-bound dynamics training and evaluation."""

    torch_threads: int = 0
    dataloader_workers: int = 0

    def __post_init__(self) -> None:
        if self.torch_threads < 0:
            raise ValueError("runtime.torch_threads must be non-negative")
        if self.dataloader_workers < 0:
            raise ValueError("runtime.dataloader_workers must be non-negative")


@dataclass(frozen=True)
class DynamicsExportConfig:
    """Export bundle paths for the trained dynamics model."""

    bundle_dir: str
    checkpoint_name: str = "checkpoint.pt"
    exported_program_name: str = "dynamics.pt2"
    metadata_name: str = DEFAULT_DYNAMICS_METADATA_NAME

    def __post_init__(self) -> None:
        if not self.bundle_dir:
            raise ValueError("export.bundle_dir must be non-empty")
        if not self.checkpoint_name:
            raise ValueError("export.checkpoint_name must be non-empty")
        if not self.exported_program_name:
            raise ValueError("export.exported_program_name must be non-empty")
        if self.metadata_name != DEFAULT_DYNAMICS_METADATA_NAME:
            raise ValueError(f"export.metadata_name must be {DEFAULT_DYNAMICS_METADATA_NAME!r}")


@dataclass(frozen=True)
class DynamicsTrainConfig:
    """Full training configuration for the Phase-6 dynamics model."""

    seed: int
    output_dir: str
    data: DynamicsDataConfig
    model: DynamicsModelConfig
    optimization: DynamicsOptimizationConfig
    evaluation: DynamicsEvaluationConfig
    runtime: DynamicsRuntimeConfig
    export: DynamicsExportConfig

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not self.output_dir:
            raise ValueError("output_dir must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DynamicsTrainConfig":
        """Parse the training config from a JSON object."""
        return cls(
            seed=int(payload.get("seed", 0)),
            output_dir=str(payload["output_dir"]),
            data=DynamicsDataConfig(**_mapping(payload, "data")),
            model=DynamicsModelConfig(**_mapping(payload, "model")),
            optimization=DynamicsOptimizationConfig(**_mapping(payload, "optimization")),
            evaluation=DynamicsEvaluationConfig(**_mapping(payload, "evaluation")),
            runtime=DynamicsRuntimeConfig(**dict(payload.get("runtime", {}))),
            export=DynamicsExportConfig(**_mapping(payload, "export")),
        )


def load_proposer_train_config(path: Path) -> ProposerTrainConfig:
    """Load a proposer training config from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: training config root must be an object")
    return ProposerTrainConfig.from_dict(payload)


def load_dynamics_train_config(path: Path) -> DynamicsTrainConfig:
    """Load a dynamics training config from JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: training config root must be an object")
    return DynamicsTrainConfig.from_dict(payload)


def resolve_repo_path(repo_root: Path, configured_path: str) -> Path:
    """Resolve a repo-relative or absolute config path."""
    path = Path(configured_path)
    return path if path.is_absolute() else repo_root / path


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object")
    return dict(value)
