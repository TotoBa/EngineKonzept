"""Dataset schemas, artifact loaders, and reproducible build helpers."""

from train.datasets.artifacts import (
    DynamicsTrainingExample,
    POSITION_FEATURE_SIZE,
    ProposerTrainingExample,
    SYMBOLIC_MAX_LEGAL_CANDIDATES,
    SymbolicProposerTrainingExample,
    dynamics_artifact_name,
    dynamics_symbolic_action_feature_spec,
    load_dataset_examples,
    load_dynamics_examples,
    load_proposer_examples,
    load_split_examples,
    materialize_dynamics_artifacts,
    materialize_proposer_artifacts,
    materialize_symbolic_proposer_artifacts,
    pack_position_features,
    position_feature_spec,
    proposer_artifact_name,
    symbolic_proposer_artifact_name,
    symbolic_proposer_feature_spec,
)
from train.datasets.builder import BuiltDataset, build_dataset
from train.datasets.io import write_dataset_artifacts
from train.datasets.pgn_policy import (
    PgnPolicySamplingConfig,
    sample_policy_records_from_pgns,
    training_split_ratios,
    verification_split_ratios,
)
from train.datasets.schema import DatasetExample, RawPositionRecord, SplitRatios, WdlTarget
from train.datasets.sources import SUPPORTED_SOURCE_FORMATS, load_raw_records


def module_purpose() -> str:
    """Describe the current responsibility of the dataset package."""
    return "Dataset schemas, exact-rule labeling, artifact loading, and feature packing for proposer and dynamics"


__all__ = [
    "BuiltDataset",
    "DatasetExample",
    "DynamicsTrainingExample",
    "POSITION_FEATURE_SIZE",
    "PgnPolicySamplingConfig",
    "ProposerTrainingExample",
    "RawPositionRecord",
    "SYMBOLIC_MAX_LEGAL_CANDIDATES",
    "SUPPORTED_SOURCE_FORMATS",
    "SplitRatios",
    "SymbolicProposerTrainingExample",
    "WdlTarget",
    "build_dataset",
    "dynamics_artifact_name",
    "dynamics_symbolic_action_feature_spec",
    "load_dataset_examples",
    "load_dynamics_examples",
    "load_proposer_examples",
    "load_raw_records",
    "load_split_examples",
    "materialize_dynamics_artifacts",
    "materialize_proposer_artifacts",
    "materialize_symbolic_proposer_artifacts",
    "module_purpose",
    "pack_position_features",
    "position_feature_spec",
    "proposer_artifact_name",
    "symbolic_proposer_artifact_name",
    "symbolic_proposer_feature_spec",
    "sample_policy_records_from_pgns",
    "training_split_ratios",
    "verification_split_ratios",
    "write_dataset_artifacts",
]
