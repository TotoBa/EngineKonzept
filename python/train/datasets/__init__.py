"""Dataset schemas, artifact loaders, and reproducible build helpers."""

from train.datasets.artifacts import (
    POSITION_FEATURE_SIZE,
    ProposerTrainingExample,
    load_dataset_examples,
    load_proposer_examples,
    load_split_examples,
    pack_position_features,
    position_feature_spec,
    proposer_artifact_name,
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
    return "Dataset schemas, exact-rule labeling, artifact loading, and feature packing"


__all__ = [
    "BuiltDataset",
    "DatasetExample",
    "POSITION_FEATURE_SIZE",
    "PgnPolicySamplingConfig",
    "ProposerTrainingExample",
    "RawPositionRecord",
    "SUPPORTED_SOURCE_FORMATS",
    "SplitRatios",
    "WdlTarget",
    "build_dataset",
    "load_dataset_examples",
    "load_proposer_examples",
    "load_raw_records",
    "load_split_examples",
    "module_purpose",
    "pack_position_features",
    "position_feature_spec",
    "proposer_artifact_name",
    "sample_policy_records_from_pgns",
    "training_split_ratios",
    "verification_split_ratios",
    "write_dataset_artifacts",
]
