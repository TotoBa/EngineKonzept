"""Dataset schemas and reproducible build helpers for Phase 4."""

from train.datasets.builder import BuiltDataset, build_dataset
from train.datasets.io import write_dataset_artifacts
from train.datasets.schema import DatasetExample, RawPositionRecord, SplitRatios, WdlTarget
from train.datasets.sources import SUPPORTED_SOURCE_FORMATS, load_raw_records


def module_purpose() -> str:
    """Describe the current responsibility of the dataset package."""
    return "Dataset schemas, exact-rule labeling, and reproducible build helpers"


__all__ = [
    "BuiltDataset",
    "DatasetExample",
    "RawPositionRecord",
    "SUPPORTED_SOURCE_FORMATS",
    "SplitRatios",
    "WdlTarget",
    "build_dataset",
    "load_raw_records",
    "module_purpose",
    "write_dataset_artifacts",
]
