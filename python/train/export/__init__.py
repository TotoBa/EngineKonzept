"""Model export helpers for proposer and dynamics bundles."""

from train.export.dynamics import build_dynamics_export_metadata, export_dynamics_bundle
from train.export.proposer import build_export_metadata, export_proposer_bundle


def module_purpose() -> str:
    """Describe the current responsibility of the export package."""
    return "torch.export proposer/dynamics bundles and Rust-facing metadata"

__all__ = [
    "build_dynamics_export_metadata",
    "build_export_metadata",
    "export_dynamics_bundle",
    "export_proposer_bundle",
    "module_purpose",
]
