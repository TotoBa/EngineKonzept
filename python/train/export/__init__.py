"""Model export helpers for the Phase-5 proposer."""

from train.export.proposer import build_export_metadata, export_proposer_bundle


def module_purpose() -> str:
    """Describe the current responsibility of the export package."""
    return "torch.export proposer bundles and Rust-facing metadata"


__all__ = ["build_export_metadata", "export_proposer_bundle", "module_purpose"]
