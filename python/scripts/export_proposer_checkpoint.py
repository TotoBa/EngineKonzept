"""Export a proposer checkpoint into a Rust-loadable bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from train.config import ProposerTrainConfig, resolve_repo_path
from train.export.proposer import export_proposer_bundle
from train.models.proposer import LegalityPolicyProposer

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path)
    args = parser.parse_args(argv)

    payload = torch.load(args.checkpoint, map_location="cpu")
    config = ProposerTrainConfig.from_dict(dict(payload["training_config"]))
    bundle_dir = (
        args.bundle_dir
        if args.bundle_dir is not None
        else resolve_repo_path(REPO_ROOT, config.export.bundle_dir)
    )
    model = LegalityPolicyProposer(
        architecture=config.model.architecture,
        hidden_dim=config.model.hidden_dim,
        hidden_layers=config.model.hidden_layers,
        dropout=config.model.dropout,
    )
    model.load_state_dict(dict(payload["model_state_dict"]))
    export_paths = export_proposer_bundle(
        model,
        config=config,
        bundle_dir=bundle_dir,
        validation_metrics=dict(payload.get("validation_metrics", {})),
    )
    print(json.dumps(export_paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
