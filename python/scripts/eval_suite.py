"""Evaluate proposer checkpoints on dataset artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.trainers import evaluate_proposer_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Run a proposer evaluation on one dataset split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--legality-threshold", type=float, default=0.5)
    args = parser.parse_args()

    metrics = evaluate_proposer_checkpoint(
        _resolve_repo_path(args.checkpoint),
        dataset_path=_resolve_repo_path(args.dataset_path),
        split=args.split,
        legality_threshold=args.legality_threshold,
    )
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
