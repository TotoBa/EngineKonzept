"""Evaluate a trained Phase-7 opponent head on one artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.trainers.opponent import evaluate_opponent_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, action="append", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    metrics = evaluate_opponent_checkpoint(
        _resolve_repo_path(args.checkpoint),
        dataset_paths=[_resolve_repo_path(path) for path in args.dataset_path],
        top_k=args.top_k,
    )
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
