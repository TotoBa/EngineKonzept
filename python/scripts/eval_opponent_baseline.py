"""Evaluate the Phase-7 exact symbolic reply-scorer baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval import evaluate_symbolic_opponent_baseline


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, action="append", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    metrics = evaluate_symbolic_opponent_baseline(
        _resolve_repo_path(args.checkpoint),
        dataset_paths=[_resolve_repo_path(path) for path in args.dataset_path],
        split=args.split,
    )
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
