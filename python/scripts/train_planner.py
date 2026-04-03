"""Train the first bounded planner head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.config import load_planner_train_config
from train.trainers import train_planner


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Train the first exact-candidate bounded root planner arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config = load_planner_train_config(_resolve_repo_path(args.config))
    run = train_planner(config, repo_root=REPO_ROOT)
    print(json.dumps(run.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
