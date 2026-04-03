"""Train the first explicit Phase-7 opponent head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.config import load_opponent_train_config
from train.trainers.opponent import train_opponent


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    run = train_opponent(load_opponent_train_config(config_path), repo_root=REPO_ROOT)
    print(json.dumps(run.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
