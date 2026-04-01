"""Train the Phase-5 legality/policy proposer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.config import load_proposer_train_config
from train.trainers import train_proposer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "python" / "configs" / "phase5_proposer_v1.json"


def main() -> int:
    """Load the configured dataset, train the proposer, and export the bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the proposer training config JSON",
    )
    args = parser.parse_args()

    config = load_proposer_train_config(args.config)
    result = train_proposer(config, repo_root=REPO_ROOT)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
