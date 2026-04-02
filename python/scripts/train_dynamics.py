"""Train the Phase-6 action-conditioned latent dynamics model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.config import load_dynamics_train_config
from train.trainers import train_dynamics

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "python" / "configs" / "phase6_dynamics_v1.json"


def main() -> int:
    """Load the configured dataset, train the dynamics model, and export the bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the dynamics training config JSON",
    )
    args = parser.parse_args()

    config = load_dynamics_train_config(args.config)
    result = train_dynamics(config, repo_root=REPO_ROOT)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
