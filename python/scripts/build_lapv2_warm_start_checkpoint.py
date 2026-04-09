"""Materialize one LAPv2 init checkpoint from an existing LAPv1 T2 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    from train.trainers import (
        build_lapv2_warm_start_checkpoint,
        load_lapv1_train_config,
    )

    target_config = load_lapv1_train_config(_resolve_repo_path(args.target_config))
    result = build_lapv2_warm_start_checkpoint(
        _resolve_repo_path(args.source_checkpoint),
        target_config=target_config,
        output_checkpoint=_resolve_repo_path(args.output_checkpoint),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
