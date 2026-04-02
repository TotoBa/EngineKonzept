"""Backfill optional dynamics_<split>.jsonl files for an existing dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.datasets import materialize_dynamics_artifacts

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    dataset_dir = _resolve_repo_path(args.dataset_dir)
    written_counts = materialize_dynamics_artifacts(dataset_dir, repo_root=REPO_ROOT)
    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "written_counts": written_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
