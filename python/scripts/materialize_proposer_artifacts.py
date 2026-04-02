"""Backfill optional proposer_<split>.jsonl files for an existing dataset directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.datasets import materialize_proposer_artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    written_counts = materialize_proposer_artifacts(args.dataset_dir)
    print(
        json.dumps(
            {
                "dataset_dir": str(args.dataset_dir),
                "written_counts": written_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
