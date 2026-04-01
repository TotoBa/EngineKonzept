"""Build a labeled dataset from raw positions using the exact Rust rules oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.datasets import SUPPORTED_SOURCE_FORMATS, build_dataset, load_raw_records
from train.datasets.io import write_dataset_artifacts
from train.datasets.schema import SplitRatios


def main(argv: Sequence[str] | None = None) -> int:
    """Build a reproducible dataset and write JSONL artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-format", choices=SUPPORTED_SOURCE_FORMATS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-name")
    parser.add_argument("--seed", default="phase-4")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args(argv)

    records = load_raw_records(
        args.input,
        args.source_format,
        source_name=args.source_name,
    )
    dataset = build_dataset(
        records,
        ratios=SplitRatios(
            train=args.train_ratio,
            validation=args.validation_ratio,
            test=args.test_ratio,
        ),
        seed=args.seed,
        repo_root=_repo_root(),
    )
    write_dataset_artifacts(args.output_dir, dataset)

    print(json.dumps(dataset.summary, indent=2, sort_keys=True))
    print(f"Wrote dataset artifacts to {args.output_dir}")
    return 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
