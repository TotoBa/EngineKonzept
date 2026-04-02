"""Build offline search traces over exact legal candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.datasets import load_split_examples
from train.datasets.search_traces import (
    build_search_trace_examples,
    search_traces_artifact_name,
    write_search_trace_artifact,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--teacher-engine", type=Path, required=True)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--movetime-ms", type=int)
    parser.add_argument(
        "--multipv",
        type=int,
        default=8,
        help="0 means score the full exact legal candidate set",
    )
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    args = parser.parse_args(argv)

    output_path = (
        args.output_path
        if args.output_path is not None
        else args.dataset_dir / search_traces_artifact_name(args.split)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    examples = load_split_examples(args.dataset_dir, args.split)
    built = build_search_trace_examples(
        examples,
        teacher_engine_path=args.teacher_engine,
        nodes=args.nodes,
        depth=args.depth,
        movetime_ms=args.movetime_ms,
        multipv=args.multipv,
        policy_temperature_cp=args.policy_temperature_cp,
        max_examples=args.max_examples,
    )
    write_search_trace_artifact(output_path, built)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "output_path": str(output_path),
        "teacher_engine": str(args.teacher_engine),
        "example_count": len(built),
        "average_pv_length": (
            sum(example.pv_length for example in built) / len(built)
            if built
            else 0.0
        ),
        "average_coverage_ratio": (
            sum(example.teacher_coverage_ratio for example in built) / len(built)
            if built
            else 0.0
        ),
        "nodes": args.nodes,
        "depth": args.depth,
        "movetime_ms": args.movetime_ms,
        "multipv": args.multipv,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
