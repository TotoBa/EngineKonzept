"""Build offline curriculum buckets from search traces and disagreement artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.datasets import (
    load_search_disagreement_examples,
    load_search_trace_examples,
)
from train.datasets.search_curriculum import (
    build_search_curriculum_examples,
    search_curriculum_artifact_name,
    write_search_curriculum_artifact,
)
from train.datasets.search_disagreements import search_disagreements_artifact_name
from train.datasets.search_traces import search_traces_artifact_name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--trace-artifact", type=Path)
    parser.add_argument("--disagreement-artifact", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--forced-gap-cp", type=float, default=80.0)
    parser.add_argument("--unstable-gap-cp", type=float, default=20.0)
    parser.add_argument("--large-rank-threshold", type=int, default=4)
    parser.add_argument("--disagreement-advantage-cp", type=float, default=80.0)
    parser.add_argument("--high-policy-l1", type=float, default=0.75)
    args = parser.parse_args(argv)

    trace_artifact = (
        args.trace_artifact
        if args.trace_artifact is not None
        else args.dataset_dir / search_traces_artifact_name(args.split)
    )
    disagreement_artifact = (
        args.disagreement_artifact
        if args.disagreement_artifact is not None
        else args.dataset_dir / search_disagreements_artifact_name(args.split)
    )
    output_path = (
        args.output_path
        if args.output_path is not None
        else args.dataset_dir / search_curriculum_artifact_name(args.split)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trace_examples = load_search_trace_examples(trace_artifact)
    disagreement_examples = load_search_disagreement_examples(disagreement_artifact)
    built = build_search_curriculum_examples(
        trace_examples,
        disagreement_examples,
        forced_gap_cp=args.forced_gap_cp,
        unstable_gap_cp=args.unstable_gap_cp,
        large_rank_threshold=args.large_rank_threshold,
        disagreement_advantage_cp=args.disagreement_advantage_cp,
        high_policy_l1=args.high_policy_l1,
    )
    write_search_curriculum_artifact(output_path, built)

    bucket_counts: dict[str, int] = {}
    for example in built:
        for label in example.bucket_labels:
            bucket_counts[label] = bucket_counts.get(label, 0) + 1

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "trace_artifact": str(trace_artifact),
        "disagreement_artifact": str(disagreement_artifact),
        "output_path": str(output_path),
        "example_count": len(built),
        "bucket_counts": bucket_counts,
        "average_curriculum_priority": (
            sum(example.curriculum_priority for example in built) / len(built)
            if built
            else 0.0
        ),
        "forced_gap_cp": args.forced_gap_cp,
        "unstable_gap_cp": args.unstable_gap_cp,
        "large_rank_threshold": args.large_rank_threshold,
        "disagreement_advantage_cp": args.disagreement_advantage_cp,
        "high_policy_l1": args.high_policy_l1,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
