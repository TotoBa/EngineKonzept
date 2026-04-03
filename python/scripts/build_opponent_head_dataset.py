"""Build the first OpponentHeadV1 dataset from traces, curriculum, and exact successor states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.datasets import (
    load_search_curriculum_examples,
    load_search_trace_examples,
    load_split_examples,
)
from train.datasets.opponent_head import (
    build_opponent_head_examples,
    opponent_head_artifact_name,
    write_opponent_head_artifact,
)
from train.datasets.search_curriculum import search_curriculum_artifact_name
from train.datasets.search_traces import search_traces_artifact_name


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--trace-artifact", type=Path)
    parser.add_argument("--curriculum-artifact", type=Path)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args(argv)

    dataset_dir = _resolve_repo_path(args.dataset_dir)
    trace_artifact = (
        _resolve_repo_path(args.trace_artifact)
        if args.trace_artifact is not None
        else dataset_dir / search_traces_artifact_name(args.split)
    )
    curriculum_artifact = (
        _resolve_repo_path(args.curriculum_artifact)
        if args.curriculum_artifact is not None
        else dataset_dir / search_curriculum_artifact_name(args.split)
    )
    output_path = (
        _resolve_repo_path(args.output_path)
        if args.output_path is not None
        else dataset_dir / opponent_head_artifact_name(args.split)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_examples = load_split_examples(dataset_dir, args.split)
    trace_examples = load_search_trace_examples(trace_artifact)
    curriculum_examples = load_search_curriculum_examples(curriculum_artifact)
    built = build_opponent_head_examples(
        dataset_examples,
        trace_examples,
        curriculum_examples,
        repo_root=REPO_ROOT,
    )
    write_opponent_head_artifact(output_path, built)

    summary = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "trace_artifact": str(trace_artifact),
        "curriculum_artifact": str(curriculum_artifact),
        "output_path": str(output_path),
        "example_count": len(built),
        "reply_supervised_count": sum(
            1 for example in built if example.teacher_reply_action_index is not None
        ),
        "average_pressure_target": (
            sum(example.pressure_target for example in built) / len(built)
            if built
            else 0.0
        ),
        "average_uncertainty_target": (
            sum(example.uncertainty_target for example in built) / len(built)
            if built
            else 0.0
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
