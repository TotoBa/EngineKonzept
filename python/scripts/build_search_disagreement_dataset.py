"""Build offline disagreement labels between symbolic proposer ranking and a search teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.datasets import load_search_teacher_examples, load_split_examples
from train.datasets.search_disagreements import (
    build_search_disagreement_examples,
    search_disagreements_artifact_name,
    write_search_disagreement_artifact,
)
from train.datasets.search_teacher import search_teacher_artifact_name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-artifact", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args(argv)

    teacher_artifact = (
        args.teacher_artifact
        if args.teacher_artifact is not None
        else args.dataset_dir / search_teacher_artifact_name(args.split)
    )
    output_path = (
        args.output_path
        if args.output_path is not None
        else args.dataset_dir / search_disagreements_artifact_name(args.split)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_examples = load_split_examples(args.dataset_dir, args.split)
    teacher_examples = load_search_teacher_examples(teacher_artifact)
    built = build_search_disagreement_examples(
        dataset_examples,
        teacher_examples,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        max_examples=args.max_examples,
    )
    write_search_disagreement_artifact(output_path, built)

    disagreement_count = sum(1 for example in built if example.top1_disagrees)
    summary = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "teacher_artifact": str(teacher_artifact),
        "checkpoint": str(args.checkpoint),
        "output_path": str(output_path),
        "example_count": len(built),
        "disagreement_count": disagreement_count,
        "disagreement_rate": (disagreement_count / len(built)) if built else 0.0,
        "average_teacher_rank_of_proposer_top1": (
            sum(example.teacher_rank_of_proposer_top1 for example in built) / len(built)
            if built
            else 0.0
        ),
        "average_proposer_rank_of_teacher_top1": (
            sum(example.proposer_rank_of_teacher_top1 for example in built) / len(built)
            if built
            else 0.0
        ),
        "average_teacher_top1_advantage_cp": (
            sum(example.teacher_top1_advantage_cp for example in built) / len(built)
            if built
            else 0.0
        ),
        "average_policy_l1_distance": (
            sum(example.policy_l1_distance for example in built) / len(built)
            if built
            else 0.0
        ),
        "top_k": args.top_k,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
