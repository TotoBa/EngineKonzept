"""Build PlannerHeadV1 artifacts from exact workflow slices and bounded reply signals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.datasets import (
    build_planner_head_examples,
    planner_head_artifact_name,
    write_planner_head_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--search-teacher-path", type=Path, required=True)
    parser.add_argument("--search-curriculum-path", type=Path)
    parser.add_argument("--proposer-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent-mode",
        choices=("none", "symbolic", "learned"),
        default="learned",
    )
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    dataset_dir = _resolve_repo_path(args.dataset_dir)
    teacher_path = _resolve_repo_path(args.search_teacher_path)
    curriculum_path = (
        _resolve_repo_path(args.search_curriculum_path)
        if args.search_curriculum_path is not None
        else None
    )
    proposer_checkpoint = _resolve_repo_path(args.proposer_checkpoint)
    opponent_checkpoint = (
        _resolve_repo_path(args.opponent_checkpoint)
        if args.opponent_checkpoint is not None
        else None
    )
    output_path = (
        _resolve_repo_path(args.output_path)
        if args.output_path is not None
        else dataset_dir / planner_head_artifact_name(args.split)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = build_planner_head_examples(
        dataset_dir=dataset_dir,
        split=args.split,
        search_teacher_path=teacher_path,
        search_curriculum_path=curriculum_path,
        proposer_checkpoint=proposer_checkpoint,
        opponent_mode=args.opponent_mode,
        opponent_checkpoint=opponent_checkpoint,
        root_top_k=args.root_top_k,
        max_examples=args.max_examples,
        repo_root=REPO_ROOT,
    )
    write_planner_head_artifact(output_path, examples)
    summary = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "search_teacher_path": str(teacher_path),
        "search_curriculum_path": str(curriculum_path) if curriculum_path is not None else None,
        "proposer_checkpoint": str(proposer_checkpoint),
        "opponent_mode": args.opponent_mode,
        "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint is not None else None,
        "root_top_k": args.root_top_k,
        "max_examples": args.max_examples,
        "output_path": str(output_path),
        "example_count": len(examples),
        "mean_curriculum_priority": round(
            sum(example.curriculum_priority for example in examples) / len(examples),
            6,
        )
        if examples
        else 0.0,
    }
    summary_path = output_path.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
