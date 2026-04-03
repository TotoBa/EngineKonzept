"""Build planner-head replay fine-tuning artifacts from replay-derived supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.datasets import (
    build_planner_head_examples_from_replay,
    planner_head_artifact_name,
    write_planner_head_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-replay-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--proposer-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path)
    parser.add_argument("--opponent-mode", type=str, default="none")
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    examples = build_planner_head_examples_from_replay(
        planner_replay_path=_resolve_repo_path(args.planner_replay_path),
        proposer_checkpoint=_resolve_repo_path(args.proposer_checkpoint),
        dynamics_checkpoint=(
            _resolve_repo_path(args.dynamics_checkpoint)
            if args.dynamics_checkpoint is not None
            else None
        ),
        opponent_mode=args.opponent_mode,
        opponent_checkpoint=(
            _resolve_repo_path(args.opponent_checkpoint)
            if args.opponent_checkpoint is not None
            else None
        ),
        root_top_k=args.root_top_k,
        max_examples=args.max_examples,
        repo_root=REPO_ROOT,
    )
    artifact_path = output_root / planner_head_artifact_name("train")
    write_planner_head_artifact(artifact_path, examples)
    summary = {
        "planner_replay_path": str(_resolve_repo_path(args.planner_replay_path)),
        "artifact_path": str(artifact_path),
        "example_count": len(examples),
        "root_top_k": args.root_top_k,
        "opponent_mode": args.opponent_mode,
        "mean_candidate_count": round(
            sum(len(example.candidate_action_indices) for example in examples) / len(examples),
            6,
        )
        if examples
        else 0.0,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
