"""Build replay-derived planner supervision from a selfplay replay buffer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.datasets import (
    build_planner_replay_examples,
    load_replay_buffer_entries,
    planner_replay_artifact_name,
    planner_replay_summary,
    write_planner_replay_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    replay_path = _resolve_repo_path(args.replay_path)
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    examples = build_planner_replay_examples(
        load_replay_buffer_entries(replay_path),
        split=args.split,
        include_unfinished=args.include_unfinished,
        max_examples=args.max_examples,
    )
    artifact_path = output_root / planner_replay_artifact_name(args.split)
    write_planner_replay_artifact(artifact_path, examples)
    summary = {
        "replay_path": str(replay_path),
        "artifact_path": str(artifact_path),
        "include_unfinished": bool(args.include_unfinished),
        "split": args.split,
        "summary": planner_replay_summary(examples),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
