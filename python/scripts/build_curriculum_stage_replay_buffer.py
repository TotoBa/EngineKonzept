"""Build one replay buffer directly from a versioned curriculum stage arena output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.datasets.replay_buffer import (
    build_replay_buffer_entries_from_sessions,
    load_arena_session_paths,
    replay_buffer_summary,
    write_replay_buffer_artifact,
)
from train.eval.curriculum import (
    load_selfplay_curriculum_plan,
    resolve_curriculum_stage,
)
from train.eval.selfplay import SelfplaySessionRecord


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-plan", type=Path, required=True)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--arena-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    curriculum_plan_path = _resolve_repo_path(args.curriculum_plan)
    arena_summary_path = _resolve_repo_path(args.arena_summary)
    curriculum_plan = load_selfplay_curriculum_plan(curriculum_plan_path)
    stage = resolve_curriculum_stage(curriculum_plan, stage_name=args.stage)
    output_root = (
        _resolve_repo_path(args.output_root)
        if args.output_root is not None
        else _resolve_repo_path(Path(stage.replay_buffer_output_root))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    session_paths = load_arena_session_paths(arena_summary_path)
    sessions = [
        SelfplaySessionRecord.from_json(path.read_text(encoding="utf-8"))
        for path in session_paths
    ]
    entries = build_replay_buffer_entries_from_sessions(
        sessions,
        session_labels=[path.stem for path in session_paths],
    )

    replay_path = output_root / "replay_buffer.jsonl"
    write_replay_buffer_artifact(replay_path, entries)
    summary = replay_buffer_summary(entries)
    payload = {
        "curriculum_plan": str(curriculum_plan_path),
        "stage_name": stage.name,
        "arena_summary": str(arena_summary_path),
        "session_count": len(session_paths),
        "replay_path": str(replay_path),
        "summary": summary,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **payload}, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
