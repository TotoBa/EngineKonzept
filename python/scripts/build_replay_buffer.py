"""Build a replay-buffer artifact from one or more selfplay session JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.datasets.replay_buffer import (
    build_replay_buffer_entries,
    replay_buffer_summary,
    write_replay_buffer_artifact,
)
from train.eval.selfplay import SelfplaySessionRecord


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-path", type=Path, action="append", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args()

    sessions = [
        SelfplaySessionRecord.from_json(_resolve_repo_path(path).read_text(encoding="utf-8"))
        for path in args.session_path
    ]
    entries = [
        entry
        for session in sessions
        for entry in build_replay_buffer_entries(session)
    ]
    output_path = _resolve_repo_path(args.output_path)
    write_replay_buffer_artifact(output_path, entries)
    summary = replay_buffer_summary(entries)
    summary_path = (
        _resolve_repo_path(args.summary_path)
        if args.summary_path is not None
        else output_path.with_suffix(".summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "output_path": str(output_path), "summary_path": str(summary_path)}, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
