"""Build a full row-vs-column matrix artifact from a selfplay arena summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.matrix import (
    build_selfplay_arena_matrix,
    load_selfplay_arena_summary,
    write_selfplay_arena_matrix,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-summary", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    summary_path = _resolve_repo_path(args.arena_summary)
    output_path = _resolve_repo_path(args.output_path)
    matrix = build_selfplay_arena_matrix(load_selfplay_arena_summary(summary_path))
    write_selfplay_arena_matrix(output_path, matrix)
    print(json.dumps({"output_path": str(output_path), **matrix}, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
