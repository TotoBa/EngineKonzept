"""Rebuild one arena summary.json from completed session files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    from train.eval.arena import (
        load_selfplay_arena_spec,
        rebuild_selfplay_arena_summary,
    )

    spec = load_selfplay_arena_spec(_resolve_repo_path(args.arena_spec))
    summary = rebuild_selfplay_arena_summary(
        spec=spec,
        output_root=_resolve_repo_path(args.output_root),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
