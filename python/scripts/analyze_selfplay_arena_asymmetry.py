"""Analyze color asymmetries in one completed selfplay arena summary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from train.eval.arena_asymmetry import (  # noqa: E402
    build_selfplay_arena_asymmetry_report,
    write_selfplay_arena_asymmetry_report,
)
from train.eval.matrix import load_selfplay_arena_summary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-summary", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    summary = load_selfplay_arena_summary(_resolve_repo_path(args.arena_summary))
    report = build_selfplay_arena_asymmetry_report(summary)
    write_selfplay_arena_asymmetry_report(_resolve_repo_path(args.output_path), report)
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
