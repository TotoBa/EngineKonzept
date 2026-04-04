"""Merge one or more versioned selfplay initial-FEN suites into a deduped suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.eval.initial_fens import (
    load_selfplay_initial_fen_suite,
    merge_selfplay_initial_fen_suites,
    write_selfplay_initial_fen_suite,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="One or more suite JSON paths to merge in order.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.suite:
        raise ValueError("at least one --suite path is required")

    suite_paths = [_resolve_repo_path(Path(raw_path)) for raw_path in args.suite]
    suites = [load_selfplay_initial_fen_suite(path) for path in suite_paths]
    merged_suite = merge_selfplay_initial_fen_suites(
        name=args.name,
        suites=suites,
        metadata={"source_paths": [str(path) for path in suite_paths]},
    )
    output_path = _resolve_repo_path(args.output_path)
    write_selfplay_initial_fen_suite(output_path, merged_suite)

    payload = {
        "output_path": str(output_path),
        "entry_count": len(merged_suite.entries),
        "metadata": merged_suite.metadata,
    }
    if args.summary_path is not None:
        summary_path = _resolve_repo_path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["summary_path"] = str(summary_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
