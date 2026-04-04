"""Build a versioned selfplay initial-FEN suite from TSV opening books."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.eval.initial_fens import (
    build_opening_initial_fen_suite,
    write_selfplay_initial_fen_suite,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--openings-dir", type=Path, required=True)
    parser.add_argument(
        "--entries-per-file",
        type=int,
        default=2,
        help="How many opening positions to keep from each TSV file.",
    )
    parser.add_argument(
        "--min-ply-count",
        type=int,
        default=8,
        help="Discard short openings below this ply count.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    openings_dir = _resolve_repo_path(args.openings_dir)
    tsv_paths = sorted(openings_dir.glob("*.tsv"))
    if not tsv_paths:
        raise ValueError(f"no TSV files found under {openings_dir}")

    suite = build_opening_initial_fen_suite(
        name=args.name,
        tsv_paths=tsv_paths,
        entries_per_file=args.entries_per_file,
        min_ply_count=args.min_ply_count,
    )
    output_path = _resolve_repo_path(args.output_path)
    write_selfplay_initial_fen_suite(output_path, suite)

    payload = {
        "output_path": str(output_path),
        "entry_count": len(suite.entries),
        "metadata": suite.metadata,
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
