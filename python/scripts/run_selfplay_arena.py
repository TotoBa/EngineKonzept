"""Run a checkpoint-vs-checkpoint selfplay arena from a versioned suite spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.arena import load_selfplay_arena_spec, run_selfplay_arena


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    spec = load_selfplay_arena_spec(_resolve_repo_path(args.arena_spec))
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = run_selfplay_arena(
        spec=spec,
        repo_root=REPO_ROOT,
        output_root=output_root,
    )
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
