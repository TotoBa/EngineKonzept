"""Evaluate the first bounded opponent-aware planner baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.planner import evaluate_two_ply_planner_baseline


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposer-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--search-teacher-path", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--opponent-mode", choices=("none", "symbolic", "learned"), required=True)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--reply-peak-weight", type=float, default=0.5)
    parser.add_argument("--pressure-weight", type=float, default=0.25)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    args = parser.parse_args()

    metrics = evaluate_two_ply_planner_baseline(
        _resolve_repo_path(args.proposer_checkpoint),
        dataset_dir=_resolve_repo_path(args.dataset_dir),
        search_teacher_path=_resolve_repo_path(args.search_teacher_path),
        split=args.split,
        opponent_mode=args.opponent_mode,
        opponent_checkpoint=(
            None
            if args.opponent_checkpoint is None
            else _resolve_repo_path(args.opponent_checkpoint)
        ),
        root_top_k=args.root_top_k,
        reply_peak_weight=args.reply_peak_weight,
        pressure_weight=args.pressure_weight,
        uncertainty_weight=args.uncertainty_weight,
        repo_root=REPO_ROOT,
    )
    print(json.dumps(metrics.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
