"""Build the versioned Phase-9 curriculum plan for large-corpus reruns and selfplay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.curriculum import (
    build_phase9_expanded_curriculum_plan,
    write_selfplay_curriculum_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-arena-summary",
        type=Path,
        default=Path("artifacts/phase9/arena_active_probe_v1/summary.json"),
    )
    parser.add_argument(
        "--corpus-suite-manifest",
        type=Path,
        default=Path("artifacts/datasets/phase5_current_corpus_suite_v1.json"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/phase9/curriculum_active_experimental_expanded_v1.json"),
    )
    parser.add_argument(
        "--expanded-initial-fen-suite",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plan-name",
        type=str,
        default="phase9_active_experimental_expanded_v1",
    )
    parser.add_argument(
        "--probe-replay-buffer-output-root",
        type=str,
        default="artifacts/phase9/replay_buffer_active_expanded_probe_v1",
    )
    parser.add_argument(
        "--expanded-replay-buffer-output-root",
        type=str,
        default="artifacts/phase9/replay_buffer_active_experimental_expanded_v1",
    )
    parser.add_argument(
        "--expanded-games-per-matchup",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--expanded-max-plies",
        type=int,
        default=64,
    )
    args = parser.parse_args()

    plan = build_phase9_expanded_curriculum_plan(
        repo_root=REPO_ROOT,
        source_arena_summary_path=_resolve_repo_path(args.source_arena_summary),
        corpus_suite_manifest_path=_resolve_repo_path(args.corpus_suite_manifest),
        plan_name=args.plan_name,
        probe_replay_buffer_output_root=args.probe_replay_buffer_output_root,
        expanded_replay_buffer_output_root=args.expanded_replay_buffer_output_root,
        expanded_initial_fen_suite=args.expanded_initial_fen_suite,
        expanded_games_per_matchup=args.expanded_games_per_matchup,
        expanded_max_plies=args.expanded_max_plies,
    )
    output_path = _resolve_repo_path(args.output_path)
    write_selfplay_curriculum_plan(output_path, plan)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
