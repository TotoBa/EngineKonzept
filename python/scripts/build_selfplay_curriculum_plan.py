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
    args = parser.parse_args()

    plan = build_phase9_expanded_curriculum_plan(
        repo_root=REPO_ROOT,
        source_arena_summary_path=_resolve_repo_path(args.source_arena_summary),
        corpus_suite_manifest_path=_resolve_repo_path(args.corpus_suite_manifest),
    )
    output_path = _resolve_repo_path(args.output_path)
    write_selfplay_curriculum_plan(output_path, plan)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
