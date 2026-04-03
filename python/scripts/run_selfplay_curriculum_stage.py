"""Run one versioned selfplay curriculum stage by materializing its arena spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.arena import run_selfplay_arena, write_selfplay_arena_spec
from train.eval.curriculum import (
    build_curriculum_stage_arena_spec,
    load_selfplay_curriculum_plan,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-plan", type=Path, required=True)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    curriculum_plan = load_selfplay_curriculum_plan(_resolve_repo_path(args.curriculum_plan))
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    resolved_spec = build_curriculum_stage_arena_spec(
        repo_root=REPO_ROOT,
        plan=curriculum_plan,
        stage_name=args.stage,
    )
    resolved_spec_path = output_root / "arena_spec.resolved.json"
    write_selfplay_arena_spec(resolved_spec_path, resolved_spec)

    summary = run_selfplay_arena(
        spec=resolved_spec,
        repo_root=REPO_ROOT,
        output_root=output_root,
    )
    payload = {
        "curriculum_plan": str(_resolve_repo_path(args.curriculum_plan)),
        "stage_name": args.stage,
        "resolved_arena_spec": str(resolved_spec_path),
        **summary,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **payload}, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
