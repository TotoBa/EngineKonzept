"""Run one batched Phase-9 selfplay teacher-retrain cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train.eval.selfplay_training_cycle import (
    load_selfplay_teacher_retrain_cycle_spec,
    run_selfplay_teacher_retrain_cycle,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    spec = load_selfplay_teacher_retrain_cycle_spec(_resolve_repo_path(args.config))
    summary = run_selfplay_teacher_retrain_cycle(spec=spec, repo_root=REPO_ROOT)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
