"""Write a compact comparison artifact for planner suite runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-only", type=Path, required=True)
    parser.add_argument("--symbolic-reply", type=Path, required=True)
    parser.add_argument("--learned-reply", type=Path, required=True)
    parser.add_argument("--trained-planner", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    root_only = _load_aggregate(args.root_only)
    symbolic_reply = _load_aggregate(args.symbolic_reply)
    learned_reply = _load_aggregate(args.learned_reply)
    trained_planner = _load_metrics(args.trained_planner)
    comparison = {
        "runs": {
            "root_only": root_only,
            "symbolic_reply": symbolic_reply,
            "learned_reply": learned_reply,
            "trained_planner": trained_planner,
        }
    }
    output_path = _resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def _load_aggregate(path: Path) -> dict[str, float | int]:
    payload = json.loads(_resolve_repo_path(path).read_text(encoding="utf-8"))
    if "aggregate" in payload:
        return dict(payload["aggregate"])
    return dict(payload)


def _load_metrics(path: Path) -> dict[str, float | int]:
    return dict(json.loads(_resolve_repo_path(path).read_text(encoding="utf-8")))


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
