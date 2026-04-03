"""Write a compact comparison artifact for planner suite runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-only", type=Path, required=True)
    parser.add_argument("--symbolic-reply", type=Path, required=True)
    parser.add_argument("--learned-reply", type=Path, required=True)
    parser.add_argument("--trained-planner", type=Path, required=True)
    parser.add_argument("--trained-planner-name", default="trained_planner")
    parser.add_argument("--reference-planner", type=Path)
    parser.add_argument("--reference-planner-name", default="reference_planner")
    parser.add_argument(
        "--additional-planner",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Add one or more extra trained-planner metric files to the comparison.",
    )
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    root_only = _load_aggregate(args.root_only)
    symbolic_reply = _load_aggregate(args.symbolic_reply)
    learned_reply = _load_aggregate(args.learned_reply)
    trained_planner = _load_metrics(args.trained_planner)
    runs = {
        "root_only": root_only,
        "symbolic_reply": symbolic_reply,
        "learned_reply": learned_reply,
        str(args.trained_planner_name): trained_planner,
    }
    if args.reference_planner is not None:
        runs[str(args.reference_planner_name)] = _load_metrics(args.reference_planner)
    for raw_spec in args.additional_planner:
        name, path = _parse_named_path(raw_spec)
        if name in runs:
            raise ValueError(f"duplicate planner run name: {name}")
        runs[name] = _load_metrics(Path(path))
    comparison = {"runs": runs}
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


def _parse_named_path(raw_spec: str) -> tuple[str, str]:
    if "=" not in raw_spec:
        raise ValueError("additional planner specs must use NAME=PATH")
    name, path = raw_spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name:
        raise ValueError("additional planner name must not be empty")
    if not path:
        raise ValueError("additional planner path must not be empty")
    return name, path


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
