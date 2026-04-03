"""Build planner-head artifacts over the current multi-corpus Phase-7 workflow suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from train.datasets import planner_head_artifact_name, search_curriculum_artifact_name, search_teacher_artifact_name


REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILD_SCRIPT = REPO_ROOT / "python" / "scripts" / "build_planner_head_dataset.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflow-summary",
        type=Path,
        default=Path("artifacts/phase7/opponent_workflow_corpus_suite_v1/summary.json"),
    )
    parser.add_argument("--proposer-checkpoint", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument(
        "--opponent-mode",
        choices=("none", "symbolic", "learned"),
        default="learned",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/phase8/planner_workflow_corpus_suite_v1"),
    )
    parser.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        help="Restrict the workflow build to one or more named tiers from the input summary.",
    )
    parser.add_argument("--root-top-k", type=int, default=4)
    args = parser.parse_args()

    workflow_summary = json.loads(_resolve_repo_path(args.workflow_summary).read_text(encoding="utf-8"))
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    proposer_checkpoint = _resolve_repo_path(args.proposer_checkpoint)
    dynamics_checkpoint = (
        _resolve_repo_path(args.dynamics_checkpoint)
        if args.dynamics_checkpoint is not None
        else None
    )
    opponent_checkpoint = (
        _resolve_repo_path(args.opponent_checkpoint)
        if args.opponent_checkpoint is not None
        else None
    )

    requested_tiers = set(args.tiers or [])
    source_tiers = dict(workflow_summary["tiers"])
    if requested_tiers:
        missing_tiers = sorted(requested_tiers - set(source_tiers))
        if missing_tiers:
            raise ValueError(f"requested unknown workflow tiers: {', '.join(missing_tiers)}")
        selected_tiers = {
            tier_name: source_tiers[tier_name]
            for tier_name in source_tiers
            if tier_name in requested_tiers
        }
    else:
        selected_tiers = source_tiers

    summary: dict[str, Any] = {
        "workflow_summary": str(_resolve_repo_path(args.workflow_summary)),
        "proposer_checkpoint": str(proposer_checkpoint),
        "dynamics_checkpoint": str(dynamics_checkpoint) if dynamics_checkpoint is not None else None,
        "opponent_mode": args.opponent_mode,
        "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint is not None else None,
        "tiers_requested": sorted(requested_tiers),
        "root_top_k": args.root_top_k,
        "output_root": str(output_root),
        "tiers": {},
        "train_paths": [],
        "validation_paths": [],
        "verify_paths": [],
    }

    for tier_name, tier_payload in selected_tiers.items():
        tier_summary: dict[str, Any] = {}
        for split_name, canonical_split in (
            ("train", "train"),
            ("validation", "validation"),
            ("verify", "test"),
        ):
            split_payload = dict(tier_payload[split_name])
            workflow_dir = Path(split_payload["output_dir"])
            dataset_dir = Path(split_payload["dataset_dir"])
            output_dir = output_root / f"{tier_name}_{split_name}_v1"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / planner_head_artifact_name(canonical_split)
            command = [
                sys.executable,
                str(_BUILD_SCRIPT),
                "--dataset-dir",
                str(dataset_dir),
                "--split",
                canonical_split,
                "--search-teacher-path",
                str(workflow_dir / search_teacher_artifact_name(canonical_split)),
                "--search-curriculum-path",
                str(workflow_dir / search_curriculum_artifact_name(canonical_split)),
                "--proposer-checkpoint",
                str(proposer_checkpoint),
                "--opponent-mode",
                args.opponent_mode,
                "--root-top-k",
                str(args.root_top_k),
                "--output-path",
                str(output_path),
            ]
            if dynamics_checkpoint is not None:
                command.extend(["--dynamics-checkpoint", str(dynamics_checkpoint)])
            if opponent_checkpoint is not None:
                command.extend(["--opponent-checkpoint", str(opponent_checkpoint)])
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            planner_summary = json.loads(
                (
                    output_dir / "summary.json"
                ).read_text(encoding="utf-8")
            ) if (output_dir / "summary.json").exists() else {
                "output_path": str(output_path),
                "example_count": sum(1 for _ in output_path.open(encoding="utf-8")) if output_path.exists() else 0,
            }
            tier_summary[split_name] = {
                "dataset_dir": str(dataset_dir),
                "workflow_dir": str(workflow_dir),
                "planner_head_path": str(output_path),
                "summary": planner_summary,
            }
            summary_key = "verify_paths" if split_name == "verify" else f"{split_name}_paths"
            summary[summary_key].append(str(output_path))
        summary["tiers"][tier_name] = tier_summary

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
