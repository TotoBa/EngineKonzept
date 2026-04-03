"""Materialize latent planner-head artifacts from an existing planner workflow summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from train.datasets import (
    load_planner_head_examples,
    materialize_planner_latent_features,
    planner_head_artifact_name,
    write_planner_head_artifact,
)
from train.eval.dynamics import load_dynamics_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-summary", type=Path, required=True)
    parser.add_argument("--dynamics-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        help="Restrict the latent materialization to one or more named tiers from the workflow summary.",
    )
    args = parser.parse_args(argv)

    workflow_summary_path = _resolve_repo_path(args.workflow_summary)
    workflow_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dynamics_checkpoint = _resolve_repo_path(args.dynamics_checkpoint)
    dynamics_model, _ = load_dynamics_checkpoint(dynamics_checkpoint)

    source_tiers = dict(workflow_summary["tiers"])
    requested_tiers = set(args.tiers or [])
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
        "workflow_summary": str(workflow_summary_path),
        "dynamics_checkpoint": str(dynamics_checkpoint),
        "output_root": str(output_root),
        "tiers_requested": sorted(requested_tiers),
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
            planner_head_path = Path(tier_payload[split_name]["planner_head_path"])
            print(
                f"[latent-workflow] tier={tier_name} split={split_name} input={planner_head_path}",
                file=sys.stderr,
                flush=True,
            )
            examples = load_planner_head_examples(planner_head_path)
            materialized = materialize_planner_latent_features(
                examples,
                dynamics_model=dynamics_model,
            )
            output_dir = output_root / f"{tier_name}_{split_name}_v1"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / planner_head_artifact_name(canonical_split)
            write_planner_head_artifact(output_path, materialized)
            split_summary = {
                "input_path": str(planner_head_path),
                "output_path": str(output_path),
                "example_count": len(materialized),
                "latent_state_version": 1 if materialized else None,
            }
            (output_dir / "summary.json").write_text(
                json.dumps(split_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"[latent-workflow] tier={tier_name} split={split_name} examples={len(materialized)} output={output_path}",
                file=sys.stderr,
                flush=True,
            )
            tier_summary[split_name] = {
                "dataset_dir": tier_payload[split_name]["dataset_dir"],
                "workflow_dir": tier_payload[split_name]["workflow_dir"],
                "planner_head_path": str(output_path),
                "summary": split_summary,
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
