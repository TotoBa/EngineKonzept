"""Backfill the preferred full-target expanded Phase-8 planner workflow suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train.datasets import (
    load_planner_head_examples,
    load_search_teacher_examples,
    materialize_planner_teacher_targets,
    planner_head_artifact_name,
    write_planner_head_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SOURCE_SUMMARY = Path(
    "/srv/schach/engine_training/phase8/planner_workflow_expanded_v1/summary.json"
)
_DEFAULT_OUTPUT_ROOT = Path(
    "/srv/schach/engine_training/phase8/planner_workflow_fulltargets_expanded_v2"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, default=_DEFAULT_SOURCE_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        help="Restrict the backfill to one or more named tiers from the source summary.",
    )
    args = parser.parse_args()

    source_summary_path = _resolve_repo_path(args.source_summary)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    requested_tiers = set(args.tiers or [])
    source_tiers = dict(source_summary["tiers"])
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
        "source_summary": str(source_summary_path),
        "output_root": str(output_root),
        "root_top_k": int(source_summary["root_top_k"]),
        "proposer_checkpoint": source_summary["proposer_checkpoint"],
        "dynamics_checkpoint": source_summary["dynamics_checkpoint"],
        "opponent_mode": source_summary["opponent_mode"],
        "opponent_checkpoint": source_summary["opponent_checkpoint"],
        "teacher_target_contract": "PlannerHeadV1/fulltargets_v1",
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
            split_payload = dict(tier_payload[split_name])
            source_planner_path = Path(split_payload["planner_head_path"])
            split_summary = dict(split_payload["summary"])
            teacher_path = Path(split_summary["search_teacher_path"])

            examples = load_planner_head_examples(source_planner_path)
            teacher_examples = load_search_teacher_examples(teacher_path)
            materialized = materialize_planner_teacher_targets(
                examples,
                teacher_examples=teacher_examples,
            )

            output_dir = output_root / f"{tier_name}_{split_name}_v1"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / planner_head_artifact_name(canonical_split)
            write_planner_head_artifact(output_path, materialized)

            teacher_target_fields_present = {
                "teacher_candidate_scores_cp": all(
                    example.teacher_candidate_scores_cp is not None for example in materialized
                ),
                "teacher_candidate_score_delta_targets_cp": all(
                    example.teacher_candidate_score_delta_targets_cp is not None
                    for example in materialized
                ),
                "teacher_rank_bucket_version": all(
                    example.teacher_rank_bucket_version is not None for example in materialized
                ),
                "teacher_candidate_rank_bucket_targets": all(
                    example.teacher_candidate_rank_bucket_targets is not None
                    for example in materialized
                ),
            }
            planner_summary = {
                **split_summary,
                "source_planner_head_path": str(source_planner_path),
                "output_path": str(output_path),
                "example_count": len(materialized),
                "teacher_target_contract": "PlannerHeadV1/fulltargets_v1",
                "teacher_target_fields_present": teacher_target_fields_present,
            }
            summary_path = output_dir / "summary.json"
            summary_path.write_text(
                json.dumps(planner_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tier_summary[split_name] = {
                "dataset_dir": split_payload["dataset_dir"],
                "workflow_dir": split_payload["workflow_dir"],
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
