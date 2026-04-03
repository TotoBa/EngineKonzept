"""Aggregate bounded planner-baseline metrics over the current workflow suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from train.datasets import search_teacher_artifact_name
from train.eval.planner import evaluate_two_ply_planner_baseline


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-summary", type=Path, required=True)
    parser.add_argument("--proposer-checkpoint", type=Path, required=True)
    parser.add_argument("--opponent-mode", choices=("none", "symbolic", "learned"), required=True)
    parser.add_argument("--opponent-checkpoint", type=Path)
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--reply-peak-weight", type=float, default=0.5)
    parser.add_argument("--pressure-weight", type=float, default=0.25)
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args()

    workflow_summary = json.loads(_resolve_repo_path(args.workflow_summary).read_text(encoding="utf-8"))
    proposer_checkpoint = _resolve_repo_path(args.proposer_checkpoint)
    opponent_checkpoint = (
        _resolve_repo_path(args.opponent_checkpoint)
        if args.opponent_checkpoint is not None
        else None
    )

    started_at = time.perf_counter()
    per_tier: dict[str, Any] = {}
    total_examples = 0
    teacher_covered_examples = 0
    root_top1_correct = 0.0
    reciprocal_rank_total = 0.0
    teacher_probability_total = 0.0
    reply_peak_total = 0.0
    pressure_total = 0.0
    uncertainty_total = 0.0

    for tier_name, tier_payload in dict(workflow_summary["tiers"]).items():
        verify_payload = dict(tier_payload["verify"])
        verify_output_dir = Path(verify_payload["output_dir"])
        metrics = evaluate_two_ply_planner_baseline(
            proposer_checkpoint,
            dataset_dir=Path(verify_payload["dataset_dir"]),
            search_teacher_path=verify_output_dir / search_teacher_artifact_name("test"),
            split="test",
            opponent_mode=args.opponent_mode,
            opponent_checkpoint=opponent_checkpoint,
            root_top_k=args.root_top_k,
            reply_peak_weight=args.reply_peak_weight,
            pressure_weight=args.pressure_weight,
            uncertainty_weight=args.uncertainty_weight,
            repo_root=REPO_ROOT,
        )
        per_tier[tier_name] = metrics.to_dict()
        total_examples += metrics.total_examples
        teacher_covered_examples += metrics.teacher_covered_examples
        root_top1_correct += (
            metrics.root_top1_accuracy * metrics.teacher_covered_examples
        )
        reciprocal_rank_total += (
            metrics.teacher_root_mean_reciprocal_rank * metrics.teacher_covered_examples
        )
        teacher_probability_total += (
            metrics.teacher_root_mean_probability * metrics.teacher_covered_examples
        )
        reply_peak_total += metrics.mean_reply_peak_probability * metrics.total_examples
        pressure_total += metrics.mean_pressure * metrics.total_examples
        uncertainty_total += metrics.mean_uncertainty * metrics.total_examples

    elapsed = time.perf_counter() - started_at
    aggregate = {
        "total_examples": total_examples,
        "teacher_covered_examples": teacher_covered_examples,
        "root_top1_accuracy": _ratio(root_top1_correct, teacher_covered_examples),
        "teacher_root_mean_reciprocal_rank": _ratio(
            reciprocal_rank_total,
            teacher_covered_examples,
        ),
        "teacher_root_mean_probability": _ratio(
            teacher_probability_total,
            teacher_covered_examples,
        ),
        "mean_reply_peak_probability": _ratio(reply_peak_total, total_examples),
        "mean_pressure": _ratio(pressure_total, total_examples),
        "mean_uncertainty": _ratio(uncertainty_total, total_examples),
        "examples_per_second": _ratio(total_examples, elapsed),
    }
    summary = {
        "workflow_summary": str(_resolve_repo_path(args.workflow_summary)),
        "proposer_checkpoint": str(proposer_checkpoint),
        "opponent_mode": args.opponent_mode,
        "opponent_checkpoint": str(opponent_checkpoint) if opponent_checkpoint is not None else None,
        "root_top_k": args.root_top_k,
        "reply_peak_weight": args.reply_peak_weight,
        "pressure_weight": args.pressure_weight,
        "uncertainty_weight": args.uncertainty_weight,
        "aggregate": aggregate,
        "per_tier": per_tier,
    }
    if args.output_path is not None:
        output_path = _resolve_repo_path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


if __name__ == "__main__":
    raise SystemExit(main())
