"""Materialize the expanded Phase-8 planner suite from a curriculum launch plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train.config import load_planner_train_config, resolve_repo_path
from train.trainers import evaluate_planner_checkpoint, train_planner


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CURRICULUM_PLAN = Path("artifacts/phase9/curriculum_active_experimental_expanded_v1.json")
DEFAULT_WORKFLOW_SUMMARY = Path(
    "/srv/schach/engine_training/phase8/planner_workflow_fulltargets_expanded_v2/summary.json"
)
DEFAULT_SUMMARY_PATH = Path("/srv/schach/engine_training/phase8/planner_active_experimental_expanded_v1_summary.json")
DEFAULT_COMPARE_PATH = Path("/srv/schach/engine_training/phase8/planner_active_experimental_expanded_v1_compare.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-plan", type=Path, default=DEFAULT_CURRICULUM_PLAN)
    parser.add_argument("--workflow-summary", type=Path, default=DEFAULT_WORKFLOW_SUMMARY)
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--verify-output-root", type=Path)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--compare-path", type=Path, default=DEFAULT_COMPARE_PATH)
    args = parser.parse_args()

    curriculum_plan_path = _resolve_repo_path(args.curriculum_plan)
    workflow_summary_path = _resolve_repo_path(args.workflow_summary)
    summary_path = _resolve_repo_path(args.summary_path)
    compare_path = _resolve_repo_path(args.compare_path)
    verify_output_root = (
        _resolve_repo_path(args.verify_output_root)
        if args.verify_output_root is not None
        else None
    )

    curriculum_plan = json.loads(curriculum_plan_path.read_text(encoding="utf-8"))
    workflow_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))

    requested_runs = set(args.runs or [])
    selected_runs = [
        run_payload
        for run_payload in curriculum_plan["planner_runs"]
        if not requested_runs or run_payload["name"] in requested_runs
    ]
    if requested_runs:
        missing_runs = sorted(requested_runs - {run_payload["name"] for run_payload in selected_runs})
        if missing_runs:
            raise ValueError(f"requested unknown planner runs: {', '.join(missing_runs)}")

    verify_paths = [Path(path) for path in workflow_summary["verify_paths"]]
    tier_verify_paths = {
        tier_name: Path(tier_payload["verify"]["planner_head_path"])
        for tier_name, tier_payload in workflow_summary["tiers"].items()
    }

    suite_summary: dict[str, Any] = {
        "curriculum_plan": str(curriculum_plan_path),
        "workflow_summary": str(workflow_summary_path),
        "verify_paths": [str(path) for path in verify_paths],
        "runs_requested": sorted(requested_runs),
        "runs": {},
    }

    compare_runs: dict[str, Any] = {}
    for run_payload in selected_runs:
        run_name = str(run_payload["name"])
        config_path = _resolve_repo_path(Path(run_payload["config_path"]))
        config = load_planner_train_config(config_path)
        output_dir = resolve_repo_path(REPO_ROOT, config.output_dir)
        checkpoint_path = resolve_repo_path(REPO_ROOT, config.export.bundle_dir) / config.export.checkpoint_name
        summary_file = output_dir / "summary.json"
        verify_root = verify_output_root if verify_output_root is not None else output_dir.parent
        verify_root.mkdir(parents=True, exist_ok=True)
        verify_file = verify_root / f"{output_dir.name}_verify.json"

        if not (args.skip_existing and summary_file.exists() and checkpoint_path.exists()):
            run = train_planner(config, repo_root=REPO_ROOT)
            summary_file = Path(run.summary_path)

        aggregate_metrics = evaluate_planner_checkpoint(
            checkpoint_path,
            dataset_paths=verify_paths,
            top_k=config.evaluation.top_k,
        )
        tier_metrics = {
            tier_name: evaluate_planner_checkpoint(
                checkpoint_path,
                dataset_path=dataset_path,
                top_k=config.evaluation.top_k,
            ).to_dict()
            for tier_name, dataset_path in tier_verify_paths.items()
        }
        verify_payload = {
            "model": run_name,
            "checkpoint": str(checkpoint_path),
            "config_path": str(config_path),
            "workflow_summary": str(workflow_summary_path),
            "aggregate": aggregate_metrics.to_dict(),
            "tiers": tier_metrics,
        }
        verify_file.write_text(json.dumps(verify_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        summary_payload = json.loads(summary_file.read_text(encoding="utf-8"))
        suite_summary["runs"][run_name] = {
            "config_path": str(config_path),
            "tags": list(run_payload.get("tags", [])),
            "summary_path": str(summary_file),
            "verify_path": str(verify_file),
            "checkpoint": str(checkpoint_path),
            "best_validation": summary_payload["best_validation"],
            "aggregate_verify": verify_payload["aggregate"],
        }
        compare_runs[run_name] = verify_payload["aggregate"]

    compare_payload = {
        "curriculum_plan": str(curriculum_plan_path),
        "workflow_summary": str(workflow_summary_path),
        "reference_run": (
            "planner_set_v2_expanded_v1"
            if "planner_set_v2_expanded_v1" in compare_runs
            else max(
                compare_runs,
                key=lambda name: (
                    compare_runs[name]["root_top1_accuracy"],
                    compare_runs[name]["teacher_root_mean_reciprocal_rank"],
                ),
            )
        ),
        "runs": compare_runs,
        "ranking_by_top1": _rank_runs(compare_runs, "root_top1_accuracy"),
        "ranking_by_mrr": _rank_runs(compare_runs, "teacher_root_mean_reciprocal_rank"),
    }

    summary_path.write_text(json.dumps(suite_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compare_path.write_text(json.dumps(compare_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(suite_summary, indent=2, sort_keys=True))
    return 0


def _rank_runs(runs: dict[str, Any], metric_name: str) -> list[dict[str, object]]:
    return [
        {"name": run_name, metric_name: runs[run_name][metric_name]}
        for run_name in sorted(
            runs,
            key=lambda run_name: (
                -float(runs[run_name][metric_name]),
                run_name,
            ),
        )
    ]


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
