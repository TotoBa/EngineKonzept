"""Build a full Phase-10 LAPv1 planner-head workflow over one all-data corpus tier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from train.datasets import planner_head_artifact_name, search_curriculum_artifact_name, search_teacher_artifact_name


REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_SCRIPT = REPO_ROOT / "python" / "scripts" / "build_opponent_workflow_dataset.py"
_PLANNER_HEAD_SCRIPT = REPO_ROOT / "python" / "scripts" / "build_planner_head_dataset.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset-dir", type=Path, required=True)
    parser.add_argument("--verify-dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-engine", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    train_dataset_dir = _resolve_repo_path(args.train_dataset_dir)
    verify_dataset_dir = _resolve_repo_path(args.verify_dataset_dir)
    checkpoint_path = _resolve_repo_path(args.checkpoint)
    teacher_engine = _resolve_repo_path(args.teacher_engine)
    output_root = _resolve_repo_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    train_summary = _load_dataset_summary(train_dataset_dir)
    verify_summary = _load_dataset_summary(verify_dataset_dir)

    split_specs = (
        {
            "name": "all_unique_train",
            "dataset_dir": train_dataset_dir,
            "split": "train",
            "canonical_split": "train",
            "output_dir": output_root / "all_unique_train_v1",
            "max_examples": int(train_summary["split_counts"]["train"]),
        },
        {
            "name": "all_unique_validation",
            "dataset_dir": train_dataset_dir,
            "split": "validation",
            "canonical_split": "validation",
            "output_dir": output_root / "all_unique_validation_v1",
            "max_examples": int(train_summary["split_counts"]["validation"]),
        },
        {
            "name": "all_unique_verify",
            "dataset_dir": verify_dataset_dir,
            "split": "test",
            "canonical_split": "test",
            "output_dir": output_root / "all_unique_verify_v1",
            "max_examples": int(verify_summary["split_counts"]["test"]),
        },
    )

    summary: dict[str, Any] = {
        "train_dataset_dir": str(train_dataset_dir),
        "verify_dataset_dir": str(verify_dataset_dir),
        "checkpoint": str(checkpoint_path),
        "teacher_engine": str(teacher_engine),
        "teacher_nodes": args.nodes,
        "teacher_multipv": args.multipv,
        "policy_temperature_cp": args.policy_temperature_cp,
        "top_k": args.top_k,
        "root_top_k": args.root_top_k,
        "log_every": args.log_every,
        "output_root": str(output_root),
        "tiers": {},
        "train_paths": [],
        "validation_paths": [],
        "verify_paths": [],
    }

    for split_spec in split_specs:
        split_name = str(split_spec["name"])
        output_dir = Path(split_spec["output_dir"])
        planner_head_path = output_dir / planner_head_artifact_name(str(split_spec["canonical_split"]))
        workflow_summary_path = output_dir / "summary.json"
        if not args.skip_existing or not planner_head_path.exists() or not workflow_summary_path.exists():
            _log(
                f"[workflow] building {split_name}: split={split_spec['split']} "
                f"dataset_dir={split_spec['dataset_dir']} max_examples={split_spec['max_examples']}"
            )
            _run_workflow_build(
                dataset_dir=Path(split_spec["dataset_dir"]),
                split=str(split_spec["split"]),
                checkpoint_path=checkpoint_path,
                teacher_engine=teacher_engine,
                output_dir=output_dir,
                max_examples=int(split_spec["max_examples"]),
                nodes=args.nodes,
                multipv=args.multipv,
                policy_temperature_cp=args.policy_temperature_cp,
                top_k=args.top_k,
                log_every=args.log_every,
            )
            _log(f"[workflow] building planner head for {split_name}")
            _run_planner_head_build(
                dataset_dir=Path(split_spec["dataset_dir"]),
                canonical_split=str(split_spec["canonical_split"]),
                workflow_dir=output_dir,
                checkpoint_path=checkpoint_path,
                root_top_k=args.root_top_k,
                output_path=planner_head_path,
            )
        split_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))
        planner_head_summary_path = planner_head_path.parent / "planner_head.summary.json"
        planner_head_summary = json.loads(planner_head_summary_path.read_text(encoding="utf-8"))
        summary_key = "verify_paths" if split_name.endswith("verify") else f"{split_spec['split']}_paths"
        summary[summary_key].append(str(planner_head_path))
        summary["tiers"][split_name] = {
            "dataset_dir": str(split_spec["dataset_dir"]),
            "workflow_dir": str(output_dir),
            "planner_head_path": str(planner_head_path),
            "workflow_summary": split_summary,
            "planner_head_summary": planner_head_summary,
        }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_workflow_build(
    *,
    dataset_dir: Path,
    split: str,
    checkpoint_path: Path,
    teacher_engine: Path,
    output_dir: Path,
    max_examples: int,
    nodes: int,
    multipv: int,
    policy_temperature_cp: float,
    top_k: int,
    log_every: int,
) -> None:
    command = [
        sys.executable,
        "-u",
        str(_WORKFLOW_SCRIPT),
        "--dataset-dir",
        str(dataset_dir),
        "--split",
        split,
        "--checkpoint",
        str(checkpoint_path),
        "--teacher-engine",
        str(teacher_engine),
        "--output-dir",
        str(output_dir),
        "--nodes",
        str(nodes),
        "--multipv",
        str(multipv),
        "--policy-temperature-cp",
        str(policy_temperature_cp),
        "--top-k",
        str(top_k),
        "--max-examples",
        str(max_examples),
        "--log-every",
        str(log_every),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _run_planner_head_build(
    *,
    dataset_dir: Path,
    canonical_split: str,
    workflow_dir: Path,
    checkpoint_path: Path,
    root_top_k: int,
    output_path: Path,
) -> None:
    command = [
        sys.executable,
        "-u",
        str(_PLANNER_HEAD_SCRIPT),
        "--dataset-dir",
        str(dataset_dir),
        "--split",
        canonical_split,
        "--search-teacher-path",
        str(workflow_dir / search_teacher_artifact_name(canonical_split)),
        "--search-curriculum-path",
        str(workflow_dir / search_curriculum_artifact_name(canonical_split)),
        "--proposer-checkpoint",
        str(checkpoint_path),
        "--opponent-mode",
        "none",
        "--root-top-k",
        str(root_top_k),
        "--output-path",
        str(output_path),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    planner_head_summary = json.loads((output_path.parent / "summary.json").read_text(encoding="utf-8"))
    (output_path.parent / "planner_head.summary.json").write_text(
        json.dumps(planner_head_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_dataset_summary(dataset_dir: Path) -> dict[str, Any]:
    summary_path = dataset_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"dataset summary not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _log(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
