"""Build a full Phase-10 LAPv1 planner-head workflow over one all-data corpus tier."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from train.datasets import (
    opponent_head_artifact_name,
    planner_head_artifact_name,
    search_curriculum_artifact_name,
    search_disagreements_artifact_name,
    search_teacher_artifact_name,
    search_traces_artifact_name,
)
from train.datasets.lapv1_training import lapv1_training_artifact_name


REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_SCRIPT = REPO_ROOT / "python" / "scripts" / "build_opponent_workflow_dataset.py"
_PLANNER_HEAD_SCRIPT = REPO_ROOT / "python" / "scripts" / "build_planner_head_dataset.py"
_LAPV1_TRAINING_SCRIPT = REPO_ROOT / "python" / "scripts" / "build_lapv1_training_dataset.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset-dir", type=Path, required=True)
    parser.add_argument("--verify-dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-engine", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--train-depth", type=int, default=None)
    parser.add_argument("--validation-depth", type=int, default=None)
    parser.add_argument("--verify-depth", type=int, default=None)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--policy-temperature-cp", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--root-top-k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--train-chunk-start", type=int, default=None)
    parser.add_argument("--train-chunk-end", type=int, default=None)
    parser.add_argument("--validation-chunk-start", type=int, default=None)
    parser.add_argument("--validation-chunk-end", type=int, default=None)
    parser.add_argument("--verify-chunk-start", type=int, default=None)
    parser.add_argument("--verify-chunk-end", type=int, default=None)
    parser.add_argument(
        "--only-split",
        choices=("train", "validation", "verify"),
        default=None,
        help="limit chunk building/finalize work to one canonical split",
    )
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if args.parallel_workers <= 0:
        raise ValueError("parallel-workers must be positive")
    if args.skip_finalize and args.finalize_only:
        raise ValueError("skip-finalize and finalize-only cannot be combined")
    if (
        args.nodes is None
        and args.depth is None
        and args.train_depth is None
        and args.validation_depth is None
        and args.verify_depth is None
    ):
        raise ValueError(
            "one of --nodes, --depth, --train-depth, --validation-depth, or --verify-depth must be provided"
        )

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
            "chunk_start": args.train_chunk_start,
            "chunk_end": args.train_chunk_end,
        },
        {
            "name": "all_unique_validation",
            "dataset_dir": train_dataset_dir,
            "split": "validation",
            "canonical_split": "validation",
            "output_dir": output_root / "all_unique_validation_v1",
            "max_examples": int(train_summary["split_counts"]["validation"]),
            "chunk_start": args.validation_chunk_start,
            "chunk_end": args.validation_chunk_end,
        },
        {
            "name": "all_unique_verify",
            "dataset_dir": verify_dataset_dir,
            "split": "test",
            "canonical_split": "test",
            "output_dir": output_root / "all_unique_verify_v1",
            "max_examples": int(verify_summary["split_counts"]["test"]),
            "chunk_start": args.verify_chunk_start,
            "chunk_end": args.verify_chunk_end,
        },
    )
    if args.only_split is not None:
        selected_name = {
            "train": "all_unique_train",
            "validation": "all_unique_validation",
            "verify": "all_unique_verify",
        }[args.only_split]
        split_specs = tuple(
            split_spec for split_spec in split_specs if split_spec["name"] == selected_name
        )

    summary: dict[str, Any] = {
        "train_dataset_dir": str(train_dataset_dir),
        "verify_dataset_dir": str(verify_dataset_dir),
        "checkpoint": str(checkpoint_path),
        "teacher_engine": str(teacher_engine),
        "teacher_nodes": args.nodes,
        "teacher_depth": args.depth,
        "train_teacher_depth": args.train_depth,
        "validation_teacher_depth": args.validation_depth,
        "verify_teacher_depth": args.verify_depth,
        "teacher_multipv": args.multipv,
        "policy_temperature_cp": args.policy_temperature_cp,
        "top_k": args.top_k,
        "root_top_k": args.root_top_k,
        "chunk_size": args.chunk_size,
        "parallel_workers": args.parallel_workers,
        "log_every": args.log_every,
        "skip_finalize": bool(args.skip_finalize),
        "finalize_only": bool(args.finalize_only),
        "output_root": str(output_root),
        "tiers": {},
        "train_paths": [],
        "validation_paths": [],
        "verify_paths": [],
    }

    for split_spec in split_specs:
        split_name = str(split_spec["name"])
        output_dir = Path(split_spec["output_dir"])
        total_chunks = _total_chunks_for_examples(
            max_examples=int(split_spec["max_examples"]),
            chunk_size=args.chunk_size,
        )
        chunk_start, chunk_end = _normalize_chunk_range(
            total_chunks=total_chunks,
            chunk_start=_optional_int(split_spec["chunk_start"]),
            chunk_end=_optional_int(split_spec["chunk_end"]),
        )
        chunk_records = _build_expected_chunk_records(
            output_dir=output_dir,
            canonical_split=str(split_spec["canonical_split"]),
            max_examples=int(split_spec["max_examples"]),
            chunk_size=args.chunk_size,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
        )
        planner_head_path = output_dir / planner_head_artifact_name(str(split_spec["canonical_split"]))
        workflow_summary_path = output_dir / "summary.json"
        planner_head_summary_path = output_dir / "planner_head.summary.json"
        lapv1_artifact_path = output_dir / lapv1_training_artifact_name(str(split_spec["canonical_split"]))
        lapv1_summary_path = output_dir / "lapv1.summary.json"
        if not args.finalize_only and chunk_records:
            _log(
                f"[workflow] chunked build for {split_name}: split={split_spec['split']} "
                f"max_examples={split_spec['max_examples']} chunk_size={args.chunk_size} "
                f"chunk_range={chunk_start}-{chunk_end}"
            )
            _build_chunked_split_workflow(
                dataset_dir=Path(split_spec["dataset_dir"]),
                split=str(split_spec["split"]),
                canonical_split=str(split_spec["canonical_split"]),
                checkpoint_path=checkpoint_path,
                teacher_engine=teacher_engine,
                output_dir=output_dir,
                nodes=args.nodes,
                depth=_resolve_teacher_depth_for_split(
                    split=str(split_spec["split"]),
                    default_depth=args.depth,
                    train_depth=args.train_depth,
                    validation_depth=args.validation_depth,
                    verify_depth=args.verify_depth,
                ),
                multipv=args.multipv,
                policy_temperature_cp=args.policy_temperature_cp,
                top_k=args.top_k,
                root_top_k=args.root_top_k,
                parallel_workers=args.parallel_workers,
                log_every=args.log_every,
                skip_existing=bool(args.skip_existing),
                chunk_records=chunk_records,
            )
        if not args.skip_finalize:
            _finalize_chunked_split_workflow(
                dataset_dir=Path(split_spec["dataset_dir"]),
                split=str(split_spec["split"]),
                canonical_split=str(split_spec["canonical_split"]),
                checkpoint_path=checkpoint_path,
                teacher_engine=teacher_engine,
                output_dir=output_dir,
                max_examples=int(split_spec["max_examples"]),
                chunk_size=args.chunk_size,
                nodes=args.nodes,
                depth=_resolve_teacher_depth_for_split(
                    split=str(split_spec["split"]),
                    default_depth=args.depth,
                    train_depth=args.train_depth,
                    validation_depth=args.validation_depth,
                    verify_depth=args.verify_depth,
                ),
                multipv=args.multipv,
                policy_temperature_cp=args.policy_temperature_cp,
                root_top_k=args.root_top_k,
            )
        if args.skip_finalize:
            summary["tiers"][split_name] = {
                "dataset_dir": str(split_spec["dataset_dir"]),
                "workflow_dir": str(output_dir),
                "chunk_range": [chunk_start, chunk_end],
                "chunk_count": len(chunk_records),
                "status": "chunk_build_only",
            }
            continue
        split_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))
        planner_head_summary = json.loads(planner_head_summary_path.read_text(encoding="utf-8"))
        lapv1_summary = json.loads(lapv1_summary_path.read_text(encoding="utf-8"))
        summary_key = "verify_paths" if split_name.endswith("verify") else f"{split_spec['split']}_paths"
        summary[summary_key].append(str(planner_head_path))
        lapv1_summary_key = (
            "lapv1_verify_paths"
            if split_name.endswith("verify")
            else f"lapv1_{split_spec['split']}_paths"
        )
        summary.setdefault(lapv1_summary_key, []).append(str(lapv1_artifact_path))
        summary["tiers"][split_name] = {
            "dataset_dir": str(split_spec["dataset_dir"]),
            "workflow_dir": str(output_dir),
            "planner_head_path": str(planner_head_path),
            "lapv1_path": str(lapv1_artifact_path),
            "workflow_summary": split_summary,
            "planner_head_summary": planner_head_summary,
            "lapv1_summary": lapv1_summary,
        }

    if not args.skip_finalize:
        summary_path = output_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _build_chunked_split_workflow(
    *,
    dataset_dir: Path,
    split: str,
    canonical_split: str,
    checkpoint_path: Path,
    teacher_engine: Path,
    output_dir: Path,
    nodes: int | None,
    depth: int | None,
    multipv: int,
    policy_temperature_cp: float,
    top_k: int,
    root_top_k: int,
    parallel_workers: int,
    log_every: int,
    skip_existing: bool,
    chunk_records: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = output_dir / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    chunk_jobs = [
        {
            "chunk_index": int(record["index"]),
            "total_chunks": int(record["total_chunks"]),
            "start_index": int(record["start_index"]),
            "example_count": int(record["example_count"]),
            "chunk_dir": Path(record["dir"]),
        }
        for record in chunk_records
    ]

    if parallel_workers == 1:
        chunk_records = [
            _process_chunk_job(
                job=job,
                dataset_dir=dataset_dir,
                split=split,
                canonical_split=canonical_split,
                checkpoint_path=checkpoint_path,
                teacher_engine=teacher_engine,
                nodes=nodes,
                depth=depth,
                multipv=multipv,
                policy_temperature_cp=policy_temperature_cp,
                top_k=top_k,
                root_top_k=root_top_k,
                log_every=log_every,
                skip_existing=skip_existing,
            )
            for job in chunk_jobs
        ]
    else:
        chunk_records = []
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(
                    _process_chunk_job,
                    job=job,
                    dataset_dir=dataset_dir,
                    split=split,
                    canonical_split=canonical_split,
                    checkpoint_path=checkpoint_path,
                    teacher_engine=teacher_engine,
                    nodes=nodes,
                    depth=depth,
                    multipv=multipv,
                    policy_temperature_cp=policy_temperature_cp,
                    top_k=top_k,
                    root_top_k=root_top_k,
                    log_every=log_every,
                    skip_existing=skip_existing,
                )
                for job in chunk_jobs
            ]
            for future in as_completed(futures):
                chunk_records.append(future.result())
        chunk_records.sort(key=lambda record: int(record["index"]))


def _finalize_chunked_split_workflow(
    *,
    dataset_dir: Path,
    split: str,
    canonical_split: str,
    checkpoint_path: Path,
    teacher_engine: Path,
    output_dir: Path,
    max_examples: int,
    chunk_size: int,
    nodes: int | None,
    depth: int | None,
    multipv: int,
    policy_temperature_cp: float,
    root_top_k: int,
) -> None:
    chunk_records = _build_expected_chunk_records(
        output_dir=output_dir,
        canonical_split=canonical_split,
        max_examples=max_examples,
        chunk_size=chunk_size,
        chunk_start=1,
        chunk_end=None,
    )
    _validate_chunk_records_complete(
        chunk_records=chunk_records,
        canonical_split=canonical_split,
    )
    _merge_chunk_artifact(
        output_path=output_dir / search_teacher_artifact_name(canonical_split),
        chunk_paths=[
            Path(record["dir"]) / search_teacher_artifact_name(canonical_split)
            for record in chunk_records
        ],
    )
    _merge_chunk_artifact(
        output_path=output_dir / search_traces_artifact_name(canonical_split),
        chunk_paths=[
            Path(record["dir"]) / search_traces_artifact_name(canonical_split)
            for record in chunk_records
        ],
    )
    _merge_chunk_artifact(
        output_path=output_dir / search_disagreements_artifact_name(canonical_split),
        chunk_paths=[
            Path(record["dir"]) / search_disagreements_artifact_name(canonical_split)
            for record in chunk_records
        ],
    )
    _merge_chunk_artifact(
        output_path=output_dir / search_curriculum_artifact_name(canonical_split),
        chunk_paths=[
            Path(record["dir"]) / search_curriculum_artifact_name(canonical_split)
            for record in chunk_records
        ],
    )
    _merge_chunk_artifact(
        output_path=output_dir / planner_head_artifact_name(canonical_split),
        chunk_paths=[Path(record["planner_head_path"]) for record in chunk_records],
    )
    _merge_chunk_artifact(
        output_path=output_dir / lapv1_training_artifact_name(canonical_split),
        chunk_paths=[Path(record["lapv1_path"]) for record in chunk_records],
    )

    opponent_chunk_paths = [
        Path(record["dir"]) / opponent_head_artifact_name(canonical_split)
        for record in chunk_records
        if (Path(record["dir"]) / opponent_head_artifact_name(canonical_split)).exists()
    ]
    if opponent_chunk_paths:
        _merge_chunk_artifact(
            output_path=output_dir / opponent_head_artifact_name(canonical_split),
            chunk_paths=opponent_chunk_paths,
        )

    workflow_summary = _aggregate_workflow_chunk_summaries(
        dataset_dir=dataset_dir,
        split=split,
        output_dir=output_dir,
        chunk_records=chunk_records,
        checkpoint_path=checkpoint_path,
        teacher_engine=teacher_engine,
        nodes=nodes,
        depth=depth,
        multipv=multipv,
        policy_temperature_cp=policy_temperature_cp,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(workflow_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    planner_head_summary = _aggregate_planner_head_chunk_summaries(
        dataset_dir=dataset_dir,
        split=canonical_split,
        output_path=output_dir / planner_head_artifact_name(canonical_split),
        chunk_records=chunk_records,
        checkpoint_path=checkpoint_path,
        root_top_k=root_top_k,
    )
    (output_dir / "planner_head.summary.json").write_text(
        json.dumps(planner_head_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lapv1_summary = _aggregate_lapv1_chunk_summaries(
        split=canonical_split,
        output_path=output_dir / lapv1_training_artifact_name(canonical_split),
        chunk_records=chunk_records,
    )
    (output_dir / "lapv1.summary.json").write_text(
        json.dumps(lapv1_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _aggregate_workflow_chunk_summaries(
    *,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
    chunk_records: Sequence[dict[str, Any]],
    checkpoint_path: Path,
    teacher_engine: Path,
    nodes: int | None,
    depth: int | None,
    multipv: int,
    policy_temperature_cp: float,
) -> dict[str, Any]:
    chunk_summaries = [
        json.loads(Path(record["workflow_summary_path"]).read_text(encoding="utf-8"))
        for record in chunk_records
    ]
    example_count = sum(int(summary["example_count"]) for summary in chunk_summaries)
    return {
        "dataset_dir": str(dataset_dir),
        "split": split,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "teacher_engine": str(teacher_engine),
        "teacher_nodes": nodes,
        "teacher_depth": depth,
        "teacher_multipv": multipv,
        "example_count": example_count,
        "reply_supervised_count": sum(
            int(summary.get("reply_supervised_count", 0))
            for summary in chunk_summaries
        ),
        "teacher_coverage_ratio": _weighted_mean(
            chunk_summaries,
            weight_key="example_count",
            value_key="teacher_coverage_ratio",
        ),
        "curriculum_priority_mean": _weighted_mean(
            chunk_summaries,
            weight_key="example_count",
            value_key="curriculum_priority_mean",
        ),
        "disagreement_rate": _weighted_mean(
            chunk_summaries,
            weight_key="example_count",
            value_key="disagreement_rate",
        ),
        "policy_temperature_cp": policy_temperature_cp,
        "chunk_count": len(chunk_records),
        "chunks": list(chunk_records),
    }


def _aggregate_planner_head_chunk_summaries(
    *,
    dataset_dir: Path,
    split: str,
    output_path: Path,
    chunk_records: Sequence[dict[str, Any]],
    checkpoint_path: Path,
    root_top_k: int,
) -> dict[str, Any]:
    chunk_summaries = [
        json.loads(Path(record["planner_head_summary_path"]).read_text(encoding="utf-8"))
        for record in chunk_records
    ]
    return {
        "dataset_dir": str(dataset_dir),
        "split": split,
        "search_teacher_path": str(output_path.parent / search_teacher_artifact_name(split)),
        "search_curriculum_path": str(output_path.parent / search_curriculum_artifact_name(split)),
        "proposer_checkpoint": str(checkpoint_path),
        "dynamics_checkpoint": None,
        "opponent_mode": "none",
        "opponent_checkpoint": None,
        "root_top_k": root_top_k,
        "max_examples": sum(int(summary["example_count"]) for summary in chunk_summaries),
        "output_path": str(output_path),
        "example_count": sum(int(summary["example_count"]) for summary in chunk_summaries),
        "mean_curriculum_priority": _weighted_mean(
            chunk_summaries,
            weight_key="example_count",
            value_key="mean_curriculum_priority",
        ),
        "chunk_count": len(chunk_records),
        "chunks": list(chunk_records),
    }


def _aggregate_lapv1_chunk_summaries(
    *,
    split: str,
    output_path: Path,
    chunk_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    chunk_summaries = [
        json.loads(Path(record["lapv1_summary_path"]).read_text(encoding="utf-8"))
        for record in chunk_records
    ]
    return {
        "split": split,
        "output_path": str(output_path),
        "example_count": sum(int(summary["example_count"]) for summary in chunk_summaries),
        "mean_candidate_count": _weighted_mean(
            chunk_summaries,
            weight_key="example_count",
            value_key="mean_candidate_count",
        ),
        "mean_curriculum_priority": _weighted_mean(
            chunk_summaries,
            weight_key="example_count",
            value_key="mean_curriculum_priority",
        ),
        "chunk_count": len(chunk_records),
        "chunks": list(chunk_records),
    }


def _weighted_mean(
    summaries: Sequence[dict[str, Any]],
    *,
    weight_key: str,
    value_key: str,
) -> float:
    total_weight = sum(float(summary.get(weight_key, 0)) for summary in summaries)
    if total_weight <= 0.0:
        return 0.0
    weighted_sum = sum(
        float(summary.get(weight_key, 0)) * float(summary.get(value_key, 0.0))
        for summary in summaries
    )
    return weighted_sum / total_weight


def _resolve_teacher_depth_for_split(
    *,
    split: str,
    default_depth: int | None,
    train_depth: int | None,
    validation_depth: int | None,
    verify_depth: int | None,
) -> int | None:
    if split == "train" and train_depth is not None:
        return train_depth
    if split == "validation" and validation_depth is not None:
        return validation_depth
    if split == "test" and verify_depth is not None:
        return verify_depth
    return default_depth


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _total_chunks_for_examples(*, max_examples: int, chunk_size: int) -> int:
    return math.ceil(max_examples / chunk_size) if max_examples else 0


def _normalize_chunk_range(
    *,
    total_chunks: int,
    chunk_start: int | None,
    chunk_end: int | None,
) -> tuple[int, int]:
    if total_chunks <= 0:
        return 1, 0
    start = 1 if chunk_start is None else chunk_start
    end = total_chunks if chunk_end is None else chunk_end
    if start < 1 or end < 1:
        raise ValueError("chunk range indices must be positive")
    if start > end:
        raise ValueError("chunk_start must be <= chunk_end")
    if end > total_chunks:
        raise ValueError(f"chunk_end {end} exceeds total_chunks {total_chunks}")
    return start, end


def _build_expected_chunk_records(
    *,
    output_dir: Path,
    canonical_split: str,
    max_examples: int,
    chunk_size: int,
    chunk_start: int,
    chunk_end: int | None,
) -> list[dict[str, Any]]:
    total_chunks = _total_chunks_for_examples(max_examples=max_examples, chunk_size=chunk_size)
    start, end = _normalize_chunk_range(
        total_chunks=total_chunks,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
    )
    chunk_root = output_dir / "chunks"
    records: list[dict[str, Any]] = []
    for chunk_index, start_index in enumerate(range(0, max_examples, chunk_size), start=1):
        if chunk_index < start or chunk_index > end:
            continue
        example_count = min(chunk_size, max_examples - start_index)
        chunk_dir = chunk_root / f"chunk_{chunk_index:04d}_{start_index:08d}"
        records.append(
            {
                "index": chunk_index,
                "total_chunks": total_chunks,
                "start_index": start_index,
                "example_count": example_count,
                "dir": str(chunk_dir),
                "workflow_summary_path": str(chunk_dir / "workflow.summary.json"),
                "planner_head_path": str(chunk_dir / planner_head_artifact_name(canonical_split)),
                "planner_head_summary_path": str(chunk_dir / "planner_head.summary.json"),
                "lapv1_path": str(chunk_dir / lapv1_training_artifact_name(canonical_split)),
                "lapv1_summary_path": str(chunk_dir / "lapv1.summary.json"),
            }
        )
    return records


def _validate_chunk_records_complete(
    *,
    chunk_records: Sequence[dict[str, Any]],
    canonical_split: str,
) -> None:
    required_paths = []
    for record in chunk_records:
        chunk_dir = Path(record["dir"])
        required_paths.extend(
            [
                Path(record["workflow_summary_path"]),
                chunk_dir / search_teacher_artifact_name(canonical_split),
                chunk_dir / search_traces_artifact_name(canonical_split),
                chunk_dir / search_disagreements_artifact_name(canonical_split),
                chunk_dir / search_curriculum_artifact_name(canonical_split),
                Path(record["planner_head_path"]),
                Path(record["planner_head_summary_path"]),
                Path(record["lapv1_path"]),
                Path(record["lapv1_summary_path"]),
            ]
        )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing chunk artifacts required for finalize:\n" + "\n".join(missing[:20])
        )


def _merge_chunk_artifact(*, output_path: Path, chunk_paths: Sequence[Path]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_handle:
        for chunk_path in chunk_paths:
            if not chunk_path.exists():
                continue
            with chunk_path.open("r", encoding="utf-8") as chunk_handle:
                for raw_line in chunk_handle:
                    if raw_line.strip():
                        output_handle.write(raw_line.rstrip("\n"))
                        output_handle.write("\n")


def _process_chunk_job(
    *,
    job: dict[str, Any],
    dataset_dir: Path,
    split: str,
    canonical_split: str,
    checkpoint_path: Path,
    teacher_engine: Path,
    nodes: int | None,
    depth: int | None,
    multipv: int,
    policy_temperature_cp: float,
    top_k: int,
    root_top_k: int,
    log_every: int,
    skip_existing: bool,
) -> dict[str, Any]:
    chunk_index = int(job["chunk_index"])
    total_chunks = int(job["total_chunks"])
    start_index = int(job["start_index"])
    example_count = int(job["example_count"])
    chunk_dir = Path(job["chunk_dir"])
    workflow_summary_path = chunk_dir / "workflow.summary.json"
    planner_head_path = chunk_dir / planner_head_artifact_name(canonical_split)
    planner_head_summary_path = chunk_dir / "planner_head.summary.json"
    lapv1_path = chunk_dir / lapv1_training_artifact_name(canonical_split)
    lapv1_summary_path = chunk_dir / "lapv1.summary.json"
    _log(
        f"[workflow:{split}] chunk {chunk_index}/{total_chunks} "
        f"start={start_index} count={example_count}"
    )
    if not skip_existing or not workflow_summary_path.exists():
        _run_workflow_build(
            dataset_dir=dataset_dir,
            split=split,
            checkpoint_path=checkpoint_path,
            teacher_engine=teacher_engine,
            output_dir=chunk_dir,
            start_index=start_index,
            max_examples=example_count,
            nodes=nodes,
            depth=depth,
            multipv=multipv,
            policy_temperature_cp=policy_temperature_cp,
            top_k=top_k,
            log_every=log_every,
        )
    if not skip_existing or not planner_head_path.exists() or not planner_head_summary_path.exists():
        _run_planner_head_build(
            dataset_dir=dataset_dir,
            canonical_split=canonical_split,
            workflow_dir=chunk_dir,
            checkpoint_path=checkpoint_path,
            start_index=start_index,
            max_examples=example_count,
            root_top_k=root_top_k,
            output_path=planner_head_path,
        )
    if not skip_existing or not lapv1_path.exists() or not lapv1_summary_path.exists():
        _run_lapv1_training_artifact_build(
            planner_head_path=planner_head_path,
            output_path=lapv1_path,
            log_every=log_every,
        )
    return {
        "index": chunk_index,
        "start_index": start_index,
        "example_count": example_count,
        "dir": str(chunk_dir),
        "workflow_summary_path": str(workflow_summary_path),
        "planner_head_path": str(planner_head_path),
        "planner_head_summary_path": str(planner_head_summary_path),
        "lapv1_path": str(lapv1_path),
        "lapv1_summary_path": str(lapv1_summary_path),
    }


def _run_workflow_build(
    *,
    dataset_dir: Path,
    split: str,
    checkpoint_path: Path,
    teacher_engine: Path,
    output_dir: Path,
    start_index: int,
    max_examples: int,
    nodes: int | None,
    depth: int | None,
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
        "--multipv",
        str(multipv),
        "--policy-temperature-cp",
        str(policy_temperature_cp),
        "--top-k",
        str(top_k),
        "--start-index",
        str(start_index),
        "--max-examples",
        str(max_examples),
        "--log-every",
        str(log_every),
        "--skip-opponent-head",
    ]
    if depth is not None:
        command.extend(["--depth", str(depth)])
    elif nodes is not None:
        command.extend(["--nodes", str(nodes)])
    else:
        raise ValueError("one of nodes or depth must be provided")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    summary_path = output_dir / "summary.json"
    workflow_summary_path = output_dir / "workflow.summary.json"
    workflow_summary_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")


def _run_planner_head_build(
    *,
    dataset_dir: Path,
    canonical_split: str,
    workflow_dir: Path,
    checkpoint_path: Path,
    start_index: int,
    max_examples: int,
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
        "--start-index",
        str(start_index),
        "--max-examples",
        str(max_examples),
        "--output-path",
        str(output_path),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    summary_path = output_path.parent / "summary.json"
    planner_head_summary_path = output_path.parent / "planner_head.summary.json"
    planner_head_summary_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")


def _run_lapv1_training_artifact_build(
    *,
    planner_head_path: Path,
    output_path: Path,
    log_every: int,
) -> None:
    command = [
        sys.executable,
        "-u",
        str(_LAPV1_TRAINING_SCRIPT),
        "--planner-head-path",
        str(planner_head_path),
        "--output-path",
        str(output_path),
        "--log-every",
        str(log_every),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


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
