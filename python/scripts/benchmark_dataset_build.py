"""Benchmark end-to-end dataset builds across multiple oracle schedules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Sequence

from train.datasets import SUPPORTED_SOURCE_FORMATS, build_dataset, load_raw_records
from train.datasets.io import write_dataset_artifacts
from train.datasets.schema import RawPositionRecord
from train.datasets.schema import SplitRatios

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-format", choices=SUPPORTED_SOURCE_FORMATS, required=True)
    parser.add_argument("--source-name")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-out", type=Path)
    parser.add_argument("--seed", default="phase-4")
    parser.add_argument("--records", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Benchmark config in the form name:workers:batch_size",
    )
    args = parser.parse_args(argv)

    records = load_raw_records(
        _resolve_repo_path(args.input),
        args.source_format,
        source_name=args.source_name,
    )
    if args.records > 0:
        records = _select_records(records, target_count=args.records)
    if not records:
        raise ValueError("benchmark input must produce at least one record")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    configs = [_parse_config(value) for value in args.config]
    results: list[dict[str, object]] = []
    baseline_digests: dict[str, str] | None = None

    for name, workers, batch_size in configs:
        output_dir = args.output_root / name
        if output_dir.exists():
            shutil.rmtree(output_dir)

        repeat_seconds: list[float] = []
        dataset = None
        for repeat_index in range(args.repeats):
            if output_dir.exists():
                shutil.rmtree(output_dir)
            started = time.perf_counter()
            dataset = build_dataset(
                records,
                ratios=SplitRatios(),
                seed=args.seed,
                repo_root=REPO_ROOT,
                oracle_workers=workers,
                oracle_batch_size=batch_size,
            )
            write_dataset_artifacts(output_dir, dataset)
            repeat_seconds.append(time.perf_counter() - started)
        assert dataset is not None

        digests = {
            "dataset_jsonl": _sha256(output_dir / "dataset.jsonl"),
            "train_jsonl": _sha256(output_dir / "train.jsonl"),
            "validation_jsonl": _sha256(output_dir / "validation.jsonl"),
            "test_jsonl": _sha256(output_dir / "test.jsonl"),
        }
        if baseline_digests is None:
            baseline_digests = digests
        elif digests != baseline_digests:
            raise RuntimeError(
                f"benchmark config {name!r} produced different output digests: {digests}"
            )

        results.append(
            {
                "name": name,
                "oracle_workers": workers,
                "oracle_batch_size": batch_size,
                "seconds": round(sum(repeat_seconds) / len(repeat_seconds), 6),
                "repeat_seconds": [round(value, 6) for value in repeat_seconds],
                "records_per_second": round(
                    len(records) / (sum(repeat_seconds) / len(repeat_seconds)),
                    3,
                ),
                "oracle_schedule": dataset.summary["oracle_schedule"],
                "digests": digests,
            }
        )

    fastest = min(results, key=lambda result: float(result["seconds"]))
    summary = {
        "input": str(_resolve_repo_path(args.input)),
        "record_count": len(records),
        "repeats": args.repeats,
        "results": results,
        "fastest": {
            "name": fastest["name"],
            "seconds": fastest["seconds"],
            "records_per_second": fastest["records_per_second"],
        },
    }
    rendered = json.dumps(summary, indent=2)
    if args.artifact_out is not None:
        args.artifact_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _parse_config(value: str) -> tuple[str, int, int]:
    name, workers, batch_size = value.split(":", maxsplit=2)
    return name, int(workers), int(batch_size)


def _select_records(
    records: Sequence[RawPositionRecord],
    *,
    target_count: int,
) -> list[RawPositionRecord]:
    if target_count <= 0:
        return list(records)
    if not records:
        return []
    if target_count <= len(records):
        return list(records[:target_count])

    expanded: list[RawPositionRecord] = []
    for index in range(target_count):
        template = records[index % len(records)]
        expanded.append(
            RawPositionRecord(
                sample_id=f"{template.sample_id}:bench:{index}",
                fen=template.fen,
                source=template.source,
                selected_move_uci=template.selected_move_uci,
                result=template.result,
                metadata=dict(template.metadata),
            )
        )
    return expanded


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
