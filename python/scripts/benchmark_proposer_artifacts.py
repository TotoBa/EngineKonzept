"""Benchmark proposer loading and artifact size with and without lean split files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import time
from typing import Sequence

from train.config import ProposerTrainConfig
from train.datasets import (
    load_proposer_examples,
    materialize_proposer_artifacts,
)
from train.datasets.schema import SUPPORTED_SPLITS
from train.trainers import train_proposer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-out", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--train-epochs", type=int, default=1)
    args = parser.parse_args(argv)

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.train_epochs <= 0:
        raise ValueError("--train-epochs must be positive")

    dataset_dir = _resolve_path(args.dataset_dir)
    output_root = _resolve_path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    full_dir = output_root / "full"
    lean_dir = output_root / "lean"
    shutil.copytree(dataset_dir, full_dir)
    shutil.copytree(dataset_dir, lean_dir)
    written_counts = materialize_proposer_artifacts(lean_dir)

    full_results = _benchmark_variant(full_dir, repeats=args.repeats)
    lean_results = _benchmark_variant(lean_dir, repeats=args.repeats)
    output = {
        "dataset_dir": str(dataset_dir),
        "runtime": _runtime_metadata(),
        "repeats": args.repeats,
        "written_counts": written_counts,
        "full": full_results,
        "lean": lean_results,
        "speedup": {
            split: round(
                full_results[split]["seconds"] / lean_results[split]["seconds"], 3  # type: ignore[index]
            )
            for split in SUPPORTED_SPLITS
            if float(lean_results[split]["seconds"]) > 0.0  # type: ignore[index]
        },
        "size_bytes": {
            "full": _measure_full_bytes(full_dir),
            "lean": _measure_lean_bytes(lean_dir),
        },
    }
    if args.train_config is not None:
        output["training"] = _benchmark_training(
            config_path=_resolve_path(args.train_config),
            full_dir=full_dir,
            lean_dir=lean_dir,
            output_root=output_root,
            epochs=args.train_epochs,
        )
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.artifact_out is not None:
        args.artifact_out.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def _benchmark_variant(dataset_dir: Path, *, repeats: int) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for split in sorted(SUPPORTED_SPLITS):
        timings: list[float] = []
        example_count = 0
        for _ in range(repeats):
            started = time.perf_counter()
            examples = load_proposer_examples(dataset_dir, split)
            timings.append(time.perf_counter() - started)
            example_count = len(examples)
        average = sum(timings) / len(timings)
        results[split] = {
            "example_count": example_count,
            "seconds": round(average, 6),
            "repeat_seconds": [round(value, 6) for value in timings],
            "examples_per_second": round(example_count / average, 3) if average > 0.0 else 0.0,
        }
    return results


def _measure_full_bytes(dataset_dir: Path) -> dict[str, int]:
    return {
        split: (dataset_dir / f"{split}.jsonl").stat().st_size
        for split in sorted(SUPPORTED_SPLITS)
        if (dataset_dir / f"{split}.jsonl").exists()
    }


def _measure_lean_bytes(dataset_dir: Path) -> dict[str, int]:
    return {
        split: (dataset_dir / f"proposer_{split}.jsonl").stat().st_size
        for split in sorted(SUPPORTED_SPLITS)
        if (dataset_dir / f"proposer_{split}.jsonl").exists()
    }


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _benchmark_training(
    *,
    config_path: Path,
    full_dir: Path,
    lean_dir: Path,
    output_root: Path,
    epochs: int,
) -> dict[str, dict[str, object]]:
    base_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(base_payload, dict):
        raise ValueError(f"{config_path}: training config root must be an object")

    results: dict[str, dict[str, object]] = {}
    for name, dataset_dir in (("full", full_dir), ("lean", lean_dir)):
        payload = json.loads(json.dumps(base_payload))
        payload["output_dir"] = str(output_root / f"train_{name}")
        payload["data"]["dataset_path"] = str(dataset_dir)
        payload["optimization"]["epochs"] = epochs
        payload["export"]["bundle_dir"] = str(output_root / f"bundle_{name}")
        config = ProposerTrainConfig.from_dict(payload)

        started = time.perf_counter()
        run = train_proposer(config, repo_root=Path(__file__).resolve().parents[2])
        elapsed = time.perf_counter() - started
        first_epoch = run.history[0]
        results[name] = {
            "seconds": round(elapsed, 6),
            "train_examples_per_second": first_epoch["train"]["examples_per_second"],
            "validation_examples_per_second": first_epoch["validation"]["examples_per_second"],
            "best_validation": run.best_validation,
        }

    results["speedup"] = {
        "seconds": round(
            float(results["full"]["seconds"]) / float(results["lean"]["seconds"]), 3
        ),
        "train_examples_per_second": round(
            float(results["lean"]["train_examples_per_second"])
            / float(results["full"]["train_examples_per_second"]),
            3,
        ),
    }
    return results


def _runtime_metadata() -> dict[str, object]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
