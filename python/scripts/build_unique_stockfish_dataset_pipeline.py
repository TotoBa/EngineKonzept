"""Run the unique PGN/Stockfish corpus builder and materialize current dataset artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from train.datasets import (  # noqa: E402
    SplitRatios,
    build_dataset,
    load_raw_records,
    materialize_dynamics_artifacts,
    materialize_symbolic_proposer_artifacts,
    training_split_ratios,
    verification_split_ratios,
)
from train.datasets.io import write_dataset_artifacts  # noqa: E402

_UNIQUE_CORPUS_SCRIPT_PATH = Path(__file__).with_name("build_unique_stockfish_pgn_corpus.py")
_UNIQUE_CORPUS_SPEC = importlib.util.spec_from_file_location(
    "build_unique_stockfish_pgn_corpus",
    _UNIQUE_CORPUS_SCRIPT_PATH,
)
assert _UNIQUE_CORPUS_SPEC is not None and _UNIQUE_CORPUS_SPEC.loader is not None
_UNIQUE_CORPUS_MODULE = importlib.util.module_from_spec(_UNIQUE_CORPUS_SPEC)
sys.modules[_UNIQUE_CORPUS_SPEC.name] = _UNIQUE_CORPUS_MODULE
_UNIQUE_CORPUS_SPEC.loader.exec_module(_UNIQUE_CORPUS_MODULE)

UniqueCorpusConfig = _UNIQUE_CORPUS_MODULE.UniqueCorpusConfig
build_unique_corpus_from_pgns = _UNIQUE_CORPUS_MODULE.build_unique_corpus_from_pgns
export_unique_corpus_snapshot = _UNIQUE_CORPUS_MODULE.export_unique_corpus_snapshot


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--train-output-dir", type=Path)
    parser.add_argument("--verify-output-dir", type=Path)
    parser.add_argument("--snapshot-only", action="store_true")
    parser.add_argument("--source-name", default="stockfish-unique-pgn")
    parser.add_argument("--seed", default="phase5-stockfish-unique-current-v1")
    parser.add_argument("--oracle-workers", type=int, default=1)
    parser.add_argument("--oracle-batch-size", type=int, default=0)
    parser.add_argument("--skip-symbolic-proposer-artifacts", action="store_true")
    parser.add_argument("--skip-dynamics-artifacts", action="store_true")

    parser.add_argument("--pgn-root", type=Path)
    parser.add_argument("--glob", default="**/*.pgn")
    parser.add_argument("--engine-path", type=Path, default=Path("/usr/games/stockfish18"))
    parser.add_argument("--target-train-records", type=int, default=1_000_000)
    parser.add_argument("--target-verify-records", type=int, default=1_000)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=80)
    parser.add_argument("--ply-stride", type=int, default=2)
    parser.add_argument("--engine-nodes", type=int, default=1500)
    parser.add_argument("--hash-mb", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--split-seed", default="phase5-stockfish-unique-v1")
    parser.add_argument("--verify-divisor", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--max-games", type=int, default=0)
    args = parser.parse_args(argv)

    work_dir = _resolve_repo_path(args.work_dir)
    train_output_dir = _resolve_repo_path(args.train_output_dir) if args.train_output_dir else work_dir / "current_train_dataset"
    verify_output_dir = _resolve_repo_path(args.verify_output_dir) if args.verify_output_dir else work_dir / "current_verify_dataset"

    build_summary: dict[str, Any] | None = None
    if args.snapshot_only:
        if args.pgn_root is not None:
            raise ValueError("--pgn-root cannot be used together with --snapshot-only")
    else:
        if args.pgn_root is None:
            raise ValueError("--pgn-root is required unless --snapshot-only is set")
        pgn_root = _resolve_repo_path(args.pgn_root)
        pgn_paths = sorted(path for path in pgn_root.glob(args.glob) if path.is_file())
        if not pgn_paths:
            raise ValueError(f"no PGNs matched {args.glob!r} under {pgn_root}")
        build_summary = build_unique_corpus_from_pgns(
            pgn_paths,
            config=UniqueCorpusConfig(
                engine_path=_resolve_repo_path(args.engine_path),
                work_dir=work_dir,
                target_train_records=args.target_train_records,
                target_verify_records=args.target_verify_records,
                min_ply=args.min_ply,
                max_ply=args.max_ply,
                ply_stride=args.ply_stride,
                engine_nodes=args.engine_nodes,
                hash_mb=args.hash_mb,
                threads=args.threads,
                split_seed=args.split_seed,
                verify_divisor=args.verify_divisor,
                progress_every=args.progress_every,
                max_games=args.max_games,
                export_jsonl_on_complete=True,
            ),
        )

    export_summary = export_unique_corpus_snapshot(work_dir)
    artifact_summary = _materialize_current_datasets(
        work_dir=work_dir,
        train_output_dir=train_output_dir,
        verify_output_dir=verify_output_dir,
        source_name=args.source_name,
        seed=args.seed,
        oracle_workers=args.oracle_workers,
        oracle_batch_size=args.oracle_batch_size,
        write_symbolic_proposer_artifacts=not args.skip_symbolic_proposer_artifacts,
        write_dynamics_artifacts=not args.skip_dynamics_artifacts,
    )

    summary = {
        "work_dir": str(work_dir),
        "snapshot_only": args.snapshot_only,
        "build_summary": build_summary,
        "export_summary": export_summary,
        "artifact_summary": artifact_summary,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _materialize_current_datasets(
    *,
    work_dir: Path,
    train_output_dir: Path,
    verify_output_dir: Path,
    source_name: str,
    seed: str,
    oracle_workers: int,
    oracle_batch_size: int,
    write_symbolic_proposer_artifacts: bool,
    write_dynamics_artifacts: bool,
) -> dict[str, Any]:
    train_raw_path = work_dir / "train_raw.jsonl"
    verify_raw_path = work_dir / "verify_raw.jsonl"
    if not train_raw_path.exists():
        raise FileNotFoundError(f"train raw export not found: {train_raw_path}")
    if not verify_raw_path.exists():
        raise FileNotFoundError(f"verify raw export not found: {verify_raw_path}")

    train_summary = _build_current_dataset(
        raw_path=train_raw_path,
        output_dir=train_output_dir,
        source_name=source_name,
        seed=seed,
        ratios=training_split_ratios(),
        oracle_workers=oracle_workers,
        oracle_batch_size=oracle_batch_size,
        write_symbolic_proposer_artifacts=write_symbolic_proposer_artifacts,
        write_dynamics_artifacts=write_dynamics_artifacts,
    )
    verify_summary = _build_current_dataset(
        raw_path=verify_raw_path,
        output_dir=verify_output_dir,
        source_name=source_name,
        seed=seed,
        ratios=verification_split_ratios(),
        oracle_workers=oracle_workers,
        oracle_batch_size=oracle_batch_size,
        write_symbolic_proposer_artifacts=write_symbolic_proposer_artifacts,
        write_dynamics_artifacts=write_dynamics_artifacts,
    )
    return {
        "train_dataset": train_summary,
        "verify_dataset": verify_summary,
    }


def _build_current_dataset(
    *,
    raw_path: Path,
    output_dir: Path,
    source_name: str,
    seed: str,
    ratios: SplitRatios,
    oracle_workers: int,
    oracle_batch_size: int,
    write_symbolic_proposer_artifacts: bool,
    write_dynamics_artifacts: bool,
) -> dict[str, Any]:
    records = load_raw_records(raw_path, "jsonl", source_name=source_name)
    dataset = build_dataset(
        records,
        ratios=ratios,
        seed=seed,
        repo_root=_repo_root(),
        oracle_workers=oracle_workers,
        oracle_batch_size=oracle_batch_size,
    )
    write_dataset_artifacts(
        output_dir,
        dataset,
        write_proposer_artifacts=True,
    )

    symbolic_counts: dict[str, int] | None = None
    dynamics_counts: dict[str, int] | None = None
    if write_symbolic_proposer_artifacts:
        symbolic_counts = materialize_symbolic_proposer_artifacts(output_dir)
    if write_dynamics_artifacts:
        dynamics_counts = materialize_dynamics_artifacts(output_dir, repo_root=_repo_root())

    return {
        "raw_path": str(raw_path),
        "output_dir": str(output_dir),
        "summary": dataset.summary,
        "symbolic_proposer_artifacts": symbolic_counts,
        "dynamics_artifacts": dynamics_counts,
    }


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
