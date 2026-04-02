"""Build bounded Stockfish-labeled policy datasets from PGN files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from train.datasets import build_dataset, write_dataset_artifacts
from train.datasets.pgn_policy import (
    PgnPolicySamplingConfig,
    sample_policy_records_from_pgns,
    training_split_ratios,
    verification_split_ratios,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Sample PGNs, label positions with Stockfish, and emit train/verify datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, nargs="+", required=True, help="PGN files to scan")
    parser.add_argument("--engine-path", type=Path, default=Path("/usr/games/stockfish18"))
    parser.add_argument("--train-output-dir", type=Path, required=True)
    parser.add_argument("--verify-output-dir", type=Path, required=True)
    parser.add_argument("--raw-output-dir", type=Path)
    parser.add_argument("--max-train-records", type=int, default=128)
    parser.add_argument("--max-verify-records", type=int, default=32)
    parser.add_argument("--min-ply", type=int, default=8)
    parser.add_argument("--max-ply", type=int, default=80)
    parser.add_argument("--samples-per-game", type=int, default=2)
    parser.add_argument("--engine-nodes", type=int, default=2000)
    parser.add_argument("--hash-mb", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--split-seed", default="phase5-stockfish-pgn-v1")
    parser.add_argument("--oracle-workers", type=int, default=1)
    parser.add_argument("--oracle-batch-size", type=int, default=0)
    parser.add_argument(
        "--write-proposer-artifacts",
        action="store_true",
        dest="write_proposer_artifacts",
        help="emit proposer_<split>.jsonl files with packed training features",
    )
    parser.add_argument(
        "--no-proposer-artifacts",
        action="store_false",
        dest="write_proposer_artifacts",
        help="skip proposer_<split>.jsonl output even for larger Phase-5 dataset builds",
    )
    parser.set_defaults(write_proposer_artifacts=True)
    args = parser.parse_args(argv)

    selection = sample_policy_records_from_pgns(
        args.pgn,
        config=PgnPolicySamplingConfig(
            engine_path=args.engine_path,
            max_train_records=args.max_train_records,
            max_verify_records=args.max_verify_records,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
            samples_per_game=args.samples_per_game,
            engine_nodes=args.engine_nodes,
            hash_mb=args.hash_mb,
            threads=args.threads,
            split_seed=args.split_seed,
        ),
    )

    train_dataset = build_dataset(
        selection.train_records,
        ratios=training_split_ratios(),
        seed=f"{args.split_seed}:train",
        repo_root=_repo_root(),
        oracle_workers=args.oracle_workers,
        oracle_batch_size=args.oracle_batch_size,
    )
    verify_dataset = build_dataset(
        selection.verify_records,
        ratios=verification_split_ratios(),
        seed=f"{args.split_seed}:verify",
        repo_root=_repo_root(),
        oracle_workers=args.oracle_workers,
        oracle_batch_size=args.oracle_batch_size,
    )

    write_dataset_artifacts(
        args.train_output_dir,
        train_dataset,
        write_proposer_artifacts=args.write_proposer_artifacts,
    )
    write_dataset_artifacts(
        args.verify_output_dir,
        verify_dataset,
        write_proposer_artifacts=args.write_proposer_artifacts,
    )

    if args.raw_output_dir is not None:
        args.raw_output_dir.mkdir(parents=True, exist_ok=True)
        _write_raw_jsonl(args.raw_output_dir / "train_raw.jsonl", selection.train_records)
        _write_raw_jsonl(args.raw_output_dir / "verify_raw.jsonl", selection.verify_records)
        (args.raw_output_dir / "selection_summary.json").write_text(
            json.dumps(selection.summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "selection": selection.summary,
                "train_summary": train_dataset.summary,
                "verify_summary": verify_dataset.summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _write_raw_jsonl(path: Path, records: Sequence[object]) -> None:
    path.write_text(
        "\n".join(json.dumps(_record_to_dict(record), sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _record_to_dict(record: object) -> dict[str, object]:
    payload = getattr(record, "__dict__", None)
    if not isinstance(payload, dict):
        raise TypeError(f"cannot serialize raw record of type {type(record)!r}")
    return dict(payload)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
