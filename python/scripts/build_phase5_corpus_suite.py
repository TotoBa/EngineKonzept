"""Materialize and validate the current Phase-5 corpus tiers used by later phases."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

from train.datasets import (
    SplitRatios,
    load_raw_records,
    materialize_proposer_artifacts,
    materialize_dynamics_artifacts,
    materialize_symbolic_proposer_artifacts,
    training_split_ratios,
    verification_split_ratios,
)
from train.datasets.oracle import label_records_with_oracle
from train.datasets.schema import DatasetExample, PositionEncoding, TacticalAnnotations, WdlTarget
from train.datasets.splits import assign_splits


@dataclass(frozen=True)
class ExistingDatasetTierSpec:
    name: str
    train_dataset_dir: Path
    verify_dataset_dir: Path


@dataclass(frozen=True)
class RawDatasetTierSpec:
    name: str
    raw_dir: Path
    train_output_dir: Path
    verify_output_dir: Path
    source_name: str
    seed: str


CURRENT_PHASE5_CORPUS_TIERS: dict[str, ExistingDatasetTierSpec | RawDatasetTierSpec] = {
    "pgn_10k": ExistingDatasetTierSpec(
        name="pgn_10k",
        train_dataset_dir=Path("artifacts/datasets/phase5_stockfish_pgn_train_pi_10k_v1"),
        verify_dataset_dir=Path("artifacts/datasets/phase5_stockfish_pgn_verify_pi_10k_v1"),
    ),
    "merged_unique_122k": ExistingDatasetTierSpec(
        name="merged_unique_122k",
        train_dataset_dir=Path("artifacts/datasets/phase5_stockfish_merged_unique_train_v1"),
        verify_dataset_dir=Path("artifacts/datasets/phase5_stockfish_merged_unique_verify_v1"),
    ),
    "unique_pi_400k": RawDatasetTierSpec(
        name="unique_pi_400k",
        raw_dir=Path("artifacts/datasets/phase5_stockfish_unique_pi_400k_v1"),
        train_output_dir=Path("artifacts/datasets/phase5_stockfish_unique_pi_400k_train_v1"),
        verify_output_dir=Path("artifacts/datasets/phase5_stockfish_unique_pi_400k_verify_v1"),
        source_name="stockfish-unique-pgn",
        seed="phase5-stockfish-unique-pi-400k-v1",
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        action="append",
        choices=sorted(CURRENT_PHASE5_CORPUS_TIERS),
        help="corpus tier(s) to materialize; defaults to all known tiers",
    )
    parser.add_argument("--artifact-out", type=Path, required=True)
    parser.add_argument("--oracle-workers", type=int, default=4)
    parser.add_argument("--oracle-batch-size", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=5000)
    args = parser.parse_args(argv)

    tier_names = args.tier or sorted(CURRENT_PHASE5_CORPUS_TIERS)
    manifest = build_phase5_corpus_suite(
        tier_names=tier_names,
        artifact_out=_resolve_repo_path(args.artifact_out),
        oracle_workers=args.oracle_workers,
        oracle_batch_size=args.oracle_batch_size,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_phase5_corpus_suite(
    *,
    tier_names: Sequence[str],
    artifact_out: Path,
    oracle_workers: int,
    oracle_batch_size: int,
    chunk_size: int,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = repo_root or _repo_root()
    manifest: dict[str, Any] = {
        "oracle_workers": oracle_workers,
        "oracle_batch_size": oracle_batch_size,
        "tiers": {},
    }
    for tier_name in tier_names:
        spec = CURRENT_PHASE5_CORPUS_TIERS[tier_name]
        if isinstance(spec, ExistingDatasetTierSpec):
            tier_summary = _validate_existing_tier(spec, repo_root=resolved_root)
        else:
            tier_summary = _build_raw_tier(
                spec,
                repo_root=resolved_root,
                oracle_workers=oracle_workers,
                oracle_batch_size=oracle_batch_size,
                chunk_size=chunk_size,
            )
        manifest["tiers"][tier_name] = tier_summary

    artifact_out.parent.mkdir(parents=True, exist_ok=True)
    artifact_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _validate_existing_tier(
    spec: ExistingDatasetTierSpec,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    train_summary = _validate_dataset_dir(repo_root / spec.train_dataset_dir)
    verify_summary = _validate_dataset_dir(repo_root / spec.verify_dataset_dir)
    return {
        "mode": "existing",
        "train_dataset_dir": str(repo_root / spec.train_dataset_dir),
        "verify_dataset_dir": str(repo_root / spec.verify_dataset_dir),
        "train_dataset": train_summary,
        "verify_dataset": verify_summary,
    }


def _build_raw_tier(
    spec: RawDatasetTierSpec,
    *,
    repo_root: Path,
    oracle_workers: int,
    oracle_batch_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    raw_dir = repo_root / spec.raw_dir
    train_output_dir = repo_root / spec.train_output_dir
    verify_output_dir = repo_root / spec.verify_output_dir

    train_summary = _build_current_dataset_from_raw(
        raw_path=raw_dir / "train_raw.jsonl",
        output_dir=train_output_dir,
        source_name=spec.source_name,
        seed=spec.seed,
        ratios=training_split_ratios(),
        repo_root=repo_root,
        oracle_workers=oracle_workers,
        oracle_batch_size=oracle_batch_size,
        chunk_size=chunk_size,
    )
    verify_summary = _build_current_dataset_from_raw(
        raw_path=raw_dir / "verify_raw.jsonl",
        output_dir=verify_output_dir,
        source_name=spec.source_name,
        seed=spec.seed,
        ratios=verification_split_ratios(),
        repo_root=repo_root,
        oracle_workers=oracle_workers,
        oracle_batch_size=oracle_batch_size,
        chunk_size=chunk_size,
    )

    return {
        "mode": "raw_materialized",
        "raw_dir": str(raw_dir),
        "train_dataset_dir": str(train_output_dir),
        "verify_dataset_dir": str(verify_output_dir),
        "train_dataset": train_summary,
        "verify_dataset": verify_summary,
    }


def _build_current_dataset_from_raw(
    *,
    raw_path: Path,
    output_dir: Path,
    source_name: str,
    seed: str,
    ratios: SplitRatios,
    repo_root: Path,
    oracle_workers: int,
    oracle_batch_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    records = load_raw_records(raw_path, "jsonl", source_name=source_name)
    summary = _stream_write_dataset_artifacts(
        records,
        output_dir=output_dir,
        ratios=ratios,
        seed=seed,
        repo_root=repo_root,
        oracle_workers=oracle_workers,
        oracle_batch_size=oracle_batch_size,
        chunk_size=chunk_size,
    )
    materialize_proposer_artifacts(output_dir)
    symbolic_counts = materialize_symbolic_proposer_artifacts(output_dir)
    dynamics_counts = materialize_dynamics_artifacts(output_dir, repo_root=repo_root)
    return {
        "raw_path": str(raw_path),
        "summary": summary,
        "symbolic_proposer_artifacts": symbolic_counts,
        "dynamics_artifacts": dynamics_counts,
    }


def _validate_dataset_dir(dataset_dir: Path) -> dict[str, Any]:
    summary_path = dataset_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing dataset summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    split_names = sorted(summary.get("split_counts", {}))
    required_files = ["dataset.jsonl", "summary.json"]
    required_files.extend(f"{split}.jsonl" for split in split_names)
    required_files.extend(f"proposer_{split}.jsonl" for split in split_names)
    required_files.extend(f"proposer_symbolic_{split}.jsonl" for split in split_names)
    required_files.extend(f"dynamics_{split}.jsonl" for split in split_names)

    missing = [
        file_name for file_name in required_files if not (dataset_dir / file_name).exists()
    ]
    if missing:
        raise FileNotFoundError(f"{dataset_dir}: missing required artifacts: {missing}")

    return {
        "summary": summary,
        "validated_files": required_files,
    }


def _stream_write_dataset_artifacts(
    records: Sequence[Any],
    *,
    output_dir: Path,
    ratios: SplitRatios,
    seed: str,
    repo_root: Path,
    oracle_workers: int,
    oracle_batch_size: int,
    chunk_size: int,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_assignments = assign_splits(records, ratios=ratios, seed=seed)
    oracle_schedule = _oracle_schedule(
        len(records),
        oracle_workers=oracle_workers,
        oracle_batch_size=oracle_batch_size,
    )
    effective_batch_size = oracle_schedule["effective_batch_size"]
    summary = _DatasetSummaryAccumulator()

    dataset_handle = (output_dir / "dataset.jsonl").open("w", encoding="utf-8")
    split_handles = {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "validation": (output_dir / "validation.jsonl").open("w", encoding="utf-8"),
        "test": (output_dir / "test.jsonl").open("w", encoding="utf-8"),
    }
    try:
        for chunk_start in range(0, len(records), chunk_size):
            chunk_records = list(records[chunk_start : chunk_start + chunk_size])
            chunk_splits = split_assignments[chunk_start : chunk_start + chunk_size]
            chunk_outputs = _label_records_chunked(
                chunk_records,
                repo_root=repo_root,
                oracle_workers=oracle_workers,
                oracle_batch_size=effective_batch_size,
            )
            for record, split_name, oracle_payload in zip(
                chunk_records,
                chunk_splits,
                chunk_outputs,
                strict=True,
            ):
                example = _dataset_example_from_payload(
                    record,
                    split_name,
                    oracle_payload,
                )
                line = json.dumps(example.to_dict(), sort_keys=True) + "\n"
                dataset_handle.write(line)
                split_handles[split_name].write(line)
                summary.add(example)
    finally:
        dataset_handle.close()
        for handle in split_handles.values():
            handle.close()

    payload = summary.finalize()
    payload["oracle_schedule"] = oracle_schedule
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _label_records_chunked(
    records: Sequence[Any],
    *,
    repo_root: Path,
    oracle_workers: int,
    oracle_batch_size: int,
) -> list[dict[str, object]]:
    if not records:
        return []
    if oracle_workers <= 1 or oracle_batch_size >= len(records):
        return label_records_with_oracle(records, repo_root=repo_root)

    batches = [
        list(records[index : index + oracle_batch_size])
        for index in range(0, len(records), oracle_batch_size)
    ]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=oracle_workers) as executor:
        batch_outputs = list(
            executor.map(
                lambda batch: label_records_with_oracle(batch, repo_root=repo_root),
                batches,
            )
        )
    return [output for batch in batch_outputs for output in batch]


def _dataset_example_from_payload(
    record: Any,
    split_name: str,
    oracle_payload: dict[str, object],
) -> DatasetExample:
    annotations = TacticalAnnotations.from_oracle_dict(oracle_payload["annotations"])
    position_encoding = PositionEncoding.from_oracle_dict(oracle_payload["position_encoding"])
    selected_action = oracle_payload["selected_action_encoding"]
    return DatasetExample(
        sample_id=record.sample_id,
        split=split_name,
        source=record.source,
        fen=record.fen,
        side_to_move=str(oracle_payload["side_to_move"]),
        selected_move_uci=record.selected_move_uci,
        selected_action_encoding=None
        if selected_action is None
        else [int(value) for value in selected_action],
        next_fen=oracle_payload["next_fen"],
        legal_moves=[str(move) for move in oracle_payload["legal_moves"]],
        legal_action_encodings=[
            [int(value) for value in action]
            for action in oracle_payload["legal_action_encodings"]
        ],
        position_encoding=position_encoding,
        wdl_target=_derive_wdl_target(
            result=record.result,
            side_to_move=str(oracle_payload["side_to_move"]),
            annotations=annotations,
        ),
        annotations=annotations,
        result=record.result,
        metadata=record.metadata,
    )


@dataclass
class _DatasetSummaryAccumulator:
    total_examples: int = 0
    selected_move_count: int = 0
    next_state_count: int = 0
    split_counts: Counter[str] = field(default_factory=Counter)
    source_counts: Counter[str] = field(default_factory=Counter)
    wdl_counts: Counter[str] = field(default_factory=Counter)
    fen_set: set[str] = field(default_factory=set)
    legal_move_min: int | None = None
    legal_move_max: int = 0
    legal_move_total: int = 0
    piece_count_min: int | None = None
    piece_count_max: int = 0
    piece_count_total: int = 0
    annotation_counts: Counter[str] = field(default_factory=Counter)

    def add(self, example: DatasetExample) -> None:
        self.total_examples += 1
        self.selected_move_count += int(example.selected_move_uci is not None)
        self.next_state_count += int(example.next_fen is not None)
        self.split_counts[example.split] += 1
        self.source_counts[example.source] += 1
        self.wdl_counts[
            "missing" if example.wdl_target is None else _wdl_label(example.wdl_target)
        ] += 1
        self.fen_set.add(example.fen)

        legal_count = int(example.annotations.legal_move_count)
        piece_count = int(example.annotations.piece_count)
        self.legal_move_total += legal_count
        self.piece_count_total += piece_count
        self.legal_move_max = max(self.legal_move_max, legal_count)
        self.piece_count_max = max(self.piece_count_max, piece_count)
        self.legal_move_min = legal_count if self.legal_move_min is None else min(
            self.legal_move_min,
            legal_count,
        )
        self.piece_count_min = piece_count if self.piece_count_min is None else min(
            self.piece_count_min,
            piece_count,
        )

        self.annotation_counts["in_check"] += int(example.annotations.in_check)
        self.annotation_counts["checkmate"] += int(example.annotations.is_checkmate)
        self.annotation_counts["stalemate"] += int(example.annotations.is_stalemate)
        self.annotation_counts["has_legal_en_passant"] += int(
            example.annotations.has_legal_en_passant
        )
        self.annotation_counts["has_legal_castle"] += int(example.annotations.has_legal_castle)
        self.annotation_counts["has_legal_promotion"] += int(
            example.annotations.has_legal_promotion
        )
        self.annotation_counts["is_low_material_endgame"] += int(
            example.annotations.is_low_material_endgame
        )

    def finalize(self) -> dict[str, Any]:
        total = max(1, self.total_examples)
        return {
            "total_examples": self.total_examples,
            "split_counts": dict(self.split_counts),
            "source_counts": dict(self.source_counts),
            "wdl_counts": dict(self.wdl_counts),
            "selected_move_count": self.selected_move_count,
            "next_state_count": self.next_state_count,
            "unique_fens": len(self.fen_set),
            "annotation_counts": {
                "in_check": self.annotation_counts["in_check"],
                "checkmate": self.annotation_counts["checkmate"],
                "stalemate": self.annotation_counts["stalemate"],
                "has_legal_en_passant": self.annotation_counts["has_legal_en_passant"],
                "has_legal_castle": self.annotation_counts["has_legal_castle"],
                "has_legal_promotion": self.annotation_counts["has_legal_promotion"],
                "is_low_material_endgame": self.annotation_counts["is_low_material_endgame"],
            },
            "legal_move_count": {
                "min": 0 if self.legal_move_min is None else self.legal_move_min,
                "max": self.legal_move_max,
                "mean": round(self.legal_move_total / total, 4),
            },
            "piece_count": {
                "min": 0 if self.piece_count_min is None else self.piece_count_min,
                "max": self.piece_count_max,
                "mean": round(self.piece_count_total / total, 4),
            },
        }


def _oracle_schedule(
    record_count: int,
    *,
    oracle_workers: int,
    oracle_batch_size: int,
) -> dict[str, int]:
    effective_batch_size = oracle_batch_size or _default_oracle_batch_size(
        record_count,
        oracle_workers=oracle_workers,
    )
    batch_count = 0 if record_count == 0 or effective_batch_size == 0 else math.ceil(
        record_count / effective_batch_size
    )
    return {
        "record_count": record_count,
        "oracle_workers": oracle_workers,
        "requested_batch_size": oracle_batch_size,
        "effective_batch_size": effective_batch_size,
        "batch_count": batch_count,
    }


def _default_oracle_batch_size(record_count: int, *, oracle_workers: int) -> int:
    if record_count <= 0:
        return 0
    if oracle_workers <= 1:
        return record_count
    return min(500, max(1, math.ceil(record_count / oracle_workers)))


def _derive_wdl_target(
    *,
    result: str | None,
    side_to_move: str,
    annotations: TacticalAnnotations,
) -> WdlTarget | None:
    if result == "1-0":
        return WdlTarget(win=1, draw=0, loss=0) if side_to_move == "w" else WdlTarget(
            win=0,
            draw=0,
            loss=1,
        )
    if result == "0-1":
        return WdlTarget(win=0, draw=0, loss=1) if side_to_move == "w" else WdlTarget(
            win=1,
            draw=0,
            loss=0,
        )
    if result == "1/2-1/2":
        return WdlTarget(win=0, draw=1, loss=0)
    if annotations.is_checkmate:
        return WdlTarget(win=0, draw=0, loss=1)
    if annotations.is_stalemate:
        return WdlTarget(win=0, draw=1, loss=0)
    return None


def _wdl_label(target: WdlTarget) -> str:
    if target.win:
        return "win"
    if target.draw:
        return "draw"
    return "loss"


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else _repo_root() / path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    raise SystemExit(main())
