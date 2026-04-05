"""Merge multiple Phase-5 raw corpora into one FEN-unique train/verify corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from train.datasets import RawPositionRecord, load_raw_records


@dataclass(frozen=True)
class RawCorpusSourceSpec:
    """One raw Phase-5 corpus source directory."""

    name: str
    raw_dir: Path

    @property
    def train_raw_path(self) -> Path:
        return self.raw_dir / "train_raw.jsonl"

    @property
    def verify_raw_path(self) -> Path:
        return self.raw_dir / "verify_raw.jsonl"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        action="append",
        dest="source_dirs",
        type=Path,
        required=True,
        help="Raw corpus directory containing train_raw.jsonl and verify_raw.jsonl. "
        "Source order defines replacement priority; later sources win on duplicates.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    source_specs = [
        RawCorpusSourceSpec(name=source_dir.name, raw_dir=_resolve_repo_path(source_dir))
        for source_dir in args.source_dirs
    ]
    output_dir = _resolve_repo_path(args.output_dir)
    summary = merge_phase5_raw_corpora(
        source_specs=source_specs,
        output_dir=output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def merge_phase5_raw_corpora(
    *,
    source_specs: Sequence[RawCorpusSourceSpec],
    output_dir: Path,
) -> dict[str, Any]:
    """Merge raw corpora with verify-over-train precedence and later-source replacement."""
    if not source_specs:
        raise ValueError("merge requires at least one source spec")

    output_dir.mkdir(parents=True, exist_ok=True)

    verify_records_by_fen: dict[str, RawPositionRecord] = {}
    train_records_by_fen: dict[str, RawPositionRecord] = {}
    verify_priority_by_fen: dict[str, int] = {}
    train_priority_by_fen: dict[str, int] = {}

    collisions_train_to_verify = 0
    skipped_train_due_to_verify = 0
    replaced_verify_duplicates = 0
    replaced_train_duplicates = 0

    source_summary: dict[str, dict[str, int]] = {}

    for priority, spec in enumerate(source_specs):
        train_records = load_raw_records(
            spec.train_raw_path,
            "jsonl",
            source_name=spec.name,
        )
        verify_records = load_raw_records(
            spec.verify_raw_path,
            "jsonl",
            source_name=spec.name,
        )
        source_summary[spec.name] = {
            "priority": priority,
            "train_input": len(train_records),
            "verify_input": len(verify_records),
        }

        for record in verify_records:
            existing_priority = verify_priority_by_fen.get(record.fen)
            if existing_priority is None:
                verify_records_by_fen[record.fen] = record
                verify_priority_by_fen[record.fen] = priority
            elif priority >= existing_priority:
                verify_records_by_fen[record.fen] = record
                verify_priority_by_fen[record.fen] = priority
                replaced_verify_duplicates += 1

            if record.fen in train_records_by_fen:
                del train_records_by_fen[record.fen]
                del train_priority_by_fen[record.fen]
                collisions_train_to_verify += 1

        for record in train_records:
            if record.fen in verify_records_by_fen:
                skipped_train_due_to_verify += 1
                continue

            existing_priority = train_priority_by_fen.get(record.fen)
            if existing_priority is None:
                train_records_by_fen[record.fen] = record
                train_priority_by_fen[record.fen] = priority
            elif priority >= existing_priority:
                train_records_by_fen[record.fen] = record
                train_priority_by_fen[record.fen] = priority
                replaced_train_duplicates += 1

    train_records = list(train_records_by_fen.values())
    verify_records = list(verify_records_by_fen.values())

    train_raw_path = output_dir / "train_raw.jsonl"
    verify_raw_path = output_dir / "verify_raw.jsonl"
    _write_raw_records(train_raw_path, train_records)
    _write_raw_records(verify_raw_path, verify_records)

    selection_summary = {
        "source_order": [spec.name for spec in source_specs],
        "sources": source_summary,
        "collisions_train_to_verify": collisions_train_to_verify,
        "skipped_train_due_to_verify": skipped_train_due_to_verify,
        "replaced_verify_duplicates": replaced_verify_duplicates,
        "replaced_train_duplicates": replaced_train_duplicates,
        "train_records": len(train_records),
        "verify_records": len(verify_records),
        "train_unique_fens": len(train_records_by_fen),
        "verify_unique_fens": len(verify_records_by_fen),
        "verify_train_overlap": len(set(train_records_by_fen) & set(verify_records_by_fen)),
        "train_source_counts": _source_counts(train_records),
        "verify_source_counts": _source_counts(verify_records),
        "train_raw_path": str(train_raw_path),
        "verify_raw_path": str(verify_raw_path),
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selection_summary


def _write_raw_records(path: Path, records: Sequence[RawPositionRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "sample_id": record.sample_id,
                "fen": record.fen,
                "source": record.source,
                "selected_move_uci": record.selected_move_uci,
                "result": record.result,
                "metadata": record.metadata,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _source_counts(records: Sequence[RawPositionRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record.source] = counts.get(record.source, 0) + 1
    return counts


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


if __name__ == "__main__":
    raise SystemExit(main())
