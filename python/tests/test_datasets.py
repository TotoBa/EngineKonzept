"""Tests for the Phase-4 dataset pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from train.datasets import (
    RawPositionRecord,
    SplitRatios,
    build_dataset,
    load_raw_records,
    write_dataset_artifacts,
)
from train.datasets.splits import assign_splits


def test_load_raw_records_supports_edge_cases_and_epd(tmp_path: Path) -> None:
    edge_cases = tmp_path / "edge_cases.txt"
    edge_cases.write_text("castle|r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1\n", encoding="utf-8")
    edge_records = load_raw_records(edge_cases, "edge-cases")

    assert len(edge_records) == 1
    assert edge_records[0].sample_id == "edge_cases:castle"
    assert edge_records[0].metadata["name"] == "castle"

    epd = tmp_path / "suite.epd"
    epd.write_text("4k3/8/8/8/8/8/8/4K3 w - - ; id 'quiet';\n", encoding="utf-8")
    epd_records = load_raw_records(epd, "epd", source_name="perftsuite")

    assert len(epd_records) == 1
    assert epd_records[0].sample_id == "perftsuite:1"
    assert epd_records[0].fen == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


def test_load_raw_records_supports_jsonl_selected_move_labels(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "policy_seed.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "sample_id": "policy-seed:startpos-e2e4",
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "source": "policy-seed",
                "selected_move_uci": "e2e4",
                "metadata": {"theme": "opening"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_raw_records(jsonl_path, "jsonl")

    assert len(records) == 1
    assert records[0].sample_id == "policy-seed:startpos-e2e4"
    assert records[0].selected_move_uci == "e2e4"
    assert records[0].metadata["theme"] == "opening"


def test_assign_splits_is_deterministic() -> None:
    records = [
        RawPositionRecord(
            sample_id=f"sample-{index}",
            fen="4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            source="synthetic",
        )
        for index in range(5)
    ]

    first = assign_splits(records, ratios=SplitRatios(0.6, 0.2, 0.2), seed="alpha")
    second = assign_splits(records, ratios=SplitRatios(0.6, 0.2, 0.2), seed="alpha")

    assert first == second
    assert first.count("train") == 3
    assert first.count("validation") == 1
    assert first.count("test") == 1


def test_build_dataset_uses_exact_rule_oracle_and_writes_artifacts(tmp_path: Path) -> None:
    records = [
        RawPositionRecord(
            sample_id="startpos:e2e4",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            source="jsonl",
            selected_move_uci="e2e4",
            result="1-0",
        ),
        RawPositionRecord(
            sample_id="mate:black",
            fen="7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
            source="jsonl",
        ),
    ]

    dataset = build_dataset(
        records,
        ratios=SplitRatios(0.5, 0.0, 0.5),
        seed="phase4-test",
        repo_root=_repo_root(),
    )

    assert len(dataset.examples) == 2
    first, second = dataset.examples
    assert first.selected_action_encoding == [12, 28, 0]
    assert first.next_fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert first.wdl_target is not None and first.wdl_target.win == 1

    assert second.annotations.is_checkmate
    assert second.wdl_target is not None and second.wdl_target.loss == 1
    assert dataset.summary["annotation_counts"]["checkmate"] == 1

    output_dir = tmp_path / "dataset"
    write_dataset_artifacts(output_dir, dataset)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_examples"] == 2
    assert (output_dir / "dataset.jsonl").exists()
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]
