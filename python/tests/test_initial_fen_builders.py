from __future__ import annotations

from pathlib import Path

from train.eval.initial_fens import (
    SelfplayInitialFenEntry,
    SelfplayInitialFenSuite,
    build_opening_initial_fen_suite,
    merge_selfplay_initial_fen_suites,
)


def test_build_opening_initial_fen_suite_selects_positions_per_file(tmp_path: Path) -> None:
    openings_path = tmp_path / "openings.tsv"
    openings_path.write_text(
        "eco\tname\tpgn\n"
        "A00\tOpening A\t1. e4 e5 2. Nf3 Nc6\n"
        "A01\tOpening B\t1. d4 d5 2. c4 e6\n"
        "A02\tOpening C\t1. c4 e5 2. Nc3 Nf6\n",
        encoding="utf-8",
    )

    suite = build_opening_initial_fen_suite(
        name="openings_v1",
        tsv_paths=[openings_path],
        entries_per_file=2,
        min_ply_count=4,
    )

    assert suite.name == "openings_v1"
    assert len(suite.entries) == 2
    assert all(entry.tier == "thor_openings" for entry in suite.entries)
    assert all("opening" in entry.tags for entry in suite.entries)
    assert suite.metadata["entries_per_file"] == 2


def test_merge_selfplay_initial_fen_suites_dedupes_by_fen() -> None:
    duplicate_fen = "8/8/8/8/8/8/8/K6k w - - 0 1"
    suite_a = SelfplayInitialFenSuite(
        name="suite_a",
        entries=[
            SelfplayInitialFenEntry(
                fen=duplicate_fen,
                tier="a",
                sample_id="a:1",
                source_path="a.jsonl",
                result="*",
                selection_score=1.0,
            )
        ],
    )
    suite_b = SelfplayInitialFenSuite(
        name="suite_b",
        entries=[
            SelfplayInitialFenEntry(
                fen=duplicate_fen,
                tier="b",
                sample_id="b:1",
                source_path="b.jsonl",
                result="*",
                selection_score=2.0,
            ),
            SelfplayInitialFenEntry(
                fen="8/8/8/8/8/8/7k/K7 w - - 0 1",
                tier="b",
                sample_id="b:2",
                source_path="b.jsonl",
                result="*",
                selection_score=3.0,
            ),
        ],
    )

    merged = merge_selfplay_initial_fen_suites(
        name="merged_v1",
        suites=[suite_a, suite_b],
    )

    assert merged.name == "merged_v1"
    assert [entry.sample_id for entry in merged.entries] == ["a:1", "b:2"]
    assert merged.metadata["source_suite_names"] == ["suite_a", "suite_b"]
