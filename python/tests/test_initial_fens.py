from __future__ import annotations

import json
from pathlib import Path

from scripts.build_selfplay_initial_fen_suite import main as build_initial_fen_suite_main
from train.eval.initial_fens import load_selfplay_initial_fen_suite


def _write_row(
    path: Path,
    *,
    sample_id: str,
    fen: str,
    result: str,
    legal_move_count: int,
    in_check: bool = False,
    gives_check: bool = False,
    capture: bool = False,
) -> None:
    row = {
        "sample_id": sample_id,
        "fen": fen,
        "side_to_move": "w",
        "result": result,
        "annotations": {
            "is_checkmate": False,
            "is_stalemate": False,
            "legal_move_count": legal_move_count,
            "piece_count": 20,
            "in_check": in_check,
            "selected_move_gives_check": gives_check,
            "selected_move_is_capture": capture,
            "has_legal_promotion": False,
            "selected_move_is_promotion": False,
            "has_legal_castle": False,
            "selected_move_is_castle": False,
            "has_legal_en_passant": False,
            "selected_move_is_en_passant": False,
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_initial_fen_suite_round_trip(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "spec_version": 1,
                "name": "suite",
                "entries": [
                    {
                        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
                        "tier": "pgn_10k",
                        "sample_id": "sample_1",
                        "source_path": "dataset.jsonl",
                        "result": "1-0",
                        "selection_score": 3.0,
                        "tags": ["decisive"],
                        "metadata": {},
                    }
                ],
                "metadata": {"tier_summaries": []},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    suite = load_selfplay_initial_fen_suite(suite_path)
    assert suite.name == "suite"
    assert suite.fen_list() == ["8/8/8/8/8/8/8/K6k w - - 0 1"]


def test_build_initial_fen_suite_prefers_more_decisive_rows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    _write_row(
        dataset_path,
        sample_id="quiet_draw",
        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        result="1/2-1/2",
        legal_move_count=14,
    )
    _write_row(
        dataset_path,
        sample_id="capture_win",
        fen="8/8/8/8/8/8/8/K5qk w - - 0 1",
        result="1-0",
        legal_move_count=10,
        capture=True,
    )
    _write_row(
        dataset_path,
        sample_id="check_win",
        fen="8/8/8/8/8/8/8/K5rk w - - 0 1",
        result="1-0",
        legal_move_count=8,
        in_check=True,
        gives_check=True,
    )

    output_path = tmp_path / "suite.json"
    summary_path = tmp_path / "summary.json"
    exit_code = build_initial_fen_suite_main(
        [
            "--name",
            "suite",
            "--dataset",
            f"tier_a={dataset_path}",
            "--per-tier",
            "1",
            "--output-path",
            str(output_path),
            "--summary-path",
            str(summary_path),
        ]
    )
    assert exit_code == 0
    suite = load_selfplay_initial_fen_suite(output_path)
    assert len(suite.entries) == 1
    assert suite.entries[0].sample_id == "check_win"
