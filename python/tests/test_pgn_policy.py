"""Tests for bounded PGN sampling and Stockfish-policy dataset helpers."""

from __future__ import annotations

from pathlib import Path

from train.datasets.pgn_policy import (
    PgnPolicySamplingConfig,
    _normalize_result,
    _select_game_candidates,
    training_split_ratios,
    verification_split_ratios,
)

try:
    import chess.pgn
except ModuleNotFoundError:  # pragma: no cover - import checked in the test
    chess = None


def test_split_ratios_for_pgn_policy_datasets_are_expected() -> None:
    train = training_split_ratios()
    verify = verification_split_ratios()

    assert (train.train, train.validation, train.test) == (0.9, 0.1, 0.0)
    assert (verify.train, verify.validation, verify.test) == (0.0, 0.0, 1.0)


def test_normalize_result_filters_unknown_values() -> None:
    assert _normalize_result("1-0") == "1-0"
    assert _normalize_result("0-1") == "0-1"
    assert _normalize_result("1/2-1/2") == "1/2-1/2"
    assert _normalize_result("*") is None


def test_select_game_candidates_is_bounded_and_deterministic(tmp_path: Path) -> None:
    import pytest

    chess_pgn = pytest.importorskip("chess.pgn")
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(
        """[Event "Sample"]\n[White "Alpha"]\n[Black "Beta"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. c3 Nf6 5. d4 exd4 6. cxd4 Bb4+ 1-0\n""",
        encoding="utf-8",
    )

    with pgn_path.open("r", encoding="utf-8") as handle:
        game = chess_pgn.read_game(handle)

    assert game is not None
    config = PgnPolicySamplingConfig(
        engine_path=Path("/usr/games/stockfish18"),
        max_train_records=4,
        max_verify_records=2,
        min_ply=4,
        max_ply=12,
        samples_per_game=2,
        engine_nodes=100,
    )

    first = _select_game_candidates(
        game,
        source_name="sample",
        game_index=1,
        config=config,
    )
    second = _select_game_candidates(
        game,
        source_name="sample",
        game_index=1,
        config=config,
    )

    assert len(first) == 2
    assert [candidate.sample_id for candidate in first] == [candidate.sample_id for candidate in second]
    assert all(candidate.metadata["source_pgn"] == "sample" for candidate in first)
    assert all(candidate.result == "1-0" for candidate in first)
