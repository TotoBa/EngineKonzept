from __future__ import annotations

from train.eval.matrix import build_selfplay_arena_matrix


def test_build_selfplay_arena_matrix_aggregates_both_directions() -> None:
    matrix = build_selfplay_arena_matrix(
        {
            "arena_name": "arena",
            "arena_spec_version": 1,
            "aggregate": {"game_count": 4, "matchup_count": 2},
            "standings": {
                "a": {"games": 4, "wins": 2, "draws": 2, "losses": 0, "unfinished": 0, "score": 3.0},
                "b": {"games": 4, "wins": 0, "draws": 2, "losses": 2, "unfinished": 0, "score": 1.0},
            },
            "matchups": [
                {
                    "white_agent": "a",
                    "black_agent": "b",
                    "game_count": 2,
                    "white_score": 1.5,
                    "black_score": 0.5,
                    "result_counts": {"1-0": 1, "1/2-1/2": 1},
                    "termination_counts": {"checkmate": 1, "threefold_repetition": 1},
                },
                {
                    "white_agent": "b",
                    "black_agent": "a",
                    "game_count": 2,
                    "white_score": 0.5,
                    "black_score": 1.5,
                    "result_counts": {"0-1": 1, "1/2-1/2": 1},
                    "termination_counts": {"checkmate": 1, "threefold_repetition": 1},
                },
            ],
        }
    )

    assert matrix["agent_names"] == ["a", "b"]
    assert matrix["ranking_by_score_rate"][0]["agent"] == "a"

    a_vs_b = matrix["matrix"]["a"]["b"]
    assert a_vs_b["game_count"] == 4
    assert a_vs_b["score"] == 3.0
    assert a_vs_b["score_rate"] == 0.75
    assert a_vs_b["wins"] == 2
    assert a_vs_b["draws"] == 2
    assert a_vs_b["losses"] == 0
    assert a_vs_b["termination_counts"] == {"checkmate": 2, "threefold_repetition": 2}

    b_vs_a = matrix["matrix"]["b"]["a"]
    assert b_vs_a["game_count"] == 4
    assert b_vs_a["score"] == 1.0
    assert b_vs_a["score_rate"] == 0.25
    assert b_vs_a["wins"] == 0
    assert b_vs_a["draws"] == 2
    assert b_vs_a["losses"] == 2
