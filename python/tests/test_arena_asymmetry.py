from __future__ import annotations

import json
from pathlib import Path

from train.eval.arena_asymmetry import build_selfplay_arena_asymmetry_report


def test_build_selfplay_arena_asymmetry_report_flags_color_skew(tmp_path: Path) -> None:
    session_white_black = tmp_path / "a_vs_b.json"
    session_black_white = tmp_path / "b_vs_a.json"
    session_white_black.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "white_agent": "agent_a",
                        "black_agent": "agent_b",
                        "result": "0-1",
                        "termination_reason": "checkmate",
                    },
                    {
                        "white_agent": "agent_a",
                        "black_agent": "agent_b",
                        "result": "0-1",
                        "termination_reason": "checkmate",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    session_black_white.write_text(
        json.dumps(
            {
                "games": [
                    {
                        "white_agent": "agent_b",
                        "black_agent": "agent_a",
                        "result": "0-1",
                        "termination_reason": "checkmate",
                    },
                    {
                        "white_agent": "agent_b",
                        "black_agent": "agent_a",
                        "result": "0-1",
                        "termination_reason": "checkmate",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = {
        "arena_name": "test_arena",
        "arena_spec_version": 1,
        "matchups": [
            {
                "white_agent": "agent_a",
                "black_agent": "agent_b",
                "session_path": str(session_white_black),
            },
            {
                "white_agent": "agent_b",
                "black_agent": "agent_a",
                "session_path": str(session_black_white),
            },
        ],
    }

    report = build_selfplay_arena_asymmetry_report(summary)

    assert report["overall"]["games"] == 4
    assert report["overall"]["white_wins"] == 0
    assert report["overall"]["black_wins"] == 4
    assert report["suspicious_pairs"]
    pair = report["suspicious_pairs"][0]
    assert pair["agent_a"] == "agent_a"
    assert pair["agent_b"] == "agent_b"
    assert pair["agent_a_color_score_delta"] == -1.0
