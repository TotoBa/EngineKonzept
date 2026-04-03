"""Build matrix-style summaries from selfplay arena outputs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping


def load_selfplay_arena_summary(path: Path) -> dict[str, Any]:
    """Load one arena summary JSON payload."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("arena summary must be a JSON object")
    return dict(payload)


def build_selfplay_arena_matrix(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Build a row-vs-column matrix from an arena summary.

    Each matrix cell is from the row agent's perspective and aggregates
    all available directed sessions against the column agent.
    """

    standings = dict(summary.get("standings") or {})
    agent_names = sorted(
        standings.keys()
        or {
            str(matchup["white_agent"])
            for matchup in list(summary.get("matchups") or [])
        }
        | {
            str(matchup["black_agent"])
            for matchup in list(summary.get("matchups") or [])
        }
    )
    matrix = {
        row_agent: {
            column_agent: _empty_matrix_cell(row_agent, column_agent)
            for column_agent in agent_names
        }
        for row_agent in agent_names
    }

    for raw_matchup in list(summary.get("matchups") or []):
        matchup = dict(raw_matchup)
        white_agent = str(matchup["white_agent"])
        black_agent = str(matchup["black_agent"])
        game_count = int(matchup["game_count"])
        white_score = float(matchup["white_score"])
        black_score = float(matchup["black_score"])
        result_counts = Counter(
            {
                str(result): int(count)
                for result, count in dict(matchup.get("result_counts") or {}).items()
            }
        )
        termination_counts = Counter(
            {
                str(reason): int(count)
                for reason, count in dict(matchup.get("termination_counts") or {}).items()
            }
        )

        _update_cell(
            matrix[white_agent][black_agent],
            game_count=game_count,
            score=white_score,
            wins=result_counts.get("1-0", 0),
            losses=result_counts.get("0-1", 0),
            draws=result_counts.get("1/2-1/2", 0),
            unfinished=result_counts.get("*", 0),
            result_counts=result_counts,
            termination_counts=termination_counts,
        )
        _update_cell(
            matrix[black_agent][white_agent],
            game_count=game_count,
            score=black_score,
            wins=result_counts.get("0-1", 0),
            losses=result_counts.get("1-0", 0),
            draws=result_counts.get("1/2-1/2", 0),
            unfinished=result_counts.get("*", 0),
            result_counts=result_counts,
            termination_counts=termination_counts,
        )

    for row_agent in agent_names:
        for column_agent in agent_names:
            cell = matrix[row_agent][column_agent]
            cell["score_rate"] = (
                round(float(cell["score"]) / int(cell["game_count"]), 6)
                if int(cell["game_count"]) > 0
                else 0.0
            )
            cell["result_counts"] = dict(sorted(dict(cell["result_counts"]).items()))
            cell["termination_counts"] = dict(sorted(dict(cell["termination_counts"]).items()))

    ranking = [
        {
            "agent": str(agent_name),
            "score": float(record.get("score", 0.0)),
            "games": int(record.get("games", 0)),
            "score_rate": round(
                float(record.get("score", 0.0)) / int(record.get("games", 1)),
                6,
            )
            if int(record.get("games", 0)) > 0
            else 0.0,
            "wins": int(record.get("wins", 0)),
            "draws": int(record.get("draws", 0)),
            "losses": int(record.get("losses", 0)),
            "unfinished": int(record.get("unfinished", 0)),
        }
        for agent_name, record in standings.items()
    ]
    ranking.sort(
        key=lambda record: (
            -float(record["score_rate"]),
            -float(record["score"]),
            str(record["agent"]),
        )
    )

    return {
        "arena_name": summary.get("arena_name"),
        "arena_spec_version": summary.get("arena_spec_version"),
        "agent_names": agent_names,
        "aggregate": dict(summary.get("aggregate") or {}),
        "standings": standings,
        "ranking_by_score_rate": ranking,
        "matrix": matrix,
    }


def write_selfplay_arena_matrix(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one arena matrix payload as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _empty_matrix_cell(row_agent: str, column_agent: str) -> dict[str, Any]:
    return {
        "row_agent": row_agent,
        "column_agent": column_agent,
        "game_count": 0,
        "score": 0.0,
        "score_rate": 0.0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "unfinished": 0,
        "result_counts": Counter(),
        "termination_counts": Counter(),
    }


def _update_cell(
    cell: dict[str, Any],
    *,
    game_count: int,
    score: float,
    wins: int,
    losses: int,
    draws: int,
    unfinished: int,
    result_counts: Counter[str],
    termination_counts: Counter[str],
) -> None:
    cell["game_count"] = int(cell["game_count"]) + game_count
    cell["score"] = round(float(cell["score"]) + score, 6)
    cell["wins"] = int(cell["wins"]) + wins
    cell["losses"] = int(cell["losses"]) + losses
    cell["draws"] = int(cell["draws"]) + draws
    cell["unfinished"] = int(cell["unfinished"]) + unfinished
    cell["result_counts"].update(result_counts)
    cell["termination_counts"].update(termination_counts)
