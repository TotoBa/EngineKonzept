"""Analyze color asymmetries in completed selfplay arena runs."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping


def build_selfplay_arena_asymmetry_report(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a color-asymmetry report for one completed arena summary."""

    overall = Counter()
    agent_profiles: dict[str, dict[str, float | int | Counter[str]]] = defaultdict(
        lambda: {
            "games_as_white": 0,
            "games_as_black": 0,
            "score_as_white": 0.0,
            "score_as_black": 0.0,
            "result_counts_as_white": Counter(),
            "result_counts_as_black": Counter(),
        }
    )
    pair_profiles: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_matchup in list(summary.get("matchups") or []):
        session_path = Path(str(raw_matchup["session_path"]))
        session_payload = json.loads(session_path.read_text(encoding="utf-8"))
        for raw_game in list(session_payload.get("games") or []):
            white_agent = str(raw_game["white_agent"])
            black_agent = str(raw_game["black_agent"])
            result = str(raw_game.get("result", "*"))
            white_score, black_score = _scores_for_result(result)

            overall["games"] += 1
            overall[f"result:{result}"] += 1
            if result == "1-0":
                overall["white_wins"] += 1
            elif result == "0-1":
                overall["black_wins"] += 1
            elif result == "1/2-1/2":
                overall["draws"] += 1
            else:
                overall["unfinished"] += 1
            overall["white_score_sum"] += white_score
            overall["black_score_sum"] += black_score

            white_profile = agent_profiles[white_agent]
            white_profile["games_as_white"] = int(white_profile["games_as_white"]) + 1
            white_profile["score_as_white"] = float(white_profile["score_as_white"]) + white_score
            white_profile["result_counts_as_white"].update([result])  # type: ignore[arg-type]

            black_profile = agent_profiles[black_agent]
            black_profile["games_as_black"] = int(black_profile["games_as_black"]) + 1
            black_profile["score_as_black"] = float(black_profile["score_as_black"]) + black_score
            black_profile["result_counts_as_black"].update([result])  # type: ignore[arg-type]

            pair_key = tuple(sorted((white_agent, black_agent)))
            pair_profile = pair_profiles.setdefault(
                pair_key,
                {
                    "agent_a": pair_key[0],
                    "agent_b": pair_key[1],
                    "games": 0,
                    "white_wins": 0,
                    "black_wins": 0,
                    "draws": 0,
                    "unfinished": 0,
                    "agent_a_games_as_white": 0,
                    "agent_a_games_as_black": 0,
                    "agent_a_score_as_white": 0.0,
                    "agent_a_score_as_black": 0.0,
                    "termination_counts": Counter(),
                },
            )
            pair_profile["games"] += 1
            pair_profile["termination_counts"].update(
                [str(raw_game.get("termination_reason", "unknown"))]
            )
            if result == "1-0":
                pair_profile["white_wins"] += 1
            elif result == "0-1":
                pair_profile["black_wins"] += 1
            elif result == "1/2-1/2":
                pair_profile["draws"] += 1
            else:
                pair_profile["unfinished"] += 1

            if white_agent == pair_key[0]:
                pair_profile["agent_a_games_as_white"] += 1
                pair_profile["agent_a_score_as_white"] += white_score
            else:
                pair_profile["agent_a_games_as_black"] += 1
                pair_profile["agent_a_score_as_black"] += black_score

    per_agent = []
    for agent_name in sorted(agent_profiles):
        profile = agent_profiles[agent_name]
        games_as_white = int(profile["games_as_white"])
        games_as_black = int(profile["games_as_black"])
        score_as_white = float(profile["score_as_white"])
        score_as_black = float(profile["score_as_black"])
        per_agent.append(
            {
                "agent": agent_name,
                "games_as_white": games_as_white,
                "games_as_black": games_as_black,
                "score_as_white": round(score_as_white, 6),
                "score_as_black": round(score_as_black, 6),
                "score_rate_as_white": (
                    round(score_as_white / games_as_white, 6) if games_as_white > 0 else 0.0
                ),
                "score_rate_as_black": (
                    round(score_as_black / games_as_black, 6) if games_as_black > 0 else 0.0
                ),
                "color_score_delta": round(
                    (
                        (score_as_white / games_as_white if games_as_white > 0 else 0.0)
                        - (score_as_black / games_as_black if games_as_black > 0 else 0.0)
                    ),
                    6,
                ),
                "result_counts_as_white": dict(
                    sorted(profile["result_counts_as_white"].items())  # type: ignore[union-attr]
                ),
                "result_counts_as_black": dict(
                    sorted(profile["result_counts_as_black"].items())  # type: ignore[union-attr]
                ),
            }
        )

    per_pair = []
    for pair_key in sorted(pair_profiles):
        profile = pair_profiles[pair_key]
        games = int(profile["games"])
        white_score_rate = (
            overall_rate := (
                (profile["white_wins"] + 0.5 * profile["draws"]) / games if games > 0 else 0.0
            )
        )
        black_score_rate = 1.0 - overall_rate if games > 0 else 0.0
        agent_a_games_as_white = int(profile["agent_a_games_as_white"])
        agent_a_games_as_black = int(profile["agent_a_games_as_black"])
        agent_a_score_as_white = float(profile["agent_a_score_as_white"])
        agent_a_score_as_black = float(profile["agent_a_score_as_black"])
        agent_a_white_rate = (
            agent_a_score_as_white / agent_a_games_as_white
            if agent_a_games_as_white > 0
            else 0.0
        )
        agent_a_black_rate = (
            agent_a_score_as_black / agent_a_games_as_black
            if agent_a_games_as_black > 0
            else 0.0
        )
        per_pair.append(
            {
                "agent_a": profile["agent_a"],
                "agent_b": profile["agent_b"],
                "games": games,
                "white_wins": int(profile["white_wins"]),
                "black_wins": int(profile["black_wins"]),
                "draws": int(profile["draws"]),
                "unfinished": int(profile["unfinished"]),
                "white_score_rate": round(white_score_rate, 6),
                "black_score_rate": round(black_score_rate, 6),
                "agent_a_score_rate_as_white": round(agent_a_white_rate, 6),
                "agent_a_score_rate_as_black": round(agent_a_black_rate, 6),
                "agent_a_color_score_delta": round(
                    agent_a_white_rate - agent_a_black_rate,
                    6,
                ),
                "termination_counts": dict(
                    sorted(profile["termination_counts"].items())  # type: ignore[union-attr]
                ),
            }
        )

    per_pair.sort(
        key=lambda entry: (
            -abs(float(entry["agent_a_color_score_delta"])),
            -int(entry["games"]),
            str(entry["agent_a"]),
            str(entry["agent_b"]),
        )
    )
    suspicious_pairs = [
        entry
        for entry in per_pair
        if int(entry["games"]) >= 4 and abs(float(entry["agent_a_color_score_delta"])) >= 0.5
    ]

    total_games = int(overall["games"])
    return {
        "arena_name": summary.get("arena_name"),
        "arena_spec_version": summary.get("arena_spec_version"),
        "overall": {
            "games": total_games,
            "white_wins": int(overall["white_wins"]),
            "black_wins": int(overall["black_wins"]),
            "draws": int(overall["draws"]),
            "unfinished": int(overall["unfinished"]),
            "white_score_rate": (
                round(float(overall["white_score_sum"]) / total_games, 6)
                if total_games > 0
                else 0.0
            ),
            "black_score_rate": (
                round(float(overall["black_score_sum"]) / total_games, 6)
                if total_games > 0
                else 0.0
            ),
        },
        "per_agent": per_agent,
        "per_pair": per_pair,
        "suspicious_pairs": suspicious_pairs,
    }


def write_selfplay_arena_asymmetry_report(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    """Write one arena asymmetry report as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _scores_for_result(result: str) -> tuple[float, float]:
    if result == "1-0":
        return 1.0, 0.0
    if result == "0-1":
        return 0.0, 1.0
    if result == "1/2-1/2":
        return 0.5, 0.5
    return 0.0, 0.0
