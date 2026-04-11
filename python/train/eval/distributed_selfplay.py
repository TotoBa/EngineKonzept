"""Distributed shard helpers for pre-verify LAP selfplay."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path

from train.eval.agent_spec import load_selfplay_agent_spec
from train.eval.initial_fens import load_selfplay_initial_fen_suite
from train.eval.phase10_campaign import Phase10Lapv1ArenaCampaignSpec, resolve_repo_path
from train.eval.selfplay import (
    STARTING_FEN,
    SelfplayMaxPliesAdjudicationSpec,
    SelfplaySessionRecord,
    open_max_plies_adjudicator,
    run_selfplay_session,
)


def run_phase10_pre_verify_selfplay_shard(
    *,
    spec: Phase10Lapv1ArenaCampaignSpec,
    repo_root: Path,
    agent_spec_path: Path,
    agent_name: str,
    output_root: Path,
    shard_index: int,
    starting_game_index: int,
    games: int,
    max_plies: int,
) -> dict[str, object]:
    """Run one tracked-LAP selfplay shard and persist its session plus shard summary."""
    if shard_index <= 0:
        raise ValueError("shard_index must be positive")
    if starting_game_index < 0:
        raise ValueError("starting_game_index must be non-negative")
    if games <= 0:
        raise ValueError("games must be positive")
    if max_plies <= 0:
        raise ValueError("max_plies must be positive")

    agent_spec = load_selfplay_agent_spec(agent_spec_path)
    if agent_spec.agent_kind != "lapv1":
        raise ValueError(f"pre-verify selfplay currently requires lapv1 agent specs, got {agent_spec.agent_kind}")
    from train.eval.lapv1_runtime import build_lapv1_runtime_from_spec

    runtime = build_lapv1_runtime_from_spec(agent_spec, repo_root=repo_root)

    selected_openings = _select_openings(
        spec=spec,
        repo_root=repo_root,
        starting_game_index=starting_game_index,
        games=games,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    sessions_root = output_root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    shards_root = output_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    adjudication = (
        SelfplayMaxPliesAdjudicationSpec.from_dict(dict(spec.max_plies_adjudication))
        if spec.max_plies_adjudication is not None
        else None
    )
    adjudication_cm = (
        open_max_plies_adjudicator(adjudication) if adjudication is not None else nullcontext(None)
    )
    with adjudication_cm as adjudicator:
        session = run_selfplay_session(
            white_agent=runtime,
            black_agent=runtime,
            repo_root=repo_root,
            games=games,
            initial_fens=[entry["fen"] for entry in selected_openings],
            max_plies=max_plies,
            adjudicator=adjudicator,
            adjudication_spec=adjudication,
        )
    session_path = sessions_root / f"selfplay_shard_{shard_index:04d}.json"
    session_path.write_text(
        json.dumps(session.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "campaign_name": spec.name,
        "agent_name": agent_name,
        "agent_spec_path": str(agent_spec_path),
        "shard_index": shard_index,
        "starting_game_index": starting_game_index,
        "games": games,
        "max_plies": max_plies,
        "session_path": str(session_path),
        "selected_openings": selected_openings,
        "aggregate": dict(session.to_dict()["aggregate"]),
    }
    summary_path = shards_root / f"selfplay_shard_{shard_index:04d}.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def rebuild_phase10_pre_verify_selfplay_summary(
    *,
    output_root: Path,
    agent_name: str,
    agent_spec_path: Path,
) -> dict[str, object]:
    """Rebuild the aggregate selfplay summary from shard session files."""
    sessions_root = output_root / "sessions"
    if not sessions_root.exists():
        raise ValueError(f"{sessions_root}: pre-verify selfplay sessions directory does not exist")
    session_paths = sorted(sessions_root.glob("selfplay_shard_*.json"))
    if not session_paths:
        raise ValueError(f"{sessions_root}: no pre-verify selfplay sessions found")

    combined_games = []
    result_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    move_counts: list[int] = []
    for session_path in session_paths:
        session = SelfplaySessionRecord.from_json(session_path.read_text(encoding="utf-8"))
        combined_games.extend(session.games)
        for game in session.games:
            result_counts[game.result] += 1
            termination_counts[game.termination_reason] += 1
            move_counts.append(game.move_count)
    aggregate = {
        "game_count": len(combined_games),
        "mean_move_count": round(sum(move_counts) / len(move_counts), 3) if move_counts else 0.0,
        "result_counts": dict(sorted(result_counts.items())),
        "termination_counts": dict(sorted(termination_counts.items())),
        "session_count": len(session_paths),
    }
    summary = {
        "agent_name": agent_name,
        "agent_spec_path": str(agent_spec_path),
        "aggregate": aggregate,
        "session_paths": [str(path) for path in session_paths],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _select_openings(
    *,
    spec: Phase10Lapv1ArenaCampaignSpec,
    repo_root: Path,
    starting_game_index: int,
    games: int,
) -> list[dict[str, object]]:
    if not spec.initial_fen_suite_path:
        return [
            {
                "fen": STARTING_FEN,
                "opening_index": 0,
                "sample_id": "startpos",
                "source_path": "builtin:startpos",
            }
            for _index in range(games)
        ]

    suite = load_selfplay_initial_fen_suite(
        resolve_repo_path(repo_root, Path(spec.initial_fen_suite_path))
    )
    entries = list(suite.entries)
    if spec.pre_verify_selfplay_opening_selection_seed is not None:
        import random

        rng = random.Random(spec.pre_verify_selfplay_opening_selection_seed)
        rng.shuffle(entries)
    selected = []
    for offset in range(games):
        entry_index = (starting_game_index + offset) % len(entries)
        entry = entries[entry_index]
        selected.append(
            {
                "fen": entry.fen,
                "opening_index": entry_index,
                "sample_id": entry.sample_id,
                "source_path": entry.source_path,
            }
        )
    return selected
