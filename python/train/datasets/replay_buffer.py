"""Replay-buffer artifacts derived from small exact selfplay sessions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from train.eval.selfplay import SelfplayGameRecord, SelfplaySessionRecord


@dataclass(frozen=True)
class ReplayBufferEntry:
    """One replay-buffer row derived from a selfplay move."""

    sample_id: str
    game_id: str
    ply_index: int
    side_to_move: str
    fen: str
    move_uci: str
    action_index: int
    next_fen: str
    selector_name: str
    white_agent: str
    black_agent: str
    legal_candidate_count: int
    considered_candidate_count: int
    proposer_score: float
    planner_score: float
    reply_peak_probability: float
    pressure: float
    uncertainty: float
    game_result: str
    outcome_pov: str
    termination_reason: str
    game_move_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "game_id": self.game_id,
            "ply_index": self.ply_index,
            "side_to_move": self.side_to_move,
            "fen": self.fen,
            "move_uci": self.move_uci,
            "action_index": self.action_index,
            "next_fen": self.next_fen,
            "selector_name": self.selector_name,
            "white_agent": self.white_agent,
            "black_agent": self.black_agent,
            "legal_candidate_count": self.legal_candidate_count,
            "considered_candidate_count": self.considered_candidate_count,
            "proposer_score": self.proposer_score,
            "planner_score": self.planner_score,
            "reply_peak_probability": self.reply_peak_probability,
            "pressure": self.pressure,
            "uncertainty": self.uncertainty,
            "game_result": self.game_result,
            "outcome_pov": self.outcome_pov,
            "termination_reason": self.termination_reason,
            "game_move_count": self.game_move_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ReplayBufferEntry":
        return cls(
            sample_id=str(payload["sample_id"]),
            game_id=str(payload["game_id"]),
            ply_index=int(payload["ply_index"]),
            side_to_move=str(payload["side_to_move"]),
            fen=str(payload["fen"]),
            move_uci=str(payload["move_uci"]),
            action_index=int(payload["action_index"]),
            next_fen=str(payload["next_fen"]),
            selector_name=str(payload["selector_name"]),
            white_agent=str(payload["white_agent"]),
            black_agent=str(payload["black_agent"]),
            legal_candidate_count=int(payload["legal_candidate_count"]),
            considered_candidate_count=int(payload["considered_candidate_count"]),
            proposer_score=float(payload["proposer_score"]),
            planner_score=float(payload["planner_score"]),
            reply_peak_probability=float(payload["reply_peak_probability"]),
            pressure=float(payload["pressure"]),
            uncertainty=float(payload["uncertainty"]),
            game_result=str(payload["game_result"]),
            outcome_pov=str(payload["outcome_pov"]),
            termination_reason=str(payload["termination_reason"]),
            game_move_count=int(payload["game_move_count"]),
        )

    @classmethod
    def from_json(cls, line: str) -> "ReplayBufferEntry":
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("replay buffer entry must be a JSON object")
        return cls.from_dict(payload)


def build_replay_buffer_entries(session: SelfplaySessionRecord) -> list[ReplayBufferEntry]:
    """Flatten one selfplay session into replay-buffer rows."""
    entries: list[ReplayBufferEntry] = []
    for game in session.games:
        entries.extend(_entries_from_game(game))
    return entries


def load_replay_buffer_entries(path: Path) -> list[ReplayBufferEntry]:
    """Load replay-buffer rows from JSONL."""
    if not path.exists():
        raise FileNotFoundError(f"replay buffer artifact not found: {path}")
    entries: list[ReplayBufferEntry] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entries.append(ReplayBufferEntry.from_json(line))
    return entries


def write_replay_buffer_artifact(
    path: Path,
    entries: Sequence[ReplayBufferEntry],
) -> None:
    """Write replay-buffer rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(entry.to_dict(), sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )


def replay_buffer_summary(entries: Sequence[ReplayBufferEntry]) -> dict[str, object]:
    """Compute a compact aggregate summary for a replay-buffer artifact."""
    if not entries:
        return {
            "entry_count": 0,
            "game_count": 0,
            "mean_considered_candidate_count": 0.0,
            "mean_game_move_count": 0.0,
            "outcome_counts": {},
            "selector_counts": {},
            "termination_counts": {},
        }
    selector_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    termination_counts: dict[str, int] = {}
    game_lengths: dict[str, int] = {}
    for entry in entries:
        selector_counts[entry.selector_name] = selector_counts.get(entry.selector_name, 0) + 1
        outcome_counts[entry.outcome_pov] = outcome_counts.get(entry.outcome_pov, 0) + 1
        termination_counts[entry.termination_reason] = termination_counts.get(
            entry.termination_reason, 0
        ) + 1
        game_lengths[entry.game_id] = entry.game_move_count
    return {
        "entry_count": len(entries),
        "game_count": len(game_lengths),
        "mean_considered_candidate_count": round(
            sum(entry.considered_candidate_count for entry in entries) / len(entries),
            3,
        ),
        "mean_game_move_count": round(sum(game_lengths.values()) / len(game_lengths), 3),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "selector_counts": dict(sorted(selector_counts.items())),
        "termination_counts": dict(sorted(termination_counts.items())),
    }


def _entries_from_game(game: SelfplayGameRecord) -> list[ReplayBufferEntry]:
    built: list[ReplayBufferEntry] = []
    for move in game.moves:
        built.append(
            ReplayBufferEntry(
                sample_id=f"{game.game_id}:{move.ply_index}",
                game_id=game.game_id,
                ply_index=move.ply_index,
                side_to_move=move.side_to_move,
                fen=move.fen,
                move_uci=move.move_uci,
                action_index=move.action_index,
                next_fen=move.next_fen,
                selector_name=move.selector_name,
                white_agent=game.white_agent,
                black_agent=game.black_agent,
                legal_candidate_count=move.legal_candidate_count,
                considered_candidate_count=move.considered_candidate_count,
                proposer_score=move.proposer_score,
                planner_score=move.planner_score,
                reply_peak_probability=move.reply_peak_probability,
                pressure=move.pressure,
                uncertainty=move.uncertainty,
                game_result=game.result,
                outcome_pov=_outcome_pov(game.result, move.side_to_move),
                termination_reason=game.termination_reason,
                game_move_count=game.move_count,
            )
        )
    return built


def _outcome_pov(result: str, side_to_move: str) -> str:
    if result == "1/2-1/2":
        return "draw"
    if result == "*":
        return "unfinished"
    if side_to_move == "w":
        return "win" if result == "1-0" else "loss"
    if side_to_move == "b":
        return "win" if result == "0-1" else "loss"
    raise ValueError(f"unsupported side_to_move: {side_to_move}")
